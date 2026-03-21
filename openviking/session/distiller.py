# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Pattern Distillation for OpenViking.

Scans case memories, clusters similar ones via cosine similarity,
and consolidates each cluster into a single reusable pattern.
"""

import fcntl
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openviking.core.context import Context, Vectorize
from openviking.prompts import render_prompt
from openviking.server.identity import RequestContext
from openviking.storage import VikingDBManager
from openviking.storage.expr import Eq
from openviking_cli.utils import get_logger
from openviking_cli.utils.config import get_openviking_config

logger = get_logger(__name__)

# Lock staleness threshold: skip if lock is younger than 1 hour.
_LOCK_MAX_AGE_SECS = 3600


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""

    scanned: int = 0
    clusters_found: int = 0
    patterns_created: int = 0
    errors: int = 0
    pattern_uris: List[str] = field(default_factory=list)


class PatternDistiller:
    """Consolidates similar case memories into reusable patterns."""

    def __init__(
        self,
        vikingdb: VikingDBManager,
        viking_fs: Any,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
    ):
        self.vikingdb = vikingdb
        self.viking_fs = viking_fs
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

    async def consolidate(
        self,
        scope_uri: str,
        ctx: RequestContext,
        dry_run: bool = False,
    ) -> ConsolidationResult:
        """Main entry point. Scans cases/, clusters similar, consolidates.

        Args:
            scope_uri: Agent space URI, e.g. ``viking://agent/{space}``.
            ctx: Request context.
            dry_run: If True, log what would happen without writing.

        Returns:
            Summary of the consolidation operation.
        """
        result = ConsolidationResult()
        scope_hash = hashlib.md5(scope_uri.encode()).hexdigest()[:12]
        lock_path = f"/tmp/openviking_distill_{scope_hash}.lock"

        # File-lock to prevent concurrent runs on the same scope.
        lock_fd = None
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                # Lock held by another process — check age.
                try:
                    lock_age = time.time() - os.fstat(lock_fd).st_mtime
                except OSError:
                    lock_age = 0
                if lock_age < _LOCK_MAX_AGE_SECS:
                    logger.info(
                        "[PatternDistiller] Skipping %s: lock held (age=%.0fs)",
                        scope_uri,
                        lock_age,
                    )
                    return result
                # Stale lock — force acquire.
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Touch lock file so other processes see fresh mtime.
            os.utime(lock_fd)

            result = await self._do_consolidate(scope_uri, ctx, dry_run)
        except Exception as e:
            logger.error("[PatternDistiller] Error consolidating %s: %s", scope_uri, e, exc_info=True)
            result.errors += 1
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                except OSError:
                    pass

        return result

    async def _do_consolidate(
        self,
        scope_uri: str,
        ctx: RequestContext,
        dry_run: bool,
    ) -> ConsolidationResult:
        """Core consolidation logic."""
        result = ConsolidationResult()
        cases_uri = f"{scope_uri}/memories/cases/"

        # Step 1: Snapshot — list case files.
        try:
            entries = await self.viking_fs.ls(cases_uri, ctx=ctx)
        except Exception as e:
            logger.warning("[PatternDistiller] Failed to list %s: %s", cases_uri, e)
            return result

        md_entries = [
            e for e in entries
            if isinstance(e, dict)
            and isinstance(e.get("name"), str)
            and e["name"].endswith(".md")
            and not e.get("isDir", False)
        ]
        if not md_entries:
            logger.debug("[PatternDistiller] No case files in %s", cases_uri)
            return result

        result.scanned = len(md_entries)

        # Step 2: Get vectors for cases from vector DB.
        vectors = await self._get_case_vectors(cases_uri, ctx)
        if len(vectors) < self.min_cluster_size:
            logger.debug(
                "[PatternDistiller] Not enough vectors (%d) for clustering in %s",
                len(vectors),
                cases_uri,
            )
            return result

        # Step 3: Cluster by cosine similarity using union-find.
        clusters = self._cluster_vectors(vectors)
        result.clusters_found = len(clusters)

        if not clusters:
            logger.debug("[PatternDistiller] No clusters found in %s", cases_uri)
            return result

        logger.info(
            "[PatternDistiller] Found %d clusters in %s (scanned=%d)",
            len(clusters),
            cases_uri,
            result.scanned,
        )

        # Step 4: Consolidate each cluster.
        for cluster_uris in clusters:
            try:
                pattern_uri = await self._consolidate_cluster(
                    cluster_uris, scope_uri, ctx, dry_run
                )
                if pattern_uri:
                    result.patterns_created += 1
                    result.pattern_uris.append(pattern_uri)
            except Exception as e:
                logger.error(
                    "[PatternDistiller] Failed to consolidate cluster: %s", e, exc_info=True
                )
                result.errors += 1

        logger.info(
            "[PatternDistiller] Consolidation complete for %s: "
            "clusters=%d, patterns_created=%d, errors=%d",
            scope_uri,
            result.clusters_found,
            result.patterns_created,
            result.errors,
        )
        return result

    async def _get_case_vectors(
        self,
        cases_uri_prefix: str,
        ctx: RequestContext,
    ) -> List[Tuple[str, List[float], str]]:
        """Fetch (uri, vector, abstract) tuples for case memories.

        Uses vikingdb.scroll() with level=2 filter, scoped to the cases URI prefix.
        """
        from openviking.storage.expr import And

        vectors: List[Tuple[str, List[float], str]] = []
        filter_expr = And(conds=[Eq("level", 2)])
        cursor: Optional[str] = None

        while True:
            records, next_cursor = await self.vikingdb.scroll(
                filter=filter_expr,
                limit=100,
                cursor=cursor,
                output_fields=["uri", "abstract", "dense_vector"],
                ctx=ctx,
            )
            if not records:
                break

            for record in records:
                uri = record.get("uri", "")
                if not uri.startswith(cases_uri_prefix):
                    continue
                vec = record.get("dense_vector")
                if not vec or not isinstance(vec, list):
                    continue
                abstract = record.get("abstract", "")
                vectors.append((uri, vec, abstract))

            cursor = next_cursor
            if cursor is None:
                break

        return vectors

    def _cluster_vectors(
        self,
        vectors: List[Tuple[str, List[float], str]],
    ) -> List[List[str]]:
        """Cluster vectors using union-find with cosine similarity threshold.

        Returns clusters with >= min_cluster_size members.
        """
        from openviking.session.memory_deduplicator import MemoryDeduplicator

        n = len(vectors)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Pairwise cosine similarity — O(n^2) but n is small per scope.
        for i in range(n):
            for j in range(i + 1, n):
                sim = MemoryDeduplicator._cosine_similarity(vectors[i][1], vectors[j][1])
                if sim >= self.similarity_threshold:
                    union(i, j)

        # Group by root.
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        # Filter to clusters meeting minimum size and collect URIs.
        clusters: List[List[str]] = []
        for members in groups.values():
            if len(members) >= self.min_cluster_size:
                clusters.append([vectors[m][0] for m in members])

        return clusters

    async def _consolidate_cluster(
        self,
        cluster_uris: List[str],
        scope_uri: str,
        ctx: RequestContext,
        dry_run: bool,
    ) -> Optional[str]:
        """Consolidate a cluster of case memories into a single pattern.

        Returns the URI of the created pattern, or None on failure / dry_run.
        """
        # Read full content of each member.
        memory_texts: List[str] = []
        for uri in cluster_uris:
            try:
                content = await self.viking_fs.read_file(uri, ctx=ctx)
                if content:
                    memory_texts.append(f"---\nURI: {uri}\n{content}")
            except Exception as e:
                logger.warning("[PatternDistiller] Failed to read %s: %s", uri, e)

        if len(memory_texts) < self.min_cluster_size:
            return None

        memories_formatted = "\n\n".join(memory_texts)

        if dry_run:
            logger.info(
                "[PatternDistiller] DRY-RUN: would consolidate %d memories into pattern",
                len(memory_texts),
            )
            return None

        # LLM consolidation.
        prompt = render_prompt(
            "compression.pattern_consolidation",
            {
                "count": len(memory_texts),
                "memories": memories_formatted,
                "category": "cases",
                "output_language": "",
            },
        )

        vlm = get_openviking_config().vlm
        if not vlm or not vlm.is_available():
            logger.warning("[PatternDistiller] VLM not available for consolidation")
            return None

        response = await vlm.get_completion_async(prompt)
        if not response or not response.strip():
            logger.warning("[PatternDistiller] Empty LLM response for consolidation")
            return None

        # Write pattern file.
        content_hash = hashlib.md5(response.encode()).hexdigest()[:12]
        pattern_uri = f"{scope_uri}/memories/patterns/consolidated_{content_hash}.md"

        await self.viking_fs.write_file(pattern_uri, response, ctx=ctx)

        # Build context for vectorization.
        pattern_context = Context(
            uri=pattern_uri,
            parent_uri=f"{scope_uri}/memories/patterns/",
            context_type="memory",
            level=2,
            abstract=response[:200].replace("\n", " ").strip(),
            meta={
                "consolidated_from": cluster_uris,
            },
        )
        pattern_context.set_vectorize(Vectorize(text=response))

        # Enqueue for vectorization.
        from openviking.storage.queuefs.embedding_msg_converter import EmbeddingMsgConverter

        embedding_msg = EmbeddingMsgConverter.from_context(pattern_context)
        if embedding_msg:
            await self.vikingdb.enqueue_embedding_msg(embedding_msg)

        # Record semantic change for patterns/ directory.
        from openviking.storage.queuefs import get_queue_manager
        from openviking.storage.queuefs.semantic_msg import SemanticMsg

        try:
            queue_manager = get_queue_manager()
            semantic_queue = queue_manager.get_queue(queue_manager.SEMANTIC, allow_create=True)
            msg = SemanticMsg(
                uri=f"{scope_uri}/memories/patterns/",
                context_type="memory",
                account_id=ctx.account_id,
                user_id=ctx.user.user_id,
                agent_id=ctx.user.agent_id,
                role=ctx.role.value,
                changes={"added": [pattern_uri], "modified": [], "deleted": []},
            )
            await semantic_queue.enqueue(msg)
        except Exception as e:
            logger.warning("[PatternDistiller] Failed to enqueue semantic msg: %s", e)

        logger.info(
            "[PatternDistiller] Created pattern %s from %d memories",
            pattern_uri,
            len(cluster_uris),
        )
        return pattern_uri
