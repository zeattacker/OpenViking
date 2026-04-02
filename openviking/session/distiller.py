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
    skipped_duplicates: int = 0
    skipped_stale: int = 0
    errors: int = 0
    pattern_uris: List[str] = field(default_factory=list)


class PatternDistiller:
    """Consolidates similar case memories into reusable patterns."""

    # Threshold for considering a new pattern a duplicate of an existing one.
    PATTERN_DEDUP_THRESHOLD = 0.90

    def __init__(
        self,
        vikingdb: VikingDBManager,
        viking_fs: Any,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        pattern_dedup_threshold: Optional[float] = None,
    ):
        self.vikingdb = vikingdb
        self.viking_fs = viking_fs
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.pattern_dedup_threshold = (
            pattern_dedup_threshold
            if pattern_dedup_threshold is not None
            else self.PATTERN_DEDUP_THRESHOLD
        )
        config = get_openviking_config()
        self._embedder = config.embedding.get_embedder()

    async def consolidate(
        self,
        scope_uri: str,
        ctx: RequestContext,
        dry_run: bool = False,
        subdirectory: str = "cases",
    ) -> ConsolidationResult:
        """Main entry point. Scans a memory subdirectory, clusters similar, consolidates.

        Args:
            scope_uri: Agent space URI, e.g. ``viking://agent/{space}``.
            ctx: Request context.
            dry_run: If True, log what would happen without writing.
            subdirectory: Memory subdirectory to scan (e.g. ``cases``, ``entities``).

        Returns:
            Summary of the consolidation operation.
        """
        result = ConsolidationResult()
        scope_hash = hashlib.md5(f"{scope_uri}/{subdirectory}".encode()).hexdigest()[:12]
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

            result = await self._do_consolidate(scope_uri, ctx, dry_run, subdirectory)
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
        subdirectory: str = "cases",
    ) -> ConsolidationResult:
        """Core consolidation logic."""
        result = ConsolidationResult()
        cases_uri = f"{scope_uri}/memories/{subdirectory}/"

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

        # Step 2: Get vectors for cases from vector DB, filtering stale entries.
        vectors = await self._get_case_vectors(cases_uri, ctx, md_entries)
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
                elif not dry_run:
                    # None return without dry_run means dedup skip or hash collision.
                    result.skipped_duplicates += 1
            except Exception as e:
                logger.error(
                    "[PatternDistiller] Failed to consolidate cluster: %s", e, exc_info=True
                )
                result.errors += 1

        logger.info(
            "[PatternDistiller] Consolidation complete for %s: "
            "clusters=%d, patterns_created=%d, skipped_dup=%d, errors=%d",
            scope_uri,
            result.clusters_found,
            result.patterns_created,
            result.skipped_duplicates,
            result.errors,
        )
        return result

    async def _get_case_vectors(
        self,
        cases_uri_prefix: str,
        ctx: RequestContext,
        fs_entries: Optional[List[Dict]] = None,
    ) -> List[Tuple[str, List[float], str]]:
        """Fetch (uri, vector, abstract) tuples for case memories.

        Uses vikingdb.scroll() with level=2 filter, scoped to the cases URI prefix.
        Skips chunk URIs (``#chunk_NNNN``) to avoid inflating cluster counts —
        only the parent file URI is kept.  When a file has multiple chunks,
        the first chunk's vector is used as the representative.

        If *fs_entries* is provided, vectors whose parent URI does not correspond
        to an existing filesystem entry are silently dropped.  This prevents
        stale vectors from already-archived cases from being included.
        """
        from openviking.storage.expr import And

        # Build a set of known filesystem URIs for fast membership checks.
        known_fs_uris: Optional[set] = None
        if fs_entries:
            known_fs_uris = set()
            for e in fs_entries:
                name = e.get("name", "")
                if name:
                    known_fs_uris.add(f"{cases_uri_prefix}{name}")

        seen_parents: Dict[str, bool] = {}
        vectors: List[Tuple[str, List[float], str]] = []
        filter_expr = And(conds=[Eq("level", 2)])
        cursor: Optional[str] = None

        while True:
            records, next_cursor = await self.vikingdb.scroll(
                filter=filter_expr,
                limit=100,
                cursor=cursor,
                output_fields=["uri", "abstract", "vector"],
                ctx=ctx,
            )
            if not records:
                break

            for record in records:
                uri = record.get("uri", "")
                if not uri.startswith(cases_uri_prefix):
                    continue

                # Normalise chunk URIs to their parent file.
                parent_uri = uri.split("#")[0]
                if parent_uri in seen_parents:
                    continue
                seen_parents[parent_uri] = True

                # Skip stale vectors whose files no longer exist on filesystem.
                if known_fs_uris is not None and parent_uri not in known_fs_uris:
                    logger.debug(
                        "[PatternDistiller] Skipping stale vector %s (file gone)",
                        parent_uri,
                    )
                    continue

                vec = record.get("vector")
                if not vec or not isinstance(vec, list):
                    continue
                abstract = record.get("abstract", "")
                vectors.append((parent_uri, vec, abstract))

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

        # --- Dedup check: skip if existing pattern is semantically equivalent ---
        existing_match = await self._find_duplicate_pattern(response, scope_uri, ctx)
        if existing_match:
            logger.info(
                "[PatternDistiller] Skipping duplicate pattern (matches %s), "
                "archiving %d source cases",
                existing_match,
                len(cluster_uris),
            )
            # Still archive source cases — they are covered by the existing pattern.
            await self._archive_source_cases(cluster_uris, ctx)
            return None  # Caller increments skipped_duplicates via sentinel

        # Write pattern file.
        content_hash = hashlib.md5(response.encode()).hexdigest()[:12]
        pattern_uri = f"{scope_uri}/memories/patterns/consolidated_{content_hash}.md"

        # Guard against overwriting an existing file with the same hash.
        try:
            existing = await self.viking_fs.read_file(pattern_uri, ctx=ctx)
            if existing:
                logger.info(
                    "[PatternDistiller] Pattern file %s already exists (hash collision), "
                    "archiving %d source cases",
                    pattern_uri,
                    len(cluster_uris),
                )
                await self._archive_source_cases(cluster_uris, ctx)
                return None
        except Exception:
            pass  # File doesn't exist — proceed to write.

        await self.viking_fs.write_file(pattern_uri, response, ctx=ctx)

        # Archive original source cases after pattern creation.
        await self._archive_source_cases(cluster_uris, ctx)

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

    async def _find_duplicate_pattern(
        self,
        content: str,
        scope_uri: str,
        ctx: RequestContext,
    ) -> Optional[str]:
        """Check if *content* is semantically duplicate of an existing pattern.

        Embeds the candidate content, searches the patterns/ directory via vector
        similarity, and returns the URI of the first match above
        ``self.pattern_dedup_threshold``.  Returns ``None`` if no duplicate found
        or if the embedder is unavailable.
        """
        if not self._embedder:
            return None

        try:
            embed_result = self._embedder.embed(content, is_query=True)
            query_vector = embed_result.dense_vector
            if not query_vector:
                return None
        except Exception as e:
            logger.warning("[PatternDistiller] Failed to embed candidate pattern: %s", e)
            return None

        patterns_uri_prefix = f"{scope_uri}/memories/patterns/"

        # Derive owner_space from scope_uri for the search filter.
        # scope_uri is e.g. "viking://agent/{space}" or "viking://user/{space}".
        owner_space: Optional[str] = None
        parts = scope_uri.rstrip("/").rsplit("/", 1)
        if len(parts) == 2:
            owner_space = parts[1]

        try:
            results = await self.vikingdb.search_similar_memories(
                owner_space=owner_space,
                category_uri_prefix=patterns_uri_prefix,
                query_vector=query_vector,
                limit=3,
                ctx=ctx,
            )
        except Exception as e:
            logger.warning("[PatternDistiller] Pattern dedup search failed: %s", e)
            return None

        for result in results:
            score = float(result.get("_score", result.get("score", 0)) or 0)
            uri = result.get("uri", "")
            if score >= self.pattern_dedup_threshold and uri.startswith(patterns_uri_prefix):
                logger.debug(
                    "[PatternDistiller] Duplicate pattern match: %s (score=%.4f)",
                    uri,
                    score,
                )
                return uri

        return None

    async def _archive_source_cases(
        self,
        cluster_uris: List[str],
        ctx: RequestContext,
    ) -> None:
        """Move source case files to ``_archive/`` subdirectory and remove vectors."""
        for source_uri in cluster_uris:
            try:
                parts = source_uri.rsplit("/", 1)
                archive_uri = f"{parts[0]}/_archive/{parts[1]}"
                await self.viking_fs.mv(source_uri, archive_uri, ctx=ctx)
                # Remove vectors for archived source — they are superseded by
                # the consolidated pattern and should not consume vector DB space.
                try:
                    await self.vikingdb.delete_uris(ctx, [archive_uri])
                except Exception as e:
                    logger.debug(
                        "[PatternDistiller] Vector cleanup for %s: %s", archive_uri, e
                    )
            except Exception as e:
                logger.warning(
                    "[PatternDistiller] Failed to archive %s: %s", source_uri, e
                )
