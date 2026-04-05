# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Episode migration from v1 to v2 storage path.

Migrates episodes from viking://user/{space}/episodes/ (v1 path)
to viking://user/{space}/memories/episodes/ (v2 path) without deleting originals.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from openviking.core.context import Context, Vectorize
from openviking.server.identity import RequestContext
from openviking.storage.viking_fs import get_viking_fs
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

_EPISODE_FILENAME_RE = re.compile(r"^ep_([a-f0-9-]+)_(\d{8}T\d{6})\.md$")
_SKIP_NAMES = {"_archive", ".overview.md", ".abstract.md"}


@dataclass
class MigrationResult:
    migrated: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)


async def _enqueue_episode_embedding(
    uri: str,
    content: str,
    user_space: str,
    account_id: str,
    vikingdb,
) -> None:
    """Enqueue a migrated episode for vector embedding."""
    from openviking.storage.queuefs.embedding_msg_converter import EmbeddingMsgConverter

    # Extract abstract from first heading
    abstract = ""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# Episode:"):
            abstract = line.replace("# Episode:", "").strip()
            break
        elif line.startswith("# ") and not abstract:
            abstract = line.lstrip("# ").strip()
            break
    if not abstract:
        abstract = content[:120].strip()

    parent_uri = uri.rsplit("/", 1)[0] if "/" in uri else uri
    episode_ctx = Context(
        uri=uri,
        parent_uri=parent_uri,
        is_leaf=True,
        abstract=abstract,
        context_type="memory",
        category="episodes",
        account_id=account_id,
    )
    episode_ctx.set_vectorize(Vectorize(text=content))

    embedding_msg = EmbeddingMsgConverter.from_context(episode_ctx)
    if embedding_msg:
        await vikingdb.enqueue_embedding_msg(embedding_msg)


async def migrate_v1_episodes(
    user_space: str = "default",
    dry_run: bool = True,
    ctx: Optional[RequestContext] = None,
    vikingdb=None,
) -> MigrationResult:
    """Migrate v1 episodes to v2 storage path.

    Args:
        user_space: User space name to migrate.
        dry_run: If True, only report what would be migrated without writing.
        ctx: Request context for VikingFS operations.
        vikingdb: VikingDBManager for embedding enqueue. If None, files are
            written but not vectorized (manual re-index needed).

    Returns:
        MigrationResult with counts and errors.
    """
    viking_fs = get_viking_fs()
    if not viking_fs:
        return MigrationResult(errors=["VikingFS not available"])

    result = MigrationResult()
    v1_dir = f"viking://user/{user_space}/episodes"
    v2_dir = f"viking://user/{user_space}/memories/episodes"

    # List v1 episodes
    try:
        entries = await viking_fs.ls(v1_dir, ctx=ctx)
    except Exception as e:
        return MigrationResult(errors=[f"Failed to list {v1_dir}: {e}"])

    if not entries:
        logger.info("No v1 episodes found at %s", v1_dir)
        return result

    for entry in entries:
        name = entry if isinstance(entry, str) else getattr(entry, "name", str(entry))

        # Skip archive dirs, overview/abstract files
        base_name = name.rsplit("/", 1)[-1] if "/" in name else name
        if base_name in _SKIP_NAMES or base_name.startswith("_"):
            result.skipped += 1
            continue

        if not _EPISODE_FILENAME_RE.match(base_name):
            result.skipped += 1
            continue

        v1_uri = f"{v1_dir}/{base_name}" if not name.startswith("viking://") else name
        v2_uri = f"{v2_dir}/{base_name}"

        try:
            # Check if already migrated
            try:
                existing = await viking_fs.read_file(v2_uri, ctx=ctx)
                if existing:
                    logger.debug("Already migrated, skipping: %s", base_name)
                    result.skipped += 1
                    continue
            except Exception:
                pass  # File doesn't exist at v2 path, proceed with migration

            # Read v1 content
            content = await viking_fs.read_file(v1_uri, ctx=ctx)
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")

            if not content or not content.strip():
                result.skipped += 1
                continue

            if dry_run:
                logger.info("[DRY RUN] Would migrate: %s -> %s", v1_uri, v2_uri)
                result.migrated += 1
                continue

            # Write to v2 path
            await viking_fs.write_file(v2_uri, content, ctx=ctx)

            # Enqueue for vector embedding
            if vikingdb:
                account_id = ctx.account_id if ctx else "default"
                try:
                    await _enqueue_episode_embedding(
                        v2_uri, content, user_space, account_id, vikingdb
                    )
                    logger.info("Migrated + enqueued: %s -> %s", v1_uri, v2_uri)
                except Exception as embed_err:
                    logger.warning(
                        "Migrated but embedding failed for %s: %s", v2_uri, embed_err
                    )
            else:
                logger.info("Migrated (no embedding): %s -> %s", v1_uri, v2_uri)

            result.migrated += 1

        except Exception as e:
            err_msg = f"Failed to migrate {base_name}: {e}"
            logger.error(err_msg)
            result.errors.append(err_msg)

    mode = "DRY RUN" if dry_run else "LIVE"
    logger.info(
        "[%s] Migration complete: migrated=%d, skipped=%d, errors=%d",
        mode,
        result.migrated,
        result.skipped,
        len(result.errors),
    )
    return result
