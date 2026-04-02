# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Distillation Scheduler for OpenViking.

Periodically runs pattern consolidation and memory decay (archival).
Modeled after WatchScheduler's asyncio periodic task pattern.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from openviking.server.identity import RequestContext, Role
from openviking.session.distiller import PatternDistiller
from openviking.session.memory_archiver import MemoryArchiver
from openviking.storage import VikingDBManager
from openviking_cli.utils import get_logger
from openviking_cli.utils.config.open_viking_config import OpenVikingConfig

logger = get_logger(__name__)


class DistillationScheduler:
    """Schedules periodic distillation (consolidation + decay) tasks."""

    def __init__(
        self,
        distiller: PatternDistiller,
        archiver: MemoryArchiver,
        config: OpenVikingConfig,
        vikingdb: VikingDBManager,
    ):
        self._distiller = distiller
        self._archiver = archiver
        self._config = config
        self._vikingdb = vikingdb
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start background consolidation and decay loops."""
        if self._running:
            logger.warning("[DistillationScheduler] Already running")
            return

        self._running = True
        distill_cfg = self._config.distillation

        if distill_cfg.consolidation_enabled:
            self._tasks.append(asyncio.create_task(self._consolidation_loop()))
            logger.info(
                "[DistillationScheduler] Consolidation loop started "
                "(interval=%dh)",
                distill_cfg.consolidation_interval_hours,
            )

        if distill_cfg.decay_enabled:
            self._tasks.append(asyncio.create_task(self._decay_loop()))
            logger.info(
                "[DistillationScheduler] Decay loop started "
                "(interval=%dh)",
                distill_cfg.decay_check_interval_hours,
            )

        if distill_cfg.semantic_regen_enabled:
            self._tasks.append(asyncio.create_task(self._semantic_regen_loop()))
            logger.info(
                "[DistillationScheduler] Semantic regen loop started "
                "(daily at %02d:00 UTC)",
                distill_cfg.semantic_regen_hour_utc,
            )

        if distill_cfg.archive_gc_enabled:
            self._tasks.append(asyncio.create_task(self._archive_gc_loop()))
            logger.info(
                "[DistillationScheduler] Archive GC loop started "
                "(interval=%dh, max_age=%dd)",
                distill_cfg.archive_gc_interval_hours,
                distill_cfg.archive_gc_max_age_days,
            )

    async def stop(self) -> None:
        """Stop all background loops."""
        if not self._running:
            return

        self._running = False
        for t in self._tasks:
            t.cancel()

        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("[DistillationScheduler] Stopped")

    async def _consolidation_loop(self) -> None:
        """Periodically consolidate similar case memories into patterns."""
        distill_cfg = self._config.distillation
        interval_secs = distill_cfg.consolidation_interval_hours * 3600
        first_run = True

        while self._running:
            if first_run:
                # Run immediately on startup, then switch to interval
                first_run = False
                await asyncio.sleep(30)  # brief delay for full initialization
            else:
                try:
                    await asyncio.sleep(interval_secs)
                except asyncio.CancelledError:
                    break

            if not self._running:
                break

            try:
                directories = distill_cfg.consolidation_directories or ["cases"]
                # Skip agent scopes — too few cases per agent to form meaningful
                # clusters.  Consolidation is only valuable for user-scoped
                # memories (entities, events) where file counts grow large.
                scopes = await self._get_user_scopes()
                for scope in scopes:
                    for subdir in directories:
                        ctx = self._make_ctx()
                        result = await self._distiller.consolidate(
                            scope, ctx, dry_run=False, subdirectory=subdir,
                        )
                        if result.patterns_created > 0:
                            logger.info(
                                "[DistillationScheduler] Consolidated %d patterns "
                                "for %s/%s",
                                result.patterns_created,
                                scope,
                                subdir,
                            )
            except Exception as e:
                logger.error(
                    "[DistillationScheduler] Consolidation cycle error: %s", e, exc_info=True
                )

    async def _decay_loop(self) -> None:
        """Periodically archive cold memories."""
        distill_cfg = self._config.distillation
        interval_secs = distill_cfg.decay_check_interval_hours * 3600
        first_run = True

        while self._running:
            if first_run:
                # Run immediately on startup, then switch to interval
                first_run = False
                await asyncio.sleep(60)  # brief delay for full initialization
            else:
                try:
                    await asyncio.sleep(interval_secs)
                except asyncio.CancelledError:
                    break

            if not self._running:
                break

            try:
                scopes = await self._get_decay_scopes()
                for scope in scopes:
                    ctx = self._make_ctx()
                    # Decay memories
                    memories_uri = f"{scope}/memories/"
                    result = await self._archiver.scan_and_archive(memories_uri, ctx=ctx)
                    if result.archived > 0:
                        logger.info(
                            "[DistillationScheduler] Archived %d cold memories for %s",
                            result.archived,
                            scope,
                        )
                    # Decay episodes
                    episodes_uri = f"{scope}/episodes/"
                    result = await self._archiver.scan_and_archive(episodes_uri, ctx=ctx)
                    if result.archived > 0:
                        logger.info(
                            "[DistillationScheduler] Archived %d cold episodes for %s",
                            result.archived,
                            scope,
                        )
            except Exception as e:
                logger.error(
                    "[DistillationScheduler] Decay cycle error: %s", e, exc_info=True
                )

    async def _get_agent_scopes(self) -> List[str]:
        """Enumerate active agent spaces.

        Returns list of scope URIs like ``viking://agent/{space}``.
        """
        return await self._ls_scopes("viking://agent/")

    async def _get_user_scopes(self) -> List[str]:
        """Enumerate user spaces for consolidation.

        Returns list of scope URIs like ``viking://user/{id}``.
        """
        return await self._ls_scopes("viking://user/")

    async def _get_decay_scopes(self) -> List[str]:
        """Enumerate scopes eligible for memory decay.

        Currently handles user scopes only. Decay is scope-agnostic:
        it only checks hotness_score on L2 vectors, so it is safe
        for user memories.
        """
        return await self._ls_scopes("viking://user/")

    async def _ls_scopes(self, root_uri: str) -> List[str]:
        """List child directories under *root_uri* as scope URIs."""
        try:
            from openviking.storage.viking_fs import get_viking_fs

            viking_fs = get_viking_fs()
            if not viking_fs:
                return []

            entries = await viking_fs.ls(root_uri)
            scopes = []
            for entry in entries:
                if isinstance(entry, dict) and entry.get("isDir"):
                    name = entry.get("name", "")
                    if name and not name.startswith("."):
                        scopes.append(f"{root_uri}{name}")
            return scopes
        except Exception as e:
            logger.warning("[DistillationScheduler] Failed to enumerate %s: %s", root_uri, e)
            return []

    _last_file_counts: Dict[str, int] = {}

    async def _semantic_regen_loop(self) -> None:
        """Full semantic overview regeneration at a fixed daily time.

        Runs at ``semantic_regen_hour_utc`` (default 21:00 UTC = 04:00 WIB).
        Only triggers full regen for directories whose file count changed by
        at least ``semantic_regen_min_file_delta`` since the last run.
        """
        distill_cfg = self._config.distillation
        target_hour = distill_cfg.semantic_regen_hour_utc
        min_delta = distill_cfg.semantic_regen_min_file_delta

        while self._running:
            # Sleep until next target time
            now = datetime.now(timezone.utc)
            target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait_secs = (target - now).total_seconds()
            logger.info(
                "[DistillationScheduler] Semantic regen sleeping %.0fs until %s UTC",
                wait_secs,
                target.strftime("%Y-%m-%d %H:%M"),
            )

            try:
                await asyncio.sleep(wait_secs)
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            try:
                await self._run_semantic_regen(min_delta)
            except Exception as e:
                logger.error(
                    "[DistillationScheduler] Semantic regen error: %s", e, exc_info=True
                )

    async def _run_semantic_regen(self, min_delta: int) -> None:
        """Check file counts and enqueue full regen for changed directories."""
        from openviking.storage.queuefs.queue_manager import get_queue_manager
        from openviking.storage.queuefs.semantic_msg import SemanticMsg
        from openviking.storage.viking_fs import get_viking_fs

        viking_fs = get_viking_fs()
        if not viking_fs:
            return

        scopes = await self._get_user_scopes()
        # Memory subdirectories + episodes
        subdirs = ["memories/entities", "memories/events", "memories/preferences", "episodes"]

        for scope in scopes:
            for subdir in subdirs:
                dir_uri = f"{scope}/{subdir}"
                try:
                    entries = await viking_fs.ls(dir_uri)
                    file_count = sum(
                        1 for e in entries
                        if isinstance(e, dict)
                        and not e.get("isDir", False)
                        and not e.get("name", "").startswith(".")
                    )
                except Exception:
                    continue

                prev = self._last_file_counts.get(dir_uri, 0)
                delta = abs(file_count - prev)
                self._last_file_counts[dir_uri] = file_count

                if prev == 0:
                    # First run — record baseline, skip regen
                    logger.info(
                        "[DistillationScheduler] Semantic regen baseline: %s = %d files",
                        dir_uri, file_count,
                    )
                    continue

                if delta < min_delta:
                    logger.debug(
                        "[DistillationScheduler] Semantic regen skip %s (delta=%d < %d)",
                        dir_uri, delta, min_delta,
                    )
                    continue

                logger.info(
                    "[DistillationScheduler] Semantic regen triggered for %s "
                    "(files: %d → %d, delta=%d)",
                    dir_uri, prev, file_count, delta,
                )

                # Enqueue with changes=None to force full LLM regen
                queue_mgr = get_queue_manager()
                semantic_queue = queue_mgr.get_queue(queue_mgr.SEMANTIC)
                ctx = self._make_ctx()
                sem_msg = SemanticMsg(
                    uri=dir_uri,
                    context_type="memory",
                    recursive=True,
                    account_id=ctx.user.account_id,
                    user_id=ctx.user.user_id,
                    agent_id=ctx.user.agent_id,
                    role="root",
                    changes=None,  # Force full regen
                )
                await semantic_queue.enqueue(sem_msg)
                logger.info(
                    "[DistillationScheduler] Enqueued full semantic regen for %s",
                    dir_uri,
                )

    async def _archive_gc_loop(self) -> None:
        """Periodically delete old archived files to reclaim storage."""
        from openviking.storage.viking_fs import get_viking_fs
        from openviking.utils.time_utils import parse_iso_datetime

        distill_cfg = self._config.distillation
        interval_secs = distill_cfg.archive_gc_interval_hours * 3600
        max_age_days = distill_cfg.archive_gc_max_age_days
        first_run = True

        while self._running:
            if first_run:
                first_run = False
                await asyncio.sleep(120)
            else:
                try:
                    await asyncio.sleep(interval_secs)
                except asyncio.CancelledError:
                    break

            if not self._running:
                break

            try:
                viking_fs = get_viking_fs()
                if not viking_fs:
                    continue

                now = datetime.now(timezone.utc)
                cutoff = now - timedelta(days=max_age_days)
                total_deleted = 0

                scopes = await self._get_user_scopes()
                for scope in scopes:
                    ctx = self._make_ctx()
                    # Scan _archive dirs under each memory subdirectory and episodes
                    archive_dirs = [
                        f"{scope}/memories/entities/_archive",
                        f"{scope}/memories/events/_archive",
                        f"{scope}/memories/cases/_archive",
                        f"{scope}/memories/patterns/_archive",
                        f"{scope}/memories/preferences/_archive",
                        f"{scope}/episodes/_archive",
                    ]
                    for archive_dir in archive_dirs:
                        try:
                            entries = await viking_fs.ls(archive_dir, ctx=ctx)
                        except Exception:
                            continue

                        for entry in entries:
                            if not isinstance(entry, dict):
                                continue
                            name = entry.get("name", "")
                            if not name or name.startswith(".") or entry.get("isDir"):
                                continue

                            mod_time_raw = entry.get("modTime", "")
                            if not mod_time_raw:
                                continue

                            try:
                                mod_time = parse_iso_datetime(mod_time_raw)
                            except Exception:
                                continue

                            if mod_time < cutoff:
                                file_uri = entry.get("uri") or f"{archive_dir}/{name}"
                                try:
                                    await viking_fs.rm(file_uri, ctx=ctx)
                                    total_deleted += 1
                                except Exception as e:
                                    logger.warning(
                                        "[DistillationScheduler] Archive GC failed to delete %s: %s",
                                        file_uri,
                                        e,
                                    )

                if total_deleted > 0:
                    logger.info(
                        "[DistillationScheduler] Archive GC deleted %d old files (max_age=%dd)",
                        total_deleted,
                        max_age_days,
                    )
            except Exception as e:
                logger.error(
                    "[DistillationScheduler] Archive GC error: %s", e, exc_info=True
                )

    def _make_ctx(self) -> RequestContext:
        """Create a RequestContext for background operations."""
        from openviking_cli.session.user_id import UserIdentifier

        user = UserIdentifier(
            self._config.default_account or "default",
            self._config.default_user or "default",
            self._config.default_agent or "default",
        )
        return RequestContext(user=user, role=Role.ROOT)
