# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Distillation Scheduler for OpenViking.

Periodically runs pattern consolidation and memory decay (archival).
Modeled after WatchScheduler's asyncio periodic task pattern.
"""

import asyncio
from typing import Any, List, Optional

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
                    memories_uri = f"{scope}/memories/"
                    result = await self._archiver.scan_and_archive(memories_uri, ctx=ctx)
                    if result.archived > 0:
                        logger.info(
                            "[DistillationScheduler] Archived %d cold memories for %s",
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

    def _make_ctx(self) -> RequestContext:
        """Create a RequestContext for background operations."""
        from openviking_cli.session.user_id import UserIdentifier

        user = UserIdentifier(
            self._config.default_account or "default",
            self._config.default_user or "default",
            self._config.default_agent or "default",
        )
        return RequestContext(user=user, role=Role.ROOT)
