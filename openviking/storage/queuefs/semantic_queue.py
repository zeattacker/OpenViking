# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""SemanticQueue: Semantic extraction queue."""

import time
from typing import Dict, Optional, Tuple

from openviking_cli.utils.logger import get_logger

from .named_queue import NamedQueue
from .semantic_msg import SemanticMsg

logger = get_logger(__name__)

# Debounce window: skip re-enqueue of the same parent URI within this period.
# Mitigates repeated full-directory recomputation on high-frequency memory writes.
# See: https://github.com/volcengine/OpenViking/issues/769
SEMANTIC_DEDUP_WINDOW_SECONDS = 300  # 5 minutes


class SemanticQueue(NamedQueue):
    """Semantic extraction queue for async generation of .abstract.md and .overview.md."""

    # Track recently enqueued URIs: uri -> (timestamp, merged_changes)
    _recent_enqueues: Dict[str, Tuple[float, Dict[str, set]]] = {}

    async def enqueue(self, msg: SemanticMsg) -> str:
        """Serialize SemanticMsg object and store in queue.

        Applies time-windowed dedup for memory-context messages: if the same
        parent URI was enqueued within SEMANTIC_DEDUP_WINDOW_SECONDS, the new
        message is skipped (its changes are already covered by the pending job).
        """
        if msg.context_type == "memory" and msg.uri:
            now = time.monotonic()
            # Evict stale entries
            stale = [
                uri
                for uri, (ts, _) in self._recent_enqueues.items()
                if now - ts > SEMANTIC_DEDUP_WINDOW_SECONDS
            ]
            for uri in stale:
                del self._recent_enqueues[uri]

            if msg.uri in self._recent_enqueues:
                prev_ts, _ = self._recent_enqueues[msg.uri]
                remaining = int(SEMANTIC_DEDUP_WINDOW_SECONDS - (now - prev_ts))
                logger.info(
                    f"[SemanticQueue] Dedup: skipping re-enqueue for {msg.uri} "
                    f"({remaining}s remaining in window)"
                )
                return ""

            self._recent_enqueues[msg.uri] = (now, {})
            logger.info(
                f"[SemanticQueue] Enqueuing {msg.uri} (dedup window={SEMANTIC_DEDUP_WINDOW_SECONDS}s)"
            )

        return await super().enqueue(msg.to_dict())

    async def dequeue(self) -> Optional[SemanticMsg]:
        """Get message from queue and deserialize to SemanticMsg object."""
        data_dict = await super().dequeue()
        if not data_dict:
            return None

        if "data" in data_dict and isinstance(data_dict["data"], str):
            try:
                return SemanticMsg.from_json(data_dict["data"])
            except Exception as e:
                logger.debug(f"[SemanticQueue] Failed to parse message data: {e}")
                return None

        try:
            return SemanticMsg.from_dict(data_dict)
        except Exception as e:
            logger.debug(f"[SemanticQueue] Failed to create SemanticMsg from dict: {e}")
            return None

    async def peek(self) -> Optional[SemanticMsg]:
        """Peek at queue head message."""
        data_dict = await super().peek()
        if not data_dict:
            return None

        if "data" in data_dict and isinstance(data_dict["data"], str):
            try:
                return SemanticMsg.from_json(data_dict["data"])
            except Exception:
                return None

        try:
            return SemanticMsg.from_dict(data_dict)
        except Exception:
            return None
