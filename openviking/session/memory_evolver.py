# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Memory Evolution for OpenViking.

Enriches existing memories with new supporting evidence without replacing
original content. Used when the deduplicator decides EVOLVE — the candidate
reinforces or supplements an existing memory.
"""

import hashlib
from datetime import datetime, timezone
from typing import Optional

from openviking.core.context import Context, Vectorize
from openviking.prompts import render_prompt
from openviking.server.identity import RequestContext
from openviking_cli.utils import get_logger
from openviking_cli.utils.config import get_openviking_config

from .memory_extractor import CandidateMemory

logger = get_logger(__name__)

MAX_EVOLUTION_COUNT = 3


class MemoryEvolver:
    """Evolves existing memories by appending new evidence."""

    async def evolve(
        self,
        candidate: CandidateMemory,
        target_memory: Context,
        viking_fs,
        ctx: RequestContext,
    ) -> bool:
        """Evolve an existing memory with new evidence from candidate.

        Reads the existing memory content, calls LLM to produce an enriched
        version that preserves the original and appends new evidence, then
        writes back and updates metadata.

        Args:
            candidate: New evidence to incorporate.
            target_memory: Existing memory to enrich.
            viking_fs: VikingFS instance for file I/O.
            ctx: Request context.

        Returns:
            True on success, False on failure.
        """
        # Guard: cap evolution count to prevent infinite reprocessing loops.
        meta = dict(target_memory.meta or {})
        evolution_count = int(meta.get("evolution_count", 0))
        if evolution_count >= MAX_EVOLUTION_COUNT:
            logger.warning(
                "Skipping evolution for %s: evolution_count %d >= cap %d",
                target_memory.uri,
                evolution_count,
                MAX_EVOLUTION_COUNT,
            )
            return False

        try:
            existing_content = await viking_fs.read_file(target_memory.uri, ctx=ctx)
        except Exception as e:
            logger.error("Failed to read existing memory %s: %s", target_memory.uri, e)
            return False

        existing_abstract = (
            getattr(target_memory, "abstract", "")
            or (target_memory.meta or {}).get("abstract", "")
        )

        prompt = render_prompt(
            "compression.memory_evolution",
            {
                "existing_abstract": existing_abstract,
                "existing_content": existing_content or "",
                "new_abstract": candidate.abstract,
                "new_content": candidate.content,
                "category": candidate.category.value,
                "output_language": getattr(candidate, "language", None) or "",
            },
        )

        vlm = get_openviking_config().vlm
        if not vlm or not vlm.is_available():
            logger.warning("VLM not available for memory evolution")
            return False

        try:
            from openviking_cli.utils.llm import parse_json_from_response

            response = await vlm.get_completion_async(prompt)
            payload = parse_json_from_response(response) or {}
            logger.debug("Evolution LLM response: %s", payload)

            evolved_abstract = payload.get("abstract", "")
            evolved_content = payload.get("content", "")

            if not evolved_content:
                logger.warning("Evolution LLM returned empty content for %s", target_memory.uri)
                return False

            # Skip write if content unchanged (prevents reprocessing loop)
            existing_hash = hashlib.md5((existing_content or "").encode()).hexdigest()
            evolved_hash = hashlib.md5(evolved_content.encode()).hexdigest()
            if existing_hash == evolved_hash:
                logger.info(
                    "Evolution produced identical content for %s, skipping write",
                    target_memory.uri,
                )
                return False

            # Write evolved content back
            await viking_fs.write_file(target_memory.uri, evolved_content, ctx=ctx)

            # Update metadata
            meta = dict(target_memory.meta or {})
            evolution_count = int(meta.get("evolution_count", 0))
            meta["evolution_count"] = evolution_count + 1
            meta["last_confirmed"] = datetime.now(timezone.utc).isoformat()
            if payload.get("overview"):
                meta["overview"] = payload["overview"]
            target_memory.meta = meta

            # Update abstract
            if evolved_abstract:
                target_memory.abstract = evolved_abstract

            # Set vectorization text for re-indexing
            target_memory.set_vectorize(Vectorize(text=evolved_content))

            logger.info(
                "Evolved memory %s (evolution_count=%d, abstract=%s)",
                target_memory.uri,
                meta["evolution_count"],
                target_memory.abstract,
            )
            return True

        except Exception as e:
            logger.error("Failed to evolve memory %s: %s", target_memory.uri, e, exc_info=True)
            return False
