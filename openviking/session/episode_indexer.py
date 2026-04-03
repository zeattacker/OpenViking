# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Episode Indexer for OpenViking.

Generates structured episode summaries from session conversations and stores them
as searchable episodes alongside existing memories.
Includes score-based dedup to prevent near-duplicate episodes.
"""

from datetime import datetime, timezone
from typing import List, Optional

from openviking.core.context import Context, Vectorize
from openviking.message import Message
from openviking.prompts import render_prompt
from openviking.server.identity import RequestContext
from openviking.storage.viking_fs import get_viking_fs
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils import get_logger
from openviking_cli.utils.config import get_openviking_config

logger = get_logger(__name__)

# Score-based episode dedup thresholds (no LLM needed).
# Derived from empirical pairwise tests on known duplicate/non-duplicate memories.
EPISODE_SKIP_THRESHOLD = 0.92   # Near-exact duplicate conversation → skip
EPISODE_EVOLVE_THRESHOLD = 0.75  # Same topic, new details → evolve (append)


class EpisodeIndexer:
    """Generates and indexes episode summaries from session conversations."""

    def __init__(self, vikingdb=None):
        """Initialize episode indexer.

        Args:
            vikingdb: VikingDBManager for episode dedup vector search.
                      If None, dedup is skipped (always creates).
        """
        self._vikingdb = vikingdb
        config = get_openviking_config()
        self._embedder = config.embedding.get_embedder()

    @staticmethod
    def _compute_token_budget(message_count: int) -> int:
        """Compute max output tokens based on conversation length."""
        if message_count <= 5:
            return 300
        elif message_count <= 15:
            return 500
        elif message_count <= 40:
            return 800
        else:
            return 1200

    # Fallback patterns used when config is not available or trivial_filter is disabled.
    _TRIVIAL_PATTERNS = [
        "heartbeat",
        "heartbeat_ok",
        "heartteat_ok",
        "health check",
        "health_check",
        "system check",
        "system status",
        "ping",
    ]

    _MIN_CONTENT_CHARS = 200

    @staticmethod
    def _is_trivial(formatted_messages: str, message_count: int) -> bool:
        """Return True if the conversation is too trivial for an episode.

        Uses configurable patterns from memory.trivial_filter when available,
        falls back to hardcoded patterns otherwise.
        """
        from openviking_cli.utils.config import get_openviking_config

        try:
            config = get_openviking_config()
            trivial_config = config.memory.trivial_filter
            patterns = trivial_config.patterns
            min_chars = trivial_config.min_content_chars
            min_msgs = trivial_config.min_message_count
        except Exception:
            patterns = EpisodeIndexer._TRIVIAL_PATTERNS
            min_chars = EpisodeIndexer._MIN_CONTENT_CHARS
            min_msgs = 3

        text_lower = formatted_messages.lower()

        # Check for trivial keyword patterns
        for pattern in patterns:
            if pattern.lower() in text_lower:
                return True

        # Very short conversations with minimal content
        if message_count <= min_msgs and len(formatted_messages) < min_chars:
            return True

        return False

    @staticmethod
    def _format_messages(messages: List[Message]) -> str:
        """Format messages as [role]: content for the prompt."""
        lines = []
        for msg in messages:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            if not content:
                # Try to get text from parts
                parts = getattr(msg, "parts", [])
                text_parts = []
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                    elif hasattr(part, "content") and part.content:
                        text_parts.append(str(part.content))
                content = "\n".join(text_parts) if text_parts else ""
            if content:
                lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)

    async def _find_similar_episode(
        self,
        episode_content: str,
        user: Optional[UserIdentifier],
        ctx: RequestContext,
    ) -> tuple[Optional[Context], float]:
        """Find the most similar existing episode using vector search.

        Returns (similar_episode, score) or (None, 0.0) if no similar episode.
        """
        if not self._vikingdb or not self._embedder:
            return None, 0.0

        try:
            from openviking.models.embedder.base import EmbedResult
            from openviking.storage.expr import And, Eq, In

            embed_result: EmbedResult = self._embedder.embed(
                episode_content[:2000], is_query=True,
            )
            query_vector = embed_result.dense_vector

            user_space = user.user_space_name() if user else "default"
            episodes_prefix = f"viking://user/{user_space}/episodes/"

            conds = [
                Eq("level", 2),
                Eq("account_id", ctx.account_id),
                In("uri", [episodes_prefix]),
            ]
            owner_space = user.user_space_name() if user else None
            if owner_space:
                conds.append(Eq("owner_space", owner_space))

            results = await self._vikingdb.search(
                query_vector=query_vector,
                filter=And(conds),
                limit=3,
                ctx=ctx,
            )

            if not results:
                return None, 0.0

            # Find the top-scoring non-archived result
            best_score = 0.0
            best_ctx = None
            for r in results:
                uri = r.get("uri", "")
                if "/_archive/" in uri:
                    continue
                score = float(r.get("_score", r.get("score", 0)) or 0)
                if score > best_score:
                    best_score = score
                    best_ctx = Context.from_dict(r)

            return best_ctx, best_score

        except Exception as e:
            logger.debug("Episode dedup search failed (non-fatal): %s", e)
            return None, 0.0

    async def _evolve_episode(
        self,
        existing: Context,
        new_content: str,
        ctx: RequestContext,
    ) -> Optional[Context]:
        """Append new episode content to an existing episode file."""
        viking_fs = get_viking_fs()
        if not viking_fs:
            return None

        try:
            old_content = await viking_fs.read_file(existing.uri, ctx=ctx)
            if isinstance(old_content, bytes):
                old_content = old_content.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("Cannot read existing episode %s: %s", existing.uri, e)
            return None

        # Append a separator + new content
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        evolved_content = (
            f"{old_content}\n\n"
            f"---\n"
            f"## Follow-up ({timestamp})\n\n"
            f"{new_content}"
        )

        try:
            await viking_fs.write_file(existing.uri, evolved_content, ctx=ctx)
        except Exception as e:
            logger.error("Failed to write evolved episode %s: %s", existing.uri, e)
            return None

        # Update abstract and vectorize text
        existing.set_vectorize(Vectorize(text=evolved_content))
        meta = dict(existing.meta or {})
        meta["evolution_count"] = int(meta.get("evolution_count", 0)) + 1
        meta["last_confirmed"] = datetime.now(timezone.utc).isoformat()
        existing.meta = meta

        logger.info(
            "Evolved episode %s (evolution_count=%d)",
            existing.uri, meta["evolution_count"],
        )
        return existing

    async def generate_episode(
        self,
        messages: List[Message],
        user: Optional[UserIdentifier],
        session_id: Optional[str],
        ctx: Optional[RequestContext] = None,
    ) -> Optional[Context]:
        """Generate an episode summary from session messages.

        Args:
            messages: List of session messages
            user: User identifier
            session_id: Session identifier
            ctx: Request context

        Returns:
            Context object for the episode, or None if generation fails
        """
        if not messages or not ctx:
            return None

        message_count = len(messages)
        if message_count < 2:
            logger.debug("Skipping episode generation: fewer than 2 messages")
            return None

        formatted_messages = self._format_messages(messages)
        if not formatted_messages.strip():
            logger.debug("Skipping episode generation: no message content")
            return None

        # Skip trivial conversations: heartbeats, system checks, short
        # cron-triggered sessions that don't contain meaningful dialogue.
        if self._is_trivial(formatted_messages, message_count):
            logger.info("Skipping episode generation: trivial conversation")
            return None

        max_output_tokens = self._compute_token_budget(message_count)

        # Determine agent_id
        agent_id = ""
        if ctx and ctx.user:
            agent_id = getattr(ctx.user, "agent_id", "") or ""

        # Determine output language
        config = get_openviking_config()
        output_language = getattr(config, "default_language", "auto") or "auto"

        # Render the episode summary prompt
        prompt = render_prompt(
            "compression.episode_summary",
            {
                "messages": formatted_messages,
                "message_count": message_count,
                "agent_id": agent_id,
                "output_language": output_language,
                "max_output_tokens": max_output_tokens,
            },
        )

        if not prompt:
            logger.error("Failed to render episode summary prompt")
            return None

        # Get LLM completion
        vlm = get_openviking_config().vlm
        if not vlm or not vlm.is_available():
            logger.warning("VLM not available for episode generation")
            return None
        episode_content = await vlm.get_completion_async(prompt)

        if not episode_content or not episode_content.strip():
            logger.warning("Episode generation returned empty content")
            return None

        episode_content = episode_content.strip()

        # --- Episode dedup: score-based, no LLM needed ---
        similar_ep, score = await self._find_similar_episode(
            episode_content, user, ctx,
        )
        if similar_ep and score >= EPISODE_SKIP_THRESHOLD:
            logger.info(
                "Episode dedup: SKIP (score=%.4f >= %.2f) similar=%s",
                score, EPISODE_SKIP_THRESHOLD, similar_ep.uri,
            )
            return None

        if similar_ep and score >= EPISODE_EVOLVE_THRESHOLD:
            logger.info(
                "Episode dedup: EVOLVE (score=%.4f >= %.2f) into=%s",
                score, EPISODE_EVOLVE_THRESHOLD, similar_ep.uri,
            )
            evolved = await self._evolve_episode(similar_ep, episode_content, ctx)
            return evolved  # May be None on failure; caller handles it

        # --- No duplicate found or score too low: create new episode ---

        # Build episode URI
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        session_prefix = (session_id or "unknown")[:12]
        user_space = user.user_space_name() if user else "default"
        episode_filename = f"ep_{session_prefix}_{timestamp}.md"
        episode_uri = f"viking://user/{user_space}/episodes/{episode_filename}"
        parent_uri = f"viking://user/{user_space}/episodes"

        # Write episode file
        viking_fs = get_viking_fs()
        if not viking_fs:
            logger.error("VikingFS not available for episode writing")
            return None

        await viking_fs.write_file(episode_uri, episode_content, ctx=ctx)
        logger.info(f"Episode file written: {episode_uri}")

        # Extract abstract from first meaningful line
        abstract = ""
        for line in episode_content.split("\n"):
            line = line.strip()
            if line.startswith("# Episode:"):
                abstract = line.replace("# Episode:", "").strip()
                break
            elif line.startswith("# ") and not abstract:
                abstract = line.lstrip("# ").strip()
                break
        if not abstract:
            abstract = episode_content[:120].strip()

        # Build Context object
        episode_ctx = Context(
            uri=episode_uri,
            parent_uri=parent_uri,
            is_leaf=True,
            abstract=abstract,
            context_type="memory",
            category="episodes",
            session_id=session_id,
            user=user,
            account_id=ctx.account_id if ctx else "default",
        )

        # Set vectorize text to the full episode content for searchability
        episode_ctx.set_vectorize(Vectorize(text=episode_content))

        return episode_ctx
