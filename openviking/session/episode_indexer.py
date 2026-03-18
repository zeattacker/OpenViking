# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Episode Indexer for OpenViking.

Generates structured episode summaries from session conversations and stores them
as searchable episodes alongside existing memories.
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


class EpisodeIndexer:
    """Generates and indexes episode summaries from session conversations."""

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
