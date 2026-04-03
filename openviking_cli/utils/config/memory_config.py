# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class TrivialFilterConfig(BaseModel):
    """Configuration for filtering trivial/automated sessions from memory extraction.

    Prevents cron jobs, heartbeats, and other scheduled runs from polluting
    user memory with system/infrastructure entities.
    """

    enabled: bool = Field(
        default=False,
        description="Enable trivial session filtering for memory extraction",
    )
    patterns: List[str] = Field(
        default_factory=lambda: [
            "heartbeat",
            "heartbeat_ok",
            "health check",
            "health_check",
            "system check",
            "system status",
            "ping",
        ],
        description=(
            "Case-insensitive keyword patterns. If any pattern appears in the "
            "session messages, memory extraction is skipped for that session."
        ),
    )
    min_content_chars: int = Field(
        default=200,
        description=(
            "Minimum characters of substantive content for a short session "
            "(<=3 messages) to be considered non-trivial."
        ),
    )
    min_message_count: int = Field(
        default=3,
        description="Message count threshold for the min_content_chars check",
    )

    model_config = {"extra": "forbid"}


class MemoryConfig(BaseModel):
    """Memory configuration for OpenViking."""

    version: str = Field(
        default="v1",
        description="Memory implementation version: 'v1' (legacy) or 'v2' (new templating system)",
    )
    agent_scope_mode: str = Field(
        default="user+agent",
        description=(
            "Agent memory namespace mode: 'user+agent' keeps agent memory isolated by "
            "(user_id, agent_id), while 'agent' shares agent memory across users of the same agent."
        ),
    )

    custom_templates_dir: str = Field(
        default="",
        description="Custom memory templates directory. If set, templates from this directory will be loaded in addition to built-in templates",
    )

    trivial_filter: TrivialFilterConfig = Field(
        default_factory=TrivialFilterConfig,
        description="Filter trivial/automated sessions (heartbeats, cron jobs) from memory extraction",
    )

    model_config = {"extra": "forbid"}

    @field_validator("agent_scope_mode")
    @classmethod
    def validate_agent_scope_mode(cls, value: str) -> str:
        if value not in {"user+agent", "agent"}:
            raise ValueError("memory.agent_scope_mode must be 'user+agent' or 'agent'")
        return value

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MemoryConfig":
        """Create configuration from dictionary."""
        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
