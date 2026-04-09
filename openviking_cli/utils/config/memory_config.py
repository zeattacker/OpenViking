# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator


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


class EpisodeConfig(BaseModel):
    """Configuration for episodic memory generation.

    Controls session-level narrative summary generation that runs as a
    separate post-extraction step after the v2 ReAct memory loop.
    """

    enabled: bool = Field(
        default=True,
        description="Enable episode generation during memory extraction",
    )
    dedup_skip_threshold: float = Field(
        default=0.92,
        description=(
            "Cosine similarity threshold at or above which a near-duplicate "
            "episode is skipped entirely."
        ),
    )
    dedup_evolve_threshold: float = Field(
        default=0.75,
        description=(
            "Cosine similarity threshold at or above which an existing episode "
            "is evolved (new content appended as a follow-up)."
        ),
    )
    min_messages: int = Field(
        default=2,
        description="Minimum message count required to trigger episode generation",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_thresholds(self) -> "EpisodeConfig":
        if self.dedup_evolve_threshold >= self.dedup_skip_threshold:
            raise ValueError(
                "episodes.dedup_evolve_threshold must be less than "
                "episodes.dedup_skip_threshold"
            )
        return self


class RecallConfig(BaseModel):
    """Configuration for memory recall ranking.

    Controls category-based score boosts applied during retrieval so that
    different memory types (episodic, semantic, procedural) are ranked
    appropriately for the query context.
    """

    category_boosts: Dict[str, float] = Field(
        default_factory=lambda: {
            "episodes": 0.15,
            "events": 0.05,
        },
        description=(
            "Category-based multiplicative score boosts for memory recall ranking. "
            "Positive values boost, negative values demote. "
            "Categories not listed default to 0.0 (no boost)."
        ),
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

    episodes: EpisodeConfig = Field(
        default_factory=EpisodeConfig,
        description="Episodic memory generation settings (session-level narrative summaries)",
    )

    recall: RecallConfig = Field(
        default_factory=RecallConfig,
        description="Memory recall ranking settings (category-based score boosts)",
    )

    small_model_mode: bool = Field(
        default=False,
        description=(
            "Enable extraction adaptations for small (~8B) models that lack "
            "function calling or nested-JSON-schema instruction following. "
            "When True: forces compact JSON schema (~200 tokens), disables "
            "tool calling (extraction_text_mode), and requests json_object "
            "response_format. Default False preserves existing behavior for "
            "larger instruction-tuned models (Qwen 35B, GPT-4, etc.)."
        ),
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
