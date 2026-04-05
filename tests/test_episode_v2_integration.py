# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Tests for episodic memory v2 integration.

Covers: config toggle, v2 storage path, dedup config, trivial filter word-boundary,
category-based recall boost, and v1→v2 migration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openviking_cli.utils.config.memory_config import (
    EpisodeConfig,
    MemoryConfig,
    RecallConfig,
)


# ── Config Tests ──


class TestEpisodeConfig:
    def test_defaults(self):
        cfg = EpisodeConfig()
        assert cfg.enabled is True
        assert cfg.dedup_skip_threshold == 0.92
        assert cfg.dedup_evolve_threshold == 0.75
        assert cfg.min_messages == 2

    def test_disabled(self):
        cfg = EpisodeConfig(enabled=False)
        assert cfg.enabled is False

    def test_custom_thresholds(self):
        cfg = EpisodeConfig(dedup_skip_threshold=0.95, dedup_evolve_threshold=0.80)
        assert cfg.dedup_skip_threshold == 0.95
        assert cfg.dedup_evolve_threshold == 0.80

    def test_threshold_validation_evolve_must_be_less_than_skip(self):
        with pytest.raises(ValueError, match="dedup_evolve_threshold must be less"):
            EpisodeConfig(dedup_skip_threshold=0.80, dedup_evolve_threshold=0.90)

    def test_threshold_validation_equal_not_allowed(self):
        with pytest.raises(ValueError, match="dedup_evolve_threshold must be less"):
            EpisodeConfig(dedup_skip_threshold=0.80, dedup_evolve_threshold=0.80)


class TestRecallConfig:
    def test_defaults(self):
        cfg = RecallConfig()
        assert cfg.category_boosts["episodes"] == 0.15
        assert cfg.category_boosts["events"] == 0.05
        assert cfg.category_boosts.get("entities", 0.0) == 0.0

    def test_custom_boosts(self):
        cfg = RecallConfig(category_boosts={"episodes": 0.30, "tools": -0.10})
        assert cfg.category_boosts["episodes"] == 0.30
        assert cfg.category_boosts["tools"] == -0.10


class TestMemoryConfigWithEpisodes:
    def test_memory_config_has_episodes(self):
        cfg = MemoryConfig()
        assert isinstance(cfg.episodes, EpisodeConfig)
        assert cfg.episodes.enabled is True

    def test_memory_config_has_recall(self):
        cfg = MemoryConfig()
        assert isinstance(cfg.recall, RecallConfig)

    def test_from_dict_with_episodes(self):
        cfg = MemoryConfig.from_dict({
            "version": "v2",
            "episodes": {"enabled": False, "min_messages": 5},
            "recall": {"category_boosts": {"episodes": 0.25}},
        })
        assert cfg.episodes.enabled is False
        assert cfg.episodes.min_messages == 5
        assert cfg.recall.category_boosts["episodes"] == 0.25


# ── EpisodeIndexer Tests ──


class TestEpisodeIndexerV2Path:
    """Test that episode URIs use the v2 memories/episodes path."""

    async def test_generate_episode_uses_v2_path(self):
        """Generated episodes should be stored at memories/episodes/."""
        from openviking.session.episode_indexer import EpisodeIndexer

        # Mock config for both __init__ and generate_episode
        mock_config = MagicMock()
        mock_config.memory.episodes = EpisodeConfig(enabled=True)
        mock_config.memory.trivial_filter.enabled = False
        mock_config.memory.trivial_filter.patterns = []
        mock_config.memory.trivial_filter.min_content_chars = 200
        mock_config.memory.trivial_filter.min_message_count = 3
        mock_config.default_language = "auto"
        mock_config.vlm.is_available.return_value = True
        mock_config.vlm.get_completion_async = AsyncMock(
            return_value="# Episode: Test Episode\n\n## Summary\nTest summary."
        )
        mock_config.embedding.get_embedder.return_value = None

        with patch(
            "openviking.session.episode_indexer.get_openviking_config",
            return_value=mock_config,
        ):
            indexer = EpisodeIndexer(vikingdb=None)

        # Mock VikingFS
        mock_fs = AsyncMock()
        mock_fs.write_file = AsyncMock()

        # Mock messages
        msg1 = MagicMock()
        msg1.role = "user"
        msg1.content = (
            "Tell me about database migrations in detail please. "
            "I need to understand schema versioning, rollback strategies, "
            "and how to handle data transforms safely in production."
        )
        msg2 = MagicMock()
        msg2.role = "assistant"
        msg2.content = (
            "Database migrations involve schema changes and data transforms. "
            "Key strategies include versioned migration files, blue-green deployments, "
            "and careful rollback procedures to ensure data integrity."
        )

        mock_ctx = MagicMock()
        mock_ctx.account_id = "test-account"
        mock_ctx.user = MagicMock()
        mock_ctx.user.user_space_name.return_value = "default"
        mock_ctx.user.agent_id = "test-agent"

        mock_user = MagicMock()
        mock_user.user_space_name.return_value = "default"

        with (
            patch(
                "openviking.session.episode_indexer.get_openviking_config",
                return_value=mock_config,
            ),
            patch(
                "openviking.session.episode_indexer.get_viking_fs",
                return_value=mock_fs,
            ),
            patch(
                "openviking.session.episode_indexer.render_prompt",
                return_value="test prompt",
            ),
        ):
            result = await indexer.generate_episode(
                messages=[msg1, msg2],
                user=mock_user,
                session_id="test-session-123",
                ctx=mock_ctx,
            )

        assert result is not None
        assert "memories/episodes/" in result.uri
        assert result.uri.startswith("viking://user/default/memories/episodes/ep_")
        assert result.category == "episodes"
        assert result.context_type == "memory"


class TestEpisodeIndexerConfigDisabled:
    """Test that disabled config prevents episode generation."""

    async def test_disabled_returns_none(self):
        from openviking.session.episode_indexer import EpisodeIndexer

        mock_config = MagicMock()
        mock_config.memory.episodes = EpisodeConfig(enabled=False)
        mock_config.embedding.get_embedder.return_value = None

        with patch(
            "openviking.session.episode_indexer.get_openviking_config",
            return_value=mock_config,
        ):
            indexer = EpisodeIndexer(vikingdb=None)

        msg = MagicMock()
        msg.role = "user"
        msg.content = "Hello world"

        mock_ctx = MagicMock()

        with patch(
            "openviking.session.episode_indexer.get_openviking_config",
            return_value=mock_config,
        ):
            result = await indexer.generate_episode(
                messages=[msg, msg, msg],
                user=MagicMock(),
                session_id="test",
                ctx=mock_ctx,
            )
        assert result is None


# ── Trivial Filter Tests ──


class TestTrivialFilterWordBoundary:
    """Test that trivial filter uses word-boundary matching."""

    def test_ping_does_not_match_shopping(self):
        from openviking.session.episode_indexer import EpisodeIndexer

        # Mock config with trivial filter that includes "ping"
        mock_config = MagicMock()
        mock_config.memory.trivial_filter.patterns = ["ping"]
        mock_config.memory.trivial_filter.min_content_chars = 200
        mock_config.memory.trivial_filter.min_message_count = 3

        with patch(
            "openviking.session.episode_indexer.get_openviking_config",
            return_value=mock_config,
        ):
            # "shopping" should NOT be flagged as trivial
            assert EpisodeIndexer._is_trivial("I went shopping today", 5) is False
            # "ping" alone should be flagged
            assert EpisodeIndexer._is_trivial("ping", 5) is True
            # "helping" should NOT be flagged
            assert EpisodeIndexer._is_trivial("I was helping the team", 5) is False

    def test_heartbeat_does_not_match_compound(self):
        from openviking.session.episode_indexer import EpisodeIndexer

        mock_config = MagicMock()
        mock_config.memory.trivial_filter.patterns = ["heartbeat"]
        mock_config.memory.trivial_filter.min_content_chars = 200
        mock_config.memory.trivial_filter.min_message_count = 3

        with patch(
            "openviking.session.episode_indexer.get_openviking_config",
            return_value=mock_config,
        ):
            assert EpisodeIndexer._is_trivial("heartbeat check ok", 5) is True
            assert EpisodeIndexer._is_trivial("my heartbeat is fast", 5) is True


# ── Category Boost Tests ──


class TestCategoryBoost:
    """Test category-based score boost in retriever."""

    def test_get_category_boost_from_config(self):
        from openviking.retrieve.hierarchical_retriever import HierarchicalRetriever

        retriever = HierarchicalRetriever(
            storage=MagicMock(), embedder=None, rerank_config=None
        )

        mock_config = MagicMock()
        mock_config.memory.recall.category_boosts = {"episodes": 0.20, "events": 0.10}

        with patch(
            "openviking_cli.utils.config.get_openviking_config",
            return_value=mock_config,
        ):
            assert retriever._get_category_boost("episodes") == 0.20
            assert retriever._get_category_boost("events") == 0.10
            assert retriever._get_category_boost("entities") == 0.0

    def test_get_category_boost_fallback(self):
        from openviking.retrieve.hierarchical_retriever import HierarchicalRetriever

        retriever = HierarchicalRetriever(
            storage=MagicMock(), embedder=None, rerank_config=None
        )

        # When config is unavailable, use defaults
        with patch(
            "openviking_cli.utils.config.get_openviking_config",
            side_effect=Exception("no config"),
        ):
            assert retriever._get_category_boost("episodes") == 0.15
            assert retriever._get_category_boost("unknown") == 0.0


# ── Migration Tests ──


class TestEpisodeMigrator:
    async def test_migrate_dry_run(self):
        from openviking.session.episode_migrator import migrate_v1_episodes

        mock_fs = AsyncMock()
        mock_fs.ls = AsyncMock(return_value=[
            "ep_abc12345-678_20260324T120000.md",
            "ep_def98765-432_20260325T140000.md",
            ".overview.md",
            "_archive",
        ])
        mock_fs.read_file = AsyncMock(
            side_effect=[
                # First call: check v2 exists -> raise (not migrated)
                Exception("not found"),
                # Second call: read v1 content
                "# Episode: Test\n\n## Summary\nTest content.",
                # Third call: check v2 exists -> raise
                Exception("not found"),
                # Fourth call: read v1 content
                "# Episode: Another\n\n## Summary\nMore content.",
            ]
        )

        mock_ctx = MagicMock()

        with patch(
            "openviking.session.episode_migrator.get_viking_fs",
            return_value=mock_fs,
        ):
            result = await migrate_v1_episodes(
                user_space="default", dry_run=True, ctx=mock_ctx
            )

        assert result.migrated == 2
        assert result.skipped == 2  # .overview.md + _archive
        assert len(result.errors) == 0
        # Dry run should NOT call write_file
        mock_fs.write_file.assert_not_called()

    async def test_migrate_no_viking_fs(self):
        from openviking.session.episode_migrator import migrate_v1_episodes

        with patch(
            "openviking.session.episode_migrator.get_viking_fs",
            return_value=None,
        ):
            result = await migrate_v1_episodes()

        assert len(result.errors) == 1
        assert "VikingFS not available" in result.errors[0]
