# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dedup consolidation pipeline in PatternDistiller."""

from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openviking.session.distiller import ConsolidationResult, PatternDistiller


@dataclass
class FakeEmbedResult:
    dense_vector: List[float]


class FakeEmbedder:
    def embed(self, text: str, is_query: bool = False) -> FakeEmbedResult:
        # Deterministic fake: hash text into a 3-dim vector.
        h = hash(text) % 1000
        return FakeEmbedResult(dense_vector=[h / 1000, (h + 1) / 1000, (h + 2) / 1000])


def _make_distiller(
    vikingdb=None,
    viking_fs=None,
    pattern_dedup_threshold: float = 0.90,
) -> PatternDistiller:
    """Build a PatternDistiller with mocked deps."""
    with patch(
        "openviking.session.distiller.get_openviking_config"
    ) as mock_config:
        mock_config.return_value.embedding.get_embedder.return_value = FakeEmbedder()
        distiller = PatternDistiller(
            vikingdb=vikingdb or AsyncMock(),
            viking_fs=viking_fs or AsyncMock(),
            pattern_dedup_threshold=pattern_dedup_threshold,
        )
    return distiller


class TestConsolidationResultFields:
    """ConsolidationResult has the new dedup tracking fields."""

    def test_default_values(self):
        r = ConsolidationResult()
        assert r.skipped_duplicates == 0
        assert r.skipped_stale == 0

    def test_fields_assignable(self):
        r = ConsolidationResult(skipped_duplicates=3, skipped_stale=1)
        assert r.skipped_duplicates == 3
        assert r.skipped_stale == 1


class TestFindDuplicatePattern:
    """_find_duplicate_pattern uses vector search to detect existing duplicates."""

    async def test_no_embedder_returns_none(self):
        distiller = _make_distiller()
        distiller._embedder = None
        result = await distiller._find_duplicate_pattern("content", "viking://agent/s", MagicMock())
        assert result is None

    async def test_no_similar_returns_none(self):
        vikingdb = AsyncMock()
        vikingdb.search_similar_memories = AsyncMock(return_value=[])
        distiller = _make_distiller(vikingdb=vikingdb)

        result = await distiller._find_duplicate_pattern(
            "some content", "viking://agent/space1", MagicMock()
        )
        assert result is None

    async def test_low_score_returns_none(self):
        vikingdb = AsyncMock()
        vikingdb.search_similar_memories = AsyncMock(
            return_value=[
                {"uri": "viking://agent/space1/memories/patterns/old.md", "_score": 0.70}
            ]
        )
        distiller = _make_distiller(vikingdb=vikingdb)

        result = await distiller._find_duplicate_pattern(
            "content", "viking://agent/space1", MagicMock()
        )
        assert result is None

    async def test_high_score_returns_matching_uri(self):
        match_uri = "viking://agent/space1/memories/patterns/consolidated_abc.md"
        vikingdb = AsyncMock()
        vikingdb.search_similar_memories = AsyncMock(
            return_value=[{"uri": match_uri, "_score": 0.95}]
        )
        distiller = _make_distiller(vikingdb=vikingdb)

        result = await distiller._find_duplicate_pattern(
            "content", "viking://agent/space1", MagicMock()
        )
        assert result == match_uri

    async def test_high_score_wrong_prefix_ignored(self):
        vikingdb = AsyncMock()
        vikingdb.search_similar_memories = AsyncMock(
            return_value=[
                {"uri": "viking://agent/space1/memories/cases/foo.md", "_score": 0.99}
            ]
        )
        distiller = _make_distiller(vikingdb=vikingdb)

        result = await distiller._find_duplicate_pattern(
            "content", "viking://agent/space1", MagicMock()
        )
        assert result is None

    async def test_search_failure_returns_none(self):
        vikingdb = AsyncMock()
        vikingdb.search_similar_memories = AsyncMock(side_effect=RuntimeError("db down"))
        distiller = _make_distiller(vikingdb=vikingdb)

        result = await distiller._find_duplicate_pattern(
            "content", "viking://agent/space1", MagicMock()
        )
        assert result is None


class TestArchiveSourceCases:
    """_archive_source_cases moves files to _archive/ subdirectory."""

    async def test_archives_all_uris(self):
        viking_fs = AsyncMock()
        distiller = _make_distiller(viking_fs=viking_fs)
        ctx = MagicMock()

        uris = [
            "viking://agent/s/memories/cases/a.md",
            "viking://agent/s/memories/cases/b.md",
        ]
        await distiller._archive_source_cases(uris, ctx)

        assert viking_fs.mv.call_count == 2
        viking_fs.mv.assert_any_call(
            "viking://agent/s/memories/cases/a.md",
            "viking://agent/s/memories/cases/_archive/a.md",
            ctx=ctx,
        )

    async def test_archive_failure_logs_warning(self):
        viking_fs = AsyncMock()
        viking_fs.mv = AsyncMock(side_effect=OSError("permission denied"))
        distiller = _make_distiller(viking_fs=viking_fs)

        # Should not raise.
        await distiller._archive_source_cases(
            ["viking://agent/s/memories/cases/a.md"], MagicMock()
        )


class TestGetCaseVectorsStaleFiltering:
    """_get_case_vectors filters out stale vectors not present in filesystem."""

    async def test_stale_vector_filtered(self):
        vikingdb = AsyncMock()
        vikingdb.scroll = AsyncMock(
            return_value=(
                [
                    {
                        "uri": "viking://agent/s/memories/cases/live.md",
                        "abstract": "live",
                        "vector": [0.1, 0.2],
                    },
                    {
                        "uri": "viking://agent/s/memories/cases/stale.md",
                        "abstract": "stale",
                        "vector": [0.3, 0.4],
                    },
                ],
                None,
            )
        )
        distiller = _make_distiller(vikingdb=vikingdb)

        # Only "live.md" exists on filesystem.
        fs_entries = [{"name": "live.md"}]
        vectors = await distiller._get_case_vectors(
            "viking://agent/s/memories/cases/",
            MagicMock(),
            fs_entries=fs_entries,
        )

        assert len(vectors) == 1
        assert vectors[0][0] == "viking://agent/s/memories/cases/live.md"

    async def test_no_fs_entries_skips_filter(self):
        vikingdb = AsyncMock()
        vikingdb.scroll = AsyncMock(
            return_value=(
                [
                    {
                        "uri": "viking://agent/s/memories/cases/a.md",
                        "abstract": "a",
                        "vector": [0.1, 0.2],
                    },
                ],
                None,
            )
        )
        distiller = _make_distiller(vikingdb=vikingdb)

        # No fs_entries → all vectors pass.
        vectors = await distiller._get_case_vectors(
            "viking://agent/s/memories/cases/",
            MagicMock(),
        )
        assert len(vectors) == 1
