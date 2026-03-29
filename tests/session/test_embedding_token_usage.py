# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SessionMeta.embedding_token_usage field (commit ea240a2)."""

import importlib.util
import sys

import pytest


def _load_session_meta():
    """Load SessionMeta directly from session.py, bypassing __init__.py.

    The session package __init__.py pulls in heavy dependencies (litellm,
    volcengine, etc.) that may not be available in lightweight test envs.
    SessionMeta itself is a pure dataclass with no external deps.
    """
    try:
        from openviking.session.session import SessionMeta
        return SessionMeta
    except (ImportError, ModuleNotFoundError):
        import pathlib
        session_py = pathlib.Path(__file__).resolve().parents[2] / "openviking" / "session" / "session.py"
        spec = importlib.util.spec_from_file_location("openviking.session.session", str(session_py))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["openviking.session.session"] = mod
        spec.loader.exec_module(mod)
        return mod.SessionMeta


SessionMeta = _load_session_meta()


class TestSessionMetaEmbeddingTokenUsage:
    """Verify embedding_token_usage in SessionMeta dataclass."""

    def test_default_value(self):
        meta = SessionMeta()
        assert meta.embedding_token_usage == {"total_tokens": 0}

    def test_to_dict_includes_embedding_token_usage(self):
        meta = SessionMeta(
            session_id="test-session",
            embedding_token_usage={"total_tokens": 42},
        )
        d = meta.to_dict()
        assert "embedding_token_usage" in d
        assert d["embedding_token_usage"]["total_tokens"] == 42

    def test_from_dict_with_embedding_data(self):
        data = {
            "session_id": "test-session",
            "embedding_token_usage": {"total_tokens": 100},
        }
        meta = SessionMeta.from_dict(data)
        assert meta.embedding_token_usage["total_tokens"] == 100

    def test_from_dict_without_embedding_data(self):
        data = {"session_id": "test-session"}
        meta = SessionMeta.from_dict(data)
        assert meta.embedding_token_usage == {"total_tokens": 0}

    def test_to_dict_from_dict_roundtrip(self):
        original = SessionMeta(
            session_id="roundtrip",
            embedding_token_usage={"total_tokens": 256},
            llm_token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )
        d = original.to_dict()
        restored = SessionMeta.from_dict(d)
        assert restored.embedding_token_usage == original.embedding_token_usage
        assert restored.llm_token_usage == original.llm_token_usage

    def test_accumulation(self):
        meta = SessionMeta()
        meta.embedding_token_usage["total_tokens"] += 50
        meta.embedding_token_usage["total_tokens"] += 30
        assert meta.embedding_token_usage["total_tokens"] == 80

    def test_combined_token_reporting(self):
        meta = SessionMeta(
            llm_token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            embedding_token_usage={"total_tokens": 200},
        )
        combined_total = (
            meta.llm_token_usage["total_tokens"]
            + meta.embedding_token_usage["total_tokens"]
        )
        assert combined_total == 350
