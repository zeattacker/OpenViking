# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Search endpoints for OpenViking HTTP Server."""

import asyncio
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.server.telemetry import run_operation
from openviking.telemetry import TelemetryRequest
from openviking_cli.utils import get_logger

logger = get_logger(__name__)


def _track_recall_background(ctx: RequestContext, uris: List[str]) -> None:
    """Fire-and-forget: increment active_count for recalled URIs."""
    if not uris:
        return

    async def _do_track() -> None:
        try:
            service = get_service()
            if service.vikingdb_manager:
                await service.vikingdb_manager.increment_active_count(ctx, uris)
        except Exception as e:
            logger.debug("Background track-recall failed: %s", e)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_do_track())
    except RuntimeError:
        pass


def _extract_uris_from_result(result: Any) -> List[str]:
    """Extract unique URIs from a FindResult (dict or object)."""
    uris: List[str] = []
    seen: set = set()

    if isinstance(result, dict):
        for key in ("memories", "resources", "skills"):
            for item in result.get(key, []):
                uri = item.get("uri", "") if isinstance(item, dict) else getattr(item, "uri", "")
                if uri and uri not in seen:
                    uris.append(uri)
                    seen.add(uri)
    else:
        for key in ("memories", "resources", "skills"):
            for item in getattr(result, key, []):
                uri = getattr(item, "uri", "")
                if uri and uri not in seen:
                    uris.append(uri)
                    seen.add(uri)
    return uris


def _sanitize_floats(obj: Any) -> Any:
    """Recursively replace inf/nan with 0.0 to ensure JSON compliance."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return 0.0
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj

router = APIRouter(prefix="/api/v1/search", tags=["search"])


class FindRequest(BaseModel):
    """Request model for find."""

    query: str
    target_uri: str = ""
    limit: int = 10
    node_limit: Optional[int] = None
    score_threshold: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None
    telemetry: TelemetryRequest = False


class SearchRequest(BaseModel):
    """Request model for search with session."""

    query: str
    target_uri: str = ""
    session_id: Optional[str] = None
    limit: int = 10
    node_limit: Optional[int] = None
    score_threshold: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None
    telemetry: TelemetryRequest = False


class TrackRecallRequest(BaseModel):
    """Request model for recall tracking."""

    uris: list[str]


@router.post("/track-recall")
async def track_recall(
    request: TrackRecallRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Increment active_count for recalled memory URIs."""
    service = get_service()
    updated = await service.vikingdb_manager.increment_active_count(_ctx, request.uris)
    return Response(status="ok", result={"updated": updated})


class GrepRequest(BaseModel):
    """Request model for grep."""

    uri: str
    pattern: str
    case_insensitive: bool = False
    node_limit: Optional[int] = None


class GlobRequest(BaseModel):
    """Request model for glob."""

    pattern: str
    uri: str = "viking://"
    node_limit: Optional[int] = None


@router.post("/find")
async def find(
    request: FindRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Semantic search without session context."""
    service = get_service()
    actual_limit = request.node_limit if request.node_limit is not None else request.limit
    execution = await run_operation(
        operation="search.find",
        telemetry=request.telemetry,
        fn=lambda: service.search.find(
            query=request.query,
            ctx=_ctx,
            target_uri=request.target_uri,
            limit=actual_limit,
            score_threshold=request.score_threshold,
            filter=request.filter,
        ),
    )
    result = execution.result
    # Auto track-recall for returned URIs (fire-and-forget).
    _track_recall_background(_ctx, _extract_uris_from_result(result))
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    result = _sanitize_floats(result)
    return Response(
        status="ok",
        result=result,
        telemetry=execution.telemetry,
    ).model_dump(exclude_none=True)


@router.post("/search")
async def search(
    request: SearchRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Semantic search with optional session context."""
    service = get_service()

    async def _search():
        session = None
        if request.session_id:
            session = service.sessions.session(_ctx, request.session_id)
            await session.load()
        actual_limit = request.node_limit if request.node_limit is not None else request.limit
        return await service.search.search(
            query=request.query,
            ctx=_ctx,
            target_uri=request.target_uri,
            session=session,
            limit=actual_limit,
            score_threshold=request.score_threshold,
            filter=request.filter,
        )

    execution = await run_operation(
        operation="search.search",
        telemetry=request.telemetry,
        fn=_search,
    )
    result = execution.result
    # Auto track-recall for returned URIs (fire-and-forget).
    _track_recall_background(_ctx, _extract_uris_from_result(result))
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    result = _sanitize_floats(result)
    return Response(
        status="ok",
        result=result,
        telemetry=execution.telemetry,
    ).model_dump(exclude_none=True)


@router.post("/grep")
async def grep(
    request: GrepRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Content search with pattern."""
    service = get_service()
    result = await service.fs.grep(
        request.uri,
        request.pattern,
        ctx=_ctx,
        case_insensitive=request.case_insensitive,
        node_limit=request.node_limit,
    )
    return Response(status="ok", result=result)


@router.post("/glob")
async def glob(
    request: GlobRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """File pattern matching."""
    service = get_service()
    result = await service.fs.glob(
        request.pattern, ctx=_ctx, uri=request.uri, node_limit=request.node_limit
    )
    return Response(status="ok", result=result)
