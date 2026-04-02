# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Debug endpoints for OpenViking HTTP Server.

Provides debug API for system diagnostics.
- /api/v1/debug/health - Quick health check
- /api/v1/debug/vector/scroll - Paginated vector records
- /api/v1/debug/vector/count - Count vector records
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext, Role
from openviking.server.models import ErrorInfo, Response
from openviking.storage import VikingDBManagerProxy

router = APIRouter(prefix="/api/v1/debug", tags=["debug"])


@router.get("/health")
async def debug_health(
    _ctx: RequestContext = Depends(get_request_context),
):
    """Quick health check."""
    service = get_service()
    is_healthy = service.debug.is_healthy()
    return Response(status="ok", result={"healthy": is_healthy})


@router.get("/vector/scroll")
async def debug_vector_scroll(
    limit: int = Query(100, ge=1, le=1000),
    cursor: Optional[str] = None,
    uri: Optional[str] = None,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Get paginated vector records with tenant isolation."""
    service = get_service()
    if not service.vikingdb_manager:
        return Response(
            status="error",
            error=ErrorInfo(code="NO_VECTOR_DB", message="Vector DB not initialized"),
        )

    proxy = VikingDBManagerProxy(service.vikingdb_manager, _ctx)

    filter_expr = None
    if uri:
        filter_expr = {"op": "must", "field": "uri", "conds": [uri]}

    records, next_cursor = await proxy.scroll(filter=filter_expr, limit=limit, cursor=cursor)

    return Response(status="ok", result={"records": records, "next_cursor": next_cursor})


@router.get("/vector/count")
async def debug_vector_count(
    filter: Optional[str] = None,
    uri: Optional[str] = None,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Get count of vector records with tenant isolation."""
    import json

    service = get_service()
    if not service.vikingdb_manager:
        return Response(
            status="error",
            error=ErrorInfo(code="NO_VECTOR_DB", message="Vector DB not initialized"),
        )

    proxy = VikingDBManagerProxy(service.vikingdb_manager, _ctx)

    filter_expr = None
    if filter:
        try:
            filter_expr = json.loads(filter)
        except json.JSONDecodeError:
            return Response(
                status="error",
                error=ErrorInfo(code="INVALID_FILTER", message="Invalid filter JSON"),
            )

    if uri:
        uri_filter = {"op": "must", "field": "uri", "conds": [uri]}
        if filter_expr:
            # For combining filters, we should use And from expr, but for simplicity, let's use RawDSL for now
            from openviking.storage.expr import And, RawDSL

            if isinstance(filter_expr, dict):
                filter_expr = RawDSL(filter_expr)
            uri_filter = RawDSL(uri_filter)
            filter_expr = And([filter_expr, uri_filter])
        else:
            filter_expr = uri_filter

    count = await proxy.count(filter=filter_expr)
    return Response(status="ok", result={"count": count})


@router.post("/distill/dry-run")
async def debug_distill_dry_run(
    scope: str = Query(
        "viking://user/default",
        description="Scope URI to consolidate (e.g. viking://user/default)",
    ),
    subdirectory: str = Query(
        "entities",
        description="Memory subdirectory to scan (e.g. entities, cases)",
    ),
    similarity_threshold: Optional[float] = Query(
        None,
        description="Override cosine similarity threshold (default from config, typically 0.85)",
    ),
    min_cluster_size: Optional[int] = Query(
        None,
        description="Override minimum cluster size (default from config, typically 3)",
    ),
    ctx: RequestContext = Depends(get_request_context),
):
    """Dry-run consolidation on a memory subdirectory.

    Returns cluster analysis without writing any files.
    """
    from openviking.session.distiller import PatternDistiller
    from openviking_cli.utils.config import get_openviking_config

    service = get_service()
    if not service.vikingdb_manager:
        return Response(
            status="error",
            error=ErrorInfo(code="NO_VECTOR_DB", message="Vector DB not initialized"),
        )

    config = get_openviking_config()
    threshold = similarity_threshold or config.distillation.consolidation_similarity_threshold
    cluster_size = min_cluster_size or config.distillation.consolidation_min_cluster_size

    distiller = PatternDistiller(
        vikingdb=service.vikingdb_manager,
        viking_fs=service.viking_fs,
        similarity_threshold=threshold,
        min_cluster_size=cluster_size,
        pattern_dedup_threshold=config.distillation.consolidation_pattern_dedup_threshold,
    )

    # Override ctx role to ROOT for background operation.
    ctx = RequestContext(user=ctx.user, role=Role.ROOT)

    result = await distiller.consolidate(
        scope, ctx, dry_run=True, subdirectory=subdirectory,
    )

    return Response(
        status="ok",
        result={
            "scope": scope,
            "subdirectory": subdirectory,
            "similarity_threshold": threshold,
            "min_cluster_size": cluster_size,
            "scanned": result.scanned,
            "clusters_found": result.clusters_found,
            "patterns_would_create": result.patterns_created,
            "errors": result.errors,
        },
    )


@router.post("/distill/execute")
async def debug_distill_execute(
    scope: str = Query(
        "viking://user/default",
        description="Scope URI to consolidate",
    ),
    subdirectory: str = Query(
        "entities",
        description="Memory subdirectory to scan",
    ),
    similarity_threshold: Optional[float] = Query(None),
    min_cluster_size: Optional[int] = Query(None),
    ctx: RequestContext = Depends(get_request_context),
):
    """Execute consolidation on a memory subdirectory (writes pattern files)."""
    from openviking.session.distiller import PatternDistiller
    from openviking_cli.utils.config import get_openviking_config

    service = get_service()
    if not service.vikingdb_manager:
        return Response(
            status="error",
            error=ErrorInfo(code="NO_VECTOR_DB", message="Vector DB not initialized"),
        )

    config = get_openviking_config()
    threshold = similarity_threshold or config.distillation.consolidation_similarity_threshold
    cluster_size = min_cluster_size or config.distillation.consolidation_min_cluster_size

    distiller = PatternDistiller(
        vikingdb=service.vikingdb_manager,
        viking_fs=service.viking_fs,
        similarity_threshold=threshold,
        min_cluster_size=cluster_size,
        pattern_dedup_threshold=config.distillation.consolidation_pattern_dedup_threshold,
    )

    ctx = RequestContext(user=ctx.user, role=Role.ROOT)

    result = await distiller.consolidate(
        scope, ctx, dry_run=False, subdirectory=subdirectory,
    )

    return Response(
        status="ok",
        result={
            "scope": scope,
            "subdirectory": subdirectory,
            "similarity_threshold": threshold,
            "min_cluster_size": cluster_size,
            "scanned": result.scanned,
            "clusters_found": result.clusters_found,
            "patterns_created": result.patterns_created,
            "skipped_duplicates": result.skipped_duplicates,
            "pattern_uris": result.pattern_uris,
            "errors": result.errors,
        },
    )


@router.post("/decay/dry-run")
async def debug_decay_dry_run(
    scope: str = Query(
        "viking://user/default/memories/",
        description="Scope URI to scan for cold memories",
    ),
    min_age_days: Optional[int] = Query(
        None, description="Override minimum age in days (default from config)",
    ),
    threshold: Optional[float] = Query(
        None, description="Override hotness threshold (default 0.1)",
    ),
    ctx: RequestContext = Depends(get_request_context),
):
    """Dry-run decay scan: show which memories would be archived."""
    from openviking.session.memory_archiver import MemoryArchiver
    from openviking_cli.utils.config import get_openviking_config

    service = get_service()
    if not service.vikingdb_manager:
        return Response(
            status="error",
            error=ErrorInfo(code="NO_VECTOR_DB", message="Vector DB not initialized"),
        )

    config = get_openviking_config()
    age = min_age_days if min_age_days is not None else config.distillation.decay_min_age_days
    thresh = threshold if threshold is not None else MemoryArchiver.DEFAULT_THRESHOLD

    archiver = MemoryArchiver(
        viking_fs=service.viking_fs,
        storage=service.vikingdb_manager,
        threshold=thresh,
        min_age_days=age,
    )

    ctx = RequestContext(user=ctx.user, role=Role.ROOT)
    candidates = await archiver.scan(scope, ctx=ctx)

    return Response(
        status="ok",
        result={
            "scope": scope,
            "min_age_days": age,
            "threshold": thresh,
            "total_candidates": len(candidates),
            "candidates": [
                {
                    "uri": c.uri,
                    "score": round(c.score, 4),
                    "active_count": c.active_count,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                    "context_type": c.context_type,
                }
                for c in candidates[:50]
            ],
        },
    )
