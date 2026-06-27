"""app/api/v1/endpoints/traces.py"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import User, LLMTrace
from app.repositories import TraceRepository
from app.schemas.traces import TraceCreate, TraceResponse, FeedbackCreate
from app.schemas.common import PaginatedResponse

router = APIRouter()


@router.post("/", response_model=TraceResponse, status_code=201)
async def log_trace(
    data: TraceCreate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Log an LLM trace. Called by the MANGOS SDK after each LLM call.
    Lightweight — must return fast so it doesn't slow down the application.
    """
    trace = LLMTrace(
        organization_id=current_user.organization_id,
        **data.model_dump(),
    )
    repo = TraceRepository(session)
    trace = await repo.create(trace)
    await session.commit()
    await session.refresh(trace)
    return trace


@router.get("/", response_model=PaginatedResponse[TraceResponse])
async def list_traces(
    run_id: Optional[uuid.UUID] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = TraceRepository(session)
    offset = (page - 1) * page_size
    if run_id:
        items, total = await repo.get_by_run(run_id, current_user.organization_id, offset, page_size)
    else:
        items, total = await repo.list(current_user.organization_id, offset, page_size)

    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@router.get("/{trace_id}", response_model=TraceResponse)
async def get_trace(
    trace_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = TraceRepository(session)
    trace = await repo.get_by_id(trace_id, current_user.organization_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@router.post("/{trace_id}/feedback", response_model=TraceResponse)
async def submit_feedback(
    trace_id: uuid.UUID,
    data: FeedbackCreate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Submit end-user feedback (thumbs up/down) on a trace.
    Used for RLHF data collection and monitoring user satisfaction trends.
    """
    repo = TraceRepository(session)
    trace = await repo.get_by_id(trace_id, current_user.organization_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    trace.user_feedback_score = data.score
    if not trace.metadata:
        trace.metadata = {}
    trace.metadata["feedback_label"] = data.label
    trace.metadata["feedback_comment"] = data.comment

    await session.commit()
    await session.refresh(trace)
    return trace


# Make Optional available
from typing import Optional
