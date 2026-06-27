"""app/api/v1/endpoints/evaluations.py"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import User
from app.repositories import EvaluationRepository
from app.schemas.evaluations import EvaluationResultResponse
from app.schemas.common import PaginatedResponse

router = APIRouter()


@router.get("/", response_model=PaginatedResponse[EvaluationResultResponse])
async def list_evaluations(
    run_id: uuid.UUID = Query(...),
    metric: str = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List individual evaluation results for a run, optionally filtered by metric."""
    repo = EvaluationRepository(session)
    filters = {"experiment_run_id": run_id}
    if metric:
        filters["metric_name"] = metric
    offset = (page - 1) * page_size
    items, total = await repo.list(current_user.organization_id, offset, page_size, **filters)
    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        has_next=(offset + page_size) < total,
    )
