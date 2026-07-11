from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import User
from app.repositories import DriftRepository, TraceRepository
from app.schemas.drift import DriftReportResponse, DriftSummary
from app.schemas.common import PaginatedResponse

router = APIRouter()


@router.get("/summary", response_model=DriftSummary)
async def get_drift_summary(
    model_name: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # dashboard drift summary with the overall health status and which metrics are drifting.
    repo = DriftRepository(session)
    latest = await repo.get_latest(current_user.organization_id, model_name)

    if not latest:
        return DriftSummary(
            overall_drift_score=0.0,
            drifting_metrics=[],
            latest_report_id=None,
            last_checked_at=None,
            status="unknown",
        )

    drifting = [
        d["metric_name"]
        for d in (latest.metric_details or [])
        if d.get("drift_detected")
    ]

    if latest.drift_score >= 0.7:
        status = "critical"
    elif latest.drift_score >= 0.3:
        status = "warning"
    else:
        status = "healthy"

    return DriftSummary(
        overall_drift_score=latest.drift_score,
        drifting_metrics=drifting,
        latest_report_id=str(latest.id),
        last_checked_at=latest.created_at,
        status=status,
    )


@router.get("/reports", response_model=PaginatedResponse[DriftReportResponse])
async def list_drift_reports(
    model_name: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = DriftRepository(session)
    filters = {}
    if model_name:
        filters["model_name"] = model_name
    offset = (page - 1) * page_size
    items, total = await repo.list(current_user.organization_id, offset, page_size, **filters)
    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@router.post("/run", status_code=202)
async def trigger_drift_check(
    background_tasks: BackgroundTasks,
    model_name: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Manually trigger a drift detection run (outside of the 5-min Celery schedule).
    Useful for on-demand checks after a deployment.
    """
    async def run_check():
        from app.monitoring.drift_detection import DriftDetector
        trace_repo = TraceRepository(session)
        traces = await trace_repo.get_recent_for_org(
            current_user.organization_id, hours=25, model_name=model_name
        )
        detector = DriftDetector()
        await detector.detect_all(
            org_id=current_user.organization_id,
            traces=traces,
            session=session,
        )

    background_tasks.add_task(run_check)
    return {"message": "Drift check queued", "status": "accepted"}
