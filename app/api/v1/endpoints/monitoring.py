"""app/api/v1/endpoints/monitoring.py"""
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import User
from app.repositories import MonitoringRepository
from app.schemas.monitoring import MetricResponse, MetricAggregateResponse, TimeSeriesPoint

router = APIRouter()

AVAILABLE_METRICS = [
    "latency_ms", "cost_usd", "hallucination_score",
    "answer_relevance", "faithfulness", "failure_rate", "feedback_score",
]


@router.get("/metrics", response_model=List[MetricAggregateResponse])
async def get_metrics_overview(
    hours: int = Query(default=24, ge=1, le=720),
    model_name: Optional[str] = Query(default=None),
    metrics: Optional[str] = Query(default=None, description="comma-separated metric names"),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fetch aggregated metrics for the dashboard overview.
    Returns current value, trend, and time-series for each metric.
    """
    repo = MonitoringRepository(session)
    requested = metrics.split(",") if metrics else AVAILABLE_METRICS

    results = []
    for metric_name in requested:
        time_series_records = await repo.get_time_series(
            current_user.organization_id, metric_name, hours, model_name
        )
        if not time_series_records:
            continue

        values = [r.metric_value for r in time_series_records]
        current_value = values[-1] if values else 0
        previous_value = values[0] if len(values) > 1 else None
        delta_pct = None
        if previous_value and previous_value != 0:
            delta_pct = round(((current_value - previous_value) / abs(previous_value)) * 100, 2)

        trend = "stable"
        if delta_pct is not None:
            if delta_pct > 5:
                trend = "up"
            elif delta_pct < -5:
                trend = "down"

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        results.append(MetricAggregateResponse(
            metric_name=metric_name,
            current_value=current_value,
            previous_value=previous_value,
            delta_pct=delta_pct,
            trend=trend,
            time_series=[
                TimeSeriesPoint(timestamp=r.window_start, value=r.metric_value)
                for r in time_series_records
            ],
            p50=sorted_vals[int(n * 0.5)] if n > 0 else None,
            p95=sorted_vals[int(n * 0.95)] if n > 1 else None,
            p99=sorted_vals[int(n * 0.99)] if n > 2 else None,
        ))

    return results
