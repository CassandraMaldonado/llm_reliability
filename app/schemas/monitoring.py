
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    value: float


class MetricResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organization_id: uuid.UUID
    metric_name: str
    metric_value: float
    model_name: Optional[str]
    window_start: datetime
    window_end: datetime
    sample_count: int
    metadata: Dict[str, Any]
    created_at: datetime


class MetricAggregateResponse(BaseModel):
    """Aggregated metric stats for dashboard KPI cards."""
    metric_name: str
    current_value: float
    previous_value: Optional[float]     # previous window for delta calculation
    delta_pct: Optional[float]
    trend: str  # up, down, stable
    time_series: List[TimeSeriesPoint]
    p50: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
