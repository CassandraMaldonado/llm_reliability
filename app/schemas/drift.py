"""app/schemas/drift.py"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class DriftMetricDetail(BaseModel):
    metric_name: str
    drift_detected: bool
    drift_type: Optional[str]    # distribution, anomaly, threshold
    severity: Optional[str]      # critical, warning
    baseline_mean: Optional[float]
    current_mean: Optional[float]
    delta_pct: Optional[float]
    p_value: Optional[float]     # KS test p-value
    z_score: Optional[float]


class DriftReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organization_id: uuid.UUID
    model_name: Optional[str]
    baseline_window_hours: int
    current_window_hours: int
    drift_detected: bool
    drift_score: float           # 0-1 composite drift score
    metrics_analyzed: int
    metrics_drifted: int
    metric_details: List[DriftMetricDetail]
    alert_generated: bool
    created_at: datetime


class DriftSummary(BaseModel):
    """Dashboard-friendly drift overview."""
    overall_drift_score: float
    drifting_metrics: List[str]
    latest_report_id: Optional[str]
    last_checked_at: Optional[datetime]
    status: str  # healthy, warning, critical
