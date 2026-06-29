"""app/schemas/evaluations.py"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class EvaluationResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    experiment_run_id: uuid.UUID
    trace_id: Optional[uuid.UUID]
    dataset_row_id: Optional[uuid.UUID]
    metric_name: str
    score: Optional[float]
    passed: Optional[bool]
    threshold: Optional[float]
    reasoning: Optional[str]   # LLM judge explanation
    raw_output: Dict[str, Any]
    latency_ms: Optional[float]
    created_at: datetime


class EvaluationSummary(BaseModel):
    """Aggregated evaluation metrics for a run."""
    run_id: str
    total_evaluated: int
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {mean, p50, p95, pass_rate}
    overall_pass_rate: float
    cost_usd: float
