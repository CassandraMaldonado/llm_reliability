"""app/schemas/traces.py"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TraceCreate(BaseModel):
    """
    SDK-facing schema: what the mangos.trace() context manager POSTs.
    Designed to be minimal — just what the SDK can capture automatically.
    """
    experiment_run_id: Optional[uuid.UUID] = None
    model_name: str
    provider: str
    input_text: str
    output_text: str
    system_prompt: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None    # group traces from one user session
    user_id: Optional[str] = None       # end-user identifier (hashed for privacy)


class TraceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organization_id: uuid.UUID
    experiment_run_id: Optional[uuid.UUID]
    model_name: str
    provider: str
    input_text: str
    output_text: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_ms: Optional[float]
    cost_usd: Optional[float]
    user_feedback_score: Optional[float]
    metadata: Dict[str, Any]
    tags: List[str]
    created_at: datetime


class FeedbackCreate(BaseModel):
    """End-user thumbs up/down or star rating feedback on a trace."""
    score: float = Field(ge=0.0, le=1.0, description="0=negative, 1=positive, 0.5=neutral")
    label: Optional[str] = Field(default=None, description="thumbs_up, thumbs_down, star_1..5")
    comment: Optional[str] = None
