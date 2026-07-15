# Alerts.
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

# condition format: <metric> <operator> <threshold>
# avg_hallucination_score > 0.6
# avg_latency_ms > 5000
# failure_rate > 0.05
    
class AlertRuleCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None
    metric: str  # hallucination_score, latency_ms, cost_usd, failure_rate, relevance_score
    operator: str  # gt, lt, gte, lte, eq
    threshold: float
    severity: str = Field(default="warning")  # info, warning, critical
    window_minutes: int = Field(default=60, ge=5, le=10080)  # 5min to 7 days
    notification_channels: List[str] = Field(
        default_factory=list,
        description="webhook, email, slack"
    )
    notification_config: Dict[str, Any] = Field(default_factory=dict)
    model_name: Optional[str] = None  # scope to specific model, or None for all
    is_active: bool = True


class AlertRuleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    threshold: Optional[float] = None
    severity: Optional[str] = None
    window_minutes: Optional[int] = None
    notification_channels: Optional[List[str]] = None
    notification_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class AlertRuleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    description: Optional[str]
    metric: str
    operator: str
    threshold: float
    severity: str
    window_minutes: int
    notification_channels: List[str]
    model_name: Optional[str]
    is_active: bool
    last_triggered_at: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organization_id: uuid.UUID
    alert_rule_id: uuid.UUID
    severity: str
    metric: str
    current_value: float
    threshold: float
    model_name: Optional[str]
    message: str
    acknowledged: bool
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    metadata: Dict[str, Any]
    created_at: datetime
