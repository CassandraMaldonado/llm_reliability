"""
app/models/

SQLAlchemy 2.0 ORM models.
These define the database tables in Python.
Kept separate from Pydantic schemas (schemas/ is for API serialization).

Design notes:
- All UUIDs as primary keys
- organization_id on every model (multi-tenancy)
- Timestamps on everything
- JSONB for flexible storage (hyperparameters, metadata, config)
"""
import uuid
from datetime import datetime
from typing import Optional, List, Any, Dict

from sqlalchemy import (
    String, Text, Boolean, Float, Integer, DateTime,
    ForeignKey, func, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class TimestampMixin:
    """Mixin that adds created_at and updated_at to any model."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )


class SoftDeleteMixin:
    """Mixin that adds soft-delete support via deleted_at column."""
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


# ─────────────────────────────────────────────────────────────────────────────
# ORGANIZATION
# ─────────────────────────────────────────────────────────────────────────────

class Organization(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    plan: Mapped[str] = mapped_column(String(50), nullable=False, default="free")
    settings: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="organization")
    projects: Mapped[List["Project"]] = relationship("Project", back_populates="organization")


# ─────────────────────────────────────────────────────────────────────────────
# USER
# ─────────────────────────────────────────────────────────────────────────────

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="member")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")
    api_keys: Mapped[List["APIKey"]] = relationship("APIKey", back_populates="created_by_user")


# ─────────────────────────────────────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────────────────────────────────────

class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(12), nullable=False)
    hashed_key: Mapped[str] = mapped_column(String(255), nullable=False)
    scopes: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    created_by_user: Mapped["User"] = relationship("User", back_populates="api_keys")

    @property
    def is_valid(self) -> bool:
        from datetime import timezone
        now = datetime.now(timezone.utc)
        if self.revoked_at:
            return False
        if self.expires_at and self.expires_at < now:
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT
# ─────────────────────────────────────────────────────────────────────────────

class Project(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    metadata_: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))

    organization: Mapped["Organization"] = relationship("Organization", back_populates="projects")
    experiments: Mapped[List["Experiment"]] = relationship("Experiment", back_populates="project")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

class Experiment(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id"))
    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")
    baseline_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    metadata_: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))

    project: Mapped["Project"] = relationship("Project", back_populates="experiments")
    runs: Mapped[List["ExperimentRun"]] = relationship("ExperimentRun", back_populates="experiment")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT RUN
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRun(Base, TimestampMixin):
    __tablename__ = "experiment_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("experiments.id"))
    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))

    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_template: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    hyperparameters: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Aggregated metrics (denormalized for fast dashboard queries)
    total_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completed_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    failed_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    avg_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p50_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p95_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p99_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    avg_answer_relevance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_faithfulness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_hallucination_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_semantic_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_toxicity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    tags: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    metadata_: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="runs")
    traces: Mapped[List["LLMTrace"]] = relationship("LLMTrace", back_populates="run")


# ─────────────────────────────────────────────────────────────────────────────
# LLM TRACE
# ─────────────────────────────────────────────────────────────────────────────

class LLMTrace(Base):
    __tablename__ = "llm_traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("experiment_runs.id"), nullable=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="experiment")
    trace_group_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    span_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    messages: Mapped[Optional[List[Dict]]] = mapped_column(JSONB, nullable=True)
    completion: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    finish_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    time_to_first_token_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    hyperparameters: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    expected_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context_documents: Mapped[Optional[List[Dict]]] = mapped_column(JSONB, nullable=True)

    status: Mapped[str] = mapped_column(String(50), nullable=False, default="success")
    error_code: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    user_feedback_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    user_feedback_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[Optional["ExperimentRun"]] = relationship("ExperimentRun", back_populates="traces")
    evaluation_results: Mapped[List["EvaluationResult"]] = relationship("EvaluationResult", back_populates="trace")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION RESULT
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("llm_traces.id"))
    run_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("experiment_runs.id"), nullable=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    raw_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    passed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evaluator_model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    metadata_: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    trace: Mapped["LLMTrace"] = relationship("LLMTrace", back_populates="evaluation_results")


# ─────────────────────────────────────────────────────────────────────────────
# ALERT RULE & ALERT
# ─────────────────────────────────────────────────────────────────────────────

class AlertRule(Base, TimestampMixin):
    __tablename__ = "alert_rules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    condition: Mapped[str] = mapped_column(String(50), nullable=False)
    threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    severity: Mapped[str] = mapped_column(String(50), nullable=False, default="warning")
    model_filter: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    evaluation_window_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    cooldown_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    notification_channels: Mapped[List[Dict]] = mapped_column(JSONB, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))

    alerts: Mapped[List["Alert"]] = relationship("Alert", back_populates="rule")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("alert_rules.id"))
    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="open")
    severity: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    metric_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threshold_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context_: Mapped[Dict[str, Any]] = mapped_column("context", JSONB, nullable=False, default=dict)
    acknowledged_by_user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    rule: Mapped["AlertRule"] = relationship("AlertRule", back_populates="alerts")
