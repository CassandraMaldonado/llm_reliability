"""
app/schemas/

Pydantic v2 schemas for API request/response serialization.

Design Principles:
- Schemas are SEPARATE from ORM models (never return raw SQLAlchemy objects)
- Three schema variants per resource: Create, Update, Response
- from_attributes=True on all Response schemas (ORM → Pydantic conversion)
- Strict types: no silent coercion of str → int etc.
- All UUIDs as uuid.UUID (not str), serialized as strings in JSON

Why separate from models?
- ORM models have DB-specific types (JSONB, UUID dialect)
- API contracts should be stable even when DB schema evolves
- Enables field-level permission control (hide internal fields from API)
"""
from app.schemas.common import (
    PaginatedResponse,
    CursorPage,
    ErrorResponse,
    SuccessResponse,
    UUIDStr,
)
from app.schemas.organizations import (
    OrganizationCreate,
    OrganizationUpdate,
    OrganizationResponse,
)
from app.schemas.auth import (
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    ApiKeyCreate,
    ApiKeyResponse,
    UserCreate,
    UserResponse,
)
from app.schemas.experiments import (
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse,
    ExperimentRunCreate,
    ExperimentRunResponse,
    RunCompareRequest,
    RunCompareResponse,
)
from app.schemas.traces import (
    TraceCreate,
    TraceResponse,
    FeedbackCreate,
)
from app.schemas.evaluations import (
    EvaluationResultResponse,
    EvaluationSummary,
)
from app.schemas.datasets import (
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetRowCreate,
    DatasetRowResponse,
)
from app.schemas.rag import (
    RAGEvaluationCreate,
    RAGEvaluationResponse,
    RAGCompareRequest,
    RAGCompareResponse,
)
from app.schemas.monitoring import (
    MetricResponse,
    MetricAggregateResponse,
    TimeSeriesPoint,
)
from app.schemas.alerts import (
    AlertRuleCreate,
    AlertRuleUpdate,
    AlertRuleResponse,
    AlertResponse,
)
from app.schemas.drift import (
    DriftReportResponse,
    DriftSummary,
)

__all__ = [
    "PaginatedResponse", "CursorPage", "ErrorResponse", "SuccessResponse", "UUIDStr",
    "OrganizationCreate", "OrganizationUpdate", "OrganizationResponse",
    "LoginRequest", "TokenResponse", "RefreshRequest",
    "ApiKeyCreate", "ApiKeyResponse", "UserCreate", "UserResponse",
    "ExperimentCreate", "ExperimentUpdate", "ExperimentResponse",
    "ExperimentRunCreate", "ExperimentRunResponse",
    "RunCompareRequest", "RunCompareResponse",
    "TraceCreate", "TraceResponse", "FeedbackCreate",
    "EvaluationResultResponse", "EvaluationSummary",
    "DatasetCreate", "DatasetUpdate", "DatasetResponse",
    "DatasetRowCreate", "DatasetRowResponse",
    "RAGEvaluationCreate", "RAGEvaluationResponse",
    "RAGCompareRequest", "RAGCompareResponse",
    "MetricResponse", "MetricAggregateResponse", "TimeSeriesPoint",
    "AlertRuleCreate", "AlertRuleUpdate", "AlertRuleResponse", "AlertResponse",
    "DriftReportResponse", "DriftSummary",
]
