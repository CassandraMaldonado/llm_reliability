# Shared Pydantic primitives used across all schemas.

Pagination Strategy: Cursor-based (not offset-based)
- Why cursor-based? Offset pagination breaks on inserts: if you're on page 3
  and someone inserts a row, your page 4 duplicates a row.
- Cursor = base64(last_seen_id + last_seen_created_at) for stable ordering.
- Enterprise standard: Stripe, GitHub, Slack all use cursor pagination.

Error Format: RFC 7807 Problem Details (industry standard)
- type: URI identifying the problem type
- title: human-readable summary
- detail: specific explanation for this occurrence
"""
import uuid
from datetime import datetime
from typing import Generic, List, Optional, TypeVar, Annotated, Any, Dict

from pydantic import BaseModel, ConfigDict, Field

# Type alias for UUID fields that serialize as strings in JSON
UUIDStr = uuid.UUID


T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Standard paginated list response.
    All list endpoints return this shape.
    
    Example response:
    {
        "items": [...],
        "total": 1523,
        "page": 1,
        "page_size": 20,
        "has_next": true,
        "next_cursor": "eyJpZCI6IjEyMyJ9"
    }
    """
    items: List[T]
    total: int
    page: int = 1
    page_size: int = 20
    has_next: bool = False
    next_cursor: Optional[str] = None


class CursorPage(BaseModel):
    """Cursor pagination input parameters."""
    cursor: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


class ErrorResponse(BaseModel):
    """
    RFC 7807 Problem Details error format.
    Consistent error shape across all endpoints.
    """
    error: str                          # machine-readable error code e.g. "NOT_FOUND"
    message: str                        # human-readable message
    details: Optional[Dict[str, Any]] = None   # optional structured details
    request_id: Optional[str] = None   # for correlation with logs


class SuccessResponse(BaseModel):
    """Simple success acknowledgment for operations that don't return data."""
    success: bool = True
    message: str = "Operation completed successfully"


class TimeRangeFilter(BaseModel):
    """Common time range filter for monitoring/metrics endpoints."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    window_hours: Optional[int] = Field(default=24, ge=1, le=720)  # max 30 days
