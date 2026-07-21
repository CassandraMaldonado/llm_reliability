# Shared Pydantic primitives used across all schemas.

# pagination strategy is cursor based.

import uuid
from datetime import datetime
from typing import Generic, List, Optional, TypeVar, Annotated, Any, Dict

from pydantic import BaseModel, ConfigDict, Field

# type alias for UUID fields that serialize as strings in JSON.
UUIDStr = uuid.UUID


T = TypeVar("T")


# standard paginated list response, all list endpoints return this shape.
    
    #Example response:
    #{
        #"items": [...],
        #"total": 1523,
        #"page": 1,
        #"page_size": 20,
        #"has_next": true,
        #"next_cursor": "eyJpZCI6IjEyMyJ9"
    #}

class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int = 1
    page_size: int = 20
    has_next: bool = False
    next_cursor: Optional[str] = None

# cursor pagination input parameters.
class CursorPage(BaseModel):
    cursor: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)

# RFC 7807 Problem Details error format.
class ErrorResponse(BaseModel):
    error: str                          
    message: str                        
    details: Optional[Dict[str, Any]] = None   # optional structured details.
    request_id: Optional[str] = None   

# success acknowledgment for operations that don't return data.
class SuccessResponse(BaseModel):
    success: bool = True
    message: str = "Operation completed successfully"

# common time range filtering for metrics endpoints.
class TimeRangeFilter(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    window_hours: Optional[int] = Field(default=24, ge=1, le=720)  # max 30 days
