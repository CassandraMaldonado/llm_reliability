import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DatasetRowCreate(BaseModel):
    question: str
    expected_answer: Optional[str] = None
    context: Optional[str] = None           # for RAG.
    retrieved_contexts: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetRowResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_id: uuid.UUID
    question: str
    expected_answer: Optional[str]
    context: Optional[str]
    retrieved_contexts: Optional[List[str]]
    metadata: Dict[str, Any]
    created_at: datetime


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    rows: List[DatasetRowCreate] = Field(default_factory=list)


class DatasetUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    description: Optional[str]
    row_count: int
    tags: List[str]
    created_at: datetime
    updated_at: Optional[datetime]
