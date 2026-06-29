"""app/schemas/auth.py — Authentication and user schemas."""
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=1, max_length=255)
    organization_name: Optional[str] = None  # If provided, creates new org


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str
    role: str
    organization_id: uuid.UUID
    is_active: bool
    created_at: datetime


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshRequest(BaseModel):
    refresh_token: str


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = None
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)


class ApiKeyResponse(BaseModel):
    """
    NOTE: key_prefix + raw_key are only returned once at creation.
    After that, only key_prefix is accessible (full key is hashed in DB).
    Enterprise pattern: same as AWS IAM secret access keys.
    """
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    key_prefix: str
    raw_key: Optional[str] = None   # Only populated on creation
    description: Optional[str]
    is_active: bool
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    created_at: datetime
