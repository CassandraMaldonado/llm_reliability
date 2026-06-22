# Shared FastAPI dependencies for authentication and services.

# How it works:
# Routes handlers get their dependencies.
# Each request gets its own database session.
# The current user is loaded once per request and reused when needed.

import uuid
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import User
from app.repositories import UserRepository
from app.services.auth_service import AuthService, decode_token

bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    session: AsyncSession = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header),
) -> User:
    """
    Resolve the authenticated user from either:
    1. Bearer JWT token (browser/interactive use)
    2. X-API-Key header (SDK/programmatic use)
    
    Returns 401 if neither is valid.
    """
    if credentials:
        try:
            payload = decode_token(credentials.credentials)
            if payload.get("type") != "access":
                raise HTTPException(status_code=401, detail="Invalid token type")
            user_repo = UserRepository(session)
            user = await user_repo.get_by_id_any_org(uuid.UUID(payload["sub"]))
            if not user or not user.is_active:
                raise HTTPException(status_code=401, detail="User not found or inactive")
            return user
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    if api_key:
        auth_service = AuthService(session)
        user = await auth_service.authenticate_api_key(api_key)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return user

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Use Bearer token or X-API-Key header.",
    )


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role not in ("admin", "owner"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def require_org(org_id: uuid.UUID, current_user: User = Depends(get_current_user)) -> uuid.UUID:
    if current_user.organization_id != org_id:
        raise HTTPException(status_code=403, detail="Access denied to this organization")
    return org_id
