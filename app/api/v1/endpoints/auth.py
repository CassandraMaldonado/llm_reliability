# Authentice the endpoints.
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import User
from app.schemas.auth import (
    LoginRequest, TokenResponse, RefreshRequest,
    UserCreate, UserResponse,
    ApiKeyCreate, ApiKeyResponse,
)
from app.services.auth_service import AuthService

router = APIRouter()


# Register a new user. 
@router.post("/register", response_model=UserResponse, status_code=201)
async def register(data: UserCreate, session: AsyncSession = Depends(get_db)):
    service = AuthService(session)
    user, _ = await service.register(data)
    await session.commit()
    return user

# Authenticate returns JWT access and refresh tokens.
@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest, session: AsyncSession = Depends(get_db)):
    service = AuthService(session)
    result = await service.login(data)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    _, tokens = result
    return tokens


@router.post("/refresh", response_model=TokenResponse)
async def refresh(data: RefreshRequest, session: AsyncSession = Depends(get_db)):
    """Exchange a refresh token for a new access token."""
    service = AuthService(session)
    tokens = await service.refresh(data.refresh_token)
    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    return tokens


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user

# Create a new API key.
@router.post("/api-keys", response_model=ApiKeyResponse, status_code=201)
async def create_api_key(
    data: ApiKeyCreate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AuthService(session)
    api_key, raw_key = await service.create_api_key(
        org_id=current_user.organization_id,
        user_id=current_user.id,
        data=data,
    )
    await session.commit()

    response = ApiKeyResponse.model_validate(api_key)
    response.raw_key = raw_key  # One-time reveal
    return response
