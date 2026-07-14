# Authentication.

import hashlib
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import User, ApiKey, Organization
from app.repositories import UserRepository, ApiKeyRepository, OrganizationRepository
from app.schemas.auth import (
    LoginRequest, TokenResponse, UserCreate, ApiKeyCreate, ApiKeyResponse
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

KEY_PREFIX = "mg_"


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# SHA-256 hash of raw key.
def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()

# returns (raw_key, key_hash).
def generate_api_key() -> Tuple[str, str]:
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, hash_api_key(raw)


def create_access_token(user_id: uuid.UUID, org_id: uuid.UUID) -> str:
    payload = {
        "sub": str(user_id),
        "org": str(org_id),
        "type": "access",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        ),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


def create_refresh_token(user_id: uuid.UUID) -> str:
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "jti": secrets.token_urlsafe(16),  # unique token ID for revocation
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        ),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])


class AuthService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.users = UserRepository(session)
        self.api_keys = ApiKeyRepository(session)
        self.orgs = OrganizationRepository(session)

    async def register(self, data: UserCreate) -> Tuple[User, TokenResponse]:
        """Register user, optionally create org."""
        # Create org if name provided
        if data.organization_name:
            slug = data.organization_name.lower().replace(" ", "-")
            org = Organization(
                name=data.organization_name,
                slug=slug,
            )
            self.session.add(org)
            await self.session.flush()
        else:
            # Default org (for single-tenant or dev mode)
            result = await self.orgs.get_by_slug("default")
            if not result:
                org = Organization(name="Default", slug="default")
                self.session.add(org)
                await self.session.flush()
            else:
                org = result

        user = User(
            organization_id=org.id,
            email=data.email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
            role="admin" if data.organization_name else "member",
        )
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)

        tokens = self._issue_tokens(user)
        return user, tokens

    async def login(self, data: LoginRequest) -> Optional[Tuple[User, TokenResponse]]:
        user = await self.users.get_by_email(data.email)
        if not user or not verify_password(data.password, user.hashed_password):
            return None
        if not user.is_active:
            return None
        return user, self._issue_tokens(user)

    async def refresh(self, refresh_token: str) -> Optional[TokenResponse]:
        try:
            payload = decode_token(refresh_token)
            if payload.get("type") != "refresh":
                return None
            user = await self.users.get_by_id_any_org(uuid.UUID(payload["sub"]))
            if not user or not user.is_active:
                return None
            return self._issue_tokens(user)
        except jwt.PyJWTError:
            return None

    def _issue_tokens(self, user: User) -> TokenResponse:
        access = create_access_token(user.id, user.organization_id)
        refresh = create_refresh_token(user.id)
        return TokenResponse(
            access_token=access,
            refresh_token=refresh,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def create_api_key(
        self, org_id: uuid.UUID, user_id: uuid.UUID, data: ApiKeyCreate
    ) -> Tuple[ApiKey, str]:
        raw_key, key_hash = generate_api_key()
        key_prefix = raw_key[:12]  # "mg_" + 9 chars for display

        expires_at = None
        if data.expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=data.expires_in_days)

        api_key = ApiKey(
            organization_id=org_id,
            user_id=user_id,
            name=data.name,
            description=data.description,
            key_hash=key_hash,
            key_prefix=key_prefix,
            expires_at=expires_at,
        )
        self.session.add(api_key)
        await self.session.flush()
        await self.session.refresh(api_key)
        return api_key, raw_key

    # Validate API key, return associated user.
    async def authenticate_api_key(self, raw_key: str) -> Optional[User]:
        key_hash = hash_api_key(raw_key)
        api_key = await self.api_keys.get_by_key_hash(key_hash)
        if not api_key:
            return None
        if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
            return None
        # Update last used.
        await self.api_keys.update_last_used(api_key)
        return await self.users.get_by_id_any_org(api_key.user_id)
