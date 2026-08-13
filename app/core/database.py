# Async SQLAlchemy 2.0 database session management.

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.core.config import settings

# SQLAlchemy declarative base.
class Base(DeclarativeBase):
    pass


# creates AsyncEngine.
def create_engine(for_testing: bool = False) -> AsyncEngine:
    pool_kwargs = (
        {"poolclass": NullPool}
        if for_testing
        else {
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
            "pool_pre_ping": True,   # validates the connection before using it.
            "pool_recycle": 3600,
        }
    )
    return create_async_engine(
        str(settings.DATABASE_URL),
        echo=settings.DATABASE_ECHO,
        **pool_kwargs,
    )


engine = create_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # keeps objects accessible after commit.
    autocommit=False,
    autoflush=False,
)

# FastAPI dependency that yields a database session.
    # Session lifecycle:
    # 1. Open session at request start and yield to route handler.
    # 2. Commit on success or rollback on exception.
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# context version for use outside of FastAPI.
@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
