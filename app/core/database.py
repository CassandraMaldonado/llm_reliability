"""
app/core/database.py

Async SQLAlchemy 2.0 database session management.

Enterprise pattern:
- AsyncEngine + AsyncSession for non-blocking I/O (critical for FastAPI)
- Session-per-request via dependency injection
- Never share sessions across requests (not thread/task safe)
- Connection pool tuned for production load
"""
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


class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base.
    All ORM models inherit from this.
    Putting it here (not in models/) avoids circular imports.
    """
    pass


def create_engine(for_testing: bool = False) -> AsyncEngine:
    """
    Create AsyncEngine with production-tuned pool settings.

    for_testing=True uses NullPool (no connection reuse) which is required
    when using pytest-asyncio with transaction rollback fixtures.
    """
    pool_kwargs = (
        {"poolclass": NullPool}
        if for_testing
        else {
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
            "pool_pre_ping": True,   # validates connection before use (handles DB restarts)
            "pool_recycle": 3600,    # recycle connections after 1 hour (prevent stale TCP)
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
    expire_on_commit=False,  # Keep objects accessible after commit (FastAPI response serialization)
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session.

    Usage in route:
        @router.get("/things")
        async def list_things(db: AsyncSession = Depends(get_db)):
            ...

    Session lifecycle:
    1. Open session at request start
    2. Yield to route handler
    3. Commit on success OR rollback on exception
    4. Always close session (returns connection to pool)

    Enterprise note: always rollback explicitly in except — don't rely on
    session closing to handle rollback. Explicit is safer.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager version for use outside of FastAPI (e.g., Celery tasks).

    Usage:
        async with get_db_context() as db:
            result = await some_repo.find(db, id=task_id)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
