# Base repository that contains common CRUD operations used across the project.

# I decided to use the Repository Pattern to keep the database logic separate from the service layer. Instead of having SQLAlchemy queries spread throughout
the code, all database interactions live in one place.

import uuid
from typing import Generic, List, Optional, Tuple, Type, TypeVar

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import Base

ModelType = TypeVar("ModelType", bound=Base)

# base repository that includes reusable CRUD methods and pagination.
# other repositories extend this class by setting `model = YourModel`.
class BaseRepository(Generic[ModelType]):

    model: Type[ModelType]

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, id: uuid.UUID, org_id: uuid.UUID) -> Optional[ModelType]:
        """Get by primary key, scoped to org (row-level tenancy)."""
        result = await self.session.execute(
            select(self.model).where(
                and_(
                    self.model.id == id,
                    self.model.organization_id == org_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        org_id: uuid.UUID,
        offset: int = 0,
        limit: int = 20,
        **filters,
    ) -> Tuple[List[ModelType], int]:
        """Return (items, total_count) for pagination."""
        conditions = [self.model.organization_id == org_id]

        # Apply soft-delete filter if model supports it
        if hasattr(self.model, "deleted_at"):
            conditions.append(self.model.deleted_at.is_(None))

        # Apply additional filters
        for key, value in filters.items():
            if value is not None and hasattr(self.model, key):
                conditions.append(getattr(self.model, key) == value)

        where_clause = and_(*conditions)

        # Count query
        count_result = await self.session.execute(
            select(func.count()).select_from(self.model).where(where_clause)
        )
        total = count_result.scalar_one()

        # Data query
        result = await self.session.execute(
            select(self.model)
            .where(where_clause)
            .order_by(self.model.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        items = list(result.scalars().all())

        return items, total

    async def create(self, obj: ModelType) -> ModelType:
        self.session.add(obj)
        await self.session.flush()  # flush to get DB-generated values (id, created_at)
        await self.session.refresh(obj)
        return obj

    async def update(self, obj: ModelType, **updates) -> ModelType:
        for key, value in updates.items():
            if value is not None:
                setattr(obj, key, value)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj

    async def soft_delete(self, obj: ModelType) -> ModelType:
        """Mark as deleted without removing from DB — preserves audit trail."""
        from datetime import datetime, timezone
        obj.deleted_at = datetime.now(timezone.utc)
        await self.session.flush()
        return obj

    async def hard_delete(self, obj: ModelType) -> None:
        """Permanent delete. Only use for non-critical data (e.g. temp files)."""
        await self.session.delete(obj)
        await self.session.flush()
