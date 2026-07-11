import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import User, Dataset, DatasetRow
from app.repositories import DatasetRepository
from app.schemas.datasets import (
    DatasetCreate, DatasetUpdate, DatasetResponse,
    DatasetRowCreate, DatasetRowResponse,
)
from app.schemas.common import PaginatedResponse

router = APIRouter()


@router.post("/", response_model=DatasetResponse, status_code=201)
async def create_dataset(
    data: DatasetCreate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = DatasetRepository(session)

    dataset = Dataset(
        organization_id=current_user.organization_id,
        name=data.name,
        description=data.description,
        tags=data.tags,
        row_count=len(data.rows),
    )
    dataset = await repo.create(dataset)

    for row_data in data.rows:
        row = DatasetRow(dataset_id=dataset.id, **row_data.model_dump())
        session.add(row)

    await session.commit()
    await session.refresh(dataset)
    return dataset


@router.get("/", response_model=PaginatedResponse[DatasetResponse])
async def list_datasets(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = DatasetRepository(session)
    offset = (page - 1) * page_size
    items, total = await repo.list(current_user.organization_id, offset, page_size)
    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = DatasetRepository(session)
    dataset = await repo.get_by_id(dataset_id, current_user.organization_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/{dataset_id}/rows", response_model=PaginatedResponse[DatasetRowResponse])
async def get_dataset_rows(
    dataset_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = DatasetRepository(session)
    dataset = await repo.get_by_id(dataset_id, current_user.organization_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    offset = (page - 1) * page_size
    items, total = await repo.get_rows(dataset_id, offset, page_size)
    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@router.post("/{dataset_id}/rows", response_model=List[DatasetRowResponse], status_code=201)
async def add_rows(
    dataset_id: uuid.UUID,
    rows: List[DatasetRowCreate],
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    #appends rows to an existing dataset.
    repo = DatasetRepository(session)
    dataset = await repo.get_by_id(dataset_id, current_user.organization_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    new_rows = []
    for row_data in rows:
        row = DatasetRow(dataset_id=dataset_id, **row_data.model_dump())
        session.add(row)
        new_rows.append(row)

    dataset.row_count += len(rows)
    await session.commit()
    for row in new_rows:
        await session.refresh(row)
    return new_rows


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = DatasetRepository(session)
    dataset = await repo.get_by_id(dataset_id, current_user.organization_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await repo.soft_delete(dataset)
    await session.commit()
