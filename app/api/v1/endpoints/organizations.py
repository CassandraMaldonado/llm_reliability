"""app/api/v1/endpoints/organizations.py"""
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_current_admin, get_db
from app.models import User
from app.repositories import OrganizationRepository
from app.schemas.organizations import OrganizationResponse, OrganizationUpdate

router = APIRouter()


@router.get("/me", response_model=OrganizationResponse)
async def get_my_org(
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the current user's organization."""
    repo = OrganizationRepository(session)
    org = await repo.get_by_id(current_user.organization_id, current_user.organization_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org


@router.patch("/me", response_model=OrganizationResponse)
async def update_my_org(
    data: OrganizationUpdate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin),
):
    """Update organization settings. Admin only."""
    repo = OrganizationRepository(session)
    org = await repo.get_by_id(current_user.organization_id, current_user.organization_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    await repo.update(org, **data.model_dump(exclude_none=True))
    await session.commit()
    await session.refresh(org)
    return org
