"""app/api/v1/endpoints/alerts.py"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import User, AlertRule, Alert
from app.repositories import AlertRuleRepository, AlertRepository
from app.schemas.alerts import (
    AlertRuleCreate, AlertRuleUpdate, AlertRuleResponse, AlertResponse
)
from app.schemas.common import PaginatedResponse

router = APIRouter()


# ── Alert Rules ──────────────────────────────────────────────────────────────

@router.post("/rules", response_model=AlertRuleResponse, status_code=201)
async def create_alert_rule(
    data: AlertRuleCreate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create an alert rule.
    The Celery Beat monitoring task evaluates all active rules every 5 minutes.
    """
    rule = AlertRule(
        organization_id=current_user.organization_id,
        **data.model_dump(),
    )
    repo = AlertRuleRepository(session)
    rule = await repo.create(rule)
    await session.commit()
    await session.refresh(rule)
    return rule


@router.get("/rules", response_model=PaginatedResponse[AlertRuleResponse])
async def list_alert_rules(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = AlertRuleRepository(session)
    offset = (page - 1) * page_size
    items, total = await repo.list(current_user.organization_id, offset, page_size)
    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@router.patch("/rules/{rule_id}", response_model=AlertRuleResponse)
async def update_alert_rule(
    rule_id: uuid.UUID,
    data: AlertRuleUpdate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = AlertRuleRepository(session)
    rule = await repo.get_by_id(rule_id, current_user.organization_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    await repo.update(rule, **data.model_dump(exclude_none=True))
    await session.commit()
    await session.refresh(rule)
    return rule


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_alert_rule(
    rule_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = AlertRuleRepository(session)
    rule = await repo.get_by_id(rule_id, current_user.organization_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    await repo.soft_delete(rule)
    await session.commit()


# ── Active Alerts ─────────────────────────────────────────────────────────────

@router.get("/", response_model=PaginatedResponse[AlertResponse])
async def list_alerts(
    resolved: bool = Query(default=False),
    severity: str = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = AlertRepository(session)
    if not resolved:
        items = await repo.get_unresolved(current_user.organization_id, limit=page_size)
        total = len(items)
    else:
        offset = (page - 1) * page_size
        items, total = await repo.list(current_user.organization_id, offset, page_size)

    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        has_next=False,
    )


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = AlertRepository(session)
    alert = await repo.get_by_id(alert_id, current_user.organization_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert = await repo.acknowledge(alert)
    await session.commit()
    return alert


@router.post("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert(
    alert_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = AlertRepository(session)
    alert = await repo.get_by_id(alert_id, current_user.organization_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert = await repo.resolve(alert)
    await session.commit()
    return alert
