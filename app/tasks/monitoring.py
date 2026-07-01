# Metrics aggregation: roll up raw traces into MonitoringMetric records.
# Alert evaluation: check all active alert rules against current metrics.

# Alert architecture:
# - Alert rules are stored in DB.
# - Beat task gets all active rules and evaluates each against the latest metric value.
# - Notifications: webhook and email.


import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from celery import shared_task

from app.core.database import get_db_context
from app.models import MonitoringMetric, Alert

logger = logging.getLogger(__name__)

OPERATOR_MAP = {
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "eq": lambda a, b: abs(a - b) < 0.001,
}

METRIC_TO_TRACE_FIELD = {
    "latency_ms": "latency_ms",
    "cost_usd": "cost_usd",
    "hallucination_score": None,   
    "answer_relevance": None,      
    "faithfulness": None,        
    "failure_rate": None,          # error_count / total
    "feedback_score": "user_feedback_score",
}


@shared_task(
    name="tasks.aggregate_metrics",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def aggregate_metrics_task(self):
    """
    Roll up the last 5 minutes of LLMTrace records into MonitoringMetric rows.
    Runs every 5 minutes via Celery Beat.
    
    Why aggregate instead of query raw traces?
    - Traces can be millions of rows; metric aggregates are thousands
    - Dashboard queries hit pre-aggregated data: <10ms response time
    - Time-series visualizations need fixed-interval buckets
    - Pattern: same as how Datadog, Prometheus, and InfluxDB work
    """
    asyncio.run(_aggregate_metrics_async())


async def _aggregate_metrics_async():
    from sqlalchemy import select, func, and_
    from app.models import LLMTrace, EvaluationResult, Organization
    from app.repositories import MonitoringRepository

    async with get_db_context() as session:
        window_end = datetime.now(timezone.utc)
        window_start = window_end - timedelta(minutes=5)

        # Get all active orgs
        result = await session.execute(
            select(Organization.id).where(Organization.deleted_at.is_(None))
        )
        org_ids = [row[0] for row in result.all()]

        for org_id in org_ids:
            await _aggregate_for_org(session, org_id, window_start, window_end)

        await session.commit()
        logger.info(f"Metrics aggregated for {len(org_ids)} orgs")


async def _aggregate_for_org(session, org_id, window_start, window_end):
    from sqlalchemy import select, func, and_
    from app.models import LLMTrace

    # Get distinct model names in this window
    result = await session.execute(
        select(LLMTrace.model_name)
        .where(and_(
            LLMTrace.organization_id == org_id,
            LLMTrace.created_at >= window_start,
            LLMTrace.created_at < window_end,
        ))
        .distinct()
    )
    model_names = [row[0] for row in result.all()]
    model_names.append(None)  # Also aggregate across all models

    for model_name in model_names:
        conditions = [
            LLMTrace.organization_id == org_id,
            LLMTrace.created_at >= window_start,
            LLMTrace.created_at < window_end,
        ]
        if model_name:
            conditions.append(LLMTrace.model_name == model_name)

        result = await session.execute(
            select(
                func.avg(LLMTrace.latency_ms).label("avg_latency"),
                func.avg(LLMTrace.cost_usd).label("avg_cost"),
                func.avg(LLMTrace.user_feedback_score).label("avg_feedback"),
                func.count(LLMTrace.id).label("total"),
            ).where(and_(*conditions))
        )
        row = result.one()

        if not row.total:
            continue

        metrics_to_store = {
            "latency_ms": row.avg_latency,
            "cost_usd": row.avg_cost,
            "feedback_score": row.avg_feedback,
        }

        for metric_name, value in metrics_to_store.items():
            if value is None:
                continue
            metric = MonitoringMetric(
                organization_id=org_id,
                metric_name=metric_name,
                metric_value=float(value),
                model_name=model_name,
                window_start=window_start,
                window_end=window_end,
                sample_count=int(row.total),
                metadata={},
            )
            session.add(metric)


@shared_task(
    name="tasks.evaluate_alert_rules",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def evaluate_alert_rules_task(self):
    """
    Evaluate all active alert rules. Fires alerts when thresholds are breached.
    Runs every 5 minutes via Celery Beat.
    """
    asyncio.run(_evaluate_alerts_async())


async def _evaluate_alerts_async():
    from sqlalchemy import select
    from app.models import Organization
    from app.repositories import AlertRuleRepository, AlertRepository, MonitoringRepository

    async with get_db_context() as session:
        result = await session.execute(
            select(Organization.id).where(Organization.deleted_at.is_(None))
        )
        org_ids = [row[0] for row in result.all()]

        for org_id in org_ids:
            rule_repo = AlertRuleRepository(session)
            alert_repo = AlertRepository(session)
            metric_repo = MonitoringRepository(session)

            rules = await rule_repo.get_active_rules(org_id)

            for rule in rules:
                # Get latest metric value for this rule
                latest = await metric_repo.get_latest_value(
                    org_id, rule.metric, rule.model_name
                )
                if not latest:
                    continue

                current_value = latest.metric_value
                operator_fn = OPERATOR_MAP.get(rule.operator)
                if not operator_fn:
                    continue

                threshold_breached = operator_fn(current_value, rule.threshold)
                if not threshold_breached:
                    continue

                # Deduplication: skip if this rule already has an unresolved alert
                existing_alerts = await alert_repo.get_unresolved(org_id, limit=1000)
                already_firing = any(
                    a.alert_rule_id == rule.id for a in existing_alerts
                )
                if already_firing:
                    continue

                # Create alert
                alert = Alert(
                    organization_id=org_id,
                    alert_rule_id=rule.id,
                    severity=rule.severity,
                    metric=rule.metric,
                    current_value=current_value,
                    threshold=rule.threshold,
                    model_name=rule.model_name,
                    message=(
                        f"[{rule.severity.upper()}] {rule.name}: "
                        f"{rule.metric} = {current_value:.4f} "
                        f"({rule.operator} {rule.threshold})"
                    ),
                    acknowledged=False,
                    metadata={"window_minutes": rule.window_minutes},
                )
                session.add(alert)
                await session.flush()

                # Update rule last_triggered_at
                rule.last_triggered_at = datetime.now(timezone.utc)

                # Fire notifications
                for channel in (rule.notification_channels or []):
                    await _fire_notification(channel, rule.notification_config, alert)

        await session.commit()


async def _fire_notification(channel: str, config: Dict, alert: Alert):
    """
    Dispatch alert notification.
    
    Supported channels:
    - webhook: HTTP POST to any URL (powers Slack incoming webhooks, PagerDuty, etc.)
    - email: POST to internal email service (implement with SendGrid/SES)
    
    Why webhook-first?
    - One mechanism covers Slack, Teams, PagerDuty, OpsGenie, Discord
    - No per-integration code needed
    - Standard in monitoring tools (Grafana, Datadog)
    """
    if channel == "webhook":
        url = config.get("url")
        if not url:
            return
        payload = {
            "severity": alert.severity,
            "metric": alert.metric,
            "value": alert.current_value,
            "threshold": alert.threshold,
            "message": alert.message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(url, json=payload)
        except Exception as e:
            logger.warning(f"Webhook notification failed: {e}")

    elif channel == "email":
        # Placeholder: implement with SendGrid or AWS SES
        email = config.get("email")
        logger.info(f"Would send email to {email}: {alert.message}")
