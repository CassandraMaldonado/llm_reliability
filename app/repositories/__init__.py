"""
app/repositories/

All domain repositories inheriting from BaseRepository.
Each adds domain-specific query methods beyond basic CRUD.
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, func, and_, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base import BaseRepository
from app.models import (
    Organization, User, ApiKey,
    Experiment, ExperimentRun, LLMTrace, EvaluationResult,
    Dataset, DatasetRow, RAGEvaluation, MonitoringMetric,
    AlertRule, Alert, DriftReport,
)


# ─────────────────────────────────────────────────────────────────────────────
# ORGANIZATION & USER
# ─────────────────────────────────────────────────────────────────────────────

class OrganizationRepository(BaseRepository[Organization]):
    model = Organization

    async def get_by_slug(self, slug: str) -> Optional[Organization]:
        result = await self.session.execute(
            select(Organization).where(Organization.slug == slug)
        )
        return result.scalar_one_or_none()


class UserRepository(BaseRepository[User]):
    model = User

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_by_id_any_org(self, user_id: uuid.UUID) -> Optional[User]:
        """For auth middleware where we don't yet know the org."""
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()


class ApiKeyRepository(BaseRepository[ApiKey]):
    model = ApiKey

    async def get_by_key_hash(self, key_hash: str) -> Optional[ApiKey]:
        result = await self.session.execute(
            select(ApiKey).where(
                and_(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
            )
        )
        return result.scalar_one_or_none()

    async def update_last_used(self, api_key: ApiKey) -> None:
        api_key.last_used_at = datetime.now(timezone.utc)
        await self.session.flush()


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRepository(BaseRepository[Experiment]):
    model = Experiment

    async def get_with_run_count(
        self, org_id: uuid.UUID, offset: int = 0, limit: int = 20
    ) -> Tuple[List[Dict], int]:
        """Return experiments with their run counts in one query."""
        query = (
            select(
                Experiment,
                func.count(ExperimentRun.id).label("run_count")
            )
            .outerjoin(ExperimentRun, ExperimentRun.experiment_id == Experiment.id)
            .where(
                and_(
                    Experiment.organization_id == org_id,
                    Experiment.deleted_at.is_(None)
                )
            )
            .group_by(Experiment.id)
            .order_by(desc(Experiment.created_at))
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(query)
        rows = result.all()

        count_result = await self.session.execute(
            select(func.count(Experiment.id)).where(
                and_(
                    Experiment.organization_id == org_id,
                    Experiment.deleted_at.is_(None)
                )
            )
        )
        total = count_result.scalar_one()
        return rows, total


class ExperimentRunRepository(BaseRepository[ExperimentRun]):
    model = ExperimentRun

    async def get_by_experiment(
        self, experiment_id: uuid.UUID, org_id: uuid.UUID
    ) -> List[ExperimentRun]:
        result = await self.session.execute(
            select(ExperimentRun).where(
                and_(
                    ExperimentRun.experiment_id == experiment_id,
                    ExperimentRun.organization_id == org_id,
                )
            ).order_by(desc(ExperimentRun.created_at))
        )
        return list(result.scalars().all())

    async def get_multiple(
        self, run_ids: List[uuid.UUID], org_id: uuid.UUID
    ) -> List[ExperimentRun]:
        result = await self.session.execute(
            select(ExperimentRun).where(
                and_(
                    ExperimentRun.id.in_(run_ids),
                    ExperimentRun.organization_id == org_id,
                )
            )
        )
        return list(result.scalars().all())

    async def update_status(
        self, run_id: uuid.UUID, status: str, error_message: Optional[str] = None
    ) -> None:
        result = await self.session.execute(
            select(ExperimentRun).where(ExperimentRun.id == run_id)
        )
        run = result.scalar_one_or_none()
        if run:
            run.status = status
            if error_message:
                run.error_message = error_message
            if status == "running":
                run.started_at = datetime.now(timezone.utc)
            elif status in ("completed", "failed"):
                run.completed_at = datetime.now(timezone.utc)
            await self.session.flush()

    async def update_aggregated_metrics(
        self, run_id: uuid.UUID, metrics: Dict[str, Any]
    ) -> None:
        result = await self.session.execute(
            select(ExperimentRun).where(ExperimentRun.id == run_id)
        )
        run = result.scalar_one_or_none()
        if run:
            for key, value in metrics.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            await self.session.flush()


# ─────────────────────────────────────────────────────────────────────────────
# TRACES
# ─────────────────────────────────────────────────────────────────────────────

class TraceRepository(BaseRepository[LLMTrace]):
    model = LLMTrace

    async def get_by_run(
        self, run_id: uuid.UUID, org_id: uuid.UUID,
        offset: int = 0, limit: int = 50
    ) -> Tuple[List[LLMTrace], int]:
        where = and_(
            LLMTrace.experiment_run_id == run_id,
            LLMTrace.organization_id == org_id,
        )
        count_result = await self.session.execute(
            select(func.count()).select_from(LLMTrace).where(where)
        )
        total = count_result.scalar_one()

        result = await self.session.execute(
            select(LLMTrace).where(where)
            .order_by(desc(LLMTrace.created_at))
            .offset(offset).limit(limit)
        )
        return list(result.scalars().all()), total

    async def get_recent_for_org(
        self, org_id: uuid.UUID, hours: int = 24, model_name: Optional[str] = None
    ) -> List[LLMTrace]:
        """Used by drift detection to get baseline and current window traces."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        conditions = [
            LLMTrace.organization_id == org_id,
            LLMTrace.created_at >= cutoff,
        ]
        if model_name:
            conditions.append(LLMTrace.model_name == model_name)

        result = await self.session.execute(
            select(LLMTrace).where(and_(*conditions))
            .order_by(LLMTrace.created_at.asc())
        )
        return list(result.scalars().all())


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATIONS
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationRepository(BaseRepository[EvaluationResult]):
    model = EvaluationResult

    async def get_summary_for_run(self, run_id: uuid.UUID) -> Dict[str, Any]:
        """Aggregate evaluation scores per metric for a run."""
        result = await self.session.execute(
            select(
                EvaluationResult.metric_name,
                func.avg(EvaluationResult.score).label("mean"),
                func.percentile_cont(0.5).within_group(
                    EvaluationResult.score
                ).label("p50"),
                func.percentile_cont(0.95).within_group(
                    EvaluationResult.score
                ).label("p95"),
                func.sum(
                    func.cast(EvaluationResult.passed, Integer)
                ).label("pass_count"),
                func.count(EvaluationResult.id).label("total"),
            )
            .where(EvaluationResult.experiment_run_id == run_id)
            .group_by(EvaluationResult.metric_name)
        )
        return {row.metric_name: dict(row._mapping) for row in result.all()}


# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────

class DatasetRepository(BaseRepository[Dataset]):
    model = Dataset

    async def get_rows(
        self, dataset_id: uuid.UUID, offset: int = 0, limit: int = 100
    ) -> Tuple[List[DatasetRow], int]:
        where = DatasetRow.dataset_id == dataset_id

        count_result = await self.session.execute(
            select(func.count()).select_from(DatasetRow).where(where)
        )
        total = count_result.scalar_one()

        result = await self.session.execute(
            select(DatasetRow).where(where).offset(offset).limit(limit)
        )
        return list(result.scalars().all()), total


# ─────────────────────────────────────────────────────────────────────────────
# RAG EVALUATIONS
# ─────────────────────────────────────────────────────────────────────────────

class RAGRepository(BaseRepository[RAGEvaluation]):
    model = RAGEvaluation

    async def get_grouped_stats(
        self, org_id: uuid.UUID, group_by: str, eval_ids: List[uuid.UUID]
    ) -> List[Dict[str, Any]]:
        """Group RAG evaluations by config dimension for comparison."""
        group_col = getattr(RAGEvaluation, group_by, None)
        if group_col is None:
            return []

        result = await self.session.execute(
            select(
                group_col.label("group_value"),
                func.count(RAGEvaluation.id).label("count"),
                func.avg(RAGEvaluation.retrieval_precision).label("avg_retrieval_precision"),
                func.avg(RAGEvaluation.retrieval_recall).label("avg_retrieval_recall"),
                func.avg(RAGEvaluation.context_relevance).label("avg_context_relevance"),
                func.avg(RAGEvaluation.groundedness).label("avg_groundedness"),
                func.avg(RAGEvaluation.answer_correctness).label("avg_answer_correctness"),
            )
            .where(
                and_(
                    RAGEvaluation.organization_id == org_id,
                    RAGEvaluation.id.in_(eval_ids),
                )
            )
            .group_by(group_col)
        )
        return [dict(row._mapping) for row in result.all()]


# ─────────────────────────────────────────────────────────────────────────────
# MONITORING & METRICS
# ─────────────────────────────────────────────────────────────────────────────

class MonitoringRepository(BaseRepository[MonitoringMetric]):
    model = MonitoringMetric

    async def get_time_series(
        self,
        org_id: uuid.UUID,
        metric_name: str,
        hours: int = 24,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        conditions = [
            MonitoringMetric.organization_id == org_id,
            MonitoringMetric.metric_name == metric_name,
            MonitoringMetric.window_start >= cutoff,
        ]
        if model_name:
            conditions.append(MonitoringMetric.model_name == model_name)

        result = await self.session.execute(
            select(MonitoringMetric)
            .where(and_(*conditions))
            .order_by(MonitoringMetric.window_start.asc())
        )
        return list(result.scalars().all())

    async def get_latest_value(
        self, org_id: uuid.UUID, metric_name: str, model_name: Optional[str] = None
    ) -> Optional[MonitoringMetric]:
        conditions = [
            MonitoringMetric.organization_id == org_id,
            MonitoringMetric.metric_name == metric_name,
        ]
        if model_name:
            conditions.append(MonitoringMetric.model_name == model_name)

        result = await self.session.execute(
            select(MonitoringMetric)
            .where(and_(*conditions))
            .order_by(desc(MonitoringMetric.window_end))
            .limit(1)
        )
        return result.scalar_one_or_none()


# ─────────────────────────────────────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────────────────────────────────────

class AlertRuleRepository(BaseRepository[AlertRule]):
    model = AlertRule

    async def get_active_rules(self, org_id: uuid.UUID) -> List[AlertRule]:
        result = await self.session.execute(
            select(AlertRule).where(
                and_(
                    AlertRule.organization_id == org_id,
                    AlertRule.is_active == True,
                )
            )
        )
        return list(result.scalars().all())


class AlertRepository(BaseRepository[Alert]):
    model = Alert

    async def get_unresolved(
        self, org_id: uuid.UUID, limit: int = 50
    ) -> List[Alert]:
        result = await self.session.execute(
            select(Alert).where(
                and_(
                    Alert.organization_id == org_id,
                    Alert.resolved_at.is_(None),
                )
            )
            .order_by(desc(Alert.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def acknowledge(self, alert: Alert) -> Alert:
        alert.acknowledged = True
        alert.acknowledged_at = datetime.now(timezone.utc)
        await self.session.flush()
        return alert

    async def resolve(self, alert: Alert) -> Alert:
        alert.resolved_at = datetime.now(timezone.utc)
        await self.session.flush()
        return alert


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT
# ─────────────────────────────────────────────────────────────────────────────

class DriftRepository(BaseRepository[DriftReport]):
    model = DriftReport

    async def get_latest(
        self, org_id: uuid.UUID, model_name: Optional[str] = None
    ) -> Optional[DriftReport]:
        conditions = [DriftReport.organization_id == org_id]
        if model_name:
            conditions.append(DriftReport.model_name == model_name)

        result = await self.session.execute(
            select(DriftReport)
            .where(and_(*conditions))
            .order_by(desc(DriftReport.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()
