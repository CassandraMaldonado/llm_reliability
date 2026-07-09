# Stat drift detection for production LLM metrics.

# 1. Statistical (KS test): Detects distribution shift over time windows.
# 2. Threshold-based: Simple rule-based alerting.
# 3. Z-score anomaly: Detects sudden spikes outside normal range.

#LLM providers silently update models. Hallucination rates, latency and cost all drift without warning. 
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

import numpy as np
from scipy import stats as scipy_stats
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.models import LLMTrace, EvaluationResult, DriftReport, Alert, AlertRule
from app.core.config import settings

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    STATISTICAL = "statistical"       # KS test detected distribution shift
    THRESHOLD = "threshold"           # Value crossed a fixed threshold
    ANOMALY = "anomaly"               # Z-score spike detected
    SUSTAINED = "sustained"           # Metric degraded and stayed there


@dataclass
class MetricWindow:
    """A time window of metric values for a specific model."""
    model_name: str
    provider: str
    metric_name: str
    values: List[float]
    window_start: datetime
    window_end: datetime
    sample_count: int

    @property
    def mean(self) -> Optional[float]:
        return float(np.mean(self.values)) if self.values else None

    @property
    def std(self) -> Optional[float]:
        return float(np.std(self.values)) if len(self.values) > 1 else None

    @property
    def p50(self) -> Optional[float]:
        return float(np.percentile(self.values, 50)) if self.values else None

    @property
    def p95(self) -> Optional[float]:
        return float(np.percentile(self.values, 95)) if self.values else None


@dataclass
class DriftResult:
    """Result of a drift detection check."""
    is_drift: bool
    drift_type: DriftType
    drift_score: float                 # KS statistic, Z-score, or 0/1 for threshold
    p_value: Optional[float]
    baseline: MetricWindow
    current: MetricWindow
    description: str
    severity: str = "warning"          # info, warning, critical
    metadata: Dict[str, Any] = field(default_factory=dict)


class DriftDetector:
    """
    Detects statistical drift in LLM metric time-series.

    Usage:
        detector = DriftDetector(db_session)
        results = await detector.detect_all(
            organization_id=org_id,
            model_name="gpt-4o",
            provider="openai",
        )
    """

    # Which metrics to monitor and their drift config
    METRIC_CONFIG: Dict[str, Dict[str, Any]] = {
        "latency_ms": {
            "higher_is_worse": True,
            "threshold_critical": 10000,   # 10s is critical
            "threshold_warning": 5000,     # 5s is warning
            "description": "Response latency",
            "unit": "ms",
        },
        "cost_usd": {
            "higher_is_worse": True,
            "threshold_critical": 0.10,    # $0.10 per call is very expensive
            "threshold_warning": 0.05,
            "description": "Per-call cost",
            "unit": "USD",
        },
        "hallucination_score": {
            "higher_is_worse": False,      # higher = less hallucination = better
            "threshold_critical": 0.6,     # below 0.6 = lots of hallucinations
            "threshold_warning": 0.75,
            "description": "Hallucination rate",
            "unit": "score",
        },
        "answer_relevance": {
            "higher_is_worse": False,
            "threshold_critical": 0.5,
            "threshold_warning": 0.65,
            "description": "Answer relevance",
            "unit": "score",
        },
        "faithfulness": {
            "higher_is_worse": False,
            "threshold_critical": 0.6,
            "threshold_warning": 0.75,
            "description": "Faithfulness to context",
            "unit": "score",
        },
    }

    def __init__(
        self,
        db: AsyncSession,
        ks_pvalue_threshold: float = None,
        zscore_threshold: float = None,
    ):
        self.db = db
        self.ks_pvalue_threshold = ks_pvalue_threshold or settings.DRIFT_KS_PVALUE_THRESHOLD
        self.zscore_threshold = zscore_threshold or settings.DRIFT_ZSCORE_THRESHOLD

    async def get_metric_window(
        self,
        organization_id: UUID,
        model_name: str,
        provider: str,
        metric_name: str,
        window_start: datetime,
        window_end: datetime,
    ) -> MetricWindow:
        """
        Pull metric values for a given time window.
        Handles both built-in trace metrics (latency, cost) and
        computed eval metrics (hallucination, relevance).
        """
        if metric_name in ("latency_ms", "cost_usd", "total_tokens"):
            # Built-in trace-level metrics
            column_map = {
                "latency_ms": LLMTrace.latency_ms,
                "cost_usd": LLMTrace.cost_usd,
                "total_tokens": LLMTrace.total_tokens,
            }
            col = column_map[metric_name]
            query = (
                select(col)
                .where(
                    and_(
                        LLMTrace.organization_id == organization_id,
                        LLMTrace.model_name == model_name,
                        LLMTrace.provider == provider,
                        LLMTrace.created_at >= window_start,
                        LLMTrace.created_at < window_end,
                        LLMTrace.status == "success",
                        col.isnot(None),
                    )
                )
            )
        else:
            # Evaluation metrics from evaluation_results table
            query = (
                select(EvaluationResult.score)
                .join(LLMTrace, EvaluationResult.trace_id == LLMTrace.id)
                .where(
                    and_(
                        EvaluationResult.organization_id == organization_id,
                        EvaluationResult.metric_name == metric_name,
                        LLMTrace.model_name == model_name,
                        LLMTrace.provider == provider,
                        EvaluationResult.evaluated_at >= window_start,
                        EvaluationResult.evaluated_at < window_end,
                    )
                )
            )

        result = await self.db.execute(query)
        values = [float(row[0]) for row in result.fetchall() if row[0] is not None]

        return MetricWindow(
            model_name=model_name,
            provider=provider,
            metric_name=metric_name,
            values=values,
            window_start=window_start,
            window_end=window_end,
            sample_count=len(values),
        )

    def _ks_test(self, baseline: MetricWindow, current: MetricWindow) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution shift.

        Returns (statistic, p_value).
        Low p-value (< threshold) = distributions are significantly different = drift.

        Why KS test:
        - Non-parametric (doesn't assume normal distribution)
        - Works on any continuous metric
        - Detects shape changes, not just mean shift
        - Industry standard for data drift detection (used in Evidently AI, Great Expectations)
        """
        if len(baseline.values) < 10 or len(current.values) < 10:
            # Not enough data for a reliable test
            return 0.0, 1.0
        statistic, p_value = scipy_stats.ks_2samp(baseline.values, current.values)
        return float(statistic), float(p_value)

    def _zscore_anomaly(self, baseline: MetricWindow, current: MetricWindow) -> float:
        """
        Z-score of current mean vs baseline distribution.

        Z-score > threshold = current mean is unusually far from baseline mean.
        More sensitive to mean shift than KS test.
        Use alongside KS for complementary coverage.
        """
        if not baseline.values or not current.values:
            return 0.0
        if baseline.std is None or baseline.std == 0:
            return 0.0

        z_score = abs((current.mean - baseline.mean) / baseline.std)
        return float(z_score)

    def _threshold_check(
        self,
        current: MetricWindow,
        metric_config: Dict[str, Any],
    ) -> Tuple[bool, str, float]:
        """
        Simple threshold rule: is the metric above/below a fixed limit?

        Returns (triggered, severity, current_value).
        """
        if not current.values:
            return False, "info", 0.0

        # Use p95 for latency/cost (tail matters), mean for quality metrics
        check_value = current.p95 if metric_config.get("higher_is_worse") else current.mean
        if check_value is None:
            return False, "info", 0.0

        higher_is_worse = metric_config.get("higher_is_worse", True)
        critical_threshold = metric_config.get("threshold_critical")
        warning_threshold = metric_config.get("threshold_warning")

        if higher_is_worse:
            if critical_threshold and check_value > critical_threshold:
                return True, "critical", check_value
            if warning_threshold and check_value > warning_threshold:
                return True, "warning", check_value
        else:
            if critical_threshold and check_value < critical_threshold:
                return True, "critical", check_value
            if warning_threshold and check_value < warning_threshold:
                return True, "warning", check_value

        return False, "info", check_value

    async def detect_metric_drift(
        self,
        organization_id: UUID,
        model_name: str,
        provider: str,
        metric_name: str,
    ) -> Optional[DriftResult]:
        """
        Run all drift detection strategies for one metric.
        Returns DriftResult if drift detected, None otherwise.
        """
        now = datetime.now(timezone.utc)

        baseline_end = now - timedelta(hours=settings.DRIFT_CURRENT_WINDOW_HOURS)
        baseline_start = baseline_end - timedelta(hours=settings.DRIFT_BASELINE_WINDOW_HOURS)
        current_start = now - timedelta(hours=settings.DRIFT_CURRENT_WINDOW_HOURS)

        baseline = await self.get_metric_window(
            organization_id, model_name, provider, metric_name,
            baseline_start, baseline_end
        )
        current = await self.get_metric_window(
            organization_id, model_name, provider, metric_name,
            current_start, now
        )

        # Need minimum data to be meaningful
        if baseline.sample_count < 5 or current.sample_count < 3:
            logger.debug(
                f"Insufficient data for drift detection: "
                f"{metric_name} on {model_name} "
                f"(baseline={baseline.sample_count}, current={current.sample_count})"
            )
            return None

        metric_config = self.METRIC_CONFIG.get(metric_name, {})

        # 1. Statistical drift (KS test)
        ks_stat, ks_pvalue = self._ks_test(baseline, current)
        if ks_pvalue < self.ks_pvalue_threshold:
            severity = "critical" if ks_pvalue < 0.01 else "warning"
            return DriftResult(
                is_drift=True,
                drift_type=DriftType.STATISTICAL,
                drift_score=ks_stat,
                p_value=ks_pvalue,
                baseline=baseline,
                current=current,
                severity=severity,
                description=(
                    f"Statistical distribution shift detected in {metric_name} "
                    f"for {model_name}. "
                    f"KS statistic={ks_stat:.3f}, p-value={ks_pvalue:.4f}. "
                    f"Baseline mean={baseline.mean:.3f}, Current mean={current.mean:.3f}"
                ),
                metadata={"ks_statistic": ks_stat, "ks_pvalue": ks_pvalue},
            )

        # 2. Z-score anomaly
        z_score = self._zscore_anomaly(baseline, current)
        if z_score > self.zscore_threshold:
            severity = "critical" if z_score > self.zscore_threshold * 1.5 else "warning"
            return DriftResult(
                is_drift=True,
                drift_type=DriftType.ANOMALY,
                drift_score=z_score,
                p_value=None,
                baseline=baseline,
                current=current,
                severity=severity,
                description=(
                    f"Anomaly detected in {metric_name} for {model_name}. "
                    f"Z-score={z_score:.2f} (threshold={self.zscore_threshold}). "
                    f"Current mean {current.mean:.3f} vs baseline {baseline.mean:.3f} ± {baseline.std:.3f}"
                ),
                metadata={"z_score": z_score},
            )

        # 3. Threshold check
        if metric_config:
            triggered, severity, check_value = self._threshold_check(current, metric_config)
            if triggered:
                return DriftResult(
                    is_drift=True,
                    drift_type=DriftType.THRESHOLD,
                    drift_score=check_value,
                    p_value=None,
                    baseline=baseline,
                    current=current,
                    severity=severity,
                    description=(
                        f"Threshold breach: {metric_name} for {model_name} "
                        f"reached {check_value:.3f} {metric_config.get('unit', '')} "
                        f"({severity} threshold)"
                    ),
                    metadata={"threshold_value": check_value, "metric_config": metric_config},
                )

        return None  # No drift detected

    async def detect_all(
        self,
        organization_id: UUID,
        model_name: str,
        provider: str,
    ) -> List[DriftResult]:
        """
        Run drift detection for all monitored metrics concurrently.
        Returns list of drift results where drift was detected.
        """
        tasks = [
            self.detect_metric_drift(organization_id, model_name, provider, metric_name)
            for metric_name in self.METRIC_CONFIG.keys()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        drift_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Drift detection error for {list(self.METRIC_CONFIG.keys())[i]}: {result}"
                )
            elif result is not None:
                drift_results.append(result)

        return drift_results
