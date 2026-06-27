"""app/api/v1/endpoints/runs.py — Experiment run endpoints (separate from experiments.py)."""
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import User
from app.repositories import ExperimentRunRepository, EvaluationRepository
from app.schemas.experiments import ExperimentRunResponse, RunCompareRequest, RunCompareResponse, RunMetricComparison
from app.schemas.evaluations import EvaluationSummary
from app.schemas.common import PaginatedResponse

router = APIRouter()


@router.get("/{run_id}", response_model=ExperimentRunResponse)
async def get_run(
    run_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = ExperimentRunRepository(session)
    run = await repo.get_by_id(run_id, current_user.organization_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/summary", response_model=EvaluationSummary)
async def get_run_summary(
    run_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get aggregated evaluation metrics for a completed run."""
    run_repo = ExperimentRunRepository(session)
    run = await run_repo.get_by_id(run_id, current_user.organization_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    eval_repo = EvaluationRepository(session)
    metrics = await eval_repo.get_summary_for_run(run_id)

    return EvaluationSummary(
        run_id=str(run_id),
        total_evaluated=run.sample_count,
        metrics={
            name: {
                "mean": float(data.get("mean") or 0),
                "p50": float(data.get("p50") or 0),
                "p95": float(data.get("p95") or 0),
                "pass_rate": (data.get("pass_count") or 0) / max(data.get("total", 1), 1),
            }
            for name, data in metrics.items()
        },
        overall_pass_rate=sum(
            (d.get("pass_count") or 0) for d in metrics.values()
        ) / max(sum(d.get("total", 1) for d in metrics.values()), 1),
        cost_usd=float(run.total_cost_usd or 0),
    )


@router.post("/compare", response_model=RunCompareResponse)
async def compare_runs(
    data: RunCompareRequest,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Side-by-side metric comparison of 2-10 runs.
    Identifies winner per metric and overall champion.
    """
    repo = ExperimentRunRepository(session)
    runs = await repo.get_multiple(data.run_ids, current_user.organization_id)

    if len(runs) < 2:
        raise HTTPException(status_code=400, detail="At least 2 valid runs required")

    # Metrics where lower is better
    lower_is_better = {"avg_latency_ms", "avg_cost_usd", "avg_hallucination_score", "error_count"}

    comparisons: List[RunMetricComparison] = []
    wins: dict = {str(run.id): 0 for run in runs}

    for metric in data.metrics:
        values = {}
        for run in runs:
            v = getattr(run, metric, None)
            if v is not None:
                values[str(run.id)] = float(v)

        if not values:
            comparisons.append(RunMetricComparison(metric=metric, values={}, winner_run_id=None, delta_pct=None))
            continue

        if metric in lower_is_better:
            winner_id = min(values, key=values.__getitem__)
        else:
            winner_id = max(values, key=values.__getitem__)

        wins[winner_id] = wins.get(winner_id, 0) + 1

        sorted_vals = sorted(values.values(), reverse=(metric not in lower_is_better))
        delta_pct = None
        if len(sorted_vals) >= 2 and sorted_vals[-1] != 0:
            delta_pct = round(((sorted_vals[0] - sorted_vals[-1]) / abs(sorted_vals[-1])) * 100, 2)

        comparisons.append(RunMetricComparison(
            metric=metric,
            values=values,
            winner_run_id=winner_id,
            delta_pct=delta_pct,
        ))

    overall_winner = max(wins, key=wins.__getitem__) if wins else None
    winner_run = next((r for r in runs if str(r.id) == overall_winner), None)
    summary = (
        f"{winner_run.model_name} ({winner_run.provider}) wins {wins.get(overall_winner, 0)}/{len(data.metrics)} metrics."
        if winner_run else "No clear winner."
    )

    return RunCompareResponse(
        run_ids=[str(r.id) for r in runs],
        runs=runs,
        metric_comparisons=comparisons,
        overall_winner_run_id=overall_winner,
        summary=summary,
    )
