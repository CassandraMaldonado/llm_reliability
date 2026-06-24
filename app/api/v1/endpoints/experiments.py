"""
app/api/v1/endpoints/experiments.py

Experiment tracking endpoints.

These are the highest-traffic endpoints in the platform.
Designed to be as fast as possible — DB writes are async,
long-running evals are offloaded to Celery.
"""
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.models import Experiment, ExperimentRun, LLMTrace
from app.core.config import settings

router = APIRouter()


# ── Pydantic Schemas ─────────────────────────────────────────────────────────
# Note: In a real project, these live in app/schemas/experiments.py
# Included here for completeness.

class ExperimentCreate(BaseModel):
    project_id: uuid.UUID
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    tags: List[str] = []
    metadata: dict = {}


class ExperimentResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    description: Optional[str]
    status: str
    tags: List[str]
    run_count: Optional[int] = None

    class Config:
        from_attributes = True


class RunCreate(BaseModel):
    """
    Create a new experiment run.
    The run defines WHAT to test (model + prompt + dataset).
    """
    experiment_id: uuid.UUID
    dataset_id: Optional[uuid.UUID] = None              # Eval against a dataset
    provider: str = Field(..., pattern="^(openai|anthropic|gemini|huggingface)$")
    model_name: str = Field(..., min_length=1, max_length=255)
    model_version: Optional[str] = None
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    prompt_version: Optional[str] = None
    hyperparameters: dict = Field(
        default={"temperature": 0.7, "max_tokens": 1024},
        description="Model hyperparameters: temperature, top_p, max_tokens, etc."
    )
    metrics_to_evaluate: List[str] = Field(
        default=["answer_relevance", "faithfulness", "hallucination_score", "semantic_similarity"],
        description="Which metrics to compute. Leave empty to skip auto-evaluation."
    )
    tags: List[str] = []
    metadata: dict = {}


class TraceCreate(BaseModel):
    """
    Log a single LLM call. Used by the SDK for production tracing.
    """
    run_id: Optional[uuid.UUID] = None
    source: str = Field(default="production", pattern="^(experiment|production)$")
    provider: str
    model_name: str
    system_prompt: Optional[str] = None
    user_prompt: str
    messages: Optional[List[dict]] = None
    completion: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    hyperparameters: dict = {}
    expected_output: Optional[str] = None
    context_documents: Optional[List[dict]] = None
    status: str = "success"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict = {}


class RunComparisonRequest(BaseModel):
    run_ids: List[uuid.UUID] = Field(..., min_length=2, max_length=10)
    metrics: Optional[List[str]] = None  # None = all metrics


# ── Experiment Endpoints ──────────────────────────────────────────────────────

@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    payload: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
    # current_user: User = Depends(get_current_user),  # Auth in production
):
    """
    Create a new experiment.

    An experiment is a named study — e.g., "GPT-4o vs Claude on customer support".
    It contains multiple runs, each with different models/prompts/configs.
    """
    experiment = Experiment(
        project_id=payload.project_id,
        organization_id=uuid.uuid4(),  # From auth in production
        name=payload.name,
        description=payload.description,
        tags=payload.tags,
        metadata_=payload.metadata,
        created_by_user_id=uuid.uuid4(),  # From auth in production
    )
    db.add(experiment)
    await db.flush()

    return ExperimentResponse(
        id=experiment.id,
        project_id=experiment.project_id,
        name=experiment.name,
        description=experiment.description,
        status=experiment.status,
        tags=experiment.tags,
        run_count=0,
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a single experiment with run count."""
    result = await db.execute(
        select(Experiment).where(
            and_(
                Experiment.id == experiment_id,
                Experiment.deleted_at.is_(None),
            )
        )
    )
    experiment = result.scalar_one_or_none()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Get run count efficiently
    count_result = await db.execute(
        select(func.count(ExperimentRun.id))
        .where(ExperimentRun.experiment_id == experiment_id)
    )
    run_count = count_result.scalar()

    return ExperimentResponse(
        id=experiment.id,
        project_id=experiment.project_id,
        name=experiment.name,
        description=experiment.description,
        status=experiment.status,
        tags=experiment.tags,
        run_count=run_count,
    )


@router.get("/", response_model=List[ExperimentResponse])
async def list_experiments(
    project_id: Optional[uuid.UUID] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    List experiments with optional filtering.

    Pagination: offset-based for simplicity in MVP.
    Enterprise upgrade: cursor-based pagination for datasets > 10k rows
    (offset gets slow with large offsets due to sequential scan).
    """
    query = select(Experiment).where(Experiment.deleted_at.is_(None))

    if project_id:
        query = query.where(Experiment.project_id == project_id)
    if status:
        query = query.where(Experiment.status == status)

    query = query.order_by(desc(Experiment.created_at)).limit(limit).offset(offset)
    result = await db.execute(query)
    experiments = result.scalars().all()

    return [
        ExperimentResponse(
            id=e.id,
            project_id=e.project_id,
            name=e.name,
            description=e.description,
            status=e.status,
            tags=e.tags,
        )
        for e in experiments
    ]


# ── Run Endpoints ─────────────────────────────────────────────────────────────

@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    payload: RunCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Create and trigger an experiment run.

    Returns 202 Accepted immediately — the actual evaluation runs async.
    The response includes a task_id to poll for status.

    Why 202 not 201:
    RFC 7231: 202 means "the request has been accepted for processing,
    but the processing has not been completed." Perfect for async jobs.
    """
    run = ExperimentRun(
        experiment_id=payload.experiment_id,
        organization_id=uuid.uuid4(),  # From auth in production
        provider=payload.provider,
        model_name=payload.model_name,
        model_version=payload.model_version,
        system_prompt=payload.system_prompt,
        prompt_template=payload.prompt_template,
        prompt_version=payload.prompt_version,
        hyperparameters=payload.hyperparameters,
        tags=payload.tags,
        metadata_=payload.metadata,
        status="pending",
        created_by_user_id=uuid.uuid4(),
    )
    db.add(run)
    await db.flush()

    # Dispatch to Celery asynchronously
    # In production: from app.tasks.evaluation import run_evaluation_task
    # task = run_evaluation_task.delay(str(run.id), payload.metrics_to_evaluate)
    # run.celery_task_id = task.id

    return {
        "run_id": str(run.id),
        "status": "pending",
        "message": "Run created. Evaluation will begin shortly.",
        # "task_id": task.id,  # Uncomment when Celery is wired
    }


@router.get("/runs/{run_id}")
async def get_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get run status and aggregated metrics.
    Poll this to check if an async eval run is complete.
    """
    result = await db.execute(
        select(ExperimentRun).where(ExperimentRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "id": str(run.id),
        "experiment_id": str(run.experiment_id),
        "provider": run.provider,
        "model_name": run.model_name,
        "status": run.status,
        "hyperparameters": run.hyperparameters,
        "metrics": {
            "avg_latency_ms": run.avg_latency_ms,
            "p95_latency_ms": run.p95_latency_ms,
            "total_cost_usd": run.total_cost_usd,
            "avg_answer_relevance": run.avg_answer_relevance,
            "avg_faithfulness": run.avg_faithfulness,
            "avg_hallucination_score": run.avg_hallucination_score,
            "avg_semantic_similarity": run.avg_semantic_similarity,
            "avg_toxicity_score": run.avg_toxicity_score,
        },
        "progress": {
            "total": run.total_samples,
            "completed": run.completed_samples,
            "failed": run.failed_samples,
        },
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


@router.post("/runs/compare")
async def compare_runs(
    payload: RunComparisonRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Side-by-side comparison of multiple runs.

    Returns a structured diff showing which run wins on each metric.
    This is the core "which prompt/model is better?" feature.

    Enterprise use: Gate prompt deployments — only ship if new run
    beats baseline on all critical metrics.
    """
    result = await db.execute(
        select(ExperimentRun).where(ExperimentRun.id.in_(payload.run_ids))
    )
    runs = result.scalars().all()

    if len(runs) != len(payload.run_ids):
        raise HTTPException(status_code=404, detail="One or more runs not found")

    # Build comparison table
    metrics = payload.metrics or [
        "avg_latency_ms", "total_cost_usd", "avg_answer_relevance",
        "avg_faithfulness", "avg_hallucination_score", "avg_semantic_similarity",
    ]

    comparison = {
        "runs": [],
        "winner_by_metric": {},
        "summary": {},
    }

    for run in runs:
        run_data = {
            "id": str(run.id),
            "model_name": run.model_name,
            "provider": run.provider,
            "prompt_version": run.prompt_version,
            "status": run.status,
            "metrics": {},
        }
        for metric in metrics:
            run_data["metrics"][metric] = getattr(run, metric, None)
        comparison["runs"].append(run_data)

    # Determine winner per metric
    for metric in metrics:
        values = {
            str(run.id): getattr(run, metric, None)
            for run in runs
            if getattr(run, metric, None) is not None
        }
        if not values:
            comparison["winner_by_metric"][metric] = None
            continue

        # Lower is better for latency/cost, higher is better for quality metrics
        lower_is_better = metric in ("avg_latency_ms", "total_cost_usd", "avg_cost_usd")
        winner_id = min(values, key=values.get) if lower_is_better else max(values, key=values.get)
        comparison["winner_by_metric"][metric] = winner_id

    return comparison


# ── Trace Endpoints ───────────────────────────────────────────────────────────

@router.post("/traces", status_code=status.HTTP_201_CREATED)
async def log_trace(
    payload: TraceCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Log a single LLM call trace.

    This is called by the MANGOS Python SDK automatically wrapping
    every LLM call. Lightweight path — just persist and return.
    Auto-evaluation is triggered as a background task if configured.

    SDK usage:
        from mangos import trace
        with trace(model="gpt-4o", prompt=...) as t:
            response = openai.chat.completions.create(...)
            t.record(response)
    """
    trace = LLMTrace(
        run_id=payload.run_id,
        organization_id=uuid.uuid4(),  # From SDK auth header in production
        source=payload.source,
        provider=payload.provider,
        model_name=payload.model_name,
        system_prompt=payload.system_prompt,
        user_prompt=payload.user_prompt,
        messages=payload.messages,
        completion=payload.completion,
        finish_reason=payload.finish_reason,
        prompt_tokens=payload.prompt_tokens,
        completion_tokens=payload.completion_tokens,
        total_tokens=payload.total_tokens,
        cost_usd=payload.cost_usd,
        latency_ms=payload.latency_ms,
        time_to_first_token_ms=payload.time_to_first_token_ms,
        hyperparameters=payload.hyperparameters,
        expected_output=payload.expected_output,
        context_documents=payload.context_documents,
        status=payload.status,
        error_code=payload.error_code,
        error_message=payload.error_message,
        metadata_=payload.metadata,
    )
    db.add(trace)
    await db.flush()

    # Background: auto-eval if expected_output provided
    # background_tasks.add_task(auto_evaluate_trace, trace_id=trace.id)

    return {
        "trace_id": str(trace.id),
        "status": "logged",
    }


@router.post("/traces/{trace_id}/feedback")
async def submit_feedback(
    trace_id: uuid.UUID,
    score: float = Query(..., ge=0.0, le=1.0),
    text: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit user feedback for a specific LLM response.

    Maps to the thumbs up/down in your product UI.
    Normalized: 1.0 = thumbs up, 0.0 = thumbs down.
    Used to correlate automatic eval scores with human judgment.
    """
    result = await db.execute(select(LLMTrace).where(LLMTrace.id == trace_id))
    trace = result.scalar_one_or_none()
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    trace.user_feedback_score = score
    trace.user_feedback_text = text

    return {"trace_id": str(trace_id), "feedback_recorded": True}
