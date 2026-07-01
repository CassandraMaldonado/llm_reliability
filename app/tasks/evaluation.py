# Celery tasks for LLM evaluation.

# - Tasks report progress for real-time UI updates.
- All state lives in PostgreSQL, not Celery result backend
  (Celery results expire; DB records don't)
- Tasks must handle partial failures (skip failed rows, not abort entire run)

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

import numpy as np
from celery import Celery, Task
from celery.utils.log import get_task_logger

from app.core.config import settings

logger = get_task_logger(__name__)

# ── Celery App ────────────────────────────────────────────────────────────────
celery_app = Celery(
    "mangos",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,               # Report STARTED state (important for progress tracking)
    task_acks_late=True,                   # Ack AFTER completion (prevents lost tasks on worker crash)
    worker_prefetch_multiplier=1,          # Process one task at a time per worker (fair queue)
    task_soft_time_limit=settings.CELERY_TASK_TIMEOUT_SECONDS,
    task_time_limit=settings.CELERY_TASK_TIMEOUT_SECONDS + 60,
    task_max_retries=settings.CELERY_MAX_RETRIES,

    # Priority queues
    task_queues={
        "high": {"exchange": "high", "routing_key": "high"},
        "default": {"exchange": "default", "routing_key": "default"},
        "low": {"exchange": "low", "routing_key": "low"},
    },
    task_default_queue="default",

    # Beat schedule for periodic tasks
    beat_schedule={
        "drift-detection": {
            "task": "app.tasks.monitoring.run_drift_detection",
            "schedule": settings.DRIFT_DETECTION_INTERVAL_MINUTES * 60,  # seconds
            "options": {"queue": "low"},
        },
        "aggregate-monitoring-metrics": {
            "task": "app.tasks.monitoring.aggregate_monitoring_metrics",
            "schedule": 300,  # Every 5 minutes
            "options": {"queue": "low"},
        },
    },
)


class DatabaseTask(Task):
    """
    Base Celery task class that provides async database access.

    Pattern: Tasks need a DB session, but Celery workers are not FastAPI.
    We use get_db_context() to get a session inside task execution.
    The session is NOT shared between tasks (each task gets its own).
    """
    abstract = True
    _loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def run_async(self, coro):
        """Run a coroutine synchronously from within a Celery task."""
        return self.loop.run_until_complete(coro)


# ── Evaluation Task ────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="app.tasks.evaluation.run_evaluation_task",
    queue="default",
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
)
def run_evaluation_task(
    self: DatabaseTask,
    run_id: str,
    metric_names: List[str],
    evaluator_api_key: Optional[str] = None,
    evaluator_model: str = "gpt-4o-mini",
) -> dict:
    """
    Main evaluation task. Runs all specified metrics against a dataset.

    Flow:
    1. Load ExperimentRun + associated dataset rows from DB
    2. For each row: call the target LLM, log trace, run eval metrics
    3. Aggregate results, update ExperimentRun with summary stats
    4. Trigger alert check if thresholds configured

    Task is idempotent: if it fails mid-run, it picks up from where it left off
    (completed traces are skipped on retry).

    Returns summary statistics for the run.
    """
    return self.run_async(_run_evaluation_async(
        task=self,
        run_id=UUID(run_id),
        metric_names=metric_names,
        evaluator_api_key=evaluator_api_key,
        evaluator_model=evaluator_model,
    ))


async def _run_evaluation_async(
    task: DatabaseTask,
    run_id: UUID,
    metric_names: List[str],
    evaluator_api_key: Optional[str],
    evaluator_model: str,
) -> dict:
    """
    Async implementation of the evaluation pipeline.
    Separated from the Celery task wrapper for testability.
    """
    from app.core.database import get_db_context
    from app.models import ExperimentRun, DatasetRow, LLMTrace, EvaluationResult
    from app.evaluators.metrics import EvaluationRunner, MetricInput
    from sqlalchemy import select, and_

    logger.info(f"Starting evaluation for run {run_id}")

    async with get_db_context() as db:
        # 1. Load the run
        result = await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
        run = result.scalar_one_or_none()
        if not run:
            raise ValueError(f"Run {run_id} not found")

        # Mark as running
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        await db.flush()

        try:
            # 2. Load dataset if configured (otherwise eval existing traces)
            dataset_rows = []
            if hasattr(run, 'dataset_id') and run.metadata_.get('dataset_id'):
                dataset_id = run.metadata_['dataset_id']
                rows_result = await db.execute(
                    select(DatasetRow)
                    .where(DatasetRow.dataset_id == dataset_id)
                    .order_by(DatasetRow.row_index)
                )
                dataset_rows = rows_result.scalars().all()

            total = len(dataset_rows)
            run.total_samples = total
            await db.flush()

            # 3. Build evaluation runner
            runner = EvaluationRunner.from_metric_names(
                metric_names=metric_names,
                evaluator_api_key=evaluator_api_key,
                evaluator_model=evaluator_model,
            )

            # 4. For each dataset row, call LLM + evaluate
            all_metric_scores: dict = {name: [] for name in metric_names}
            latencies: List[float] = []
            costs: List[float] = []
            completed = 0
            failed = 0

            for i, row in enumerate(dataset_rows):
                try:
                    # Update progress
                    task.update_state(
                        state="PROGRESS",
                        meta={"current": i, "total": total, "run_id": str(run_id)}
                    )

                    # Call the target LLM (simplified — production would use provider clients)
                    trace_start = time.monotonic()
                    llm_response = await _call_llm(run, row.question)
                    latency_ms = (time.monotonic() - trace_start) * 1000

                    # Log the trace
                    trace = LLMTrace(
                        run_id=run_id,
                        organization_id=run.organization_id,
                        source="experiment",
                        provider=run.provider,
                        model_name=run.model_name,
                        system_prompt=run.system_prompt,
                        user_prompt=row.question,
                        completion=llm_response.get("content"),
                        prompt_tokens=llm_response.get("prompt_tokens"),
                        completion_tokens=llm_response.get("completion_tokens"),
                        total_tokens=llm_response.get("total_tokens"),
                        cost_usd=llm_response.get("cost_usd"),
                        latency_ms=latency_ms,
                        hyperparameters=run.hyperparameters,
                        expected_output=row.expected_answer,
                        context_documents=row.context_documents,
                        status="success",
                    )
                    db.add(trace)
                    await db.flush()

                    if latency_ms:
                        latencies.append(latency_ms)
                    if llm_response.get("cost_usd"):
                        costs.append(llm_response["cost_usd"])

                    # Run evaluation metrics
                    eval_input = MetricInput(
                        question=row.question,
                        actual_output=llm_response.get("content", ""),
                        expected_output=row.expected_answer,
                        context_documents=[
                            doc.get("text", "") for doc in (row.context_documents or [])
                        ],
                    )
                    metric_results = await runner.run(eval_input)

                    # Store individual metric results
                    for metric_result in metric_results:
                        eval_record = EvaluationResult(
                            trace_id=trace.id,
                            run_id=run_id,
                            organization_id=run.organization_id,
                            metric_name=metric_result.metric_name,
                            score=metric_result.score,
                            passed=metric_result.passed,
                            threshold=metric_result.threshold,
                            reasoning=metric_result.reasoning,
                            evaluator_model=metric_result.evaluator_model,
                            metadata_=metric_result.metadata,
                        )
                        db.add(eval_record)
                        all_metric_scores[metric_result.metric_name].append(metric_result.score)

                    await db.flush()
                    completed += 1

                except Exception as e:
                    logger.error(f"Failed to evaluate row {i} for run {run_id}: {e}")
                    failed += 1
                    continue

            # 5. Compute aggregate stats and update run
            run.completed_samples = completed
            run.failed_samples = failed

            if latencies:
                run.avg_latency_ms = float(np.mean(latencies))
                run.p50_latency_ms = float(np.percentile(latencies, 50))
                run.p95_latency_ms = float(np.percentile(latencies, 95))
                run.p99_latency_ms = float(np.percentile(latencies, 99))

            if costs:
                run.total_cost_usd = float(sum(costs))
                run.avg_cost_usd = float(np.mean(costs))

            metric_field_map = {
                "answer_relevance": "avg_answer_relevance",
                "faithfulness": "avg_faithfulness",
                "hallucination_score": "avg_hallucination_score",
                "semantic_similarity": "avg_semantic_similarity",
                "toxicity_score": "avg_toxicity_score",
            }
            for metric_name, scores in all_metric_scores.items():
                if scores:
                    field_name = metric_field_map.get(metric_name)
                    if field_name:
                        setattr(run, field_name, float(np.mean(scores)))

            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            await db.flush()

            logger.info(
                f"Evaluation complete for run {run_id}: "
                f"{completed} completed, {failed} failed"
            )

            return {
                "run_id": str(run_id),
                "status": "completed",
                "completed_samples": completed,
                "failed_samples": failed,
                "avg_latency_ms": run.avg_latency_ms,
                "total_cost_usd": run.total_cost_usd,
            }

        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc)
            await db.flush()
            logger.error(f"Run {run_id} failed: {e}")
            raise


async def _call_llm(run: "ExperimentRun", prompt: str) -> dict:
    """
    Call the appropriate LLM provider for a run.
    Returns normalized response dict regardless of provider.

    Cost calculation uses per-token pricing tables.
    Production: prices should be fetched from config/DB, not hardcoded.
    """
    import httpx

    # Token cost tables (per 1M tokens, USD) — update as providers change pricing
    OPENAI_COSTS = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    }

    if run.provider == "openai":
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured")

        messages = []
        if run.system_prompt:
            messages.append({"role": "system", "content": run.system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": run.model_name,
                    "messages": messages,
                    **run.hyperparameters,
                }
            )
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Calculate cost
        cost_per_m = OPENAI_COSTS.get(run.model_name, {"input": 5.0, "output": 15.0})
        cost_usd = (
            prompt_tokens * cost_per_m["input"] / 1_000_000
            + completion_tokens * cost_per_m["output"] / 1_000_000
        )

        return {
            "content": data["choices"][0]["message"]["content"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens", 0),
            "cost_usd": cost_usd,
            "finish_reason": data["choices"][0].get("finish_reason"),
        }

    elif run.provider == "anthropic":
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")

        async with httpx.AsyncClient(timeout=60.0) as client:
            body = {
                "model": run.model_name,
                "max_tokens": run.hyperparameters.get("max_tokens", 1024),
                "messages": [{"role": "user", "content": prompt}],
            }
            if run.system_prompt:
                body["system"] = run.system_prompt

            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        return {
            "content": data["content"][0]["text"],
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            "cost_usd": None,  # Add Anthropic pricing table
            "finish_reason": data.get("stop_reason"),
        }

    else:
        raise ValueError(f"Unsupported provider: {run.provider}")
