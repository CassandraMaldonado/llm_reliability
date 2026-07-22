# Experiment and run schemas.

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None
    project_id: Optional[uuid.UUID] = None
    tags: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)


class ExperimentUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    description: Optional[str]
    status: str
    tags: List[str]
    config: Dict[str, Any]
    run_count: int
    created_at: datetime
    updated_at: Optional[datetime]


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT RUNS
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRunCreate(BaseModel):
    """
    Create a new experiment run.
    
    Hyperparameters are stored in JSONB — this is intentional.
    LLM hyperparameters evolve (new models add new params) and using JSONB
    avoids schema migrations every time a new provider appears.
    """
    experiment_id: uuid.UUID
    dataset_id: Optional[uuid.UUID] = None
    model_name: str = Field(min_length=1, max_length=100)
    provider: str = Field(min_length=1, max_length=50)  # openai, anthropic, gemini
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="temperature, top_p, max_tokens, etc."
    )
    evaluation_metrics: List[str] = Field(
        default_factory=lambda: ["answer_relevance", "faithfulness", "hallucination"],
        description="Which metrics to compute for this run"
    )
    max_samples: Optional[int] = Field(default=None, ge=1, le=10000)


class ExperimentRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    experiment_id: uuid.UUID
    model_name: str
    provider: str
    status: str  # pending, running, completed, failed
    system_prompt: Optional[str]
    hyperparameters: Dict[str, Any]
    # Aggregated metrics (populated after run completes)
    avg_latency_ms: Optional[float]
    avg_cost_usd: Optional[float]
    total_cost_usd: Optional[float]
    avg_hallucination_score: Optional[float]
    avg_answer_relevance: Optional[float]
    avg_faithfulness: Optional[float]
    avg_semantic_similarity: Optional[float]
    sample_count: int
    error_count: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime


class RunCompareRequest(BaseModel):
    run_ids: List[uuid.UUID] = Field(min_length=2, max_length=10)
    metrics: List[str] = Field(
        default_factory=lambda: [
            "avg_latency_ms", "avg_cost_usd",
            "avg_hallucination_score", "avg_answer_relevance", "avg_faithfulness"
        ]
    )


class RunMetricComparison(BaseModel):
    metric: str
    values: Dict[str, Optional[float]]  # run_id -> value
    winner_run_id: Optional[str]        # run with best value (None if tie)
    delta_pct: Optional[float]          # % improvement of best vs worst


class RunCompareResponse(BaseModel):
    run_ids: List[str]
    runs: List[ExperimentRunResponse]
    metric_comparisons: List[RunMetricComparison]
    overall_winner_run_id: Optional[str]  # run that wins most metrics
    summary: str                           # human-readable summary
