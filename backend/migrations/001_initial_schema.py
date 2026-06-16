"""
MANGOS — Initial Database Schema
Alembic migration: 001_initial_schema

Design notes:
- All tables use UUID primary keys (not serial ints) for distributed safety
- organization_id on every tenant table enables future multi-tenancy
- JSONB columns used for flexible metadata / parameters without sacrificing queryability
- created_at / updated_at on every table (audit trail)
- Soft deletes via deleted_at (never hard-delete experiment data)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import uuid

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ─────────────────────────────────────────
    # ORGANIZATIONS  (multi-tenancy root)
    # ─────────────────────────────────────────
    op.create_table(
        "organizations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False, unique=True),
        sa.Column("plan", sa.String(50), nullable=False, server_default="free"),
        sa.Column("settings", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ─────────────────────────────────────────
    # USERS
    # ─────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("role", sa.String(50), nullable=False, server_default="member"),  # owner, admin, member, viewer
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )

    # ─────────────────────────────────────────
    # API KEYS  (programmatic access)
    # ─────────────────────────────────────────
    op.create_table(
        "api_keys",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("created_by_user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("key_prefix", sa.String(12), nullable=False),       # shown to user e.g. "mg_live_abc1"
        sa.Column("hashed_key", sa.String(255), nullable=False),       # SHA-256 hash, never store raw
        sa.Column("scopes", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ─────────────────────────────────────────
    # PROJECTS  (logical grouping of experiments)
    # ─────────────────────────────────────────
    op.create_table(
        "projects",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("tags", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_by_user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ─────────────────────────────────────────
    # EXPERIMENTS  (a named study comparing prompts/models)
    # ─────────────────────────────────────────
    op.create_table(
        "experiments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("project_id", UUID(as_uuid=True), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),  # active, archived
        sa.Column("baseline_run_id", UUID(as_uuid=True), nullable=True),   # FK to experiment_runs, set later
        sa.Column("tags", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_by_user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ─────────────────────────────────────────
    # EXPERIMENT RUNS  (one run = one full eval pass with a config)
    # ─────────────────────────────────────────
    op.create_table(
        "experiment_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("experiment_id", UUID(as_uuid=True), sa.ForeignKey("experiments.id"), nullable=False),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        # Model configuration
        sa.Column("provider", sa.String(50), nullable=False),           # openai, anthropic, gemini, huggingface
        sa.Column("model_name", sa.String(255), nullable=False),        # gpt-4o, claude-3-5-sonnet, etc.
        sa.Column("model_version", sa.String(100), nullable=True),
        # Prompt configuration
        sa.Column("system_prompt", sa.Text, nullable=True),
        sa.Column("prompt_template", sa.Text, nullable=True),
        sa.Column("prompt_version", sa.String(50), nullable=True),
        # Hyperparameters stored as JSONB for flexibility
        sa.Column("hyperparameters", JSONB, nullable=False, server_default="{}"),
        # e.g. {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1024}
        # Status
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        # pending, running, completed, failed, cancelled
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        # Aggregate metrics (denormalized for fast dashboard queries)
        sa.Column("total_samples", sa.Integer, nullable=True),
        sa.Column("completed_samples", sa.Integer, nullable=True),
        sa.Column("failed_samples", sa.Integer, nullable=True),
        sa.Column("avg_latency_ms", sa.Float, nullable=True),
        sa.Column("p50_latency_ms", sa.Float, nullable=True),
        sa.Column("p95_latency_ms", sa.Float, nullable=True),
        sa.Column("p99_latency_ms", sa.Float, nullable=True),
        sa.Column("total_cost_usd", sa.Float, nullable=True),
        sa.Column("avg_cost_usd", sa.Float, nullable=True),
        sa.Column("total_prompt_tokens", sa.Integer, nullable=True),
        sa.Column("total_completion_tokens", sa.Integer, nullable=True),
        sa.Column("avg_answer_relevance", sa.Float, nullable=True),
        sa.Column("avg_faithfulness", sa.Float, nullable=True),
        sa.Column("avg_hallucination_score", sa.Float, nullable=True),
        sa.Column("avg_semantic_similarity", sa.Float, nullable=True),
        sa.Column("avg_toxicity_score", sa.Float, nullable=True),
        sa.Column("tags", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by_user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )

    # ─────────────────────────────────────────
    # LLM TRACES  (individual LLM calls)
    # Core of the observability system. One row per LLM call.
    # ─────────────────────────────────────────
    op.create_table(
        "llm_traces",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("run_id", UUID(as_uuid=True), sa.ForeignKey("experiment_runs.id"), nullable=True),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        # Source: was this from an experiment, or from production SDK instrumentation?
        sa.Column("source", sa.String(50), nullable=False, server_default="experiment"),  # experiment, production
        sa.Column("trace_group_id", UUID(as_uuid=True), nullable=True),  # group related multi-turn calls
        sa.Column("span_index", sa.Integer, nullable=True),              # position within a trace group
        # Input
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("system_prompt", sa.Text, nullable=True),
        sa.Column("user_prompt", sa.Text, nullable=False),
        sa.Column("messages", JSONB, nullable=True),                     # full message array for multi-turn
        # Output
        sa.Column("completion", sa.Text, nullable=True),
        sa.Column("finish_reason", sa.String(50), nullable=True),        # stop, length, content_filter
        # Tokens & cost
        sa.Column("prompt_tokens", sa.Integer, nullable=True),
        sa.Column("completion_tokens", sa.Integer, nullable=True),
        sa.Column("total_tokens", sa.Integer, nullable=True),
        sa.Column("cost_usd", sa.Float, nullable=True),
        # Timing
        sa.Column("latency_ms", sa.Float, nullable=True),
        sa.Column("time_to_first_token_ms", sa.Float, nullable=True),
        # Hyperparameters at call time
        sa.Column("hyperparameters", JSONB, nullable=False, server_default="{}"),
        # Ground truth (for eval)
        sa.Column("expected_output", sa.Text, nullable=True),
        sa.Column("context_documents", JSONB, nullable=True),           # RAG context passed to model
        # Status
        sa.Column("status", sa.String(50), nullable=False, server_default="success"),  # success, error, timeout
        sa.Column("error_code", sa.String(100), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        # User feedback
        sa.Column("user_feedback_score", sa.Float, nullable=True),       # 1-5 thumbs up/down normalized
        sa.Column("user_feedback_text", sa.Text, nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ─────────────────────────────────────────
    # EVALUATION RESULTS  (per-trace metric scores)
    # ─────────────────────────────────────────
    op.create_table(
        "evaluation_results",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("trace_id", UUID(as_uuid=True), sa.ForeignKey("llm_traces.id"), nullable=False),
        sa.Column("run_id", UUID(as_uuid=True), sa.ForeignKey("experiment_runs.id"), nullable=True),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("metric_name", sa.String(100), nullable=False),
        # e.g. answer_relevance, faithfulness, hallucination_score, semantic_similarity, toxicity
        sa.Column("metric_version", sa.String(50), nullable=True),      # version of evaluator used
        sa.Column("score", sa.Float, nullable=False),                    # normalized 0.0 – 1.0
        sa.Column("raw_score", sa.Float, nullable=True),                 # provider-native score
        sa.Column("passed", sa.Boolean, nullable=True),                  # did it meet threshold?
        sa.Column("threshold", sa.Float, nullable=True),                 # threshold used
        sa.Column("reasoning", sa.Text, nullable=True),                  # LLM-based evaluator explanation
        sa.Column("evaluator_model", sa.String(255), nullable=True),     # which model scored this
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ─────────────────────────────────────────
    # DATASETS  (reusable eval question sets)
    # ─────────────────────────────────────────
    op.create_table(
        "datasets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("dataset_type", sa.String(50), nullable=False, server_default="qa"),  # qa, rag, chat
        sa.Column("row_count", sa.Integer, nullable=True),
        sa.Column("schema_definition", JSONB, nullable=False, server_default="{}"),
        sa.Column("tags", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_by_user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "dataset_rows",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("dataset_id", UUID(as_uuid=True), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("row_index", sa.Integer, nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("expected_answer", sa.Text, nullable=True),
        sa.Column("context_documents", JSONB, nullable=True),  # for RAG eval: gold context
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ─────────────────────────────────────────
    # RAG EVALUATIONS
    # ─────────────────────────────────────────
    op.create_table(
        "rag_evaluations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("run_id", UUID(as_uuid=True), sa.ForeignKey("experiment_runs.id"), nullable=False),
        sa.Column("trace_id", UUID(as_uuid=True), sa.ForeignKey("llm_traces.id"), nullable=False),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        # RAG configuration
        sa.Column("embedding_model", sa.String(255), nullable=False),   # text-embedding-3-large, bge-large, etc.
        sa.Column("chunk_size", sa.Integer, nullable=True),
        sa.Column("chunk_overlap", sa.Integer, nullable=True),
        sa.Column("retrieval_strategy", sa.String(100), nullable=True), # dense, sparse, hybrid, mmr
        sa.Column("top_k", sa.Integer, nullable=True),
        # Retrieved context
        sa.Column("retrieved_chunks", JSONB, nullable=True),            # list of {text, score, source}
        sa.Column("gold_context", JSONB, nullable=True),                # expected context from dataset
        # Scores
        sa.Column("retrieval_precision", sa.Float, nullable=True),
        sa.Column("retrieval_recall", sa.Float, nullable=True),
        sa.Column("context_relevance", sa.Float, nullable=True),
        sa.Column("groundedness", sa.Float, nullable=True),
        sa.Column("answer_correctness", sa.Float, nullable=True),
        sa.Column("context_utilization", sa.Float, nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ─────────────────────────────────────────
    # MONITORING METRICS  (time-series production metrics)
    # ─────────────────────────────────────────
    op.create_table(
        "monitoring_metrics",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("metric_name", sa.String(100), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_size_minutes", sa.Integer, nullable=False),
        sa.Column("sample_count", sa.Integer, nullable=False),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ─────────────────────────────────────────
    # ALERT RULES
    # ─────────────────────────────────────────
    op.create_table(
        "alert_rules",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("metric_name", sa.String(100), nullable=False),
        sa.Column("condition", sa.String(50), nullable=False),           # gt, lt, gte, lte, anomaly
        sa.Column("threshold", sa.Float, nullable=True),
        sa.Column("severity", sa.String(50), nullable=False, server_default="warning"),  # info, warning, critical
        sa.Column("model_filter", sa.String(255), nullable=True),        # null = all models
        sa.Column("evaluation_window_minutes", sa.Integer, nullable=False, server_default="60"),
        sa.Column("cooldown_minutes", sa.Integer, nullable=False, server_default="30"),
        sa.Column("notification_channels", JSONB, nullable=False, server_default="[]"),
        # [{"type": "email", "address": "..."}, {"type": "webhook", "url": "..."}]
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("last_triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by_user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )

    # ─────────────────────────────────────────
    # ALERTS  (triggered alert instances)
    # ─────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("rule_id", UUID(as_uuid=True), sa.ForeignKey("alert_rules.id"), nullable=False),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="open"),  # open, acknowledged, resolved
        sa.Column("severity", sa.String(50), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("metric_value", sa.Float, nullable=True),
        sa.Column("threshold_value", sa.Float, nullable=True),
        sa.Column("context", JSONB, nullable=False, server_default="{}"),
        sa.Column("acknowledged_by_user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("triggered_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ─────────────────────────────────────────
    # DRIFT REPORTS
    # ─────────────────────────────────────────
    op.create_table(
        "drift_reports",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("metric_name", sa.String(100), nullable=False),
        sa.Column("drift_type", sa.String(50), nullable=False),         # statistical, threshold, anomaly
        sa.Column("baseline_window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("baseline_window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("baseline_mean", sa.Float, nullable=True),
        sa.Column("current_mean", sa.Float, nullable=True),
        sa.Column("baseline_std", sa.Float, nullable=True),
        sa.Column("current_std", sa.Float, nullable=True),
        sa.Column("drift_score", sa.Float, nullable=True),              # KS statistic or Z-score
        sa.Column("p_value", sa.Float, nullable=True),
        sa.Column("is_significant", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("alert_id", UUID(as_uuid=True), sa.ForeignKey("alerts.id"), nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ─────────────────────────────────────────
    # INDEXES  (critical for dashboard query performance)
    # ─────────────────────────────────────────

    # Traces — most queried table, needs aggressive indexing
    op.create_index("ix_llm_traces_org_created", "llm_traces", ["organization_id", "created_at"])
    op.create_index("ix_llm_traces_org_model", "llm_traces", ["organization_id", "model_name"])
    op.create_index("ix_llm_traces_run_id", "llm_traces", ["run_id"])
    op.create_index("ix_llm_traces_trace_group", "llm_traces", ["trace_group_id"])
    op.create_index("ix_llm_traces_source_org", "llm_traces", ["source", "organization_id"])

    # Eval results — commonly queried by metric name per run
    op.create_index("ix_eval_results_trace", "evaluation_results", ["trace_id"])
    op.create_index("ix_eval_results_run_metric", "evaluation_results", ["run_id", "metric_name"])

    # Monitoring metrics — time-series queries
    op.create_index("ix_monitoring_org_model_metric_window", "monitoring_metrics",
                    ["organization_id", "model_name", "metric_name", "window_start"])

    # Experiments
    op.create_index("ix_experiments_project", "experiments", ["project_id"])
    op.create_index("ix_experiment_runs_experiment", "experiment_runs", ["experiment_id"])
    op.create_index("ix_experiment_runs_status", "experiment_runs", ["status", "organization_id"])

    # Alerts
    op.create_index("ix_alerts_org_status", "alerts", ["organization_id", "status"])
    op.create_index("ix_alerts_rule_triggered", "alerts", ["rule_id", "triggered_at"])

    # Dataset rows
    op.create_index("ix_dataset_rows_dataset", "dataset_rows", ["dataset_id", "row_index"])

    # RAG evaluations
    op.create_index("ix_rag_evals_run", "rag_evaluations", ["run_id"])

    # Drift reports
    op.create_index("ix_drift_reports_org_model_metric", "drift_reports",
                    ["organization_id", "model_name", "metric_name", "detected_at"])


def downgrade() -> None:
    # Drop in reverse dependency order
    tables = [
        "drift_reports", "alerts", "alert_rules", "monitoring_metrics",
        "rag_evaluations", "evaluation_results", "llm_traces",
        "dataset_rows", "datasets", "experiment_runs", "experiments",
        "projects", "api_keys", "users", "organizations",
    ]
    for table in tables:
        op.drop_table(table)
