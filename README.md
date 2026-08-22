# LLM Reliability Platform

An open-source, production garde platform for LLM evaluation, observability and reliability.

---

## What it does

| Capability | Details |
|---|---|
| **Experiment tracking** | A/B test prompts, models, and providers. Compare runs side-by-side with automatic winner detection. |
| **Automated evaluation** | 7 metrics (answer relevance, faithfulness, hallucination, toxicity, semantic similarity, context relevance) via LLM and local sentence transformers. |
| **RAG pipeline analysis** | Retrieval precision/recall, groundedness, answer correctness. Compare embedding models, chunk sizes and retrieval strategies head-to-head. |
| **Production tracing** | Instrument any LLM call in 3 lines via the Python SDK. Captures latency, cost, token counts and user feedback. |
| **Statistical drift detection** | KS test and Z-score anomaly detection on latency, cost, and quality metrics. Alerts fire before users notice. |
| **Alert engine** | Configurable rules with webhook dispatch (Slack, PagerDuty, OpsGenie, anything). |
| **Dashboard** | UI with live metric gauges, time-series charts and experiment comparison tables. |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         MANGOS Stack                         │
├──────────────┬───────────────┬──────────────┬───────────-────┤
│  React + TS  │  FastAPI      │  Celery      │  PostgreSQL    │
│  Dashboard   │  REST API     │  Workers     │  + JSONB       │
│  Port 5173   │  Port 8000    │  Beat (5min) │                │
├──────────────┴───────────────┴──────────────┴─────────────-──┤
│              Redis (cache + Celery broker)                   │
└──────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **Repository pattern** — services never touch SQLAlchemy directly; fully mockable for testing
- **202 for eval jobs** — evaluations take minutes; HTTP must return immediately
- **Concurrent metrics** — `asyncio.gather()` runs all metrics in parallel; wall time = slowest metric
- **KS test for drift** — non-parametric, handles non-normal distributions, industry standard
- **UUID primary keys** — distributed-safe, no sequential scan risk in future sharding

---

## Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/yourhandle/mangos.git
cd mangos
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY and SECRET_KEY
```

### 2. Start the stack

```bash
docker compose up -d
```

This starts: PostgreSQL 16, Redis 7, FastAPI API server, Celery worker, Celery Beat scheduler, Flower (Celery UI at :5555), React frontend, Nginx.

### 3. Run migrations

```bash
docker compose exec api alembic upgrade head
```

### 4. Open the dashboard

```
http://localhost:80
```

API docs (Swagger UI):
```
http://localhost:8000/docs
```

---

## Instrument your app (SDK)

```bash
pip install mangos-sdk
```

```python
import mangos_sdk as mangos

# Once at startup
mangos.init(api_key="mg_...", base_url="https://your-mangos.com")

# Wrap any LLM call
async with mangos.trace(model="gpt-4o", provider="openai", input_text=prompt) as t:
    response = await openai_client.chat.completions.create(...)
    t.set_output(response.choices[0].message.content)
    t.set_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)

# Submit user feedback
await mangos.get_client().submit_feedback(trace_id, score=1.0, label="thumbs_up")
```

The SDK is **fire-and-forget** — trace submission happens in a background task. If MANGOS is unreachable, your app keeps running (errors are logged, never raised).

---

## Evaluation metrics

| Metric | Method | Cost |
|---|---|---|
| Semantic similarity | Local `sentence-transformers` (MiniLM) |
| Answer relevance | GPT-4o-mini judge | 
| Faithfulness | GPT-4o-mini judge |
| Hallucination | GPT-4o-mini judge |
| Toxicity | GPT-4o-mini judge |
| Context relevance | GPT-4o-mini judge |
| RAG groundedness | GPT-4o-mini judge |

All metrics run **concurrently** — evaluating 7 metrics takes the same time as evaluating 1.

---

## Drift detection

Runs every 5 minutes via Celery Beat. Three detection strategies:

- **KS test** — Kolmogorov-Smirnov two-sample test on metric distributions. Flags when the distribution shifts (not just the mean). `p < 0.05` triggers warning.
- **Z-score anomaly** — Flags individual values > 2.5 standard deviations from the 24-hour baseline.
- **Threshold breach** — Hard limits (e.g. `latency_ms > 10,000` = critical, `hallucination_score > 0.6` = critical).

Drift generates an Alert record and fires configured notification channels.

---

## Development

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Run tests
pytest tests/unit/           # Fast, no DB needed
pytest tests/integration/    # Requires test DB (see conftest.py)

# Lint + type check
ruff check app/
mypy app/

# Run API locally (no Docker)
uvicorn app.main:app --reload
```

---

## Environment variables

See [`.env.example`](.env.example) for the full list. Required to get started:

| Variable | Description |
|---|---|
| `SECRET_KEY` | 64-char random string for JWT signing |
| `DATABASE_URL` | PostgreSQL async URL |
| `REDIS_URL` | Redis URL |
| `OPENAI_API_KEY` | Used by LLM-as-judge evaluations |

Optional (enables more features):
- `ANTHROPIC_API_KEY` — run evaluations against Claude models
- `SENTRY_DSN` — error tracking
- `SENDGRID_API_KEY` — email alerts


