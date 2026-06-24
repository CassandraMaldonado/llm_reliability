"""
app/api/v1/router.py + endpoint definitions

Full REST API for the MANGOS platform.

API Design Principles:
- RESTful resource naming (/experiments, /runs, /traces)
- Consistent error format: {error, message, details}
- Pagination on all list endpoints (cursor-based for large datasets)
- Async everywhere — no blocking database calls in route handlers
- All business logic lives in services/, not here
- Route handlers: validate → call service → return schema
"""
from fastapi import APIRouter
from app.api.v1.endpoints import (
    auth,
    experiments,
    runs,
    traces,
    evaluations,
    datasets,
    rag,
    monitoring,
    alerts,
    drift,
    organizations,
)

api_router = APIRouter()

# Mount all sub-routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(organizations.router, prefix="/organizations", tags=["Organizations"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
api_router.include_router(runs.router, prefix="/runs", tags=["Experiment Runs"])
api_router.include_router(traces.router, prefix="/traces", tags=["LLM Traces"])
api_router.include_router(evaluations.router, prefix="/evaluations", tags=["Evaluations"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(rag.router, prefix="/rag", tags=["RAG Evaluation"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
api_router.include_router(drift.router, prefix="/drift", tags=["Drift Detection"])
