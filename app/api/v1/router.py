# Full REST API.

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

# all sub-routers.
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
