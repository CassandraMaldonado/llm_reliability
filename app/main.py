"""
app/main.py

FastAPI application factory and root configuration.

Enterprise patterns:
- Application factory pattern (not global app object) enables testing
- Lifespan context manager for startup/shutdown (replaces deprecated on_event)
- Structured logging (JSON in production, human-readable in dev)
- Request ID middleware for distributed tracing
- Health check endpoint (required for k8s liveness/readiness probes)
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1.router import api_router

# ── Structured logging setup ────────────────────────────────────────────────
# structlog gives us JSON logs in production (parseable by Datadog/CloudWatch)
# and colored dev logs locally. Same code path, different renderers.
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
        if settings.ENVIRONMENT == "production"
        else structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Code before yield: startup.
    Code after yield: shutdown.

    Replaces deprecated @app.on_event("startup") pattern.
    """
    log.info("MANGOS starting up", environment=settings.ENVIRONMENT)

    # In development, auto-create tables (migrations handle prod)
    if settings.ENVIRONMENT == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("Database tables verified")

    yield

    # Shutdown: close DB connection pool
    await engine.dispose()
    log.info("MANGOS shut down cleanly")


def create_app() -> FastAPI:
    """
    Application factory.
    Returns a configured FastAPI instance.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Enterprise-grade LLM evaluation, monitoring, and reliability platform. "
            "Track experiments, detect drift, measure quality, and prevent hallucinations."
        ),
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID Middleware ────────────────────────────────────────────────
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """
        Inject X-Request-ID header on every request/response.
        Critical for distributed tracing — correlate logs across services.
        """
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.monotonic()
        response: Response = await call_next(request)
        duration_ms = (time.monotonic() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-MS"] = f"{duration_ms:.2f}"

        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=f"{duration_ms:.2f}",
            request_id=request_id,
        )
        return response

    # ── Global Exception Handler ─────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            request_id=getattr(request.state, "request_id", "unknown"),
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", "unknown"),
            }
        )

    # ── Routes ───────────────────────────────────────────────────────────────
    app.include_router(api_router, prefix=settings.API_PREFIX)

    # ── Health Check ─────────────────────────────────────────────────────────
    @app.get("/health", tags=["system"])
    async def health_check():
        """
        Kubernetes liveness/readiness probe endpoint.
        Returns 200 if the service is healthy.
        Enterprise: Add DB ping check for readiness probe.
        """
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }

    @app.get("/", tags=["system"])
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
        }

    return app


app = create_app()
