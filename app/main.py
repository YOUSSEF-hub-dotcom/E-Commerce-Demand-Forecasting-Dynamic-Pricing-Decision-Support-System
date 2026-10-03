"""Application factory: FastAPI app wiring, middleware, lifecycle."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.api import explain, health, prediction, pricing
from app.config import Settings
from app.db.database import InferenceStore
from app.exception_handlers import (
    app_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from app.errors import AppError
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.payload_limit import PayloadSizeLimitMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_id import RequestIdMiddleware
from app.services.state import AppState

logger = logging.getLogger("deploy.app")

APP_VERSION = "0.6.0"


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def create_app(
    settings: Settings | None = None,
    *,
    state: AppState | None = None,
    load_on_startup: bool = True,
) -> FastAPI:
    settings = settings or Settings()
    _configure_logging(settings)

    app = FastAPI(
        title="E-Commerce Demand Forecasting & Dynamic Pricing API",
        version=APP_VERSION,
        description=(
            "Business-level demand forecast, price recommendation and SHAP "
            "explainability over the frozen Production demand model "
            "(models:/demand_forecasting_final/2)."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.settings = settings
    app.state.inference_store = InferenceStore(
        settings.database_url, timeout_s=settings.persistence_timeout_s
    )
    app.state.app_state = state or AppState(settings)
    store: InferenceStore = app.state.inference_store

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware, settings=settings)
    app.add_middleware(RequestIdMiddleware, settings=settings)
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(PayloadSizeLimitMiddleware, settings=settings)

    app.include_router(health.router)
    app.include_router(prediction.router)
    app.include_router(pricing.router)
    app.include_router(explain.router)

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime_state: AppState = app.state.app_state
        # Inference DB: attempted but never fatal. On failure requests report
        # ``inference_logged=false`` instead of failing.
        try:
            store.create_schema()
            store.ping()
            runtime_state.mark_component("database", "ok")
        except Exception:
            logger.exception(
                "inference database unavailable at startup; requests will report "
                "inference_logged=false"
            )
            runtime_state.mark_component(
                "database",
                "not_ready",
                "inference log database unreachable; inference_logged=false",
            )
        # Heavy startup loads happen inside load_all (each step recorded).
        if load_on_startup and not runtime_state.loaded:
            try:
                runtime_state.load_all()
            except Exception:
                logger.exception("startup load failed; /ready will report not_ready")
        yield
        if store.engine is not None:
            store.engine.dispose()

    app.router.lifespan_context = lifespan
    return app


app = create_app()