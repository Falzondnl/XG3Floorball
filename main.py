"""
XG3 Floorball Microservice
==========================
Port: 8037 (local dev) / injected via $PORT (Railway)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import PORT, SERVICE_NAME, SERVICE_VERSION
from ml.predictor import FloorballPredictor

# ---------------------------------------------------------------------------
# Sentry error monitoring — set SENTRY_DSN env var in Railway to activate
# ---------------------------------------------------------------------------
import os as _os_sentry
_SENTRY_DSN = _os_sentry.getenv("SENTRY_DSN", "")
if _SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            integrations=[
                StarletteIntegration(transaction_style="endpoint"),
                FastApiIntegration(transaction_style="endpoint"),
            ],
            traces_sample_rate=float(_os_sentry.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.05")),
            environment=_os_sentry.getenv("ENVIRONMENT", "production"),
        )
        print(f"[Sentry] Initialized for {_os_sentry.getenv('RAILWAY_SERVICE_NAME', 'unknown')}")
    except ImportError:
        pass  # sentry-sdk not installed — non-fatal

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


# ---------------------------------------------------------------------------
# Lifespan: load models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting %s v%s...", SERVICE_NAME, SERVICE_VERSION)

    predictor = FloorballPredictor()
    try:
        predictor.load()
    except Exception as exc:
        logger.error("Predictor load failed: %s — service will start in degraded mode", exc)

    app.state.predictor = predictor
    logger.info(
        "Startup complete. Predictor ready=%s, regimes=%s",
        predictor.ready,
        predictor.available_regimes(),
    )

    yield

    logger.info("Shutting down %s.", SERVICE_NAME)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="XG3 Floorball Microservice",
    description="Tier-1 Floorball prediction + pricing (Win/Draw/Win, Handicap, Totals)",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

from api.routes.health import router as health_router
from api.routes.live import router as live_router
from api.routes.matches import router as matches_router
from api.routes.admin import router as admin_router
from api.routes.fixtures import router as fixtures_router
from api.routes.settlement import router as settlement_router


app.include_router(health_router)
app.include_router(live_router)
app.include_router(matches_router)
app.include_router(admin_router)
app.include_router(fixtures_router)
app.include_router(settlement_router)



@app.get("/")
async def root() -> dict:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )
