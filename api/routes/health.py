"""
Health endpoints — mandatory XG3 pattern.
GET /health        → {"status": "ok", "service": "floorball"}
GET /health/ready  → {"status": "ok", "ready": true}
GET /health/live   → {"status": "ok"}
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from config import SERVICE_NAME, SERVICE_VERSION

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    predictor = getattr(request.app.state, "predictor", None)
    ready = predictor is not None and predictor.ready
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "ready": ready,
    }


@router.get("/health/ready")
async def health_ready(request: Request) -> dict:
    predictor = getattr(request.app.state, "predictor", None)
    ready = predictor is not None and predictor.ready

    regimes = []
    if predictor is not None:
        regimes = predictor.available_regimes()

    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "service": SERVICE_NAME,
        "regimes_loaded": regimes,
    }


@router.get("/health/live")
async def health_live() -> dict:
    return {"status": "ok"}
