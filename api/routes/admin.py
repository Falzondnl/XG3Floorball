"""
Admin endpoints — model status, training metadata.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Request

from config import MODELS_DIR, SERVICE_NAME, SERVICE_VERSION

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/floorball/admin")


@router.get("/status")
async def status(request: Request) -> dict:
    """Overall service and model status."""
    predictor = getattr(request.app.state, "predictor", None)

    model_files: dict[str, dict] = {}
    for regime_dir in MODELS_DIR.iterdir() if MODELS_DIR.exists() else []:
        if not regime_dir.is_dir():
            continue
        regime = regime_dir.name
        model_files[regime] = {}
        for fname in ["ensemble.pkl", "calibrator.pkl", "extractor.pkl"]:
            fpath = regime_dir / fname
            model_files[regime][fname] = {
                "exists": fpath.exists(),
                "size_bytes": fpath.stat().st_size if fpath.exists() else 0,
            }

    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "predictor_ready": predictor is not None and predictor.ready,
        "regimes_loaded": predictor.available_regimes() if predictor else [],
        "model_files": model_files,
    }


@router.get("/model-health")
async def model_health(request: Request) -> dict:
    """Detailed model health check."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return {"status": "not_loaded"}

    return {
        "status": "loaded" if predictor.ready else "not_ready",
        "regimes": predictor.available_regimes(),
        "ensemble_types": {
            r: getattr(ens, "mode", "unknown")
            for r, ens in predictor._ensembles.items()
        },
        "calibrators_loaded": list(predictor._calibrators.keys()),
    }
