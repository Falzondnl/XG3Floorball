"""
Floorball Predictor
===================
Loads trained ensemble artefacts and serves P(home_win) predictions.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import ELO_DEFAULT, MODELS_DIR, REGIME_MEN, REGIME_WOMEN
from ml.calibrator import FloorballCalibrator
from ml.features import FEATURE_COLUMNS, FloorballFeatureExtractor

logger = logging.getLogger(__name__)

_ELO_STATE_PATH = MODELS_DIR / "r0" / "elo_state.json"


def _load_team_elo_state() -> dict[int, float]:
    """Load pre-computed team ELO from models/r0/elo_state.json.

    Returns dict mapping team_id (int) -> float ELO rating.
    Falls back to empty dict if file not present (caller defaults to ELO_DEFAULT).
    """
    if not _ELO_STATE_PATH.exists():
        logger.warning("elo_state.json not found at %s — team ELO lookup unavailable", _ELO_STATE_PATH)
        return {}
    try:
        with open(_ELO_STATE_PATH) as fh:
            data = json.load(fh)
        raw = data.get("team_elos", {})
        result = {int(k): float(v) for k, v in raw.items()}
        logger.info("Floorball ELO state loaded: %d teams from %s", len(result), _ELO_STATE_PATH)
        return result
    except Exception as exc:
        logger.warning("Failed to load elo_state.json: %s — team ELO lookup disabled", exc)
        return {}


# Module-level singleton — loaded once at import time (same process lifetime as predictor)
_TEAM_ELO: dict[int, float] = _load_team_elo_state()


@dataclass
class PredictionInput:
    home_team_id: int
    away_team_id: int
    tournament_name: str
    is_women: bool = False
    # Optional override fields — when caller has pre-computed context
    home_elo: Optional[float] = None
    away_elo: Optional[float] = None
    form_pts_home: Optional[float] = None
    form_pts_away: Optional[float] = None
    form_gd_home: Optional[float] = None
    form_gd_away: Optional[float] = None
    goals_home_avg5: Optional[float] = None
    goals_away_avg5: Optional[float] = None
    h2h_home_win_rate: Optional[float] = None
    competition_level: Optional[int] = None
    is_playoff: Optional[int] = None


@dataclass
class PredictionResult:
    home_win_prob: float
    away_win_prob: float
    draw_prob: float          # residual for international draw scenarios
    regime: str
    model_mode: str
    raw_prob: float
    calibrated: bool = True
    confidence: str = "medium"
    features_used: dict = field(default_factory=dict)


class FloorballPredictor:
    """
    Thread-safe prediction engine.
    Loaded once at startup; reused across requests.
    """

    def __init__(self) -> None:
        self._ensembles: dict[str, object] = {}
        self._calibrators: dict[str, FloorballCalibrator] = {}
        self._ready = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all available regimes."""
        for regime in [REGIME_MEN, REGIME_WOMEN]:
            regime_dir = MODELS_DIR / regime
            ens_path = regime_dir / "ensemble.pkl"
            cal_path = regime_dir / "calibrator.pkl"

            if not ens_path.exists():
                logger.warning("[%s] ensemble.pkl not found — regime unavailable", regime)
                continue

            try:
                with open(ens_path, "rb") as fh:
                    self._ensembles[regime] = pickle.load(fh)
                logger.info("[%s] Ensemble loaded from %s", regime, ens_path)
            except Exception as exc:
                logger.error("[%s] Failed to load ensemble: %s", regime, exc)
                continue

            if cal_path.exists():
                try:
                    self._calibrators[regime] = FloorballCalibrator.load(cal_path)
                    logger.info("[%s] Calibrator loaded", regime)
                except Exception as exc:
                    logger.warning("[%s] Calibrator load failed: %s", regime, exc)
            else:
                logger.warning("[%s] calibrator.pkl not found — raw probs used", regime)

        self._ready = bool(self._ensembles)
        logger.info(
            "Predictor ready=%s — loaded regimes: %s",
            self._ready,
            list(self._ensembles.keys()),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    def available_regimes(self) -> list[str]:
        return list(self._ensembles.keys())

    def predict(self, inp: PredictionInput) -> PredictionResult:
        regime = REGIME_WOMEN if inp.is_women else REGIME_MEN

        # Fall back to men's model if women's not available
        if regime not in self._ensembles:
            if REGIME_MEN in self._ensembles:
                logger.debug("Women model unavailable, falling back to men's model")
                regime = REGIME_MEN
            else:
                raise RuntimeError("No prediction models loaded")

        ensemble = self._ensembles[regime]

        # Build feature row
        from config import COMPETITION_LEVELS, WOMEN_KEYWORDS
        tn = inp.tournament_name or ""
        comp_level = inp.competition_level
        if comp_level is None:
            tn_lower = tn.lower()
            comp_level = 1
            for key, lv in COMPETITION_LEVELS.items():
                if key in tn_lower:
                    comp_level = lv
                    break

        is_playoff = inp.is_playoff
        if is_playoff is None:
            is_playoff = 1 if any(
                kw in tn.lower() for kw in ("playoff", "final")
            ) else 0

        is_women_flag = 1 if inp.is_women else 0

        # Use provided values, then elo_state.json lookup, then ELO default
        if inp.home_elo is not None:
            home_elo = inp.home_elo
        else:
            home_elo = _TEAM_ELO.get(inp.home_team_id, ELO_DEFAULT)
            if home_elo == ELO_DEFAULT:
                logger.debug("home_team_id=%d not in ELO state — using default %.0f", inp.home_team_id, ELO_DEFAULT)

        if inp.away_elo is not None:
            away_elo = inp.away_elo
        else:
            away_elo = _TEAM_ELO.get(inp.away_team_id, ELO_DEFAULT)
            if away_elo == ELO_DEFAULT:
                logger.debug("away_team_id=%d not in ELO state — using default %.0f", inp.away_team_id, ELO_DEFAULT)

        feat = {
            "elo_home_pre": home_elo,
            "elo_away_pre": away_elo,
            "elo_diff": home_elo - away_elo,
            "form_pts_home_last5": inp.form_pts_home if inp.form_pts_home is not None else 1.5,
            "form_pts_away_last5": inp.form_pts_away if inp.form_pts_away is not None else 1.5,
            "form_gd_home_last5": inp.form_gd_home if inp.form_gd_home is not None else 0.0,
            "form_gd_away_last5": inp.form_gd_away if inp.form_gd_away is not None else 0.0,
            "goals_scored_home_avg5": inp.goals_home_avg5 if inp.goals_home_avg5 is not None else 5.5,
            "goals_scored_away_avg5": inp.goals_away_avg5 if inp.goals_away_avg5 is not None else 5.5,
            "h2h_home_win_rate": inp.h2h_home_win_rate if inp.h2h_home_win_rate is not None else 0.5,
            "competition_level": comp_level,
            "is_playoff": is_playoff,
            "is_women": is_women_flag,
        }

        X = pd.DataFrame([feat])[FEATURE_COLUMNS]
        raw_prob = float(ensemble.predict_proba(X)[0])

        calibrated = False
        if regime in self._calibrators:
            cal_arr = self._calibrators[regime].calibrate(np.array([raw_prob]))
            prob = float(cal_arr[0])
            calibrated = True
        else:
            prob = raw_prob

        prob = float(np.clip(prob, 0.02, 0.98))

        # For domestic leagues: no draw possible in regulation (OT settles it)
        # For international/WC group stage: draws possible; we split residual
        draw_prob = 0.0
        if inp.competition_level is not None and inp.competition_level >= 4:
            # World championship group stage — allocate ~8% draw probability
            draw_prob = 0.08
            prob = prob * (1.0 - draw_prob)
        away_prob = 1.0 - prob - draw_prob

        # Confidence
        abs_edge = abs(prob - 0.5)
        if abs_edge > 0.2:
            confidence = "high"
        elif abs_edge > 0.1:
            confidence = "medium"
        else:
            confidence = "low"

        return PredictionResult(
            home_win_prob=round(prob, 4),
            away_win_prob=round(away_prob, 4),
            draw_prob=round(draw_prob, 4),
            regime=regime,
            model_mode=getattr(ensemble, "mode", "unknown"),
            raw_prob=round(raw_prob, 4),
            calibrated=calibrated,
            confidence=confidence,
            features_used=feat,
        )
