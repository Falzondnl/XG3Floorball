"""
Live repricing for the XG3 Floorball microservice.

Endpoint
--------
  POST /api/v1/floorball/live/price

Domain context
--------------
  IFF/domestic floorball: 3 periods × 20 min = 60 min regulation.
  WDW market (Win/Draw/Win) — draws count in regulation; OT possible.
  Average combined goals ≈ 10–12 per game; typical line 10.5.
  ELO home advantage empirically calibrated to +50 pts.

  Live repricing inputs:
    - home_score / away_score         — current score
    - period (1–3)                    — current period
    - time_remaining_seconds          — seconds left in regulation (max 3600)
    - home_elo / away_elo             — team ELO ratings
    - pinnacle_home_odds / draw_odds / away_odds — optional; triggers 80/20 logit blend
      (Three-way Pinnacle blend when all three are supplied.)
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/floorball/live", tags=["live"])

# Floorball — 3 periods × 20 min = 60 min = 3600 s regulation
_REGULATION_SECONDS: float = 3600.0

_EPS = 1e-9

# ---------------------------------------------------------------------------
# Auto-suspension state — stale feed detection
# ---------------------------------------------------------------------------

# match_id → timestamp of last live-price request received
_last_event_times: Dict[str, datetime] = {}

# Seconds without a new event before markets are auto-suspended
_STALE_THRESHOLD_S: float = 30.0


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class LivePriceRequest(BaseModel):
    match_id: str = Field(..., description="Unique match identifier")
    home_team_id: int = Field(..., description="Sofascore home team ID")
    away_team_id: int = Field(..., description="Sofascore away team ID")
    home_score: int = Field(0, ge=0, description="Current home score")
    away_score: int = Field(0, ge=0, description="Current away score")
    period: int = Field(1, ge=1, le=3, description="Current period (1–3)")
    time_remaining_seconds: int = Field(
        3600, ge=0, le=3600,
        description="Seconds remaining in regulation (max 3600)",
    )
    home_elo: float = Field(1500.0, ge=500.0, le=3000.0, description="Home team ELO rating")
    away_elo: float = Field(1500.0, ge=500.0, le=3000.0, description="Away team ELO rating")
    # Optional live Pinnacle odds — all three required to activate blend
    pinnacle_home_odds: Optional[float] = Field(
        None, gt=1.0, le=100.0, description="Pinnacle decimal odds for home win"
    )
    pinnacle_draw_odds: Optional[float] = Field(
        None, gt=1.0, le=100.0, description="Pinnacle decimal odds for draw"
    )
    pinnacle_away_odds: Optional[float] = Field(
        None, gt=1.0, le=100.0, description="Pinnacle decimal odds for away win"
    )


class LivePriceResponse(BaseModel):
    match_id: str
    home_team_id: int
    away_team_id: int
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    home_score: int
    away_score: int
    period: int
    time_remaining_seconds: int
    blend_source: str
    live: bool
    elapsed_ms: float
    request_id: str
    pricing_timestamp: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _elo_prob(home_elo: float, away_elo: float, home_advantage: float = 50.0) -> float:
    """ELO win probability for the home team."""
    diff = home_elo - away_elo + home_advantage
    return 1.0 / (1.0 + math.pow(10.0, -diff / 400.0))


def _score_state_update(
    base_prob: float,
    score_diff: int,
    time_remaining_frac: float,
) -> float:
    """
    Update home win probability based on live score state.

    The per-goal shift grows as time_remaining_frac decreases (late goals
    carry more decisive information than early goals).
    """
    time_gone_frac = 1.0 - time_remaining_frac
    per_goal_shift = 0.05 * time_gone_frac
    max_shift = 0.38 * time_gone_frac
    shift = max(-max_shift, min(max_shift, score_diff * per_goal_shift))
    return max(_EPS, min(1.0 - _EPS, base_prob + shift))


def _draw_prob(
    home_prob: float,
    score_diff: int,
    time_remaining_frac: float,
) -> float:
    """
    Contextual draw probability.

    Highest when scores are level and meaningful time remains.
    Collapses to near-zero as game approaches end.
    """
    symmetry = 1.0 - min(1.0, abs(score_diff) * 0.30)   # 0 if 3+ apart, 1 if tied
    base_draw = 0.10 * symmetry * time_remaining_frac
    return float(max(0.0, min(0.20, base_draw)))


def _logit(p: float) -> float:
    p = max(_EPS, min(1.0 - _EPS, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _pinnacle_three_way_blend(
    model_home: float,
    model_draw: float,
    model_away: float,
    pin_home_odds: float,
    pin_draw_odds: float,
    pin_away_odds: float,
    model_weight: float = 0.80,
) -> tuple[float, float, float]:
    """
    80/20 logit-space blend for each outcome individually, then re-normalise.

    Pinnacle vig is removed before blending.
    """
    vig_total = (1.0 / pin_home_odds) + (1.0 / pin_draw_odds) + (1.0 / pin_away_odds)
    if vig_total <= 0.0:
        raise ValueError(f"Invalid vig total: {vig_total}")
    pin_home = (1.0 / pin_home_odds) / vig_total
    pin_draw = (1.0 / pin_draw_odds) / vig_total
    pin_away = (1.0 / pin_away_odds) / vig_total

    blended_home_logit = model_weight * _logit(model_home) + (1.0 - model_weight) * _logit(pin_home)
    blended_draw_logit = model_weight * _logit(model_draw) + (1.0 - model_weight) * _logit(pin_draw)
    blended_away_logit = model_weight * _logit(model_away) + (1.0 - model_weight) * _logit(pin_away)

    raw_home = _sigmoid(blended_home_logit)
    raw_draw = _sigmoid(blended_draw_logit)
    raw_away = _sigmoid(blended_away_logit)

    total = raw_home + raw_draw + raw_away
    return raw_home / total, raw_draw / total, raw_away / total


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/price",
    response_model=LivePriceResponse,
    summary="Reprice a floorball match based on current in-play state",
)
async def live_price(req: LivePriceRequest) -> LivePriceResponse:
    """
    Reprice a floorball match from its current live state.

    Steps
    -----
    1. ELO-based pre-match probability (home advantage +50 ELO pts).
    2. Score-state probability update weighted by time elapsed.
    3. Contextual draw probability (tied + time remaining).
    4. Normalise three-way distribution.
    5. If all three Pinnacle live odds supplied, apply per-outcome 80/20
       logit-space blend and re-normalise.
    """
    rid = str(uuid.uuid4())
    t0 = time.monotonic()

    # ------------------------------------------------------------------
    # Auto-suspend check — stale feed detection
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc)
    last_event = _last_event_times.get(req.match_id)
    if last_event is not None:
        stale_seconds = (now - last_event).total_seconds()
        if stale_seconds > _STALE_THRESHOLD_S:
            log.warning(
                "floorball.auto_suspend_triggered match_id=%s sport=floorball stale_seconds=%.1f",
                req.match_id,
                stale_seconds,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "match_id": req.match_id,
                    "suspended": True,
                    "reason": "feed_timeout",
                    "stale_seconds": round(stale_seconds, 1),
                },
            )
    _last_event_times[req.match_id] = now

    time_remaining_frac = max(0.0, min(1.0, req.time_remaining_seconds / _REGULATION_SECONDS))
    score_diff = req.home_score - req.away_score

    # Step 1 — ELO base
    base_prob = _elo_prob(req.home_elo, req.away_elo)

    # Step 2 — score-state update
    adjusted_home = _score_state_update(base_prob, score_diff, time_remaining_frac)

    # Step 3 — draw probability
    draw_prob = _draw_prob(adjusted_home, score_diff, time_remaining_frac)

    # Step 4 — normalise three-way
    raw_home = max(_EPS, adjusted_home - draw_prob / 2.0)
    raw_away = max(_EPS, 1.0 - adjusted_home - draw_prob / 2.0)
    raw_draw = max(0.0, draw_prob)
    total = raw_home + raw_draw + raw_away
    home_win_prob = raw_home / total
    draw_prob_norm = raw_draw / total
    away_win_prob = raw_away / total

    blend_source = "model_elo_score_state"

    # Step 5 — optional three-way Pinnacle blend
    if (
        req.pinnacle_home_odds is not None
        and req.pinnacle_draw_odds is not None
        and req.pinnacle_away_odds is not None
    ):
        try:
            home_win_prob, draw_prob_norm, away_win_prob = _pinnacle_three_way_blend(
                model_home=home_win_prob,
                model_draw=draw_prob_norm,
                model_away=away_win_prob,
                pin_home_odds=req.pinnacle_home_odds,
                pin_draw_odds=req.pinnacle_draw_odds,
                pin_away_odds=req.pinnacle_away_odds,
            )
            blend_source = "pinnacle_blend_80_20_three_way"
        except Exception as exc:
            log.warning(
                "floorball_live_pinnacle_blend_failed match_id=%s error=%s",
                req.match_id,
                exc,
            )

    elapsed_ms = round((time.monotonic() - t0) * 1000, 3)

    log.info(
        "floorball_live_price match_id=%s home_win=%.4f draw=%.4f away_win=%.4f "
        "score=%d-%d period=%d blend=%s elapsed_ms=%.3f",
        req.match_id,
        home_win_prob,
        draw_prob_norm,
        away_win_prob,
        req.home_score,
        req.away_score,
        req.period,
        blend_source,
        elapsed_ms,
    )

    return LivePriceResponse(
        match_id=req.match_id,
        home_team_id=req.home_team_id,
        away_team_id=req.away_team_id,
        home_win_prob=round(home_win_prob, 6),
        draw_prob=round(draw_prob_norm, 6),
        away_win_prob=round(away_win_prob, 6),
        home_score=req.home_score,
        away_score=req.away_score,
        period=req.period,
        time_remaining_seconds=req.time_remaining_seconds,
        blend_source=blend_source,
        live=True,
        elapsed_ms=elapsed_ms,
        request_id=rid,
        pricing_timestamp=datetime.now(timezone.utc).isoformat(),
    )
