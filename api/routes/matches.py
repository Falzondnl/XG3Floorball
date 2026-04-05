"""
Match prediction and pricing endpoints.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from config import ELO_DEFAULT
from ml.predictor import PredictionInput
from pricing.markets import FloorballMarketPricer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/floorball")

pricer = FloorballMarketPricer()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    home_team_id: int = Field(..., description="Sofascore team ID (home)")
    away_team_id: int = Field(..., description="Sofascore team ID (away)")
    tournament_name: str = Field(..., description="Tournament name e.g. 'Svenska Superligan'")
    is_women: bool = Field(False, description="Women's competition flag")
    home_elo: Optional[float] = Field(None, description="Home team ELO (overrides DB lookup)")
    away_elo: Optional[float] = Field(None, description="Away team ELO (overrides DB lookup)")
    form_pts_home: Optional[float] = Field(None, ge=0.0, le=3.0)
    form_pts_away: Optional[float] = Field(None, ge=0.0, le=3.0)
    form_gd_home: Optional[float] = Field(None)
    form_gd_away: Optional[float] = Field(None)
    goals_home_avg5: Optional[float] = Field(None, ge=0.0)
    goals_away_avg5: Optional[float] = Field(None, ge=0.0)
    h2h_home_win_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    competition_level: Optional[int] = Field(None, ge=1, le=5)
    is_playoff: Optional[int] = Field(None, ge=0, le=1)
    custom_totals_line: Optional[float] = Field(None, ge=3.0, le=30.0)
    # Pinnacle live odds — optional; all three required to activate 80/20 three-way blend
    pinnacle_home_odds: Optional[float] = Field(
        None, gt=1.0, le=100.0, description="Pinnacle decimal odds for home win"
    )
    pinnacle_draw_odds: Optional[float] = Field(
        None, gt=1.0, le=100.0, description="Pinnacle decimal odds for draw"
    )
    pinnacle_away_odds: Optional[float] = Field(
        None, gt=1.0, le=100.0, description="Pinnacle decimal odds for away win"
    )

    @field_validator("home_elo", "away_elo")
    @classmethod
    def validate_elo(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 500 or v > 3000):
            raise ValueError("ELO must be in range [500, 3000]")
        return v


class PredictResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    tournament_name: str
    home_win_prob: float
    away_win_prob: float
    draw_prob: float
    regime: str
    model_mode: str
    confidence: str
    markets: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """
    Get prediction and priced markets for a floorball match.
    Returns Win/Draw/Win, Asian Handicap, and Total Goals markets.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.ready:
        raise HTTPException(
            status_code=503,
            detail="Prediction models not loaded. Service starting up.",
        )

    inp = PredictionInput(
        home_team_id=body.home_team_id,
        away_team_id=body.away_team_id,
        tournament_name=body.tournament_name,
        is_women=body.is_women,
        home_elo=body.home_elo,
        away_elo=body.away_elo,
        form_pts_home=body.form_pts_home,
        form_pts_away=body.form_pts_away,
        form_gd_home=body.form_gd_home,
        form_gd_away=body.form_gd_away,
        goals_home_avg5=body.goals_home_avg5,
        goals_away_avg5=body.goals_away_avg5,
        h2h_home_win_rate=body.h2h_home_win_rate,
        competition_level=body.competition_level,
        is_playoff=body.is_playoff,
    )

    try:
        result = predictor.predict(inp)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    # Optional Pinnacle 80/20 three-way logit-space blend
    home_win_prob = result.home_win_prob
    draw_prob = result.draw_prob
    away_win_prob = result.away_win_prob
    if (
        body.pinnacle_home_odds is not None
        and body.pinnacle_draw_odds is not None
        and body.pinnacle_away_odds is not None
    ):
        try:
            import math as _math
            _eps = 1e-9
            vig_total = (
                1.0 / body.pinnacle_home_odds
                + 1.0 / body.pinnacle_draw_odds
                + 1.0 / body.pinnacle_away_odds
            )
            pin_home = (1.0 / body.pinnacle_home_odds) / vig_total
            pin_draw = (1.0 / body.pinnacle_draw_odds) / vig_total
            pin_away = (1.0 / body.pinnacle_away_odds) / vig_total

            def _logit(p: float) -> float:
                p = max(_eps, min(1.0 - _eps, p))
                return _math.log(p / (1.0 - p))

            def _sigmoid(x: float) -> float:
                return 1.0 / (1.0 + _math.exp(-x))

            raw_home = _sigmoid(0.80 * _logit(home_win_prob) + 0.20 * _logit(pin_home))
            raw_draw = _sigmoid(0.80 * _logit(draw_prob) + 0.20 * _logit(pin_draw))
            raw_away = _sigmoid(0.80 * _logit(away_win_prob) + 0.20 * _logit(pin_away))
            blend_total = raw_home + raw_draw + raw_away
            home_win_prob = raw_home / blend_total
            draw_prob = raw_draw / blend_total
            away_win_prob = raw_away / blend_total
        except Exception as _blend_exc:
            logger.warning(
                "floorball_pinnacle_blend_failed home=%d away=%d error=%s",
                body.home_team_id,
                body.away_team_id,
                _blend_exc,
            )

    try:
        priced = pricer.price(
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            draw_prob=draw_prob,
            regime=result.regime,
            model_mode=result.model_mode,
            confidence=result.confidence,
            custom_totals_line=body.custom_totals_line,
        )
    except Exception as exc:
        logger.exception("Pricing failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pricing error: {exc}") from exc

    return PredictResponse(
        home_team_id=body.home_team_id,
        away_team_id=body.away_team_id,
        tournament_name=body.tournament_name,
        home_win_prob=home_win_prob,
        away_win_prob=away_win_prob,
        draw_prob=draw_prob,
        regime=result.regime,
        model_mode=result.model_mode,
        confidence=result.confidence,
        markets=priced.to_dict()["markets"],
    )


@router.get("/markets/price-match")
async def price_match(
    home_team_id: int,
    away_team_id: int,
    tournament_name: str,
    is_women: bool = False,
    request: Request = None,
) -> dict:
    """
    GET convenience endpoint for pricing a match with default features.
    Uses only ELO (defaults to 1500 if teams unknown) + tournament metadata.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.ready:
        raise HTTPException(status_code=503, detail="Models not ready")

    inp = PredictionInput(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        tournament_name=tournament_name,
        is_women=is_women,
    )

    try:
        result = predictor.predict(inp)
        priced = pricer.price(
            home_win_prob=result.home_win_prob,
            away_win_prob=result.away_win_prob,
            draw_prob=result.draw_prob,
            regime=result.regime,
            model_mode=result.model_mode,
            confidence=result.confidence,
        )
        return {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "tournament_name": tournament_name,
            **priced.to_dict(),
        }
    except Exception as exc:
        logger.exception("price-match failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
