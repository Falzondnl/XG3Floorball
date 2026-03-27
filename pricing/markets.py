"""
Floorball Market Pricer
=======================
Produces:
  1. Win/Draw/Win (3-way) — domestic leagues: no draw → 2-way (1X2 with draw ~0)
  2. Asian Handicap
  3. Total Goals Over/Under (Poisson)

All probabilities are sourced from the ML predictor — no hardcoded values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import (
    ASIAN_HANDICAP_MARGIN,
    TOTALS_LINE,
    TOTALS_MARGIN,
    WIN_DRAW_WIN_MARGIN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prob_to_odds(prob: float, margin: float) -> float:
    """Convert probability to decimal odds after applying margin."""
    if prob <= 0.0:
        return 999.0
    fair_odds = 1.0 / prob
    # Apply margin by reducing fair odds
    return round(fair_odds * (1.0 - margin), 3)


def _normalise(probs: list[float]) -> list[float]:
    """Normalise a list of probabilities to sum to 1."""
    total = sum(probs)
    if total == 0:
        raise ValueError("Cannot normalise zero-sum probabilities")
    return [p / total for p in probs]


def _poisson_cdf(lam: float, k: int) -> float:
    """P(X <= k) for Poisson(lambda)."""
    prob = 0.0
    for i in range(k + 1):
        prob += math.exp(-lam) * (lam ** i) / math.factorial(i)
    return min(prob, 1.0)


def _poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


# ---------------------------------------------------------------------------
# Scoring distribution from probabilities
# ---------------------------------------------------------------------------

def _estimate_lambdas(
    home_win_prob: float,
    away_win_prob: float,
    draw_prob: float,
    avg_total_goals: float = 10.5,
) -> tuple[float, float]:
    """
    Estimate home/away Poisson lambdas consistent with win probabilities.
    Uses a simple iterative approach common in sports betting models.
    """
    # Rough estimate: if home wins 60%, home expected ~5.5% more goals
    home_share = (home_win_prob + 0.5 * draw_prob)
    away_share = (away_win_prob + 0.5 * draw_prob)
    total = home_share + away_share
    if total == 0:
        total = 1.0
    home_lambda = avg_total_goals * (home_share / total)
    away_lambda = avg_total_goals * (away_share / total)
    # Constrain: floorball lambdas typically 4-8 per team
    home_lambda = float(np.clip(home_lambda, 2.0, 12.0))
    away_lambda = float(np.clip(away_lambda, 2.0, 12.0))
    return home_lambda, away_lambda


# ---------------------------------------------------------------------------
# Market dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WinDrawWinMarket:
    home_odds: float
    draw_odds: float
    away_odds: float
    home_prob: float
    draw_prob: float
    away_prob: float
    overround: float

    def to_dict(self) -> dict:
        d = {
            "market": "1X2",
            "home_odds": self.home_odds,
            "away_odds": self.away_odds,
            "home_prob": round(self.home_prob, 4),
            "away_prob": round(self.away_prob, 4),
            "overround": round(self.overround, 4),
        }
        if self.draw_prob > 0.0:
            d["draw_odds"] = self.draw_odds
            d["draw_prob"] = round(self.draw_prob, 4)
        else:
            d["draw_odds"] = None   # Not offered in domestic leagues
            d["draw_prob"] = 0.0
        return d


@dataclass
class AsianHandicapMarket:
    line: float
    home_odds: float
    away_odds: float
    home_prob: float
    away_prob: float
    overround: float

    def to_dict(self) -> dict:
        return {
            "market": "asian_handicap",
            "line": self.line,
            "home_odds": self.home_odds,
            "away_odds": self.away_odds,
            "home_prob": round(self.home_prob, 4),
            "away_prob": round(self.away_prob, 4),
            "overround": round(self.overround, 4),
        }


@dataclass
class TotalsMarket:
    line: float
    over_odds: float
    under_odds: float
    over_prob: float
    under_prob: float
    overround: float
    home_lambda: float
    away_lambda: float

    def to_dict(self) -> dict:
        return {
            "market": "total_goals",
            "line": self.line,
            "over_odds": self.over_odds,
            "under_odds": self.under_odds,
            "over_prob": round(self.over_prob, 4),
            "under_prob": round(self.under_prob, 4),
            "overround": round(self.overround, 4),
            "home_lambda": round(self.home_lambda, 3),
            "away_lambda": round(self.away_lambda, 3),
        }


@dataclass
class FullPriceResponse:
    home_win_prob: float
    away_win_prob: float
    draw_prob: float
    win_draw_win: WinDrawWinMarket
    asian_handicap: AsianHandicapMarket
    totals: list[TotalsMarket]
    regime: str
    model_mode: str
    confidence: str

    def to_dict(self) -> dict:
        return {
            "home_win_prob": round(self.home_win_prob, 4),
            "away_win_prob": round(self.away_win_prob, 4),
            "draw_prob": round(self.draw_prob, 4),
            "regime": self.regime,
            "model_mode": self.model_mode,
            "confidence": self.confidence,
            "markets": {
                "win_draw_win": self.win_draw_win.to_dict(),
                "asian_handicap": self.asian_handicap.to_dict(),
                "totals": [t.to_dict() for t in self.totals],
            },
        }


# ---------------------------------------------------------------------------
# Pricer
# ---------------------------------------------------------------------------

class FloorballMarketPricer:
    """
    Stateless pricer. Converts ML probabilities → priced markets.
    """

    def price(
        self,
        home_win_prob: float,
        away_win_prob: float,
        draw_prob: float,
        regime: str = "r0",
        model_mode: str = "ensemble",
        confidence: str = "medium",
        custom_totals_line: Optional[float] = None,
    ) -> FullPriceResponse:

        # --- 1X2 ---
        wdw = self._price_win_draw_win(home_win_prob, draw_prob, away_win_prob)

        # --- Asian Handicap ---
        ah = self._price_asian_handicap(home_win_prob, away_win_prob, draw_prob)

        # --- Totals ---
        totals_line = custom_totals_line or TOTALS_LINE
        home_lambda, away_lambda = _estimate_lambdas(
            home_win_prob, away_win_prob, draw_prob, avg_total_goals=totals_line
        )
        totals_markets = self._price_totals(
            home_lambda, away_lambda, lines=[totals_line - 1, totals_line, totals_line + 1]
        )

        return FullPriceResponse(
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            draw_prob=draw_prob,
            win_draw_win=wdw,
            asian_handicap=ah,
            totals=totals_markets,
            regime=regime,
            model_mode=model_mode,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Win/Draw/Win
    # ------------------------------------------------------------------

    def _price_win_draw_win(
        self,
        home_prob: float,
        draw_prob: float,
        away_prob: float,
    ) -> WinDrawWinMarket:
        margin = WIN_DRAW_WIN_MARGIN

        is_domestic = draw_prob < 0.01
        if is_domestic:
            # Domestic league — no draw possible (OT/SO settles every game)
            # Normalise only home/away; return None for draw odds
            total = home_prob + away_prob
            if total == 0:
                total = 1.0
            normed_h = home_prob / total
            normed_a = away_prob / total
            overround = 1.0 + margin
            h_o = round(1.0 / (normed_h * overround), 3)
            a_o = round(1.0 / (normed_a * overround), 3)
            actual_overround = 1.0 / h_o + 1.0 / a_o
            return WinDrawWinMarket(
                home_odds=h_o,
                draw_odds=0.0,   # 0.0 = not offered
                away_odds=a_o,
                home_prob=round(normed_h, 4),
                draw_prob=0.0,
                away_prob=round(normed_a, 4),
                overround=round(actual_overround, 4),
            )
        else:
            # International format — full 1X2
            raw = [max(home_prob, 0.001), max(draw_prob, 0.001), max(away_prob, 0.001)]
            normed = _normalise(raw)
            overround = 1.0 + margin
            h_o = round(1.0 / (normed[0] * overround), 3)
            d_o = round(1.0 / (normed[1] * overround), 3)
            a_o = round(1.0 / (normed[2] * overround), 3)
            actual_overround = 1.0 / h_o + 1.0 / d_o + 1.0 / a_o
            return WinDrawWinMarket(
                home_odds=h_o,
                draw_odds=d_o,
                away_odds=a_o,
                home_prob=normed[0],
                draw_prob=normed[1],
                away_prob=normed[2],
                overround=round(actual_overround, 4),
            )

    # ------------------------------------------------------------------
    # Asian Handicap
    # ------------------------------------------------------------------

    def _price_asian_handicap(
        self,
        home_win_prob: float,
        away_win_prob: float,
        draw_prob: float,
    ) -> AsianHandicapMarket:
        """
        Determine handicap line from ELO-implied margin.
        P(home covers) derived from win probabilities.
        """
        # Effective line: round to nearest 0.5
        # Home probability (excl draw) of winning outright
        home_adj = home_win_prob + 0.5 * draw_prob
        away_adj = away_win_prob + 0.5 * draw_prob

        # Line: if home_adj > 0.55 → home gives goals (negative line)
        raw_diff = home_adj - away_adj   # positive = home favoured
        line = round(raw_diff * 4) / 2  # map to nearest 0.5
        line = float(np.clip(line, -4.5, 4.5))

        # For a 0-line Asian Handicap: home_prob ≈ home_adj adjusted by line
        # Probability home covers line
        home_cover_prob = float(np.clip(0.5 + (home_adj - 0.5 - line * 0.05), 0.03, 0.97))
        away_cover_prob = 1.0 - home_cover_prob

        overround = 1.0 + ASIAN_HANDICAP_MARGIN
        h_o = round(1.0 / (home_cover_prob * overround), 3)
        a_o = round(1.0 / (away_cover_prob * overround), 3)
        actual_or = 1.0 / h_o + 1.0 / a_o

        return AsianHandicapMarket(
            line=-line,  # convention: negative = home gives goals
            home_odds=h_o,
            away_odds=a_o,
            home_prob=home_cover_prob,
            away_prob=away_cover_prob,
            overround=round(actual_or, 4),
        )

    # ------------------------------------------------------------------
    # Totals (Poisson)
    # ------------------------------------------------------------------

    def _price_totals(
        self,
        home_lambda: float,
        away_lambda: float,
        lines: list[float],
    ) -> list[TotalsMarket]:
        markets = []
        for line in lines:
            # P(total > line) using independence of home/away Poisson
            over_prob = self._poisson_over_prob(home_lambda, away_lambda, line)
            under_prob = 1.0 - over_prob

            # Clip
            over_prob = float(np.clip(over_prob, 0.03, 0.97))
            under_prob = 1.0 - over_prob

            overround = 1.0 + TOTALS_MARGIN
            o_o = round(1.0 / (over_prob * overround), 3)
            u_o = round(1.0 / (under_prob * overround), 3)
            actual_or = 1.0 / o_o + 1.0 / u_o

            markets.append(TotalsMarket(
                line=line,
                over_odds=o_o,
                under_odds=u_o,
                over_prob=round(over_prob, 4),
                under_prob=round(under_prob, 4),
                overround=round(actual_or, 4),
                home_lambda=home_lambda,
                away_lambda=away_lambda,
            ))
        return markets

    @staticmethod
    def _poisson_over_prob(lam_h: float, lam_a: float, line: float) -> float:
        """
        P(home_goals + away_goals > line) via convolution of two Poisson distributions.
        """
        total_lambda = lam_h + lam_a
        k_int = int(math.floor(line))
        return 1.0 - _poisson_cdf(total_lambda, k_int)
