"""
Floorball Feature Extractor
===========================
Derives match-level features from historical Sofascore event data.

Features (all pre-match, no leakage):
  - ELO rating home / away (computed from historical results, updated after each match)
  - ELO rating difference
  - Recent form home / away (W/D/L last 5 games → points rate 3/1/0)
  - Recent goal difference last 5 for each team
  - H2H win rate (home team wins in last 10 H2H encounters)
  - Home scoring average last 5
  - Away scoring average last 5
  - Competition level (ordinal 1-5)
  - Is home/away an international team flag
  - Is playoff flag
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from config import (
    COMPETITION_LEVELS,
    ELO_DEFAULT,
    ELO_HOME_ADVANTAGE,
    ELO_K_INITIAL,
    ELO_K_SETTLED,
    WOMEN_KEYWORDS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ELO engine
# ---------------------------------------------------------------------------

def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _k_factor(games_played: int) -> float:
    return ELO_K_INITIAL if games_played < 20 else ELO_K_SETTLED


def build_elo_timeline(
    df: pd.DataFrame,
) -> dict[int, list[float]]:
    """
    Walk matches in chronological order and build per-team ELO series.
    Returns dict: team_id -> list of ELO snapshots (one per match, BEFORE the match).
    """
    df_sorted = df.sort_values("start_timestamp").reset_index(drop=True)
    elo: dict[int, float] = defaultdict(lambda: ELO_DEFAULT)
    games_played: dict[int, int] = defaultdict(int)

    # We store the PRE-match ELO snapshot for each row index
    pre_match_elo_home: list[float] = []
    pre_match_elo_away: list[float] = []

    for _, row in df_sorted.iterrows():
        hid = int(row["home_team_id"])
        aid = int(row["away_team_id"])
        wc = row["winner_code"]

        h_elo = elo[hid]
        a_elo = elo[aid]
        pre_match_elo_home.append(h_elo)
        pre_match_elo_away.append(a_elo)

        # Update ELO only for finished matches with a known winner
        if row["status_type"] != "finished" or pd.isna(wc):
            games_played[hid] += 1
            games_played[aid] += 1
            continue

        wc_int = int(wc)
        # winner_code: 1=home win, 2=away win, 3=draw
        if wc_int == 1:
            actual_h, actual_a = 1.0, 0.0
        elif wc_int == 2:
            actual_h, actual_a = 0.0, 1.0
        else:
            actual_h, actual_a = 0.5, 0.5

        # Apply home advantage in ELO calculation
        exp_h = _expected_score(h_elo + ELO_HOME_ADVANTAGE, a_elo)
        exp_a = 1.0 - exp_h

        k_h = _k_factor(games_played[hid])
        k_a = _k_factor(games_played[aid])

        elo[hid] = h_elo + k_h * (actual_h - exp_h)
        elo[aid] = a_elo + k_a * (actual_a - exp_a)
        games_played[hid] += 1
        games_played[aid] += 1

    df_sorted["elo_home_pre"] = pre_match_elo_home
    df_sorted["elo_away_pre"] = pre_match_elo_away
    return df_sorted


# ---------------------------------------------------------------------------
# Recent form
# ---------------------------------------------------------------------------

def _compute_recent_form(
    df_sorted: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    For each match row, compute:
      - form_pts_home_last5:  points (3/1/0) per game in last `window` matches
      - form_gd_home_last5:   average goal difference in last `window` matches
      - form_pts_away_last5
      - form_gd_away_last5
    Uses only PAST matches (strict temporal ordering, no leakage).
    """
    # Build per-team history of (timestamp, points, goal_diff)
    team_history: dict[int, list[tuple[float, float, float]]] = defaultdict(list)

    form_pts_home: list[float] = []
    form_gd_home: list[float] = []
    form_pts_away: list[float] = []
    form_gd_away: list[float] = []
    goals_scored_home: list[float] = []
    goals_scored_away: list[float] = []

    for _, row in df_sorted.iterrows():
        hid = int(row["home_team_id"])
        aid = int(row["away_team_id"])
        ts = float(row["start_timestamp"])

        def _last_n(tid: int) -> tuple[float, float, float, float]:
            hist = team_history[tid]
            if not hist:
                return 1.0, 0.0, 5.0, 5.0  # neutral defaults
            recent = [h for h in hist if h[0] < ts][-window:]
            if not recent:
                return 1.0, 0.0, 5.0, 5.0
            pts = np.mean([r[1] for r in recent])
            gd = np.mean([r[2] for r in recent])
            gs = np.mean([r[3] for r in recent])
            return pts, gd, gs, gs

        h_pts, h_gd, h_gs, _ = _last_n(hid)
        a_pts, a_gd, _, a_gs = _last_n(aid)

        form_pts_home.append(h_pts)
        form_gd_home.append(h_gd)
        form_pts_away.append(a_pts)
        form_gd_away.append(a_gd)
        goals_scored_home.append(h_gs)
        goals_scored_away.append(a_gs)

        # Update history after recording features
        wc = row["winner_code"]
        hs = row.get("home_score_current", np.nan)
        as_ = row.get("away_score_current", np.nan)

        if row["status_type"] == "finished" and not pd.isna(wc):
            wc_int = int(wc)
            if wc_int == 1:
                h_pts_upd, a_pts_upd = 3.0, 0.0
            elif wc_int == 2:
                h_pts_upd, a_pts_upd = 0.0, 3.0
            else:
                h_pts_upd, a_pts_upd = 1.0, 1.0

            if not pd.isna(hs) and not pd.isna(as_):
                h_gd_upd = float(hs) - float(as_)
                a_gd_upd = float(as_) - float(hs)
                h_gs_upd = float(hs)
                a_gs_upd = float(as_)
            else:
                h_gd_upd = a_gd_upd = 0.0
                h_gs_upd = a_gs_upd = 5.0

            team_history[hid].append((ts, h_pts_upd, h_gd_upd, h_gs_upd))
            team_history[aid].append((ts, a_pts_upd, a_gd_upd, a_gs_upd))

    df_sorted = df_sorted.copy()
    df_sorted["form_pts_home_last5"] = form_pts_home
    df_sorted["form_gd_home_last5"] = form_gd_home
    df_sorted["form_pts_away_last5"] = form_pts_away
    df_sorted["form_gd_away_last5"] = form_gd_away
    df_sorted["goals_scored_home_avg5"] = goals_scored_home
    df_sorted["goals_scored_away_avg5"] = goals_scored_away
    return df_sorted


# ---------------------------------------------------------------------------
# H2H
# ---------------------------------------------------------------------------

def _compute_h2h(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    h2h_home_win_rate: rate of home-team wins in last 10 H2H matches.
    Uses only strictly past matches.
    """
    h2h_history: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)

    h2h_rates: list[float] = []

    for _, row in df_sorted.iterrows():
        hid = int(row["home_team_id"])
        aid = int(row["away_team_id"])
        ts = float(row["start_timestamp"])
        key1 = (min(hid, aid), max(hid, aid))

        hist = h2h_history[key1]
        past = [h for h in hist if h[0] < ts][-10:]

        if not past:
            h2h_rates.append(0.5)
        else:
            home_wins = sum(1 for h in past if h[1] == hid)
            h2h_rates.append(home_wins / len(past))

        wc = row["winner_code"]
        if row["status_type"] == "finished" and not pd.isna(wc):
            wc_int = int(wc)
            winner_id = hid if wc_int == 1 else (aid if wc_int == 2 else 0)
            h2h_history[key1].append((ts, winner_id))

    df_sorted = df_sorted.copy()
    df_sorted["h2h_home_win_rate"] = h2h_rates
    return df_sorted


# ---------------------------------------------------------------------------
# Competition level
# ---------------------------------------------------------------------------

def _competition_level(tournament_name: str) -> int:
    name_lower = tournament_name.lower() if isinstance(tournament_name, str) else ""
    for key, level in COMPETITION_LEVELS.items():
        if key in name_lower:
            return level
    return 1


def _is_playoff(tournament_name: str) -> int:
    name_lower = tournament_name.lower() if isinstance(tournament_name, str) else ""
    return 1 if "playoff" in name_lower or "final" in name_lower else 0


def _is_women(tournament_name: str) -> int:
    if not isinstance(tournament_name, str):
        return 0
    for kw in WOMEN_KEYWORDS:
        if kw in tournament_name:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "elo_home_pre",
    "elo_away_pre",
    "elo_diff",
    "form_pts_home_last5",
    "form_pts_away_last5",
    "form_gd_home_last5",
    "form_gd_away_last5",
    "goals_scored_home_avg5",
    "goals_scored_away_avg5",
    "h2h_home_win_rate",
    "competition_level",
    "is_playoff",
    "is_women",
]


class FloorballFeatureExtractor:
    """
    Stateless transformer: given a DataFrame of raw Sofascore events,
    returns a feature matrix (X) and target vector (y).
    """

    def fit_transform(
        self, df_raw: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Returns (X, y, enriched_df) where:
          y = 1 if home wins, 0 if away wins (draws excluded for binary target,
              or kept as 0.5 for soft-label — here excluded for cleaner training)
        """
        df = df_raw.copy()
        df = df[df["status_type"] == "finished"].copy()
        df = df[df["winner_code"].notna()].copy()
        df["winner_code"] = df["winner_code"].astype(int)

        # Sort by time (CRITICAL for temporal integrity)
        df = df.sort_values("start_timestamp").reset_index(drop=True)

        logger.info("Building ELO timeline on %d finished matches...", len(df))
        df = build_elo_timeline(df)

        logger.info("Computing recent form...")
        df = _compute_recent_form(df)

        logger.info("Computing H2H...")
        df = _compute_h2h(df)

        # Competition metadata
        df["competition_level"] = df["tournament_name"].apply(_competition_level)
        df["is_playoff"] = df["tournament_name"].apply(_is_playoff)
        df["is_women"] = df["tournament_name"].apply(_is_women)

        # Derived
        df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]

        # Target: 1=home win, 0=away win; exclude draws (winner_code=3)
        df_binary = df[df["winner_code"].isin([1, 2])].copy()
        df_binary["target"] = (df_binary["winner_code"] == 1).astype(int)

        X = df_binary[FEATURE_COLUMNS].copy()
        y = df_binary["target"]

        logger.info(
            "Feature matrix: %d rows, %d cols. Target balance: %.1f%% home wins",
            len(X),
            len(FEATURE_COLUMNS),
            100.0 * y.mean(),
        )
        return X, y, df_binary

    def transform_single(
        self,
        home_elo: float,
        away_elo: float,
        form_pts_home: float,
        form_pts_away: float,
        form_gd_home: float,
        form_gd_away: float,
        goals_home_avg5: float,
        goals_away_avg5: float,
        h2h_home_win_rate: float,
        competition_level: int,
        is_playoff: int,
        is_women: int,
    ) -> dict[str, Any]:
        return {
            "elo_home_pre": home_elo,
            "elo_away_pre": away_elo,
            "elo_diff": home_elo - away_elo,
            "form_pts_home_last5": form_pts_home,
            "form_pts_away_last5": form_pts_away,
            "form_gd_home_last5": form_gd_home,
            "form_gd_away_last5": form_gd_away,
            "goals_scored_home_avg5": goals_home_avg5,
            "goals_scored_away_avg5": goals_away_avg5,
            "h2h_home_win_rate": h2h_home_win_rate,
            "competition_level": competition_level,
            "is_playoff": is_playoff,
            "is_women": is_women,
        }
