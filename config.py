"""
Floorball Microservice Configuration
XG3 Enterprise — Port 8037
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"

DATA_SOURCES = {
    "sofascore_events": Path(
        "D:/codex/Data/othersports/02_derived/floorball/sofascore_events.csv"
    ),
    "sofascore_event_details": Path(
        "D:/codex/Data/othersports/02_derived/floorball/sofascore_event_details.csv"
    ),
    "iff_rankings_men": Path(
        "D:/codex/Data/othersports/02_derived/floorball/iff_rankings_table_2.csv"
    ),
    "iff_rankings_women": Path(
        "D:/codex/Data/othersports/02_derived/floorball/iff_rankings_table_3.csv"
    ),
}

# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
SERVICE_NAME = "floorball"
SERVICE_VERSION = "1.0.0"
PORT = int(os.getenv("PORT", "8000"))  # Railway injects PORT; default 8000

# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
MIN_ROWS_FOR_ENSEMBLE = 500   # below this → logistic fallback

# Regime definitions
REGIME_MEN = "r0"
REGIME_WOMEN = "r1"

WOMEN_KEYWORDS = ("women", "Women", "woman", "female", "w ")

# Competition level weights (higher = more elite)
COMPETITION_LEVELS: dict[str, int] = {
    "world championship": 5,
    "world championship, women": 5,
    "euro floorball tour": 4,
    "champions cup": 4,
    "svenska superligan": 3,
    "f-liiga": 3,
    "eliteserien": 3,
    "national league": 3,
    "unihockey prime league": 3,
    "extraliga": 3,
    "superliga": 3,
    "allsvenskan": 2,
    "floorball league": 2,
    "ufl": 2,
}

# ELO
ELO_K_INITIAL = 32
ELO_K_SETTLED = 20
ELO_DEFAULT = 1500.0
ELO_HOME_ADVANTAGE = 50.0   # floorball has meaningful home advantage

# Pricing margins
WIN_DRAW_WIN_MARGIN = 0.08    # 8%
ASIAN_HANDICAP_MARGIN = 0.05  # 5%
TOTALS_MARGIN = 0.05          # 5%
TOTALS_LINE = 10.5            # typical total goals line
