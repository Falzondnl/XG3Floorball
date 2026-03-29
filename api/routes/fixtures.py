"""
Floorball fixture discovery via Optic Odds v3 API.

GET /api/v1/floorball/games          → upcoming fixtures (today)
GET /api/v1/floorball/games/upcoming → alias

Requires OPTIC_ODDS_API_KEY env var.  Returns [] gracefully when not set.
Optic Odds sport identifier: "floorball" (confirmed in live API audit).
"""
from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional

import httpx
import structlog
from fastapi import APIRouter, Query

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/floorball")

_OPTIC_BASE = "https://api.opticodds.com"
_SPORT = "floorball"


def _api_key() -> Optional[str]:
    return os.getenv("OPTIC_ODDS_API_KEY") or None


async def _fetch_fixtures_for_date(date_str: str) -> List[Dict[str, Any]]:
    key = _api_key()
    if not key:
        return []
    try:
        async with httpx.AsyncClient(
            base_url=_OPTIC_BASE,
            headers={"X-Api-Key": key, "Accept": "application/json"},
            timeout=httpx.Timeout(20.0, connect=8.0),
        ) as client:
            resp = await client.get(
                "/v3/fixtures",
                params={"sport": _SPORT, "date": date_str},
            )
            resp.raise_for_status()
            return resp.json().get("data", [])
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "floorball_fixtures_http_error",
            status=exc.response.status_code,
            date=date_str,
        )
        return []
    except Exception as exc:
        logger.warning("floorball_fixtures_error", date=date_str, error=str(exc))
        return []


@router.get("/games", response_model=List[Dict[str, Any]])
@router.get("/games/upcoming", response_model=List[Dict[str, Any]])
async def list_games(
    date_str: Optional[str] = Query(
        None,
        alias="date",
        description="ISO date YYYY-MM-DD (default: today)",
        example="2026-04-01",
    ),
    days: int = Query(
        1,
        ge=1,
        le=14,
        description="Number of days to fetch (default 1)",
    ),
) -> List[Dict[str, Any]]:
    """
    List upcoming floorball fixtures from Optic Odds.

    Returns empty list when OPTIC_ODDS_API_KEY is not configured.
    """
    if not _api_key():
        logger.debug("floorball_fixtures_disabled", reason="OPTIC_ODDS_API_KEY not set")
        return []

    start_date = date_str or str(datetime.date.today())

    results: List[Dict[str, Any]] = []
    base = datetime.date.fromisoformat(start_date)
    for i in range(days):
        day_str = str(base + datetime.timedelta(days=i))
        day_fixtures = await _fetch_fixtures_for_date(day_str)
        results.extend(day_fixtures)

    logger.info(
        "floorball_fixtures_fetched",
        start=start_date,
        days=days,
        count=len(results),
    )
    return results
