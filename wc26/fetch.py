"""
Fetch and cache FIFA WC 2026 match data.

Primary:  soccerdata / FBref — lineups, player stats, shots + xG
Fallback: ScraperFC / Sofascore — shot coords, player ratings
Schedule: openfootball worldcup.json (GitHub raw)

All raw pulls are cached as Parquet.  Every network call is preceded by a
3-second sleep to be polite to upstream servers.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
DELAY = 3.0

OPENFOOTBALL_URL = (
    "https://raw.githubusercontent.com/openfootball/world-cup/master/2026/worldcup.json"
)

# FBref league label used by soccerdata
_FBREF_LEAGUE = "FIFA World Cup"


# ── cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{key}.parquet"


def _load(key: str) -> Optional[pd.DataFrame]:
    p = _cache_path(key)
    return pd.read_parquet(p) if p.exists() else None


def _save(df: pd.DataFrame, key: str) -> None:
    df.to_parquet(_cache_path(key))


def _sleep() -> None:
    time.sleep(DELAY)


# ── openfootball fixtures ─────────────────────────────────────────────────────

def fetch_fixtures() -> pd.DataFrame:
    """WC 2026 fixtures from openfootball (cached)."""
    key = "fixtures_wc26"
    if (df := _load(key)) is not None:
        return df

    _sleep()
    resp = requests.get(OPENFOOTBALL_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for rnd in data.get("rounds", []):
        for m in rnd.get("matches", []):
            t1 = m.get("team1", {})
            t2 = m.get("team2", {})
            rows.append({
                "round": rnd.get("name"),
                "date": m.get("date"),
                "time": m.get("time"),
                "team1": t1.get("name") if isinstance(t1, dict) else t1,
                "team2": t2.get("name") if isinstance(t2, dict) else t2,
                "score1": m.get("score1"),
                "score2": m.get("score2"),
                "group": (
                    m.get("group", {}).get("name")
                    if isinstance(m.get("group"), dict)
                    else m.get("group")
                ),
            })

    df = pd.DataFrame(rows)
    _save(df, key)
    return df


# ── FBref helpers ─────────────────────────────────────────────────────────────

def _fbref_instance(season: int = 2026):
    """Return a configured soccerdata FBref scraper (import is lazy)."""
    import soccerdata as sd
    return sd.FBref(leagues=_FBREF_LEAGUE, seasons=season)


def _fbref_schedule(season: int = 2026) -> pd.DataFrame:
    key = f"fbref_schedule_{season}"
    if (df := _load(key)) is not None:
        return df

    _sleep()
    df = _fbref_instance(season).read_schedule().reset_index()
    _save(df, key)
    return df


def _find_game_id(home: str, away: str, season: int = 2026) -> Optional[str]:
    schedule = _fbref_schedule(season)
    for home_col, away_col in [("home_team", "away_team"), ("away_team", "home_team")]:
        if home_col not in schedule.columns:
            continue
        mask = schedule[home_col].str.contains(home, case=False, na=False) & \
               schedule[away_col].str.contains(away, case=False, na=False)
        hits = schedule[mask]
        if not hits.empty:
            return str(hits.iloc[0]["game_id"])
    return None


def _fetch_shots(fbref_id: str, season: int = 2026) -> Optional[pd.DataFrame]:
    key = f"shots_{fbref_id}"
    if (df := _load(key)) is not None:
        return df

    _sleep()
    try:
        df = _fbref_instance(season).read_shot_events(match_id=fbref_id).reset_index()
        _save(df, key)
        return df
    except Exception as exc:
        logger.warning("FBref shots failed for %s: %s", fbref_id, exc)
        return None


def _fetch_player_stats(fbref_id: str, season: int = 2026) -> Optional[pd.DataFrame]:
    key = f"player_stats_{fbref_id}"
    if (df := _load(key)) is not None:
        return df

    fbref = _fbref_instance(season)
    parts = []
    for stat_type in ("summary", "passing", "defense", "misc"):
        _sleep()
        try:
            part = fbref.read_player_match_stats(stat_type=stat_type, match_id=fbref_id)
            part = part.reset_index()
            part["stat_type"] = stat_type
            parts.append(part)
        except Exception as exc:
            logger.warning("FBref player stats (%s) failed: %s", stat_type, exc)

    if not parts:
        return None
    df = pd.concat(parts, axis=0, ignore_index=True)
    _save(df, key)
    return df


def _fetch_lineups(fbref_id: str, season: int = 2026) -> Optional[pd.DataFrame]:
    key = f"lineups_{fbref_id}"
    if (df := _load(key)) is not None:
        return df

    _sleep()
    try:
        df = _fbref_instance(season).read_lineup(match_id=fbref_id).reset_index()
        _save(df, key)
        return df
    except Exception as exc:
        logger.warning("FBref lineup failed: %s", exc)
        return None


def fetch_season_stats(season: int = 2026) -> Optional[pd.DataFrame]:
    """
    Season-wide FBref player stats used to compute per-position percentiles
    for the pizza chart.
    """
    key = f"season_player_stats_{season}"
    if (df := _load(key)) is not None:
        return df

    _sleep()
    try:
        fbref = _fbref_instance(season)
        dfs = []
        for stat_type in ("standard", "shooting", "passing", "defense"):
            _sleep()
            try:
                part = fbref.read_player_season_stats(stat_type=stat_type).reset_index()
                part["stat_type"] = stat_type
                dfs.append(part)
            except Exception as exc:
                logger.warning("Season stats (%s) failed: %s", stat_type, exc)
        if not dfs:
            return None
        df = pd.concat(dfs, axis=0, ignore_index=True)
        _save(df, key)
        return df
    except Exception as exc:
        logger.warning("fetch_season_stats failed: %s", exc)
        return None


# ── Sofascore fallback ────────────────────────────────────────────────────────

def _sofascore_fallback(home: str, away: str) -> dict:
    """
    Try to pull shot coords and player ratings from Sofascore via ScraperFC.
    Returns an empty dict if anything goes wrong — this is strictly optional.
    """
    result: dict = {}
    try:
        from ScraperFC import Sofascore  # noqa: PLC0415

        ss = Sofascore()
        _sleep()

        # ScraperFC ≥3 API: get_match() accepts team names or searches
        match_id = None
        try:
            active = ss.get_active_team_match(home)
            candidates = active if isinstance(active, list) else [active]
            for cand in candidates:
                h = str(cand.get("homeTeam", {}).get("name", ""))
                a = str(cand.get("awayTeam", {}).get("name", ""))
                if away.lower() in a.lower():
                    match_id = cand.get("id")
                    break
        except Exception as exc:
            logger.debug("Sofascore match lookup failed: %s", exc)

        if match_id is None:
            return result

        _sleep()
        try:
            shots = ss.scrape_shots(match_id)
            if shots is not None and not shots.empty:
                result["shots"] = shots
        except Exception as exc:
            logger.debug("Sofascore shots failed: %s", exc)

        _sleep()
        try:
            ratings = ss.scrape_player_stats(match_id)
            if ratings is not None and not ratings.empty:
                result["ratings"] = ratings
        except Exception as exc:
            logger.debug("Sofascore ratings failed: %s", exc)

    except ImportError:
        logger.debug("ScraperFC not installed; skipping Sofascore fallback")
    except Exception as exc:
        logger.warning("Sofascore fallback error: %s", exc)

    return result


# ── score extraction ──────────────────────────────────────────────────────────

def _extract_score(
    shots: Optional[pd.DataFrame],
    home: Optional[str],
    away: Optional[str],
    fbref_id: str,
    season: int,
) -> dict:
    """Derive final scoreline from shots DataFrame or schedule."""
    if shots is not None and not shots.empty:
        goal_col = next(
            (c for c in ["outcome", "result", "shot_outcome"] if c in shots.columns),
            None,
        )
        team_col = next(
            (c for c in ["team", "squad", "home_team"] if c in shots.columns),
            None,
        )
        if goal_col and team_col:
            goals = shots[shots[goal_col].str.lower().str.contains("goal", na=False)]
            if home:
                hg = int(goals[team_col].str.lower().str.contains(home.lower(), na=False).sum())
            else:
                # First unique team value is home
                teams = shots[team_col].dropna().unique()
                hg = int((goals[team_col] == teams[0]).sum()) if len(teams) else 0
            if away:
                ag = int(goals[team_col].str.lower().str.contains(away.lower(), na=False).sum())
            else:
                teams = shots[team_col].dropna().unique()
                ag = int((goals[team_col] == teams[1]).sum()) if len(teams) > 1 else 0
            return {"home": hg, "away": ag}

    try:
        schedule = _fbref_schedule(season)
        row = schedule[schedule["game_id"].astype(str) == str(fbref_id)]
        if not row.empty:
            return {
                "home": int(row.iloc[0].get("home_score") or 0),
                "away": int(row.iloc[0].get("away_score") or 0),
            }
    except Exception:
        pass

    return {"home": 0, "away": 0}


# ── public API ────────────────────────────────────────────────────────────────

def fetch_match_report(
    home: Optional[str] = None,
    away: Optional[str] = None,
    fbref_id: Optional[str] = None,
    season: int = 2026,
) -> dict:
    """
    Pull the full match report for a WC 2026 fixture.

    Pass ``fbref_id`` directly (the hash-string from FBref URLs) or provide
    ``home`` + ``away`` team names to look up the ID automatically.

    Returns a dict with keys:
        meta         – home, away, fbref_id, season, score
        shots        – DataFrame of shot events (may be None)
        player_stats – DataFrame of per-player match stats (may be None)
        lineups      – DataFrame of lineups (may be None)
        sofascore    – dict with optional "shots" / "ratings" DFs
    """
    if fbref_id is None:
        if not (home and away):
            raise ValueError("Supply fbref_id OR both home and away team names")
        fbref_id = _find_game_id(home, away, season)
        if fbref_id is None:
            raise LookupError(
                f"No FBref match found for '{home}' vs '{away}' in {season}. "
                "Check spelling or supply fbref_id directly."
            )
        logger.info("Resolved FBref game_id: %s", fbref_id)

    shots = _fetch_shots(fbref_id, season)
    player_stats = _fetch_player_stats(fbref_id, season)
    lineups = _fetch_lineups(fbref_id, season)

    sofascore: dict = {}
    has_coords = (
        shots is not None
        and not shots.empty
        and bool({"x", "y", "start_x", "location_x"} & set(shots.columns))
    )
    if not has_coords and home and away:
        logger.info("Shot coordinates missing from FBref — trying Sofascore fallback")
        sofascore = _sofascore_fallback(home, away)
        if "shots" in sofascore and (shots is None or shots.empty):
            shots = sofascore["shots"]

    score = _extract_score(shots, home, away, fbref_id, season)

    return {
        "meta": {
            "home": home,
            "away": away,
            "fbref_id": fbref_id,
            "season": season,
            "score": score,
        },
        "shots": shots,
        "player_stats": player_stats,
        "lineups": lineups,
        "sofascore": sofascore,
    }
