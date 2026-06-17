"""
Fetch and cache FIFA WC 2026 match data.

Source priority (each falls through to the next on failure):
  1. FIFA PMSR PDF   — official PDF from FIFA Training Centre; works on
                       GitHub Actions (no IP block); has score, lineups,
                       team xG, shot log (no per-shot x/y/xG)
  2. FBref           — full shot events + xG; blocked on GitHub Actions
                       cloud IPs but works locally
  3. StatsBomb       — open data; activates when they publish WC26
  4. Sofascore       — blocked from cloud IPs; last resort

Schedule: openfootball worldcup.json (GitHub raw)

All raw pulls are cached as Parquet / JSON.  Every network call is
preceded by a 3-second sleep to be polite to upstream servers.
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
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)

# FBref league label used by soccerdata
_FBREF_LEAGUE = "INT-World Cup"


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

def _flatten_matches(data: dict) -> list[dict]:
    """openfootball uses either a flat 'matches' list or nested 'rounds'."""
    if "matches" in data:
        return [(m.get("round"), m) for m in data["matches"]]
    out = []
    for rnd in data.get("rounds", []):
        for m in rnd.get("matches", []):
            out.append((rnd.get("name"), m))
    return out


def _team_name(value) -> Optional[str]:
    if isinstance(value, dict):
        return value.get("name")
    return value


def fetch_fixtures(force_refresh: bool = False) -> pd.DataFrame:
    """WC 2026 fixtures from openfootball (cached)."""
    key = "fixtures_wc26"
    if not force_refresh and (df := _load(key)) is not None:
        return df

    _sleep()
    resp = requests.get(OPENFOOTBALL_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for round_name, m in _flatten_matches(data):
        score = m.get("score") or {}
        ft = score.get("ft") if isinstance(score, dict) else None
        rows.append({
            "round": round_name,
            "date": m.get("date"),
            "time": m.get("time"),
            "team1": _team_name(m.get("team1")),
            "team2": _team_name(m.get("team2")),
            "score1": (ft[0] if ft else m.get("score1")),
            "score2": (ft[1] if ft else m.get("score2")),
            "group": (
                m.get("group", {}).get("name")
                if isinstance(m.get("group"), dict)
                else m.get("group")
            ),
            "ground": m.get("ground"),
        })

    df = pd.DataFrame(rows)
    _save(df, key)
    return df


# ── FBref helpers ─────────────────────────────────────────────────────────────

def _fbref_instance(season: int = 2026):
    """Return a configured soccerdata FBref scraper (import is lazy)."""
    import soccerdata as sd
    return sd.FBref(leagues=_FBREF_LEAGUE, seasons=season)


def _fbref_schedule(season: int = 2026, force_refresh: bool = False) -> pd.DataFrame:
    key = f"fbref_schedule_{season}"
    if not force_refresh and (df := _load(key)) is not None:
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
    if fbref_id is None and not (home and away):
        raise ValueError("Supply fbref_id OR both home and away team names")

    shots = player_stats = lineups = None
    score: Optional[dict] = None
    team_stats: dict = {}

    # ── source 1: FIFA PMSR PDF — always works from cloud runners ────────────
    if home and away:
        try:
            from wc26 import fifa_pdf  # noqa: PLC0415

            logger.info("Trying FIFA PMSR PDF for %s vs %s", home, away)
            pdf_data = fifa_pdf.fetch_match(home, away)
            if pdf_data is not None:
                logger.info("FIFA PDF data loaded")
                score = pdf_data["score"]
                team_stats = pdf_data.get("team_stats", {})
                if lineups is None or (isinstance(lineups, pd.DataFrame) and lineups.empty):
                    lineups = pdf_data.get("lineups")
                # shots from PDF have no x/y/xg — store separately; don't
                # block other sources from providing richer shot data
                if pdf_data.get("shots") is not None and not pdf_data["shots"].empty:
                    shots = pdf_data["shots"]  # may be overwritten below with richer data
        except Exception as exc:
            logger.warning("FIFA PDF source failed: %s", exc)

    # ── source 2: FBref (blocked on GitHub Actions; works locally) ───────────
    # Skip entirely when FIFA PDF already gave a valid score — avoids 4-minute
    # retry storm on cloud runners where FBref is Cloudflare-blocked.
    _fifa_has_data = score is not None and (score.get("home", 0) or score.get("away", 0)
                                             or (shots is not None and not shots.empty))
    if not _fifa_has_data:
        try:
            if fbref_id is None and home and away:
                fbref_id = _find_game_id(home, away, season)
                if fbref_id:
                    logger.info("Resolved FBref game_id: %s", fbref_id)
            if fbref_id:
                fbref_shots = _fetch_shots(fbref_id, season)
                if fbref_shots is not None and not fbref_shots.empty:
                    shots = fbref_shots  # richer: has x/y/xg
                fbref_stats = _fetch_player_stats(fbref_id, season)
                if fbref_stats is not None and not fbref_stats.empty:
                    player_stats = fbref_stats
                fbref_lineups = _fetch_lineups(fbref_id, season)
                if fbref_lineups is not None and not fbref_lineups.empty:
                    lineups = fbref_lineups
        except Exception as exc:
            logger.warning("FBref unavailable (%s)", exc)

    # ── source 3: StatsBomb open data (activates when WC26 published) ────────
    if home and away:
        _has_xg = (
            shots is not None
            and not shots.empty
            and "xg" in shots.columns
            and shots["xg"].notna().any()
        )
        if not _has_xg:
            try:
                from wc26 import statsbomb  # noqa: PLC0415

                sb = statsbomb.fetch_match(home, away)
                if sb is not None:
                    logger.info("Using StatsBomb open data")
                    if sb["shots"] is not None and not sb["shots"].empty:
                        shots = sb["shots"]
                    if lineups is None or (isinstance(lineups, pd.DataFrame) and lineups.empty):
                        lineups = sb["lineups"]
                    if player_stats is None or (isinstance(player_stats, pd.DataFrame) and player_stats.empty):
                        player_stats = sb["player_stats"]
                    if score is None or score == {"home": 0, "away": 0}:
                        score = sb["score"]
            except Exception as exc:
                logger.warning("StatsBomb fallback failed: %s", exc)

    # ── source 4: Sofascore (last resort; also blocked on cloud) ─────────────
    sofascore: dict = {}
    _has_xg2 = (
        shots is not None
        and not shots.empty
        and "xg" in shots.columns
        and shots["xg"].notna().any()
    )
    if not _has_xg2 and home and away:
        logger.info("Trying Sofascore fallback")
        sofascore = _sofascore_fallback(home, away)
        if "shots" in sofascore:
            shots = sofascore["shots"]

    # ── finalise score ────────────────────────────────────────────────────────
    if score is None or score == {"home": 0, "away": 0}:
        if shots is not None and not shots.empty:
            try:
                score = _extract_score(shots, home, away, fbref_id or "", season)
            except Exception:
                score = {"home": 0, "away": 0}
        else:
            score = {"home": 0, "away": 0}

    return {
        "meta": {
            "home": home,
            "away": away,
            "fbref_id": fbref_id,
            "season": season,
            "score": score,
            "team_stats": team_stats,
        },
        "shots": shots,
        "player_stats": player_stats,
        "lineups": lineups,
        "sofascore": sofascore,
    }
