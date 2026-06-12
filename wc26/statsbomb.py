"""
StatsBomb open data source (raw GitHub JSON — works from any IP, including
GitHub Actions runners, because it's just raw.githubusercontent.com).

StatsBomb published free match data *during* the 2022 World Cup; this module
automatically activates if/when they do the same for WC 2026.  Until the
2026 season appears in competitions.json, fetch_match() returns None.

Pitch coords are converted from StatsBomb 120×80 to Opta 0–100 here so the
rest of the pipeline never needs to care.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import requests

from wc26.fetch import _load, _save, _sleep

logger = logging.getLogger(__name__)

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMP_WORLD_CUP = 43
SEASON_NAME = "2026"

_UA = {"User-Agent": "wc26-analytics (github.com/anup4khandelwal/wc26-analytics)"}


def _get_json(url: str) -> Optional[list | dict]:
    _sleep()
    try:
        resp = requests.get(url, headers=_UA, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("StatsBomb fetch failed for %s: %s", url, exc)
        return None


def find_season_id(season_name: str = SEASON_NAME) -> Optional[int]:
    """Season id for the World Cup season, or None if not published yet."""
    comps = _get_json(f"{SB_BASE}/competitions.json")
    if not comps:
        return None
    for c in comps:
        if c.get("competition_id") == COMP_WORLD_CUP and c.get("season_name") == season_name:
            return int(c["season_id"])
    logger.info("StatsBomb open data has no World Cup %s season yet", season_name)
    return None


def _team_matches(name: str, target: str) -> bool:
    name, target = name.lower(), target.lower()
    return name in target or target in name


def find_match(home: str, away: str, season_name: str = SEASON_NAME) -> Optional[dict]:
    season_id = find_season_id(season_name)
    if season_id is None:
        return None
    matches = _get_json(f"{SB_BASE}/matches/{COMP_WORLD_CUP}/{season_id}.json")
    if not matches:
        return None
    for m in matches:
        h = m.get("home_team", {}).get("home_team_name", "")
        a = m.get("away_team", {}).get("away_team_name", "")
        if (_team_matches(h, home) and _team_matches(a, away)) or \
           (_team_matches(h, away) and _team_matches(a, home)):
            return m
    return None


# ── StatsBomb position names → our abbreviations (initials, GK special) ───────

def _abbrev_position(position_name: str) -> str:
    name = str(position_name).strip()
    if not name or name.lower() == "goalkeeper":
        return "GK"
    return "".join(w[0] for w in name.split()).upper()


def _build_shots(events: list, mid: int) -> pd.DataFrame:
    rows = []
    for ev in events:
        if ev.get("type", {}).get("name") != "Shot":
            continue
        if ev.get("period") == 5:  # penalty shootout — not part of match xG
            continue
        shot = ev.get("shot", {})
        loc = ev.get("location") or [60, 40]
        rows.append({
            "x": loc[0] / 120 * 100,
            "y": loc[1] / 80 * 100,
            "xg": shot.get("statsbomb_xg", 0.0),
            "outcome": shot.get("outcome", {}).get("name", ""),
            "minute": ev.get("minute", 0),
            "player": ev.get("player", {}).get("name", ""),
            "team": ev.get("team", {}).get("name", ""),
            "match_id": mid,
        })
    return pd.DataFrame(rows)


def _build_lineups(lineups_raw: list) -> pd.DataFrame:
    rows = []
    for team in lineups_raw or []:
        team_name = team.get("team_name", "")
        for p in team.get("lineup", []):
            positions = p.get("positions") or [{}]
            pos_name = positions[0].get("position", "")
            rows.append({
                "player": p.get("player_name", ""),
                "team": team_name,
                "position": _abbrev_position(pos_name),
            })
    return pd.DataFrame(rows)


def _build_player_stats(events: list, lineups: pd.DataFrame) -> pd.DataFrame:
    stats: dict[str, dict] = {}

    def bump(player: str, team: str, key: str, amount: float = 1):
        if not player:
            return
        entry = stats.setdefault(player, {"player": player, "team": team})
        entry[key] = entry.get(key, 0) + amount

    for ev in events:
        etype = ev.get("type", {}).get("name", "")
        player = ev.get("player", {}).get("name", "")
        team = ev.get("team", {}).get("name", "")

        if etype == "Shot":
            shot = ev.get("shot", {})
            bump(player, team, "xg", shot.get("statsbomb_xg", 0.0))
            if shot.get("outcome", {}).get("name") == "Goal":
                bump(player, team, "goals")
        elif etype == "Pass":
            pas = ev.get("pass", {})
            bump(player, team, "passes")
            if "outcome" not in pas:  # StatsBomb omits outcome on completed passes
                bump(player, team, "passes_completed")
            if pas.get("goal_assist"):
                bump(player, team, "assists")
            if pas.get("shot_assist"):
                bump(player, team, "key_passes")
        elif etype == "Duel":
            if "Tackle" in ev.get("duel", {}).get("type", {}).get("name", ""):
                bump(player, team, "tackles")
        elif etype == "Interception":
            bump(player, team, "interceptions")
        elif etype == "Clearance":
            bump(player, team, "clearances")
        elif etype == "Block":
            bump(player, team, "blocked_shots")
        elif etype == "Pressure":
            bump(player, team, "pressures")

    df = pd.DataFrame(list(stats.values()))
    if df.empty:
        return df

    if "passes" in df.columns:
        completed = df.get("passes_completed", 0)
        df["passes_completed_pct"] = (completed / df["passes"] * 100).round(1)

    if not lineups.empty:
        df = df.merge(lineups[["player", "position"]], on="player", how="left")

    return df.fillna(0)


def fetch_match(home: str, away: str, season_name: str = SEASON_NAME) -> Optional[dict]:
    """
    Full match data from StatsBomb open data, or None if unavailable.

    Returns dict with keys: shots, lineups, player_stats (DataFrames) and
    score ({"home": int, "away": int} in the caller's home/away order).
    """
    cache_key = f"sb_{home}_{away}".replace(" ", "_").lower()
    if (cached := _load(f"{cache_key}_shots")) is not None:
        meta = _load(f"{cache_key}_meta")
        score = None
        if meta is not None and not meta.empty:
            score = {"home": int(meta.iloc[0]["home"]), "away": int(meta.iloc[0]["away"])}
        return {
            "shots": cached,
            "lineups": _load(f"{cache_key}_lineups"),
            "player_stats": _load(f"{cache_key}_stats"),
            "score": score,
        }

    m = find_match(home, away, season_name)
    if m is None:
        return None

    mid = m["match_id"]
    logger.info("StatsBomb match found: id=%s", mid)

    events = _get_json(f"{SB_BASE}/events/{mid}.json")
    if not events:
        return None
    lineups_raw = _get_json(f"{SB_BASE}/lineups/{mid}.json")

    shots = _build_shots(events, mid)
    lineups = _build_lineups(lineups_raw or [])
    player_stats = _build_player_stats(events, lineups)

    # score in the caller's home/away order
    sb_home = m.get("home_team", {}).get("home_team_name", "")
    if _team_matches(sb_home, home):
        score = {"home": int(m.get("home_score", 0)), "away": int(m.get("away_score", 0))}
    else:
        score = {"home": int(m.get("away_score", 0)), "away": int(m.get("home_score", 0))}

    if not shots.empty:
        _save(shots, f"{cache_key}_shots")
    if not lineups.empty:
        _save(lineups, f"{cache_key}_lineups")
    if not player_stats.empty:
        _save(player_stats, f"{cache_key}_stats")
    _save(pd.DataFrame([score]), f"{cache_key}_meta")

    return {
        "shots": shots,
        "lineups": lineups,
        "player_stats": player_stats,
        "score": score,
    }
