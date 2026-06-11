"""
Build a Twitter / X thread as a Markdown file.

Each tweet is numbered, hard-capped at 280 characters.
Thread structure:
  1  Scoreline + headline stat
  2  xG comparison
  3–4  Auto-picked insights (xG efficiency, big chances, top performer, etc.)
  5  Shot volume / accuracy
  6  Thread footer (image list + data credit)
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


def _norm_shots(shots: pd.DataFrame) -> pd.DataFrame:
    df = shots.copy()
    xgc = _col(df, ["xg", "psxg", "expected_goals", "xGoal", "xg_shot"])
    df["_xg"] = pd.to_numeric(df[xgc], errors="coerce").fillna(0.05) if xgc else 0.05
    outc = _col(df, ["outcome", "result", "shot_outcome"])
    df["_goal"] = df[outc].astype(str).str.lower().str.contains("goal", na=False) if outc else False
    tc = _col(df, ["team", "squad", "home_team", "team_name"])
    df["_team"] = df[tc].astype(str) if tc else ""
    mc = _col(df, ["minute", "min", "time"])
    df["_minute"] = pd.to_numeric(df[mc], errors="coerce").fillna(0) if mc else 0.0
    pc = _col(df, ["player", "name", "player_name"])
    df["_player"] = df[pc].astype(str) if pc else ""
    return df


def _cap(text: str, limit: int = 278) -> str:
    """Trim to limit, appending '…' if truncated."""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


# ── insight generators ────────────────────────────────────────────────────────

def _insight_xg_efficiency(
    shots: pd.DataFrame,
    home: str,
    away: str,
    sc: dict,
) -> Optional[str]:
    """Who over/under-performed xG?"""
    h = shots[shots["_team"].str.lower().str.contains(home.lower(), na=False)]
    a = shots[shots["_team"].str.lower().str.contains(away.lower(), na=False)]
    h_xg = h["_xg"].sum()
    a_xg = a["_xg"].sum()
    h_g = sc.get("home", 0)
    a_g = sc.get("away", 0)

    h_diff = h_g - h_xg
    a_diff = a_g - a_xg

    if abs(h_diff) < 0.3 and abs(a_diff) < 0.3:
        return None

    if abs(h_diff) >= abs(a_diff):
        team, goals, xg, diff = home, h_g, h_xg, h_diff
    else:
        team, goals, xg, diff = away, a_g, a_xg, a_diff

    if diff > 0:
        return (
            f"Clinical night for {team}: {goals}g from {xg:.2f} xG "
            f"(+{diff:.2f} vs expectation). Every chance taken counted."
        )
    else:
        return (
            f"Costly miss for {team}: only {goals}g from {xg:.2f} xG "
            f"({diff:.2f} vs expectation). The xG says they deserved more."
        )


def _insight_big_chances(shots: pd.DataFrame, home: str, away: str) -> Optional[str]:
    """Teams' big chances (xG ≥ 0.30)."""
    BIG_XG = 0.30
    h = shots[
        shots["_team"].str.lower().str.contains(home.lower(), na=False)
        & (shots["_xg"] >= BIG_XG)
    ]
    a = shots[
        shots["_team"].str.lower().str.contains(away.lower(), na=False)
        & (shots["_xg"] >= BIG_XG)
    ]
    h_sc = h["_goal"].sum()
    a_sc = a["_goal"].sum()

    parts = []
    if len(h) > 0:
        parts.append(f"{home}: {int(h_sc)}/{len(h)} big chances converted")
    if len(a) > 0:
        parts.append(f"{away}: {int(a_sc)}/{len(a)} big chances converted")

    if not parts:
        return None
    return "Big chances (xG ≥0.30)  ·  " + "  |  ".join(parts)


def _insight_best_chance(shots: pd.DataFrame) -> Optional[str]:
    """Single best chance of the game."""
    if shots.empty:
        return None
    top = shots.loc[shots["_xg"].idxmax()]
    player = str(top["_player"]) if top["_player"] else "Unknown"
    team = str(top["_team"])
    xg = top["_xg"]
    minute = int(top["_minute"])
    result = "scored" if top["_goal"] else "didn't score"
    return (
        f"Best chance: {player} ({team}) had a {xg:.2f} xG opportunity "
        f"in the {minute}′ — {result}."
    )


def _insight_top_performer(
    player_stats: Optional[pd.DataFrame],
    home: str,
    away: str,
) -> Optional[str]:
    """Highest-rated / most creative outfield player."""
    if player_stats is None or player_stats.empty:
        return None

    name_c = _col(player_stats, ["player", "name", "player_name"])
    if name_c is None:
        return None

    # prefer key-pass or xG chain column; fall back to shots
    metric_c = _col(
        player_stats,
        ["key_passes", "sca", "gca", "xg_chain", "xg_assisted", "goals", "assists"],
    )
    team_c = _col(player_stats, ["team", "squad", "home_team", "team_name"])

    if metric_c is None:
        return None

    ps = player_stats.copy()
    ps["_metric"] = pd.to_numeric(ps[metric_c], errors="coerce").fillna(0)
    best = ps.loc[ps["_metric"].idxmax()]

    player = str(best[name_c])
    val = best["_metric"]
    team = str(best[team_c]) if team_c else "—"

    return (
        f"🌟 Standout: {player} ({team}) led the match "
        f"with {val:.0f} {metric_c.replace('_', ' ')} — the creative engine."
    )


def _insight_shot_accuracy(shots: pd.DataFrame, home: str, away: str) -> Optional[str]:
    """Shot accuracy comparison."""
    sot_c = _col(shots, ["shots_on_target", "on_target", "sot"])
    if sot_c is None:
        # infer from outcome
        shots = shots.copy()
        shots["_on_target"] = shots["_goal"] | shots.get("outcome", pd.Series(dtype=str)).astype(str).str.lower().str.contains("saved", na=False)
    else:
        shots = shots.copy()
        shots["_on_target"] = shots[sot_c].astype(bool)

    h = shots[shots["_team"].str.lower().str.contains(home.lower(), na=False)]
    a = shots[shots["_team"].str.lower().str.contains(away.lower(), na=False)]

    def pct(df: pd.DataFrame) -> str:
        total = len(df)
        sot = df["_on_target"].sum() if "_on_target" in df.columns else 0
        if total == 0:
            return "0/0"
        return f"{int(sot)}/{total}"

    return f"Shots on target  ·  {home}: {pct(h)}  |  {away}: {pct(a)}"


# ── main function ─────────────────────────────────────────────────────────────

def compose_thread(
    match: dict,
    viz_paths: list[Path],
    output_dir: Path,
    handle: str = "@anup4khandelwal",
) -> Path:
    """
    Write a numbered tweet thread to output_dir/<match_key>/thread.md.

    Parameters
    ----------
    match       Full match report dict from fetch.fetch_match_report()
    viz_paths   List of PNG paths already generated by viz.*
    output_dir  Root output directory
    handle      Twitter handle (including @)
    """
    meta = match.get("meta", {})
    home = meta.get("home", "Home")
    away = meta.get("away", "Away")
    sc = meta.get("score", {"home": 0, "away": 0})
    shots_raw = match.get("shots")
    player_stats = match.get("player_stats")

    h_xg = a_xg = 0.0
    shots_norm: Optional[pd.DataFrame] = None
    if shots_raw is not None and not shots_raw.empty:
        shots_norm = _norm_shots(shots_raw)
        h_shots = shots_norm[shots_norm["_team"].str.lower().str.contains(home.lower(), na=False)]
        a_shots = shots_norm[shots_norm["_team"].str.lower().str.contains(away.lower(), na=False)]
        h_xg = h_shots["_xg"].sum()
        a_xg = a_shots["_xg"].sum()

    tweets: list[str] = []

    # 1 — scoreline
    xg_winner = home if h_xg >= a_xg else away
    xg_gap = abs(h_xg - a_xg)
    tweet1 = _cap(
        f"🏆 FINAL | {home} {sc['home']}–{sc['away']} {away}\n\n"
        f"xG: {home} {h_xg:.2f} – {a_xg:.2f} {away}\n"
        f"{'Dominant' if xg_gap > 0.5 else 'Narrow'} xG edge: {xg_winner}\n\n"
        f"Full thread 🧵👇"
    )
    tweets.append(tweet1)

    # 2 — xG summary
    h_shots_n = len(h_shots) if shots_norm is not None else 0
    a_shots_n = len(a_shots) if shots_norm is not None else 0
    tweet2 = _cap(
        f"📊 xG breakdown\n"
        f"{home}: {h_xg:.2f} xG from {h_shots_n} shots\n"
        f"{away}: {a_xg:.2f} xG from {a_shots_n} shots\n\n"
        f"{'Both teams created chances but quality was key.'  if abs(h_xg - a_xg) < 0.4 else f'{xg_winner} were clearly the better side on the numbers.'}"
    )
    tweets.append(tweet2)

    # 3–4 — auto-picked insights
    insights: list[str] = []
    if shots_norm is not None:
        for fn in [
            lambda: _insight_xg_efficiency(shots_norm, home, away, sc),
            lambda: _insight_big_chances(shots_norm, home, away),
            lambda: _insight_best_chance(shots_norm),
            lambda: _insight_shot_accuracy(shots_norm, home, away),
        ]:
            ins = fn()
            if ins:
                insights.append(ins)
            if len(insights) == 3:
                break

    for i, ins in enumerate(insights[:3]):
        tweets.append(_cap(f"{'📈' if i == 0 else '🎯' if i == 1 else '⚡'} {ins}"))

    # top performer
    top_perf = _insight_top_performer(player_stats, home, away)
    if top_perf:
        tweets.append(_cap(top_perf))

    # footer
    png_names = [p.name for p in viz_paths if str(p).endswith(".png")]
    images_note = "  ".join(f"[{n}]" for n in png_names) if png_names else ""
    tweet_n = _cap(
        f"Graphics attached 📸\n{images_note}\n\n"
        f"Data: FBref / Opta  ·  {handle}\n"
        f"#FIFAWorldCup2026 #WC26 #{home.replace(' ', '')} #{away.replace(' ', '')}"
    )
    tweets.append(tweet_n)

    # write thread.md
    match_key = f"{home}_vs_{away}".replace(" ", "_")
    out_path = output_dir / match_key / "thread.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Thread: {home} {sc['home']}–{sc['away']} {away}",
        f"<!-- auto-generated by wc26/compose.py -->",
        "",
    ]
    for i, tweet in enumerate(tweets, 1):
        lines += [
            f"## Tweet {i}",
            "",
            tweet,
            "",
            f"*{len(tweet)} / 280 chars*",
            "",
            "---",
            "",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
