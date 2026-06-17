"""
Visualizations for WC 2026 match reports.

All plots use a dark #0e1117 theme, export at 1600×900 px (100 dpi),
and carry a "Data: FBref/Opta | <handle>" footer.

Public API
----------
shot_map(match, output_dir, handle)
xg_race(match, output_dir, handle)
player_pizza(player_name, match, output_dir, season_stats, handle)
pass_network(team, match, output_dir, handle)
defensive_actions(player_name, match, output_dir, handle)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mplsoccer import Pitch, VerticalPitch, PyPizza

logger = logging.getLogger(__name__)

# ── design tokens ─────────────────────────────────────────────────────────────
BG = "#0e1117"
PITCH_LINE = "#3d4046"
TEXT = "#e8e8e8"
MUTED = "#888888"
HOME_C = "#00d4aa"   # teal
AWAY_C = "#ff6b6b"   # coral
GOAL_C = "#ffd700"   # gold
GRID_C = "#2a2d35"

FIG_W, FIG_H = 16, 9
DPI = 100

# font sizes — large enough to read on mobile
T_TITLE = 26
T_SUB = 17
T_LABEL = 14
T_TICK = 12
T_FOOT = 11


# ── low-level helpers ─────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def _brand(fig: plt.Figure, handle: str) -> None:
    fig.text(
        0.5, 0.013,
        f"Data: FIFA PMSR / Sofascore  ·  {handle}",
        ha="center", va="bottom",
        fontsize=T_FOOT, color=MUTED,
        transform=fig.transFigure,
    )


def _team_names(match: dict) -> tuple[str, str]:
    meta = match.get("meta", {})
    return meta.get("home", "Home"), meta.get("away", "Away")


def _score(match: dict) -> dict:
    return match.get("meta", {}).get("score", {"home": 0, "away": 0})


def _match_key(match: dict) -> str:
    h, a = _team_names(match)
    return f"{h}_vs_{a}".replace(" ", "_")


# ── coordinate normalisation ──────────────────────────────────────────────────

_X_CANDIDATES = ["x", "start_x", "x_shot", "location_x", "pos_x"]
_Y_CANDIDATES = ["y", "start_y", "y_shot", "location_y", "pos_y"]
_XG_CANDIDATES = ["xg", "psxg", "expected_goals", "xGoal", "xg_shot"]
_OUTCOME_CANDIDATES = ["outcome", "result", "shot_outcome", "type"]
_TEAM_CANDIDATES = ["team", "squad", "home_team", "team_name"]
_MIN_CANDIDATES = ["minute", "min", "time", "event_time"]
_PLAYER_CANDIDATES = ["player", "name", "player_name"]


def _col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


def _norm_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with columns norm_x, norm_y (opta 0-100 space),
    norm_xg (float), is_goal (bool), team, minute.
    """
    df = shots.copy()

    # coordinates
    xc = _col(df, _X_CANDIDATES)
    yc = _col(df, _Y_CANDIDATES)
    if xc:
        df["norm_x"] = pd.to_numeric(df[xc], errors="coerce")
        if df["norm_x"].max(skipna=True) > 105:  # statsbomb 0-120 → opta 0-100
            df["norm_x"] = df["norm_x"] / 120 * 100
    else:
        df["norm_x"] = np.nan
    if yc:
        df["norm_y"] = pd.to_numeric(df[yc], errors="coerce")
        if df["norm_y"].max(skipna=True) > 81:   # statsbomb 0-80 → opta 0-100
            df["norm_y"] = df["norm_y"] / 80 * 100
    else:
        df["norm_y"] = np.nan

    # xG
    xgc = _col(df, _XG_CANDIDATES)
    df["norm_xg"] = pd.to_numeric(df[xgc], errors="coerce").fillna(0.05) if xgc else 0.05

    # is_goal
    outc = _col(df, _OUTCOME_CANDIDATES)
    if outc:
        df["is_goal"] = df[outc].astype(str).str.lower().str.contains("goal", na=False)
    else:
        df["is_goal"] = False

    # team
    tc = _col(df, _TEAM_CANDIDATES)
    df["_team"] = df[tc].astype(str) if tc else ""

    # minute
    mc = _col(df, _MIN_CANDIDATES)
    df["_minute"] = pd.to_numeric(df[mc], errors="coerce").fillna(0) if mc else 0.0

    return df


# ── formation-based position coordinates (for pass network / def actions) ─────

_POS_MAP = {
    "GK":  (5, 50),
    "RB":  (25, 82), "RCB": (22, 67), "CB":  (22, 50), "LCB": (22, 33), "LB": (25, 18),
    "RWB": (40, 88), "LWB": (40, 12),
    "RDM": (38, 68), "CDM": (38, 50), "LDM": (38, 32),
    "RM":  (50, 88), "RCM": (50, 66), "CM":  (50, 50), "LCM": (50, 34), "LM": (50, 12),
    "RAM": (65, 68), "CAM": (65, 50), "LAM": (65, 32),
    "RW":  (80, 85), "RF":  (76, 68), "CF":  (80, 50), "LF":  (76, 32), "LW": (80, 15),
    "SS":  (74, 38), "FW":  (80, 50),
}

_POS_GROUPS = {
    "GK":  ["GK"],
    "DEF": ["RB", "RCB", "CB", "LCB", "LB", "RWB", "LWB"],
    "MID": ["RDM", "CDM", "LDM", "RM", "RCM", "CM", "LCM", "LM", "RAM", "CAM", "LAM"],
    "FWD": ["RW", "RF", "CF", "LF", "LW", "SS", "FW"],
}


def _pos_to_xy(pos: str) -> tuple[float, float]:
    pos = str(pos).upper().strip()
    return _POS_MAP.get(pos, _POS_MAP.get(pos[:2], (50, 50)))


def _lineup_positions(lineups: Optional[pd.DataFrame], team: str) -> pd.DataFrame:
    """Return DataFrame with columns player, pos, x, y for a team.

    Assigns formation-aware (x, y) coordinates in opta space (0-100 × 0-100)
    so the network looks like a real formation rather than rigid vertical columns.
    Attack direction is left → right (x=0 GK goal, x=100 opponent goal).
    """
    if lineups is None or lineups.empty:
        return pd.DataFrame(columns=["player", "pos", "x", "y"])

    name_c = _col(lineups, ["player", "name", "player_name"])
    pos_c = _col(lineups, ["position", "pos", "formation_place"])
    team_c = _col(lineups, _TEAM_CANDIDATES)

    if name_c is None:
        return pd.DataFrame(columns=["player", "pos", "x", "y"])

    df = lineups.copy()
    if team_c:
        df = df[df[team_c].astype(str).str.lower().str.contains(team.lower(), na=False)]
    if df.empty:
        return pd.DataFrame(columns=["player", "pos", "x", "y"])

    def _broad(pos: str) -> str:
        p = str(pos).upper().strip()
        if p == "GK": return "GK"
        if p in ("DF", "RB", "LB", "CB", "RCB", "LCB", "RWB", "LWB"): return "DF"
        if p in ("MF", "CM", "CDM", "CAM", "RM", "LM", "RDM", "LDM",
                 "RCM", "LCM", "RAM", "LAM"): return "MF"
        return "FW"

    # Formation-aware (x, y) templates for each group size.
    # y is lateral position (15=right flank, 85=left flank, 50=centre).
    # x is depth (5=GK, 22=defensive line, 50=midfield, 78=attacking line).
    _TEMPLATES: dict[str, dict[int, list[tuple[float, float]]]] = {
        "GK": {1: [(5, 50)]},
        "DF": {
            2: [(23, 33), (23, 67)],
            3: [(23, 22), (22, 50), (23, 78)],
            4: [(24, 17), (21, 39), (21, 61), (24, 83)],
            5: [(28, 10), (24, 28), (21, 50), (24, 72), (28, 90)],
        },
        "MF": {
            1: [(50, 50)],
            2: [(45, 33), (45, 67)],
            3: [(42, 50), (55, 25), (55, 75)],       # DM + 2 wide
            4: [(40, 35), (40, 65), (57, 22), (57, 78)],
            5: [(42, 50), (50, 28), (50, 72), (62, 20), (62, 80)],
            6: [(40, 30), (40, 70), (52, 15), (52, 50), (52, 85), (62, 50)],
        },
        "FW": {
            1: [(80, 50)],
            2: [(78, 28), (78, 72)],
            3: [(75, 17), (82, 50), (75, 83)],
            4: [(73, 15), (80, 38), (80, 62), (73, 85)],
        },
    }

    groups: dict[str, list[str]] = {"GK": [], "DF": [], "MF": [], "FW": []}
    for _, row in df.iterrows():
        pos = str(row[pos_c]).upper().strip() if pos_c else "MF"
        groups[_broad(pos)].append(str(row[name_c]))

    rows: list[dict] = []
    for grp, players in groups.items():
        n = len(players)
        if n == 0:
            continue
        templates = _TEMPLATES.get(grp, {})
        # Find closest template size; fall back to evenly spread
        coords = templates.get(n)
        if coords is None:
            best = min(templates.keys(), key=lambda k: abs(k - n)) if templates else None
            if best:
                base = templates[best]
                # interpolate y positions to fit n players
                base_y = [c[1] for c in base]
                ys = np.linspace(min(base_y), max(base_y), n)
                x_avg = np.mean([c[0] for c in base])
                coords = [(x_avg, float(y)) for y in ys]
            else:
                x_avg = {"GK": 5, "DF": 22, "MF": 50, "FW": 78}.get(grp, 50)
                coords = [(x_avg, float(y)) for y in np.linspace(15, 85, n)]

        for player, (x_val, y_val) in zip(players, coords):
            rows.append({"player": player, "pos": grp, "x": float(x_val), "y": float(y_val)})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["player", "pos", "x", "y"])


# ── 1. shot map ───────────────────────────────────────────────────────────────

def _shot_outcomes_chart(
    match: dict, output_dir: Path, handle: str
) -> Path:
    """
    Fallback when shot x/y coordinates are unavailable (FIFA PDF source).
    Renders a grouped bar chart of shot outcomes + team xG side by side.
    """
    shots_raw = match.get("shots")
    home, away = _team_names(match)
    sc = _score(match)
    team_stats = match.get("meta", {}).get("team_stats", {})

    CATEGORIES = ["Goal", "On Target", "Off Target", "Blocked", "Incomplete"]

    def _count_outcomes(team_name: str) -> dict[str, int]:
        counts = {k: 0 for k in CATEGORIES}
        if shots_raw is None or shots_raw.empty:
            return counts
        outc_c = _col(shots_raw, _OUTCOME_CANDIDATES)
        team_c = _col(shots_raw, _TEAM_CANDIDATES)
        if outc_c is None:
            return counts
        ts = shots_raw
        if team_c is not None:
            ts = shots_raw[shots_raw[team_c].astype(str).str.lower()
                           .str.contains(team_name.lower(), na=False)]
        for _, row in ts.iterrows():
            outcome = str(row[outc_c]).lower()
            if "goal" in outcome:
                counts["Goal"] += 1
            elif "on target" in outcome or "saved" in outcome:
                counts["On Target"] += 1
            elif "off target" in outcome or "off" in outcome:
                counts["Off Target"] += 1
            elif "blocked" in outcome or "block" in outcome:
                counts["Blocked"] += 1
            elif "incomplete" in outcome or "deflected" in outcome:
                counts["Incomplete"] += 1
        return counts

    h_counts = _count_outcomes(home)
    a_counts = _count_outcomes(away)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.14, wspace=0.28)

    CAT_COLORS = {
        "Goal":       GOAL_C,
        "On Target":  HOME_C,
        "Off Target": AWAY_C,
        "Blocked":    "#8ecae6",
        "Incomplete": MUTED,
    }

    for ax, team, counts, color, total_shots_key, xg_key in [
        (axes[0], home, h_counts, HOME_C, "shots_home", "xg_home"),
        (axes[1], away, a_counts, AWAY_C, "shots_away", "xg_away"),
    ]:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(axis='x', colors=TEXT, labelsize=T_LABEL)
        ax.tick_params(axis='y', length=0, labelsize=T_LABEL, labelcolor=TEXT)

        bars = [counts[c] for c in CATEGORIES]
        bar_colors = [CAT_COLORS[c] for c in CATEGORIES]
        y_pos = range(len(CATEGORIES))

        rects = ax.barh(y_pos, bars, color=bar_colors, height=0.55,
                        edgecolor=BG, linewidth=1.5)
        for rect, val in zip(rects, bars):
            if val > 0:
                ax.text(val + 0.1, rect.get_y() + rect.get_height() / 2,
                        str(val), va="center", ha="left",
                        color=TEXT, fontsize=T_LABEL, fontweight="bold")

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(CATEGORIES, fontsize=T_LABEL, color=TEXT)
        ax.set_xlim(0, max(max(bars) + 2, 5))

        total = team_stats.get(total_shots_key, sum(bars))
        xg = team_stats.get(xg_key, 0.0)
        goals = sc["home"] if team == home else sc["away"]
        subtitle = f"{goals}g  ·  {total} shots  ·  xG {xg:.2f}"
        ax.set_title(team, color=color, fontsize=T_LABEL + 4, fontweight="bold", pad=8)
        ax.set_xlabel(subtitle, color=MUTED, fontsize=T_TICK)

    # colour legend at the bottom
    legend_handles = [
        mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CATEGORIES
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=len(CATEGORIES),
        facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT,
        fontsize=T_TICK, bbox_to_anchor=(0.5, 0.01),
    )

    title = f"{home}  {sc['home']}–{sc['away']}  {away}  —  Shots"
    fig.suptitle(title, color=TEXT, fontsize=T_TITLE, fontweight="bold", y=0.95)
    _brand(fig, handle)

    path = output_dir / _match_key(match) / "shot_map.png"
    return _save(fig, path)


def shot_map(match: dict, output_dir: Path, handle: str = "@anup4khandelwal") -> Path:
    """
    Two attacking half-pitches with shots sized by xG.
    Falls back to a shot-outcomes bar chart when coordinates are unavailable
    (e.g. FIFA PDF source, which has no x/y data).
    """
    shots_raw = match.get("shots")

    # Detect whether we have real x/y coordinates
    has_coords = False
    if shots_raw is not None and not shots_raw.empty:
        xc = _col(shots_raw, _X_CANDIDATES)
        if xc is not None:
            has_coords = shots_raw[xc].notna().any()

    if not has_coords:
        return _shot_outcomes_chart(match, output_dir, handle)

    home, away = _team_names(match)
    sc = _score(match)
    title = f"{home}  {sc['home']}–{sc['away']}  {away}"

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.06, wspace=0.05)

    for ax, team, color, label in [
        (axes[0], home, HOME_C, f"{home}  ({sc['home']}g)"),
        (axes[1], away, AWAY_C, f"{away}  ({sc['away']}g)"),
    ]:
        pitch = VerticalPitch(
            pitch_type="opta",
            half=True,
            pitch_color=BG,
            line_color=PITCH_LINE,
            line_zorder=2,
            linewidth=1.5,
        )
        pitch.draw(ax=ax)
        ax.set_facecolor(BG)

        if shots_raw is not None and not shots_raw.empty:
            shots = _norm_shots(shots_raw)
            team_shots = shots[shots["_team"].str.lower().str.contains(team.lower(), na=False)]
        else:
            team_shots = pd.DataFrame()

        if not team_shots.empty:
            regular = team_shots[~team_shots["is_goal"]]
            goals = team_shots[team_shots["is_goal"]]

            pitch.scatter(
                regular["norm_x"], regular["norm_y"],
                s=(regular["norm_xg"].clip(0.01, 1) * 600 + 50),
                c=color, alpha=0.55, edgecolors=BG, linewidths=0.8, zorder=5, ax=ax,
            )
            if not goals.empty:
                pitch.scatter(
                    goals["norm_x"], goals["norm_y"],
                    s=(goals["norm_xg"].clip(0.01, 1) * 900 + 200),
                    c=GOAL_C, marker="*", edgecolors=BG, linewidths=0.5, zorder=6, ax=ax,
                )
        else:
            ax.text(50, 75, "No shot data", ha="center", va="center",
                    fontsize=T_LABEL, color=MUTED)

        ax.set_title(label, color=color, fontsize=T_LABEL + 2, fontweight="bold", pad=8)

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HOME_C,
               markersize=10, label="Shot (home)", alpha=0.8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=AWAY_C,
               markersize=10, label="Shot (away)", alpha=0.8),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=GOAL_C,
               markersize=14, label="Goal"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#aaa",
               markersize=6, label="Dot size ∝ xG"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4,
        facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT,
        fontsize=T_TICK, bbox_to_anchor=(0.5, 0.03),
    )

    fig.suptitle(title, color=TEXT, fontsize=T_TITLE, fontweight="bold", y=0.96)
    _brand(fig, handle)

    path = output_dir / _match_key(match) / "shot_map.png"
    return _save(fig, path)


# ── 2. xG race ────────────────────────────────────────────────────────────────

def xg_race(match: dict, output_dir: Path, handle: str = "@anup4khandelwal") -> Path:
    """
    Cumulative xG step chart with goal markers.

    When per-shot xG is unavailable (FIFA PDF source), distributes the team's
    total xG evenly across their shots so the totals are still correct.
    """
    shots_raw = match.get("shots")
    home, away = _team_names(match)
    sc = _score(match)
    team_stats = match.get("meta", {}).get("team_stats", {})

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.10)

    if shots_raw is not None and not shots_raw.empty:
        shots = _norm_shots(shots_raw)

        # Detect whether we have real per-shot xG values
        # (norm_xg is all 0.05 default when no xG column was present)
        xgc = _col(shots_raw, _XG_CANDIDATES)
        has_real_xg = (xgc is not None and
                       shots_raw[xgc].notna().any() and
                       shots_raw[xgc].astype(float).sum() > 0)

        for team, color, label_suffix, ts_key in [
            (home, HOME_C, f"  {sc['home']}g", "xg_home"),
            (away, AWAY_C, f"  {sc['away']}g", "xg_away"),
        ]:
            ts = shots[shots["_team"].str.lower().str.contains(team.lower(), na=False)]
            ts = ts.sort_values("_minute").copy()

            if ts.empty:
                continue

            # When no real per-shot xG: scale uniform xG to match FIFA team total
            if not has_real_xg and ts_key in team_stats:
                total_xg = float(team_stats[ts_key])
                n = len(ts)
                ts["norm_xg"] = total_xg / n if n else 0.0

            minutes = [0.0] + ts["_minute"].tolist() + [95.0]
            cum_xg = [0.0] + ts["norm_xg"].cumsum().tolist()
            cum_xg.append(cum_xg[-1])

            ax.step(minutes, cum_xg, where="post", color=color, linewidth=2.5,
                    label=f"{team}{label_suffix}  ({cum_xg[-1]:.2f} xG)")

            # goal markers
            goals = ts[ts["is_goal"]]
            for _, row in goals.iterrows():
                prev = ts[ts["_minute"] <= row["_minute"]]["norm_xg"].sum()
                ax.axvline(row["_minute"], color=GOAL_C, linestyle="--",
                           linewidth=1, alpha=0.6)
                ax.text(row["_minute"] + 0.5, prev + 0.02,
                        "⚽", fontsize=13, color=GOAL_C, va="bottom")
    else:
        ax.text(0.5, 0.5, "No shot data available", ha="center", va="center",
                fontsize=T_LABEL, color=MUTED, transform=ax.transAxes)

    # half-time line
    ax.axvline(45, color=GRID_C, linestyle=":", linewidth=1.2)
    ax.text(45.5, ax.get_ylim()[1] * 0.97, "HT", color=MUTED,
            fontsize=T_TICK - 1, va="top")

    ax.set_xlabel("Minute", fontsize=T_LABEL, color=TEXT)
    ax.set_ylabel("Cumulative xG", fontsize=T_LABEL, color=TEXT)
    ax.tick_params(colors=TEXT, labelsize=T_TICK)
    ax.spines[:].set_color(GRID_C)
    ax.set_xlim(0, 96)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)

    ax.legend(facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT,
              fontsize=T_LABEL, loc="upper left")

    title = f"{home}  {sc['home']}–{sc['away']}  {away}  —  xG Race"
    fig.suptitle(title, color=TEXT, fontsize=T_TITLE, fontweight="bold", y=0.96)
    _brand(fig, handle)

    path = output_dir / _match_key(match) / "xg_race.png"
    return _save(fig, path)


# ── 3. player pizza ───────────────────────────────────────────────────────────

_PIZZA_METRICS: dict[str, list[str]] = {
    "FWD": [
        "goals", "xg_per90", "shots_on_target_pct", "assists",
        "xa_per90", "progressive_passes", "take_ons_completed",
        "touches_att_pen", "pressures_success_pct",
    ],
    "MID": [
        "passes_completed_pct", "progressive_passes", "key_passes",
        "xg_per90", "xa_per90", "tackles_won",
        "interceptions", "carries_progressive", "pressures",
    ],
    "DEF": [
        "tackles_won_pct", "interceptions", "clearances",
        "aerials_won_pct", "passes_completed_pct",
        "progressive_passes", "blocks", "errors_leading_to_shot",
        "pressures_success_pct",
    ],
    "GK": [
        "save_pct", "psxg_net_per90", "passes_completed_pct",
        "long_passes_completed_pct", "crosses_stopped_pct",
        "goal_kicks_launched_pct", "avg_goal_kick_length",
        "sweeper_actions", "psxg_per_shot_on_target",
    ],
}

_PIZZA_LABELS: dict[str, list[str]] = {
    "FWD": [
        "Goals", "xG p90", "SoT %", "Assists",
        "xA p90", "Prog Passes", "Take-ons",
        "Att-Pen Touches", "Press Succ %",
    ],
    "MID": [
        "Pass Cmp %", "Prog Passes", "Key Passes",
        "xG p90", "xA p90", "Tackles Won",
        "Interceptions", "Prog Carries", "Pressures",
    ],
    "DEF": [
        "Tackle Win %", "Interceptions", "Clearances",
        "Aerial Win %", "Pass Cmp %",
        "Prog Passes", "Blocks", "Errors→Shot",
        "Press Succ %",
    ],
    "GK": [
        "Save %", "PSxG+/- p90", "Pass Cmp %",
        "Long Pass %", "Cross Stop %",
        "GK Kick Launched %", "Kick Length",
        "Sweeper Actions", "PSxG/SoT",
    ],
}

_SLICE_COLORS = {
    "FWD": [HOME_C] * 4 + [AWAY_C] * 3 + ["#8ecae6"] * 2,
    "MID": ["#8ecae6"] * 3 + [HOME_C] * 2 + [AWAY_C] * 4,
    "DEF": [AWAY_C] * 4 + ["#8ecae6"] * 5,
    "GK":  [HOME_C] * 3 + [AWAY_C] * 3 + ["#8ecae6"] * 3,
}


def _player_group(pos: str) -> str:
    pos_up = str(pos).upper().strip()
    for group, positions in _POS_GROUPS.items():
        if pos_up in positions or pos_up.startswith(group[:2]):
            return group
    if "K" in pos_up:
        return "GK"
    if any(pos_up.startswith(p) for p in ["F", "S", "A"]):
        return "FWD"
    if any(pos_up.startswith(p) for p in ["D", "B"]):
        return "DEF"
    return "MID"


def _compute_percentiles(
    player_name: str,
    player_stats: pd.DataFrame,
    season_stats: Optional[pd.DataFrame],
    group: str,
) -> list[float]:
    """
    Compute 0-100 percentiles for pizza chart metrics.
    Falls back to 50 when a stat column is missing.
    """
    metrics = _PIZZA_METRICS.get(group, _PIZZA_METRICS["MID"])
    result = []

    pop: Optional[pd.DataFrame] = None
    if season_stats is not None and not season_stats.empty:
        pos_key = next(
            (c for c in ["position", "pos", "formation_place"] if c in season_stats.columns),
            None,
        )
        if pos_key:
            pop = season_stats[
                season_stats[pos_key].astype(str).str.upper().str[:3].isin(
                    {p[:3] for p in _POS_GROUPS.get(group, [])}
                )
            ]
        else:
            pop = season_stats

    name_c = _col(player_stats, _PLAYER_CANDIDATES)
    player_row = None
    if name_c:
        hits = player_stats[
            player_stats[name_c].astype(str).str.lower().str.contains(
                player_name.lower(), na=False
            )
        ]
        if not hits.empty:
            player_row = hits.iloc[0]

    for metric in metrics:
        if player_row is None or metric not in player_row.index:
            result.append(50.0)
            continue
        val = pd.to_numeric(player_row[metric], errors="coerce")
        if pd.isna(val):
            result.append(50.0)
            continue
        if pop is not None and metric in pop.columns:
            col_vals = pd.to_numeric(pop[metric], errors="coerce").dropna()
            if len(col_vals) > 1:
                pct = float((col_vals < val).mean() * 100)
                result.append(round(pct, 1))
                continue
        result.append(50.0)

    return result


def player_pizza(
    player_name: str,
    match: dict,
    output_dir: Path,
    season_stats: Optional[pd.DataFrame] = None,
    handle: str = "@anup4khandelwal",
) -> Path:
    """Percentile pizza chart vs all WC 2026 players in the same position."""
    player_stats = match.get("player_stats")
    lineups = match.get("lineups")
    home, away = _team_names(match)

    # determine position group
    pos = "MID"
    if lineups is not None and not lineups.empty:
        name_c = _col(lineups, _PLAYER_CANDIDATES)
        pos_c = _col(lineups, ["position", "pos"])
        if name_c and pos_c:
            hit = lineups[lineups[name_c].astype(str).str.lower().str.contains(
                player_name.lower(), na=False
            )]
            if not hit.empty:
                pos = str(hit.iloc[0][pos_c]).upper()
    group = _player_group(pos)

    if player_stats is not None:
        pcts = _compute_percentiles(player_name, player_stats, season_stats, group)
    else:
        pcts = [50.0] * len(_PIZZA_METRICS[group])

    labels = _PIZZA_LABELS.get(group, _PIZZA_LABELS["MID"])
    slice_colors = _SLICE_COLORS.get(group, [HOME_C] * len(labels))

    baker = PyPizza(
        params=labels[: len(pcts)],
        background_color=BG,
        straight_line_color=GRID_C,
        straight_line_lw=1,
        last_circle_color=GRID_C,
        last_circle_lw=2,
        other_circle_lw=0,
        inner_circle_size=20,
    )

    fig, ax = baker.make_pizza(
        pcts,
        figsize=(10, 10.5),
        color_blank_space="same",
        slice_colors=slice_colors[: len(pcts)],
        value_bck_colors=slice_colors[: len(pcts)],
        blank_alpha=0.35,
        kwargs_slices={
            "edgecolor": BG,
            "zorder": 2,
            "linewidth": 0.8,
        },
        kwargs_params={
            "color": TEXT,
            "fontsize": 12,
            "va": "center",
        },
        kwargs_values={
            "color": "#ffffff",
            "fontsize": 11,
            "zorder": 3,
            "bbox": {
                "edgecolor": BG,
                "facecolor": GRID_C,
                "boxstyle": "round,pad=0.2",
                "lw": 0.5,
            },
        },
    )

    fig.set_facecolor(BG)

    ax.set_title(
        f"{player_name}  ·  WC 2026 percentiles  [{group}]",
        color=TEXT, fontsize=T_SUB, fontweight="bold", pad=15,
    )
    _brand(fig, handle)

    safe_name = player_name.replace(" ", "_")
    path = output_dir / _match_key(match) / f"pizza_{safe_name}.png"
    return _save(fig, path)


# ── 4. pass network ───────────────────────────────────────────────────────────

def pass_network(
    team: str,
    match: dict,
    output_dir: Path,
    handle: str = "@anup4khandelwal",
) -> Path:
    """
    Approximate pass network using formation positions and per-player passing
    volume from FBref.  Node size ∝ passes attempted; edges connect neighbours
    in the formation.
    """
    lineups = match.get("lineups")
    player_stats = match.get("player_stats")
    home, away = _team_names(match)
    sc = _score(match)

    is_home = team.lower() in home.lower()
    color = HOME_C if is_home else AWAY_C

    pos_df = _lineup_positions(lineups, team)

    # enrich with passing volumes — FBref preferred, position-based fallback
    # Typical WC game estimates by position (creates meaningful size variation)
    _POS_PASS_DEFAULT = {"GK": 28, "DF": 52, "MF": 68, "FW": 38}
    pass_vol: dict[str, float] = {}
    if player_stats is not None and not player_stats.empty:
        pass_stats = player_stats[player_stats.get("stat_type", pd.Series([])) == "passing"] \
            if "stat_type" in player_stats.columns else player_stats
        name_c = _col(pass_stats, _PLAYER_CANDIDATES)
        att_c = next(
            (c for c in ["passes", "passes_attempted", "passes_total_distance"] if c in pass_stats.columns),
            None,
        )
        if name_c and att_c:
            for _, row in pass_stats.iterrows():
                nm = str(row[name_c])
                vol = pd.to_numeric(row[att_c], errors="coerce")
                if not pd.isna(vol):
                    pass_vol[nm] = float(vol)

    # Apply position-based fallback for players without FBref data
    if not pos_df.empty:
        for _, row in pos_df.iterrows():
            nm = str(row["player"])
            if nm not in pass_vol:
                pass_vol[nm] = float(_POS_PASS_DEFAULT.get(str(row.get("pos", "MF")), 45))

    pitch = Pitch(
        pitch_type="opta",
        pitch_color=BG,
        line_color=PITCH_LINE,
        line_zorder=2,
        linewidth=1.5,
    )
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor(BG)
    ax.set_facecolor(BG)

    if pos_df.empty:
        ax.text(50, 50, "No lineup data", ha="center", va="center",
                fontsize=T_LABEL, color=MUTED)
    else:
        # draw edges between positional neighbours (distance-based)
        positions = pos_df[["x", "y"]].values
        n = len(positions)
        edge_drawn: set = set()
        # Connect each player to their 3 nearest neighbours (always forms a network)
        from scipy.spatial import cKDTree  # noqa: PLC0415
        try:
            tree = cKDTree(positions)
            k = min(4, n)  # 3 neighbours + self
            dists, idxs = tree.query(positions, k=k)
            for i in range(n):
                for jj in range(1, k):
                    j = idxs[i][jj]
                    pair = (min(i, j), max(i, j))
                    if pair not in edge_drawn and dists[i][jj] < 50:
                        w = max(0.5, 2.5 - dists[i][jj] / 20)
                        ax.plot(
                            [positions[i][0], positions[j][0]],
                            [positions[i][1], positions[j][1]],
                            color=color, alpha=0.22, linewidth=w, zorder=3,
                        )
                        edge_drawn.add(pair)
        except ImportError:
            # fallback to distance threshold if scipy unavailable
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < 38:
                        ax.plot(
                            [positions[i][0], positions[j][0]],
                            [positions[i][1], positions[j][1]],
                            color=color, alpha=0.22, linewidth=1.5, zorder=3,
                        )
                        edge_drawn.add((i, j))

        # draw nodes
        for _, row in pos_df.iterrows():
            vol = pass_vol.get(str(row["player"]), 30.0)
            size = np.clip(vol * 4, 200, 2500)
            ax.scatter(
                row["x"], row["y"],
                s=size, c=color, edgecolors=BG, linewidths=2,
                zorder=5, alpha=0.9,
            )
            short_name = str(row["player"]).split()[-1]
            ax.text(
                row["x"], row["y"] - 5.5,
                short_name, ha="center", va="top",
                fontsize=T_TICK - 1, color=TEXT,
                fontweight="bold",
            )

    title = f"{team}  —  Pass Network  ({home} {sc['home']}–{sc['away']} {away})"
    fig.suptitle(title, color=TEXT, fontsize=T_TITLE, fontweight="bold", y=0.97)

    note = "Node size ∝ passes attempted  ·  Edges connect positional neighbours"
    fig.text(0.5, 0.055, note, ha="center", fontsize=T_TICK, color=MUTED)
    _brand(fig, handle)

    safe_team = team.replace(" ", "_")
    path = output_dir / _match_key(match) / f"pass_network_{safe_team}.png"
    return _save(fig, path)


# ── 5. match timeline ─────────────────────────────────────────────────────────

def match_timeline(
    match: dict,
    output_dir: Path,
    handle: str = "@anup4khandelwal",
) -> Path:
    """
    Horizontal timeline (0–95 min) with home (top) and away (bottom) lanes.
    Shots are semi-transparent circles; goals are gold stars with labels.
    """
    shots_raw = match.get("shots")
    home, away = _team_names(match)
    sc = _score(match)

    fig, ax = plt.subplots(figsize=(FIG_W, 5), facecolor=BG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.15)
    ax.set_facecolor(BG)

    # Lane centres: home=+1, away=-1
    LANE_HOME, LANE_AWAY = 1.0, -1.0

    if shots_raw is not None and not shots_raw.empty:
        shots = _norm_shots(shots_raw)
        for team_name, lane, color in [
            (home, LANE_HOME, HOME_C),
            (away, LANE_AWAY, AWAY_C),
        ]:
            ts = shots[shots["_team"].str.lower().str.contains(team_name.lower(), na=False)]
            if ts.empty:
                continue

            regular = ts[~ts["is_goal"]]
            goals = ts[ts["is_goal"]]

            # Plot shot circles
            ax.scatter(
                regular["_minute"],
                [lane] * len(regular),
                s=60, c=color, alpha=0.40, edgecolors="none", zorder=4,
            )

            # Plot goal stars
            for _, row in goals.iterrows():
                minute = int(row["_minute"])
                ax.scatter(
                    minute, lane,
                    s=280, c=GOAL_C, marker="*", edgecolors=BG,
                    linewidths=0.8, zorder=6,
                )
                # Get last name
                player_str = str(row.get("player", row.get("_player", "")))
                lastname = player_str.split()[-1] if player_str.strip() else ""
                label = f"⚽ {minute}' {lastname}"
                vert_offset = 0.22 if lane > 0 else -0.22
                va = "bottom" if lane > 0 else "top"
                ax.text(
                    minute, lane + vert_offset, label,
                    ha="center", va=va, fontsize=9, color=GOAL_C,
                    fontweight="bold", zorder=7,
                )

    # HT and ET lines
    ax.axvline(45, color=GRID_C, linestyle="--", linewidth=1.5, zorder=2)
    ax.text(45.5, 1.55, "HT", color=MUTED, fontsize=T_TICK - 1, va="center")
    ax.axvline(90, color=GRID_C, linestyle=":", linewidth=1.0, zorder=2, alpha=0.6)
    ax.text(90.5, 1.55, "ET", color=MUTED, fontsize=T_TICK - 1, va="center", alpha=0.6)

    # Lane labels
    ax.text(-1, LANE_HOME, home, ha="right", va="center",
            fontsize=T_TICK, color=HOME_C, fontweight="bold")
    ax.text(-1, LANE_AWAY, away, ha="right", va="center",
            fontsize=T_TICK, color=AWAY_C, fontweight="bold")

    # Horizontal lane lines
    ax.axhline(LANE_HOME, color=HOME_C, linewidth=0.6, alpha=0.3, zorder=1)
    ax.axhline(LANE_AWAY, color=AWAY_C, linewidth=0.6, alpha=0.3, zorder=1)

    ax.set_xlim(-2, 97)
    ax.set_ylim(-1.8, 2.0)
    ax.set_xlabel("Minute", fontsize=T_LABEL, color=TEXT)
    ax.set_xticks(range(0, 96, 15))
    ax.tick_params(colors=TEXT, labelsize=T_TICK)
    ax.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(GRID_C)

    title = f"{home}  {sc['home']}–{sc['away']}  {away}  —  Match Timeline"
    fig.suptitle(title, color=TEXT, fontsize=T_TITLE, fontweight="bold", y=0.96)
    _brand(fig, handle)

    path = output_dir / _match_key(match) / "match_timeline.png"
    return _save(fig, path)


# ── 6. shot creation ──────────────────────────────────────────────────────────

def shot_creation(
    match: dict,
    output_dir: Path,
    handle: str = "@anup4khandelwal",
) -> Path:
    """
    Two side-by-side horizontal bar charts: delivery_type and body_part
    breakdowns for home and away teams.
    """
    shots_raw = match.get("shots")
    home, away = _team_names(match)
    sc = _score(match)

    DELIVERY_CATS = ["Pass", "Cross", "Corner", "Free Kick", "Dribble", "Set Piece", "Other"]
    BODY_CATS = ["Right Foot", "Left Foot", "Head", "Other"]

    def _normalise_delivery(val: str) -> str:
        v = str(val).strip().lower()
        if "corner" in v:
            return "Corner"
        if "cross" in v:
            return "Cross"
        if "free" in v or "freekick" in v:
            return "Free Kick"
        if "dribble" in v or "carry" in v:
            return "Dribble"
        if "set" in v and "piece" in v:
            return "Set Piece"
        if "pass" in v:
            return "Pass"
        return "Other"

    def _normalise_body(val: str) -> str:
        v = str(val).strip().lower()
        if "right" in v and ("foot" in v or "rf" in v):
            return "Right Foot"
        if "left" in v and ("foot" in v or "lf" in v):
            return "Left Foot"
        if "head" in v:
            return "Head"
        return "Other"

    def _count_for_team(team_name: str):
        del_counts = {k: 0 for k in DELIVERY_CATS}
        body_counts = {k: 0 for k in BODY_CATS}
        if shots_raw is None or shots_raw.empty:
            return del_counts, body_counts
        tc = _col(shots_raw, _TEAM_CANDIDATES)
        del_c = next((c for c in ["delivery_type", "technique", "shot_technique"]
                      if c in shots_raw.columns), None)
        body_c = next((c for c in ["body_part", "shot_body_part", "bodypart"]
                       if c in shots_raw.columns), None)
        ts = shots_raw
        if tc:
            ts = shots_raw[shots_raw[tc].astype(str).str.lower().str.contains(
                team_name.lower(), na=False)]
        if del_c:
            for val in ts[del_c].dropna():
                cat = _normalise_delivery(str(val))
                del_counts[cat] = del_counts.get(cat, 0) + 1
        if body_c:
            for val in ts[body_c].dropna():
                cat = _normalise_body(str(val))
                body_counts[cat] = body_counts.get(cat, 0) + 1
        return del_counts, body_counts

    h_del, h_body = _count_for_team(home)
    a_del, a_body = _count_for_team(away)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.10, wspace=0.38)

    panel_data = [
        (axes[0], "Delivery Type", DELIVERY_CATS, h_del, a_del),
        (axes[1], "Body Part", BODY_CATS, h_body, a_body),
    ]

    for ax, panel_title, cats, h_counts, a_counts in panel_data:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=TEXT, labelsize=T_TICK)

        y_pos = np.arange(len(cats))
        bar_h = 0.35

        h_vals = [h_counts[c] for c in cats]
        a_vals = [a_counts[c] for c in cats]

        ax.barh(y_pos + bar_h / 2, h_vals, height=bar_h,
                color=HOME_C, alpha=0.85, label=home, edgecolor=BG)
        ax.barh(y_pos - bar_h / 2, a_vals, height=bar_h,
                color=AWAY_C, alpha=0.85, label=away, edgecolor=BG)

        for i, (hv, av) in enumerate(zip(h_vals, a_vals)):
            if hv > 0:
                ax.text(hv + 0.05, i + bar_h / 2, str(hv),
                        va="center", ha="left", fontsize=T_TICK, color=TEXT)
            if av > 0:
                ax.text(av + 0.05, i - bar_h / 2, str(av),
                        va="center", ha="left", fontsize=T_TICK, color=TEXT)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(cats, fontsize=T_TICK, color=TEXT)
        ax.set_title(panel_title, color=TEXT, fontsize=T_LABEL + 2,
                     fontweight="bold", pad=8)
        ax.xaxis.grid(True, color=GRID_C, linewidth=0.8)
        ax.set_axisbelow(True)

    legend_elements = [
        mpatches.Patch(facecolor=HOME_C, label=home),
        mpatches.Patch(facecolor=AWAY_C, label=away),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT,
               fontsize=T_TICK, bbox_to_anchor=(0.5, 0.02))

    title = f"{home}  {sc['home']}–{sc['away']}  {away}  —  Shot Creation"
    fig.suptitle(title, color=TEXT, fontsize=T_TITLE, fontweight="bold", y=0.95)
    _brand(fig, handle)

    path = output_dir / _match_key(match) / "shot_creation.png"
    return _save(fig, path)


# ── 7. first / second half ────────────────────────────────────────────────────

def first_second_half(
    match: dict,
    output_dir: Path,
    handle: str = "@anup4khandelwal",
) -> Path:
    """
    2x2 grid comparing H1 (<=45 min) vs H2 (>45 min) for both teams:
    Shots, xG, Goals, On-Target.
    """
    shots_raw = match.get("shots")
    home, away = _team_names(match)
    sc = _score(match)
    team_stats = match.get("meta", {}).get("team_stats", {})

    METRICS = ["Shots", "xG", "Goals", "On Target"]

    def _half_stats(team_name: str) -> dict:
        result = {m: {"H1": 0.0, "H2": 0.0} for m in METRICS}
        if shots_raw is None or shots_raw.empty:
            return result
        shots = _norm_shots(shots_raw)
        tc = _col(shots, _TEAM_CANDIDATES)
        ts = shots
        if tc:
            ts = shots[shots[tc].astype(str).str.lower().str.contains(
                team_name.lower(), na=False)]
        if ts.empty:
            return result

        h1 = ts[ts["_minute"] <= 45]
        h2 = ts[ts["_minute"] > 45]

        result["Shots"]["H1"] = float(len(h1))
        result["Shots"]["H2"] = float(len(h2))

        # Scale xG to FIFA totals if available
        xg_key = "xg_home" if team_name.lower() in home.lower() else "xg_away"
        total_xg_fifa = team_stats.get(xg_key)
        raw_h1_xg = float(h1["norm_xg"].sum())
        raw_h2_xg = float(h2["norm_xg"].sum())
        if total_xg_fifa is not None and (raw_h1_xg + raw_h2_xg) > 0:
            scale = float(total_xg_fifa) / (raw_h1_xg + raw_h2_xg)
            result["xG"]["H1"] = round(raw_h1_xg * scale, 2)
            result["xG"]["H2"] = round(raw_h2_xg * scale, 2)
        else:
            result["xG"]["H1"] = round(raw_h1_xg, 2)
            result["xG"]["H2"] = round(raw_h2_xg, 2)

        result["Goals"]["H1"] = float(h1["is_goal"].sum())
        result["Goals"]["H2"] = float(h2["is_goal"].sum())

        outc_c = _col(ts, _OUTCOME_CANDIDATES)
        if outc_c:
            on_tgt_h1 = h1[
                h1[outc_c].astype(str).str.lower().str.contains(
                    "on target|saved|goal", na=False)
            ]
            on_tgt_h2 = h2[
                h2[outc_c].astype(str).str.lower().str.contains(
                    "on target|saved|goal", na=False)
            ]
            result["On Target"]["H1"] = float(len(on_tgt_h1))
            result["On Target"]["H2"] = float(len(on_tgt_h2))
        return result

    h_stats = _half_stats(home)
    a_stats = _half_stats(away)

    # Lighter shade via reduced alpha (encode as color with transparency)
    HOME_C2 = HOME_C + "88"
    AWAY_C2 = AWAY_C + "88"

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.12,
                        hspace=0.45, wspace=0.30)

    GROUP_LABELS = ["H1 Home", "H1 Away", "H2 Home", "H2 Away"]
    BAR_COLORS = [HOME_C, AWAY_C, HOME_C2, AWAY_C2]

    for idx, metric in enumerate(METRICS):
        ax = axes[idx // 2][idx % 2]
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(GRID_C)
        ax.tick_params(colors=TEXT, labelsize=T_TICK)

        vals = [
            h_stats[metric]["H1"],
            a_stats[metric]["H1"],
            h_stats[metric]["H2"],
            a_stats[metric]["H2"],
        ]
        x_pos = np.arange(len(GROUP_LABELS))
        bars = ax.bar(x_pos, vals, color=BAR_COLORS, edgecolor=BG,
                      width=0.55, zorder=3)
        ax.yaxis.grid(True, color=GRID_C, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(GROUP_LABELS, fontsize=T_TICK - 1, color=TEXT,
                           rotation=20, ha="right")
        ax.set_title(metric, color=TEXT, fontsize=T_LABEL, fontweight="bold", pad=6)

        fmt = ".2f" if metric == "xG" else ".0f"
        for bar, val in zip(bars, vals):
            if val > 0:
                label_str = f"{val:{fmt}}"
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                        label_str, ha="center", va="bottom",
                        fontsize=T_TICK, color=TEXT, fontweight="bold")

    title = f"{home}  {sc['home']}–{sc['away']}  {away}  —  First vs Second Half"
    fig.suptitle(title, color=TEXT, fontsize=T_TITLE, fontweight="bold", y=0.95)
    _brand(fig, handle)

    path = output_dir / _match_key(match) / "first_second_half.png"
    return _save(fig, path)


# ── 8. shot conversion table ──────────────────────────────────────────────────

def shot_conversion_table(
    match: dict,
    output_dir: Path,
    handle: str = "@anup4khandelwal",
) -> Path:
    """
    Matplotlib table of per-player shot stats, max 14 rows.
    """
    shots_raw = match.get("shots")
    home, away = _team_names(match)
    sc = _score(match)

    COL_HEADERS = ["Player", "Team", "Shots", "Goals", "On Tgt", "Blocked", "Off Tgt"]
    MAX_ROWS = 14

    rows: list = []

    if shots_raw is not None and not shots_raw.empty:
        pc = _col(shots_raw, _PLAYER_CANDIDATES)
        tc = _col(shots_raw, _TEAM_CANDIDATES)
        outc = _col(shots_raw, _OUTCOME_CANDIDATES)

        if pc is not None:
            for player, grp in shots_raw.groupby(pc):
                if grp.empty:
                    continue
                team_val = str(grp.iloc[0][tc]) if tc else ""
                shots_total = len(grp)
                goals = 0
                on_tgt = 0
                blocked = 0
                off_tgt = 0
                if outc:
                    for outcome_val in grp[outc].astype(str).str.lower():
                        if "goal" in outcome_val:
                            goals += 1
                            on_tgt += 1
                        elif "on target" in outcome_val or "saved" in outcome_val:
                            on_tgt += 1
                        elif "blocked" in outcome_val or "block" in outcome_val:
                            blocked += 1
                        elif ("off" in outcome_val or "wide" in outcome_val
                              or "high" in outcome_val):
                            off_tgt += 1
                rows.append([str(player), team_val, shots_total, goals,
                              on_tgt, blocked, off_tgt])

    # Sort by shots descending
    rows.sort(key=lambda r: r[2], reverse=True)
    rows = rows[:MAX_ROWS]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.08)

    if not rows:
        ax.text(0.5, 0.5, "No shot data available", ha="center", va="center",
                fontsize=T_LABEL, color=MUTED, transform=ax.transAxes)
    else:
        n_rows = len(rows)
        n_cols = len(COL_HEADERS)
        cell_colors = []
        for i, row in enumerate(rows):
            row_team = row[1]
            base_color = "#161b26" if i % 2 == 0 else "#1e2535"
            row_colors = [base_color] * n_cols
            if home.lower() in row_team.lower():
                row_colors[0] = "#003d30"
            elif away.lower() in row_team.lower():
                row_colors[0] = "#3d1010"
            cell_colors.append(row_colors)

        # Display rows: shorten player name if too long
        display_rows = []
        for row in rows:
            player_short = str(row[0])
            if len(player_short) > 22:
                player_short = player_short[:20] + "…"
            display_rows.append([player_short] + row[1:])

        table = ax.table(
            cellText=display_rows,
            colLabels=COL_HEADERS,
            cellColours=cell_colors,
            colColours=[GOAL_C] * n_cols,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(T_TICK)
        table.scale(1.0, 1.8)

        # Style header cells
        for col_idx in range(n_cols):
            cell = table[(0, col_idx)]
            cell.set_text_props(color=BG, fontweight="bold", fontsize=T_TICK)
            cell.set_facecolor(GOAL_C)

        # Style data cells
        for row_idx in range(1, n_rows + 1):
            for col_idx in range(n_cols):
                cell = table[(row_idx, col_idx)]
                cell.set_text_props(color=TEXT, fontsize=T_TICK - 1)

    title = f"{home}  {sc['home']}–{sc['away']}  {away}  —  Shot Conversion"
    fig.suptitle(title, color=TEXT, fontsize=T_TITLE, fontweight="bold", y=0.95)
    _brand(fig, handle)

    path = output_dir / _match_key(match) / "shot_conversion_table.png"
    return _save(fig, path)


# ── 9. player ratings card ────────────────────────────────────────────────────

def player_ratings_card(
    match: dict,
    output_dir: Path,
    handle: str = "@anup4khandelwal",
) -> Path:
    """
    Top 5 performers per team — ratings bars if Sofascore available, else shots.
    """
    home, away = _team_names(match)
    sc = _score(match)
    shots_raw = match.get("shots")
    sofascore = match.get("sofascore", {})

    # Try to get Sofascore ratings
    ratings_df = sofascore.get("ratings") if isinstance(sofascore, dict) else None
    use_ratings = False
    rating_col = None
    if (ratings_df is not None and isinstance(ratings_df, pd.DataFrame)
            and not ratings_df.empty):
        rating_col = next(
            (c for c in ["rating", "Rating", "sofascore_rating"]
             if c in ratings_df.columns),
            None,
        )
        if rating_col is not None:
            use_ratings = True

    def _top5_ratings(team_name: str) -> list:
        tc = _col(ratings_df, _TEAM_CANDIDATES)
        pc = _col(ratings_df, _PLAYER_CANDIDATES)
        if pc is None:
            return []
        df = ratings_df
        if tc:
            df = ratings_df[ratings_df[tc].astype(str).str.lower().str.contains(
                team_name.lower(), na=False)]
        if df.empty:
            return []
        df = df.copy()
        df["_rating_num"] = pd.to_numeric(df[rating_col], errors="coerce")
        df = df.dropna(subset=["_rating_num"]).sort_values("_rating_num", ascending=False)
        top = df.head(5)
        return [(str(row[pc]), float(row["_rating_num"])) for _, row in top.iterrows()]

    def _top5_shots(team_name: str) -> list:
        if shots_raw is None or shots_raw.empty:
            return []
        pc = _col(shots_raw, _PLAYER_CANDIDATES)
        tc = _col(shots_raw, _TEAM_CANDIDATES)
        if pc is None:
            return []
        df = shots_raw
        if tc:
            df = shots_raw[shots_raw[tc].astype(str).str.lower().str.contains(
                team_name.lower(), na=False)]
        if df.empty:
            return []
        counts = df.groupby(pc).size().sort_values(ascending=False).head(5)
        return [(str(player), float(cnt)) for player, cnt in counts.items()]

    if use_ratings:
        h_top = _top5_ratings(home)
        a_top = _top5_ratings(away)
        bar_label = "Rating (0-10)"
        x_max = 10.0
        metric_label = "Sofascore Rating"
    else:
        h_top = _top5_shots(home)
        a_top = _top5_shots(away)
        all_vals = [v for _, v in h_top + a_top]
        x_max = float(max(all_vals)) + 1 if all_vals else 5.0
        bar_label = "Shots"
        metric_label = "Shots"

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.10, wspace=0.40)

    for ax, team_name, top5, color in [
        (axes[0], home, h_top, HOME_C),
        (axes[1], away, a_top, AWAY_C),
    ]:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=TEXT, labelsize=T_TICK)

        if not top5:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=T_LABEL, color=MUTED, transform=ax.transAxes)
        else:
            players = [p for p, _ in top5]
            vals = [v for _, v in top5]
            short_names = [p.split()[-1] if len(p) > 15 else p for p in players]
            y_pos = np.arange(len(players))

            bars = ax.barh(y_pos, vals, color=color, alpha=0.85,
                           edgecolor=BG, height=0.55, zorder=3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(short_names, fontsize=T_TICK, color=TEXT)
            ax.set_xlim(0, x_max)
            ax.xaxis.grid(True, color=GRID_C, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_xlabel(bar_label, fontsize=T_TICK, color=MUTED)
            ax.invert_yaxis()

            fmt = ".1f" if use_ratings else ".0f"
            for bar, val in zip(bars, vals):
                ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                        f"{val:{fmt}}", va="center", ha="left",
                        fontsize=T_TICK, color=TEXT, fontweight="bold")

        ax.set_title(team_name, color=color, fontsize=T_LABEL + 4,
                     fontweight="bold", pad=8)

    title = (f"{home}  {sc['home']}–{sc['away']}  {away}"
             f"  —  Top Performers ({metric_label})")
    fig.suptitle(title, color=TEXT, fontsize=T_TITLE, fontweight="bold", y=0.95)
    _brand(fig, handle)

    path = output_dir / _match_key(match) / "player_ratings_card.png"
    return _save(fig, path)


# ── 10. momentum chart ────────────────────────────────────────────────────────

def momentum_chart(
    match: dict,
    output_dir: Path,
    handle: str = "@anup4khandelwal",
) -> Path:
    """
    Rolling 5-minute window shot differential + cumulative xG lines.
    """
    shots_raw = match.get("shots")
    home, away = _team_names(match)
    sc = _score(match)
    team_stats = match.get("meta", {}).get("team_stats", {})

    WINDOW = 5
    bins = list(range(0, 96, WINDOW))  # 0,5,10,...,90
    bin_labels = [b + WINDOW / 2 for b in bins]  # centre of each bin

    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax2 = ax1.twinx()

    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.88, bottom=0.10)

    if shots_raw is not None and not shots_raw.empty:
        shots = _norm_shots(shots_raw)

        xgc = _col(shots_raw, _XG_CANDIDATES)
        has_real_xg = (xgc is not None and
                       shots_raw[xgc].notna().any() and
                       shots_raw[xgc].astype(float).sum() > 0)

        home_shots = shots[shots["_team"].str.lower().str.contains(
            home.lower(), na=False)].copy()
        away_shots = shots[shots["_team"].str.lower().str.contains(
            away.lower(), na=False)].copy()

        # Scale xG to FIFA totals if needed
        if not has_real_xg:
            for ts, key in [(home_shots, "xg_home"), (away_shots, "xg_away")]:
                if key in team_stats and len(ts) > 0:
                    total_xg = float(team_stats[key])
                    ts["norm_xg"] = total_xg / len(ts)

        # Shot differential per window
        h_counts = np.zeros(len(bins))
        a_counts = np.zeros(len(bins))
        for i, b in enumerate(bins):
            h_counts[i] = float(((home_shots["_minute"] >= b)
                                  & (home_shots["_minute"] < b + WINDOW)).sum())
            a_counts[i] = float(((away_shots["_minute"] >= b)
                                  & (away_shots["_minute"] < b + WINDOW)).sum())

        diff = h_counts - a_counts

        for i, (mid, d) in enumerate(zip(bin_labels, diff)):
            color = HOME_C if d >= 0 else AWAY_C
            ax1.bar(mid, d, width=WINDOW * 0.85, color=color, alpha=0.75,
                    edgecolor=BG, zorder=3)

        # Cumulative xG lines on secondary axis
        for ts, color, label in [
            (home_shots, HOME_C, home),
            (away_shots, AWAY_C, away),
        ]:
            ts_sorted = ts.sort_values("_minute")
            if ts_sorted.empty:
                continue
            minutes = [0.0] + ts_sorted["_minute"].tolist() + [95.0]
            cum_xg = [0.0] + ts_sorted["norm_xg"].cumsum().tolist()
            cum_xg.append(cum_xg[-1])
            ax2.step(minutes, cum_xg, where="post", color=color,
                     linewidth=1.5, alpha=0.6, linestyle="--",
                     label=f"{label} xG ({cum_xg[-1]:.2f})")
    else:
        ax1.text(0.5, 0.5, "No shot data available", ha="center", va="center",
                 fontsize=T_LABEL, color=MUTED, transform=ax1.transAxes)

    # HT line
    ax1.axvline(45, color=GRID_C, linestyle="--", linewidth=1.5, zorder=2)
    ax1.text(45.5, ax1.get_ylim()[1] * 0.9, "HT", color=MUTED,
             fontsize=T_TICK - 1, va="top")
    ax1.axhline(0, color=MUTED, linewidth=0.8, zorder=2)

    ax1.set_xlabel("Minute", fontsize=T_LABEL, color=TEXT)
    ax1.set_ylabel("Shot Differential (Home − Away)", fontsize=T_TICK, color=TEXT)
    ax2.set_ylabel("Cumulative xG", fontsize=T_TICK, color=MUTED)
    ax1.tick_params(colors=TEXT, labelsize=T_TICK)
    ax2.tick_params(colors=MUTED, labelsize=T_TICK)
    ax1.spines[:].set_color(GRID_C)
    ax2.spines[:].set_color(GRID_C)
    ax1.set_xlim(0, 96)
    ax1.yaxis.grid(True, color=GRID_C, linewidth=0.6, linestyle=":")
    ax1.set_axisbelow(True)

    legend_elements = [
        mpatches.Patch(facecolor=HOME_C, label=f"{home} dominant"),
        mpatches.Patch(facecolor=AWAY_C, label=f"{away} dominant"),
        Line2D([0], [0], color=HOME_C, linewidth=1.5, linestyle="--",
               label=f"{home} xG"),
        Line2D([0], [0], color=AWAY_C, linewidth=1.5, linestyle="--",
               label=f"{away} xG"),
    ]
    ax1.legend(handles=legend_elements, facecolor=BG, edgecolor=GRID_C,
               labelcolor=TEXT, fontsize=T_TICK, loc="upper left")

    title = (f"{home}  {sc['home']}–{sc['away']}  {away}"
             f"  —  Momentum (5-min windows)")
    fig.suptitle(title, color=TEXT, fontsize=T_TITLE, fontweight="bold", y=0.96)
    _brand(fig, handle)

    path = output_dir / _match_key(match) / "momentum.png"
    return _save(fig, path)


# ── 11. defensive actions ─────────────────────────────────────────────────────

_DEF_METRICS = [
    ("tackles", HOME_C, "Tackles", "o"),
    ("interceptions", AWAY_C, "Interceptions", "D"),
    ("clearances", "#8ecae6", "Clearances", "s"),
    ("blocked_shots", "#ffd166", "Blocks", "P"),
]


def defensive_actions(
    player_name: str,
    match: dict,
    output_dir: Path,
    handle: str = "@anup4khandelwal",
) -> Path:
    """
    Defensive action summary for a player, displayed as a stacked bar chart
    overlaid on a half-pitch with approximate defensive zone indication.
    """
    player_stats = match.get("player_stats")
    lineups = match.get("lineups")
    home, away = _team_names(match)
    sc = _score(match)

    # determine which team this player is on
    player_team = home
    if lineups is not None and not lineups.empty:
        name_c = _col(lineups, _PLAYER_CANDIDATES)
        team_c = _col(lineups, _TEAM_CANDIDATES)
        if name_c and team_c:
            hit = lineups[
                lineups[name_c].astype(str).str.lower().str.contains(
                    player_name.lower(), na=False
                )
            ]
            if not hit.empty:
                player_team = str(hit.iloc[0][team_c])

    is_home = player_team.lower() in home.lower()
    color = HOME_C if is_home else AWAY_C

    # collect defensive stats for the player
    stats: dict[str, float] = {}
    if player_stats is not None and not player_stats.empty:
        name_c = _col(player_stats, _PLAYER_CANDIDATES)
        if name_c:
            hit = player_stats[
                player_stats[name_c].astype(str).str.lower().str.contains(
                    player_name.lower(), na=False
                )
            ]
            if not hit.empty:
                row = hit.iloc[0]
                for col_name in ["tackles", "tackles_won", "interceptions",
                                  "clearances", "blocked_shots", "blocks",
                                  "errors", "aerial_duels_won", "pressures"]:
                    if col_name in row.index:
                        val = pd.to_numeric(row[col_name], errors="coerce")
                        if not pd.isna(val):
                            stats[col_name] = float(val)

    pos_df = _lineup_positions(lineups, player_team)
    player_xy = (50, 50)
    if not pos_df.empty and "player" in pos_df.columns:
        hit = pos_df[pos_df["player"].astype(str).str.lower().str.contains(
            player_name.lower(), na=False
        )]
        if not hit.empty:
            player_xy = (float(hit.iloc[0]["x"]), float(hit.iloc[0]["y"]))

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax_pitch = fig.add_axes([0.02, 0.08, 0.46, 0.82])
    ax_bar = fig.add_axes([0.54, 0.12, 0.42, 0.72])

    # left: half-pitch with player position marker
    pitch = Pitch(
        pitch_type="opta",
        pitch_color=BG,
        line_color=PITCH_LINE,
        half=True,
        line_zorder=2,
        linewidth=1.5,
    )
    pitch.draw(ax=ax_pitch)
    ax_pitch.set_facecolor(BG)

    # defensive zone circle
    zone_radius = 18
    circle = plt.Circle(
        (player_xy[0], player_xy[1]),
        zone_radius,
        color=color,
        fill=True,
        alpha=0.08,
        zorder=3,
    )
    ax_pitch.add_patch(circle)
    ax_pitch.scatter(
        player_xy[0], player_xy[1],
        s=400, c=color, edgecolors=BG,
        linewidths=2.5, zorder=6,
    )
    short = player_name.split()[-1]
    ax_pitch.text(
        player_xy[0], player_xy[1] - 7, short,
        ha="center", va="top", fontsize=T_LABEL,
        color=TEXT, fontweight="bold",
    )

    # right: bar chart
    ax_bar.set_facecolor(BG)
    ax_bar.spines[:].set_color(GRID_C)
    ax_bar.tick_params(colors=TEXT, labelsize=T_LABEL)

    display_stats = {
        "Tackles": stats.get("tackles", stats.get("tackles_won", 0)),
        "Interceptions": stats.get("interceptions", 0),
        "Clearances": stats.get("clearances", 0),
        "Blocks": stats.get("blocked_shots", stats.get("blocks", 0)),
        "Aerial Wins": stats.get("aerial_duels_won", 0),
        "Pressures": stats.get("pressures", 0),
    }

    bar_colors = [HOME_C, AWAY_C, "#8ecae6", "#ffd166", "#c77dff", "#ff9f1c"]
    keys = list(display_stats.keys())
    vals = [display_stats[k] for k in keys]

    bars = ax_bar.barh(keys, vals, color=bar_colors, edgecolor=BG,
                       height=0.55, zorder=3)
    ax_bar.xaxis.grid(True, color=GRID_C, linewidth=0.8)
    ax_bar.set_axisbelow(True)
    ax_bar.set_xlabel("Count", fontsize=T_LABEL, color=TEXT)

    for bar, val in zip(bars, vals):
        if val > 0:
            ax_bar.text(
                val + 0.05, bar.get_y() + bar.get_height() / 2,
                str(int(val)),
                va="center", fontsize=T_LABEL, color=TEXT, fontweight="bold",
            )

    if not any(vals):
        ax_bar.text(0.5, 0.5, "No defensive data", ha="center", va="center",
                    fontsize=T_LABEL, color=MUTED, transform=ax_bar.transAxes)

    title = (
        f"{player_name}  —  Defensive Actions\n"
        f"{home} {sc['home']}–{sc['away']} {away}"
    )
    fig.suptitle(title, color=TEXT, fontsize=T_TITLE - 2, fontweight="bold", y=0.97)
    _brand(fig, handle)

    safe_name = player_name.replace(" ", "_")
    path = output_dir / _match_key(match) / f"def_actions_{safe_name}.png"
    return _save(fig, path)
