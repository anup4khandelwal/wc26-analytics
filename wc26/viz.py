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
        f"Data: FBref / Opta  ·  {handle}",
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
    """Return DataFrame with columns player, pos, x, y for a team."""
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

    rows = []
    for _, row in df.iterrows():
        pos = str(row[pos_c]).upper().strip() if pos_c else "CM"
        x, y = _pos_to_xy(pos)
        rows.append({"player": row[name_c], "pos": pos, "x": x, "y": y})

    return pd.DataFrame(rows)


# ── 1. shot map ───────────────────────────────────────────────────────────────

def shot_map(match: dict, output_dir: Path, handle: str = "@WC26Analytics") -> Path:
    """
    Two attacking half-pitches side by side.
    Dots sized by xG; gold stars for goals.
    """
    shots_raw = match.get("shots")
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

            # missed / saved shots
            pitch.scatter(
                regular["norm_x"].fillna(85),
                regular["norm_y"].fillna(50),
                s=(regular["norm_xg"].clip(0.01, 1) * 600 + 50),
                c=color,
                alpha=0.55,
                edgecolors=BG,
                linewidths=0.8,
                zorder=5,
                ax=ax,
            )
            # goals
            if not goals.empty:
                pitch.scatter(
                    goals["norm_x"].fillna(95),
                    goals["norm_y"].fillna(50),
                    s=(goals["norm_xg"].clip(0.01, 1) * 900 + 200),
                    c=GOAL_C,
                    marker="*",
                    edgecolors=BG,
                    linewidths=0.5,
                    zorder=6,
                    ax=ax,
                )
        else:
            ax.text(
                50, 75, "No shot data",
                ha="center", va="center", fontsize=T_LABEL,
                color=MUTED, transform=ax.transData,
            )

        ax.set_title(label, color=color, fontsize=T_LABEL + 2, fontweight="bold", pad=8)

    # legend
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

def xg_race(match: dict, output_dir: Path, handle: str = "@WC26Analytics") -> Path:
    """Cumulative xG step chart with goal markers."""
    shots_raw = match.get("shots")
    home, away = _team_names(match)
    sc = _score(match)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.10)

    if shots_raw is not None and not shots_raw.empty:
        shots = _norm_shots(shots_raw)

        for team, color, label_suffix in [
            (home, HOME_C, f"  {sc['home']}g"),
            (away, AWAY_C, f"  {sc['away']}g"),
        ]:
            ts = shots[shots["_team"].str.lower().str.contains(team.lower(), na=False)]
            ts = ts.sort_values("_minute")

            if ts.empty:
                continue

            minutes = [0.0] + ts["_minute"].tolist() + [95.0]
            cum_xg = [0.0] + ts["norm_xg"].cumsum().tolist()
            cum_xg.append(cum_xg[-1])

            ax.step(minutes, cum_xg, where="post", color=color, linewidth=2.5,
                    label=f"{team}{label_suffix}  ({cum_xg[-1]:.2f} xG)")

            # goal markers
            goals = ts[ts["is_goal"]]
            goal_cum = 0.0
            for _, row in goals.iterrows():
                prev = ts[ts["_minute"] <= row["_minute"]]["norm_xg"].sum()
                goal_cum = prev
                ax.axvline(row["_minute"], color=GOAL_C, linestyle="--",
                           linewidth=1, alpha=0.6)
                ax.text(
                    row["_minute"] + 0.5, goal_cum + 0.02,
                    "⚽", fontsize=13, color=GOAL_C, va="bottom",
                )
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
    handle: str = "@WC26Analytics",
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
    handle: str = "@WC26Analytics",
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

    # enrich with passing volumes
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
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 28:  # neighbour threshold in opta space
                    if (i, j) not in edge_drawn:
                        ax.plot(
                            [positions[i][0], positions[j][0]],
                            [positions[i][1], positions[j][1]],
                            color=color, alpha=0.25, linewidth=1.5, zorder=3,
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


# ── 5. defensive actions ──────────────────────────────────────────────────────

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
    handle: str = "@WC26Analytics",
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
