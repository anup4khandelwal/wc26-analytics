#!/usr/bin/env python3
"""
Demo run using synthetic WC 2026 match data (USA vs Mexico, Group A).

Generates all standard charts + thread.md without hitting FBref, so it
works offline and is safe to run any time to verify the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUTPUT_DIR = Path("output")
HANDLE = "@anup4khandelwal"


# ── synthetic data builders ───────────────────────────────────────────────────

def _make_shots() -> pd.DataFrame:
    """Realistic shot-event table for a 2–1 result (USA 2 – Mexico 1)."""
    rows = [
        # USA shots (attacking left→right, opta x>50 = attacking half)
        # x, y, xg, outcome, minute, player, team
        (88, 48, 0.38, "Goal",   14, "Pulisic",    "USA"),
        (75, 35, 0.08, "Saved",  22, "Weah",       "USA"),
        (83, 60, 0.12, "Missed", 29, "Adams",      "USA"),
        (92, 50, 0.61, "Goal",   57, "Ferreira",   "USA"),
        (69, 42, 0.06, "Saved",  63, "Musah",      "USA"),
        (78, 55, 0.09, "Blocked",71, "Dest",       "USA"),
        (95, 52, 0.34, "Saved",  82, "Pulisic",    "USA"),
        (71, 28, 0.04, "Missed", 88, "McKennie",   "USA"),
        # Mexico shots
        (84, 46, 0.27, "Goal",   38, "Lozano",     "Mexico"),
        (76, 62, 0.07, "Saved",  45, "Vega",       "Mexico"),
        (88, 50, 0.19, "Saved",  52, "Raul J",     "Mexico"),
        (70, 38, 0.05, "Missed", 67, "Guardado",   "Mexico"),
        (93, 55, 0.41, "Missed", 74, "Lozano",     "Mexico"),
        (81, 45, 0.11, "Blocked",79, "H. Moreno",  "Mexico"),
        (97, 48, 0.47, "Saved",  90, "Jimenez",    "Mexico"),
    ]
    df = pd.DataFrame(rows, columns=["x", "y", "xg", "outcome", "minute", "player", "team"])
    # add small random jitter so dots don't overlap perfectly
    df["x"] += RNG.uniform(-2, 2, len(df))
    df["y"] += RNG.uniform(-4, 4, len(df))
    df["x"] = df["x"].clip(50, 100)
    df["y"] = df["y"].clip(0, 100)
    return df


def _make_player_stats() -> pd.DataFrame:
    rows = [
        # player, team, position, goals, assists, key_passes, tackles, interceptions,
        #         clearances, passes_completed_pct, xg, xg_assisted, stat_type
        ("Turner",    "USA", "GK",  0, 0, 0, 0, 0, 2, 72, 0.00, 0.00, "summary"),
        ("Dest",      "USA", "RB",  0, 0, 1, 2, 1, 1, 81, 0.09, 0.08, "summary"),
        ("Zimmermann","USA", "CB",  0, 0, 0, 1, 2, 4, 88, 0.00, 0.00, "summary"),
        ("Long",      "USA", "CB",  0, 0, 0, 2, 1, 5, 85, 0.00, 0.00, "summary"),
        ("Robinson",  "USA", "LB",  0, 1, 2, 1, 2, 0, 79, 0.05, 0.11, "summary"),
        ("Adams",     "USA", "CDM", 0, 0, 1, 4, 3, 1, 84, 0.12, 0.04, "summary"),
        ("McKennie",  "USA", "CM",  0, 0, 1, 2, 1, 0, 76, 0.04, 0.07, "summary"),
        ("Musah",     "USA", "CM",  0, 0, 2, 1, 1, 0, 78, 0.06, 0.14, "summary"),
        ("Weah",      "USA", "RW",  0, 0, 1, 0, 0, 0, 71, 0.08, 0.05, "summary"),
        ("Ferreira",  "USA", "CF",  1, 0, 0, 0, 0, 0, 68, 0.61, 0.00, "summary"),
        ("Pulisic",   "USA", "LW",  1, 0, 2, 0, 0, 0, 74, 0.72, 0.12, "summary"),
        ("Ochoa",     "Mexico","GK",0, 0, 0, 0, 0, 1, 70, 0.00, 0.00, "summary"),
        ("Sanchez",   "Mexico","RB",0, 0, 1, 2, 1, 2, 77, 0.02, 0.06, "summary"),
        ("H. Moreno", "Mexico","CB",0, 0, 0, 1, 2, 6, 82, 0.11, 0.00, "summary"),
        ("Montes",    "Mexico","CB",0, 0, 0, 2, 1, 4, 80, 0.00, 0.00, "summary"),
        ("Alvarez T.","Mexico","LB",0, 0, 1, 1, 1, 1, 76, 0.03, 0.09, "summary"),
        ("Guardado",  "Mexico","CDM",0,0, 2, 3, 2, 0, 89, 0.05, 0.15, "summary"),
        ("Herrera",   "Mexico","CM",0, 0, 1, 2, 0, 0, 83, 0.02, 0.08, "summary"),
        ("Vega",      "Mexico","RW",0, 0, 2, 0, 0, 0, 70, 0.07, 0.18, "summary"),
        ("Raul J",    "Mexico","CF",0, 0, 1, 0, 0, 0, 65, 0.19, 0.05, "summary"),
        ("Lozano",    "Mexico","LW",1, 0, 1, 1, 0, 0, 68, 0.68, 0.09, "summary"),
        ("Jimenez",   "Mexico","CF",0, 0, 0, 0, 0, 0, 60, 0.47, 0.00, "summary"),
    ]
    cols = [
        "player", "team", "position", "goals", "assists", "key_passes",
        "tackles", "interceptions", "clearances", "passes_completed_pct",
        "xg", "xg_assisted", "stat_type",
    ]
    return pd.DataFrame(rows, columns=cols)


def _make_lineups() -> pd.DataFrame:
    rows = [
        ("Turner",    "USA", "GK"),
        ("Dest",      "USA", "RB"),
        ("Zimmermann","USA", "CB"),
        ("Long",      "USA", "CB"),
        ("Robinson",  "USA", "LB"),
        ("Adams",     "USA", "CDM"),
        ("McKennie",  "USA", "CM"),
        ("Musah",     "USA", "CM"),
        ("Weah",      "USA", "RW"),
        ("Ferreira",  "USA", "CF"),
        ("Pulisic",   "USA", "LW"),
        ("Ochoa",     "Mexico", "GK"),
        ("Sanchez",   "Mexico", "RB"),
        ("H. Moreno", "Mexico", "CB"),
        ("Montes",    "Mexico", "CB"),
        ("Alvarez T.","Mexico", "LB"),
        ("Guardado",  "Mexico", "CDM"),
        ("Herrera",   "Mexico", "CM"),
        ("Vega",      "Mexico", "RW"),
        ("Raul J",    "Mexico", "CF"),
        ("Lozano",    "Mexico", "LW"),
    ]
    return pd.DataFrame(rows, columns=["player", "team", "position"])


def _build_match() -> dict:
    shots = _make_shots()
    player_stats = _make_player_stats()
    lineups = _make_lineups()
    return {
        "meta": {
            "home": "USA",
            "away": "Mexico",
            "fbref_id": "demo_wc26_usa_mex",
            "season": 2026,
            "score": {"home": 2, "away": 1},
        },
        "shots": shots,
        "player_stats": player_stats,
        "lineups": lineups,
        "sofascore": {},
    }


# ── run ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless

    from wc26 import viz
    from wc26.compose import compose_thread

    match = _build_match()
    print("Demo match: USA 2–1 Mexico  (Group A · WC 2026)")

    viz_paths = []

    print("  → shot_map…")
    viz_paths.append(viz.shot_map(match, OUTPUT_DIR, handle=HANDLE))

    print("  → xg_race…")
    viz_paths.append(viz.xg_race(match, OUTPUT_DIR, handle=HANDLE))

    for team in ["USA", "Mexico"]:
        print(f"  → pass_network ({team})…")
        viz_paths.append(viz.pass_network(team, match, OUTPUT_DIR, handle=HANDLE))

    print("  → player_pizza (Pulisic)…")
    viz_paths.append(
        viz.player_pizza("Pulisic", match, OUTPUT_DIR,
                         season_stats=match["player_stats"],
                         handle=HANDLE)
    )

    print("  → defensive_actions (Adams)…")
    viz_paths.append(viz.defensive_actions("Adams", match, OUTPUT_DIR, handle=HANDLE))

    print("  → thread.md…")
    thread = compose_thread(match, viz_paths, OUTPUT_DIR, handle=HANDLE)

    print(f"\nAll output written to {OUTPUT_DIR}/USA_vs_Mexico/")
    for p in viz_paths:
        print(f"  {p.relative_to(OUTPUT_DIR)}")
    print(f"  {thread.relative_to(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
