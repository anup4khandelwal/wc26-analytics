#!/usr/bin/env python3
"""
Generate a full match report (graphics + tweet thread) for a WC 2026 fixture.

Usage
-----
# By team names:
python scripts/run_match.py --home "Brazil" --away "Argentina"

# By FBref match ID:
python scripts/run_match.py --fbref-id abc1234ef

# Specify which graphics to produce (comma-separated):
python scripts/run_match.py --home "France" --away "Germany" \\
    --charts shot_map,xg_race,pass_network \\
    --handle "@myhandle"

# Pizza / defensive charts for a specific player:
python scripts/run_match.py --home "Spain" --away "England" \\
    --player "Pedri" --charts pizza,defensive

Output is written to output/<Home>_vs_<Away>/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WC 2026 match analytics — generate graphics + tweet thread"
    )
    p.add_argument("--home", default=None, help="Home team name")
    p.add_argument("--away", default=None, help="Away team name")
    p.add_argument("--fbref-id", default=None, dest="fbref_id",
                   help="FBref match ID (overrides --home/--away)")
    p.add_argument("--season", type=int, default=2026, help="Season year (default: 2026)")
    p.add_argument(
        "--charts",
        default="shot_map,xg_race,pass_network",
        help=(
            "Comma-separated list of charts to generate.\n"
            "Options: shot_map, xg_race, pass_network, pizza, defensive\n"
            "Use 'all' for everything."
        ),
    )
    p.add_argument("--player", default=None,
                   help="Player name (required for pizza / defensive charts)")
    p.add_argument("--handle", default="@WC26Analytics",
                   help="Twitter handle for branding footer")
    p.add_argument("--output-dir", default="output", dest="output_dir",
                   help="Root output directory (default: output)")
    p.add_argument("--no-thread", action="store_true",
                   help="Skip generating thread.md")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    if args.fbref_id is None and not (args.home and args.away):
        logger.error("Supply --fbref-id OR both --home and --away")
        sys.exit(1)

    charts_req = (
        {"shot_map", "xg_race", "pass_network", "pizza", "defensive"}
        if args.charts == "all"
        else {c.strip() for c in args.charts.split(",")}
    )

    # ── fetch ──────────────────────────────────────────────────────────────────
    from wc26.fetch import fetch_match_report, fetch_season_stats

    logger.info("Fetching match report…")
    match = fetch_match_report(
        home=args.home,
        away=args.away,
        fbref_id=args.fbref_id,
        season=args.season,
    )

    meta = match["meta"]
    home = meta["home"] or args.home or "Home"
    away = meta["away"] or args.away or "Away"
    sc = meta["score"]
    logger.info("Match: %s %s–%s %s", home, sc["home"], sc["away"], away)

    # ── visualisations ─────────────────────────────────────────────────────────
    from wc26 import viz

    viz_paths: list[Path] = []

    if "shot_map" in charts_req:
        logger.info("Generating shot map…")
        p = viz.shot_map(match, output_dir, handle=args.handle)
        viz_paths.append(p)
        logger.info("  → %s", p)

    if "xg_race" in charts_req:
        logger.info("Generating xG race chart…")
        p = viz.xg_race(match, output_dir, handle=args.handle)
        viz_paths.append(p)
        logger.info("  → %s", p)

    if "pass_network" in charts_req:
        for team in [home, away]:
            logger.info("Generating pass network for %s…", team)
            p = viz.pass_network(team, match, output_dir, handle=args.handle)
            viz_paths.append(p)
            logger.info("  → %s", p)

    if "pizza" in charts_req:
        player = args.player
        if player is None:
            logger.warning("--player required for pizza chart; skipping")
        else:
            logger.info("Fetching season stats for pizza percentiles…")
            season_stats = fetch_season_stats(args.season)
            logger.info("Generating pizza chart for %s…", player)
            p = viz.player_pizza(
                player, match, output_dir,
                season_stats=season_stats,
                handle=args.handle,
            )
            viz_paths.append(p)
            logger.info("  → %s", p)

    if "defensive" in charts_req:
        player = args.player
        if player is None:
            logger.warning("--player required for defensive actions chart; skipping")
        else:
            logger.info("Generating defensive actions chart for %s…", player)
            p = viz.defensive_actions(player, match, output_dir, handle=args.handle)
            viz_paths.append(p)
            logger.info("  → %s", p)

    # ── thread ─────────────────────────────────────────────────────────────────
    if not args.no_thread:
        from wc26.compose import compose_thread

        logger.info("Composing tweet thread…")
        thread_path = compose_thread(
            match, viz_paths, output_dir, handle=args.handle
        )
        logger.info("  → %s", thread_path)

    logger.info("Done. %d graphic(s) written to %s/", len(viz_paths), output_dir)


if __name__ == "__main__":
    main()
