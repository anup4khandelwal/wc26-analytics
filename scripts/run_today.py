#!/usr/bin/env python3
"""
Auto-discover and process all WC 2026 matches that completed today (or late
yesterday in UTC) and don't yet have output files.

Called by the scheduled GitHub Actions workflow.  Safe to run multiple times —
matches that already have a thread.md are skipped.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta, timezone, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
HANDLE = "@anup4khandelwal"
SEASON = 2026

# Kick-off slots run until ~23:00 ET = 03:00 UTC next day, so we also
# check yesterday's date to catch overnight finishers.
def _check_dates() -> set[str]:
    today = date.today()
    yesterday = today - timedelta(days=1)
    return {str(today), str(yesterday)}


def _already_done(home: str, away: str) -> bool:
    match_key = f"{home}_vs_{away}".replace(" ", "_")
    return (OUTPUT_DIR / match_key / "thread.md").exists()


def _find_completed_matches(season: int = SEASON) -> list[dict]:
    """
    Pull the FBref schedule (cache-busted so scores are fresh) and return
    matches that:
      - fall on today or yesterday (UTC)
      - have a non-null score (i.e. the match is finished)
      - don't already have output
    """
    from wc26.fetch import _fbref_schedule  # noqa: PLC0415

    logger.info("Fetching latest FBref schedule (force-refresh)…")
    schedule = _fbref_schedule(season, force_refresh=True)

    # normalise column names
    col = lambda candidates: next(  # noqa: E731
        (c for c in candidates if c in schedule.columns), None
    )
    date_c = col(["date", "game_date"])
    home_c = col(["home_team", "home"])
    away_c = col(["away_team", "away"])
    id_c = col(["game_id", "match_id", "id"])
    hscore_c = col(["home_score", "score_home", "score1"])
    ascore_c = col(["away_score", "score_away", "score2"])

    if date_c is None or home_c is None or away_c is None:
        logger.error(
            "Unexpected FBref schedule columns: %s", list(schedule.columns)
        )
        sys.exit(1)

    check_dates = _check_dates()
    logger.info("Checking for completed matches on: %s", check_dates)

    matches = []
    for _, row in schedule.iterrows():
        row_date = str(row[date_c])[:10]
        if row_date not in check_dates:
            continue

        # require at least home score to confirm match is finished
        if hscore_c and (row[hscore_c] is None or str(row[hscore_c]) in ("", "nan", "None")):
            logger.debug("No score yet for %s vs %s — skipping", row[home_c], row[away_c])
            continue

        home = str(row[home_c])
        away = str(row[away_c])
        fbref_id = str(row[id_c]) if id_c else None

        if _already_done(home, away):
            logger.info("Already processed: %s vs %s", home, away)
            continue

        matches.append({"home": home, "away": away, "fbref_id": fbref_id})

    return matches


def _run_match(home: str, away: str, fbref_id: str | None) -> bool:
    """Fetch + visualise + compose for one match.  Returns True on success."""
    from wc26.fetch import fetch_match_report
    from wc26 import viz
    from wc26.compose import compose_thread

    logger.info("── Processing: %s vs %s (id=%s) ──", home, away, fbref_id)

    try:
        match = fetch_match_report(home=home, away=away, fbref_id=fbref_id, season=SEASON)
    except Exception as exc:
        logger.error("fetch_match_report failed: %s", exc)
        return False

    sc = match["meta"]["score"]
    logger.info("  Score: %s %s–%s %s", home, sc["home"], sc["away"], away)

    viz_paths: list[Path] = []
    for fn_name, kwargs in [
        ("shot_map",    {}),
        ("xg_race",     {}),
    ]:
        try:
            p = getattr(viz, fn_name)(match, OUTPUT_DIR, handle=HANDLE, **kwargs)
            viz_paths.append(p)
            logger.info("  %s → %s", fn_name, p.name)
        except Exception as exc:
            logger.warning("  %s failed: %s", fn_name, exc)

    for team in [home, away]:
        try:
            p = viz.pass_network(team, match, OUTPUT_DIR, handle=HANDLE)
            viz_paths.append(p)
            logger.info("  pass_network (%s) → %s", team, p.name)
        except Exception as exc:
            logger.warning("  pass_network (%s) failed: %s", team, exc)

    try:
        thread_path = compose_thread(match, viz_paths, OUTPUT_DIR, handle=HANDLE)
        logger.info("  thread → %s", thread_path)
    except Exception as exc:
        logger.error("  compose_thread failed: %s", exc)
        return False

    return True


def main() -> None:
    matches = _find_completed_matches()

    if not matches:
        logger.info("Nothing to process — all done or no matches today.")
        return

    logger.info("Found %d match(es) to process.", len(matches))
    success, failed = 0, 0
    for m in matches:
        ok = _run_match(m["home"], m["away"], m["fbref_id"])
        if ok:
            success += 1
        else:
            failed += 1

    logger.info("Finished: %d succeeded, %d failed.", success, failed)
    if failed:
        sys.exit(1)  # non-zero so the Actions step is marked red


if __name__ == "__main__":
    main()
