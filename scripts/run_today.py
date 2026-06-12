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

# A match is considered "completed" this long after kick-off.
COMPLETION_BUFFER = timedelta(hours=2, minutes=30)
# Only look back this far so early-tournament runs don't reprocess everything.
LOOKBACK = timedelta(hours=36)


def _already_done(home: str, away: str) -> bool:
    match_key = f"{home}_vs_{away}".replace(" ", "_")
    return (OUTPUT_DIR / match_key / "thread.md").exists()


def _parse_kickoff(date_str: str, time_str: str | None) -> datetime | None:
    """
    Parse openfootball date + time into an aware UTC datetime.
    Time looks like '13:00 UTC-6' or '18:00 UTC+2' (or may be missing).
    """
    try:
        d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    except ValueError:
        return None

    if not time_str:
        # no kickoff time — treat as 12:00 UTC so date-only entries still work
        return d.replace(hour=12, tzinfo=timezone.utc)

    parts = str(time_str).split()
    try:
        hh, mm = parts[0].split(":")
        offset_hours = 0.0
        if len(parts) > 1 and parts[1].upper().startswith("UTC"):
            off = parts[1][3:]
            if off:
                offset_hours = float(off)
        local = d.replace(hour=int(hh), minute=int(mm))
        return (local - timedelta(hours=offset_hours)).replace(tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return d.replace(hour=12, tzinfo=timezone.utc)


def _find_completed_matches(season: int = SEASON) -> list[dict]:
    """
    Discover matches from the openfootball fixture list (plain JSON — no
    scraping) whose kick-off was more than COMPLETION_BUFFER ago and less
    than LOOKBACK ago, and that don't already have output.

    Match data itself still comes from FBref via fetch_match_report().
    """
    from wc26.fetch import fetch_fixtures  # noqa: PLC0415

    logger.info("Fetching openfootball fixtures (force-refresh)…")
    fixtures = fetch_fixtures(force_refresh=True)

    now = datetime.now(timezone.utc)
    matches = []
    for _, row in fixtures.iterrows():
        home, away = row.get("team1"), row.get("team2")
        if not home or not away:
            continue  # TBD knockout slot

        kickoff = _parse_kickoff(row.get("date"), row.get("time"))
        if kickoff is None:
            continue

        finished_at = kickoff + COMPLETION_BUFFER
        if not (now - LOOKBACK <= finished_at <= now):
            continue

        home, away = str(home), str(away)
        if _already_done(home, away):
            logger.info("Already processed: %s vs %s", home, away)
            continue

        logger.info("Completed match found: %s vs %s (KO %s UTC)",
                    home, away, kickoff.strftime("%Y-%m-%d %H:%M"))
        matches.append({"home": home, "away": away, "fbref_id": None})

    return matches


def _run_match(home: str, away: str, fbref_id: str | None) -> bool:
    """
    Fetch + visualise + compose for one match.

    Returns True on success, False on a hard error, or None when no data
    source has the match yet (a clean skip — retried on the next run since
    no thread.md is written).
    """
    from wc26.fetch import fetch_match_report
    from wc26 import viz
    from wc26.compose import compose_thread

    logger.info("── Processing: %s vs %s (id=%s) ──", home, away, fbref_id)

    try:
        match = fetch_match_report(home=home, away=away, fbref_id=fbref_id, season=SEASON)
    except Exception as exc:
        logger.error("fetch_match_report failed: %s", exc)
        return False

    shots = match.get("shots")
    if shots is None or shots.empty:
        logger.info(
            "  No shot data available from any source yet — skipping, "
            "will retry on the next scheduled run."
        )
        return None

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
    success, failed, waiting = 0, 0, 0
    for m in matches:
        ok = _run_match(m["home"], m["away"], m["fbref_id"])
        if ok is True:
            success += 1
        elif ok is None:
            waiting += 1
        else:
            failed += 1

    logger.info(
        "Finished: %d succeeded, %d waiting for data, %d failed.",
        success, waiting, failed,
    )
    if failed:
        sys.exit(1)  # non-zero so the Actions step is marked red


if __name__ == "__main__":
    main()
