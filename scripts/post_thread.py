#!/usr/bin/env python3
"""
Post WC 2026 match tweet threads to Twitter / X.

Scans output/ for thread.md files created/modified in the last MAX_AGE_HOURS
(so only the current run's matches are posted).  A .tweeted marker file inside
each match directory prevents double-posting on re-runs.

Credentials from environment variables (set as GitHub Secrets):
    TWITTER_API_KEY, TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
MAX_AGE_HOURS = 4   # only post threads modified within this window


# ── thread.md parser ──────────────────────────────────────────────────────────

def _parse_tweets(thread_md: Path) -> list[str]:
    """Parse thread.md into a list of tweet strings (one per tweet)."""
    text = thread_md.read_text(encoding="utf-8")
    tweets: list[str] = []

    for section in re.split(r'\n---\n', text):
        lines = section.strip().splitlines()
        tweet_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                tweet_lines.append("")
                continue
            if stripped.startswith("# Thread:"):
                continue
            if stripped.startswith("<!--"):
                continue
            if stripped.startswith("## Tweet"):
                continue
            if re.match(r'^\*\d+ / \d+ chars\*$', stripped):
                continue
            tweet_lines.append(line)

        tweet_text = "\n".join(tweet_lines).strip()
        if tweet_text:
            tweets.append(tweet_text)

    return tweets


# ── Twitter client ────────────────────────────────────────────────────────────

def _build_client():
    try:
        import tweepy
    except ImportError:
        logger.error("tweepy not installed — add it to pip install in the workflow")
        return None

    api_key              = os.environ.get("TWITTER_API_KEY")
    api_secret           = os.environ.get("TWITTER_API_SECRET")
    access_token         = os.environ.get("TWITTER_ACCESS_TOKEN")
    access_token_secret  = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

    missing = [k for k, v in {
        "TWITTER_API_KEY": api_key,
        "TWITTER_API_SECRET": api_secret,
        "TWITTER_ACCESS_TOKEN": access_token,
        "TWITTER_ACCESS_TOKEN_SECRET": access_token_secret,
    }.items() if not v]

    if missing:
        logger.error("Missing environment variables: %s", ", ".join(missing))
        return None

    return tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )


def _post_thread(client, tweets: list[str], match_name: str) -> bool:
    """Post tweets as a thread.  Returns True on full success."""
    prev_id = None
    for i, text in enumerate(tweets, 1):
        if len(text) > 280:
            text = text[:279] + "…"
        try:
            kwargs: dict = {"text": text}
            if prev_id:
                kwargs["in_reply_to_tweet_id"] = prev_id
            resp = client.create_tweet(**kwargs)
            prev_id = resp.data["id"]
            logger.info("  %d/%d posted (id=%s)", i, len(tweets), prev_id)
            if i < len(tweets):
                time.sleep(2)
        except Exception as exc:
            logger.error("  Tweet %d failed: %s", i, exc)
            return False
    logger.info("Thread posted: %s (%d tweets)", match_name, len(tweets))
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    client = _build_client()
    if client is None:
        sys.exit(1)

    now = time.time()
    posted = failed = skipped = 0

    for thread_md in sorted(OUTPUT_DIR.glob("*/thread.md")):
        match_dir  = thread_md.parent
        match_name = match_dir.name
        marker     = match_dir / ".tweeted"

        if marker.exists():
            logger.debug("Already tweeted: %s", match_name)
            skipped += 1
            continue

        age_h = (now - thread_md.stat().st_mtime) / 3600
        if age_h > MAX_AGE_HOURS:
            logger.debug("Skipping old thread (%s): %.1fh old", match_name, age_h)
            skipped += 1
            continue

        tweets = _parse_tweets(thread_md)
        if not tweets:
            logger.warning("No tweets parsed from %s — skipping", thread_md)
            skipped += 1
            continue

        logger.info("Posting thread for %s (%d tweets)…", match_name, len(tweets))
        ok = _post_thread(client, tweets, match_name)
        if ok:
            marker.write_text("ok\n")
            posted += 1
        else:
            failed += 1

    logger.info("Done: %d posted, %d skipped, %d failed.", posted, skipped, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
