"""
FIFA Post Match Summary Report (PMSR) — official free PDF data source.

FIFA publishes a Post Match Summary Report PDF for every WC 2026 match at:
  https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub.php

This module:
  1. Scrapes the hub page to find the correct PDF URL.
  2. Downloads and caches the PDF locally.
  3. Parses it with PyMuPDF (fitz) — no network calls, no IP-blocking.

What's available in PMSRs:
  ✓  Official match score
  ✓  Lineups with GK/DF/MF/FW positions, goals, cards, substitutions
  ✓  Team xG totals, possession, pass stats
  ✓  Per-shot log: minute, player, outcome, body_part  (no x/y, no per-shot xG)

Works from GitHub Actions runners — no Cloudflare, no auth required.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FIFA_HUB = (
    "https://www.fifatrainingcentre.com"
    "/en/fifa-world-cup-2026/match-report-hub.php"
)
FIFA_BASE = "https://www.fifatrainingcentre.com"
CACHE_DIR = Path("cache")

_UA = {"User-Agent": "wc26-analytics (github.com/anup4khandelwal/wc26-analytics)"}

# ── FIFA 3-letter country codes ──────────────────────────────────────────────
# Maps common name variants → FIFA code(s) used in PDF filenames.
_NAME_TO_CODE: dict[str, str] = {
    "mexico": "MEX", "south africa": "RSA",
    "korea republic": "KOR", "south korea": "KOR", "korea": "KOR",
    "czechia": "CZE", "czech republic": "CZE",
    "canada": "CAN",
    "bosnia and herzegovina": "BIH", "bosnia": "BIH",
    "usa": "USA", "united states": "USA", "us": "USA",
    "paraguay": "PAR",
    "haiti": "HAI",
    "scotland": "SCO",
    "australia": "AUS",
    "türkiye": "TUR", "turkey": "TUR", "turkiye": "TUR",
    "brazil": "BRA",
    "morocco": "MAR",
    "qatar": "QAT",
    "switzerland": "SUI",
    "spain": "ESP",
    "argentina": "ARG",
    "france": "FRA",
    "england": "ENG",
    "germany": "GER",
    "portugal": "POR",
    "netherlands": "NED",
    "belgium": "BEL",
    "uruguay": "URU",
    "colombia": "COL",
    "chile": "CHI",
    "peru": "PER",
    "ecuador": "ECU",
    "venezuela": "VEN",
    "bolivia": "BOL",
    "nigeria": "NGA",
    "senegal": "SEN",
    "cameroon": "CMR",
    "ghana": "GHA",
    "ivory coast": "CIV", "côte d'ivoire": "CIV", "cote d'ivoire": "CIV",
    "egypt": "EGY",
    "tunisia": "TUN",
    "algeria": "ALG",
    "japan": "JPN",
    "saudi arabia": "KSA",
    "iran": "IRN",
    "new zealand": "NZL",
    "costa rica": "CRC",
    "honduras": "HON",
    "panama": "PAN",
    "jamaica": "JAM",
    "trinidad and tobago": "TTO", "trinidad": "TTO",
    "serbia": "SRB",
    "croatia": "CRO",
    "denmark": "DEN",
    "sweden": "SWE",
    "norway": "NOR",
    "poland": "POL",
    "ukraine": "UKR",
    "romania": "ROU",
    "hungary": "HUN",
    "slovakia": "SVK",
    "slovenia": "SVN",
    "austria": "AUT",
    "wales": "WAL",
    "ireland": "IRL",
    "northern ireland": "NIR",
    "greece": "GRE",
    "turkey": "TUR",
    "israel": "ISR",
    "russia": "RUS",
    "iceland": "ISL",
    "albania": "ALB",
    "north macedonia": "MKD",
    "montenegro": "MNE",
    "georgia": "GEO",
    "armenia": "ARM",
    "azerbaijan": "AZE",
    "kazakhstan": "KAZ",
    "uzbekistan": "UZB",
    "china": "CHN",
    "indonesia": "IDN",
    "thailand": "THA",
    "vietnam": "VIE",
    "philippines": "PHI",
    "india": "IND",
    "iraq": "IRQ",
    "jordan": "JOR",
    "oman": "OMA",
    "bahrain": "BHR",
    "kuwait": "KUW",
    "uae": "UAE", "united arab emirates": "UAE",
    "cuba": "CUB",
    "el salvador": "SLV",
    "guatemala": "GUA",
    "nicaragua": "NCA",
    "venezuela": "VEN",
    "guyana": "GUY",
    "suriname": "SUR",
    "curacao": "CUW",
    "bermuda": "BER",
    "cayman islands": "CAY",
    "dr congo": "COD", "congo": "CGO",
    "kenya": "KEN",
    "tanzania": "TAN",
    "uganda": "UGA",
    "ethiopia": "ETH",
    "mali": "MLI",
    "burkina faso": "BFA",
    "benin": "BEN",
    "cape verde": "CPV",
    "mozambique": "MOZ",
    "zimbabwe": "ZIM",
    "zambia": "ZAM",
    "namibia": "NAM",
    "angola": "ANG",
    "gabon": "GAB",
    "equatorial guinea": "EQG",
    "comoros": "COM",
}


def _name_to_code(name: str) -> str:
    """Best-effort FIFA 3-letter code for a team name."""
    n = name.lower().strip()
    if n in _NAME_TO_CODE:
        return _NAME_TO_CODE[n]
    # Try partial match
    for key, code in _NAME_TO_CODE.items():
        if key in n or n in key:
            return code
    # Fallback: first 3 upper chars
    return n[:3].upper()


def _codes_match(c1: str, c2: str, name: str) -> bool:
    """True if c1 matches the FIFA code we'd derive from name."""
    derived = _name_to_code(name)
    return c1.upper() == derived or c2.upper() == derived


def _parse_pdf_link(href: str) -> tuple[str, str, str]:
    """
    Extract (match_num, home_code, away_code) from a PMSR PDF href.
    e.g. '/media/.../PMSR-M07-BRA-V-MAR POST-V2.pdf'
    """
    stem = Path(href).stem.upper()
    stem = re.sub(r'[\s\-]*(POST|V\d+|FINAL)[\s\-]*$', '', stem.strip())
    m = re.search(r'M(\d+)[\s\-]+([A-Z]+)[\s\-]V[\s\-]([A-Z]+)', stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return "", "", ""


# ── Hub page scraping ────────────────────────────────────────────────────────

_INDEX_CACHE = Path("cache/_fifa_hub_index.json")


def _fetch_hub_index(force: bool = False) -> list[dict]:
    """Return [{match_num, home_code, away_code, url}, ...] from FIFA hub."""
    if not force and _INDEX_CACHE.exists():
        try:
            import json
            return json.loads(_INDEX_CACHE.read_text())
        except Exception:
            pass

    try:
        import time; time.sleep(3)
        resp = requests.get(FIFA_HUB, headers=_UA, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("FIFA hub unreachable: %s", exc)
        return []

    pdf_hrefs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', resp.text, re.IGNORECASE)
    if not pdf_hrefs:
        # Some sites put URLs in data attributes or JS strings
        pdf_hrefs = re.findall(r'["\']([^"\']*PMSR[^"\']*\.pdf)["\']', resp.text, re.IGNORECASE)

    entries = []
    for href in pdf_hrefs:
        match_num, home_code, away_code = _parse_pdf_link(href)
        if not home_code:
            continue
        url = href if href.startswith("http") else FIFA_BASE + href
        entries.append({"match_num": match_num, "home_code": home_code,
                        "away_code": away_code, "url": url})

    if entries:
        import json
        _INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _INDEX_CACHE.write_text(json.dumps(entries, indent=2))
        logger.info("FIFA hub: %d PDFs indexed", len(entries))

    return entries


def find_pdf_url(home: str, away: str) -> Optional[str]:
    """Find the PDF URL for a match. Returns None if not published yet."""
    hc = _name_to_code(home)
    ac = _name_to_code(away)

    for entry in _fetch_hub_index():
        ehc, eac = entry["home_code"], entry["away_code"]
        if (ehc == hc and eac == ac) or (ehc == ac and eac == hc):
            logger.info("FIFA PDF: %s vs %s → %s", home, away, entry["url"])
            return entry["url"]

    # Re-fetch (might be newly published)
    for entry in _fetch_hub_index(force=True):
        ehc, eac = entry["home_code"], entry["away_code"]
        if (ehc == hc and eac == ac) or (ehc == ac and eac == hc):
            return entry["url"]

    logger.info("FIFA PDF not yet published for %s vs %s", home, away)
    return None


def _pdf_cache_path(home: str, away: str) -> Path:
    key = f"{home}_{away}".replace(" ", "_").lower()
    return CACHE_DIR / f"fifa_pmsr_{key}.pdf"


def download_pdf(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    try:
        import time; time.sleep(3)
        resp = requests.get(url, headers=_UA, timeout=120, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(1 << 16):
                f.write(chunk)
        logger.info("Downloaded FIFA PDF (%d KB) → %s",
                    dest.stat().st_size // 1024, dest.name)
        return True
    except Exception as exc:
        logger.warning("FIFA PDF download failed: %s", exc)
        return False


# ── PDF parsing with PyMuPDF ─────────────────────────────────────────────────

def _page_words(page) -> list[tuple[float, float, str]]:
    """Return [(x, y, word)] for all non-empty words on a page."""
    return [
        (w[0], w[1], w[4])
        for w in page.get_text("words")
        if w[4].strip()
    ]


def _group_rows(words: list[tuple], y_tol: float = 4.0) -> list[list[tuple]]:
    """Cluster words into rows by y-coordinate."""
    if not words:
        return []
    rows: list[list[tuple]] = []
    current_y = words[0][1]
    current_row: list[tuple] = []
    for w in sorted(words, key=lambda t: (t[1], t[0])):
        if abs(w[1] - current_y) > y_tol:
            if current_row:
                rows.append(sorted(current_row, key=lambda t: t[0]))
            current_row = [w]
            current_y = w[1]
        else:
            current_row.append(w)
    if current_row:
        rows.append(sorted(current_row, key=lambda t: t[0]))
    return rows


def _row_text(row: list[tuple]) -> str:
    return " ".join(w[2] for w in row)


# ── Page parsers ─────────────────────────────────────────────────────────────

def _parse_cover(page) -> dict:
    """Page 0: match metadata — teams, score, date, venue."""
    text = page.get_text("text")
    result: dict = {}

    # Score pattern: "1 - 1" or "2–0"
    m = re.search(r'\b(\d)\s*[-–]\s*(\d)\b', text)
    if m:
        result["score_home"] = int(m.group(1))
        result["score_away"] = int(m.group(2))

    # Date pattern: 2026-06-xx or DD Month YYYY
    dm = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if dm:
        result["date"] = dm.group(1)
    else:
        dm2 = re.search(r'(\d{1,2}\s+\w+\s+2026)', text)
        if dm2:
            result["date"] = dm2.group(1)

    # Team names are the two largest text blocks — heuristic: look for lines
    # in ALL CAPS that look like country names (≥3 chars, no numbers)
    caps_lines = [l.strip() for l in text.splitlines()
                  if re.match(r'^[A-Z][A-Z\s]{2,}$', l.strip())]
    if len(caps_lines) >= 2:
        result["home_team_raw"] = caps_lines[0]
        result["away_team_raw"] = caps_lines[1]

    return result


def _parse_lineups(page) -> dict:
    """
    Page 1: starting XI for home (left column) and away (right column).

    Home (Belgium-style, left col):  [num] [POS] [Name…]   x ~ 40–260
    Away (Egypt-style, right col):   [Name…] [POS] [num]   x ~ 730–910

    Uses x-coordinate separation — home < 40 % of page width, away > 60 %.
    Only includes starting XI (stops at SUBSTITUTES heading).
    """
    words = _page_words(page)
    if not words:
        return {"home": [], "away": []}

    w = page.rect.width
    cut_lo = w * 0.40   # home words have x < cut_lo
    cut_hi = w * 0.55   # away words have x > cut_hi

    home_words = [(x, y, t) for x, y, t in words if x < cut_lo]
    away_words = [(x, y, t) for x, y, t in words if x > cut_hi]

    def _clean(s: str) -> str:
        """Remove stray tokens from player name strings."""
        s = re.sub(r"\s*\d+'\s*", " ", s)          # sub markers "86'"
        s = re.sub(r"\s*\d+-\d+-\d+-?\d?\s*", " ", s)  # formation "4-2-3-1"
        s = re.sub(r"\s+\d+\s*$", "", s)           # trailing jersey numbers
        s = re.sub(r"^\d+\s+", "", s)              # leading jersey numbers
        return s.strip()

    def _extract_home(wds: list[tuple]) -> list[dict]:
        """[num] [POS] [Name…] — Belgium-style."""
        players, in_starting = [], True
        for row in _group_rows(wds):
            txt = _row_text(row)
            if "SUBSTITUTE" in txt.upper():
                in_starting = False
                continue
            if not in_starting:
                continue
            m = re.match(r'(\d{1,2})\s+(GK|DF|MF|FW)\s+(.+)', txt)
            if m:
                players.append({
                    "number": int(m.group(1)),
                    "position": m.group(2),
                    "name": _clean(m.group(3)),
                })
        return players

    _NON_PLAYER = {"STARTING", "SUBSTITUTES", "FORMATION", "DISTRIBUTION",
                   "MATCH", "SUMMARY", "TEAMS", "PLAYER", "COACH"}

    def _extract_away(wds: list[tuple]) -> list[dict]:
        """[Name…] [POS] [num] — Egypt-style (reversed)."""
        players, in_starting = [], True
        for row in _group_rows(wds):
            txt = _row_text(row)
            if "SUBSTITUTE" in txt.upper():
                in_starting = False
                continue
            if not in_starting:
                continue
            # Strip leading sub-minute markers before the name
            txt = re.sub(r"^(\d+'\s*)+", "", txt).strip()
            # Position and number at the end: "MOSTAFA SHOUBIR GK 23" or "FW10"
            m = re.search(r'^(.+?)\s+(GK|DF|MF|FW)\s*(\d{1,2})$', txt)
            if m:
                name = _clean(m.group(1))
                # Skip rows where non-player keywords leaked into the name
                name_words = set(name.upper().split())
                if name_words & _NON_PLAYER:
                    name = " ".join(w for w in name.split()
                                    if w.upper() not in _NON_PLAYER).strip()
                if name:
                    players.append({
                        "number": int(m.group(3)),
                        "position": m.group(2),
                        "name": name,
                    })
        return players

    return {
        "home": _extract_home(home_words),
        "away": _extract_away(away_words),
    }


def _parse_team_stats(page) -> dict:
    """
    Page 2: key statistics.

    The two-column layout extracts as:
        [home_value]          ← left column (home team)
        [stat name]           ← centre
        [away_value]          ← right column (away team)

    So the number BEFORE each label is home, number AFTER is away.
    """
    text = page.get_text("text")
    stats: dict = {}

    # xG — value before label = home, value after label = away
    # Raw: "1.07\nxG (Expected Goals)\n0.73\n"
    xg = re.search(r'([\d.]+)\s*\n[^\n]*xG[^\n]*\n\s*([\d.]+)', text, re.IGNORECASE)
    if xg:
        stats["xg_home"] = float(xg.group(1))
        stats["xg_away"] = float(xg.group(2))

    # Possession — "45.5%\n…Possession…\n16.4%"
    poss = re.search(
        r'([\d.]+)%?\s*\n[^\n]*Possession[^\n]*\n[^\d]*([\d.]+)', text, re.IGNORECASE
    )
    if poss:
        stats["possession_home"] = float(poss.group(1))
        stats["possession_away"] = float(poss.group(2))

    # Shots — "15 (4)\nAttempts at Goal (On Target)\n14 (3)"
    shots = re.search(
        r'(\d+)\s*\([^)]+\)\s*\n[^\n]*Attempts at Goal[^\n]*\n\s*(\d+)', text, re.IGNORECASE
    )
    if shots:
        stats["shots_home"] = int(shots.group(1))
        stats["shots_away"] = int(shots.group(2))

    # Passes — "476 (415)\nTotal Passes (Complete)\n411 (339)"
    passes = re.search(
        r'(\d+)\s*\([^)]+\)\s*\n[^\n]*Total Passes[^\n]*\n\s*(\d+)', text, re.IGNORECASE
    )
    if passes:
        stats["passes_home"] = int(passes.group(1))
        stats["passes_away"] = int(passes.group(2))

    return stats


_OUTCOME_KEYWORDS = (
    "On Target", "Off Target", "Incomplete", "Blocked", "Deflected"
)
_BODY_PARTS = ("Left Foot", "Right Foot", "Head", "Chest", "Knee")
_DELIVERY_TYPES = ("Pass", "Cross", "Corner", "Freekick", "Free Kick",
                   "Ball Progression", "Loose Ball", "Other", "Dribble",
                   "Set Piece", "Direct")


def _parse_shot_log_page(page, team: str) -> list[dict]:
    """
    Shot log page layout (one row per shot, 5 sequential text lines):
        {minute}
        {Player Name}
        {Outcome}
        {Body Part}
        {Delivery Type}

    The page header (Attempts at Goal / Belgium / Time / Player / Outcome …)
    is stripped before processing.
    """
    lines = [l.strip() for l in page.get_text("text").splitlines() if l.strip()]

    # Drop header lines (contain column labels or page title)
    _SKIP = {"Attempts at Goal", "Time", "Player", "Outcome", "Body Part",
             "Delivery Type", "Belgium", "Egypt", "France", "Morocco"}
    content: list[str] = []
    for ln in lines:
        skip = any(ln.startswith(s) or s == ln for s in _SKIP)
        # Also skip the date/venue footer and standalone team name lines
        if skip or re.match(r'^\d{1,2} \w+ 202', ln) or re.match(r'^[A-Z][a-z]+ Stadium', ln):
            continue
        content.append(ln)

    shots: list[dict] = []
    i = 0
    while i < len(content):
        # Look for a minute number on its own line
        if re.match(r'^\d{1,3}$', content[i]):
            minute = int(content[i])
            if not (1 <= minute <= 130):
                i += 1
                continue
            # Peek ahead for the 4 remaining fields
            player   = content[i + 1] if i + 1 < len(content) else ""
            outcome  = content[i + 2] if i + 2 < len(content) else ""
            body_    = content[i + 3] if i + 3 < len(content) else ""
            delivery = content[i + 4] if i + 4 < len(content) else ""

            # Validate: outcome must contain a known keyword
            if any(kw in outcome for kw in _OUTCOME_KEYWORDS):
                shots.append({
                    "minute": minute,
                    "player": player,
                    "team": team,
                    "outcome": outcome,
                    "body_part": body_,
                    "delivery_type": delivery,
                    "x": float("nan"),
                    "y": float("nan"),
                    "xg": float("nan"),
                })
                i += 5
                continue
        i += 1

    return shots


def _find_shot_pages(doc, home_team: str, away_team: str) -> tuple[list[dict], list[dict]]:
    """
    Scan the document for shot log pages by looking for 'Attempts at Goal'
    heading + a column of minute numbers.  Returns (home_shots, away_shots).
    """
    home_shots: list[dict] = []
    away_shots: list[dict] = []

    for i, page in enumerate(doc):
        text = page.get_text("text").lower()
        if "attempts at goal" not in text:
            continue
        # Determine team by looking for team name on the page
        is_home = home_team.lower()[:4] in text
        is_away = away_team.lower()[:4] in text
        # Check if it has shot rows (multiple minute numbers)
        minutes = re.findall(r'\b(\d{1,3})\b', page.get_text("text"))
        valid_minutes = [int(m) for m in minutes if 1 <= int(m) <= 130]
        if len(valid_minutes) < 3:
            continue  # summary page, not log page

        team = home_team if is_home else (away_team if is_away else "")
        shots = _parse_shot_log_page(page, team)
        if is_home:
            home_shots.extend(shots)
        else:
            away_shots.extend(shots)

    return home_shots, away_shots


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path, home: str, away: str) -> Optional[dict]:
    """
    Parse a FIFA PMSR PDF and return our standard data dict.

    Returns dict with keys:
      score   - {"home": int, "away": int}
      lineups - pd.DataFrame [player, team, position]
      shots   - pd.DataFrame [minute, player, team, outcome] (no x/y/xg)
      team_stats - dict (xg_home, xg_away, possession_home, ...)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (pymupdf) not installed — cannot parse FIFA PDF")
        return None

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.error("Cannot open PDF %s: %s", pdf_path, exc)
        return None

    n = len(doc)
    logger.info("Parsing FIFA PDF: %d pages, %s vs %s", n, home, away)

    # Cover page (index 0)
    meta = _parse_cover(doc[0]) if n > 0 else {}

    # Score from cover
    score = {
        "home": meta.get("score_home", 0),
        "away": meta.get("score_away", 0),
    }

    # Lineup page (index 1)
    lineup_data = _parse_lineups(doc[1]) if n > 1 else {"home": [], "away": []}
    lineup_rows = []
    for p in lineup_data["home"]:
        lineup_rows.append({"player": p["name"], "team": home, "position": p["position"]})
    for p in lineup_data["away"]:
        lineup_rows.append({"player": p["name"], "team": away, "position": p["position"]})
    lineups = pd.DataFrame(lineup_rows) if lineup_rows else pd.DataFrame()

    # Team stats page (index 2)
    team_stats = _parse_team_stats(doc[2]) if n > 2 else {}

    # Shot log pages (scan all pages)
    home_shots, away_shots = _find_shot_pages(doc, home, away)
    all_shots = home_shots + away_shots
    shots_df = pd.DataFrame(all_shots) if all_shots else pd.DataFrame()
    if not shots_df.empty:
        # Add empty xg/x/y columns so the rest of the pipeline knows what's missing
        for col in ["xg", "x", "y"]:
            shots_df[col] = float("nan")

    doc.close()

    return {
        "score": score,
        "lineups": lineups,
        "shots": shots_df,
        "team_stats": team_stats,
    }


def fetch_match(home: str, away: str) -> Optional[dict]:
    """
    Full entry point: find PDF → download → parse.

    Returns the same dict as parse_pdf(), or None if PDF not available.
    """
    # Try cache first
    pdf_path = _pdf_cache_path(home, away)
    json_cache = pdf_path.with_suffix(".parsed.json")

    if json_cache.exists():
        try:
            import json
            data = json.loads(json_cache.read_text())
            score = data.get("score", {})
            team_stats = data.get("team_stats", {})
            # Reconstruct DataFrames
            lineups = pd.DataFrame(data.get("lineups", []))
            shots_raw = data.get("shots", [])
            shots = pd.DataFrame(shots_raw) if shots_raw else pd.DataFrame()
            if not shots.empty:
                for col in ["xg", "x", "y"]:
                    if col not in shots.columns:
                        shots[col] = float("nan")
            return {"score": score, "lineups": lineups, "shots": shots, "team_stats": team_stats}
        except Exception:
            pass

    url = find_pdf_url(home, away)
    if url is None:
        return None

    if not download_pdf(url, pdf_path):
        return None

    result = parse_pdf(pdf_path, home, away)
    if result is None:
        return None

    # Cache parsed result as JSON
    try:
        import json
        import math

        def _df_to_list(df: pd.DataFrame) -> list:
            if df is None or df.empty:
                return []
            return [{k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                     for k, v in row.items()}
                    for row in df.to_dict("records")]

        cache_data = {
            "score": result["score"],
            "team_stats": result["team_stats"],
            "lineups": _df_to_list(result["lineups"]),
            "shots": _df_to_list(result["shots"]),
        }
        json_cache.write_text(json.dumps(cache_data, indent=2))
    except Exception as exc:
        logger.debug("Could not cache FIFA parsed JSON: %s", exc)

    return result
