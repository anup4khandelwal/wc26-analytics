#!/usr/bin/env python3
"""
Build a static website from the generated match reports in output/.

Scans output/<Home>_vs_<Away>/ folders, renders a single dark-themed
index.html with every match's graphics and tweet thread, and copies the
images alongside it.  The result lands in site/ ready for GitHub Pages.

Stdlib only — no third-party dependencies, so the Pages workflow needs
nothing beyond a Python install.
"""

from __future__ import annotations

import html
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
SITE_DIR = ROOT / "site"

HANDLE = "@anup4khandelwal"
REPO_URL = "https://github.com/anup4khandelwal/wc26-analytics"

# Preferred display order for chart images within a match card.
IMAGE_ORDER = ["shot_map", "xg_race", "pass_network", "pizza", "def_actions"]

CSS = """
:root {
  --bg: #0e1117; --card: #161b26; --border: #2a3142;
  --text: #e6e9f0; --muted: #8b93a7;
  --home: #00d4aa; --away: #ff6b6b; --gold: #ffd700;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
header { text-align: center; padding: 32px 0 8px; }
header h1 { font-size: 1.9rem; letter-spacing: 0.5px; }
header h1 .accent { color: var(--home); }
header p { color: var(--muted); margin-top: 6px; font-size: 0.95rem; }
header a { color: var(--home); text-decoration: none; }
.match-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; margin-top: 28px; padding: 20px; scroll-margin-top: 16px;
}
.match-card h2 { font-size: 1.35rem; }
.match-card h2 .score { color: var(--gold); }
.imgs { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
.imgs a { display: block; }
.imgs img { width: 100%; border-radius: 8px; border: 1px solid var(--border); display: block; }
@media (max-width: 720px) { .imgs { grid-template-columns: 1fr; } }
details { margin-top: 16px; }
summary {
  cursor: pointer; color: var(--home); font-weight: 600;
  padding: 8px 0; user-select: none;
}
.thread {
  white-space: pre-wrap; background: var(--bg); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; font-size: 0.92rem; color: var(--text);
  overflow-x: auto;
}
.toc { margin-top: 20px; display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.toc a {
  background: var(--card); border: 1px solid var(--border); color: var(--text);
  padding: 6px 14px; border-radius: 999px; text-decoration: none; font-size: 0.9rem;
}
.toc a:hover { border-color: var(--home); }
.empty { text-align: center; color: var(--muted); margin-top: 48px; }
footer { text-align: center; color: var(--muted); margin-top: 48px; font-size: 0.85rem; }
footer a { color: var(--home); text-decoration: none; }
"""


def _scoreline(thread_md: Path) -> str | None:
    """First heading of thread.md looks like '# Thread: USA 2–1 Mexico'."""
    try:
        first = thread_md.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    m = re.match(r"#\s*Thread:\s*(.+)", first)
    return m.group(1).strip() if m else None


def _thread_text(thread_md: Path) -> str:
    """Strip markdown scaffolding down to the readable tweets."""
    lines = []
    for line in thread_md.read_text(encoding="utf-8").splitlines():
        if line.startswith("<!--") or line.startswith("# Thread:"):
            continue
        if line.strip() == "---":
            continue
        lines.append(re.sub(r"^##\s*", "", line))
    return "\n".join(lines).strip()


def _sorted_images(match_dir: Path) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        for i, prefix in enumerate(IMAGE_ORDER):
            if p.name.startswith(prefix):
                return (i, p.name)
        return (len(IMAGE_ORDER), p.name)

    return sorted(match_dir.glob("*.png"), key=key)


def collect_matches() -> list[dict]:
    matches = []
    if not OUTPUT_DIR.is_dir():
        return matches
    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir():
            continue
        images = _sorted_images(d)
        thread_md = d / "thread.md"
        if not images and not thread_md.exists():
            continue
        title = d.name.replace("_", " ")
        scoreline = _scoreline(thread_md) if thread_md.exists() else None
        matches.append({
            "slug": d.name,
            "title": scoreline or title,
            "images": images,
            "thread": _thread_text(thread_md) if thread_md.exists() else None,
            "mtime": max((p.stat().st_mtime for p in d.iterdir()), default=0),
        })
    # newest matches first
    return sorted(matches, key=lambda m: m["mtime"], reverse=True)


def _match_section(m: dict) -> str:
    parts = [f'<section class="match-card" id="{html.escape(m["slug"])}">']

    # gold-highlight the score inside e.g. "USA 2–1 Mexico"
    title = html.escape(m["title"])
    title = re.sub(r"(\d+\s*[–\-:]\s*\d+)", r'<span class="score">\1</span>', title, count=1)
    parts.append(f"<h2>{title}</h2>")

    if m["images"]:
        parts.append('<div class="imgs">')
        for img in m["images"]:
            rel = f'output/{m["slug"]}/{img.name}'
            ts = int(img.stat().st_mtime)
            parts.append(
                f'<a href="{rel}?v={ts}" target="_blank">'
                f'<img src="{rel}?v={ts}" alt="{html.escape(img.stem)}" loading="lazy"></a>'
            )
        parts.append("</div>")

    if m["thread"]:
        parts.append(
            "<details><summary>📝 Tweet thread</summary>"
            f'<div class="thread">{html.escape(m["thread"])}</div></details>'
        )

    parts.append("</section>")
    return "\n".join(parts)


def build() -> Path:
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir(parents=True)

    matches = collect_matches()

    # copy images so the site is self-contained
    for m in matches:
        dest = SITE_DIR / "output" / m["slug"]
        dest.mkdir(parents=True)
        for img in m["images"]:
            shutil.copy2(img, dest / img.name)

    toc = "".join(
        f'<a href="#{html.escape(m["slug"])}">{html.escape(m["title"])}</a>'
        for m in matches
    )
    body = "\n".join(_match_section(m) for m in matches) if matches else \
        '<p class="empty">No match reports yet — they appear here automatically after each WC 2026 match.</p>'

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-cache, must-revalidate">
<title>World Cup 2026 Analytics</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>🏆 World Cup <span class="accent">2026</span> Analytics</h1>
  <p>Auto-generated match graphics &amp; tweet threads · by
     <a href="https://twitter.com/{HANDLE.lstrip('@')}">{HANDLE}</a></p>
  <nav class="toc">{toc}</nav>
</header>
{body}
<footer>
  Generated by <a href="{REPO_URL}">wc26-analytics</a> ·
  data: FIFA PMSR / Sofascore / openfootball
</footer>
</div>
</body>
</html>
"""
    index = SITE_DIR / "index.html"
    index.write_text(page, encoding="utf-8")
    print(f"Built {index} with {len(matches)} match(es).")
    return index


if __name__ == "__main__":
    build()
