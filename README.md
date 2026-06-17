# wc26-analytics

Auto-generated FIFA World Cup 2026 match analytics — charts and stats committed after every match, published to GitHub Pages.

**Live site → https://anup4khandelwal.github.io/wc26-analytics/**

---

## What it produces

Each match generates four charts and a thread under `output/<Home>_vs_<Away>/`:

| File | Description |
|---|---|
| `shot_map.png` | Shot outcomes by category (Goal / On Target / Off Target / Blocked) with official xG totals |
| `xg_race.png` | Cumulative xG step chart by minute, scaled to official FIFA totals |
| `pass_network_<Team>.png` | Formation-based passing network — node size ∝ pass volume, edges ∝ combination frequency |
| `thread.md` | Match stats summary ready for posting |

## How it runs

A scheduled GitHub Actions workflow (`match_report.yml`) fires every 3 hours, detects newly completed matches, and commits the charts back to `main`. The `Deploy Site` workflow then rebuilds the GitHub Pages site automatically.

To trigger manually: **Actions → Match Report → Run workflow**

## Data sources

| Source | Role |
|---|---|
| [FIFA PMSR PDFs](https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub.php) | **Primary** — official post-match reports published within hours of each game. Provides score, lineups, team xG, possession, shots, per-shot log (minute / player / outcome / body part / delivery type) |
| [Sofascore](https://sofascore.com) via [ScraperFC](https://github.com/eddwebster/ScraperFC) | Score + shot fallback when FIFA PDF isn't yet available |
| [openfootball](https://github.com/openfootball/world-cup) | Fixture schedule and scores |

## Local development

```bash
pip install -r requirements.txt

# Process today's completed matches
PYTHONPATH=. MPLBACKEND=Agg python scripts/run_today.py

# Build the static site locally
python scripts/build_site.py
# → opens site/index.html
```

## Project layout

```
wc26/
  fifa_pdf.py     FIFA PMSR PDF downloader + parser (PyMuPDF)
  fetch.py        Data source orchestration — FIFA → StatsBomb → Sofascore
  viz.py          Chart generators (mplsoccer, matplotlib)
  compose.py      Tweet thread builder
scripts/
  run_today.py    Main pipeline — detects completed matches, runs the chain
  build_site.py   Static site generator (stdlib only)
output/           Generated charts + threads (committed to repo)
cache/            Downloaded PDFs + parsed JSON (gitignored)
```
