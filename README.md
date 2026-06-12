# wc26-analytics

FIFA World Cup 2026 match analytics pipeline — generates match graphics and
tweet-thread text as committed files, ready for manual posting.

## Output

Each run produces files under `output/<Home>_vs_<Away>/`:

| File | Description |
|---|---|
| `shot_map.png` | Both teams' shots, dot size ∝ xG, gold ★ for goals |
| `xg_race.png` | Cumulative xG step chart with goal markers |
| `pass_network_<Team>.png` | Formation-based passing hub (node size ∝ passes) |
| `pizza_<Player>.png` | Percentile pizza vs all WC26 players in same position |
| `def_actions_<Player>.png` | Defensive action bar chart + pitch zone |
| `thread.md` | Numbered tweets (≤280 chars each) ready to post |

## Website

All match reports are published automatically to GitHub Pages:

**https://anup4khandelwal.github.io/wc26-analytics/**

The `Deploy Site` workflow rebuilds the page after every match run
(`scripts/build_site.py` — stdlib only, no extra dependencies).

## Quick start

```bash
pip install -r requirements.txt

# By team names (auto-resolves FBref match ID):
python scripts/run_match.py --home "Brazil" --away "Argentina" --handle "@yourtwitterhandle"

# By FBref match ID:
python scripts/run_match.py --fbref-id abc1234ef

# With player-specific charts:
python scripts/run_match.py --home "Spain" --away "England" \
    --charts all --player "Pedri" --handle "@yourtwitterhandle"
```

## GitHub Actions

Trigger **Actions → Match Report → Run workflow** and fill in:

| Input | Description |
|---|---|
| `home_team` | e.g. `Brazil` |
| `away_team` | e.g. `Argentina` |
| `fbref_match_id` | FBref hash — overrides team names if set |
| `charts` | `shot_map,xg_race,pass_network` (default) or `all` |
| `player` | Required only for `pizza` / `defensive` charts |
| `twitter_handle` | Your handle, e.g. `@anup4khandelwal` |

The workflow commits generated files back to the repo automatically.

## Data sources

| Source | Used for |
|---|---|
| [FBref](https://fbref.com) via [soccerdata](https://github.com/probberechts/soccerdata) | Shots + xG, player stats, lineups (primary — works from residential IPs) |
| [StatsBomb open data](https://github.com/statsbomb/open-data) | Automatic fallback — raw GitHub JSON works from GitHub Actions runners. Activates when StatsBomb publishes WC26 (they published WC22 free during the tournament) |
| [Sofascore](https://sofascore.com) via [ScraperFC](https://github.com/eddwebster/ScraperFC) | Shot coords / player ratings fallback |
| [openfootball](https://github.com/openfootball/world-cup) | Fixture list + scores |

All raw pulls are cached as Parquet in `cache/` (gitignored). Every network
call is preceded by a 3-second delay.
