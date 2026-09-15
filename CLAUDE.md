# Ashcroft — Horse Racing BFSP Prediction System

## Session Setup

At the start of every session that needs the historic database or live race cards, run:

```bash
bash scripts/setup_session.sh
```

This script:
1. Verifies `gh` CLI is authenticated
2. Fetches repository variables and prompts for secrets (cached in `.env.local`)
3. Downloads the full `horse_racing.db` from S3

### First-time setup

```bash
gh auth login                     # One-time: authenticate GitHub CLI
bash scripts/setup_session.sh     # Fetches secrets + downloads S3 database
```

After first run, secrets are cached in `.env.local` (gitignored). Subsequent sessions only need:
```bash
bash scripts/setup_session.sh --db   # Just re-download latest DB from S3
```

### Required secrets (in GitHub repo secrets)

| Secret | Purpose |
|--------|---------|
| `AWS_ACCESS_KEY_ID` | S3 access for `horse_racing.db` |
| `AWS_SECRET_ACCESS_KEY` | S3 access |
| `AWS_DEFAULT_REGION` | Default: `us-east-1` |
| `HRB_USERNAME` | horseracebase.com login |
| `HRB_PASSWORD` | horseracebase.com password |
| `SMTP_USERNAME` | Email sender for predictions |
| `SMTP_PASSWORD` | Email app password |
| `BETFAIR_USERNAME` | Betfair Exchange login |
| `BETFAIR_PASSWORD` | Betfair Exchange password |
| `BETFAIR_APP_KEY` | Betfair API application key |

## Key Commands

```bash
# Train BFSP model (all custom metrics)
python train_bfsp.py

# Generate today's predictions (live from HRB)
python predict_bfsp_today.py

# Predictions from database (no HRB login needed)
python predict_bfsp_today.py --from-db --date 2025-12-30

# Backtest last 7 days
python predict_bfsp_today.py --last-n-days 7

# Daily predictions pipeline (email, auto-fetches Betfair odds if configured)
python daily_predictions.py --dry-run

# Pull actual BSPs from Betfair for yesterday's completed races
python betfair_sync.py --bsp

# Pull BSPs for a date range
python betfair_sync.py --bsp --from 2026-03-01 --to 2026-03-10

# Show live Betfair exchange odds for today
python betfair_sync.py --live

# Show only upcoming/non-completed races with live odds
python betfair_sync.py --upcoming

# Export live odds to CSV
python betfair_sync.py --live --csv live_odds.csv

# --- Research toolkit (see RESEARCH_FRAMEWORK.md) ---
# Model-vs-market scoring (Murphy decomposition, skill vs BSP, concordance, drift)
python research_lab.py score --predictions data/oos_predictions.csv

# Betfair historic price files: fetch (not from Cloudflare-blocked hosts), load, match, coverage
python betfair_prices.py --fetch --days 3 --load --match --report

# Race ABM: simulate a card, batch features for training, pattern-oriented calibration
python research_lab.py abm --db horse_racing.db --date 2026-03-12
python research_lab.py abm-features --db horse_racing.db --from 2024-01-01 --jobs 8 --out data/abm_features.parquet
python research_lab.py abm-calibrate --db horse_racing.db --from 2025-01-01 --races 200

# Cross-classified effects, causal intervention effects, GP draw bias, market diagnostics
python research_lab.py effects --db horse_racing.db --from 2023-01-01
python research_lab.py causal --db horse_racing.db --treatment first_time_headgear
python research_lab.py draw --db horse_racing.db --track Chester --dist 5
python research_lab.py market --db horse_racing.db --from 2024-01-01

# Train with the opt-in feature blocks
python train_bfsp.py --perf-features --market-features --abm-features data/abm_features.parquet
```

## Architecture

3-stage pipeline: Custom Metrics (19 proprietary) -> LightGBM regression on log(BFSP) -> Predictions

- `model/custom_metrics.py` — NFP, RB, WIV, WAX, WOA, CWO, ORR2, EPF, FSS, FCS, PFD, WPMRF, PMW, OFS, DSLR, LRP, pace, trainer-jockey, race strength (187 features total)
- `train_bfsp.py` — Walk-forward training with all custom metrics
- `predict_bfsp_today.py` — Daily BFSP predictions
- `model/trainer.py` — Win probability model (classification)
- `model/abm/` — Monte-Carlo race simulator (pace / traffic / draw) → `abm_*` features
- `model/diagnostics.py`, `effects.py`, `causal.py`, `selection.py`, `uncertainty.py`, `spatial.py`, `interpret.py`, `perf_figures.py`, `market_features.py` — research toolkit (RESEARCH_FRAMEWORK.md)
- `betfair_prices.py` — Betfair historic SP/price-movement files → `betfair_prices` table
- S3 bucket: `horseracingresults`, key: `horse_racing.db`

## Data

- **Source**: horseracebase.com (CSV export + HTML scraping) + Betfair Exchange API
- **Storage**: SQLite `horse_racing.db` backed up to S3
- **Schema**: `race_results` table with 50+ columns
- **Betfair**: `betfair_client.py` (API client), `betfair_sync.py` (BSP sync + live markets)

## Rules

- **NEVER use sample/generated data** for training or evaluation. Always use the real `horse_racing.db` from S3.

## Git Conventions

- Feature branches: `claude/<description>-<id>`
- Push with: `git push -u origin <branch>`
- Never force push to main/master
