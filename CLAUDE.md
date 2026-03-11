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

# Daily predictions pipeline (email)
python daily_predictions.py --dry-run
```

## Architecture

3-stage pipeline: Custom Metrics (19 proprietary) -> LightGBM regression on log(BFSP) -> Predictions

- `model/custom_metrics.py` — NFP, RB, WIV, WAX, WOA, CWO, ORR2, EPF, FSS, FCS, PFD, WPMRF, PMW, OFS, DSLR, LRP, pace, trainer-jockey, race strength (187 features total)
- `train_bfsp.py` — Walk-forward training with all custom metrics
- `predict_bfsp_today.py` — Daily BFSP predictions
- `model/trainer.py` — Win probability model (classification)
- S3 bucket: `horseracingresults`, key: `horse_racing.db`

## Data

- **Source**: horseracebase.com (CSV export + HTML scraping)
- **Storage**: SQLite `horse_racing.db` backed up to S3
- **Schema**: `race_results` table with 50+ columns

## Git Conventions

- Feature branches: `claude/<description>-<id>`
- Push with: `git push -u origin <branch>`
- Never force push to main/master
