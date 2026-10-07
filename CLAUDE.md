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
| `BETFAIR_APP_KEY` | Betfair API application key (the delayed key: every read-only job, the market records) |
| `BETFAIR_LIVE_APP_KEY` | Betfair's live application key (6 Oct 2026): the live trader's own process only (live-trade.yml); Betfair permits no read-only use of live data |

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

# Update an earlier run's predictions for the non-runners since: drop every runner the card as it
# stands no longer lists, lists with no jockey (withdrawn) or as a reserve not yet in, re-normalise its race
# (writes the non-runners and why beside the output; every card is read that way: declared_runners). The card
# comes from HRB, or from a file: a racecard CSV, or the HTML card page predict-now's probe_card saves
python predict_bfsp_today.py --non-runners out/predictions.csv --output-csv out/now.csv [--card onedayracecards.html]
python scripts/predictions_workbook.py --predictions out/now.csv --non-runners out/now_non_runners.csv --meta data/models/bfsp_model_meta.json --out reports/predictions.xlsx

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

# Live trading on the owner's account (the owner's decision, 30 Sep 2026; TRADING.md): the owner's rule, GBP400 to
# win, at most GBP300 a bet, no new bet after 14:00 UK and no day limit (the owner, 7 Oct), each back laid at the SP.
# Places real orders only with TRADING_LIVE=yes, and only on the UK runner (live-trade.yml); the S3 object
# trading/STOP stops new bets
TRADING_LIVE=yes python auto_trade.py --live --until 14:05   # 08:00-14:00 UK, never within 15 min of an off (owner, 7 Oct)
python auto_trade.py --live --settle

# --- Research toolkit (see RESEARCH_FRAMEWORK.md) ---
# Model-vs-market scoring (Murphy decomposition, skill vs BSP, concordance, drift)
python research_lab.py score --predictions data/oos_predictions.csv

# Market-blind staking: Kelly ladder, per-race ranks, bankroll paths (see STAKING_REPORT.md)
python research_lab.py stake --predictions data/oos_predictions.csv --bank-chart banks.png

# Calibrate the BFSP forecast against realised BFSP (removes the compression and rank bias)
python research_lab.py price-cal --predictions data/oos_predictions.csv --save models/bsp_price_calibrator.json

# --- Fast experiment loop ---
# The feature build is the hours-long half of an evaluation and is identical
# whenever the feature code and the data are. Cache it once, then iterate.
python evaluate_oos.py --feature-cache .feature_cache --output-csv data/oos_predictions.csv

# Same cached matrix, model withheld from a group of features (this is how to
# bisect a suspected leak — costs a model fit, not a rebuild)
python evaluate_oos.py --feature-cache .feature_cache --drop-features td_,tdg_,going_draw_

# Force a rebuild (the cache does this by itself when feature code changes)
python evaluate_oos.py --feature-cache .feature_cache --refresh-cache

# Betfair historic price files: fetch (not from Cloudflare-blocked hosts), load, match, coverage
python betfair_prices.py --fetch --days 3 --load --match --report
# ... on the owner's UK server: every file Betfair lists, all markets (the owner's ask, 1 Oct), archived slowly to
# s3://$CAPTURE_BUCKET/betfair_prices_raw/, one file at a time: what the archive lacks and the last week again, the last
# week first, then UK/IE racing from 2018, then the other markets from 2018, then the older files, newest first
# (betfair-prices.yml nightly from 22:30 UTC, stopping by 06:15; live-record.yml after the last race for the UK/IE
# files); the nightly job loads UK/IE racing from 2018
python betfair_prices.py --archive --max-minutes 10
python betfair_prices.py --archive --market ukwin,ukplace,irewin,ireplace --refresh-days 2 --max-minutes 5
python betfair_prices.py --pull-s3 --load --match --dir data/betfair_raw --load-from 2018-01-01 --max-files 2500

# The live market record (the owner's ask, 30 Sep): every book the trader reads (BETFAIR_RECORD=1) and all-day
# snapshots on the UK server, raw to s3://$CAPTURE_BUCKET/betfair_live/<day>/; the nightly load builds
# betfair_live_markets, betfair_live_marks (the book at 08:00-12:00 UK and T-120..T-1 min, last, settled; the BSP read at each off, from 7 Oct) and live_orders
python betfair_recorder.py --record --final --until 21:30
python betfair_recorder.py --load-db --days 3

# The other markets' data (the owner's ask, 3 Oct; reports/all_markets_priorities.md): greyhound results (GBGB),
# football (football-data.co.uk), tennis (Sackmann, tennis-data.co.uk), Australian form (Punting Form, keyed), raw and
# tables to s3://$CAPTURE_BUCKET/sources/ (source-data.yml nightly); Betfair's Historic Data per sport (UK server)
python source_data.py --source all --fetch --build --report --max-minutes 280
python betfair_historic.py --my-data
python betfair_historic.py --fetch --max-minutes 60 && python betfair_historic.py --build
# a bundle downloaded from the Historic Data website: list it in sources/betfair_imports.json (source-data.yml imports
# each once), or import a local data.tar directly
python betfair_historic.py --import-tar data.tar --sport "Greyhound Racing"
# Every sport's markets and matched money (census), the ante-post books, greyhound and 2/3/4 TBP/each-way records
python betfair_recorder.py --census-sports
python betfair_recorder.py --antepost
python betfair_recorder.py --record --event-type 4339 --tag greyhound --until 21:30
python betfair_recorder.py --record --market-types OTHER_PLACE,EACH_WAY --tag other_place --until 21:30

# The greyhound model (reports/greyhound_model.md): GBGB metrics, walk-forward win model, scored against the SP and
# Betfair (greyhound-model.yml); its paper forward test from 5 Oct 2026, the model and rules frozen, daily after the
# recorded books settle (greyhound-track.yml, ledger in s3://$CAPTURE_BUCKET/sources/greyhound/track/)
python -m greyhound.model --years 2018-2026 --folds 2024-01,2025-01,2026-01 --from 2019-01-01 --out out/report.json
python -m greyhound.track --freeze --cutoff 2026-10-05 --days 3 --summary --out out/greyhound_track.json

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
- `model/race_shape.py`, `model/draw_curve.py` — run style from past comments, race shape, position value, draw curves (`SHAPE_DRAW_FEATURES`, built by `CustomMetricsEngine`, not served: iteration 25)
- `model/intent_features.py`, `model/freshness_features.py` — the connections' choices and each trainer's record with them against the price; days since the last run in context (served: `INTENT_SERVED_FEATURES` (card-safe), `SERVED_FRESHNESS_FEATURES`; `model/bfsp_features.py PRODUCTION_BLOCKS`)
- `model/form_windows.py` — every per-run measure (win, place, WAX, NFP, pounds beaten, race-centred pounds, rating-scale performance figure, RSR, the market's view, A−E vs BSP, position vs the market's order) over career, last run, last 3, last 5 and last 3/5/10 weighted linearly by recency (served from 26 Sep: `SERVED_FORM_WINDOW_FEATURES`; iteration 29)
- `model/shape_form.py` — the shape/draw remodel: each past run read against the pace (its actual position in the race's actual shape) and draw it met, windowed; speed drawn inside/outside/near today; the market's miss (A−E) by draw cell and projected position (`SHAPE_FORM_FEATURES`, built with the shape block, not served; a `RESEARCH_BLOCKS` entry: `evaluate_oos.py --blocks shape_form`)
- `model/blocks/` — drop-in feature blocks (one file each, computed on the cached matrix, no rebuild; see docs/FAST_LOOP.md). A model that reads one is served it: the live path builds it as training did (`blocks.attach_as_trained`); `train_bfsp.py --feature-cache .feature_cache --blocks form_variants,race_relative`
- `model/blocks/quant.py` — quant metrics (owner, 30 Sep; `qm_`, not served, retired: iteration 106 found nothing beyond the served main): the yard's, rider's, sire's and horse's record against the Betfair SP as a track record (Sharpe ratio of level-stake returns, signed chi of winners against the prices, A/E) and the horse's form as a return series (volatility, downside deviation, drawdown, the market's trend); `scripts/clv_betfair.py` scores every research arm on the owner's staking (GBP250 to win on each horse whose expected CLV clears 0/3/5%, `race_book.expected_clv_walk_forward`)
- `model/race_book.py` — the race book (owner, 30 Sep): a closing-price model (conditional logit on the BSP book from the market's price now and ours) and a whole-race book of backs and lays, sized for growth at the close or at the result under a race-level value floor (theta) and a per-position underlay limit (delta); `reports/race_book_model.md`
- `train_bfsp.py` — Walk-forward training with all custom metrics
- `predict_bfsp_today.py` — Daily BFSP predictions
- `model/trainer.py` — Win probability model (classification)
- `model/abm/` — Monte-Carlo race simulator (pace / traffic / draw) → `abm_*` features
- `model/diagnostics.py`, `effects.py`, `causal.py`, `selection.py`, `uncertainty.py`, `spatial.py`, `interpret.py`, `perf_figures.py`, `market_features.py` — research toolkit (RESEARCH_FRAMEWORK.md)
- `betfair_prices.py` — Betfair historic SP/price-movement files, every market Betfair lists, archived in S3 from the UK server (`--archive`); UK/IE racing → `betfair_prices` table, each day file noted in `betfair_prices_files` (loaded by daily-results.yml)
- `betfair_recorder.py` — the live market record (read-only): the trader's books (hook in `trading/exchange.py BetfairData`) and the recorder job's snapshots (live-record.yml), day files on the UK server and in S3, loaded nightly into `betfair_live_markets`, `betfair_live_marks`, `live_orders`
- `sources/` (`source_data.py`) — the historic data for the models beyond UK/IE racing, one module a source (GBGB, football-data, tennis, Punting Form); `betfair_historic.py` — Betfair's Historic Data service (BASIC plan, every sport) to opening/closing-price tables
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
