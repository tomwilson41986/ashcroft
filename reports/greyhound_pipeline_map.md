# Greyhound pipeline map (Phase 0 of the improve-validate-test brief, 18 Nov 2026)

racing². Compiled from the code at commit aa6c64f; every line cites its file. `Store` keys sit under `s3://<bucket>/sources/` (sources/common.py); recorder day files under `betfair_live/<day>/` (betfair_recorder.py).

## Stages

| Stage | Code | Workflow (schedule, UTC) | Reads | Writes |
|---|---|---|---|---|
| Ingestion: results | sources/gbgb.py | source-data.yml (03:20) | GBGB API | gbgb/runs_<year>.parquet |
| Ingestion: Betfair price files | betfair_prices.py, sources/greyhound_prices.py | greyhound-prices.yml (22:35), greyhound-model.yml | Betfair Historic | betfair_prices_raw/, greyhound_prices/prices_<year>.parquet |
| Ingestion: GreyhoundAPI | sources/greyhoundapi.py | greyhoundapi.yml | GreyhoundAPI (GREYHOUNDAPI_KEY) | greyhoundapi/runs_*, pedigree, comments, cards |
| Ingestion: live books | betfair_recorder.py (delayed key, 60 s); greyhound/book_record.py (live key, 60 s from T-15, 5 s from T-5); greyhound/stream_record.py | paper-models.yml (05:45 to 21:30 UK) | Betfair | betfair_live/<day>/books_greyhound.csv, books_greyhound_live.csv; betfair_stream/greyhound |
| Features | greyhound/data.py (clean), greyhound/metrics.py, greyhound/pit/ (point-in-time tables), greyhound/blocks/, card.py | greyhound-pit.yml (06:40), greyhound-cards.yml | gbgb/, greyhoundapi/, weather | greyhound/pit2/<table>.parquet; greyhound/feature_cache/<hash>/ |
| Training | greyhound/stack_model.py (3 LightGBM seeds + DeepSets nets, stacked; target the SP book's shares), greyhound/memory.py (residual memory) | greyhound-live.yml (Sun 04:41) | features, prices | greyhound/live/stack/bundle.pt, meta.json |
| Research models | greyhound/fastloop.py, greyhound/model.py | greyhound-fastloop.yml, greyhound-model.yml (dispatch) | feature cache, prices | greyhound/live/oos_prices_<recipe>/ |
| Prediction | greyhound/live.py predict_stack | greyhound-live.yml (hourly 07:21-19:21) | cards, books, pit tables | greyhound/live/predictions/<day>.csv (p_model, pred_bsp) |
| Selection, staking, orders | greyhound/early_lay.py, take_close.py, quote_mid.py (live-capable); live.py, fav_back.py (paper) | paper-models.yml | predictions, live-key books; SP forecast greyhound/laytest.py day_fit | greyhound/<trader>/<mode>/<day>/ledger.csv |
| Settlement | live.py --settle, early_lay reconcile, fav_back --settle, greyhound/track.py, tradeout.py, laytest.py | greyhound-live.yml, greyhound-track.yml | recorder's settled books | settled.csv, track/, tradeout/, laytest/ |

## Findings that matter for the brief

- **Rule 4 / reduction factors: not applied anywhere.** The recorder captures `adjustment_factor` and `removal_utc` (betfair_recorder.py), but no module under greyhound/ or research/greyhound/ reads them (checked by search, 18 Nov). Research drops a race with a late non-runner (greyhound/tradeout.py:46); live settlement leaves a removed runner's P&L as NaN (greyhound/live.py:878).
- **Non-runners, vacant traps:** training drops rows with no dog or no position and void races, then counts the field (greyhound/data.py clean). Live pricing removes runners whose status is REMOVED and renormalises over the active runners (greyhound/live.py).
- **Decision times differ by trader:** live.py within 1.5 min of the off; the early lay and the favourite back at the first traded book; take and close at the poll where every runner has traded (greyhound/take_close.py).
- **Real-order switches:** early lay, quote mid, take and close each need `--live`, their `GREYHOUND_*=live` variable and a daily loss limit; S3 STOP objects stop them (greyhound/STOP, greyhound/take_close/STOP, trading/STOP). As of 18 Nov 15:00 UTC all three are stopped or on paper.
- **Validation tooling already in place:** walk-forward folds and paired bootstrap (greyhound/fastloop.py), point-in-time tests (tests/test_leakage.py, tests/test_greyhound_pit.py), Benter/blend tests (greyhound/model.py blend_test), deflated Sharpe (research/overfit.py), PBO/CSCV (research/greyhound/pbo.py).
