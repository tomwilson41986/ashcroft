# racing² cricket trader - status

Built from the owner's build spec of 4 Nov 2026. Every number below is copied from the report named beside it,
which says how it was produced. Nothing here has placed a bet or read the live Betfair key, and nothing can:
`tests/test_cricket_guards.py` enforces both.

## Where it stands (4 Nov 2026)

| Phase | State |
|---|---|
| 0 Research and backtest | Model condition met on held-out IPL chases. The two market conditions are untested: no data found (no Betfair cricket price history held). |
| 1 Paper trading | Code built and tested end to end on replays. Not running: needs a ball-feed vendor and a UK/Irish host (asks below). |
| 2-3 Live | Not started. Needs the owner's go at the Phase 1 gate. |

## Results

| What | Result | Source |
|---|---|---|
| Cricsheet ingest | PASS. 23,241 matches and 11,739,488 deliveries; rows in = rows out for every format; 0 duplicate keys; 0 nulls; a rerun adds 0 rows | `ingest_2026-11-04.md` |
| Baseline: logistic on RRR + wickets + balls left, IPL chases 2022-2026, walk-forward | log loss 0.4363 (published 0.4379, rows and split unknown) | `baseline_rrr.md` |
| Next-ball model, held out by season | log loss 1.541-1.559 vs base rate 1.586-1.602, all men's T20, 2023-2026 | `model_v1.md` |
| Chase win probability, IPL 2024-2026, 24,271 balls, calibrated (stack) | log loss 0.3814, ECE 0.0681. Same-rows baseline: 0.4014, 0.0866 | `model_v1.md` |
| By season, calibrated vs baseline log loss | 2024: 0.4013 vs 0.4005; 2025: 0.4008 vs 0.4029; 2026: 0.3426 vs 0.4007 | `model_v1.md` |
| First innings: match winner and the runs line | see the report | `model_v1_first_innings.md` |

How to read the chase result:

- The gain is almost all 2026. That season's chases were very different, and the baseline, fitted on earlier
  seasons, is badly calibrated there (ECE 0.165). 2024 and 2025 are roughly ties.
- The calibration layer was chosen by a rule fixed before the results (lowest 2024 log loss). It picked the stack,
  which made 2026 worse than the raw simulator (0.3426 vs 0.3004). The pre-registered choice stands.
- The spec's absolute gate (0.4379) is cleared by the same-rows baseline as well. Recommendation: gate on beating
  the same-rows baseline, not on a fixed number.

## What is built

| Area | Module | Tested by |
|---|---|---|
| Cricsheet ingest. Raw zips hash-deduplicated; HTML pages refused; tables; report | `cricket/ingest/cricsheet.py` | `tests/test_cricket_ingest.py` |
| Database schema and upserts (SQLite now, Aurora ready) | `cricket/db/` | `tests/test_cricket_ingest.py` |
| Match state, checked against 50 scorecards | `cricket/state/match_state.py` | `tests/test_cricket_state.py` |
| Ratings as of each match, and current ratings | `cricket/ratings/` | `tests/test_cricket_model.py` |
| Next-ball XGBoost; simulator (chase, and first innings into chase); calibration; evaluation | `cricket/model/` | `tests/test_cricket_model.py` |
| Production bundle, versioned in the store, with a model card | `cricket/model/production.py` | run below |
| Ball-feed interface with receipt times, Cricsheet replay, vendor-name mapping | `cricket/ingest/ball_feed.py`, `player_map.py` | `tests/test_cricket_live.py`, `tests/test_cricket_ingest.py` |
| Live pricing engine | `cricket/live/engine.py` | `tests/test_cricket_live.py` |
| Betfair recorder: read-only method whitelist, delayed key | `cricket/market/` | `tests/test_cricket_market.py` |
| Paper trader, sizing, fees, CLV, Phase 1 gate, session driver | `cricket/trading/`, `cricket/ledger/`, `cricket/reports/`, `cricket/live/paper.py` | `tests/test_cricket_trading.py` |

## Paper-trading readiness

The paper trader follows these rules:

- Each market's own bet delay is applied to every order.
- Fills come from the recorded ladder only, never the last traded price.
- An order lapses if the market suspends during its delay.
- A resting order fills only once the price trades through it.
- No trading in the 8 seconds after a wicket or a boundary.
- Stakes are one-third Kelly, with a 5% cap on what one match can lose and a 20% drawdown stop.
- Commission is 5% on net market winnings, and transactions are counted by the hour.
- CLV is measured against the mid at delay + 30 s and against the pre-match close.

What still stands between this build and a running paper trader:

1. **A ball-feed vendor (paid; the owner's go).** One adapter per vendor maps its events to `FeedEvent`, with names
   mapped by `PlayerMap`.
2. **A UK or Irish host for Betfair (paid; the owner's go).** GitHub's runners are refused by Betfair.
   `cricket-record.yml` waits for a runner labelled `cricket`.
3. **Betfair cricket price history (the owner's go).** The free BASIC plan through `betfair_historic.py`, or Advanced
   (£499). Needed for the two untested Phase 0 conditions.

## Runbook

```bash
python -m cricket.ingest.cricsheet --fetch --build --report
python -m cricket.model.baseline --test-from 2022
python -m cricket.model.evaluate --seasons 2023,2024,2025,2026 --paths 256
python -m cricket.model.evaluate_first --seasons 2023,2024,2025,2026 --paths 256
python -m cricket.model.production
python -m cricket.live.replay --match <cricsheet id>
python -m cricket.market.recorder --hours 6
python -m cricket.live.paper --feed-record feed.jsonl --books <day folder> --market <id> --runners '{"<sel>": "<team>"}'
```

`--root <folder>` runs any of these against a local store instead of S3.
