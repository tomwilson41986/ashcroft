# The Australian thoroughbred model

The UK/IE horse model's approach (market-blind, lag-safe metrics, walk-forward LightGBM, the closing-price model,
backs at the early price laid at the SP), rebuilt for Australia in its own package. The owner's ask, 2 Nov 2026.
The live UK/IE system is never changed for it (CLAUDE.md, Rules; `tests/test_workflow_guards.py`): this package
imports the UK/IE code and never edits it, and runs in its own jobs.

Evidence so far: `reports/international_racing.md` (the plan, sources, the port) and `reports/au_nz_market_scan.md`
(no market-only edge; the early price is the gate).

## Stages and gates

Each gate is pass/fail. A fail stops the work, and the next stage costs nothing until the gate is passed.

| Stage | Who | What | Gate |
|---|---|---|---|
| 0. Set-up | Owner | Betfair Historic Data BASIC Horse Racing plan (free); Punting Form written consent for betting use, a history quote, Starter, secret `PUNTINGFORM_API_KEY`; a second self-hosted runner on the UK server labelled `au-night`, repo variable `AU_RUNNER` | All three in place |
| 1. Data | Claude | BSP targets from 2012 (`au/data.py`, done); Punting Form form history (`sources/puntingform.py` build); early prices (`betfair_historic.py`); joins with row counts | **A**: AU early prices move to the BSP the way UK prices do |
| 2. Books | Claude | The unmodified recorder on AU books for 2 weeks, delayed key, on `au-night` | **B**: GBP15-200 backs fill before T-15 |
| 3. Model | Claude | `au/metrics.py` (the engine reused by import, an AU jurisdiction config), `au/model.py` walk-forward with the leak test | **C**: held-out CLV clears a threshold the owner approves before it is read |
| 4. Paper | Claude | `au/live.py --trade` on paper, its own workflow on `au-night`, rule written down in advance | **D**: the paper test passes |
| 5. Live | Owner's go | `au-live-trade.yml`, live key in the trader's process only, switch `AU_TRADING_LIVE=yes`, own limits, S3 `trading/STOP` honoured, settle and day report | Owner approves stakes and limits |

## What is here

- `config.py`: the jurisdiction's settings (price-file market, store keys, race times in UK hours, race-type
  families).
- `data.py`: the BSP targets, every Australian thoroughbred runner's Betfair SP and result from Sep 2012, from our
  price-file archive, reconciled with Betfair's ANZ files (`python -m au.data --build --anz-check`; workflow
  `au-data.yml`). Stored at `s3://$CAPTURE_BUCKET/sources/au/bsp_targets.parquet` with its data report beside it.

## Status (2 Nov 2026)

Stage 1 started: the BSP targets are built (au-data run 41461018554,
https://github.com/tomwilson41986/ashcroft/actions/runs/41461018554), at `s3://ashcroft/sources/au/bsp_targets.parquet`.

- 5,154 archive files (20 Sep 2012 to 1 Nov 2026; 18 empty, 0 unreadable), 4,119,258 rows in, 2,326,223 out:
  1,792,381 harness, 460 duplicates, 116 with no meeting name, 78 New Zealand dropped; rows in = out + dropped.
- 237,943 markets, 18 Sep 2012 to 31 Oct 2026, about 16,300-17,500 races a year from 2013; 237,446 with one winner,
  103 dead heats, 394 with none; 308 runners without a BSP, 696 without a result; 180 without a race number or
  distance.
- Against Betfair's ANZ files, 2020 to Oct 2026: 1,130,747 runners matched (99.87% of ours found there, 99.99% of
  theirs found here); BSP, result, race number, distance, meeting date and track agree on 100%.
- Betfair's short meeting names ('AUS / Flem') mapped to full names from the ANZ files: 178 learned, each at 99% or
  more agreement; 12,180 rows (0.5%) keep a short name, courses not raced since 2019 (Mdra 4,069, Stcr 3,089, ...).
- `off_uk` is UK local time: the ANZ clock runs 9, 10 or 11 hours ahead of it, with the two countries' clock changes.

Waiting on the owner for stage 0: the Historic Data plan, Punting Form's consent and key, the `au-night` runner.
