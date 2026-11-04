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
- 237,943 markets, 18 Sep 2012 to 31 Oct 2026, 16,263-17,517 races a year in 2013-2025; 237,446 with one winner,
  103 dead heats, 394 with none; 308 runners without a BSP, 696 without a result; 180 without a race number or
  distance.
- Against Betfair's ANZ files, 2020 to Oct 2026: 1,130,747 runners matched (99.87% of ours found there, 99.99% of
  theirs found here); BSP, result, race number, distance, meeting date and track agree on 100%.
- Betfair's short meeting names ('AUS / Flem') mapped to full names from the ANZ files: 178 learned, each at 99% or
  more agreement; 12,180 rows (0.5%) keep a short name, short names the ANZ files (2020 on) never carry (Mdra 4,069, Stcr 3,089, ...).
- `off_uk` is UK local time: the ANZ clock runs 9, 10 or 11 hours ahead of it, with the two countries' clock changes.

## The Betfair-only model (3 Nov 2026)

The owner, 3 Nov: Punting Form is too expensive; racing.com and Racenet have no permission for automated use (Racenet's
robots.txt prohibits it). So stage 3 is first tried on Betfair's own history only (`features.py`, `model.py`,
workflow `au-model.yml`; run 41598199179, https://github.com/tomwilson41986/ashcroft/actions/runs/41598199179).

- The horse is its name: Betfair's Australian selection ids change with the saddlecloth (546,099 ids for 81,348
  names in Betfair's ANZ files, 2020-2026). 151,954 horses from Sep 2012; 36 rows where one name runs twice in a day.
- Leak check (features rebuilt with the future cut away, 15 Jun 2019, 4 Mar 2023, 12 Sep 2026): nothing moved.
- Walk-forward, test years 2015-2026, 1,949,381 runners priced out of sample:
  - mean absolute log error against the BSP 0.725 (2015-2026; 0.691-0.764 by year), against 0.907 for the horse's
    last market chance and 1.071 for equal chances; R^2 of the logit 0.51-0.53. The UK model, with form, scores
    0.398-0.403 on its development window (reports/serving_note_2026-09-27.md);
  - nothing beyond the BSP: the blend's weight on the model is -0.012 to -0.034 in every year, log-loss change
    -0.036 per 1,000 races (90% -0.085 to +0.012), 183,541 races 2016-2026;
  - value at the BSP: backing where the model's chance beats the BSP's loses 5.3-5.9% after commission, worse than
    backing every runner (-3.8%): the model's prices are compressed, so its 'value' is long shots.
- Verdict: the free model learns form (its error is 20% under the last-run baseline) but carries nothing the BSP
  does not. Whether it forecasts the move from an early price to the BSP (the UK edge) needs early prices: Betfair
  Historic Data, gate A. Its out-of-sample prices are stored at `s3://ashcroft/sources/au/model_report_oos.parquet`.

## Gate A: the early prices (4 Nov 2026)

The owner's Betfair Historic Data bundle (BASIC, Horse Racing, every country; `data.tar` from Drive, kept at
`s3://ashcroft/sources/betfair_historic/bundles/horse_racing_basic_2024.tar`) streamed by `historic.py`
(au-historic run 41818048958, https://github.com/tomwilson41986/ashcroft/actions/runs/41818048958).

- The bundle is cut short: 4,404,215,808 bytes, ending inside a file after 904,002 files (last whole file
  `BASIC/2026/Oct/4/...`); read to the cut. 6 files unreadable. WIN markets Jan 2024 to early Oct 2026: Australia
  85,504 markets, GB 27,670, Ireland 7,922 (`betfair_historic/horse_racing_win_<year>.parquet`).
- The market alone: an Australian price an hour out sits under the BSP far more than a British one. A back at the
  last traded price at T-60 closed by a lay at the SP averages -2.31% (GB/IE -0.69%); at T-15 -2.17% (GB/IE -0.58%).
  Australian books trade late: a traded price at T-60 for 45.8% of runners (GB/IE 99.9%), at T-15 73.2%.
- The model (its out-of-sample prices, 2024-2026; 452,367 runners joined, 56% of the AU WIN runners: the rest are
  harness, which the model does not price) does order the drift: within each price band its top fifth beat the
  band by +6.2% CLV at T-60 and its bottom fifth fell short by 4.0%. Backing where the price is over the model's
  expected BSP:
  - T-60: excess CLV over runners at the same price +0.98% (90% +0.55 to +1.42), but CLV itself -2.87%
    (-3.30 to -2.43): the edge is smaller than Australia's drift; to the result -7.5%;
  - T-15: excess +0.12% (-0.11 to +0.37), CLV -3.00%;
  - the first traded price: CLV +2.01% (+1.62 to +2.42), excess +2.25%; to the result -1.9%. A first trade can be a
    few dollars at an odd price, and BASIC holds no book or volume, so this is not yet a price anyone could take;
  - T-24h (5.5% of runners): CLV +6.36%, excess +3.65%, fading by year (2024 +8.1%, 2025 +1.3%, 2026 -0.2%).
- Verdict: **gate A fails for the hour before the off**, where Australian books trade: the model's edge does not
  cover the drift. What is left is at the first prices, where BASIC cannot show what was takeable. Only recorded
  books at the market's opening (gate B) can test it; the trader is not built.

Waiting on the owner: whether to record the Australian books from their opening (the `au-night` runner), or park
Australia.
