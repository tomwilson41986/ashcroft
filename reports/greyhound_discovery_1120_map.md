# Greyhound discovery, 20 Nov 2026: the map

racing² - greyhounds. The owner, 20 Nov 2026: "find a way to address these. Ideate and run discovery to improve our
record on them." This file is what every inventor and checker reads first: the five weaknesses, what has been tried
against each (with its ledger tag; L = line in `reports/research_ledger.jsonl`), the data held, and the harnesses a
first test can run on. Sources: the ledger, `reports/greyhound_model.md` (GM), `reports/greyhound_edge_search.md` (ES),
`reports/greyhound_baseline_v0_card.md`, `reports/greyhound_data_audit.md`.

## The model and its five weaknesses

baseline_v0 = the live recipe `stack_mlsitd`: LightGBM trees and a neural network stacked on within-race shares of
the Betfair SP book, 747 card-safe features, fitted from 2024. Test 1 Jun - 11 Nov 2026 (run 44331106517): RMSE of
log(BSP) 0.2759; win log-loss 0.43322 against the BSP's 0.42825.

- **W1 No edge over the market.** Benter test against the BSP: weight market 0.808, model 0.196; the blend's log-loss
  0.42759 is worse than the market alone, 0.42733. At the decision-time book the model adds 0.000 a race.
- **W2 Compression.** Favourites priced too long (bias +0.12 in log price at BSP 1-2), outsiders too short (-0.23 at 15+).
- **W3 First-time runners.** RMSE 0.44 against 0.28 (2,020 of 110,788 test runners).
- **W4 Fills.** Every rule passes only on the optimistic "traded" fill and loses on the "through" fill a live order
  gets (the live early-lay rule about -2% of liability a day; live lays that matched had CLV +0.0175, unmatched +0.3511).
- **W5 The test window** 1 Jun - 11 Nov has been read about 55 times. The locked holdout is forward days from 19 Nov.

## Already tried (do not repeat)

**W1.** Blend by market depth: worse than the SP in every band (`greyhound-depth-blend-1111`, L735). Edge anatomy: 0
of 1,470 cells (`greyhound-edge-anatomy-1111`, L714). Edge models at the decision book (calibrated conditional logit,
two GBMs): all FAIL, predicted edge on taken backs about +6% against -3.5% realised (`greyhound-edge-models-1118`,
L994). Residual memory of our own error: PASS for RMSE, served (`greyhound-residual-memory-1111`, L734;
`greyhound-decay-memory-1112`). Australia: not feasible (`greyhound-au-model-1117`). Limit-on-close against our fair
price: 0 of 160 (ES family 4). 0 of 966 offline strategies survive (`greyhound-strategy-exploration-offline-1114`).
Not built: a multi-input Benter stage (roadmap 4.1); cross-market win/place/forecast (roadmap 4.7); the Betfair
adjustment factor (feature review D2).

**W2.** Seven calibration shapes (global and per-track stretch, cubic, isotonic, per-rank, learnt race stretch): each
gains under 0.001 (`greyhound-calibration-shape-1111`, L719). The favourite bias reads as regression to the mean
(reverse slope 0.856, reproduced by a placebo; `greyhound-short-odds-research-1113`, L793). Rolling recalibration:
-0.0005 to -0.0010 (`greyhound-online-recal-1115`). Live calibration `stretch_track` (0.2759 to 0.2743). Not built:
Shin-adjusted market probabilities; any forecast of the forecast's own uncertainty before the off.

**W3.** Trials table: PASS, -0.0107 (batch 26); rank correlation for dogs with no race 0.046 to 0.554. `pit_debut`:
-0.0010, z -2.7, not a pass alone. Pedigree, breeding2, trainer2, world (sire/dam abroad), origin: FAIL. trials2
PASS (-0.00102). Tables that exist: `greyhound/pit/` debut, trials, trials2, breeding, breeding2, pedigree, litter,
world, origin, trainers, trainer2. Not found: a trainer's record with its debutants against the BSP; `pit_litter`
scored alone. Not held: Irish trials, weigh-in before the off, racing-manager seeding.

**W4.** Lay backtest on stack_mlsitd: FAIL, through fill CLV -5.35% against traded +12.65%
(`greyhound-lay-backtest-mlsitd-1118`, L973). Take and close live: -4.76% CLV; its refit FAIL
(`greyhound-take-close-refit-1118`). 43 taker rules on the stream, 15 days: no candidate; the late family: every
cell negative. Limit-order study: 12 of 360 against 18 expected by chance. Resting test (queue-aware, ES family 13):
`rest_back_60_m0` +1.259% at the SP (1,652 trades, 90% 0.593 to 1.771, q 0.01) over 8 days of the 10 it needs to be
a candidate; `rest_back_60_m3` +2.264% (131 trades); `rest_lay_60_m3` +2.152% (114); the all-runner versions near
zero. `greyhound/quote_mid.py` quotes around the mid on paper, never scored. Not built: a queue-imbalance model
(roadmap 4.5); passive entries (4.6); a queue backtester validated against live fills (audit C.2).

**W5.** PBO near 0 on RMSE (`greyhound-pbo-cscv-1115`); deflated Sharpe per script with hard-coded trial counts
(`edge_models.N_TRIALS = 243`, `take_close_refit.N_TRIALS = 240`); fit-to-fit spread 0.00038. Not built: one trial
count from the ledger; one meta reading over all rules tried.

**Pending reads:** BFSP forward tests 25 Nov (`bfsp_forward.json`); favourite back gate 2 Dec
(`reports/preregistration_greyhound_fav_back.md`); pit_forward 7 Dec; early lay closed at the SP 13 Dec
(`greyhound-early-lay-sp-close-1114`).

## Data held

- **GBGB results** 2014 - today (`gbgb/runs_<year>.parquet`): position, times, sectional, trap, grade, weight, trainer,
  comments, SP. Trials (no off time) held, dropped by `greyhound.data.clean`, read by the trials table.
- **Betfair price files** (`greyhound_prices/prices_<year>.parquet`): BSP per runner, win and place markets, GB/IE.
  After the 18 Nov join fix, GBGB runs matched to a BSP: 2014-17 50-53%, 2018 80.9%, 2019 85.5%, 2020 95.2%,
  2021-26 94-99%; 351,438 fully priced races 2014-22 plus about 190,000 in 2023-26. Pre-off prices in these files
  are missing (shown as 1.0): the files give the BSP, not a path.
- **Recorder books** from 6 Oct (delayed key, about 90 s stale). **Live-key books** from 3 Nov
  (`betfair_recorder.BOOK_FIELDS`: back1-3, lay1-3 with sizes, last traded, runner and market matched, sp_near,
  sp_far, sp_actual; every 5 s from 5 min out, 60 s from 15 min; WIN and PLACE markets; the traded ladder per price).
- **Exchange Stream** from 5 Nov (`greyhound/stream_record.py`): full ladder, traded volume per price, ltp, near
  and far projected SP, SP back and lay pools; WIN and PLACE; replayed by `research/greyhound/stream_replay.py`.
- **GreyhoundAPI** (results GB/IE/AU, cards, Irish in-running comments from Nov 2021), NOAA weather, track geometry.
- Served prices: `greyhound/live/predictions/<day>.csv` from 6 Oct; out-of-sample prices of the live recipe to 11 Nov.
- Costs: 2% commission on a market's net gain (`greyhound/costs.py`, the owner's figure). Bets at the SP are always
  matched (the back side at the BSP; an SP lay by liability), so they carry no fill risk.

## Harnesses a first test can run on

- **Feature table:** `greyhound/pit/<name>.py` (`PREFIX`, `SOURCE`, `COLUMNS`, `build`), built by `greyhound-pit.yml`,
  scored by a spec in `research/greyhound/experiments/` on `greyhound-fastloop.yml` (about 2.5 hours for the stack;
  metrics: RMSE of log(BSP) with a paired by-day interval, log-loss, Benter against the BSP, splits by BSP band,
  runs before, field, track). The fast loop's cache starts 1 Jun 2020; a spec cannot filter or weight rows.
- **Model form:** `greyhound/fastloop.py` objectives softmax, pl2, binary, logbsp, share, mix, stack_share, nn_share;
  calibrations stretch, stretch_track, temper, temper_track.
- **Late market (last 300 s):** `research/greyhound/close_lab.py`: forecasts registered in `close_specs.json`
  (LightGBM with `init_score`), `trade_rules`, `band_rules`, `rest_orders` (queue-aware `stream_replay.SimOrder`,
  closed at the SP or run to the result); `greyhound-close-lab.yml`.
- **Taker rules on the stream:** `research/greyhound/stream_lab.py trades()`.
- **Pre-off rules on real fills:** `research/greyhound/take_close_refit.py takes()`, `edge_models.py`,
  `day_replay.py` (through and traded fills), `limit_study.fills`; `greyhound-api-research.yml` tasks.
- **Overfit control:** `research/overfit.py dsr_check(returns, n_trials)`; a pass needs a 90% interval above zero and
  a deflated Sharpe of at least 0.95 with every trial counted, on forward days from 19 Nov.
