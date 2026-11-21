# racing² horse racing models - Phase 0 report (18 Nov 2026)

**Verdict: iterate. baseline_v0 is frozen and reproduces; Phase 1 (the validation harness) is next.** No model has
been changed. Scope: horse racing only (the owner, 18 Nov). Targets: the BFSP (kept) and normalised beaten lengths
(added).

## Five findings

1. **The baseline reproduces.** Iteration 127 refitted the served recipe on the same six folds from a later commit:
   mean absolute log error 0.37466 against 0.37465, difference +0.000015 (90% CI -0.00015 to +0.00020), predictions
   identical on 109,396 of 143,075 runners. The only drift comes from runners with an EPYC 7763 CPU, and it is about
   a quarter of the 0.0007 bar a change must clear. Source: `baseline_v0_model_card.md`, runs 43895762307 and
   44283128808.
2. **On the BFSP target the baseline's error is 0.37465 (0.37168-0.37761).** It rises with the price (0.268 at 3-6
   to 0.561 at 50+) and the field size (0.336 at 2-7 runners to 0.428 at 16+), and jumps are worst (0.390). That is
   where a better model has most to win. Source: `phase0_baseline_v0.json`.
3. **On winners the model adds nothing to the BSP, out of sample.** Benter weight about 0.00-0.01 against the
   market's 0.98. Log-likelihood -0.35 per 1,000 races (-0.68 to -0.01). Log loss 0.2935 against the market's 0.2838
   (corrected 19 Nov to every runner in the race; the first reading kept finishers only).
   This is expected of a BSP forecaster: the money is in the price before the off against the BSP (CLV), not in
   beating the BSP on the result.
4. **On normalised beaten lengths the model's order agrees less than the BSP's.** Spearman 0.424 against 0.463 on
   race-centred pounds beaten, difference -0.039 (-0.042 to -0.036). It is the same on the field-size-adjusted
   figure and the finishing position. That gap is what a model trained on the beaten-lengths target (Phase 3) has to
   narrow before it can help the price model.
5. **The data is clean on the targets but thin on execution.** No duplicates. The BFSP is usable on 99.94% of
   2,023,508 rows and the beaten-lengths target on 93.86%. But non-runners and Rule 4 are not in the results, a
   rating of 0 means unrated (538,146 rows), order-book depth starts on 1 Oct 2026, and results stop at 16 Nov
   (HRB's pause to 25 Nov). Source: `data_audit.md`.

## Deliverables in this folder

| File | What |
|---|---|
| `pipeline_map.md` | ingestion to settlement: diagram, every script, job and table |
| `data_audit.md` | rows by year and code, drops and why, duplicates, key-field nulls, date range, gaps |
| `baseline_v0_model_card.md` | the recipe, out-of-sample metrics for both targets, the reproduction diff, weaknesses |
| `phase0_data_audit.json`, `phase0_baseline_v0.json` and their `.txt` logs | the raw outputs (run 44332233719) |
| `../../experiments/log.csv` | the experiment log, a view of `reports/research_ledger.jsonl` rebuilt by `scripts/ledger_to_csv.py` |

## Decisions taken (say if any is wrong)

- **Experiment log.** `reports/research_ledger.jsonl` stays the one experiment log (append-only, 998 lines on 18 Nov, used for
  the deflated Sharpe trial count). `experiments/log.csv` is rebuilt from it, so the trial count is never split
  between two logs.
- **Holdout.** The locked holdout is race dates from 2026-04-01 (82,545 rows), opened once in Phase 6. Every
  candidate trains and is scored on races before it.
- **Baseline.** baseline_v0 is the served recipe scored walk-forward (iteration 126). It is not the served model's
  own metadata, which is in-sample: the served model is refitted through 2026-11-12 (`holdout_metrics.in_sample:
  true`).

## Phase 1 next: what exists and what to build

| Harness item (brief) | Exists | To build |
|---|---|---|
| Walk-forward by race date, purge | `evaluate_oos.py`, `scripts/research_loop.py`: 3-month folds, 30-day purge | an embargo option; meeting-level grouping check |
| Point-in-time checks | `tests/test_feature_blocks.py` (lag, contract and leak tests on every block), the training `post_race_guard` | one test over every served feature, including synthetic leakage injection that must be caught |
| Metrics by segment with race-day intervals | `scripts/compare_oos_runs.py` (paired error, rank, code, Brier skill, concordance), `scripts/clv_betfair.py` | one module: both targets, ECE, Benter, by track, race type, field size, price band, year (from `phase0_baseline_v0_1118.py`) |
| Overfitting | `research/overfit.py` (deflated Sharpe) | probability of backtest overfitting (CSCV) |
| Harness unit tests | | probabilities sum to 1 in a race; P&L reconciled by hand on 20 bets |

## Test suite (18 Nov, this container, Python 3.11, pandas 3.1.1)

`python -m pytest -q tests` (two chart and workbook test files left out: this container's system PIL breaks
matplotlib): **3,262 passed, 48 failed, 16 errors, 12 skipped** (`test_suite_1118.txt`).

| Failing | Count | Cause | Horse model affected? |
|---|---|---|---|
| Point-in-time leak tests: `test_feature_blocks.py` (11 blocks, the day's results blanked), `test_form_windows.py` (3), `test_career_before.py` (1), `test_shape_draw_production.py` (1) | 16 | Rounding, not leakage: the largest difference is 1.8e-15 (`fw_lbsc_l1`), with no missing value gained or lost. All 164 tests in those files pass under pandas 2.2.3 and fail only under 3.1.1, whose group means round differently in the last bit | No. Every production job installs pandas 3.1.1 (predict.yml run 44228556587, research-loop.yml run 43895762307, train-bfsp.yml run 43570002191), so training and serving agree |
| `test_football_warehouse*.py` | 27 failed, 16 errors | the warehouse needs `football/warehouse/requirements.txt`, not installed here | out of scope |
| `test_greyhound_pit.py`, `test_greyhound_fastloop.py` | 5 | not investigated: greyhounds are out of scope (the owner, 18 Nov) | out of scope |

For Phase 1: `requirements.txt` sets only minimum versions (`pandas>=2.1.0`), against the standing rule to pin
dependency versions. Pin them, and make the leak tests compare at a tolerance (1e-12) far below any real leak, so
the tests check for leaks rather than for rounding.

## Phase 1 - the validation harness (19 Nov)

| Brief item | Done | Where | Check |
|---|---|---|---|
| Walk-forward by date with purge and embargo | existed (30-day purge, embargo option) | `model/bfsp_model.py fold_masks`, `tests/test_measurement_integrity.py` | passes |
| No meeting split across training and the fold | test added | `tests/test_point_in_time.py` | passes |
| Point-in-time: every served feature under a leak test | test added: all 1,181 features of each member are in the deployed set or a block | `tests/test_point_in_time.py` | passes (0 uncovered) |
| Synthetic leak injection detected | tests added: a feature reading its own race's result and one lagged by row are flagged, a correct lag is not; the same for a planted block | `tests/test_point_in_time.py` | passes |
| Features the leak test cannot see | named: 5 served features are empty on the synthetic card (`front_sustainability`, `track_draw_bias`, `horse_draw_pref`, `fr_runs_since_break`, `fr_break_len`); a new one fails the test | `tests/test_point_in_time.py` | passes |
| Leak tests read leaks, not rounding | the five exact-equality leak tests compare at 1e-12 | `tests/test_feature_blocks.py`, `test_form_windows.py`, `test_career_before.py`, `test_shape_draw_production.py` | 501 passed under pandas 3.1.1 (16 failed before on 1e-15 differences) |
| Metrics with race-day intervals, both targets | BFSP error by code, price band, field size, quarter, year, track, race type; win log loss, Brier, ECE; walk-forward Benter; order agreement with beaten lengths | `research/harness.py`, `tests/test_harness.py` | 7 passed; reproduces baseline_v0's 0.37465 exactly |
| Probabilities sum to 1 in a race | test | `tests/test_harness.py` | passes |
| P&L reconciled by hand on 20 bets | 20 backs in 5 races, lays at the BSP, 2 non-runners, 2% commission on each market's positive net, through the trader's own settlement | `tests/test_pnl_by_hand.py` | 3 passed: every bet, every commission and the day's +52.29 |
| Deflated Sharpe | existed | `research/overfit.py dsr_check` | |
| Probability of backtest overfitting (CSCV) | added | `research/overfit.py pbo_cscv`, `tests/test_overfit_pbo.py` | 3 passed (noise 0.3-0.7, a real edge under 0.05) |
| Pinned dependency versions | **not done: the owner's decision.** The UK runner keeps one virtual environment and installs `requirements.txt` (minimum versions only) into it each morning; pinning would change the live trader's pandas, numpy and LightGBM on its next start | `requirements.txt`, `.github/workflows/live-trade.yml` | |

## Phase 4 - race selection (19 Nov)

Policy: keep the live rule (`selection_policy.yaml`). The research edge is in every segment (code, class, distance,
field size, hour, month, the 40 largest courses; ledger `phase4-segments-1119`), and the expected CLV is well
calibrated. Of 40 rules chosen walk-forward, the 0% bar beats the live 3% bar by GBP29.31 a day (90% +22.68 to
+35.78; deflated Sharpe 1.0 with 40 trials; PBO 0.33; ledger `phase4-rules-1119`), but by turnover (2.1% at the
margin in research, about 1% live) into the GBP4,000 day limit: not proposed.

## Phase 5 - execution (19 Nov)

| Finding | Evidence (ledger) |
|---|---|
| The live edge is the price taken early, not the picks: the trader's horses at the morning WAP -0.30%, at the matched price +8.29%, at the settled price +3.64% | `phase5-execution-1119` |
| Non-runners' reductions take -4.66%, most on the earliest backs (05-07 UK: 30% of horses cut, -5.6 to -9.3%); the research CLV carries none, so it overstates what live can take | `phase5-execution-1119` |
| The research CLV cannot yet carry the reductions: Betfair's price files list almost no withdrawn runners (140 of 116,124 races) | `phase5-reductions-1119` |
| Fills: 05-06 UK +16.8/+15.6% at the fill, 10:00 on about +1%; ladder fills below the best back +13.8% against +6.8% at the best | `phase5-execution-1119` |
| Each top-up is worth less after the reductions: 1st +6.07%, 2nd +3.88%, 3rd +1.94%, 4th +0.43%, 5th -1.46% (a floor) | `phase5-reductions-1119` |

Proposals for the owner (nothing changed; the live rule is the owner's): a cap of 3 top-ups a horse (from 5); keep
the 05:15 start and the ladder. For research: a reduction-aware CLV from the live books' removals and reduction factors
(S3 `betfair_live/<day>/`, from October), so research and live are measured alike.

Controls already in place (`trading/config_live.json`, `TRADING.md`, `CLAUDE.md`): the S3 `trading/STOP` kill switch;
GBP4,000 a day turnover limit; a stake of GBP15-200 a horse; at most 5 top-ups; no price over 200; order references
`ash<mmdd>` (the trader adopts and lays only its own); the trading watchdog; the day's ledger reconciled each evening
against Betfair's settled figure and the account's funds (`account_day_<mmdd>.py`; 18 Nov left a GBP27.52 gap under
check). Commission: 2% of each market's positive net, as the account's daily reconciliation has matched to the penny.

## Phase 6 - the locked holdout, read once (21 Nov)

Candidate gate_stn (the served three members, each with collateral, future_form and nbl_rating) against gate_st
(served), paired by runner. Folds 28 Mar - 26 Jun and 27 Jun - 25 Sep (run 44752561907) and 26 Sep - 16 Nov, never
scored by any research run (run 44875953861). Read by `research/queries/phase6_gates_1121.py` (run 44899288436),
full output in `phase6_gates_1121.txt`. Ledger `iter132-final`, result **PROMOTE**.

| Gate | Three folds | Unseen fold alone | |
|---|---|---|---|
| 1. Paired error vs BFSP (at least 0.0007 below, CI below zero) | -0.0027 (-0.0031 to -0.0022), 83,712 runners | -0.0019 (-0.0030 to -0.0009), 18,133 runners | pass |
| 2. Brier skill vs market, not resolved worse | -0.00037 (-0.00112 to +0.00034) | -0.00017 (-0.00171 to +0.00159) | pass |
| 2. Concordance, not resolved worse | -0.00004 (-0.00116 to +0.00095) | +0.00092 (-0.00112 to +0.00318) | pass |
| 3. Net CLV at the 3% bar, 2% commission (gate_st beside) | +8.58% (+8.13 to +9.01) vs +8.26% | +6.09% (+5.31 to +6.88) vs +5.65% | pass |
| 3. Difference, gate_stn less gate_st | +0.32 pts (+0.13 to +0.52) | +0.45 pts (+0.13 to +0.78) | pass |
| 4. Deflated Sharpe, 484 trials | 1.0 (Sharpe 2.30, 200 days) | 1.0 (Sharpe 1.66, 52 days) | pass |
| 5. Point-in-time tests | 487 passed, 5 skipped | | pass |

Reported, not gates: the gain is in the flat/AW (-0.0034); jumps -0.0009 on the three folds and +0.0003
(-0.0014 to +0.0019) on the unseen fold. Top pick at flat stakes -2.72% against -1.40% (intervals overlap). The
Benter test: neither blend adds to the BSP on the winner (-0.09 and -0.04 per 1,000 races, intervals across zero).
The query's first run used 5% commission instead of the registered 2%; it passed there too.
