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
   market's 0.98. Log-likelihood -0.29 per 1,000 races (-0.58 to -0.01). Log loss 0.306 against the market's 0.296.
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

- **Experiment log.** `reports/research_ledger.jsonl` stays the one experiment log (append-only, 986 lines, used for
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
