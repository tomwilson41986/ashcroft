# Greyhound baseline_v0 model card (18 Nov 2026)

racing² - greyhounds. Phase 0 of the owner's improve-validate-test brief. baseline_v0 is the live recipe `stack_mlsitd` (batch 54's `c_trials_drawbend`), frozen as the reference every later candidate is scored against.

## Reproduction: REPRODUCED

Registered before the run as `greyhound-baseline-v0-1118` in `reports/research_ledger.jsonl`. Run: greyhound-fastloop.yml 44331106517, spec `research/greyhound/experiments/batch56_baseline_v0.json`; table job 134990399412; artifact `fastloop-baseline_v0` (result_baseline_v0.json).

| Check | Registered (batch 54) | This run | Gap | Tolerance | Pass |
|---|---|---|---|---|---|
| RMSE of log(BSP) | 0.2764 | 0.2759 | 0.0005 | 0.0008 | yes |
| Gain over c_live (RMSE) | 0.00167 | 0.0024 (90% by day 0.0021 to 0.0027, z 12.3) | 0.0007 | 0.0008 | yes |

c_live itself read 0.2784 here against 0.2781 in batch 54 (fit-to-fit spread). The gain's gap (0.0007) sits near the edge of its tolerance.

## Data window

- Training rows from 2024-01-01 (recipe `train_from`), 666,412 fit rows; the live retrain fits from June 2024.
- Test: 2026-06-01 to 2026-11-11, 20,473 races, 112,134 runners (110,788 with a BSP for the RMSE; 110,630 on the BSP-race log-loss set).
- Source tables: GBGB results (`gbgb/runs_<year>.parquet`, cleaned by `greyhound.data.clean`), Betfair price files for the BSP target (`greyhound_prices/prices_<year>.parquet`). Read many times before: this window is not a holdout. The locked holdout is forward days from 19 Nov 2026.

## Model

- Objective `stack_share`: LightGBM trees (3 seeds) and a neural network (3 seeds, `greyhound/nn.py`) stacked on within-race shares; stack weights trees 0.5699, network 0.4485.
- 747 features: the card-safe set less `lf_wt_`, `hb_days_since_season`, `ti_new_trainer`, plus 26 blocks: sim, wpred, pit_figs, pit_ability, pit_trouble, pit_connections, pit_market, pit_breeding, pit_life, pit_form, pit_draw, pit_bend, pit_xtime, pit_trials, pit_optics, pit_line, pit_salience, pit_lastrace, pit_league, pit_speed, pit_seq, pit_irish, pit_trials2, pit_ability2, pit_draw2, pit_bend2.
- Learning rate 0.08, early stop 100, stopped at 4,487 rounds; max_bin 63. Fit time 7,852 s.
- Calibration `stretch_track` read beside it (RMSE 0.2743 calibrated, b 1.0243); the gate figures above are uncalibrated.

## Metrics (test window)

| Metric | baseline_v0 | c_live | BSP |
|---|---|---|---|
| RMSE log(BSP) | 0.2759 | 0.2784 | - |
| Win log-loss (all races) | 0.43322 | 0.43339 | - |
| Log-loss vs c_live (90% by day) | -0.00017 (-0.00031 to -0.00002, z -2.0) | 0 | - |
| Win log-loss on BSP races (110,630 runners) | 0.43322 | - | 0.42825 |

Benter's test against the BSP (fitted on the first half, read on the second): weights market 0.808, model 0.196; log-loss market 0.42733, model 0.43337, blend 0.42759. The blend is worse than the market alone: the model adds nothing to the BSP.

Error breakdown (110,788 runners): bias -0.032 (priced short overall); 88.5% of the within-race spread explained; race-level error 0.0936.

| BSP band | Runners | RMSE | Bias |
|---|---|---|---|
| 1-2 | 2,430 | 0.2016 | +0.1225 |
| 2-3 | 9,747 | 0.2025 | +0.0941 |
| 3-5 | 28,441 | 0.2077 | +0.0450 |
| 5-8 | 28,226 | 0.2328 | -0.0094 |
| 8-15 | 24,542 | 0.2842 | -0.0753 |
| 15+ | 17,402 | 0.4286 | -0.2258 |

Debutants (0 runs before, 2,020 runners): RMSE 0.4413, bias -0.1173.

## Known weaknesses

- **No edge over the market.** Benter weight on the model 0.196 with a blend worse than the BSP; the decision-time Benter test (`greyhound-edge-models-1118`) gained -0.00044 a race (90% interval about +/-0.009). Every trade rule built on it so far has failed (`greyhound-take-close-refit-1118`, `greyhound-edge-models-1118`, the early lay backtest).
- **Compressed prices.** Favourites priced too long, outsiders too short (bias +0.12 at 1-2, -0.23 at 15+); `stretch_track` removes part of it.
- **Debutants** carry the largest error.
- **Data** (`reports/greyhound_data_audit.md`): fields are shrinking (six-runner races 90% of races in 2018, 52% in 2026); missing dog ids and positions rising since 2023; Rule 4 reduction factors recorded but never applied; the BSP join before 2021 was fixed on 18 Nov (2014-2017 still about half matched); the out-of-sample prices end on 11 Nov.
- **The test window is reused**: about 55 batches have read 1 Jun - 11 Nov.
