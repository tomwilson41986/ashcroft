# baseline_v0 - model card (racing², 18 Nov 2026)

The served price model's recipe, frozen as the baseline every Phase 1-6 candidate is compared with. Horse racing
(GB and IE) only.

## What it is

| | |
|---|---|
| Recipe | `st` in the research loop: the served features and blocks (`market_rating, connections_rating, pace_v2, expected_time, placement, gear, market_metrics, pedigree_bayes, improvement, debut` on top of the production blocks), without `sire_runners, damsire_runners` (`research/loop.json` at commit 85dba72) |
| Learner | LightGBM, objective huber, target `logit_norm_prob` (the logit of the race-normalised BFSP probability, `model/bfsp_model.py build_target`), 127 leaves, learning rate 0.02, 12,000 rounds, min_child_samples 500, extra_trees, float32, seed 42, deterministic |
| Features | 1,181 (`summary_st_f*.json n_features_used`) |
| Training window | each fold trains on races from 2021-01-01 (history from 2018-01-01) up to the fold, purged 30 days |
| Out-of-sample window | six 3-month folds, 2024-12-28 to 2026-03-31; 143,075 runners, 15,670 races |
| Source run | research loop iteration 126, run 43895762307, artifact `research-iter126-stack-collateral-164`, `oos_st.csv` |
| Served model (for reference) | the same recipe refitted on everything through 2026-11-12, in an ensemble with members `dml` and `rx` (`data/models/bfsp_model_meta.json`, `bfsp_ensemble.json`). Its reported holdout figures are in-sample (`holdout_metrics.in_sample: true`) and are not the baseline |
| Locked holdout | race dates from 2026-04-01 (82,545 rows, 8,864 races; `phase0_data_audit.json holdout_counts`) are not read |

## Out-of-sample metrics

Source: `outputs/horse_v0_20261118/phase0_baseline_v0.json` (query `research/queries/done/phase0_baseline_v0_1118.py`,
run 44332233719). Intervals 95%, resampled by race day.

### Target 1 - the BFSP (the owner's goal: as close as possible to the BFSP)

| Slice | Runners | Mean absolute log error | 95% CI |
|---|---|---|---|
| **All** | 143,075 | **0.37465** | 0.37168 - 0.37761 |
| Flat turf | 45,750 | 0.36159 | 0.35667 - 0.36654 |
| All-weather | 38,272 | 0.36667 | 0.36198 - 0.37161 |
| Jumps | 59,053 | 0.38993 | 0.38559 - 0.39416 |
| BFSP under 3 | 8,089 | 0.27100 | 0.26496 - 0.27756 |
| 3-6 | 23,430 | 0.26812 | 0.26457 - 0.27156 |
| 6-12 | 33,223 | 0.29466 | 0.29146 - 0.29793 |
| 12-25 | 30,773 | 0.36168 | 0.35746 - 0.36574 |
| 25-50 | 19,770 | 0.43545 | 0.43009 - 0.44042 |
| 50+ | 27,790 | 0.56136 | 0.55394 - 0.56863 |
| 2-7 runners | 32,168 | 0.33586 | 0.33190 - 0.34017 |
| 8-11 | 61,588 | 0.36996 | 0.36627 - 0.37381 |
| 12-15 | 34,822 | 0.39668 | 0.39100 - 0.40247 |
| 16+ | 14,497 | 0.42769 | 0.41818 - 0.43752 |

By quarter: 2025 Q1 0.38690, Q2 0.37374, Q3 0.35386, Q4 0.38724, 2026 Q1 0.37566 (2024 Q4: 1,360 runners only).

### Win probability against the market (normalised BSP)

15,647 races with one winner held, 133,936 runners.

| | Model | Market (BSP) |
|---|---|---|
| Log loss per runner | 0.30591 (0.30342 - 0.30865) | 0.29583 (0.29331 - 0.29835) |
| Brier per runner | 0.09047 (0.08954 - 0.09141) | 0.08780 (0.08699 - 0.08865) |
| Log loss, model minus market | +0.01008 (0.00917 - 0.01098): the market is better | |
| Calibration error (ECE, deciles of the model's probability) | 0.00293 | |

Benter test (conditional logit on log p_model and log p_market, each fold fitted on the folds before it, 13,023 races
scored): the model's weight is -0.076, 0.008, 0.006, 0.002, 0.011 across folds 1-5, against about 0.98 on the market.
The log-likelihood against the market alone is **-0.29 per 1,000 races (-0.58 to -0.01)**: out of sample the model
adds nothing to the BSP on who wins. That is expected of a model built to forecast the BSP, and matches the ledger
(`benter-blend-1020`, `bsp-calibration-1007`).

### Target 2 - normalised beaten lengths (added by the owner, 18 Nov)

Within each race (at least 4 finishers with a margin), Spearman's correlation of the price order with the outcome,
mean over races:

| Outcome | Races | Model's order | BSP's order | Model minus BSP |
|---|---|---|---|---|
| Race-centred pounds beaten (`lbsc`) | 15,168 | 0.42388 (0.41826 - 0.42949) | 0.46270 (0.45689 - 0.46836) | -0.03882 (-0.04234 to -0.03563) |
| lbsc over the field-size spread s(n) | 15,168 | 0.42831 (0.42229 - 0.43419) | 0.46769 (0.46193 - 0.47354) | -0.03938 (-0.04272 to -0.03608) |
| Normalised finishing position | 15,169 | 0.42944 (0.42375 - 0.43542) | 0.46871 (0.46282 - 0.47443) | -0.03927 (-0.04267 to -0.03594) |

The class adjustment (prize money) is constant within a race, so it does not move a within-race reading. It matters
for a model trained on the figure across races (Phase 3).

## Reproduction

Iteration 127 (run 44283128808) refitted the same recipe on the same folds from a later commit (85dba72):

| Fold | Runners | Error, iteration 126 | Error, iteration 127 | Difference | Runner CPU (126 / 127) |
|---|---|---|---|---|---|
| 0 | 23,309 | 0.38822 | 0.38822 | 0 | Xeon 8573C / EPYC 9V74 |
| 1 | 33,385 | 0.37442 | 0.37442 | 0 | EPYC 9V74 / EPYC 9V74 |
| 2 | 32,471 | 0.35316 | 0.35314 | -0.00002 | EPYC 9V45 / **EPYC 7763** |
| 3 | 30,064 | 0.38640 | 0.38640 | 0 | EPYC 9V74 / EPYC 9V74 |
| 4 | 22,638 | 0.37400 | 0.37400 | 0 | Xeon 6973P / EPYC 9V74 |
| 5 | 1,208 | 0.41623 | 0.41857 | +0.00234 | EPYC 9V74 / **EPYC 7763** |
| **All** | 143,075 | 0.37465 | 0.37466 | **+0.000015 (90% CI -0.00015 to +0.00020)** | |

Predictions are identical on 109,396 of 143,075 runners. The two folds that moved both ran on an EPYC 7763 in
iteration 127 and stopped at different rounds (11,997 against 11,999; 11,171 against 11,999). The other CPU changes
reproduce exactly. So the run-to-run noise from the runner's hardware is at most about a quarter of the 0.0007 bar a
change must clear, and the baseline reproduces. Source: the two runs' `fit-*-st-*` artifacts, `summary_st_f*.json`.

## Known weaknesses

- Error rises with the price (0.27 under 3 to 0.56 at 50+) and the field size (0.34 to 0.43).
- Jumps are worse than flat and all-weather.
- Out of sample it loses to the BSP on winners (by design: it forecasts the BSP), and its order agrees less with the
  beaten lengths than the BSP's order does (0.424 against 0.463).
- The last fold is 4 days (1,208 runners), so its figure is noisy.
- Not every runner is the same hardware: comparisons should be paired on the same folds and read against the bar,
  not single-fold differences.
