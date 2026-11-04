# Cricket model v1: next-ball XGBoost, Monte Carlo simulator, calibration

Source tables: `data/cricket/store/cricket/tables` (Cricsheet). Built by `python -m cricket.model.evaluate`. Paths per state: 256. Next-ball rounds: 95 (early stopping on 2022, fit on 2012-2021).

## Phase 0 gate, first condition: per-ball win probability on held-out IPL chases

Test seasons 2024-2026 (each calibrated on earlier seasons only; 2023 is calibration-only). Rows: the state before every delivery of the chase, 24,271 rows from 211 matches.

| model | log loss | Brier | ECE |
|---|---:|---:|---:|
| baseline: logistic on RRR, wickets, balls left | 0.4014 | 0.1296 | 0.0866 |
| simulator, raw | 0.3693 | 0.1168 | 0.0331 |
| simulator, calibrated | 0.3814 | 0.1207 | 0.0681 |
| published baseline (Mishra) | 0.4379 | | 0.0773 |

- Log loss below 0.4379: **PASS**
- ECE below 0.131: **PASS**
- Calibrated simulator beats the same-rows baseline: **yes** (-0.0200 log loss)

Calibration layer: **stack**, chosen by the rule fixed before the results: the lowest log loss on 2024 (each layer fitted on the seasons before), confirmed on the later seasons.

| layer | 2024 LL | 2025 LL | 2026 LL | pooled LL | pooled ECE |
|---|---:|---:|---:|---:|---:|
| isotonic | 0.4098 | 0.4096 | 0.3334 | 0.3841 | 0.0590 |
| platt | 0.4044 | 0.4097 | 0.3377 | 0.3837 | 0.0645 |
| stack | 0.4013 | 0.4008 | 0.3426 | 0.3814 | 0.0681 |
| baseline | 0.4005 | 0.4029 | 0.4007 | 0.4014 | 0.0866 |

- Benter combination weight against the Betfair mid, and the strategy after 5% commission: no data found (no Betfair cricket price history held; the Advanced data purchase waits on the owner's go).

## By season

| season | rows | baseline LL | raw LL | calibrated LL | baseline ECE | raw ECE | calibrated ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2023 (calibration only) | 8,428 | 0.5094 | 0.5169 | | 0.0466 | 0.0734 | |
| 2024 | 8,210 | 0.4005 | 0.3920 | 0.4013 | 0.0539 | 0.0459 | 0.0668 |
| 2025 | 7,908 | 0.4029 | 0.4168 | 0.4008 | 0.0666 | 0.0613 | 0.0558 |
| 2026 | 8,153 | 0.4007 | 0.3004 | 0.3426 | 0.1653 | 0.0721 | 0.1136 |

## By phase and required-run-rate band (test seasons)

| split | group | rows | baseline LL | calibrated LL | baseline ECE | calibrated ECE |
|---|---|---:|---:|---:|---:|---:|
| phase | powerplay | 7,944 | 0.5420 | 0.4912 | 0.1509 | 0.1106 |
| phase | middle | 11,543 | 0.3692 | 0.3666 | 0.0615 | 0.0602 |
| phase | death | 4,784 | 0.2454 | 0.2349 | 0.0596 | 0.0426 |
| rrr band | [0.0, 6.0) | 2,159 | 0.1283 | 0.1407 | 0.0242 | 0.0130 |
| rrr band | [6.0, 8.0) | 3,712 | 0.2343 | 0.1824 | 0.1703 | 0.1323 |
| rrr band | [8.0, 10.0) | 5,529 | 0.5775 | 0.5353 | 0.1855 | 0.1361 |
| rrr band | [10.0, 12.0) | 5,995 | 0.6068 | 0.5937 | 0.0637 | 0.0493 |
| rrr band | [12.0, 15.0) | 3,786 | 0.3816 | 0.3845 | 0.0175 | 0.0475 |
| rrr band | [15.0, 36.01) | 3,090 | 0.1034 | 0.0975 | 0.0106 | 0.0150 |

## Reliability, calibrated simulator (test seasons)

| p from | rows | mean p | chase won |
|---|---:|---:|---:|
| 0.0 | 3,584 | 0.031 | 0.021 |
| 0.1 | 2,217 | 0.150 | 0.126 |
| 0.2 | 2,321 | 0.250 | 0.176 |
| 0.3 | 2,216 | 0.349 | 0.314 |
| 0.4 | 2,121 | 0.450 | 0.489 |
| 0.5 | 2,151 | 0.549 | 0.682 |
| 0.6 | 2,078 | 0.648 | 0.817 |
| 0.7 | 1,857 | 0.751 | 0.897 |
| 0.8 | 2,138 | 0.851 | 0.956 |
| 0.9 | 3,588 | 0.956 | 0.984 |

## Next-ball model, held out by season

| season | scope | deliveries | log loss | base-rate log loss | accuracy |
|---|---|---:|---:|---:|---:|
| 2023 | all_t20 | 198,423 | 1.5460 | 1.5866 | 0.424 |
| 2023 | ipl | 17,307 | 1.5571 | 1.5968 | 0.427 |
| 2024 | all_t20 | 264,259 | 1.5408 | 1.5860 | 0.426 |
| 2024 | ipl | 17,103 | 1.5891 | 1.6412 | 0.405 |
| 2025 | all_t20 | 249,838 | 1.5588 | 1.5995 | 0.418 |
| 2025 | ipl | 16,463 | 1.5913 | 1.6392 | 0.411 |
| 2026 | all_t20 | 230,485 | 1.5536 | 1.6017 | 0.419 |
| 2026 | ipl | 17,008 | 1.6046 | 1.6586 | 0.404 |

Outcome classes: 0, 1, 2, 3, 4, 6, W, extra. Accuracy above 0.90 would stop the run as leakage.

## What v1 leaves out (stated, not filled)

- Who bats next is the next listed player of the XI not yet in; Cricsheet's list order is not guaranteed to be the batting order.
- Future overs are bowled by the bowling XI's experience-weighted average rating.
- No runs on wicket balls, no wickets on wides or no-balls, no free-hit flag.
- Ratings v1 have no phase x innings prior, no bowler type (no data found in Cricsheet), no matchups.
- First-innings win probability and the runs line: reports/cricket/model_v1_first_innings.md.
- Monte Carlo noise: 256 paths a state adds roughly 1/(2 x paths) to log loss near p = 0.5; the share of winning paths is smoothed to (wins + 1/2) / (paths + 1), never exactly 0 or 1.
