# Cricket model v1, first innings: match winner and the runs line

Source tables: `data/cricket/store/cricket/tables` (Cricsheet). Built by `python -m cricket.model.evaluate_first`. Paths per state: 256. Next-ball rounds: 95.
Rows: IPL first innings at the start of each over, decided non-D/L matches. Test seasons 2024-2026: 4,180 over starts from 211 matches (2023 is calibration-only).

## P(side batting first wins)

| model | log loss | Brier | ECE |
|---|---:|---:|---:|
| baseline: logistic on runs, wickets, legal balls | 0.6389 | 0.2229 | 0.0976 |
| simulator, raw | 0.6093 | 0.2123 | 0.0339 |
| simulator, calibrated | 0.6314 | 0.2203 | 0.1004 |

Calibration layer: **stack** (the lowest log loss on 2024, fitted on the seasons before; the rule of cricket/model/calibration.py).

| layer | pooled LL | pooled ECE |
|---|---:|---:|
| isotonic | 0.6360 | 0.0801 |
| platt | 0.6256 | 0.0767 |
| stack | 0.6314 | 0.1004 |

| season | over starts | baseline LL | raw LL | calibrated LL |
|---|---:|---:|---:|---:|
| 2023 | 1,418 | 0.6651 | 0.7011 | calibration only |
| 2024 | 1,413 | 0.6057 | 0.6239 | 0.6236 |
| 2025 | 1,361 | 0.6161 | 0.5984 | 0.6080 |
| 2026 | 1,406 | 0.6944 | 0.6052 | 0.6620 |

Reliability, calibrated (test seasons):

| p from | rows | mean p | batting first won |
|---|---:|---:|---:|
| 0.0 | 44 | 0.056 | 0.000 |
| 0.1 | 101 | 0.157 | 0.099 |
| 0.2 | 166 | 0.255 | 0.078 |
| 0.3 | 403 | 0.355 | 0.146 |
| 0.4 | 856 | 0.460 | 0.322 |
| 0.5 | 1,112 | 0.544 | 0.468 |
| 0.6 | 728 | 0.648 | 0.643 |
| 0.7 | 479 | 0.748 | 0.641 |
| 0.8 | 235 | 0.846 | 0.685 |
| 0.9 | 56 | 0.925 | 0.875 |

## First-innings runs line (all evaluated seasons)

| overs bowled | rows | MAE sim | MAE baseline | in 50% sim | in 50% base | in 80% sim | in 80% base | CRPS sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-0 | 282 | 29.1 | 33.3 | 0.49 | 0.23 | 0.81 | 0.45 | 20.4 |
| 1-5 | 1,410 | 25.3 | 29.3 | 0.49 | 0.27 | 0.81 | 0.52 | 17.8 |
| 6-9 | 1,128 | 20.2 | 22.3 | 0.50 | 0.36 | 0.80 | 0.67 | 14.2 |
| 10-14 | 1,405 | 14.5 | 16.4 | 0.53 | 0.48 | 0.82 | 0.79 | 10.3 |
| 15-19 | 1,373 | 8.0 | 10.1 | 0.54 | 0.69 | 0.83 | 0.96 | 5.6 |
| all | 5,598 | 17.5 | 20.1 | 0.52 | 0.44 | 0.82 | 0.72 | 12.3 |

The runs-line ranges are not calibrated after the simulation: coverage away from 0.50 and 0.80 is the simulator's own spread.
