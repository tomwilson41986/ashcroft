# Baseline: logistic on required run rate, IPL chases

Source tables: `data/cricket/store/cricket/tables` (Cricsheet). Built by `python -m cricket.model.baseline`.
Rows: the state before every delivery of the second innings, decided non-D/L men's IPL matches; 41,392 rows from 356 matches in the test seasons 2022-2026. Walk-forward: each season scored by a fit on IPL seasons 2008 to the season before only.

| model | log loss | Brier | ECE |
|---|---:|---:|---:|
| rrr | 0.4729 | 0.1538 | 0.0652 |
| rrr_wkts_balls | 0.4363 | 0.1413 | 0.0694 |
| published (Mishra) | 0.4379 | | 0.0773 |

Published figure: Mishra, arXiv 2608.14696 (research brief; not re-verified). Its rows and split are not known, so the comparison is indicative only.

## By season

| season | rows | rrr log loss | rrr ECE | rrr+wkts+balls log loss | rrr+wkts+balls ECE |
|---|---:|---:|---:|---:|---:|
| 2022 | 8,693 | 0.4950 | 0.0539 | 0.4391 | 0.0486 |
| 2023 | 8,428 | 0.5631 | 0.0644 | 0.5150 | 0.0548 |
| 2024 | 8,210 | 0.4319 | 0.0804 | 0.4048 | 0.0660 |
| 2025 | 7,908 | 0.4166 | 0.0913 | 0.4052 | 0.0690 |
| 2026 | 8,153 | 0.4518 | 0.1724 | 0.4139 | 0.1752 |

## Reliability (rrr+wkts+balls, all test seasons)

| p from | rows | mean p | chase won |
|---|---:|---:|---:|
| 0.0 | 8,500 | 0.030 | 0.048 |
| 0.1 | 4,755 | 0.149 | 0.187 |
| 0.2 | 4,364 | 0.250 | 0.312 |
| 0.3 | 3,407 | 0.349 | 0.456 |
| 0.4 | 3,332 | 0.450 | 0.574 |
| 0.5 | 3,350 | 0.549 | 0.689 |
| 0.6 | 2,806 | 0.649 | 0.800 |
| 0.7 | 2,960 | 0.749 | 0.852 |
| 0.8 | 3,145 | 0.851 | 0.919 |
| 0.9 | 4,773 | 0.956 | 0.974 |
