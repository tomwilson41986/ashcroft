# Pre-registration: the stacked team-feature move rule, paper-traded at the recorded Betfair books

racing² football research. Written 2026-11-03T09:00Z by the modelling agent, after the development screens and the
threshold sweep on 2024/25, and before any forward book of this rule was read. Code:
`research/open_close/football_stack_trial.py` (sweep, freeze, score) on the lab matrix (`football/lab.py`,
`docs/FOOTBALL_FAST_LOOP.md`). The scorer reads the JSON block at the end of this file and records this file's SHA-256
beside every result.

## What development showed

Sources: `reports/football_runs/lab_ledger.jsonl` (the entry of 2026-11-03T08:55Z, matrix
`matrix_00311bce762c678f.parquet`) and `reports/football_runs/stack_threshold_sweep_2024.json`.

The target is the early-to-close move of each side's chance, log(close / early) on de-vigged prices. Rows are same-book
only: Pinnacle to Pinnacle up to 2023/24, Betfair to Betfair in 2024/25. Scoring is walk-forward by season,
2016/17-2024/25, on 60,805 matches.

| Arm (stack = mean of ridge and LightGBM) | Pooled R2 | Fold SE | 2024/25 R2 (Betfair fold) | 2024/25 at 0.03, symmetric book |
|---|---|---|---|---|
| `team_all` (35 team features) | 4.70% | 0.29 | **4.18%** | +1.53% (90% +0.83 to +2.36, 1,007 bets) |
| `team_all` + five blocks (64) | 5.68% | 0.40 | 3.56% | +1.57% (1,140 bets) |
| `team_all` less `dis_open`, + Avg anchor (46) | 4.78% | 0.40 | 2.87% | +1.40% (1,206 bets) |

The five blocks win on the pooled Pinnacle seasons but lose on the Betfair fold. They read soft books against the sharp
open, and the sharp open is not the book traded here.

### Threshold sweep, `team_all` stack, 2024/25 only

Models were fitted on 2013/14-2023/24. Rows are takeable books only (Betfair early back sum 1.05 or under). Commission
is 2% on each market's net.

| Threshold | Bets | A tick worse | 90% low | Symmetric book | 90% low | Right way |
|---|---|---|---|---|---|---|
| 0.01 | 6,169 | -1.71% | -2.02 | -0.68% | -0.97 | 59.8% |
| 0.02 | 2,616 | -0.99% | -1.48 | +0.41% | -0.06 | 62.9% |
| 0.03 | 1,007 | -0.27% | -0.97 | +1.53% | +0.83 | 64.7% |
| **0.04** | **416** | **+1.58%** | **+0.20** | **+3.82%** | **+2.43** | **65.9%** |
| 0.05 | 190 | +4.17% | +2.10 | +7.01% | +4.97 | 73.2% |

These are football-data's untimed Betfair prices, not a book with sizes.

The 2024/25 season has now been read by about 50 lab configurations (the ledger) and by the earlier rules. The earlier
linear rule showed +0.88% on 2024/25 and -0.17% on its holdout under the same costing. Read the table as the case for a
paper trial, not as an edge.

## The rule, and how each part was chosen

### Arm: `team_all`

The 35 team features of `football.feature_eval.FAMILIES["all"]`: points, goal difference, shots on target, and
performance against Dixon-Coles and against the market. Each comes over the last 5 and last 10 matches, an EWM and
the season, plus strength of schedule, rating change and `dis_open`.

The rule was stated before choosing: **the stack arm with the best R2 on the 2024/25 fold**, the only season whose
same-book rows are Betfair's. That is the market this rule trades.

`team_all` is best (4.18% against 3.56% and 2.87%). It is also the arm whose features mean the same in training and
live:

- `dis_open` is our model against the early price of the row's own book. That is Betfair's in 2024/25 and in every
  forward match.
- The block arms read Pinnacle's open, which is stale from 23 Jul 2025 and absent from Feb 2026.

### Model

The mean of a ridge regression (alpha `lab.RIDGE_ALPHA`, features standardised on the training rows and clipped at
`lab.CLIP` SD) and a LightGBM model with the league as a category. One model a side, home and away.

The models are fitted **once**, on every same-book move of 2013/14-2024/25 (82,735 rows), and frozen. No refitting
during the trial. The frozen files are in `models/football/stack_v1/`:

| File | SHA-256 |
|---|---|
| `spec.json` | `22e1073fb826d174406ac86ee541551095bdfc8d7b1874410fd7bec4351c7df0` |
| `gbm_h.txt` | `3b430d931eecaae7c071dce6b9a6e36431a16e592e492ec79d1a9b69703a97cf` |
| `gbm_a.txt` | `86072d0ccaa24f754b701ae03844b3169e660c50ead92dac801f56b3f828e9aa` |

### Threshold: 0.04

The chooser was written in `sweep()` before it ran: **the best 90% lower bound at a symmetric book among thresholds
with at least 300 bets on 2024/25**. 0.05 has 190 bets, so 0.04 wins.

### How a trade works

- **Market:** Betfair MATCH_ODDS, home and away selections, never the draw. The 22 main divisions football-data
  prices: B1, D1, D2, E0, E1, E2, E3, EC, F1, F2, G1, I1, I2, N1, P1, SC0, SC1, SC2, SC3, SP1, SP2, T1.
- **Features:** taken from the lab matrix rebuilt on the data as it stands. Every feature is lagged to the match's
  collection day: Friday 17:00 UK for Friday-Monday matches, Tuesday 13:00 UK for Tuesday-Thursday. That is tested
  for every column (`tests/test_football_team_features.py`, `tests/test_football_lab.py`), so a later score equals
  a live one.
- **Signal:** the price forecast is minus the chance forecast.
  - Forecast shorter by more than 0.04: back at the recorded early book, lay at the recorded close.
  - Forecast longer by more than 0.04: lay, then back.
  - Every trade is greened.
- **Execution:** the paper-trial scorer's (`research/open_close/football_paper_trial.py`, steps 4-6):
  - Early book: the first recorded snapshot within 60 min after the collection moment, pre-play. Its best-back sum
    must be 1.05 or under (tight or medium).
  - Fill: at the recorded best back or best lay, only if the size shown covers GBP10. Otherwise the trade is
    skipped and logged.
  - Close: the last two-sided book before in-play.
- **Headline:** the return per stake unit at the recorded quotes, 2% commission on each market's net, with its 90%
  bootstrap interval by day (2,000 draws).

## Start and window

The trial starts at the first Friday collection on or after the first day the football recorder holds books, and not
before this file was written. It runs until the later of the 300th trade and 90 days, capped at 365 days. Until then the
gate reads "running", never a verdict.

The football recorder runs on paper-models.yml. **That runner is not yet registered**, so no forward book exists yet.

## Gate

The trial passes only if all of these hold:

- at least 300 trades;
- at least 90 days;
- the 90% lower bound of the return at the recorded quotes above zero;
- the mean CLV at the taken price above zero;
- both halves of the window above zero.

A pass licenses a decision on small real stakes by the owner. It is not itself a licence to stake. A fail retires this
rule: the threshold and arm are not re-tuned on the trial's books.

## The expectation stated in advance

The primary is expected to **fail or read close to zero**:

- The only positive development figures are on a season read many times.
- A tick worse, the 0.04 lower bound is +0.20%, barely above zero.
- The early lay spread is unrecorded.
- The earlier rules lost about their development edge on holdout.

## Power

2024/25 gave 416 takeable signals in a full season of football-data's prices. The recorder's book filter and the size
check will cut that, so 300 trades will take most of a season.

At 416 bets the 90% interval was about ±1.4 points wide. The trial can therefore tell an edge of about 1.5% or more
from zero, and no smaller one.

```json
{
 "model": "stack_ridge_lgbm_team_all_frozen",
 "arm": "team_all",
 "models_dir": "models/football/stack_v1",
 "model_sha256": {
  "spec.json": "22e1073fb826d174406ac86ee541551095bdfc8d7b1874410fd7bec4351c7df0",
  "gbm_h.txt": "3b430d931eecaae7c071dce6b9a6e36431a16e592e492ec79d1a9b69703a97cf",
  "gbm_a.txt": "86072d0ccaa24f754b701ae03844b3169e660c50ead92dac801f56b3f828e9aa"
 },
 "market_types": ["MATCH_ODDS"],
 "selections": ["H", "A"],
 "leagues_fd": ["B1", "D1", "D2", "E0", "E1", "E2", "E3", "EC", "F1", "F2", "G1", "I1", "I2", "N1", "P1",
                "SC0", "SC1", "SC2", "SC3", "SP1", "SP2", "T1"],
 "book_width": ["tight <1.02", "medium 1.02-1.05"],
 "threshold": 0.04,
 "commission": 0.02,
 "notional_gbp": 10.0,
 "headline": "return at the recorded quotes",
 "gate": {"min_bets": 300, "min_days": 90, "cap_days": 365, "roi_lo90_above": 0, "clv_mean_above": 0,
          "both_halves_above": 0},
 "start": "first Friday collection on or after the first recorded football book, not before written_utc",
 "written_utc": "2026-11-03T09:00Z"
}
```
