# oos_predictions_v1_l2_decay1.csv against oos_predictions_v0_l2.csv

*v1_l2_decay1 (--decay-rate 1.0) against the v0_l2 default. Same 2023-06-01 cached matrix, eval_from 2025-04-01, 11 folds, 2025-04-26 to 2026-03-21. The only difference is the exponential recency weight exp(-days/365), which gives a race a year old 37% of today's weight.*

107,047 paired runners over 11,671 races.

## Mean absolute log error, paired

- base **0.4633**, variant **0.4644**
- paired difference **+0.0011** (90% CI +0.0004 to +0.0018) — negative favours the variant
- **the base wins**

## By the base model's rank in the race

```
rank     n  err_base  err_variant   delta   ci_lo  ci_hi
   1 11671    0.2900       0.2895 -0.0005 -0.0018 0.0008
   2 11671    0.3568       0.3571  0.0003 -0.0012 0.0018
   3 11641    0.4069       0.4057 -0.0012 -0.0029 0.0004
  8+ 30555    0.5574       0.5598  0.0024  0.0009 0.0038
```

## Against the market

| | base | variant | variant − base (90% CI) |
|---|---|---|---|
| Brier skill vs market | -0.0412 | -0.0424 | -0.00117 (-0.00195 to -0.00047) |
| log loss | 0.29706 | 0.29736 | |
| winner-vs-loser concordance | 0.72391 | 0.72300 | -0.00091 (-0.00208 to +0.00025) |
| ...the market's | 0.75732 | 0.75732 | |
| implied book (1.0 by construction since Step A) | 1.0000 | 1.0000 |

## Top pick, flat stakes

- base    -3.94% (-6.66 to -1.43) on 11,671 bets
- variant -3.32% (-6.06 to -0.75) on 11,671 bets

Reported, not decisive: at this sample the interval is wider than any difference worth acting on.


## Decision

- paired error interval excludes zero in the variant's favour: **no**
- Brier skill not worse by more than its own interval: **yes**
- concordance not worse by more than its own interval: **yes**

**The default stands.**

