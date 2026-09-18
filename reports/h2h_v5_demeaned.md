# oos_predictions_v5_demeaned.csv against oos_predictions_v0_l2.csv

*v5_demeaned (--target demeaned_log) against the v0_l2 default. Same matrix and folds. It fits within-race differences in log(BFSP), dropping the race-level term that normalisation cancels anyway.*

107,047 paired runners over 11,671 races.

## Mean absolute log error, paired

- base **0.4633**, variant **0.4636**
- paired difference **+0.0004** (90% CI -0.0006 to +0.0013) — negative favours the variant
- **no difference the data can resolve**

## By the base model's rank in the race

```
rank     n  err_base  err_variant   delta   ci_lo   ci_hi
   1 11671    0.2900       0.2902  0.0003 -0.0013  0.0020
   2 11671    0.3568       0.3555 -0.0013 -0.0035  0.0009
   3 11641    0.4069       0.4026 -0.0043 -0.0067 -0.0019
  8+ 30555    0.5574       0.5633  0.0059  0.0038  0.0078
```

## Against the market

| | base | variant | variant − base (90% CI) |
|---|---|---|---|
| Brier skill vs market | -0.0412 | -0.0421 | -0.00087 (-0.00203 to +0.00038) |
| log loss | 0.29706 | 0.29728 | |
| winner-vs-loser concordance | 0.72391 | 0.72389 | -0.00002 (-0.00128 to +0.00144) |
| ...the market's | 0.75732 | 0.75732 | |
| implied book (1.0 by construction since Step A) | 1.0000 | 1.0000 |

## Top pick, flat stakes

- base    -3.94% (-6.66 to -1.43) on 11,671 bets
- variant -2.80% (-5.78 to +0.05) on 11,671 bets

Reported, not decisive: at this sample the interval is wider than any difference worth acting on.


## Decision

- paired error interval excludes zero in the variant's favour: **no**
- Brier skill not worse by more than its own interval: **yes**
- concordance not worse by more than its own interval: **yes**

**The default stands.**

