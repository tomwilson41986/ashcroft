# oos_predictions_v2_profit.csv against oos_predictions_v0_l2.csv

*v2_profit (--objective profit_weighted) against the v0_l2 default. Same 2023-06-01 cached matrix, eval_from 2025-04-01, 11 folds, 107,047 runners over 11,671 races. The objective weights a runner by sqrt(1/BFSP), so a 2.0 shot carries more than four times the gradient of a 50.0 shot for the same error, and penalises quoting shorter than the horse settles by 1.5x. This is the first honest measurement of it: the run was dispatched after the fixes that made the row weights reach a custom objective's gradients and removed the ~300-round initialisation handicap.*

107,047 paired runners over 11,671 races.

## Mean absolute log error, paired

- base **0.4633**, variant **0.4780**
- paired difference **+0.0147** (90% CI +0.0138 to +0.0156) — negative favours the variant
- **the base wins**

## By the base model's rank in the race

```
rank     n  err_base  err_variant   delta   ci_lo   ci_hi
   1 11671    0.2900       0.2878 -0.0022 -0.0035 -0.0008
   2 11671    0.3568       0.3565 -0.0003 -0.0018  0.0011
   3 11641    0.4069       0.4088  0.0018  0.0001  0.0036
  8+ 30555    0.5574       0.5890  0.0316  0.0294  0.0334
```

## Against the market

| | base | variant | variant − base (90% CI) |
|---|---|---|---|
| Brier skill vs market | -0.0412 | -0.0421 | -0.00090 (-0.00170 to -0.00010) |
| log loss | 0.29706 | 0.29735 | |
| winner-vs-loser concordance | 0.72391 | 0.72316 | -0.00075 (-0.00200 to +0.00053) |
| ...the market's | 0.75732 | 0.75732 | |
| implied book (1.0 by construction since Step A) | 1.0000 | 1.0000 |

## Top pick, flat stakes

- base    -3.94% (-6.66 to -1.43) on 11,671 bets
- variant -2.35% (-5.07 to +0.31) on 11,671 bets

Reported, not decisive: at this sample the interval is wider than any difference worth acting on.


## Decision

- paired error interval excludes zero in the variant's favour: **no**
- Brier skill not worse by more than its own interval: **yes**
- concordance not worse by more than its own interval: **yes**

**The default stands.**

