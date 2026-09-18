# oos_predictions_v3_profit_decay1.csv against oos_predictions_v0_l2.csv

*v3_profit_decay1 (--objective profit_weighted --decay-rate 1.0) against the v0_l2 default. Same matrix and folds. This pairing is only meaningful because of a fix made today: LightGBM discards a Dataset weight for a custom objective, so before it this run was a byte-identical duplicate of v2_profit and the recency decay had never reached a gradient in production either.*

107,047 paired runners over 11,671 races.

## Mean absolute log error, paired

- base **0.4633**, variant **0.4784**
- paired difference **+0.0151** (90% CI +0.0142 to +0.0160) — negative favours the variant
- **the base wins**

## By the base model's rank in the race

```
rank     n  err_base  err_variant   delta   ci_lo  ci_hi
   1 11671    0.2900       0.2884 -0.0016 -0.0029 0.0000
   2 11671    0.3568       0.3569  0.0000 -0.0016 0.0015
   3 11641    0.4069       0.4094  0.0024  0.0006 0.0042
  8+ 30555    0.5574       0.5880  0.0306  0.0284 0.0324
```

## Against the market

| | base | variant | variant − base (90% CI) |
|---|---|---|---|
| Brier skill vs market | -0.0412 | -0.0422 | -0.00101 (-0.00178 to -0.00020) |
| log loss | 0.29706 | 0.29734 | |
| winner-vs-loser concordance | 0.72391 | 0.72276 | -0.00115 (-0.00237 to -0.00001) |
| ...the market's | 0.75732 | 0.75732 | |
| implied book (1.0 by construction since Step A) | 1.0000 | 1.0000 |

## Top pick, flat stakes

- base    -3.94% (-6.66 to -1.43) on 11,671 bets
- variant -3.85% (-6.61 to -1.14) on 11,671 bets

Reported, not decisive: at this sample the interval is wider than any difference worth acting on.


## Decision

- paired error interval excludes zero in the variant's favour: **no**
- Brier skill not worse by more than its own interval: **yes**
- concordance not worse by more than its own interval: **yes**

**The default stands.**

