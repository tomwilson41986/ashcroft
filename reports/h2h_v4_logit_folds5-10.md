# oos_predictions_v4_logit.csv against oos_predictions_v0_l2.csv

*second half of the evaluation window. v4_logit (--target logit_norm_prob) against the v0_l2 default, same cached matrix and folds. Stability check on a variant that passed the pre-registered rule over the whole window by 0.35% of the error.*

**Folds 5-10.** 52,561 paired runners over 5,762 races, 2025-09-23 to 2026-03-21.

## Mean absolute log error, paired

- base **0.4735**, variant **0.4713**
- paired difference **-0.0022** (90% CI -0.0032 to -0.0014) — negative favours the variant
- **the variant wins**

## By the base model's rank in the race

```
rank     n  err_base  err_variant   delta   ci_lo   ci_hi
   1  5762    0.2887       0.2879 -0.0008 -0.0022  0.0009
   2  5762    0.3567       0.3580  0.0013 -0.0007  0.0033
   3  5743    0.4128       0.4111 -0.0017 -0.0041  0.0007
  8+ 14889    0.5728       0.5673 -0.0054 -0.0076 -0.0033
```

## Against the market

| | base | variant | variant − base (90% CI) |
|---|---|---|---|
| Brier skill vs market | -0.0421 | -0.0426 | -0.00043 (-0.00141 to +0.00067) |
| log loss | 0.29552 | 0.29562 | |
| winner-vs-loser concordance | 0.73090 | 0.72996 | -0.00094 (-0.00260 to +0.00070) |
| ...the market's | 0.76305 | 0.76305 | |
| implied book (1.0 by construction since Step A) | 1.0000 | 1.0000 |

## Top pick, flat stakes

- base    -3.48% (-7.07 to +0.41) on 5,762 bets
- variant -2.04% (-5.78 to +1.81) on 5,762 bets

Reported, not decisive: at this sample the interval is wider than any difference worth acting on.


## Decision

- paired error interval excludes zero in the variant's favour: **yes**
- Brier skill not worse by more than its own interval: **yes**
- concordance not worse by more than its own interval: **yes**

**The variant replaces the default.**

