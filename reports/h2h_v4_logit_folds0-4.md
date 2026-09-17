# oos_predictions_v4_logit.csv against oos_predictions_v0_l2.csv

*first half of the evaluation window. v4_logit (--target logit_norm_prob) against the v0_l2 default, same cached matrix and folds. Stability check on a variant that passed the pre-registered rule over the whole window by 0.35% of the error.*

**Folds 0-4.** 54,486 paired runners over 5,909 races, 2025-04-26 to 2025-09-22.

## Mean absolute log error, paired

- base **0.4534**, variant **0.4524**
- paired difference **-0.0010** (90% CI -0.0019 to -0.0000) — negative favours the variant
- **the variant wins**

## By the base model's rank in the race

```
rank     n  err_base  err_variant   delta   ci_lo  ci_hi
   1  5909    0.2912       0.2902 -0.0010 -0.0025 0.0004
   2  5909    0.3569       0.3555 -0.0014 -0.0032 0.0004
   3  5898    0.4012       0.4023  0.0011 -0.0012 0.0033
  8+ 15666    0.5429       0.5413 -0.0016 -0.0035 0.0002
```

## Against the market

| | base | variant | variant − base (90% CI) |
|---|---|---|---|
| Brier skill vs market | -0.0403 | -0.0404 | -0.00007 (-0.00105 to +0.00092) |
| log loss | 0.29854 | 0.29858 | |
| winner-vs-loser concordance | 0.71718 | 0.71601 | -0.00117 (-0.00278 to +0.00049) |
| ...the market's | 0.75180 | 0.75180 | |
| implied book (1.0 by construction since Step A) | 1.0000 | 1.0000 |

## Top pick, flat stakes

- base    -4.38% (-7.96 to -0.52) on 5,909 bets
- variant -4.76% (-8.42 to -0.56) on 5,909 bets

Reported, not decisive: at this sample the interval is wider than any difference worth acting on.


## Decision

- paired error interval excludes zero in the variant's favour: **yes**
- Brier skill not worse by more than its own interval: **yes**
- concordance not worse by more than its own interval: **yes**

**The variant replaces the default.**

