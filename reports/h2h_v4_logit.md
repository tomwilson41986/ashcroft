# oos_predictions_v4_logit.csv against oos_predictions_v0_l2.csv

*v4_logit (--target logit_norm_prob) against the v0_l2 default. Same matrix and folds. A reparameterisation rather than a different model: it fits the logit of the race-normalised probability instead of log(BFSP), and the served price is normalised either way.*

107,047 paired runners over 11,671 races.

## Mean absolute log error, paired

- base **0.4633**, variant **0.4617**
- paired difference **-0.0016** (90% CI -0.0023 to -0.0009) — negative favours the variant
- **the variant wins**

## By the base model's rank in the race

```
rank     n  err_base  err_variant   delta   ci_lo   ci_hi
   1 11671    0.2900       0.2891 -0.0009 -0.0020  0.0003
   2 11671    0.3568       0.3567 -0.0001 -0.0015  0.0012
   3 11641    0.4069       0.4067 -0.0003 -0.0019  0.0013
  8+ 30555    0.5574       0.5540 -0.0035 -0.0050 -0.0020
```

## Against the market

| | base | variant | variant − base (90% CI) |
|---|---|---|---|
| Brier skill vs market | -0.0412 | -0.0414 | -0.00025 (-0.00100 to +0.00051) |
| log loss | 0.29706 | 0.29713 | |
| winner-vs-loser concordance | 0.72391 | 0.72285 | -0.00106 (-0.00215 to +0.00010) |
| ...the market's | 0.75732 | 0.75732 | |
| implied book (1.0 by construction since Step A) | 1.0000 | 1.0000 |

## Top pick, flat stakes

- base    -3.94% (-6.66 to -1.43) on 11,671 bets
- variant -3.42% (-6.34 to -0.69) on 11,671 bets

Reported, not decisive: at this sample the interval is wider than any difference worth acting on.


## Decision

- paired error interval excludes zero in the variant's favour: **yes**
- Brier skill not worse by more than its own interval: **yes**
- concordance not worse by more than its own interval: **yes**

**The variant replaces the default.**

