# run10/oos_predictions.csv against run6/oos_predictions.csv

*Validation of this script against a known answer: run 6 (leaky, 36 features reading the race's own result) as the base, run 10 (the same period, folds and 215,760 rows on the fixed code) as the variant. The leak's size is independently known from STAKING_REPORT.md, so this is a measurement whose answer was fixed before it was taken.*

215,760 paired runners over 23,191 races.

## Mean absolute log error, paired

- base **0.3365**, variant **0.4651**
- paired difference **+0.1287** (90% CI +0.1273 to +0.1299) — negative favours the variant
- **the base wins**

## By the base model's rank in the race

```
rank     n  err_base  err_variant  delta  ci_lo  ci_hi
   1 23191    0.1787       0.3662 0.1875 0.1840 0.1911
   2 23191    0.2250       0.3866 0.1616 0.1584 0.1648
   3 23147    0.2682       0.4105 0.1423 0.1389 0.1455
  8+ 62878    0.4325       0.5333 0.1008 0.0982 0.1035
```

## Against the market

| | base | variant |
|---|---|---|
| Brier skill vs market | +0.0384 | -0.0445 |
| log loss | 0.27316 | 0.29605 |
| winner-vs-loser concordance | 0.77014 | 0.72129 |
| ...the market's | 0.75977 | 0.75977 |
| implied book (1.0 by construction since Step A) | 0.9652 | 0.9228 |

## Top pick, flat stakes

- base    +17.39% (+15.63 to +19.27) on 23,191 bets
- variant -4.05% (-6.01 to -2.15) on 23,191 bets

Reported, not decisive: at this sample the interval is wider than any difference worth acting on.

