# Pre-registration: the in-day rank-1 rule

Written 23 Sep 2026, before either confirming sample below has been scored. It is committed before
either run is triggered, so the git history dates it.

## The candidate

- **Model:** `scripts/outcome_model.py --mode linear --blocks inday30 --keep id_`.
  - A conditional logit on the winner over the BSP market, `ln π` and `(ln π)²`, unpenalised.
  - Plus the 21 features of `model/inday_features.py`, each as z, z² and a missing flag:
    - a 30-minute gap between a source race's scheduled off and the target's;
    - `exclude_self` on.
  - Ridge from {1, 10, 100, 1,000, 10,000}, chosen on the 3 months before each quarterly fold, then refitted on everything before it.
- **Rule:** in each race, back the runner with the highest model probability for one point at BSP, when `p·(BSP−1)·0.95 − (1−p) > 0.05`. Profit is taken after 5% commission on winnings.
- **Development record** (2023-01 → 2026-03): iterations 14–16 in `reports/research_ledger.jsonl`.
  - Iteration 15: 525 bets, +21.2% (90% +7.7% to +34.3%);
  - positive in each year and in 11 of 13 folds.

## The two confirming samples, in order

1. **Gate: 2022-01-01 → 2022-12-31**, walk-forward with training from 2021.
   - No iteration has ever scored 2022; it was only ever training data, so the rule and the block were chosen without its results.
   - Run with `--first-fold 2022-01-01 --until 2023-01-01`.
   - **The gate passes if its ROI is above zero.** If it fails, the candidate is rejected and the holdout stays locked.
2. **The locked holdout: 2026-04-01 → 2026-09-22**, scored once with `--final`, and only if the gate passes.

## Success

- **Primary** (the criterion fixed at the start of this work): the holdout's ROI has a race-bootstrap 90% interval clear of zero, and both halves of the holdout are positive.
- **Secondary** (stated here because the holdout alone is weak for this rule): 2022 and the holdout pooled have a 90% interval clear of zero, and each is positive.
  - About 123 bets are expected in the holdout.
  - If the true ROI is +10% / +20%, the primary passes with probability about 15% / 33%.
  - Pooling roughly doubles the sample.

Either way both are reported, together with:
- the number of development configurations tried before this was written: 15 valid, about 75 rule variants;
- the ROI of every other EV threshold, for reading, not selection.

## Not changed after this point

Model, features, gap, ridge grid, rule, threshold, commission and samples. A failure is reported as a failure.
