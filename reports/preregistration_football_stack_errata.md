# Errata: the stacked football rule's pre-registration

racing² football research, 4 Nov 2026. This note corrects two descriptive statements in
`reports/preregistration_football_stack.md` (SHA-256 `f124d219…`). That file is not edited, because its hash is
registered. **The rule is unchanged:** the arm, the models, the threshold, the gate and the start rule all stand, and
the frozen files in `models/football/stack_v1` still match their recorded hashes.

## What was wrong

**1. The fit is described as "Pinnacle to Pinnacle, and Betfair to Betfair in 2024/25". It is Pinnacle only.**
In the lab matrix, 2024/25's stored close is Pinnacle's (PSC), so Betfair's 2024/25 early prices (BFE) never pair with
a Betfair close in the same-book rows. The 82,735 training rows are Pinnacle's moves, 488 of them from 2024/25.

**2. The arm is said to be chosen on "the 2024/25 fold, the only season whose same-book rows are Betfair's". That fold
is 488 Pinnacle matches.**
The lab's 2024/25 R2 figures were measured on those matches: team features 4.18%, with the five blocks 3.56%,
Avg-anchored 2.87%. The stated reason for the choice, that it was measured on Betfair's own moves, does not hold.

The threshold sweep and every trading figure (`reports/football_runs/stack_threshold_sweep_2024.json`, the "BF sym" and
"BF tick" columns) did use Betfair's own 2024/25 prices (the `bfe_open_*`/`bfe_close_*` columns). Those stand.

## What this means

- The registered model is a Pinnacle-trained move model, traded and judged on Betfair's prices.
- **Read on Betfair's 2025/26 season, which neither the arm choice nor the threshold used**
  (`reports/football_runs/stack_betfair_read.json`, `research/open_close/football_stack_betfair_read.py`; 9,245
  matches):
  - home R2 4.07%, away R2 3.64%;
  - the strongest tenth of forecasts is 68-69% the right way;
  - at its registered threshold 0.04, 531 bets: -0.53% a tick off fair (90% -1.80 to +0.65), and +1.59% at a symmetric
    book (+0.39 to +2.70).
- 2025/26 was read by earlier rules, so this is development evidence for the stack model, not a holdout.
- A model fitted on Betfair's own 2024/25 moves forecasts those moves better (R2 5.5% and 6.5%). It trades worse,
  though: at its registered threshold (0.04, 5,036 bets) it loses -1.7% at a symmetric book, and still loses at a
  matched 531 bets (threshold 0.14: +1.0% at a symmetric book, 90% from -0.3%; -3.1% a tick off fair). Its extra R2
  comes largely from the early book's own noise, which is not tradeable.
- The forward paper trial is unaffected. It scores the frozen models at the recorder's books as registered.
