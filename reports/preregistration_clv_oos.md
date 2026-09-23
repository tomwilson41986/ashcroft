# Pre-registration: the early-price trade, out of sample

Written 23 Sep 2026 at about 17:35 UTC. At that point run 28, which makes the forecasts, was still running, and no forecast for April–September 2026 had been produced.

## The rule, unchanged from #73 (`scripts/clv_betfair.py`, defaults)

- **Back** a runner at Betfair's morning volume-weighted price when the model's BSP forecast is at least 22% shorter (`ln(morning / forecast) ≥ 0.2`) and at least £100 was matched in the morning.
- **Close** at BSP. Net CLV per unit is `morning / BSP − 1`, less 5% commission when positive.
- **Passes** if the 90% race-bootstrap interval of the mean net CLV is above zero.

## What is new

- **Window: 2026-04-01 → 2026-09-21**, the Betfair files' last day.
  - The rule was fixed on 1 Jan – 21 Mar 2026, so this window never set it.
  - Jan–Mar is reported alongside, from the same forecasts, for comparison only.
- **Forecaster:** run 28, `train-bfsp.yml` `evaluate_only`, variant `prod_logit_repaired`, `eval_from` 2025-12-11.
  - It uses the deployed training recipe on the **repaired** features of #74: day-lagged priors, EPF, lengths parser, debut flag. These are the features a live system would use after the merge.
  - #73 fixed the rule on the pre-repair forecasts. The difference is disclosed, not hidden: if the rule fails here, the repair may be part of why.

## Reported, whatever the outcome

- The headline and its interval.
- Month by month.
- The same-day-leak check (first race at each meeting).
- Price bands.
- The trade held to settlement, at the morning price and at BSP.
