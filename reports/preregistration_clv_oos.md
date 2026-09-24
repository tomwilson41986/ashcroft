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

## Results (appended after scoring; nothing above was changed)

The forecasts end on 6 September, the last full fold of run 28, so the window scored is 1 April – 6 September.

- **The rule: +3.54% net CLV (90% +2.86% to +4.19%) on 12,389 trades.** It passes. Every runner was −2.05%.
- **By month:**

  | Apr | May | Jun | Jul | Aug | Sep (to 6th) |
  |---|---|---|---|---|---|
  | +0.96% (interval crosses zero) | +3.60% | +4.00% | +5.68% | +3.71% | +2.71% |

- **Held to settlement:** −6.8% at BSP. It is a trading edge, not a value bet.
- **The same-day check, reported as promised: the edge now sits in later races.**
  - First three races of the day: −0.05%.
  - Race ten onwards: +4.96%.
  - By off time: about 0 at 13:00, rising to +4% to +8% in the evening.
  - In January–March the first races carried the largest share.
- **Post hoc, late non-runners explain only part of it.** Races where the morning book of the runners who ran is short return +9.5% to +27% (462 trades); without them the rule makes +3.25% (+2.60 to +3.93).
- **Two readings remain:**
  1. The morning price is staler before later races. That edge would be tradeable.
  2. The backtest forecast knows race-day facts, such as the going on the day and jockey changes, from the result rows. That edge would not.
- **The deciding test** is the live 06:00 record. See `reports/clv_betfair_2026q1.md` on #73.

## Post-hoc diagnostics (23 Sep, after the result; descriptive, not a test)

These come from `scripts/clv_diagnostics.py`, run on the same forecasts and prices. They ask whether the later-race edge follows the clock (a staler morning price) or the race's order at its meeting (race-day facts that build up through a card). They also measure two race-day channels directly:
- going changes during the meeting, read from the result rows;
- late withdrawals, whose reduction factors are estimated from the runners' morning book.

| | Jan–Mar (set the rule) | Apr–Sep (out of sample) |
|---|---|---|
| the rule | +6.10% (4,348) | +3.54% (12,389) |
| with estimated reduction factors | +5.39% | +2.87% |
| no sign of a withdrawal and no going change during the meeting | **+4.66%** (+3.04 to +6.43; 3,516) | **+2.45%** (+1.73 to +3.13; 10,310) |
| races after a going change | +14.05% (255) | +10.43% (778) |
| per hour after 11:00 | **−1.99%** (−2.78 to −1.25) | **+0.84%** (+0.46 to +1.24) |
| per race of meeting order | +1.61% (+0.67 to +2.62) | +0.29% (−0.12 to +0.71) |
| first race, afternoon meeting | +9.30% (555) | −0.58% (1,397) |
| first race, evening meeting | −6.13% (53) | +4.87% (367) |

What this says:
- **The two channels that can be measured account for only part of the edge.**
  - Going changes during a meeting inflate it where they happen, but they cover about 6% of bets.
  - Withdrawals take about 0.7 points off.
  - Without either channel, the edge is positive in both windows.
- **The timing is not stable, so it supports neither reading.**
  - In winter the edge was largest in early-afternoon races and fell with the clock.
  - In summer it rises into the evening.
  - A staler morning price would show the same clock gradient in both seasons.
  - The seasonal flip lines up with the code of racing: winter afternoons are jump racing, summer evenings are flat.
- **Still unmeasured:**
  - going changes between 06:00 and a meeting's first race;
  - jockey replacements.

  The morning card the 06:00 job now keeps will measure both.
- **The forward test** (`reports/preregistration_clv_forward.md`) remains the decider.
