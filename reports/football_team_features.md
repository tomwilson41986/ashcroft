# Football team features: strength, strength of schedule, performance added (2 Nov 2026, development only)

**Bottom line: none of the families adds meaningfully to the market's price for the result, but together they forecast
part of the move from Pinnacle's early price to its close.** In the outcome test the best family (points and goal
difference above the close's expectation) gains 0.12 per thousand in log-loss over the open and our model, with a 90%
interval from -0.01 to +0.26. That is about 4% of what the close itself adds over the open (3.4 per thousand,
`reports/football_model.md` s.3), and not significant. Strength of schedule adds nothing (-0.03, interval -0.11 to
+0.05). In the move test every family together explains 4.6% of the open-to-close move out of sample, against a
permuted floor of -0.07%. In the tenth of matches with the largest forecasts, a trade on the forecast's side would catch
a 4.0% move at fair prices, the right way 65.5% of the time, before any spread or commission. The holdout (from 2025/26)
has not been read.

Figures: `reports/football_runs/team_features_dev.json` and `reports/football_runs/team_features_dev_move_size.json`,
from `python -m football.feature_eval` on football-data.co.uk fetched in this session (`source_data.py --source
football,openfootball --fetch --build`, then `football.data --build`, local root). Code: `football/team_features.py`,
`football/feature_eval.py`. Tests: `tests/test_football_team_features.py`, including the lag-safety rewrite test (rewrite
the last day's scores, shots and closes; every feature of that day must be unchanged). It caught a same-day leak in the
windows during development, which was fixed before any figure here was produced.

## What was built

| Family | Features (home minus away unless noted) | Built from |
|---|---|---|
| Team strength (market) | `r_diff`: market-implied strength, attack plus defence in log goal-rate units, from a decayed ridge fit of the goal rates implied by past closes, refitted weekly per country; `d_own_r_d5`: its change over the last five matches | Closing 1X2 and O/U 2.5, inverted to two Poisson rates |
| Strength of schedule | `d_opp_r_*`: mean pre-match market rating of the opponents faced (last 5, last 10, EWM half-life 8 matches, season to date) | As above |
| Raw form | points, goal difference, shots-on-target difference over the same four windows | Results |
| Performance added vs the model | points (`pae_dc`) and goal difference (`gae_dc`) above the Dixon-Coles pre-match expectation | Results, walk-forward Dixon-Coles |
| Performance added vs the market | points (`pae_mkt`) and goal difference (`gae_mkt`) above the close's expectation (the horse model's A-E idea) | Results, closes |
| Disagreement | `dis_open`: our model's log supremacy less the open's | Dixon-Coles, open |

Every window is over a side's earlier matches only and day-lagged: nothing from the match day enters.

## Rows

299,619 finished matches with features. 109,325 evaluated: both sides with five or more earlier matches, the open and
the Dixon-Coles price present. By season, from 6,848 (2019/20, curtailed) to 7,797. Outcome test: 68,535 matches scored
across 2016/17 to 2024/25, each season fitted on the seasons from 2013/14 before it. Move test: 60,807 matches whose open
and close are from the same book. Every one of them is Pinnacle to Pinnacle, because Betfair's prices start in 2024/25,
which is in the scored years only for its first season. Shots on target are missing for 14-25% of the windowed rows
(lower divisions); missing values are set to the training mean.

## The outcome test (log-loss gain per thousand over open + model; 90% interval, bootstrap by match day)

| Family | Gain | 90% interval |
|---|---|---|
| Strength (market) | -0.02 | -0.04 to +0.01 |
| Strength of schedule | -0.03 | -0.11 to +0.05 |
| Raw form | 0.00 | -0.20 to +0.19 |
| Performance added vs model | +0.09 | -0.03 to +0.21 |
| Performance added vs market | +0.12 | -0.01 to +0.26 |
| Disagreement | +0.03 | -0.04 to +0.09 |
| All | -0.02 | -0.27 to +0.23 |

Against the open alone, performance added vs market is +0.15 (+0.002 to +0.29), the only interval above zero, and only
just. Our model's own weight beside the open is again negative (-0.063, -0.108 to -0.029), as in `football_model.md`.

## The move test (out-of-sample R2 of the open-to-close move in implied log supremacy)

| Family | R2 vs mean | Permuted floor | Sign right |
|---|---|---|---|
| Strength (market) | +1.27% | -0.04% | 55.5% |
| Strength of schedule | -0.01% | +0.03% | 50.6% |
| Raw form | +1.61% | -0.02% | 55.8% |
| Performance added vs model | +0.63% | -0.02% | 53.1% |
| Performance added vs market | +0.35% | -0.07% | 52.1% |
| Disagreement | -0.08% | 0.00% | 51.3% |
| All | +4.60% | -0.07% | 58.0% |

The move a trade would catch, home price, by the forecast's size (all families; fair prices, before costs):

| Forecast decile | Matches | Mean forecast | Mean move caught | Right way |
|---|---|---|---|---|
| 8 | 6,080 | 1.8% | +2.0% | 59.7% |
| 9 | 6,081 | 2.3% | +2.6% | 61.4% |
| 10 | 6,081 | 3.6% | +4.0% | 65.5% |

## What this means, and what it does not

- **For the result, team features are already in the price.** Strength, schedule and form measured against the model or
  the market add nothing a bettor could use. This matches every earlier football finding here.
- **The move is the lead, but it is Pinnacle's, not a tradeable one yet.** football-data's "open" is the price at its
  Friday or Tuesday collection, not a timed book, and Pinnacle cannot be laid. On Betfair a trade pays a tick at each end
  and 2% commission on the net. A tick is 0.02 from 2 to 3 (0.7-1.0% of the price) and 0.05 from 3 to 4 (1.25-1.7%), so a
  tick at each end costs about 1.4-3.4% in that range, more at longer prices. The top decile's 4.0% at fair prices clears
  that at the shorter prices on paper. But the earlier exchange study found a forecastable move that a tick at each end
  removed in Betfair's takeable books (`reports/open_close/model_football_exchange.md`), so this needs Betfair's own
  prices, not Pinnacle's, before it means anything.
- **Not read:** the holdout from 2025/26, any Betfair book, any cost. Nothing here is a licence to trade.

## Next

1. Score the same forecast on Betfair's own early and closing prices (football-data BFE/BFEC, 2024/25 on) a tick worse at
   each end. 2024/25 is development; 2025/26 on is the holdout. Pre-register the rule before reading the holdout.
2. Then on the recorder's books with sizes (`research/open_close/football_paper_trial.py` is the scorer's pattern),
   once the paper-models runner records them.
3. Player level: the API-Football history (49,205 matches from 2020, built 2 Nov 2026) for minutes-weighted player
   ratings and the confirmed-eleven difference.

## Betfair's own prices: development and the holdout (read once, 2 Nov 2026)

**Bottom line: the forecast holds on Betfair's prices and on new data, but a tick at each end takes all of it.** The
pre-registered rule (`reports/preregistration_football_team_move.md`, SHA-256 e08e604d..., committed before the holdout
was read) failed its gate. Figures: `reports/football_runs/team_move_dev.json`, `reports/football_runs/team_move_holdout.json`;
code: `research/open_close/football_team_move.py`.

The trade: the forecast fitted on Pinnacle's moves only (2013/14-2023/24 for development, to 2024/25 for the holdout),
applied to football-data's Betfair early (BFE) and closing (BFEC) prices; GBP1 on the home or away side, back then lay
(or lay then back), greened; a tick worse at each end on the fair prices, 2% commission on each market's net.

| Read | Books | Bets | Tick worse each end | 90% interval | At the mid | CLV | Right way | Back side as quoted |
|---|---|---|---|---|---|---|---|---|
| Development 2024/25 | 1.05 or under, threshold 0.03 | 1,316 | -0.40% | -1.15 to +0.40 | +3.85% | | 63.5% | +0.06% |
| Development 2024/25 | wide (over 1.05), 0.03 | 527 | +7.16% | +4.25 to +10.41 | +11.13% | | 69.4% | -8.37% |
| **Holdout, primary** (Jul 2025-1 Nov 2026) | 1.05 or under, 0.03 | 1,827 | **-1.10%** | **-1.64 to -0.54** | +3.27% | +1.13% | 64.0% | +0.50% |
| Holdout, secondary (found by looking) | 1.02-1.05, 0.03 | 775 | +0.37% | -0.50 to +1.22 | +4.62% | +2.59% | 67.4% | -0.04% |
| Holdout, for the record | wide, 0.03 | 543 | +4.92% | | | | | |

Gate (at least 300 bets, lower bound above zero, CLV above zero): primary fails on the lower bound; the secondary fails
on the lower bound too. In wide books the profit is again the stale quote: backing at Betfair's recorded early price
there loses (development -8.4%).

What it does show:

- **The forecast is real and holds out of sample on Betfair:** 64% the right way and +3.3% at the mid on 1,827 holdout
  bets in takeable books, from a model that never saw a Betfair price.
- **The cost is the whole problem.** Paying a tick to get in and a tick to get out turns +3.3% into -1.1%.

What could still be tested (not pre-registered, ideas only):

- **Resting orders:** offering at our own price one tick inside the spread before the move, instead of taking the
  quote, could save a tick at each end. Only the recorder's books with sizes can show whether such orders fill and how
  often they are picked off (`reports/football_quant_approach.md` s.6.4, rule 4).
- **Holding to the result instead of greening:** the entry beats the fair close by +1.13% (CLV) after paying the tick
  in, which commission on winnings roughly cancels. A test, not a finding.

## Corrected figures after the QA review (2 Nov 2026, commit aa6f196)

The QA review (`reports/football_review_and_roadmap.md` s.2.5) found that the weekly refits could let a Saturday result
or close into a Sunday or Monday feature whose early price was taken on the Friday. It also found a season-window
overlap and an optimistic cost model. Every figure above was re-run on the corrected code; the JSON files in
`reports/football_runs/` now hold the corrected run (the first run is in git history).

| Figure | First run | Corrected |
|---|---|---|
| Outcome, performance added vs market, gain over open + model (per thousand) | +0.12 (-0.01 to +0.26) | **+0.17 (+0.02 to +0.32)** |
| Outcome, performance added vs model | +0.09 (-0.03 to +0.21) | +0.14 (+0.004 to +0.28) |
| Outcome, strength of schedule | -0.03 (-0.11 to +0.05) | -0.01 (-0.06 to +0.03) |
| Move, all families, R2 vs mean (permuted floor) | 4.60% (-0.07%) | 4.51% (-0.13%) |
| Move, top decile caught, right way | 4.03%, 65.5% | 3.98%, 65.9% |
| Betfair, development, takeable, threshold 0.03, a tick off fair | -0.40% (-1.15 to +0.40) | -0.29% (-1.06 to +0.45) |
| Betfair, holdout, registered rule, a tick off fair | -1.10% (-1.64 to -0.54) | -1.11% (-1.67 to -0.61), 1,926 bets |

The leak was small: no conclusion moves. Two outcome families now have lower bounds just above zero. With 7 families
read against 2 baselines at 90%, about one such result is expected by chance, and the effect is about 5% of what the
close adds over the open. It is a lead for the stack, not a finding.

**The cost model decides the Betfair result (`reports/football_runs/team_move_cost_check.json`).** football-data
records only Betfair's back prices, so the lay price at the early book has to be assumed:

| Holdout, takeable books, threshold 0.03 | Backs (611) | Lays (1,315) | All (1,926) |
|---|---|---|---|
| A tick off the fair price (the registered headline) | | | -1.11% (-1.67 to -0.61) |
| Recorded back; lay one tick over the back | +0.29% (-0.50 to +1.08) | +3.11% (+2.55 to +3.64) | +2.22% (+1.70 to +2.72) |
| Recorded back; lay from a book symmetric about the fair price | -0.52% (-1.31 to +0.26) | -0.01% (-0.56 to +0.52) | -0.17% (-0.68 to +0.33) |

A symmetric book implies a median early spread of 4 ticks in the holdout (3 in development), so "one tick over the
back" flatters every lay. The realistic reading is break-even. The registered verdict stands (fail, on its registered
headline). The honest summary is that **the rule is between -1.1% and +2.2% depending only on a price we did not record.**
The recorder's best lay with sizes settles it, which makes it the most valuable data we can get. This re-read used the
holdout a second time, for the correction only; nothing was selected on it.
