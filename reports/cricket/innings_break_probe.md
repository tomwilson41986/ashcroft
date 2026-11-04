# Cricket - innings-break probe: the chase model against Betfair's price at the break

Source: the owner's Betfair BASIC bundle (`data/cricket/store/betfair_historic/raw/cricket`), Cricsheet, the walk-forward 2026 model. Built by `python -m cricket.model.innings_break`. Rules fixed before the results: see the module.

Matched 2026 markets: 218; break priced: 123; not priced: no first-innings market with an in-play suspension 92; no Match Odds trade inside 10 minutes of the price time 3.

## Was the break found?

- Minutes from the break to the chase's 6-over suspension: median 44, range 3-148 (80 events with that market).
- The market's move across the break (logit, close to break) against the first-innings total less the model's pre-match median: correlation -0.86 (a market that has seen the innings moves against the chasing side as the total beats expectation: negative).
- Model and market at the break, correlation of the logits: 0.83.

## Chase won: probabilities at the break

| probability | log loss |
|---|---:|
| Betfair at the break (last traded, overround removed) | 0.5331 |
| model, chase from 0 for 0 against the real target | 0.6102 |
| for reference: Betfair's pre-match close | 0.6521 |

Model minus market: +0.0770 (95% interval +0.0180 to +0.1422, 123 matches).

| Benter term | weight | s.e. | z |
|---|---:|---:|---:|
| logit(model p) | 0.018 | 0.268 | 0.07 |
| logit(market p) | 0.973 | 0.309 | 3.15 |

The model adds to the price at the break (z above 1.96): **no**.

## Sensitivity (not the pre-registered result): breaks 25-75 minutes before the chase's 6-over suspension

76 events. Betfair at the break 0.5471, model 0.6165; model minus market +0.0694 (95% interval -0.0178 to +0.1557). Benter weight on the model -0.028 (s.e. 0.414).
