# Greyhounds: can a model trade off the Betfair data? (the owner's question, 5 Oct 2026)

**Short answer.** The market is efficient a minute before the off, against public form. The early market is where
any edge is: a form-only model points to dogs that later shorten. On the untouched months that gave +3% CLV. The
return was not clearly above zero, and whether those early prices can be taken in size is not yet known.

Study: `research/greyhound_feasibility.py` (full output: `reports/greyhound_feasibility_1005.json`).

- **Data.** The owner's Betfair Historic Data bundle (GB greyhound WIN markets, Jan-Sep 2026: the BSP, the last
  price traded a minute before the off, the first price traded, and the result). Joined to GBGB results from Jul 2025:
  33,474 of 34,496 markets, every runner matched.
- **Features.** Each dog's runs before the race only: calculated and sectional times against the track and
  distance's standard, positions, win rate, run style from the comments against today's trap, grade change, weight
  change, rest, and the track's trap bias. Also each runner against its field.
- **Periods.** Fit on Jan-Apr. Rule chosen on May-Jun. Jul-Sep scored once.
- **Settlement.** 2% commission on winnings.

## 1. The market on its own

Every runner backed at the BSP returns within a few points of zero in every price band, and no band holds across
both periods. The first price traded is a median 5.7% worse than the BSP. About 127 GB win markets a day.

## 2. A minute before the off: no edge

| Jul-Sep (untouched) | Log-loss |
|---|---|
| The market's price at T-1 | 0.43366 |
| That price corrected by the form model | 0.43374 |
| The BSP | 0.43203 |

Form adds nothing to the T-1 price. The last minute itself carries information the T-1 price lacks: the BSP is
better again. The rule chosen on May-Jun (back at the SP where the model sees any edge, T-1 price 20 or shorter) gave
+0.4% on Jul-Sep (90% -2.6 to +3.3) on 207 bets a day: the market's own fairness, not an edge.

## 3. The early market: form predicts which dogs shorten

A model on form alone (no price) against the first price traded. Back at the first price where the model's expected
return there clears the edge (first price 20 or shorter).

| Edge bar | May-Jun: bets/day, CLV vs BSP, result | Jul-Sep: bets/day, CLV vs BSP, result (90%) |
|---|---|---|
| > 0 | 303, +3.3%, +2.6% | 288, +0.5%, -0.6% (-3.3 to +2.2) |
| > 0.1 | 249, +4.7%, +4.4% | 235, +1.3%, +0.7% (-2.5 to +3.8) |
| > 0.2 | 202, +6.2%, +5.7% | 190, +2.0%, +1.6% (-2.1 to +5.3) |
| **> 0.3 (chosen on May-Jun)** | 163, +8.1%, +8.1% | **154, +3.0%, +3.1% (-1.2 to +7.3)** |

The form-only model is worse than the first price as a forecast (log-loss 0.4585 against 0.4441 on Jul-Sep). It
still separates the dogs the market later backs: the CLV rises with the bar in both periods. Out of sample it is
about a third of the in-sample figure.

## 4. What is not known yet, and next

1. **Can the early prices be taken?** The basic plan gives no traded volume. Betfair's greyhound price files (in the
   S3 archive from 2018: morning and pre-play WAP and volume, BSP) answer how much trades early and at what price.
   That is the next query. It would also give eight years to test on instead of nine months.
2. **The recorded books.** The GB/IE greyhound recorder (merged 5 Oct) records every book from now. Two weeks of
   books give the depth behind each early price, and a paper forward test of the chosen rule.
3. **A better model.** Trainer form, sectional by trap, the race-level conditional logit used for horses, and the
   other markets (place, forecast).
4. **Gate before money.** As for horses: CLV positive with its 90% interval above zero on the price files' later
   years, then a forward test at fillable prices, then small stakes.
