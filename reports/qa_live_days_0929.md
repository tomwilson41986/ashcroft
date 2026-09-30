# QA after 29 September: is the live model robust?

Written 29 Sep 2026, 17:40 UTC, after the owner's "really bad day, we were getting the market wrong".

- **Forecasts:** the 06:00 files of the five live days, 25–29 Sep, as written (`s3://…/predictions/<date>.csv`). The model changed during the window: the 535 on 25 Sep, the 615 on 26–28 Sep, the gated three from 29 Sep.
- **Results:** horseracebase's day results pages, saved by the card probe (predict-now run 43).
  - 25–28 Sep: every race.
  - 29 Sep: the 29 races up to the 6.00 at Wolverhampton. Cork's last four races were not yet on the page, and the evening races had not been run.
- **Field:** the record's runners that were still on the latest card we priced that day. HRB's non-runner list is taken out for 29 Sep. Reserves the audit found are taken out where a race's field was bigger than the number that ran.
- **Prices:** probabilities are renormalised over the field. "Card" is the horseracebase card price at 06:00, which HRB takes from William Hill. SP is the returned starting price.

## Today (29 races)

- **Rank 1 won 7.** The model expected 8.9, so a day this bad or worse has a 28% chance on its own numbers.
  - The 06:00 card favourite also won 7 (8.7 expected).
- **Rank 1 at SP: −14.6 points on 29 bets (−50%).**
  - In the backtest, 7% of days are that bad at BSP.
  - 61% of backtest days lose for rank 1, and rank 1 loses 6.0% per bet at BSP over the whole window.
- **The model priced the winners as well as the morning market did.**
  - Mean log-probability of the winner: model −1.813, 06:00 card −1.814.
  - The 11:35 card scored −1.714. Later prices know more.
- **The morning market moved toward the model more than on any recorded day.**
  - The rank correlation between the model's gap to the card and the card's move to 11:35 was 0.51.
  - 15 of the 19 horses the model priced at least 22% shorter than the card had shortened by 11:35.
- **Those 19 horses returned +15.9 points backed at the card price.** Three won:
  - Saint Agatha: card 17.0, SP 6/1.
  - Last Dandelion: card 15.0, SP 10/1.
  - Red Shea: card 2.88, SP 8/11.

## The five live days (194 races, 1,988 runners)

| Day | Races | Rank 1 won | Expected | Card favourite won | Rank 1 at SP | Model − card, winner log-lik | Rule picks | Won | At card price |
|---|---|---|---|---|---|---|---|---|---|
| 25 Sep | 44 | 10 | 12.6 | 12 | −19.3 | −0.007 | 52 | 4 | −19.0 |
| 26 Sep | 53 | 17 | 15.8 | 12 | −1.6 | +0.049 | 46 | 8 | +53.8 |
| 27 Sep | 23 | 6 | 6.9 | 9 | −10.6 | −0.058 | 25 | 1 | −16.5 |
| 28 Sep | 45 | 10 | 14.0 | 14 | −14.6 | −0.016 | 27 | 3 | −7.5 |
| 29 Sep | 29 | 7 | 8.9 | 7 | −14.6 | +0.001 | 19 | 3 | +15.9 |
| **All** | **194** | **50** | **58.3** | **54 (54.1 expected)** | **−60.7 (−31%)** | **+0.001 (−0.048 to +0.050)** | **169** | **19** | **+26.6 (+16%)** |

"Rule picks" are a stand-in for the pre-registered rule: runners whose 06:00 card price was at least 22% longer than the forecast. The real rule uses Betfair's morning traded price, which we do not have for these days.

- **Rank 1 fell short of its own expectation.** It won 50 against 58.3 expected, a 10% chance. A quarter of five-day windows in the backtest fall short by as much.
- **The favourites won exactly as often as the card said they would.**
- **The rule's picks returned +26.6 points at the card price, but the interval is huge:** 90% from −33% to +71% per bet. Five days prove nothing either way.
- **The same 169 horses at SP returned −33.3 points (−20%).**
  - The whole difference is the price. 17 of the 19 winners were shorter at SP than at 06:00.
  - For example, Ramaah went from 17.0 to 8.0, Shaffron from 12.0 to 6.0, and Saint Agatha from 17.0 to 7.0.
- **Rank 1 returned −15% at the card price and −31% at SP.**

## What "getting the market wrong" is

### When rank 1 is not the market's favourite, the market is right

This holds in the backtest (5,923 races, 27 Sep 2025 – 31 Mar 2026, the served blend's walk-forward forecasts) and on the live days alike:

| Rank 1 … | Races | Won | Model said | Market said |
|---|---|---|---|---|
| … is the BSP favourite (backtest) | 3,876 | 38.8% | 36.4% | 39.5% |
| … is not the BSP favourite (backtest) | 2,047 | **19.1%** | **27.0%** | 20.0% |
| … is not the 06:00 card favourite (live) | 86 | **18.6%** | **26.3%** | — |

- The live shortfall is the backtest's shortfall. The live path is doing what the backtest did.
- **Against BSP, the model's disagreements carry no information about the result.**
  - Runners the model rates 0.5–1.0 above the BSP in log terms win 3.9%. The model says 7.7%; the BSP says 3.9%.
  - Runners rated more than 1.0 below the BSP win 13.7%. The model says 3.7%; the BSP says 13.4%.
- **A blend with the BSP gives the model a weight of 0.02.** Fitted on the first half of the backtest and scored on the second, it is no better than the BSP alone (1.6863 per race each).

### Against the morning price, the model does carry information

This is from the live days, 06:00 card:

| log(model p / card p) | Runners | Won | Model said | Card said |
|---|---|---|---|---|
| below −0.5 | 429 | 3.0% | 2.7% | 6.0% |
| −0.5 to −0.2 | 378 | 9.5% | 7.5% | 10.3% |
| −0.2 to 0.2 | 696 | 12.9% | 12.2% | 12.0% |
| 0.2 to 0.5 | 313 | 11.2% | 14.6% | 10.6% |
| above 0.5 | 172 | 11.6% | 14.0% | 7.2% |

- **A blend beats both on held-out days.** Leave-one-day-out:
  - Blend (model to the power 0.50, card to the power 0.66): 1.886 per race.
  - Card alone: 1.907.
  - Model alone: 1.906.
- **The blend was better on four of the five held-out days.**

### In short

- The model forecasts the price from form. In the morning it knows things the morning price has not yet taken in, and the market then moves its way.
- By the off, the final market has taken in what the model knows and more. Anything backed at SP on the model's word pays for that.
- Today was an ordinary bad day for top picks and favourites alike, on a day the morning market agreed with the model more than usual.

## Robustness items found

1. **History has been stale since 23 Sep.**
   - `race_results` ends on 22 Sep: horseracebase's download pause, research query `qa_stale_history_0929` (run 36603403959).
   - Share of the day's runners whose latest run was missing from the history:
     - 25 Sep: 0.8%
     - 26 Sep: 0.7%
     - 27 Sep: 0.4%
     - 28 Sep: 2.1%
     - 29 Sep: 2.1% (8 of 375)
   - Every trainer and jockey form window is also a week out of date.
   - The pause ends 30 Sep 13:53 UK. The nightly ingestion then has to backfill 23–29 Sep.
2. **There is no Betfair data after 21 Sep, and no app key.** The pre-registered test (morning traded price → BSP) cannot be scored for any live day until the price files for those days are imported, or the 06:00 job has `BETFAIR_APP_KEY`.
3. **Non-runners were priced as runners before 29 Sep.** Fixed in PR #78: 23 races on 25–29 Sep had a reserve or a blank-jockey horse in their book.

## What follows

- **Do not bet the top pick at SP or BSP on the model's word.**
  - Rank 1 loses in the backtest (−6.0% per bet at BSP) and on the live days (−31% at SP).
  - Where it disagrees with the favourite, it wins at the market's rate, not the model's.
- **The model's price is worth acting on only against a price that has not moved yet.**
  - That means the morning price, and the earlier the better.
  - That is the pre-registered early-price trade, and it is the test to finish.
- **A blended probability belongs in the daily workbook.**
  - It would sit beside the model's, and top picks that disagree with the favourite would be marked.
  - It would be fitted on the morning prices we now keep, not on these five days.
- **Score each live day in the evening, as this note does,** so a bad run is measured the day it happens.
