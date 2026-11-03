# Football expected-goals features: practice, construction and results (2 Nov 2026, development only)

**Bottom line: xG adds to the forecast of the price's move, which is the target that can be traded. It does not add to
the forecast of the result beyond the opening price.** Built as the published practice suggests (non-penalty xG,
shrunk finishing and goalkeeping, score-state and red-card adjustment, xG ratings), the xG features lift the stacked
move model from R2 4.03% to 4.79%. They lift the move caught by the strongest tenth of forecasts from 4.14% to 4.67%,
and its right-way share from 65.4% to 67.9%, on 6,377 matches scored walk-forward (mostly Betfair early-to-close
moves, 2025/26 on). That period was read before for other rules, so this is development evidence; it needs the forward
test on the recorder like everything else.

Figures: `reports/football_runs/xg_features_dev.json` (the research-aligned build) and
`reports/football_runs/xg_features_dev_first_build.json` (the first build). Code: `football/xg_features.py`, the
statistics table in `sources/api_football_history.py` (`history_stats`). Tests: `tests/test_football_xg_features.py`,
including a lag test (rewriting a Saturday's xG must not move Sunday's features or ratings).

## 1. The data

API-Football's team statistics in the history backfill: xG, the goalkeeper's goals prevented, shots (total, on and off
target, blocked, inside and outside the box), possession, passes, corners and cards, one row a side
(`api_football/history_stats`, 84,932 rows of 98,412 sides). xG is present for 29,884 side rows: the eleven top flights
and the Championship from 2023/24 (about half of 2022/23), most other second tiers and the EFL from 2025/26, and none for
the National League or Scotland's lower three. Checks on the data:

- The xG is calibrated: 1.387 a side a match against 1.401 goals; correlation with goals 0.58.
- It includes penalties: sides that took one penalty average 2.14 xG against 1.28 with none, so penalties are removed
  to make non-penalty xG.
- The provider behind API-Football's xG is not named, so we use one provider for every feature and never mix it with
  another (finding 7 below).

## 2. What the published practice says (research, 2 Nov 2026)

Each line has the page it comes from. [unverified] marks what could not be confirmed on a primary page.

1. **xG difference predicts future results better than goals or points.** Over the previous 10 matches it explains
   32.0% of the variance in the next 10 matches' points; points explain 25.3% (top five leagues, 2013/14-2016/17,
   [Brechot and Flepp](http://repec.business.uzh.ch/RePEc/zrh/wpaper/374_IBW_full.pdf)). The season-to-date xG ratio
   passes R2 0.5 for future goal ratio by match 9, goal ratio by 11, shots on target by 12; total shots never
   ([ElHabr](https://tonyelhabr.rbind.io/posts/xg-predictor-future-results/)). xG was the best predictor at every
   point of the season in a replication on 14,608 games ([ASA](https://www.americansocceranalysis.com/home/2022/7/19/the-replication-project-is-xg-the-best-predictor-of-future-results)).
2. **Penalties.** Opta gives every penalty 0.79 xG ([The Analyst](https://theanalyst.com/articles/what-are-expected-goals-xg));
   StatsBomb 0.78 and Wyscout 0.76 by a secondary source ([jobsinfootball](https://jobsinfootball.com/blog/what-is-the-xg-of-a-penalty/)).
3. **Finishing barely persists.** Goals over xG year to year is about 0.1, clear only past 60 shots
   ([Analytics FC](https://analyticsfc.co.uk/blog/2025/02/11/are-some-players-consistently-good-finishers/)); with
   empirical-Bayes shrinkage, "basically zero" ([ASA](https://www.americansocceranalysis.com/home/2023/8/28/the-replication-project-measuring-shooting-overperformance)).
4. **Goalkeeping barely persists.** Year to year 0.18-0.21 for keepers facing 50+ shots on target
   ([ASA](https://www.americansocceranalysis.com/home/2023/8/28/thomas-bayes-meet-djordje-petrovic)); goals prevented =
   xG on target conceded minus goals conceded, own goals excluded ([Premier League](https://www.premierleague.com/news/4025806)).
5. **Score effects.** Trailing sides take more of the shots ([StatsBomb](https://blogarchive.statsbomb.com/articles/soccer/score-effects/)).
   Multipliers that project shots onto a level score: trailing x0.70-0.90, leading x1.10-1.75, a man down x1.7-1.9; the
   adjustment helps forecasts only a little ([Skripnikov et al.](https://arxiv.org/html/2508.04008v1)). Play at a
   margin of two or more goals carries almost no signal ([ElHabr](https://tonyelhabr.rbind.io/posts/xg-predictor-future-results/)).
6. **Red cards.** A side a man down creates about 65% less xG a minute, the opponent about 54% more
   ([Data Between the Lines](https://databetweenthelines.substack.com/p/how-do-red-cards-impact-team-performance)).
7. **Providers differ.** Shot locations differ by "a couple of metres"; two providers disagreed on the more threatening
   side in 13% of top-five games in 2018/19 ([KU Leuven](https://dtai.cs.kuleuven.be/sports/blog/how-data-quality-affects-xg)).
8. **The market.** Betting-exchange prices overrate sides that outperformed their xG and underrate those that
   underperformed ([Flepp, Merz and Franck, Economic Inquiry 2024](https://ideas.repec.org/a/bla/ecinqu/v62y2024i1p414-429.html);
   effect sizes paywalled). An xG model on 11 Bundesliga seasons returned about 10% at average odds, not tested
   against the close ([Wilkens 2026](https://sponet.de/sponet/Record/4097217?lng=en)). No published xG model tested
   against Pinnacle or Betfair closing prices was found.
9. **Syndicates.** Starlizard is reported to "record every stat you can imagine"
   ([Bleacher Report](https://bleacherreport.com/articles/2200795-mugs-and-millionaires-inside-the-murky-world-of-professional-football-gambling));
   Benham calls the league table "notoriously inaccurate" ([SI](https://www.si.com/soccer/2015/05/15/soccer-analytics-brentford-midtjylland-matthew-benham)).
   Reports of watchers grading chance quality are [unverified].

## 3. What was built (`football/xg_features.py`)

| Family | Features (home minus away; windows: last 10, EWM half-life 8, season to date for the core) | From the practice |
|---|---|---|
| Core | Non-penalty xG for, against and difference (penalties at 0.79) | 1, 2 |
| Core, adjusted | The same projected onto a level score with eleven men: score-state multipliers by margin (trailing by one x0.85, by two or more x0.75; leading by one x1.2, by two or more x1.5) and the red-card rate change, time-weighted from the events | 5, 6 |
| Shot quality | Non-penalty xG per shot for and against, share of shots inside the box | inferred |
| Shots | Shots and shots on target, for minus against | 1 |
| xG vs market | Non-penalty xG difference minus the close's expected goal difference | 8 |
| Luck, raw | Finishing (non-penalty goals minus npxG) and keeper goals prevented, last 10 and EWM | 3, 4, 8 |
| Luck, shrunk | Finishing over 20 matches shrunk by 2,000 pseudo-shots (keeps about a tenth); keeper goals prevented over 38 matches per shot on target faced, shrunk by 400 | 3, 4 |
| xG rating | Opponent-adjusted attack and defence from a decayed ridge fit of log npxG (half-life about 300 days), refitted at each collection day; its gap to the market-implied rating | 1 |
| Game state | Share of minutes trailing | 5 |

Every feature uses only matches before the match's collection day (Friday for Friday to Monday matches, Tuesday for
Tuesday to Thursday).

## 4. Results (development; walk-forward by season, each fitted on the seasons from 2022/23 before it)

Rows: 14,423 matches with xG windows (both sides with three or more earlier xG matches). The evaluation uses ridge
regression with a fixed penalty and standardised features clipped at 4: an unregularised first build blew up on the
sparse goalkeeper columns (`xg_features_dev_first_build.json`, kept for the record). These settings were fixed before
the results were read.

**The move (early price to close, same book), 6,377 matches, 2024/25-2026/27:**

| Model | R2 vs mean | Top tenth caught | Right way |
|---|---|---|---|
| Ridge, team features | 3.44% | 3.63% | 64.4% |
| Ridge, team + xG features | 4.39% | 4.19% | 67.1% |
| Stack (ridge + LightGBM), team features | 4.03% | 4.14% | 65.4% |
| **Stack, team + xG features** | **4.79%** | **4.67%** | **67.9%** |

By family alone (ridge): xG core 1.84%, adjusted core 1.80%, shots 1.72%, xG rating 0.85%, xG quality 0.51%, game state
0.29%, xG vs market 0.16%, luck 0.07-0.13%; all xG together 1.95% (permuted floors within -1.03% to +0.45%).

**The result, over the opening price and our model (log-loss gain per thousand), 10,291 matches:** no family adds.
Core -0.76 (-1.12 to -0.41), adjusted core -0.65, quality -0.64, raw luck +0.24 (-0.27 to +0.73), shrunk luck -0.02,
xG rating -0.08. The established team features score -1.90 on this same subset, against +0.08 on the full 68,533-match
sample, so this test is too small here to read beyond "no gain".

## 5. What it means

- **xG is a real addition to the move model.** It adds about 0.8 points of R2 to the stack and half a point to the
  move caught by the strongest signals. That is the right direction against a spread of about a tick at each end, but
  the cost question (the unrecorded early lay, `reports/football_review_and_roadmap.md` s.2.6) still decides
  profitability.
- **The adjustments matter less than the base measure.** Score-state and red-card adjustment did not beat plain
  non-penalty xG (1.80% vs 1.84%), as Skripnikov et al. found ("marginal").
- **The market's xG residual is small in our sample.** xG difference against the market's expectation, and the luck
  terms, add almost nothing to the move. The Flepp result (exchanges overrate xG over-performers) is about outcomes,
  which this sample is too small to test.
- **Coverage grows.** Most second tiers have xG only from 2025/26, so the xG features cover more leagues each season.

## 6. Next

1. Put the team + xG stack into the paper-trial pipeline as a pre-registered rule, scored on the recorder's books.
2. Refit the xG features as data accrues; re-test the outcome value when the sample with xG exceeds about 25,000
   matches.
3. Tune the decay (90 days to one year) and the shrinkage walk-forward once the sample allows, and fit a Dixon-Coles
   model on npxG beside the goals model with a learned blend weight (research finding 1, top-10 feature 5).
