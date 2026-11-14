# Greyhound edge search: the ledger

The owner, 4 Nov 2026: "keep working in loops until you define a true edge and profitable strategy"; "fundamental is
always just to get the most accurate closing price, optimise towards this"; "only come back when you have 5 profitable
strategies".

**What counts.** A rule fixed before its test (chosen on a development period, read on a later holdout it never saw),
filled as it really would be (at the Betfair SP or a limit-on-close; resting orders only through the queue-aware
replay, `research/greyhound/queue_fill.py`), with its holdout return's 90% lower bound (a bootstrap over days) above
zero and its development return above zero. **Two targets, always** (the owner, 6 Nov 2026: "let's always test
against two targets; 1. BFSP 2. Win - winning the bet"): every forecast is scored on the SP (RMSE of log(BFSP)) and on
the result (the winner's log loss, its chances normalised within the race); every rule is scored closed at the SP and
left to run to the result, a pass on either counted as a test of its own. **Commission: 2%** on each market's net winnings, the account's rate (the owner, 6 Nov 2026: "it's 2%"), the standard for every greyhound test and trader (greyhound/costs.py). Every rule tried is counted here: about one in twenty clears a one-sided
90% bound by chance.

| # | Family | Data, test | Rules tried | Passing | Verdict | Source |
|---|---|---|---|---|---|---|
| 1 | Place SP against Harville on the win price known before the off (the morning WAP) | price files, GB, 2021-26 walk-forward | 16 (2% comm.) | 1 at +3.60% (0.18 to 7.28), 1 in 16 = chance | no edge | research-query 41883691562 |
| 1b | Same on the win SP (look-ahead upper bound) | price files, 2023-26 | 16 | most | not a rule: chose on the place SP itself (look-ahead on both sides) | research-query 41883691562, 41884997778 |
| 2 | Place SP against Harville on the win market's projected SP / mid, 120-10 s before the off, settled at the place SP | recorded books 6 Oct - 3 Nov (3,530 races; the recorder's, ~40 s stale near the off) | 16 | 0 | no edge on stale books; to retest on the live-key books (5 s, projected SP) as days accrue | research-query 41884997778 |
| 3 | Limit-on-close against the morning WAP (back drifts, lay steamers), win and place | price files, dev 2016-21, holdout 2022-26 | 72 | 0 | no edge | research-query 41888602917 |
| 4 | Limit-on-close against our fair price (win and place, raw and tempered), log-BSP and softmax models | fast-loop 41888253783 (fitted on 2021-23, 784,294 runners priced 2024 - 3 Nov 2026); dev 2024, holdout 2025-26 | 160 | 0 | no edge: every win rule's holdout return negative; the SP is efficient against our pre-race price | research-query 41891640383 |
| 5 | The SP's bias by segment (track, trap, field, session) as limit-on-close thresholds | price files, dev 2016-21, holdout 2022-26 as one portfolio | 17,822 (win 8,989, place 8,833); 1,500 selected on dev | 0 | no edge: the dev-selected portfolios lose in every holdout year (win backs -1.38%, 90% -2.18 to -0.60, 361,513 bets; win lays -1.25%; place backs -1.30%; place lays -1.25%): selection on noise. The older files' track names are not normalised (e.g. 'Televised Cards / Sund') | research-query 41889297892 |
| 6 | Place against win from Betfair's historic data at 1 and 15 minutes out (last traded prices), bet at the place SP; GB and AU | BASIC Jan-Sep 2026 (70,125 races paired: GB 34,174, AU 33,068); dev Jan-Apr, holdout May-Sep | 32 | 0 | no edge in GB (backs lose in both halves; lays +0.8 to +3.9% in dev, -1.9 to -4.1% in holdout). AU lays at 1 min: holdout +1.7 to +3.7% (best +3.70%, 90% 0.15 to 7.58, 11,148 bets) but dev -1.04 to +0.93%: a lead for a forward test, not a pass. 15 min out: too few place trades | research-query 41892329688 |
| 7 | Australian greyhounds: morning-drift limit-on-close and the SP's segment bias | price files AUS, dev 2016-21, holdout 2022-26 | 72 + 12,983 | 2 of 36 drift rules at 2% (lays at the SP only when the dog has shortened to 0.8-0.9 of its morning WAP: best +12.7% a unit staked, 90% 5.58 to 19.7, 7,523 lays, mean SP 37.8; +0.34% of liability); segment portfolios fail (win backs -2.23%, lays -1.15%) | candidates, not counted: 2 of 36 is the chance rate, the two overlap, dev returns not significant, and whether the account can bet AU markets is unsettled; a forward test if pursued | research-query 41893115101 |
| 8 | Trade to the SP on the closing forecast at 1 minute out (Betfair's historic data, last traded prices only) | BASIC Jan-Sep 2026, GB and AU; forecast and rules chosen Jan-Apr, read May-Sep | 16 uncapped; 96 capped (entry <= 5/10/20, fills 1 and 2 ticks through the last trade) | uncapped: passes from entries averaging 89-390, not credible (stale last prices of long shots); capped: lays where the forecast close is 5-10% above the price pass in GB and AU at both fills (e.g. GB lay, margin 0.10, entry <= 10, 1 tick: +10.72%, 90% 8.90 to 12.58, 541 trades) | not counted until checked against real books: the forecast may be reading stale last prices no one would fill | research-query 41895217384, 41896084511 |
| 8b | The trade-to-SP lays against the recorded books' real prices (the forecast fitted on Jan-Apr; GB books 60 s before the off, entered at the best price with GBP2 on offer, closed at the SP) | recorded books 7 Oct - 3 Nov, 3,477 races (mostly the recorder's) | 18 + 18 | 0 | **the edge was the fill assumption**: at the book's price every lay loses (margin 0.10: entry <= 5 -1.02%, 90% -2.73 to 0.64; <= 10 -3.28%; <= 20 -4.28%) while on the same rows a tick through the last trade 'makes' +9.5 to +16.5%; backs lose 13-24%. The forecast reads stale last prices no one offers | research-query 41897170252 |
| 9 | The stream lab: 43 taker rules at the price on offer 60/30/10 s out, closed at the SP (projected-SP gap, momentum, reversal, a control) | the recorded Exchange Stream from 5 Nov, scored nightly, pooled | 43 | pending: 10 days | registered 5 Nov (reports/greyhound_stream_lab.md); no result before then counts | greyhound-stream-lab.yml |
| 10 | The metrics database (greyhound/pit): every family added to the live recipe | fast loop, test 2026-06-01 to 2026-11-05 (read by earlier batches), both targets | 9 + 9 | 0 on both | all families together: RMSE log(BSP) 0.4130 -> 0.4101 (-0.0029, 90% -0.0038 to -0.0020); result log-loss +0.00024 (-0.00022 to +0.0007): the closing price improves, the result does not; forward test registered (pit_forward.json, from 7 Dec 2026) | fast-loop 42255735836, 42253527161; reports/greyhound_metrics_database.md |
| 11 | The stream lab's late family: its projected-SP, momentum and reversal rules decided 0, 10 and 20 s after the scheduled off | the recorded Exchange Stream, counted from 11 Nov | 42 | pending: 10 days | registered 10 Nov (reports/greyhound_model_review.md section 5) | greyhound-stream-lab.yml |
| 12 | The band test: back or lay at the price on offer when it is outside the forecast's 80% or 90% band of the SP (gbm_small_plus), 60 and 10 s before the scheduled off and 0, 10, 20 s after | the recorded Exchange Stream, counted from 11 Nov | 20 | pending: 10 days | registered 10 Nov. Practice 7-9 Nov, not counted: lays 60 s out +7.1% (26 trades) and +3.9% (50); the rest inside their noise | greyhound-close-lab.yml; practice run 42786681970 |
| 13 | The resting test: a lay at the best back price or a back at the best lay price, resting to the real off, queue-aware fills, closed at the SP; every runner, or where the forecast agrees by 0 or 3% | the recorded Exchange Stream, counted from 11 Nov | 12 | pending: 10 days | registered 10 Nov. Practice 7-9 Nov, not counted: backs resting 60 s out +0.87% with no forecast, +1.94% and +3.64% where the forecast agrees; nothing at the scheduled off; replayed fills | greyhound-close-lab.yml; practice run 42786681970 |
| 14 | Win and place at the SP on the new model (stack3_all3): limit-on-close backs and lays on its value, on a BSP + model blend, the place SP through Harville | price files, out-of-sample prices fitted before 1 Jun; dev Jun-Aug, holdout Sep-5 Nov 2026 | 46 | 0 | no edge: the blend weights the model 0.035 (log-loss 0.43791 against the BSP's 0.42853) | fast-loop 42888391928 |
| 15 | The morning move on the new model: lay where the forecast SP is above the morning WAP, back where below; greened at the SP or held; 0-2 ticks worse; morning volume floors | same | 96 (+ 96 pre-play, no data) | 48 lays (backs 0) | **the market's own drift**: laying every runner with a morning price makes +7.7% (6.2 to 9.5) with no model; the model's lift over that control +1.3% to +5.8% on the holdout (8 rules, lower bound above zero), the old model about the same. A fill question (the WAP traded; a lay there is a maker's order): a candidate for a live-key forward test, not counted | fast-loop 42892318999 |
| 16 | The limit-order grid and the trade-out study on the new model's prices | recorded books 6 Oct - 5 Nov | 360 + 26 | 11 + 2 | chance-level (about 18 expected); the early lay's rule +1.6% to +2.0% on the holdout (CLV 10.8% against 7.3% on the old price) | fast-loop 42888391928 |
| 17 | Back strategies (the owner: "back and trade out at BFSP, or back just to win"): every back order (take, join, ticks, rest at the forecast + margin) at three marks, ten gates, two fills, traded out at the SP or held to win, read on both periods | recorded books 6 Oct - 5 Nov (dev 15, holdout 7 days); three prices | 3,054 a price | 27 (new model, before 6 Oct), 23 (before 1 Jun), 5 (old recipe); about 153 by chance | **no back-to-win rule** (509 hold rules: 4 pass, none survives a 10% FDR). One trade-out family survives the FDR on the new model's prices: the favourite backed at the first traded book at the forecast SP + 20%, closed at the SP, holdout +10.50% (5.38 to 14.37), 157 trades; on the old price +7.51%, not surviving. Backing with no model loses 4% to 27%. Registered for a paper forward test from 11 Nov (reports/preregistration_greyhound_fav_back.md) | fast-loop 42906178212 |
| 18 | The early lay live on the owner's account (GBP20 then GBP100 a lay; the tested rule, the guard off from 13:46 UTC on 11 Nov) | the trader's ledger against the settled books, 3, 4 and 11 Nov 2026 | 1 | 0 | **the fills, again**: 717 lays, the matched ones' log(SP / lay price) 0.000, +0.001 and +0.021 a day, the unmatched +0.358, +0.305, +0.321; locked GBP-19.35, GBP-23.56, GBP-42.70 (11 Nov: -0.56% of GBP7,691 matched). Back on paper 11 Nov 18:16 UTC (the owner) | greyhound-diagnostic 43074468127; greyhound-edge-anatomy 43077986412; reports/greyhound_model.md 42, 43.4 |
| 19 | The edge anatomy: a lay or a back taken at the price the live-key book offers, 600 to 15 s before the scheduled off, closed at the SP; no signal, our price, the projected SP (near, far), both; five margins, four price bands | live-key books 3-11 Nov 2026 (1,017 races); development 3-8 Nov, holdout 9-11 Nov | 1,470 (754 with 100 bets) | 0 | no edge: 21 development candidates (37.7 expected by chance), none with a holdout interval above 0. Blind taking costs 1.8% to 2.9% a pound in the last minute; our price as the signal is worth about a point. On books under 15% wide the mid's error is 0.106 to 0.161 against ours 0.26 to 0.32, and the blend weights ours 1% to 12% | greyhound-edge-anatomy 43076824352, 43077986412; reports/greyhound_model.md 43 |

## The closing price (the objective)

RMSE of log(BSP), recorder books re-timed, 14 Oct - 2 Nov (closing-price run 41884995327): first traded 0.319 (mid
0.635), 5 min 0.368 (0.709), 3 min 0.225 (0.337), 1 min 0.144 (0.158). The recorder's greyhound books carry no
projected SP; the live key's do (from 2 Nov), to be scored as their days accrue.

Betfair's historic data, Jan-Sep 2026, GB, 1 minute out (research-query 41892329688): the win SP from the win market's
last traded price RMSE 0.1759; a LightGBM reading both markets' prices at 60, 15 and 1 minute out and the race's book
0.1718 (90% of the gain in mean squared error -0.00169 to -0.00077, by day; 102,144 runners May-Sep, fitted on
Jan-Apr). The place market and the book's total carry information on the win SP the win market's last price lacks.

**The closing-price lab** (research/greyhound/close_lab.py, greyhound-close-lab.yml nightly; the owner, 6 Nov: "Let's
start recursive self improvement"): every 10 s over the last 5 minutes, from the recorded Exchange Stream, six specs
registered 6 Nov, scored walk-forward by day, and counted only on days after registration. Day one, 5 Nov (73 races,
12,240 runner-checkpoints; run 42192362661), baselines only, RMSE of log(SP):

| Baseline | All | <= 60 s | 70-180 s | > 180 s |
|---|---|---|---|---|
| mid | 0.3000 | 0.1180 | 0.1938 | 0.4248 |
| last traded | 0.3216 | 0.1185 | 0.2218 | 0.4499 |
| Betfair's projected SP | 0.4454 | 0.1401 | 0.2645 | 0.6452 |

Betfair's projected SP is the worst of the three in every window on this day, so the bar a spec must clear is the best
baseline (the mid here), not the projected SP. The first fresh day for the six specs is 7 Nov, scored the night of 8
Nov.

## The nightly record (both targets: 1. BFSP, 2. Win)

**6 Nov** (117 races recorded, 19,800 runner-checkpoints; close lab run 42353917019, stream lab run 42353918865).
Practice only: no result counts yet. The close-lab specs were registered on 6 Nov, so their first fresh day is 7 Nov.
The stream-lab rules need 10 days.

Close lab, fitted on 5 Nov and scored on 6 Nov:

| | BFSP: RMSE log(SP), all | 60 s or less | over 180 s | Win: log loss |
|---|---|---|---|---|
| mid (baseline) | 0.2884 | 0.1340 | 0.4019 | 1.4904 |
| gbm_all_small | 0.2222 | 0.1402 | 0.2847 | 1.4751 |
| gbm_all | 0.2263 | 0.1484 | 0.2878 | 1.4719 |
| gbm_book | 0.2568 | 0.1387 | 0.3480 | 1.4842 |

- The specs gain far from the off. In the last minute the mid is already better than any spec.
- The win-trained specs, fitted on one day, did worse than the mid on both targets.
- ridge_all scored nothing: an infinite feature broke its standardisation. That is fixed; the spec itself is unchanged.

Stream lab, 2 days pooled: no candidate on either target among 86 tests.

The early-lay forward test's lays, replayed queue-aware through the stream, as a share of matched liability:

| Day | Closed at the SP | Held to the result (x1 / x0.5 volume) |
|---|---|---|
| 5 Nov | +1.78% | -1.65% / -0.37% |
| 6 Nov | +1.43% | +1.04% / +0.88% |

The forward test's own figure on the recorder's books for 6 Nov was +1.25%.

**7 Nov.** 121 races and 20,190 runner-checkpoints in the close lab (run 42428569854). The stream lab read 514,487 stream lines and 250 markets (run 42428572199). This was the close-lab specs' first fresh day.

- **The BFSP champion changed (the pre-registered bar cleared).** gbm_all_small replaces the mid.
  - Paired gain in squared error of log(SP): 0.0415 (90% interval 0.032 to 0.051), over 121 races.
  - RMSE: 0.2247 against the mid's 0.3030 overall, and 0.1365 against 0.1404 at 60 s or less.
  - This is one fresh day, and the champion is re-tested every night.
- **Win: the mid stays champion.** The best challenger, gbm_book, gained -0.0015 (90% interval -0.0072 to +0.0042).
- **Close lab, 7 Nov (fitted on 5-6 Nov):**

  | | BFSP: RMSE log(SP), all | 60 s or less | over 180 s | Win: log loss |
  |---|---|---|---|---|
  | gbm_all_small | 0.2247 | 0.1365 | 0.2945 | 1.5120 |
  | gbm_all | 0.2295 | 0.1407 | 0.2993 | 1.5122 |
  | mid | 0.3030 | 0.1404 | 0.4280 | 1.5095 |
  | ltp | 0.3200 | 0.1405 | 0.4554 | 1.5180 |

- **Stream lab, 3 days pooled:** no candidate among 86 tests.
- **The forward test's lays, 7 Nov** (queue-aware, as a share of matched liability):
  - closed at the SP: +0.71% (x1) and +0.64% (x0.5);
  - held to the result: -1.50% (x1) and -2.11% (x0.5).
  - Its own figure was +0.96%.

**Registered 8 Nov: the trade test on the champion's forecast** (research/greyhound/close_lab.py, `trade_rules`).
Does the price on offer lag gbm_all_small's forecast of the SP? The rules are fixed:
- At 60, 30 and 10 s out, back at the best back price when it is more than m above the forecast SP, and lay at the best lay price when it is more than m below. GBP2 or more must be shown.
- m is 3%, 6% or 10%.
- Each trade is scored closed at the SP and also held to the result, with 2% commission on a gain.

That makes 18 rules and 36 tests. Only days after 8 Nov count. A rule is a candidate on a target after 10 days with its 90% lower bound (bootstrap over races) above zero.

**8 Nov.** 109 races and 17,820 runner-checkpoints (close lab run 42538888173). The stream lab read 444,811 lines across 218 markets (run 42538890940). Collection is healthy.

- **BFSP:** gbm_all_small holds the championship. RMSE log(SP) 0.1895 against the mid's 0.2381 overall, and 0.1253 against 0.1230 at 60 s or less.
- **Win:** the mid holds. The best fresh challenger, gbm_all_small, gained 0.0037 (90% interval -0.0012 to +0.0087, 230 races), which does not clear the bar.
- **The trade test on gbm_all_small's forecast:** no counted day yet. Its first is 9 Nov.
- **Stream lab:** 4 days pooled, no candidate among 86 tests. Momentum lays still lose 2.9% to 5.6% at the SP.
- **The forward test's lays, 8 Nov** (queue-aware, share of matched liability): +1.42% at the SP and +4.28% to the result (x1); +1.41% and +4.19% (x0.5). Its own figure was +0.47%.

**9 Nov.** 136 races and 22,980 runner-checkpoints (close lab run 42726462862). Stream lab run 42726467717. Collection is healthy.

- **Win: the champion changed (the pre-registered bar cleared, only just).** gbm_all_small replaces the mid.
  - Paired gain in the winner's log loss: 0.0046 (90% interval 0.00004 to 0.0089), over 365 races.
  - The lower bound is barely above zero, and the champion is re-tested every night.
  - gbm_all_small now holds both targets.
- **BFSP:** gbm_all_small holds. RMSE log(SP) 0.1837 against the mid's 0.2760 overall, 0.1218 against 0.1235 at 60 s or less, and 0.2321 against 0.3885 over 180 s.
- **Close lab, 9 Nov (fitted on 5-8 Nov):**

  | | BFSP: RMSE log(SP), all | 60 s or less | over 180 s | Win: log loss |
  |---|---|---|---|---|
  | gbm_all_small | 0.1837 | 0.1218 | 0.2321 | 1.5625 |
  | gbm_all | 0.1857 | 0.1216 | 0.2350 | 1.5617 |
  | mid | 0.2760 | 0.1235 | 0.3885 | 1.5685 |
  | ltp | 0.2838 | 0.1255 | 0.3994 | 1.5685 |

- **The trade test on gbm_all_small's forecast: its first counted day.** 14 of the 18 rules traded, 1 to 18 trades each, so nothing is readable yet. The busiest:
  - lay_60_0.03: 18 trades, +0.78% a unit at the SP (90% -6.5% to +7.7%);
  - lay_30_0.03: 18 trades, +0.60% (-7.3% to +8.3%);
  - back_60_0.03: 10 trades, -6.64% (-12.2% to -1.1%).
  - The rule needs 10 days.
- **Stream lab:** 5 days pooled, no candidate among 86 tests. Momentum lays still lose 3% to 5% at the SP.
- **The forward test's lays, 9 Nov** (queue-aware, share of matched liability, GBP5,752 matched over 293 lays): +1.85% at the SP and +2.62% to the result (x1); +1.80% and +2.48% (x0.5). Its own figure was +1.72%.

Five days so far:

| | 5 Nov | 6 Nov | 7 Nov | 8 Nov | 9 Nov |
|---|---|---|---|---|---|
| At the SP | +1.78% | +1.43% | +0.71% | +1.42% | +1.85% |
| To the result | -1.65% | +1.04% | -1.50% | +4.28% | +2.62% |

**10 Nov: the review of the statistics, features and modelling** (reports/greyhound_model_review.md; dry runs
42768177695 and 42769218285, nothing written to the registry). What changes from tonight:

- **The champion bar is stricter (version 2).** A crown needs a paired gain 3 standard errors above zero over at
  least 3 fresh days and 300 races, and a spec keeps a crown only while it clears that against the best baseline.
  - BFSP: gbm_all_small holds (gain 0.0355 over the mid, 10.3 standard errors, 366 races).
  - Win: the crown of 9 Nov returns to the mid (gain 0.0046, 1.67 standard errors, 365 races). It was chance-sized.
- **The lab reads on after the scheduled off.** Races go off a median 22 s late (17 to 23 s by day), so a
  checkpoint is taken every 10 s until the market closes. The headline scores stay on the first five minutes.
- **Ten specs registered 10 Nov**, fresh from 11 Nov: three new feature families (the SP pool, the trade path, the
  race around the runner), by-window fits, Huber loss, a normalised book, and a conditional logit for the result.
- **q-values** beside every test in the stream lab and the trade test; a z-score needs 100 races.
- **Read every night now:** the SP as the Win ceiling (it beats the mid by 0.0085 a race at 60 s out, p 0.002, 548
  races) and what our forecast adds to the mid on the result (0.00045, p 0.57).
- **Three more tests registered 10 Nov** (ledger rows 11 to 13; the owner: "work on these and fix. then merge"),
  counted from 11 Nov: the stream lab's late family, the band test and the resting test.
- **The early lay left to rest to the real off**, read beside the gate's figure from now on. On 9 Nov it matched 306
  lays against 296 and made +1.78% at the SP against +1.85% (stream-lab run 42787435666): the clock barely matters
  for this rule.

**10 Nov.** 114 races and 19,050 runner-checkpoints in the close lab (run 42971599560). The stream lab read 442,524
lines, 228 markets and 8,845 trades (run 42971603161). Collection is healthy. The ten specs, the late family, the band
test and the resting test count from 11 Nov, so none has a counted day yet.

- **Champions (bar version 2):**
  - **BFSP:** gbm_all_small holds. Against the mid: gain 0.03486, z 11.11, 480 races over 4 days. The strongest
    fresh challenger, gbm_all, is worse (gain -0.00089, z -3.44).
  - **Win:** the mid holds. gbm_all_small's gain is 0.00514 (90% 0.00116 to 0.00894), z 2.15, 477 races; a gain this
    size needs 963 races to clear the bar.
- **Close lab, 10 Nov:**

  | | BFSP: RMSE log(SP), all | 60 s or less | Tight book | Late (after the off) | Win: log loss | Win, late |
  |---|---|---|---|---|---|---|
  | gbm_small_plus | 0.1818 | 0.1108 | 0.1176 | 0.0761 | 1.4863 | 1.4820 |
  | gbm_all_small | 0.1823 | 0.1118 | 0.1182 | 0.0790 | 1.4859 | 1.4798 |
  | mid | 0.2595 | 0.1112 | 0.1179 | 0.0773 | 1.4929 | 1.4809 |

  In the last 60 s, in a tight book and after the off, the mid is as good as the forecasts. Their gain is all earlier.
- **The review's readings, 10 Nov:**
  - **The off:** markets closed a median 18 s after the scheduled off (10% 7 s, 90% 40 s). All were open at the
    scheduled off, 22% still open 30 s after and 4% 60 s after.
  - **The SP pool:** SP backs on 89% of runner-checkpoints (median GBP58) and SP lays on 99% (median GBP15
    liability). The SP book's median is 1.0042 over 670 races.
  - **Benter's second stage:** the SP itself beats the mid on the result by 0.00758 a race at 60 s out (p 0.0016, 660
    races) and 0.0096 at 10 s (p 0.0004). gbm_all_small adds 0.00044 at 60 s (p 0.52) and 0.00008 at 10 s (p 0.78):
    nothing yet.
- **The trade test on gbm_all_small's forecast** (second counted day of 10 needed): the busiest rules are lay_10_0.03
  (34 trades, -0.19% a unit at the SP, 90% -4.8 to +3.3), lay_60_0.03 (27, -2.79%) and back_10_0.03 (19, +5.40%,
  0.89 to 10.02).
- **The band test and the resting test:** no counted day yet.
- **Stream lab, 6 days pooled, 86 tests: no candidate.**
  - rev_lay_10_0.1 to the result: +34.96% (15.7 to 43.7, q 0.02), but its deflated Sharpe is 0.034 against the
    0.95 needed, and it has 6 of 10 days.
  - Momentum still loses 2% to 5% at the SP.
- **The early lay's forward-test lays, 10 Nov** (queue-aware, GBP4,790 matched over 242 lays):
  - at the SP: +2.22% (x1) and +2.19% (x0.5);
  - to the result: +4.51% and +4.33%;
  - resting to the real off: +2.20% at the SP and +4.94% to the result;
  - its own figure: +1.9% (207 lays matched).

Six days so far:

| | 5 Nov | 6 Nov | 7 Nov | 8 Nov | 9 Nov | 10 Nov |
|---|---|---|---|---|---|---|
| At the SP | +1.78% | +1.43% | +0.71% | +1.42% | +1.85% | +2.22% |
| To the result | -1.65% | +1.04% | -1.50% | +4.28% | +2.62% | +4.51% |

### 11 Nov 2026 (close lab 43165734884, stream lab 43165738409; read 12 Nov 08:45 UTC)

- **The day.** 145 races, 24,210 runner-checkpoints (4,260 race-checkpoints). The early lay was live for part of
  this day and back on paper from the evening (the owner's word).
- **Champions.** BFSP: gbm_all_small holds against the mid (gain 0.03931, z 13.32, 625 races over 5 days); the
  strongest fresh challenger gbm_small_plus_huber: gain 0.00031, z 0.88 on its first day (145 races; a gain this
  size needs 1,578). Win: the mid; gbm_all_small's gain 0.00646, z 2.86 over 619 races (needs 690).
- **The day's BFSP RMSE** (all / last 60 s / late): the mid 0.3120 / 0.1402 / 0.1164; gbm_book_flow 0.2393 /
  0.1374 / 0.1215; clogit_win 0.2667 / 0.1665 / 0.1661; clogit_win_plus 0.2809 / 0.2067 / 0.3091.
- **The review's readings.** The market closed a median 21 s after its scheduled off (10% 8 s, 90% 60 s); still
  open 30 s after on 33%, 60 s after on 10%. The SP pool: backs on 69% (median GBP54), lays on 98% (median GBP10
  liability). The SP book over 815 races: median 1.0041. Benter's second stage: the SP itself gains 0.00872 at 60 s
  (p 0.0002); gbm_all_small 0.00051 (p 0.43) at 60 s over 619 races; gbm_small_plus_huber 0.00073 (p 0.65) over 142.
- **The trade test** (third counted day of ten): no candidate; the widest cells are back_10_0.03 +4.48% at the SP
  (26 trades, 90% 0.30 to 8.65) and lay_60_0.03 -2.75% (43 trades).
- **The band test** (first counted day): 2 trades in all; quantile coverage q10 0.124, q20 0.223, q80 0.775,
  q90 0.872. **The resting test** (first counted day): every cell within noise of zero at the SP
  (rest_back_60_all +0.094%, rest_lay_60_all +0.309%, about 640 trades each); no candidate.
- **The stream lab.** No candidate. The late family's first counted day is negative at the SP in every cell read
  (between -2.4% and -3.9%).
- **The early-lay forward test, 11 Nov.** Its own figure: 148 races, 670 lays placed, 239 matched, +2.02% of
  liability. Queue-aware replay: 300 matched, +1.46% at the SP, +0.81% to the result (traded volume x1.0; x0.5:
  +1.44%, +0.62%). The same lays resting to the real off (not the gate's figure): +1.33% at the SP, -0.28% to
  the result.

| | 11 Nov |
|---|---|
| At the SP | +1.46% |
| To the result | +0.81% |

### 12 Nov 2026 (close lab 43326793895, stream lab 43321877130; read 14 Nov 00:15 UTC)

- **The day.** 118 races, 20,100 runner-checkpoints (3,480 race-checkpoints); 456,705 stream lines, 238 markets,
  8,904 trades. The repository was private on 12 Nov to about 12:40 UTC 13 Nov, so hosted jobs queued or died: the
  scheduled close lab was queued at 23:12 UTC and ran 00:37 to 00:44 UTC 13 Nov (success); the stream lab ran 22:50
  UTC (success). The early lay ran live from 11:35 UTC on prices that froze from about 14:26 UTC (hosted price runs
  failed while the repository was private). First day of the favourite-back paper trader: its summary
  (s3 sources/greyhound/fav_back/paper/2026-11-12/summary.json) - no data found here: not read.
- **Champions.** BFSP: gbm_all_small holds against the mid (gain 0.03672, z 14.00, 743 races over 6 days); the
  strongest fresh challenger gbm_small_plus_huber: gain 0.00040 (90% 0.00002 to 0.00079), z 1.73, 263 races over 2
  days (a gain this size needs 806). Win: the mid; gbm_all_small's gain 0.00581 (90% 0.00214 to 0.00947), z 2.58 over
  735 races (needs 925).
- **The day's BFSP RMSE** (all / last 60 s / late): the mid 0.2457 / 0.1339 / 0.0988; gbm_all_small 0.1915 /
  0.1309 / 0.0937; gbm_book_flow 0.2051 / 0.1296 / 0.0937; clogit_win 0.2180 / 0.1466 / 0.1196; clogit_win_plus
  0.2383 / 0.1822 / 0.2630.
- **The review's readings.** The market closed a median 17 s after its scheduled off (10% 6 s, 90% 53 s); still
  open 30 s after on 27%, 60 s after on 8%. The SP pool: backs on 73% (median GBP50), lays on 99% (median GBP10
  liability). The SP book over 933 races: median 1.0039. Benter's second stage: the SP itself gains 0.00817 at 60 s
  (p 0.0001, 918 races); gbm_all_small 0.0001 (p 0.70) at 60 s over 735 races; gbm_small_plus_huber 0.00002
  (p 0.92) over 258.
- **The trade test** (fourth counted day of ten): no candidate; the widest cells are back_10_0.03 +3.34% at the SP
  (30 trades, 90% -0.38 to 7.13), lay_60_0.03 +0.03% (55 trades) and lay_10_0.03 -0.58% (67 trades).
- **The band test** (second counted day): 3 trades in all; quantile coverage q10 0.114, q20 0.211, q80 0.782,
  q90 0.877 (49,641 runner-checkpoints). **The resting test** (second counted day): no candidate; rest_back_60_all
  -0.213% at the SP (1,186 trades), rest_lay_60_all +0.469% (1,190, 90% -0.054 to 0.734, q 0.81), rest_lay_60_m0
  +0.798% (721, 90% -0.142 to 1.081), rest_lay_60_m3 +4.643% (29 trades, 0.194 to 7.659).
- **The stream lab** (8 days pooled, 170 tests). No candidate. The late family's second counted day: between +1.32%
  (gap_lay_late20_0.1, 22 trades) and -3.89% at the SP. rev_lay_10_0.1 to the result +24.87% (10.46 to 34.96,
  q 0.22), deflated Sharpe 0.0.
- **The early-lay forward test, 12 Nov.** Its own figure: 112 races, 523 lays placed, 170 matched, +1.42% of
  liability. Queue-aware replay: 226 matched, +1.62% at the SP, +4.92% to the result (traded volume x1.0; x0.5:
  +1.57%, +4.66%). The same lays resting to the real off (not the gate's figure): 238 matched, +1.43% at the SP,
  +4.59% to the result.

| | 11 Nov | 12 Nov |
|---|---|---|
| At the SP | +1.46% | +1.62% |
| To the result | +0.81% | +4.92% |

### 13 Nov 2026 (close lab 43525405017, stream lab 43520907174; read 14 Nov 00:15 UTC)

- **The day.** No greyhound prices were served until 16:04 UTC. The live early lay traded 16:16-21:30 UTC: 71 races
  laid, 312 lays, GBP7,859.58 asked, GBP2,044.66 matched, locked GBP-54.32 (GBP-55.41 after commission)
  (greyhound-live.yml running table, run 43511796787). The released stack (batch 32's m_stack, retrained 18:16 UTC,
  run 43447520432) priced from 19:30 UTC. The stream was recorded: 529,443 stream lines, 265 markets, 9,780 trades.
- **The close lab: not read: the scheduled run 43525405017 (23:12 UTC, success) did not score 13 Nov.** Its days
  recorded end at 12 Nov and its log has no 13 Nov line; every table repeats the 12 Nov run's. So no day table,
  champions, review readings, trade, band or resting figures for 13 Nov, and those tests' counted days stand as at
  12 Nov (trade 4, band 2, resting 2). No run was dispatched.
- **The stream lab** (9 days pooled, 170 tests; scheduled 22:38 UTC, success). No candidate. The late family's third
  counted day: between -0.34% (mom_back_late20_0.1, 145 trades) and -3.75% at the SP. rev_lay_10_0.1 to the result
  +24.11% (11.33 to 34.80, q 0.10), deflated Sharpe 0.0; gap_lay_60_0.1 +1.66% at the SP (108 trades, -1.27 to 4.65).
- **The early-lay forward test, 13 Nov.** Its own figure: 131 races, 606 lays placed, 177 matched, +1.3% of
  liability. Queue-aware replay: 236 matched, +0.93% at the SP, +1.78% to the result (traded volume x1.0; x0.5:
  +0.93%, +1.85%). The same lays resting to the real off (not the gate's figure): 238 matched, +0.79% at the SP,
  +1.57% to the result.

| | 11 Nov | 12 Nov | 13 Nov |
|---|---|---|---|
| At the SP | +1.46% | +1.62% | +0.93% |
| To the result | +0.81% | +4.92% | +1.78% |
