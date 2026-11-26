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
| 9 | The stream lab: 43 taker rules at the price on offer 60/30/10 s out, closed at the SP (projected-SP gap, momentum, reversal, a control) | the recorded Exchange Stream from 5 Nov, scored nightly, pooled | 43 | 0 | no edge: read at 10 days (5-14 Nov), no candidate; best at the SP gap_lay_60_0.1 +1.52% (-1.17 to 4.03), deflated Sharpe 9e-05. Registered 5 Nov (reports/greyhound_stream_lab.md) | greyhound-stream-lab.yml; run 43649155320 |
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

### 14 Nov 2026 (close lab 43651994554, stream lab 43649155320, greyhound-live running 43645103389, green audit 43593264909; read 15 Nov 00:13 UTC)

- **The close lab scored 13 Nov, not 14 Nov.** The scheduled run 43651994554 (23:11 UTC 14 Nov, success) lists days
  recorded 5-13 Nov: it reads the 13 Nov day the 13 Nov run missed (529,443 stream lines, 24,084 runner-checkpoints
  with an SP; 3,994 resting orders, 3,119 matched). 14 Nov is not scored yet. The figures below are 13 Nov's.
- **The day (13 Nov).** 132 races, 21,960 runner-checkpoints (3,840 race-checkpoints).
- **Champions.** BFSP: gbm_all_small holds against the mid (gain 0.03276, z 14.22, 875 races over 7 days); the
  strongest fresh challenger gbm_small_plus_huber: gain 0.00038 (90% 0.00010 to 0.00064), z 2.31, 395 races over 3
  days (a gain this size needs 692). Win: the mid; gbm_all_small's gain 0.00494 (90% 0.00181 to 0.00778), z 2.67 over
  863 races (needs 1,130).
- **The day's BFSP RMSE** (all / last 60 s / late): the mid 0.1913 / 0.1247 / 0.0899; gbm_all_small 0.1612 /
  0.1168 / 0.0882; gbm_book_flow 0.1679 / 0.1165 / 0.0868; clogit_win 0.1845 / 0.1470 / 0.1312; clogit_win_plus
  0.2040 / 0.1921 / 0.2979.
- **The review's readings.** The market closed a median 18 s after its scheduled off (10% 8 s, 90% 41 s); still
  open 30 s after on 23%, 60 s after on 5%. The SP pool: backs on 74% (median GBP47), lays on 98% (median GBP10
  liability). The SP book over 1,065 races: median 1.0039. Benter's second stage: the SP itself gains 0.00854 at 60 s
  (p 0.0000, 1,046 races); gbm_all_small 0.00017 (p 0.59) at 60 s over 863 races; gbm_small_plus_huber 0.00003
  (p 0.88) over 386.
- **The trade test** (fifth counted day of ten): no candidate; back_10_0.03 +4.05% at the SP (31 trades, 90% 0.31 to
  7.95), lay_60_0.03 +2.30% (74 trades), lay_30_0.03 +2.12% (69), lay_10_0.03 +1.11% (98).
- **The band test** (third counted day): 13 trades in all; quantile coverage q10 0.115, q20 0.209, q80 0.784,
  q90 0.882 (73,725 runner-checkpoints). **The resting test** (third counted day): no candidate; rest_back_60_all
  -0.067% at the SP (1,738 trades), rest_lay_60_all +0.117% (1,745), rest_back_60_m0 +0.681% (667), rest_back_60_m3
  +2.38% (72, -1.64 to 5.31), rest_lay_60_m3 +1.91% (42); at the scheduled off rest_back_late0_all -0.401% and
  rest_lay_late0_all -0.271%.
- **The stream lab, 14 Nov** (scheduled 22:38 UTC, success; 523,716 stream lines, 286 markets, 10,266 trades; 10
  days pooled, 85 rules, 170 tests). **The first family's tenth day (5-14 Nov): no candidate**, so row 9 is read: no
  edge. Best at the SP gap_lay_60_0.1 +1.52% (128 trades, -1.17 to 4.03, deflated Sharpe 9e-05); rev_lay_10_0.1 to
  the result +23.48% (10.61 to 32.19, q 0.08) and gap_lay_30_0.1 to the result +34.32% (11.37 to 64.83), deflated
  Sharpe 0.0 and 0.083 against the 0.95 needed. The late family (fourth counted day, pooled): every cell negative at
  the SP, from -0.92% (gap_lay_late10_0.06, 184 trades) to -4.21% (gap_back_late0_0.1).
- **The early-lay forward test, 14 Nov.** Its own figure: 137 races, 601 lays placed, 240 matched, +1.29% of
  liability. Queue-aware replay: 277 matched, +1.33% at the SP, -1.62% to the result (traded volume x1.0; x0.5:
  +1.32%, -1.60%). The same lays resting to the real off (not the gate's figure): 290 matched, +1.25% at the SP,
  -2.08% to the result.
- **The live early lay, 14 Nov (partial: the owner's greyhound/STOP at 19:29 UTC).** Live orders on the owner's
  account (the running table's early-lay section; the paper win trader in the same run is mode paper). The trader
  watched 140 races, priced 137; last race laid 19:38 UTC, then STOPPED (S3 STOP), 43 races skipped. Races laid 94;
  lays 425; GBP10,702.67 asked; GBP3,277.78 matched (30.6%); locked by green-ups GBP-196.49 (GBP-197.32 after
  commission; -6.0% of matched); GBP60.20 liability left to the SP or the result. 13 Nov: 71 races, GBP2,044.66 of
  GBP7,859.58 matched (26.0%), locked GBP-54.32 (-2.7%). More matched and a worse locked return.
  - **By lay price** (the dog-by-dog table: 151 dogs matched, GBP2,703.87; the race table's matched column, GBP3,277.78,
    is the larger figure and the log does not reconcile the two): under 5, 43 dogs, GBP1,411.43 matched, all greened,
    locked GBP-71.78 (-5.1%); 5 and over, 108 dogs, GBP1,292.44 matched, 100 greened locked GBP-124.71 (-9.6%), 8 held
    to the result (GBP4.74 matched). The table has the matched lays only, so no match rate by price.
  - **Betfair's settled bets** (green-audit 43593264909, 21:30 UTC, the same 151 dogs and GBP2,703.87): GBP-130.05
    (-4.8% of laid; a flat green-up GBP-136.12) against the ledger's GBP-196.49; 13 Nov GBP-26.65 on 106 dogs
    (GBP1,974.70, -1.35%) against GBP-54.32. The ledger overstates the loss (section 56), but 14 Nov is the worst live
    day on either reading.

| | 11 Nov | 12 Nov | 13 Nov | 14 Nov |
|---|---|---|---|---|
| At the SP | +1.46% | +1.62% | +0.93% | +1.33% |
| To the result | +0.81% | +4.92% | +1.78% | -1.62% |

Counted days: the stream lab's first family 10 of 10 (read: no candidate); its late family 4 of 10; the trade test 5
of 10; the band test 3 of 10; the resting test 3 of 10 (the close lab's three through 13 Nov, the stream lab's through 14 Nov).

### 15 Nov 2026 (close lab 43773721002, stream lab 43770393017, day replay 43768179777; read 16 Nov)

- **The live early lay stood down all day**: greyhound/STOP (the owner, 14 Nov 19:29 UTC) held all 94 races, 0 lays (running table 43766143179). GREYHOUND_EARLY_LAY still reads `live` (paper-models 43682498497: "greyhound early lay (live) exited 0"); the STOP is what stops it. The STOP also held the paper books: m10, short20, short10, spclose and spclose3 placed no lay ("no lays yet (0 dogs looked at)"), so the SP-close paper test (greyhound-early-lay-sp-close-1114) has no first day yet.
- **The day replayed** (research/greyhound/day_replay.py, the live model's prices, 94 races; GBP after commission, ROI on liability), live-key books / recorder's books: the rule as the running table fills it (traded) +306.02 (+1.41%) / +334.07 (+1.97%); the live rule with the fill the reconciliation trusts (through) -201.62 (-1.65%) / -244.25 (-3.24%); closed at the SP -402.76 (-3.29%) / -402.05 (-5.33%); spclose3 -366.57 / -340.66; m10 -432.16 / -506.28; short20 -29.60 / -70.60; short10 -75.91 / -166.35. The traded fill's 220 matched lays against the through fill's 125 are the whole difference. Descriptive (ledger greyhound-early-lay-day-replay-1115).
- **The close lab scored 14 Nov, not 15 Nov** (it runs a day behind; 14 Nov's entry left it unscored). 14 Nov: 523,716 stream lines, 22,836 runner-checkpoints with an SP; 123 races, 20,220 runner-checkpoints (3,570 race-checkpoints); 3,645 resting orders, 3,031 matched.
- **Champions.** BFSP: gbm_all_small holds against the mid (gain 0.03784, z 14.74, 998 races over 8 days); the strongest fresh challenger gbm_small_plus_huber gain 0.00041 (90% 0.00012 to 0.00070), z 2.33, 518 races over 4 days (needs 848). **Win: the crown passes from the mid to gbm_all_small**: gain 0.00558 (90% 0.00247 to 0.00840), z 3.15, 982 races over 8 days (needs 910).
- **14 Nov's BFSP RMSE** (all / last 60 s / late): mid 0.3527 / 0.1437 / 0.1088; gbm_all_small 0.2200 / 0.1355 / 0.1014; gbm_book_flow 0.2668 / 0.1351 / 0.1007; clogit_win 0.2910 / 0.1581 / 0.1355; clogit_win_plus 0.3007 / 0.1957 / 0.3004; best overall gbm_small_plus 0.2183.
- **The review.** The market closed a median 23 s after its scheduled off (10% 10 s, 90% 57 s); open 30 s after on 36%, 60 s after on 9%. SP pool: backs on 69% (median GBP35), lays on 88% (median GBP10 liability). SP book over 1,188 races median 1.0038. Benter's second stage at 60 s: the SP 0.00744 (p 0.0000, 1,165 races); gbm_all_small 9e-05 (p 0.68, 982); gbm_small_plus_huber 0.0 (p 0.96, 505).
- **The trade test** (sixth counted day of ten): no candidate. back_10_0.03 +3.81% at the SP (32 trades, 90% 0.068 to 7.526); lay_30_0.03 +2.98% (91); lay_60_0.03 +2.72% (91); lay_10_0.03 +2.07% (121, q 0.32).
- **The band test**: 16 trades over its rules; quantile coverage q10 0.117, q20 0.212, q80 0.779, q90 0.880 (96,561 runner-checkpoints).
- **The resting test** (fourth counted day): no candidate. rest_back_60_all -0.118% at the SP (2,263 trades), rest_lay_60_all +0.027% (2,265), rest_back_60_m0 +0.771% (844), rest_back_60_m3 +2.884% (89, -0.029 to 5.746), rest_lay_60_m3 +1.777% (63); at the scheduled off rest_back_late0_all -0.433%, rest_lay_late0_all -0.153%.
- **The stream lab, 15 Nov** (401,856 stream lines, 186 markets, 6,809 trades; 11 days pooled, 85 rules, 170 tests): no candidate. Best at the SP gap_lay_60_0.1 +0.875% (133 trades, -1.759 to 3.224, deflated Sharpe 2e-05). To the result gap_lay_30_0.1 +38.79% (16.90 to 66.52, q 0.33, deflated Sharpe 0.168), rev_lay_10_0.1 +17.37% (q 1.00, 0.0), against 0.95. The late family (fifth counted day): every cell negative at the SP, -0.91% (mom_back_late20_0.1, 257 trades) to -4.09% (gap_back_late0_0.1).
- **The early-lay forward test, 15 Nov**: 94 races, 437 lays placed, 176 matched, +1.97% of liability (its own traded fill). Queue-aware replay: 206 matched, +1.39% at the SP, -0.58% to the result (traded volume x1.0; x0.5 +1.37%, -0.76%). Resting to the real off: 220 matched, +1.23% at the SP, -1.02% to the result.

| | 11 Nov | 12 Nov | 13 Nov | 14 Nov | 15 Nov |
|---|---|---|---|---|---|
| At the SP | +1.46% | +1.62% | +0.93% | +1.33% | +1.39% |
| To the result | +0.81% | +4.92% | +1.78% | -1.62% | -0.58% |

- **The live model**: stack_mls (speed and sequence tables) retrained 15 Nov (43743136952: 643,830 runners, 116,752 races from 15 Jun 2024); its first price run 43760412455 priced 94 markets, 528 runners, none missed (peak 15.8 GB used, 14.5 GB of the 32 GB swap); it serves from 16 Nov's 07:21 UTC prices. Its out-of-sample prices published (fast loop 43760539283: greyhound/live/oos_prices_stack_mls, 37 days, 6 Oct - 11 Nov, 25,970 runners). The running table's favourite back fixed for a STOP day (PRs 362, 364).

Counted days: the stream lab's first family 11 days pooled (read at 10: no candidate); its late family 5 of 10; the trade test 6 of 10; the band test 4 of 10; the resting test 4 of 10 (the close lab's three through 14 Nov, the stream lab's through 15 Nov).

### 16 Nov 2026 (close lab 43977153802, stream lab 43971998763, day replay 43985632145; read 17 Nov)

- **The live early lay stood down all day**: greyhound/STOP (the owner, 14 Nov 19:29 UTC; no greyhound-stop run since) held all 136 races, 0 lays (running table 43964491117: "STOPPED (S3 STOP)"). GREYHOUND_EARLY_LAY still reads `live` (paper-models 43812370908: "greyhound early lay (live) exited 0"). The paper books m10, short20, short10, spclose and spclose3 placed no lay ("no lays yet (0 dogs looked at)"), so the SP-close paper test (greyhound-early-lay-sp-close-1114) still has no first day. The favourite back: 136 races priced, no back placed; no "not read" warning.
- **The day replayed** (43985632145): on the live-key books the live rule (through fill, greened) would have made -339.08 on 17,590.00 liability (-1.93%, 176 of 577 lays matched, 30 laid dogs won) against the traded fill's +111.56 (+0.43%, 265 matched); spclose -385.91 (-2.19%), spclose3 -385.91 (-2.21%), m10 -597.49 (-1.98%), short20 -2.59, short10 -8.50. On the recorder's books: the live rule -246.60 (-2.61%, 95 matched), the traded fill +228.85 (+1.27%).
- **The close lab scored 15 Nov**: 401,856 stream lines, 17,995 runner-checkpoints with an SP; 93 races, 15,660 runner-checkpoints (2,700 race-checkpoints); 2,789 resting orders, 2,359 matched.
- **Champions.** BFSP: gbm_all_small holds against the mid (gain 0.03589, z 15.94, 1,091 races over 9 days); the strongest fresh challenger gbm_small_plus_huber gain 0.00040 (90% 0.00015 to 0.00064), z 2.64, 611 races over 5 days (needs 811). **Win: the crown returns to the mid**: gbm_all_small gain 0.00470 (90% 0.00175 to 0.00756), z 2.66 over 1,072 races and 9 days (needs 1,284), under the bar a day after taking it.
- **15 Nov's BFSP RMSE** (all / last 60 s / late): mid 0.2232 / 0.1522 / 0.1164; gbm_all_small 0.1860 / 0.1481 / 0.1101; gbm_book_flow 0.1922 / 0.1496 / 0.1110; clogit_win 0.2111 / 0.1656 / 0.1296; clogit_win_plus 0.2342 / 0.2106 / 0.2623; best overall gbm_small_plus 0.1851. Win log-loss: mid 1.5584, oracle SP 1.5451.
- **The review.** The market closed a median 25 s after its scheduled off (10% 9 s, 90% 87 s); open 30 s after on 42%, 60 s after on 13%. SP pool: backs on 78% (median GBP53), lays on 93% (median GBP14 liability). SP book over 1,281 races median 1.0040. Benter's second stage at 60 s: the SP 0.00722 (p 0.0000, 1,255 races); gbm_all_small 1e-05 (p 0.88, 1,072); gbm_small_plus_huber 0.00024 (p 0.59, 595).
- **The trade test** (seventh counted day of ten): no candidate. back_10_0.03 +3.81% at the SP (32 trades, 90% 0.068 to 7.526); lay_30_0.03 +2.88% (108); lay_60_0.03 +2.10% (107, q 0.57); lay_10_0.03 +1.25% (143, q 0.59).
- **The band test**: 31 trades over its rules; quantile coverage q10 0.119, q20 0.216, q80 0.774, q90 0.876 (114,556 runner-checkpoints).
- **The resting test** (fifth counted day): no candidate. rest_back_60_all -0.081% at the SP (2,668 trades), rest_lay_60_all -0.102% (2,658), rest_back_60_m0 +0.773% (1,028), rest_back_60_m3 +2.328% (97, -0.527 to 4.908), rest_lay_60_m3 +2.621% (70); at the scheduled off rest_back_late0_all -0.435%, rest_lay_late0_all -0.223%.
- **The stream lab, 16 Nov** (583,858 stream lines, 270 markets, 10,172 trades; 12 days pooled, 85 rules, 170 tests): no candidate. Best at the SP gap_lay_30_0.1 +1.09% (128 trades, -0.769 to 3.184, deflated Sharpe 0.000846). To the result gap_lay_30_0.1 +38.41% (13.42 to 65.57, q 0.68, deflated Sharpe 0.136), rev_lay_10_0.1 +12.89% (q 1.00, 0.0), against 0.95. The late family (sixth counted day): every cell negative at the SP, -0.67% (gap_lay_late0_0.06, 337 trades) to -3.57% (rev_back_late10_0.1).
- **The early-lay forward test, 16 Nov**: 135 races, 613 lays placed, 185 matched, +1.27% of liability (its own traded fill). Queue-aware replay: 242 matched, +0.63% at the SP, -2.75% to the result (traded volume x1.0; x0.5 +0.58%, -2.66%). Resting to the real off: 276 matched, +0.54% at the SP, -3.08% to the result.

| | 11 Nov | 12 Nov | 13 Nov | 14 Nov | 15 Nov | 16 Nov |
|---|---|---|---|---|---|---|
| At the SP | +1.46% | +1.62% | +0.93% | +1.33% | +1.39% | +0.63% |
| To the result | +0.81% | +4.92% | +1.78% | -1.62% | -0.58% | -2.75% |

- **The live price runs, 16 Nov** (stack_mls's first day): 13 of 13 scheduled runs succeeded, each queued behind the last (started 07:41 to 20:36 UTC, last finished 21:56); every run priced 137 markets (778 runners at first, falling to 768), races_not_priced 0; peak 15,890 MB used, 17,271 MB of the 32 GB swap (43907773903). The log's model field reads stack_share (the objective); the recipe name is not logged. 137 markets priced against 136 races in the early lay and the running table: not reconciled in the logs.
- **The live model from 17 Nov**: stack_mlsi, the Irish form added (the owner: "Yes build it"; PR 382; ledger greyhound-irish-batch45-1116), retrained by greyhound-live.yml 43971830784; the sire and dam abroad failed (greyhound-world-batch46-1116). Its out-of-sample prices: fast loop 43971835560 (batch 48), which also replicates the Irish gain against batch 42.

Counted days: the stream lab's first family 12 days pooled (read at 10: no candidate); its late family 6 of 10; the trade test 7 of 10; the band test 5 of 10; the resting test 5 of 10 (the close lab's three through 15 Nov, the stream lab's through 16 Nov).

### 17 Nov 2026 (close lab 44197429985, stream lab 44192106565, day replay 44205737671; read 18 Nov)

- **The live early lay stood down all day**: greyhound/STOP held (running table 44184438191: "STOPPED (S3 STOP)", 101 races skipped, 0 lays). GREYHOUND_EARLY_LAY still reads `live` (paper-models 44026343278: "greyhound early lay (live) exited 0"). spclose and spclose3 placed no lay ("no lays yet (0 dogs looked at)"): the SP-close paper test (greyhound-early-lay-sp-close-1114) still has no first day.
- **The day replayed** (44205737671): on the live-key books the live rule (through fill, greened) -182.22 on 13,779.00 liability (-1.32%, 138 of 432 lays matched, 22 laid dogs won) against the traded fill's +348.64 (+1.52%, 230 matched); spclose -217.29 (-1.58%), spclose3 -213.00 (-1.60%), m10 -322.49 (-1.40%), short20 -6.51, short10 +20.91. On the recorder's books: the live rule -169.34 (-1.88%, 90 matched), the traded fill +154.34 (+0.91%).
- **The close lab scored 16 Nov**: 583,858 stream lines; 135 races, 22,695 runner-checkpoints (4,017 race-checkpoints). BFSP RMSE: gbm_small_plus_huber 0.1708, gbm_all_small 0.1724, mid 0.1905, ltp 0.1971. Win log-loss: gbm_small_plus_huber 1.5384, mid 1.5389, oracle SP 1.5222.
- **Champions.** BFSP: gbm_all_small holds against the mid (gain 0.03264, z 15.13, 1,226 races over 10 days); the challenger gbm_small_plus_huber gain 0.00042 (90% 0.00021 to 0.00062), z 3.37, 746 races over 6 days (needs 644 more). Win: the mid holds; gbm_all_small gain 0.00418 (0.00151 to 0.00658), z 2.71 over 1,206 races and 10 days (needs 1,493).
- **The trade, band and resting tests**: no candidate. Best trade q on target 2 lay_30_0.03 (129 trades over 8 days, q 0.00, candidate False/False); resting rest_back_60_m0 +1.05% at the SP (90% 0.286 to 1.593, q 0.17). The market closed a median 23 s after its scheduled off.
- **The stream lab, 17 Nov** (431,878 stream lines, 204 markets, 7,856 trades; 13 days pooled, 85 rules, 170 tests): no candidate. Top gap_lay_10_0.1 +0.144% at the SP (186 trades, -0.661 to 2.215, q 1.00, deflated Sharpe 4.4e-05).
- **The early-lay forward test, 17 Nov**: 100 races, 454 lays placed, 172 matched, +0.91% of liability (its own traded fill). Queue-aware replay: 228 matched, +1.95% at the SP, +0.79% to the result (x0.5: +1.94%, +0.61%). Resting to the close: 241 matched, +1.73% at the SP, +1.25% to the result.

| | 12 Nov | 13 Nov | 14 Nov | 15 Nov | 16 Nov | 17 Nov |
|---|---|---|---|---|---|---|
| At the SP | +1.62% | +0.93% | +1.33% | +1.39% | +0.63% | +1.95% |
| To the result | +4.92% | +1.78% | -1.62% | -0.58% | -2.75% | +0.79% |

- **The live price runs, 17 Nov** (stack_mlsi's first day): 13 fired, 11 succeeded, 2 cancelled by the next hour's run (12:30, 15:24; no jobs); none failed. 103 markets priced to 17:25 (races_not_priced 0), 102 at 18:27 and 19:25 (races_not_priced 1); runners 584 falling to 566. Peak 15,894 MB used and 17,427 MB of the 32 GB swap (44038751378). The GreyhoundAPI results job finished 05:45 UTC, before the 07:26 price run; no Irish-context warning.
- **The residual memory went live from the 16:26 UTC price run** (PR 396; ledger greyhound-memory-live-1117): the hourly price runs check out the merged code, so it served from that run, not from 18 Nov as first said. Runners with a memory 547/545/537/537 across the four runs; greyhound/live/oos_prices_stack_mlsi_mem written (37 days) at 16:26. The early lay was stopped, so no order read it.

Counted days: the stream lab's first family 13 days pooled (no candidate); the trade test 8 days; the resting test 6 days; the band test no candidate (the close lab through 16 Nov, the stream lab through 17 Nov).

### 18 Nov 2026 (close lab 44418814037, stream lab 44413556877, day replay 44427004281; read 19 Nov)

- **The early lay moved to paper**: GREYHOUND_EARLY_LAY reads `paper` from the 14:10 UTC run (paper-models 44321292727: "greyhound early lay (paper) exited 0"; 44246555591 at 05:46 still read `live`). greyhound/STOP held. Running table 44405941242: live rule 17 races skipped and paper rule 131 skipped ("STOPPED (S3 STOP)"), 0 lays. spclose and spclose3 placed no lay ("no lays yet (0 dogs looked at)"): the SP-close paper test still has no first day.
- **The take and close traded live, then was stopped by the owner**: bar 0.08, GBP100 a bet. Its own switch: stop 13:48 (44317551907), resume 14:09 (44321269861), stop 14:45 (44327821357, greyhound/take_close/STOP). Exit line (44321292727): 131 markets decided, 28 positions, settled net -27.45, 0 unsettled. Status 44327581198 (fills to 14:37): 27 backs, GBP394.63 staked, CLV mean -3.51%, 4 winners, -26.76 before commission; the 28th position is not itemised in any log read (no data found).
- **The day replayed** (44427004281): on the live-key books the live rule (through fill, greened) would have made -255.11 on 19,010.50 liability (-1.34%, 192 of 653 lays matched, 33 laid dogs won) against the traded fill's +415.44 (+1.26%, 333 matched); spclose -260.88 (-1.37%), spclose3 -248.66 (-1.34%), m10 -579.99 (-1.82%), short20 -11.96, short10 -45.68. On the recorder's books: the live rule -239.27 (-2.49%, 97 matched), the traded fill +224.64 (+1.16%).
- **The close lab scored 17 Nov**: 431,878 stream lines, 18,976 runner-checkpoints with an SP; 102 races, 16,950 runner-checkpoints (3,000 race-checkpoints); 3,187 resting orders, 2,639 matched. BFSP RMSE: gbm_small_plus_huber 0.1925, gbm_all_small 0.1931, mid 0.2357, ltp 0.2418. Win log-loss: gbm_small_plus_huber 1.5299, mid 1.5327, oracle SP 1.5184.
- **Champions.** BFSP: **gbm_small_plus_huber** is now named the incumbent (17 Nov's entry had gbm_all_small; this log does not say when the crown changed); it holds against the mid (gain 0.02989, z 11.94, 848 races over 7 days); its challenger gbm_small_plus gain -0.00012, z -1.05. Win: the mid holds; gbm_all_small gain 0.00399 (90% 0.00157 to 0.00635), z 2.76 over 1,306 races and 11 days (needs 1,576).
- **The review.** The market closed a median 23 s after its scheduled off (10% 6 s, 90% 53 s); open 30 s after on 36%, 60 s after on 7%. SP pool: backs on 91% (median GBP47), lays on 93% (median GBP15 liability). SP book over 1,518 races median 1.0039. Benter at 60 s: the SP 0.00705 (p 0.0000, 1,489 races); gbm_all_small 0.0001 (p 0.61, 1,306); gbm_small_plus_huber 0.00039 (p 0.42, 829).
- **The trade, band and resting tests**: no candidate. Trade: lay_30_0.03 +1.444% at the SP (139 trades, 9 days, q 0.43); back_10_0.03 +3.81% (32 trades, 0.068 to 7.526). Band: 50 trades; coverage q10 0.126, q20 0.224, q80 0.778, q90 0.877 (158,674 runner-checkpoints). Resting: rest_back_60_m0 +1.447% at the SP (1,374 trades, 0.724 to 2.03, q 0.00); rest_back_60_m3 +2.498% (109); rest_lay_60_m3 +1.877% (104); rest_back_60_all +0.171%; rest_lay_60_all -0.341%.
- **The stream lab, 18 Nov** (544,920 stream lines, 301 markets, 11,602 trades; 14 days pooled, 85 rules, 170 tests): no candidate. Top at the SP gap_lay_60_0.1 +0.217% (177 trades, -2.203 to 2.364, deflated Sharpe 3e-06). Late family (8 days): every cell negative at the SP, -1.22% (gap_lay_late10_0.06) to -3.65% (gap_back_late20_0.1).
- **The early-lay forward test, 18 Nov**: 150 races, 662 lays placed, 197 matched, +1.16% of liability (its own traded fill). Queue-aware replay: 267 matched, +1.46% at the SP, +1.13% to the result; resting to the close: 300 matched, +1.30% at the SP, +1.31% to the result.

| | 13 Nov | 14 Nov | 15 Nov | 16 Nov | 17 Nov | 18 Nov |
|---|---|---|---|---|---|---|
| At the SP | +0.93% | +1.33% | +1.39% | +0.63% | +1.95% | +1.46% |
| To the result | +1.78% | -1.62% | -0.58% | -2.75% | +0.79% | +1.13% |

- **The live price runs, 18 Nov** (stack_mlsitd, the _mem out-of-sample set): 13 fired, 10 succeeded, 3 cancelled (superseded), none failed; every run priced 153 markets, races_not_priced 0; runners 869 to 859, with a memory 825 to 816; m_sd 0.0441; no "not applied" warning. Peak 15,879 MB used (44347006662), 17,633 MB of the 32 GB swap (44267230863). GreyhoundAPI cards: 13 runs, no HTTP 500 on /racecards/today (recovered).

Counted days: the stream lab's first family 14 days pooled (no candidate); its late family 8 days; the trade test 9 days; the resting test 7 days; the band test no candidate.

### 19 Nov 2026 (close lab 44635535738, stream lab 44630527322, day replay 44643820189; read 20 Nov)

- **The early lay stayed stopped; its switch still reads `live`**: the scheduled run (paper-models 44467445587) printed "greyhound early lay (live) exited 0", so the repository variable GREYHOUND_EARLY_LAY reads `live` (the workflow takes `inputs.early_lay || vars.GREYHOUND_EARLY_LAY || 'paper'`). 18 Nov's `paper` came from that day's dispatch input, not the variable: the 18 Nov entry's "moved to paper" is corrected here. greyhound/STOP held (status 44506415493 "ON: greyhound/STOP"; running table 44623179590 "STOPPED (S3 STOP)"): 117 races skipped, 0 lays. spclose and spclose3 placed no lay ("no lays yet (0 dogs looked at)"): the SP-close paper test still has no first day. The favourite back placed none (117 races skipped; no reason in the log).
- **The take and close stayed on paper**: GREYHOUND_TAKE_CLOSE `paper`, greyhound/take_close/STOP ON (44506415493); 117 markets decided, 0 positions, settled net 0. No live status or ledger for 19 Nov (404).
- **The day replayed** (44643820189): on the live-key books the live rule (through fill, greened) would have made -294.82 on 14,266.00 liability (-2.07%, 144 of 532 lays matched, 29 laid dogs won) against the traded fill's +370.02 (+1.32%, 282 matched); spclose -307.92 (-2.16%), spclose3 -275.38 (-2.01%), m10 -551.87 (-2.08%), short20 -19.83, short10 -77.19. On the recorder's books: the live rule -237.47 (-2.75%, 88 matched), the traded fill +281.68 (+1.64%).
- **The close lab scored 18 Nov**: 544,920 stream lines, 28,313 runner-checkpoints with an SP; 149 races, 25,200 runner-checkpoints (4,410 race-checkpoints); 4,620 resting orders, 3,788 matched. BFSP RMSE: gbm_small_plus_huber 0.1922, gbm_all_small 0.1930, mid 0.3033, ltp 0.3177. Win log-loss: gbm_all_small 1.4826, gbm_small_plus_huber 1.4835, mid 1.4971, oracle SP 1.4649.
- **Champions.** BFSP: gbm_small_plus_huber holds against the mid (gain 0.03353, z 13.54, 997 races over 8 days); challenger gbm_small_plus gain -0.00014 (90% -0.00031 to 0.00004), z -1.31. **Win: gbm_all_small takes the crown from the mid** (gain 0.00505, 90% 0.00271 to 0.00719, z 3.66, 1,453 races over 12 days; the log line still prints "needs 971 races to clear the bar", which 1,453 exceeds). A forecast of the close, not a trade: Benter at 60 s gives it 4e-05 (p 0.73) beside the SP.
- **The review.** The market closed a median 19 s after its scheduled off (10% 4 s, 90% 73 s); open 30 s after on 33%, 60 s after on 14%. SP pool: backs on 89% (median GBP45), lays on 86% (median GBP16 liability). SP book over 1,667 races median 1.0038. Benter at 60 s: the SP 0.0076 (p 0.0000, 1,636 races); gbm_all_small 4e-05 (p 0.73, 1,453); gbm_small_plus_huber 0.0003 (p 0.44, 976).
- **The trade, band and resting tests.** Trade (10 days): the lab's table flags lay_30_0.03 and lay_30_0.06 on the Win column ("False / True"). lay_30_0.03: 164 trades, +2.017% at the SP (-1.064 to 4.639, q 0.40), +53.5% to the result (24.611 to 85.4, q 0.01); lay_30_0.06: 69 trades, +98.0% to the result. The table prints no deflated Sharpe: neither is read against the ledger rule (deflated Sharpe >= 0.95 with every trial counted), so neither is a pass; at the SP both intervals span zero. lay_10_0.1 +7.528% at the SP (47 trades, 9 days, 0.791 to 13.009). Band: no candidate; coverage q10 0.125, q20 0.223, q80 0.778, q90 0.877 (186,987 runner-checkpoints). Resting (8 days): rest_back_60_m0 +1.259% at the SP (1,652 trades, 0.593 to 1.771, q 0.01); rest_back_60_m3 +2.264% (131); rest_lay_60_m3 +2.152% (114); rest_back_60_all +0.149%; rest_lay_60_all -0.367%.
- **The stream lab, 19 Nov** (441,683 stream lines, 230 markets, 10,000 trades; 15 days pooled, 85 rules, 170 tests): no candidate. Top at the SP gap_lay_60_0.1 +0.304% (193 trades, -1.875 to 2.288, deflated Sharpe 1e-06). Late family (9 days): every cell negative at the SP, -1.266% (mom_back_late20_0.1) to -3.915% (gap_back_late20_0.1).
- **The early-lay forward test, 19 Nov**: 115 races, 558 lays placed, 176 matched, +1.64% of liability (GBP+281.68 on 17,142.00; its own traded fill). Queue-aware replay: 235 matched, +1.48% at the SP, -1.10% to the result (x0.5: +1.51%, -1.40%); resting to the close: 265 matched, +1.23% at the SP, +0.14% to the result.

| | 13 Nov | 14 Nov | 15 Nov | 16 Nov | 17 Nov | 18 Nov | 19 Nov |
|---|---|---|---|---|---|---|---|
| At the SP | +0.93% | +1.33% | +1.39% | +0.63% | +1.95% | +1.46% | +1.48% |
| To the result | +1.78% | -1.62% | -0.58% | -2.75% | +0.79% | +1.13% | -1.10% |

- **The live price runs, 19 Nov** (stack_mlsitd, the _mem out-of-sample set of 37 days; the log's model field reads its objective, stack_share): 13 fired, 11 succeeded, 2 cancelled while queued (superseded: 44533353196, 44566126768), none failed. Every run priced 118 markets, races_not_priced 0; runners 680 to 674, with a memory 640 to 635; m_sd 0.0495; no "not applied" warning. tables_seconds 1,831 to 3,045 a run (18 Nov: 1,861 to 3,663; the 2,902 quoted on 18 Nov was one run, 44313487247): the runs vary within the day, so 2,902 vs 1,622 was not a step change. Each run takes about 60-85 min. Peak 15,884 MB used and 17,814 MB of the 32 GB swap (44588221206). The locked holdout's first day: no log line names greyhound/live/predictions/2026-11-19.csv (no data found); the runs' predict output priced 118 markets. GreyhoundAPI cards: 12 runs, no HTTP 500.

Counted days: the stream lab's first family 15 days pooled (no candidate); its late family 9 days; the trade test 10 days (two Win flags, not read against the ledger rule); the resting test 8 days; the band test no candidate.

### 20 Nov 2026 (close lab 44835118361, stream lab 44830616864, day replay 44880490429; read 21 Nov)

- **The early lay stayed stopped; its switch still reads `live`**: paper-models 44683230422 (the only run created on 20 Nov) printed "greyhound early lay (live) exited 0"; 120 markets priced, 133 closed, locked_pnl 0.0, open_liability 0. greyhound/STOP held: running table 44824169716 printed "STOPPED (S3 STOP)", 120 races skipped, 0 lays; the paper trader logged "kill switch on: no new bets". spclose and spclose3 placed no lay ("no lays yet (0 dogs looked at)"): the SP-close paper test still has no first day. The favourite back placed none (120 races skipped; no reason logged).
- **The take and close stayed on paper**: GREYHOUND_TAKE_CLOSE `paper` (44683230422): 120 markets decided, 0 positions, settled net 0. greyhound/take_close/STOP on 20 Nov: no data found (no status run on 20 or 21 Nov; the last, 44506415493 on 19 Nov, read ON).
- **The day replayed** (44880490429): on the live-key books the live rule (through fill, greened) would have made -223.78 on 15,219.50 liability (-1.47%, 154 of 517 lays matched, 23 laid dogs won) against the traded fill's +232.25 (+0.9%, 262 matched); spclose -91.86 (-0.6%), spclose3 -94.37 (-0.64%), m10 -442.00 (-1.69%), short20 -17.67, short10 -11.53. Recorder's books: live rule -224.00 (-2.46%, 92 of 555), traded fill +199.71 (+1.08%, 188). The through fill is the harsh bound; the trusted fill (greyhound-queue-calibration-1120: queue as shown, trade-through capped, half the printed volume) is +0.64% at the SP (below).
- **The close lab scored 19 Nov**: stream-line and resting-order counts not printed (no data found). 115 races, 19,680 runner-checkpoints (3,390 race-checkpoints). BFSP RMSE: gbm_all_small 0.2046, gbm_small_path 0.2047, gbm_small_plus_huber 0.2051, mid 0.2619, ltp 0.2779. Win log-loss: gbm_win_all 1.4631, clogit_win 1.4639, gbm_all_small 1.4686, gbm_small_plus_huber 1.4691, mid 1.4715, oracle SP 1.4523.
- **Champions.** BFSP: gbm_small_plus_huber holds against the mid (gain 0.03273, z 14.89, 1,112 races over 9 days); challenger gbm_small_plus gain -0.00012 (90% -0.00029 to 0.00004), z -1.24. Win: gbm_all_small holds against the mid (gain 0.00489, z 3.75, 1,566 races over 13 days); challenger gbm_small_path gain 0.00018 (90% -0.00007 to 0.00043), z 1.18.
- **The review.** The market closed a median 24 s after its scheduled off (10% 7 s, 90% 69 s); open 30 s after on 43%, 60 s after on 14%. SP pool: backs on 76% (median GBP44), lays on 89% (median GBP12 liability). SP book over 1,782 races median 1.0041 (10% 0.9822, 90% 1.0233). Benter at 60 s: the SP 0.00732 (p 0.0000, 1,749 races); gbm_all_small 6e-05 (p 0.6621); gbm_small_plus_huber 0.00051 (p 0.2920).
- **The trade, band and resting tests.** Trade (11 days): the table flags lay_60_0.1 at the SP (35 trades, 10 days, +7.504%, 0.061 to 14.3) and lay_30_0.03, lay_30_0.06, lay_30_0.1 on Win (lay_30_0.03: 170 trades, +1.691% at the SP, -1.373 to 4.161; +55.071% to the result). The trade table prints no deflated Sharpe: no flag is read against the ledger rule and none is a pass. Band: no candidate (209,144 runner-checkpoints). Resting (9 days): rest_back_60_m0 +1.087% at the SP (1,852 trades, 0.516 to 1.654, q 0.02); rest_back_60_m3 +2.831% (141); rest_lay_60_m3 +1.431% (128); rest_back_60_all +0.118%; rest_lay_60_all -0.394%. (The resting back's gates are the bracket's, read on or after 19 Dec.)
- **The stream lab, 20 Nov** (439,860 stream lines, 250 markets, 9,684 trades; 16 days pooled, 85 rules, 170 tests): no candidate. Top at the SP gap_lay_10_0.1 +0.108% (229 trades, -0.824 to 1.876, deflated Sharpe 2.1e-05). Late family (10 days): every cell negative at the SP, -1.172% (mom_back_late20_0.1) to -8.404% (rev_lay_late20_0.1). gap_lay_30_0.1 to the result +35.896% (q 0.09), Win deflated Sharpe 0.24106: not a candidate.
- **The early-lay forward test, 20 Nov**: 121 races, 546 lays placed, 185 matched, +1.06% of liability (GBP+192.71 on 18,166.50; its own traded fill). Queue-aware replay: 246 matched, +2.2% at the SP, +4.39% to the result (x0.5: +2.15%, +4.03%). **The trusted fill**: 239 matched, 18,220.38 liability, +0.64% at the SP. Resting to the close: 268 matched, +1.84% at the SP, +3.17% to the result.

| | 13 Nov | 14 Nov | 15 Nov | 16 Nov | 17 Nov | 18 Nov | 19 Nov | 20 Nov |
|---|---|---|---|---|---|---|---|---|
| At the SP | +0.93% | +1.33% | +1.39% | +0.63% | +1.95% | +1.46% | +1.48% | +2.2% |
| To the result | +1.78% | -1.62% | -0.58% | -2.75% | +0.79% | +1.13% | -1.10% | +4.39% |

- **The live price runs, 20 Nov** (stack_mlsitd, the _mem out-of-sample set): 13 fired, 9 succeeded, 3 cancelled while queued (superseded: 44727451911, 44753331499, 44783049146), **1 failed: 44773174225** (16:25 schedule; "The hosted runner lost communication with the server", ended 18:52; log gone, BlobNotFound). No price update landed between 16:24 and 18:28 UTC. 141 markets priced to 13:24, 133 from 15:24; races_not_priced 0; runners 803 to 750, with a memory 767 to 720; m_sd 0.048 then 0.0466; no "not applied" warning. Peak 15,895 MB used and 17,719 MB swap (44710743114): the memory headroom is thin. The locked holdout's file for 20 Nov is not named in any log (no data found; written by store.put without a log line).
- **GreyhoundAPI**: 12 cards runs on 20 Nov, no HTTP 500 and no racecards/today error (06:46 run 44690177402: today 343, upcoming 109).
- **The residual on the BSP, forward on paper**: frozen 20 Nov 15:37 UTC (44759069462); the 16:48 serve (44777088142) wrote nothing (before the first forward day, 21 Nov).

Counted days: the stream lab's first family 16 days pooled (no candidate); its late family 10 days (no candidate); the trade test 11 days (flags not read against the ledger rule); the resting test 9 days; the band test no candidate.

### 21 Nov 2026 (close lab 44960598003, stream lab 44957817630, day replay 44965400720; read 22 Nov)

- **Real money: the residual's lay at the SP went live** (greyhound/sp_lay.py; the owner, 21 Nov: the bottom-10% SP lay, GBP50 a lay; to be placed too from 13:21 UTC; one GBP1,000 daily loss limit at the worst case, GBP300 before 13:21). paper-models 44883754133 (08:12-13:21, win only) and 44908471220 (13:21 to the end), each dispatched with sp_lay `live`: "greyhound lay at the SP (live) exited 0"; 82 picks served, 146 lays (79 win, 67 place), all PENDING at Betfair, 1 skip ("runner 5344573 REMOVED"), 0 holds, settled net +GBP126.34 after commission at 21:45:06 UTC with GBP100 open (status 44965402852). Two picks never decided (Romford 22:02 and 22:20): the trader stopped at 21:45 UK (fixed: to 22:30 UK, PR 480). greyhound/sp_lay/STOP and trading/STOP off all day. Betfair's reconciliation of the day comes at the next start.
- **The early lay ran on paper after 08:12; greyhound/STOP held.** Three paper-models runs: the scheduled 44872421264 (05:46, GREYHOUND_EARLY_LAY `live`) cancelled at 08:12:01 with no exit line; 44883754133 (`paper`) cancelled at 13:21:34; 44908471220 (`paper`) to the end: "greyhound early lay (paper) exited 0", 137 markets priced, 138 closed, locked_pnl 0.0. Running table 44953829616: "STOPPED (S3 STOP)", 137 races skipped, 0 lays; spclose and spclose3 "no lays yet (0 dogs looked at)": the SP-close paper test still has no first day. The favourite back placed none (137 races skipped; no reason logged). Restart gaps: about 10 s at 08:12 and 11 s at 13:21 at the job level; the traders' first lines after the second at 13:24 UTC.
- **The take and close stayed on paper**: 137 markets decided, 0 positions, settled net 0. greyhound/take_close/STOP on 21 Nov: no data found.
- **The day replayed** (44965400720): on the live-key books the live rule (through fill, greened) would have made -242.74 on 18,931.00 liability (-1.28%, 191 of 563 lays matched, 41 laid dogs won) against the traded fill's +375.38 (+1.18%, 321 matched); spclose -229.73 (-1.21%), spclose3 -251.69 (-1.38%), m10 -582.12 (-1.9%), short20 +3.66, short10 -25.23. Recorder's books: live rule -328.05 (-2.76%, 122 of 591), traded fill +239.47 (+1.18%, 208). The through fill is the harsh bound; the trusted fill is -0.3% at the SP (below).
- **The close lab scored 20 Nov**: 439,860 stream lines, 23,612 runner-checkpoints with an SP; 125 races, 21,180 runner-checkpoints (3,690 race-checkpoints); 3,793 resting orders, 3,168 matched. BFSP RMSE: gbm_all 0.2266, gbm_small_plus_bw 0.2276, gbm_all_small 0.2277, gbm_small_plus_huber 0.2306, mid 0.2721, ltp 0.2773. Win log-loss: ltp 1.6180, mid 1.6204, gbm_book_flow_sp 1.6205, gbm_all_small 1.6245, gbm_small_plus_huber 1.6250, oracle SP 1.6499 (on 20 Nov the oracle SP scored worse than the mid: unexplained, to check).
- **Champions.** BFSP: gbm_small_plus_huber holds against the mid (gain 0.03144, z 14.96, 1,237 races over 10 days); challenger gbm_small_plus_bw gain -0.00021, z -0.78. Win: gbm_all_small holds (gain 0.00424, z 3.22, 1,689 races over 14 days); challenger gbm_small_path gain 0.00008, z 0.60.
- **The review.** Closed a median 18 s after the off (10% 8 s, 90% 57 s); open 30 s after on 29%, 60 s after on 10%. SP pool: backs on 69% (median GBP41), lays on 85% (median GBP10 liability). SP book over 1,907 races median 1.0041. Benter at 60 s: the SP 0.00527 (p 0.0000); gbm_all_small 4e-05 (p 0.7151); gbm_small_plus_huber 0.00025 (p 0.4396).
- **The trade, band and resting tests.** Trade (12 days): flags lay_60_0.1 at the SP (39 trades, +7.818%, 0.831 to 13.806), lay_10_0.1 on both (52 trades; +7.619% at the SP; +72.077% to the result), lay_30_0.06 and lay_30_0.03 on Win; no deflated Sharpe printed: none is a pass. Band: no candidate. Resting (10 days): rest_back_60_m0 +1.102% at the SP (2,035 trades, 0.595 to 1.683, q 0.01) and rest_back_60_m3 +2.501% (150) flagged; no deflated Sharpe printed; the resting back's gates are the bracket's, read on or after 19 Dec.
- **The stream lab, 21 Nov** (475,638 stream lines, 280 markets, 10,134 trades; 17 days pooled, 85 rules, 170 tests): no candidate. Top at the SP gap_lay_60_0.1 +0.381% (225 trades, -1.688 to 2.184, deflated Sharpe 2e-06). Late family (11 days): every cell negative at the SP, -1.305% to -8.052%.
- **The early-lay forward test, 21 Nov**: 135 races, 591 lays placed, 208 matched, +1.18% of liability (its own traded fill). Queue-aware replay: 271 matched, +1.06% at the SP, -4.52% to the result. **The trusted fill**: 269 matched, 21,540.28 liability, -0.3% at the SP. Resting to the close: 297 matched, +0.95% at the SP, -3.84% to the result.

| | 13 Nov | 14 Nov | 15 Nov | 16 Nov | 17 Nov | 18 Nov | 19 Nov | 20 Nov | 21 Nov |
|---|---|---|---|---|---|---|---|---|---|
| At the SP | +0.93% | +1.33% | +1.39% | +0.63% | +1.95% | +1.46% | +1.48% | +2.2% | +1.06% |
| To the result | +1.78% | -1.62% | -0.58% | -2.75% | +0.79% | +1.13% | -1.10% | +4.39% | -4.52% |

- **The residual on the BSP, forward on paper**: four serves (44881994153, 44896153292, 44910782780, 44926627649), each 149 races, 829 runners; picks back_top10 77, back_top5 30, back_top2 7, lay_bottom10 82, lay_bottom5 44, lay_bottom2 15; code_changed_since_freeze false in all four. **A fault, fixed**: the 12:31 settle (44904328145) settled 21 Nov while it was still running (13 of 149 races; lay_bottom10 8 bets, net -0.1) and its reading counted it ("running: 1 of 120 days"). From 22 Nov the settle takes completed days only and the reading loads days to yesterday; 21 Nov is settled in full by 22 Nov's 12:26 run.
- **The live price runs, 21 Nov**: 13 fired, 10 succeeded, 2 cancelled while queued (superseded), **1 failed: 44889494619** (09:25 schedule; "The hosted runner lost communication with the server", the second day running; log gone): no price update between 09:51 and 12:33 UTC. 149 markets priced to 08:26 (races_not_priced 0); from 10:24, 148 with races_not_priced 1 (no reason logged). m_sd 0.0485; no "not applied" warning. Peak 15,906 MB used and 17,151 MB swap: the headroom is thin, and two lost runners in two days point at memory.
- **GreyhoundAPI**: 14 runs, no HTTP 500, no racecards/today error.

Counted days: the stream lab's first family 17 days pooled (no candidate); its late family 11 days; the trade test 12 days (flags not read against the ledger rule); the resting test 10 days; the band test no candidate; the residual's forward test 0 completed days settled (21 Nov is settled on 22 Nov).

### 22 Nov 2026 (close lab 45082043123, stream lab 45078838714, day replay 45087314920; read 23 Nov)

- **The live lay at the SP, second day** (greyhound/sp_lay.py). The scheduled run 44990829901 (05:46) read GREYHOUND_SP_LAY `paper`: the owner's setting was then a repository secret, not a variable (moved to a variable on 22 Nov). Restarted live at 11:23 (45017050921, sp_lay `live`): "greyhound lay at the SP (live) exited 0" at 22:30:13; 51 picks served, 72 lays (36 win, 36 place), all PENDING at Betfair, 20 skip rows (10 picks x 2 markets, each a runner removed after the serve), 0 holds, settled net +GBP145.67 with GBP100 open; 5 picks laid on paper only (races off before 11:23); status 45087317483. **21 Nov reconciled** at that start against Betfair's settled bets: 146 lays, 124 markets, all settled; net +GBP153.08 after commission (win +170.67 on 79 lays, place -17.59 on 67); liability laid GBP7,300.
- **The early lay ran on paper after 11:23; greyhound/STOP held all day.** The scheduled run (GREYHOUND_EARLY_LAY `live`) was cancelled at 11:22:52 with no exit line; 45017050921 (`paper`): "greyhound early lay (paper) exited 0", 83 markets priced, 83 closed, locked_pnl 0.0. Running table 45074530581: "STOPPED (S3 STOP)", 0 lays. spclose and spclose3 "no lays yet (0 dogs looked at)": the SP-close paper test still has no first day. The favourite back placed none (95 races skipped; no reason logged). The quotes on paper, 0 quoted. Restart gap about 9 s at the job level.
- **The take and close stayed on paper**: 94 markets decided, 0 positions, settled net 0.
- **The day replayed** (45087314920): on the live-key books the live rule (through fill, greened) would have made -237.00 on 11,979.00 liability (-1.98%, 121 of 404 lays matched, 21 laid dogs won) against the traded fill's +257.72 (+1.17%, 223 matched); spclose -265.87 (-2.22%), spclose3 -279.35 (-2.45%), m10 -404.67 (-1.95%), short20 -18.31, short10 -52.72. Recorder's books: live rule -220.58 (-3.42%, 66 of 437), traded fill +240.92 (+1.59%, 155). The through fill is the harsh bound; the trusted fill is -0.05% at the SP (below).
- **The close lab scored 21 Nov**: 475,638 stream lines, 24,597 runner-checkpoints with an SP; 137 races, 22,650 runner-checkpoints (4,080 race-checkpoints); 4,102 resting orders, 3,410 matched. BFSP RMSE: gbm_small_plus_bw 0.2065, gbm_small_plus_huber 0.2072, gbm_small_plus 0.2077, ltp 0.2746, mid 0.2773. Win log-loss: clogit_win_plus 1.4583, clogit_win 1.4680, gbm_win_book 1.4690, gbm_book_flow_sp 1.4709, mid 1.4713, ltp 1.4725, gbm_small_plus_huber 1.4757, gbm_all_small 1.4767, oracle SP 1.4621 (clogit_win_plus better than the oracle SP on the day; the oracle was worse than the mid on 20 Nov: both unexplained, to check).
- **Champions.** BFSP: gbm_small_plus_huber holds against the mid (gain 0.03156, z 16.15, 1,374 races over 11 days); challenger gbm_small_plus_bw gain -0.00016, z -0.70. **Win: the crown returns to the mid**: gbm_all_small gain 0.00353 (0.00150 to 0.00549), z 2.94 over 1,825 races, under the bar.
- **The review.** Closed a median 19 s after the off (10% 7 s, 90% 36 s); open 30 s after on 23%, 60 s after on 1%. SP pool: backs on 71% (median GBP50), lays on 87% (median GBP10 liability). SP book over 2,044 races median 1.0040. Benter at 60 s: the SP 0.0049 (p 0.0000); gbm_all_small 0.00012 (p 0.5043); gbm_small_plus_huber 0.00038 (p 0.3084).
- **The trade, band and resting tests.** Trade (13 days): flags lay_60_0.1 (47 trades, +7.803% at the SP), lay_10_0.1 on both (61 trades), lay_60_0.06, lay_60_0.03, lay_30_0.06, lay_30_0.03; no deflated Sharpe printed: none is a pass. Band: no candidate. Resting (11 days): rest_back_60_m0 +1.049% at the SP (2,241 trades, 0.576 to 1.637, q 0.01) and rest_back_60_m3 +2.504% (160) flagged; no deflated Sharpe printed; gates read on or after 19 Dec. A fault to check: the resting table prints some means outside their own 90% interval (rest_back_60_all 0.016 [-0.42, -0.005]; rest_back_late0_all -0.36 [-0.765, -0.398]; rest_lay_60_all -0.407 [-0.882, -0.461]).
- **The stream lab, 22 Nov** (402,256 stream lines, 190 markets, 6,951 trades; 18 days pooled, 85 rules, 170 tests): no candidate; every first-family rule negative at the SP, top gap_lay_10_0.1 -0.266%. Late family (12 days): every cell negative.
- **The early-lay forward test, 22 Nov**: 94 races, 437 lays placed, 155 matched, +1.59% of liability (its own traded fill). Queue-aware replay: 193 matched, +1.89% at the SP, +1.53% to the result. **The trusted fill**: 192 matched, 13,486.23 liability, -0.05% at the SP (the second day running below zero).

| | 13 Nov | 14 Nov | 15 Nov | 16 Nov | 17 Nov | 18 Nov | 19 Nov | 20 Nov | 21 Nov | 22 Nov |
|---|---|---|---|---|---|---|---|---|---|---|
| At the SP | +0.93% | +1.33% | +1.39% | +0.63% | +1.95% | +1.46% | +1.48% | +2.2% | +1.06% | +1.89% |
| To the result | +1.78% | -1.62% | -0.58% | -2.75% | +0.79% | +1.13% | -1.10% | +4.39% | -4.52% | +1.53% |

- **The residual on the BSP, forward on paper.** Serves (44999968316, 45014282123, 45029498202, 45045913349): each 95 races, 540 runners; lay_bottom10 51 picks, back_top10 52; code_changed_since_freeze false in all four (the serve builds its own features from the card and needs no live price run). **The first completed day settled** (45022680510, 21 Nov): 149 races served, 140 wholly priced, 9 left out; lay_bottom10 81 bets, net +4.86 (+6.0% of liability); back_top10 73 bets, net +48.47. Reading: lay_bottom10 pooled with the study +1.332% (1.042 to 1.622, t 7.56, deflated Sharpe 0.999379); gate "running: 1 of 120 days". Beside it, the live win lays on 21 Nov: 79 lays (2 missed after the 21:45 stop), +GBP170.67 on GBP3,950 of liability (+4.3%).

| 21 Nov, the same picks | Bets | Return on liability |
|---|---|---|
| Paper, lay_bottom10 at the recorder's SP | 81 | +6.0% |
| Live, win lays, Betfair's settlement | 79 | +4.3% |

- **The live price runs, 22 Nov** (stack_mlsitd, refit by the Sunday train 44986445825): 13 fired, 11 succeeded, 1 cancelled (superseded), **1 failed: 44998133641** (07:21 schedule; "The hosted runner lost communication with the server": the third day running). First price update 10:04 UTC. 95 markets priced in every run, races_not_priced 0; m_sd 0.0446 to 0.0449; no "not applied" warning. Peak 15,897 MB used and 17,734 MB swap. Three lost runners in three days point at memory.
- **GreyhoundAPI**: 14 runs, no HTTP 500, no racecards/today error.

Counted days: the stream lab's first family 18 days pooled (no candidate); its late family 12 days; the trade test 13 days (flags not read against the ledger rule); the resting test 11 days; the band test no candidate; the residual's forward test 1 completed day (21 Nov).

### 23 Nov 2026 (close lab 45287018054, stream lab 45281654812, day replay 45295128770; read 24 Nov)

- **The live lay at the SP, third day: live from the scheduled start** (paper-models 45119818208, 05:46 to 22:30, no restart; GREYHOUND_SP_LAY `live`, now a repository variable). "greyhound lay at the SP (live) exited 0": 75 picks served, 132 lays (66 win, 66 place), all PENDING at Betfair, 18 skip rows (9 picks x 2 markets, each a runner removed after the serve), 0 picks undecided, 0 holds, settled net **+GBP696.19** after commission with GBP100 open (status 45295131720). **22 Nov reconciled** at the start: 72 lays, 62 markets, 60 settled, 2 unsettled (1.265831819, 1.265831820, a race's win and place markets); net +GBP145.67 (win +11.14, place +134.52). From 24 Nov the trader reconciles its last three live days at each start (PR 485), so those two get their final figure.

| Live SP lay, net after commission | Lays | Win | Place | Day |
|---|---|---|---|---|
| 21 Nov (reconciled) | 146 | +170.67 | -17.59 | +GBP153.08 |
| 22 Nov (reconciled, 2 markets unsettled) | 72 | +11.14 | +134.52 | +GBP145.67 |
| 23 Nov (at the close, GBP100 open) | 132 | | | +GBP696.19 |
| Total | 350 | | | +GBP994.94 |

- **The early lay placed nothing; greyhound/STOP held all day.** GREYHOUND_EARLY_LAY read `live` (the owner asked for paper on 21 Nov; still to change): "greyhound early lay (live) exited 0", 131 markets priced, 134 closed, locked_pnl 0.0, open_liability 0, "recovered from Betfair's orders: {}". Running table 45274109867: "STOPPED (S3 STOP)", 131 races skipped, 0 lays. spclose, spclose3, m10, short20, short10 "no lays yet (0 dogs looked at)": the SP-close paper test still has no first day. The favourite back placed none (131 skipped; no reason logged). The quotes on paper, 0 quoted (the running table labels their section "live": a label fault, nothing sent).
- **The take and close stayed on paper**: 131 markets decided, 0 positions, settled net 0.
- **The day replayed** (45295128770): on the live-key books the live rule (through fill, greened) would have made -282.22 on 17,223.00 liability (-1.64%, 173 of 552 lays matched, 31 laid dogs won) against the traded fill's +189.18 (+0.67%, 284 matched); spclose -352.74 (-2.05%), spclose3 -324.75 (-1.92%), m10 -611.67 (-2.26%), short20 -13.16, short10 -26.85. Recorder's books: live rule -217.75 (-2.58%, 85 of 573), traded fill +180.24 (+1.03%, 178). The through fill is the harsh bound; the trusted fill is -0.12% at the SP (below).
- **The close lab scored 22 Nov**: 402,256 stream lines, 17,253 runner-checkpoints with an SP; 93 races, 15,450 runner-checkpoints (2,730 race-checkpoints); 2,727 resting orders, 2,227 matched. BFSP RMSE: gbm_small_plus 0.1948, gbm_small_path 0.1949, gbm_all 0.1955, gbm_small_plus_huber 0.1959, mid 0.2594, ltp 0.2737. Win log-loss: gbm_win_all 1.3641, gbm_small_late 1.3643, gbm_small_race 1.3644, gbm_all_small 1.3648, clogit_win_plus 1.3653, mid 1.3725, ltp 1.3760, oracle SP 1.3614 (the oracle best on 22 Nov; the 20 and 21 Nov anomalies still unexplained).
- **Champions.** BFSP: gbm_small_plus_huber holds (gain 0.03135, z 15.99, 1,467 races over 12 days). **Win: the crown goes back to gbm_all_small** (gain 0.00373, 0.00176 to 0.00566, z 3.22 over 1,916 races in 16 days).
- **The review.** Closed a median 25 s after the off (10% 13 s, 90% 52 s); open 30 s after on 34%, 60 s after on 9%. SP pool: backs on 86% (median GBP58), lays on 87% (median GBP14 liability). SP book over 2,137 races median 1.0041. Benter at 60 s: the SP 0.00509 (p 0.0000); gbm_all_small 6e-05 (p 0.6201).
- **The trade, band and resting tests.** Trade (14 days): flags lay_10_0.1 on both (67 trades, +7.768% at the SP), lay_60_0.1, lay_60_0.06 (119 trades, +5.908%, q 0.03), lay_30_0.06 on both, lay_30_0.03 on Win; no deflated Sharpe printed: none is a pass. Band: no candidate. Resting (12 days): rest_back_60_m0 +1.011% at the SP (2,372 trades, 0.547 to 1.566, q 0.01), rest_back_60_m3 +2.185% (167) and rest_lay_60_m3 +2.304% (181) flagged; gates read on or after 19 Dec. Still to fix: four resting means printed outside their own 90% interval (rest_back_60_all -0.017 [-0.447, -0.053]; rest_back_late0_all -0.372 [-0.762, -0.412]; rest_lay_60_all -0.425 [-0.925, -0.515]; rest_lay_late0_all -0.44 [-0.969, -0.617]).
- **The stream lab, 23 Nov** (554,830 stream lines, 264 markets, 10,485 trades; 19 days pooled, 85 rules, 170 tests): no candidate; every rule negative at the SP, top gap_lay_60_0.1 -0.251%. Late family (13 days): every cell negative.
- **The early-lay forward test, 23 Nov**: 132 races, 573 lays placed, 178 matched, +1.03% of liability (its own traded fill). Queue-aware replay: 233 matched, +1.04% at the SP, +2.86% to the result. **The trusted fill**: 231 matched, 17,907.56 liability, -0.12% at the SP (the third day running below zero).

| | 13 Nov | 14 Nov | 15 Nov | 16 Nov | 17 Nov | 18 Nov | 19 Nov | 20 Nov | 21 Nov | 22 Nov | 23 Nov |
|---|---|---|---|---|---|---|---|---|---|---|---|
| At the SP | +0.93% | +1.33% | +1.39% | +0.63% | +1.95% | +1.46% | +1.48% | +2.2% | +1.06% | +1.89% | +1.04% |
| To the result | +1.78% | -1.62% | -0.58% | -2.75% | +0.79% | +1.13% | -1.10% | +4.39% | -4.52% | +1.53% | +2.86% |

- **The residual on the BSP, forward on paper.** Serves (45133898536, 45159573776, 45187755911, 45220865376): each 135 races, 757 runners; lay_bottom10 75 picks; code_changed_since_freeze false in all four. The settle (45175519832): 22 Nov, 95 races served, 81 wholly priced, 14 left out; lay_bottom10 40 bets, net +0.65; back_top10 43 bets, net -15.07. Reading after 2 settled days: lay_bottom10 121 forward bets, +4.552% (1.368 to 7.736, t 2.35); pooled with the study +1.332% (t 7.57, deflated Sharpe 0.9994); gate "running: 2 of 120 days".

| The same picks | Paper lay_bottom10 (recorder's SP) | Live win lays (Betfair's settlement) |
|---|---|---|
| 21 Nov | +6.0% (81 bets) | +4.3% (79; 2 missed after the 21:45 stop) |
| 22 Nov | +1.6% (40 bets) | +0.6% (36; +11.14 on GBP1,800; 2 markets unsettled; 5 picks paper only, before the 11:23 restart) |

- **The live price runs, 23 Nov**: 13 fired, 9 succeeded, 2 cancelled (superseded), **2 failed** (45227398128, 45248727632; "The hosted runner lost communication with the server": the fourth day running; logs gone): no price update after 20:13 UTC. 135 markets priced in every run, races_not_priced 0; m_sd 0.0457 to 0.046; no "not applied" warning. Peak 15,882 MB used and 17,774 MB swap. Five lost runners in four days point at memory.
- **GreyhoundAPI**: 15 runs, no HTTP 500, no racecards/today error.

Counted days: the stream lab's first family 19 days pooled (no candidate); its late family 13 days; the trade test 14 days (flags not read against the ledger rule); the resting test 12 days; the band test no candidate; the residual's forward test 2 completed days (21 and 22 Nov).

### 24 Nov 2026 (close lab 45507870552, stream lab 45502787328, day replay 45516203190; read 25 Nov)

- **The live lay at the SP, fourth day** (paper-models 45335630299, 05:46 to 22:30, no restart; GREYHOUND_SP_LAY `live`). "greyhound lay at the SP (live) exited 0": 62 picks served, 116 lays (58 win, 58 place), 8 skip rows, 0 holds, settled net **+GBP484.45**, open liability 0.0. The start's three-day reconciliation is cut from the job's log (only its tail printed); read instead from each day's reconcile.json (greyhound-sp-lay-status.yml now prints it: runs 45516379490, 45516387026, 45516399361): 21 Nov final +GBP153.08 (win +170.67, place -17.59); 22 Nov +GBP145.67 (win +11.14, place +134.52), 2 markets still unsettled (1.265831819, 1.265831820); 23 Nov +GBP696.19 (win +436.40, place +259.80), 112 of 114 markets settled, 2 unsettled (1.265881280, 1.265881281: the GBP100 open).

| Live SP lay, net | Lays | Day |
|---|---|---|
| 21 Nov (reconciled) | 146 | +GBP153.08 |
| 22 Nov (reconciled; win +11.14, place +134.52; 2 markets still unsettled) | 72 | +GBP145.67 |
| 23 Nov (reconciled; win +436.40, place +259.80; 2 markets unsettled) | 132 | +GBP696.19 |
| 24 Nov (the trader's figure at the close, nothing open; reconciled 25 Nov) | 116 | +GBP484.45 |
| Total | 466 | +GBP1,479.39 |

- **The early lay placed nothing; greyhound/STOP held all day.** GREYHOUND_EARLY_LAY still read `live` (the owner asked for paper on 21 Nov; still to change): "greyhound early lay (live) exited 0", 101 markets priced, 103 closed, locked_pnl 0.0, open_liability 0, "recovered from Betfair's orders: {}", 23 Nov reconcile "no lays". Running table 45495545213: "STOPPED (S3 STOP)", 101 races skipped, 0 lays, matched GBP0.00. spclose, spclose3, m10, short20, short10 "no lays yet (0 dogs looked at)" (each priced 101 races): the SP-close paper test still has no first day (the third day running). The favourite back placed none (101 skipped; no reason logged). The quotes on paper, 0 quoted.
- **The take and close stayed on paper** (GREYHOUND_TAKE_CLOSE `paper`): 102 markets decided, 0 positions, settled net 0.
- **The day replayed** (45516203190): on the live-key books the live rule (through fill, greened) would have made -213.02 on 11,948.00 liability (-1.78%, 120 of 461 lays matched, 17 laid dogs won) against the traded fill's +394.43 (+1.6%, 249 matched); spclose -126.30 (-1.06%), spclose3 -149.33 (-1.29%), m10 -403.87 (-1.9%), short20 -16.07, short10 -7.81. Recorder's books: live rule -149.02 (-2.0%, 76 of 487), traded fill +297.12 (+1.74%, 174). The through fill is the harsh bound; the trusted fill is +0.47% at the SP (below).
- **The close lab scored 23 Nov**: 554,830 stream lines, 25,384 runner-checkpoints with an SP; 132 races, 21,870 runner-checkpoints (3,900 race-checkpoints); 3,983 resting orders, 3,399 matched. BFSP RMSE: gbm_small_plus 0.2103, gbm_small_plus_huber 0.2104, gbm_small_plus_bw 0.2105, gbm_small_pool 0.2106, mid 0.2973, ltp 0.3097. Win log-loss: clogit_win_plus 1.4968, ridge_all 1.4986, gbm_book 1.4992, gbm_book_flow 1.4995, gbm_small_path 1.5004, mid 1.5042, ltp 1.5051, oracle SP 1.4931 (the oracle best on 23 Nov; no spec below it. The table still shows clogit_win_plus under the oracle on 21 Nov, 1.4583 against 1.4621: unexplained).
- **Champions.** BFSP: gbm_small_plus_huber holds against the mid (gain 0.03231, z 17.47, 1,599 races over 13 days); challenger gbm_small_plus_bw gain -0.00014, z -0.64. **Win: the crown goes to gbm_book_flow** (paired against the mid: gain 0.00279, 0.00130 to 0.00425, z 3.17 over 2,046 races in 17 days); gbm_all_small no longer clears the bar (z 2.97). The report's line also says "the crown returns to mid" beside "Champion: gbm_book_flow": the wording to check.
- **The review.** Closed a median 26 s after the off (10% 12 s, 90% 135 s); open 30 s after on 41%, 60 s after on 17% (the latest closes yet). SP pool: backs on 86% (median GBP38), lays on 81% (median GBP9 liability). SP book over 2,269 races median 1.0037 (10% 0.9806, 90% 1.0233). Benter at 60 s: the SP 0.00499 (p 0.0000); gbm_book_flow 2e-05 (p 0.7648); gbm_small_plus_huber 0.00019 (p 0.4380).
- **The trade, band and resting tests.** Trade (15 days): flags lay_60_0.1 (59 trades, +8.559% at the SP), lay_10_0.1 on both (73 trades, +8.071%), lay_60_0.06 (128, +5.801%, q 0.02), lay_10_0.06 (154, +3.637%, q 0.10), lay_60_0.03 on the SP, lay_30_0.06 and lay_30_0.03 on Win; no deflated Sharpe printed: none is a pass. Band: no candidate (the SP at or under each quantile: q10 0.128, q20 0.226, q80 0.772, q90 0.870, 299,990 runner-checkpoints). Resting (13 days): rest_back_60_m0 +0.944% at the SP (2,578 trades, 0.468 to 1.453, q 0.02) and rest_lay_60_m3 +2.203% (203, 0.487 to 3.941, q 0.24) flagged; rest_back_60_m3 no longer (+1.89%, -0.041 to 4.191); gates read on or after 19 Dec. Still to fix, now five resting means printed outside their own 90% interval (rest_back_60_all -0.036 [-0.452, -0.062]; rest_back_late0_all -0.394 [-0.784, -0.445]; rest_lay_60_all -0.396 [-0.87, -0.487]; rest_lay_60_m0 -0.035 [-0.634, -0.072], new; rest_lay_late0_all -0.436 [-0.953, -0.616]).
- **The stream lab, 24 Nov** (429,030 stream lines, 206 markets, 8,035 trades; 20 days pooled, 85 rules, 170 tests): no candidate; every rule negative at the SP, top gap_lay_60_0.1 -0.376%. Late family (14 days): every cell negative.
- **The early-lay forward test, 24 Nov**: 101 races, 487 lays placed, 174 matched, +1.74% of liability (its own traded fill). Queue-aware replay: 218 matched, +2.23% at the SP, +7.35% to the result. **The trusted fill**: 214 matched, 15,216.95 liability, +0.47% at the SP (back above zero after three days below).

| | 13 Nov | 14 Nov | 15 Nov | 16 Nov | 17 Nov | 18 Nov | 19 Nov | 20 Nov | 21 Nov | 22 Nov | 23 Nov | 24 Nov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| At the SP | +0.93% | +1.33% | +1.39% | +0.63% | +1.95% | +1.46% | +1.48% | +2.2% | +1.06% | +1.89% | +1.04% | +2.23% |
| To the result | +1.78% | -1.62% | -0.58% | -2.75% | +0.79% | +1.13% | -1.10% | +4.39% | -4.52% | +1.53% | +2.86% | +7.35% |

- **The residual on the BSP, forward on paper.** Serves (45350886824, 45378370468, 45407455977, 45441574642): each 104 races, 600 runners; picks back_top10 64, back_top5 30, back_top2 12, lay_bottom10 62, lay_bottom5 34, lay_bottom2 11; code_changed_since_freeze false in all four. The settle (45394709910): 23 Nov, 135 races served, 121 wholly priced, 14 left out; lay_bottom10 65 bets, net +8.73; back_top10 78 bets, net +26.78. Reading after 3 settled days: lay_bottom10 186 forward bets, +7.657% (2.607 to 12.706, t 2.49); pooled with the study +1.344% (1.054 to 1.634, t 7.63, deflated Sharpe 0.999517); gate "running: 3 of 120 days".

| The same picks | Paper lay_bottom10 (recorder's SP) | Live win lays (Betfair's settlement) |
|---|---|---|
| 21 Nov | +6.0% (81 bets) | +4.3% (79; 2 missed after the 21:45 stop) |
| 22 Nov | +1.6% (40 bets) | +0.6% (36; 2 markets unsettled; 5 picks paper only) |
| 23 Nov | +13.4% (65 bets) | +13.2% (66; +436.40 on GBP3,300; 1 market unsettled) |

- **The live price runs, 24 Nov**: 13 fired, 11 succeeded, 2 cancelled while queued (superseded: 45384062780, 45425447182), **none failed** (after four days running with a lost runner, 20 to 23 Nov). Beside them, a dispatched predict (45414593840) and card-field (45440726369). First price update 08:52 UTC, last 20:33. 104 markets priced in every run (600 runners falling to 590), races_not_priced 0; m_sd 0.0472 to 0.0474; no "not applied" warning. Stage tables: built 2,982 s (key e262e4850895c6d8) by the dispatch, cached at 17:01 (4 s) and 17:52 (3 s), built 2,407 s on a new key (322830f456086ca3) at 19:36, then built again (1,914 s) on the same key at 20:33: the 19:36 run (45459705912) ran the code before PR 500 (39190e0) and cached at the old path; the 20:33 run (45469942622, 17ad2fa) read the new path (a folder per base key), the one rebuild PR 500 expected; earlier runs print no label. Peak 15,888 MB used and 16,815 MB swap (45374583308, to 12:47); from 12:47 swap peaks 1.6 to 3.8 GB.
- **GreyhoundAPI**: 15 runs, no HTTP 500, no racecards/today error.

Counted days: the stream lab's first family 20 days pooled (no candidate); its late family 14 days; the trade test 15 days (flags not read against the ledger rule); the resting test 13 days; the band test no candidate; the residual's forward test 3 completed days (21 to 23 Nov).

### 25 Nov 2026 (close lab 45719663794, stream lab 45714970692, day replay 45726945796; read 26 Nov)

- **The live lay at the SP, fifth day** (paper-models 45555854831, 05:46 to 22:30, no restart; GREYHOUND_SP_LAY `live`). "greyhound lay at the SP (live) exited 0": 90 picks served, 160 lays (80 win, 80 place), 20 skip rows, 0 holds, settled net **+GBP625.19**, open liability **GBP100.0** (one lay still open at the close). The 24 Nov reconciliation is cut from the job's log (only its tail printed: the 22 Nov block shows, unchanged at +GBP145.67 with 1.265831819 and 1.265831820 still unsettled).

| Live SP lay, net | Lays | Day |
|---|---|---|
| 21 Nov (reconciled; win +170.67, place -17.59) | 146 | +GBP153.08 |
| 22 Nov (reconciled; win +11.14, place +134.52; 2 markets never in Betfair's settled bets after three reconciles: voided or unmatched at the SP, not confirmed) | 72 | +GBP145.67 |
| 23 Nov (reconciled; win +436.40, place +259.80; 2 markets unsettled) | 132 | +GBP696.19 |
| 24 Nov (reconciled, every market settled; win +209.88, place +274.57; greyhound-sp-lay-status.yml 45726897150) | 116 | +GBP484.45 |
| 25 Nov (the trader's figure at the close, GBP100 open; reconciled 26 Nov) | 160 | +GBP625.19 |
| Total | 626 | +GBP2,104.58 |

- **The early lay placed nothing; greyhound/STOP held all day** (the paper trader still logged "kill switch on: no new bets" at 21:29). GREYHOUND_EARLY_LAY still read `live` (the owner asked for paper on 21 Nov; still to change): "greyhound early lay (live) exited 0", 146 markets priced, 148 closed, locked_pnl 0.0, open_liability 0, "recovered from Betfair's orders: {}", 24 Nov reconcile "no lays". Running table 45708324632 (21:52 UTC): "STOPPED (S3 STOP)", 146 races skipped, 0 lays, matched GBP0.00. spclose, spclose3, m10, short20, short10 "no lays yet (0 dogs looked at)" (each priced 146 races): the SP-close paper test still has no first day (the fourth day running). The favourite back placed none (146 skipped; no reason logged). The quotes on paper, 0 quoted (147 closed).
- **The take and close stayed on paper** (GREYHOUND_TAKE_CLOSE `paper`): 146 markets decided, 0 positions, settled net 0, 0 unsettled.
- **The day replayed** (45726945796): on the live-key books the live rule (through fill, greened) would have made -211.78 on 15,905.00 liability (-1.33%, 160 of 553 lays matched, 30 laid dogs won) against the traded fill's +219.28 (+0.8%, 275 matched); spclose -243.68 (-1.53%), spclose3 -221.54 (-1.44%), m10 -516.30 (-1.82%), short20 -22.05 (-4.18%), short10 -65.35 (-3.33%). Recorder's books: live rule -245.15 (-2.55%, 98 of 594), traded fill +76.85 (+0.45%, 172). Every through-fill version below zero on both books; the trusted fill is -0.51% at the SP (below).
- **The close lab scored 24 Nov**: 429,030 stream lines, 19,847 runner-checkpoints with an SP; 102 races, 17,370 runner-checkpoints (3,000 race-checkpoints); 3,192 resting orders, 2,696 matched. BFSP RMSE: gbm_small_plus_bw 0.2077, gbm_all_small 0.2084, gbm_small_race 0.2088, gbm_small_plus 0.2089, mid 0.3035, ltp 0.3177. Win log-loss: gbm_small_plus, gbm_small_pool and gbm_small_plus_norm 1.5447, gbm_small_race 1.5449, gbm_small_plus_huber 1.5450, ridge_all 1.5455, gbm_book_flow 1.5483, mid 1.5525, ltp 1.5528, oracle SP 1.5226 (the oracle best on 22, 23 and 24 Nov: no spec under it on any day shown). The table now shows 22 to 24 Nov only, so 21 Nov (clogit_win_plus 1.4583 under the oracle's 1.4621) has left it: still unexplained, no longer visible.
- **Champions.** BFSP: gbm_small_plus_huber holds against the mid (gain 0.03320, z 18.59, 1,701 races over 14 days); challenger gbm_small_plus_bw gain -0.00010 (-0.00038 to 0.00026), z -0.48. Win: gbm_book_flow holds against the mid (gain 0.00286, z 3.39, 2,146 races over 18 days); challenger gbm_all_small gain 0.00094 (-0.00039 to 0.00232), z 1.15 ("needs 13958 races to clear the bar"). The wording is now consistent: both lines say "Champion: gbm_book_flow" and "holds the crown against mid"; the "crown returns to mid" phrase of 24 Nov is gone.
- **The review.** Closed a median 22 s after the off (10% 9 s, 90% 76 s); open at the off on 99%, 30 s after on 31%, 60 s after on 13%. SP pool: backs on 74% (median GBP45), lays on 91% (median GBP8 liability). SP book over 2,371 races median 1.0038 (10% 0.9809, 90% 1.0234). Benter at 60 s: the SP 0.00582 (p 0.0000); gbm_book_flow 2e-05 (p 0.7935); gbm_small_plus_huber 0.0001 (p 0.5697); at 10 s gbm_small_plus_bw 0.00122 (p 0.0435), the only forecast under 0.05 (one of twelve forecast rows; not a test).
- **The trade, band and resting tests.** Trade (16 days): flags lay_60_0.1 (60 trades, +8.801% at the SP), lay_10_0.1 on both (75, +7.382%), lay_60_0.06 (135, +5.87%, q 0.01), lay_30_0.06 on both (124, +3.981%, q 0.08), lay_60_0.03 on the SP (270, +2.637%, q 0.08), lay_30_0.03 on both (276, +2.622%, q 0.09); lay_10_0.06 no longer flagged (+3.385%, -0.316 to 5.638); no deflated Sharpe printed: none is a pass. Band: no candidate (the SP at or under each quantile: q10 0.127, q20 0.226, q80 0.772, q90 0.870, 319,837 runner-checkpoints). Resting (14 days): rest_back_60_m0 +0.96% at the SP (2,713 trades, 0.454 to 1.456, q 0.02), rest_back_60_m3 flagged again (+2.025%, 180 trades, 0.255 to 4.396, q 0.27) and rest_lay_60_m3 +2.61% (224, 0.823 to 4.335, q 0.06); gates read on or after 19 Dec. Still to fix, the same five resting means printed outside their own 90% interval, none new (rest_back_60_all -0.064 [-0.477, -0.092]; rest_back_late0_all -0.402 [-0.767, -0.45]; rest_lay_60_all -0.355 [-0.823, -0.438]; rest_lay_60_m0 0.04 [-0.541, 0.023]; rest_lay_late0_all -0.464 [-0.975, -0.649]); every other interval in the close and stream labs holds its mean.
- **The stream lab, 25 Nov** (569,660 stream lines, 295 markets, 11,168 trades; 21 days pooled, 85 rules, 170 tests): no candidate; every rule negative at the SP, top gap_lay_10_0.1 -0.243%. Late family (15 days): every cell negative.
- **The early-lay forward test, 25 Nov**: 140 races, 594 lays placed, 172 matched, +0.45% of liability (its own traded fill). Queue-aware replay: 230 matched, +0.67% at the SP, -1.11% to the result. **The trusted fill**: 229 matched, 18,057.03 liability, -0.51% at the SP (below zero again after +0.47% on 24 Nov; the fourth day below zero in five).

| | 13 Nov | 14 Nov | 15 Nov | 16 Nov | 17 Nov | 18 Nov | 19 Nov | 20 Nov | 21 Nov | 22 Nov | 23 Nov | 24 Nov | 25 Nov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| At the SP | +0.93% | +1.33% | +1.39% | +0.63% | +1.95% | +1.46% | +1.48% | +2.2% | +1.06% | +1.89% | +1.04% | +2.23% | +0.67% |
| To the result | +1.78% | -1.62% | -0.58% | -2.75% | +0.79% | +1.13% | -1.10% | +4.39% | -4.52% | +1.53% | +2.86% | +7.35% | -1.11% |

- **The residual on the BSP, forward on paper.** Serves (45571147556, 45598636380, 45627511089, 45658086955): each 150 races, 842 runners; picks back_top10 88, back_top5 41, back_top2 21, lay_bottom10 90, lay_bottom5 47, lay_bottom2 10; code_changed_since_freeze false in all four. The settle (45614991872): 24 Nov, 104 races served, 94 wholly priced, 10 left out; lay_bottom10 58 bets, net +4.20; back_top10 58 bets, net +4.23. Reading after 4 settled days: lay_bottom10 244 forward bets, +7.559% (3.919 to 11.198, t 3.42); pooled with the study +1.35% (1.06 to 1.64, t 7.66, deflated Sharpe 0.999578); gate "running: 4 of 120 days". Passed: none.

| The same picks | Paper lay_bottom10 (recorder's SP) | Live win lays (Betfair's settlement) |
|---|---|---|
| 21 Nov | +6.0% (81 bets) | +4.3% (79; 2 missed after the 21:45 stop) |
| 22 Nov | +1.6% (40 bets) | +0.6% (36; 2 markets unsettled; 5 picks paper only) |
| 23 Nov | +13.4% (65 bets) | +13.2% (66; +436.40 on GBP3,300; 1 market unsettled) |
| 24 Nov | +7.2% (58 bets) | +7.2% (58; +209.88 on GBP2,900) |
| 25 Nov | settles 26 Nov (90 picks served) | reconciled 26 Nov (80 win lays; +GBP625.19 for the day, win and place, at the close) |

- **The live price runs, 25 Nov**: 13 fired, 12 succeeded, 0 cancelled, **1 failed** (45664501392, created 17:24 UTC, lost after 70 minutes in the price step: "The hosted runner lost communication with the server"; log gone). First price update 08:55 UTC, last 20:14. 150 markets priced (842 runners, then 837 from 11:25) to 13:26; from 14:26 149 markets (827 falling to 823) with races_not_priced 1 (which race: no data found in the log); m_sd 0.0451 to 0.0452; no "not applied" warning. Stage tables: built 3,051 s (key 19fef4e175a37e60, 07:27), cached 3 to 5 s at 08:28, 09:28, 10:26; built 2,179 s on a new key (aacf81014c903acb, 11:25, same commit 402e731), cached at 12:31 and 13:26; partial 2,583 s (151dccf29d880320, 14:26, commit e2c2948), partial 1,639 s (a55f1f325a84f330, 15:25), cached 16:28; partial 2,032 s (3420be8d7c659e76, 18:27), cached 19:24: 7 cached, 3 partial, 2 built. Peak 15,677 MB used (45654261965, 17:08) and 3,934 MB swap (45604302710, 12:21).
- **GreyhoundAPI**: 15 runs, all succeeded, no HTTP 500, no racecards error.

Counted days: the stream lab's first family 21 days pooled (no candidate); its late family 15 days; the trade test 16 days (flags not read against the ledger rule); the resting test 14 days; the band test no candidate; the residual's forward test 4 completed days (21 to 24 Nov).
