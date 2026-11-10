# Greyhound statistics, features and modelling: the review

racing² - greyhound research, 10 Nov 2026. The owner: "Review our statistics, features and modelling approach.
Ideate and Improve this."

Every figure below is from the closing-price lab's dry runs on the five recorded days, 5-9 Nov 2026 (runs
42768177695 and 42769218285, 556 races with every runner priced), unless another source is named. Nothing here
places a bet or touches the horse model.

## The short version

1. **The labs run on the wrong clock.** The SP is struck at the real off, and races go off a median 22 seconds after
   the scheduled time. Every lab stopped reading at the scheduled time. The lab now reads on until the market closes.
2. **The champion test was too easy to pass.** It was read every night against the best of several challengers. The
   Win crown changed on 9 Nov on evidence chance produces. The bar is now three standard errors, and the Win crown
   is back with the mid.
3. **The current champion's gain is far from the off, on wide books.** Where the spread is within 5% it does not
   beat the mid, and its forecast tells the result nothing the mid does not.
4. **The SP knows more about the result than the mid does**, even 10 seconds before the scheduled off. That is the
   prize for forecasting the last move, and the current features capture about a twentieth of it.
5. **The stream held data nobody read:** the money waiting for the SP, where trades happened, and the rest of the
   race. Three feature families and ten specs are registered from today. On three practice days the fullest of them
   beat the mid in the last stretch before the real off, where the champion does not. Practice days do not count.

## 1. The stack reviewed

| Layer | What it forecasts | Data | State |
|---|---|---|---|
| Pre-race model (`greyhound/model.py`, the fast loop, `greyhound/pit`) | The result and log(BSP) from form, market-blind | GBGB 2014-26 | The SP is efficient against it (ledger row 4). Its use is as a feature for the layers below |
| First-book forecast (`research/greyhound/limit_study.py`) | The SP from the first traded book and our price | Recorder books from 6 Oct | Drives the early lay, the only rule ahead in its forward test |
| Closing-price lab (`research/greyhound/close_lab.py`) | The SP from the last five minutes of the Exchange Stream | Stream from 5 Nov | Reviewed and extended here |
| Rule labs (stream lab, trade test, forward test) | Whether a fixed rule makes money at the SP and to the result | Stream and recorder books | 86 + 36 nightly tests, none passed |

## 2. What the review found

### 2.1 The real off is later than the scheduled off

When each WIN market really closed, against its scheduled time:

| Day | Markets | Median delay | 10% | 90% | Open at the scheduled off | 30 s after | 60 s after |
|---|---|---|---|---|---|---|---|
| 5 Nov | 73 | 17 s | 6 s | 36 s | 97% | 15% | 1% |
| 6 Nov | 117 | 22 s | 9 s | 61 s | 100% | 33% | 10% |
| 7 Nov | 121 | 23 s | 11 s | 44 s | 100% | 26% | 3% |
| 8 Nov | 109 | 22 s | 7 s | 51 s | 99% | 28% | 8% |
| 9 Nov | 136 | 23 s | 10 s | 62 s | 100% | 30% | 10% |

What this means:
- The lab's "10 seconds out" is really about 33 seconds before the SP is struck, and its window ended before the
  last and busiest stretch of trading.
- The trade test and the stream lab enter at 60, 30 and 10 seconds before the scheduled off, so they are earlier
  than their names say.
- The stream replay of the early lay's orders lapses them 5 seconds before the scheduled off
  (`stream_replay.replay`, `close_secs`), so fills in the last 20 or so seconds are not counted.

**Changed:** the lab now takes a checkpoint every 10 seconds after the scheduled off for as long as the market is
open, up to 3 minutes. The headline scores stay on the original checkpoints so the registry still compares night to
night. The late stretch is its own window. The rule labs' clocks are unchanged, because their rules are registered.

### 2.2 The champion bar let chance through

The old bar was a 90% lower bound above zero, read every night, against the best of seven challengers. On 9 Nov the
Win crown passed to gbm_all_small on a lower bound of 0.00004 at the third nightly look (run 42726462862). Nightly
looks at several challengers make a pass like that likely with no real gain.

**Changed (bar version 2):**
- A crown needs a paired gain 3 bootstrap standard errors above zero, over at least 3 fresh days and 300 races.
- The challenger is the spec with the strongest paired evidence on the days both were fresh. Raw means over
  different days no longer decide it, because days differ a lot (the mid's Win log loss was 1.43 on 8 Nov and 1.57
  on 9 Nov).
- A spec holding a crown must keep clearing the bar against the best baseline, or the crown returns to it.

Under the new bar (run 42769218285):

- **BFSP: gbm_all_small keeps the crown.** Its paired gain over the mid is 0.0355 in squared error, 10.3 standard
  errors, over 366 races on 3 fresh days.
- **Win: the crown returns to the mid.** gbm_all_small's gain is 0.0046 (90% interval 0.00004 to 0.0089), only 1.67
  standard errors, over 365 races. A gain that size needs about 1,129 races to clear the bar, about a week more at 120 races a day.

### 2.3 The champion's gain is on wide books, far from the off

RMSE of log(SP), the champion gbm_all_small against the mid:

| Day | All: mid | Champion | Over 180 s out: mid | Champion | Last 60 s: mid | Champion | Spread within 5%: mid | Champion |
|---|---|---|---|---|---|---|---|---|
| 7 Nov | 0.3030 | 0.2247 | 0.4280 | 0.2945 | 0.1404 | 0.1365 | 0.1301 | 0.1285 |
| 8 Nov | 0.2381 | 0.1895 | 0.3323 | 0.2423 | 0.1230 | 0.1253 | 0.1183 | 0.1238 |
| 9 Nov | 0.2760 | 0.1837 | 0.3885 | 0.2321 | 0.1235 | 0.1218 | 0.1132 | 0.1139 |

- The headline gain is real, but it comes from repairing a poor mid when the book is wide. With a wide spread the
  lab's mid falls back to one side of the book (`close_lab._mid`).
- Where the book is tight, which is where a trade can be done, the champion beat the mid on one day of three.
- The headline RMSE is therefore not the number to optimise. The last-minute, tight-book and late readings are now
  printed beside it every night.

### 2.4 The Win target: what a perfect SP forecast is worth, and what ours is worth

Benter's second stage asks whether a forecast tells the result anything the mid does not. One checkpoint a race:

| Forecast beside the mid | Seconds before the scheduled off | Races | Weight on the forecast's gap | Gain in log loss a race | p |
|---|---|---|---|---|---|
| The SP itself (the ceiling) | 60 | 548 | 1.20 | 0.0085 | 0.002 |
| The SP itself (the ceiling) | 10 | 548 | 1.60 | 0.0105 | 0.0007 |
| The SP itself, 30 s after | -30 | 150 | 3.12 | 0.0271 | 0.004 |
| gbm_all_small | 60 | 365 | 1.01 | 0.00045 | 0.57 |
| gbm_all_small | 10 | 365 | -1.04 | 0.00042 | 0.58 |

- **The last move carries information about the result.** The SP beats the mid on the winner even 10 seconds before
  the scheduled off. Late money is informed.
- **Our forecast captures about 5% of that** (0.00045 of 0.0085), which is not distinguishable from nothing.
- The reason is simple. In the last minute the champion explains between -4% and +6% of the move from the mid to
  the SP (from the table in 2.3). A forecast that cannot call the move cannot pick winners from it.
- The win-trained specs are worse than the mid on both targets on every day. A few hundred races cannot teach a
  boosted model the result.

**Changed:** the second stage and the SP ceiling are printed every night. The report also says how many races a
gain of the observed size needs to clear the bar. A small conditional logit for the result is registered
(`clogit_win`, `clogit_win_plus`). On the three practice days the first was worse than the mid on Win every day
and the second beat it on one day of three.

### 2.5 Data in the stream that no feature read

| Family | What it reads | Why it should matter |
|---|---|---|
| `sppool` | The money waiting for the SP: SP backers' stakes as the liability they need at the mid, SP layers' liability, the gap between them, and that gap against the money near the top of the book | The SP is where those two pools and the book balance. This is the mechanism itself |
| `path` | The move over 2 minutes, how much the price has been moving, how much of the volume came in the last minute, and where trades happened against the mid (all trades, and the last minute's) | Signed flow: trades above the mid are backers paying up |
| `race` | The runner's move less the race's, its share of the race's volume, the favourite's move, the back and lay book percentages, the trap | A race's prices move together: one dog shortening pushes the others out |

The SP pool is in the recorded stream (the recorder subscribed to it from the first day) but the replay never
parsed it. On 9 Nov SP backs were present on 75% of runner-checkpoints (median GBP50 staked) and SP lays on 98%
(median GBP10 liability).

One caution: I read the stream's `spb` sizes as back stakes and `spl` sizes as lay liabilities, from Betfair's
documentation as I remember it. I could not check that against a settled market from here. If the reading is wrong
the features are weaker, not leaky, and the forward score will show it.

### 2.6 The model's form

| Weakness | Option registered to test it |
|---|---|
| One model for every horizon, from 5 minutes out to after the scheduled off | `by_window`: one fit for each stretch |
| Squared error on a heavy-tailed target (a long shot's SP can be several times its mid) | `loss: huber` |
| Each runner forecast alone, though a race's SPs sum to about 1 (median SP book 1.0046, 10% 0.984, 90% 1.022) | `norm`: the race's forecast book rescaled |
| Thirty correlated checkpoints a runner treated as thirty independent rows | Not built: see the backlog |

### 2.7 Many tests, read one at a time

The stream lab reads 86 tests a night and the trade test 36, each against a 90% bound. About one in twenty passes
by chance, and the ledger counts about 31,000 rules tried in all.

**Changed:** both labs now print a q-value for every test: the share of false discoveries expected among all the
tests at least as strong (Benjamini and Hochberg). A z-score needs 100 races before it is printed. The registered
candidate rule is unchanged; the q-value is read beside it.

### 2.8 The early lay's forward test is stronger on the SP target than its gate has yet said

The replayed lays, as a share of matched liability (the nightly record in `reports/greyhound_edge_search.md`):

| | 5 Nov | 6 Nov | 7 Nov | 8 Nov | 9 Nov | Mean | Day-to-day spread | t (4 d.f.) |
|---|---|---|---|---|---|---|---|---|
| At the SP | +1.78% | +1.43% | +0.71% | +1.42% | +1.85% | +1.44% | 0.45% | 7.1 |
| To the result | -1.65% | +1.04% | -1.50% | +4.28% | +2.62% | +0.96% | 2.58% | 0.8 |

- Closed at the SP, five days of five are positive and the day-level t is 7.1. That is the strongest evidence in
  the ledger.
- Held to the result, the same trades are nearly six times noisier a day. The result figure is the SP figure plus
  the luck of the race, so the SP figure is the one to judge the rule on, with the result as a check that the SP is
  not biased in the prices traded.
- Three cautions stay. The fills are a replay, not real orders. The stake ladder found the return falls to about
  +0.79% at GBP20 with capped fills and is negative at GBP100 and above (run 42462215823). And five days is five
  days: the gate asks for 20 days and 1,000 trades.

This does not change the gate or the go-live decision, which is the owner's.

## 3. First readings of the new specs (practice only)

The ten specs were registered today, so nothing before 11 Nov counts. These are the days they were designed
without seeing, but they are still practice. Share of the mid's squared error removed (positive is better):

| | Last 60 s: 7 Nov | 8 Nov | 9 Nov | After the scheduled off: 7 Nov | 8 Nov | 9 Nov |
|---|---|---|---|---|---|---|
| gbm_all_small (the champion) | +5.5% | -3.8% | +2.7% | -0.5% | -6.1% | -14.4% |
| gbm_small_pool (adds the SP pool) | +7.0% | -1.1% | +5.0% | +7.2% | +6.1% | -8.5% |
| gbm_small_plus (adds all three families) | +7.0% | -1.5% | +7.9% | +6.4% | +5.5% | +5.7% |

Computed from the RMSE columns of run 42768177695 as 1 - (spec / mid) squared.

- After the scheduled off, the fullest spec beat the mid on all three days by about 6% of the squared error. The
  champion, which never saw those checkpoints in training, was worse than the mid on all three.
- Normalising the race's book made the BFSP score worse on all three days (for example 0.1871 against 0.1833 on
  9 Nov). The by-window and Huber options were mixed. All stay registered and will be scored as they are.
- On the headline RMSE the new specs are level with the champion (0.1833 against 0.1837 on 9 Nov). The gain, if it
  holds, is where trades happen.

## 4. The ideas, in the order I would do them

The first three were built and registered the same day: see section 5.

1. **Trade on the forecast's uncertainty, not a fixed margin.** The forecast's error is about 0.12 in the last
   minute and 0.23 beyond three minutes, so a 3% margin is a quarter of a standard error. Fit the 20% and 80%
   quantiles of the SP and trade only when the price on offer is outside them. Register it once the new specs have
   a week of fresh days, as a new family with its tests counted.
2. **Re-time the rule labs to the real off.** Enter at the late checkpoints, and let the replay rest orders until
   the market closes. This needs new registered rules; the old ones stay as they are.
3. **Rest orders instead of taking prices.** Every taker rule pays the spread. Use the queue-aware replay to test
   resting at the forecast SP in the last stretch, where the forecast now has some skill.
4. **Rebuild the SP from its parts.** With the SP pool and the full ladder at each checkpoint, compute the price at
   which they balance. Betfair's own projection is the worst of the three baselines in every window, so a better
   one is possible.
5. **Feed the closing forecast back into the early lay** as a paper arm: cancel or move a resting lay when the
   closing forecast turns against it.
6. **Forecast the time to the real off** from the stream (volume speeding up, the spread closing). Every
   late-window feature gets sharper with it.
7. **Judge every rule on its SP figure, with the result as a bias check** (the reasoning in 2.8), and write that
   into the gates of new rules.
8. **Learn the SP-to-result curve from the 2026 historic files** (70,125 paired races, ledger row 6), so the few
   stream days are spent on the microstructure, not on the favourite-longshot curve.
9. **Bag and weight by race, not by row,** and weight recent days more as the record grows.
10. **Track-level effects with partial pooling** once 30 days are held: the SP's gap to the mid by track and trap.
11. **Deflated Sharpe and the probability of overfitting across the 31,000 rules,** as one meta-reading of whether
    any family holds an edge at all.
12. **Score capacity in pounds, not percent:** edge times the size that would have matched.

## 5. The first three ideas, built and registered (10 Nov)

The owner, 10 Nov: "work on these and fix. then merge." All three were fixed in code before any of their output was
read (commit b128b03), count from 11 Nov, need 10 counted days, and place no bets.

| Test | What it does | Size |
|---|---|---|
| The late family (stream lab) | The lab's projected-SP, momentum and reversal rules decided 0, 10 and 20 s after the scheduled off, while the market is open | 42 rules, 84 tests |
| The band test (closing-price lab) | Fits the 10%, 20%, 80% and 90% quantiles of the SP. Backs at the best back price when it is above the upper quantile, lays at the best lay price when it is below the lower one. 60 and 10 s before the scheduled off and 0, 10, 20 s after | 20 rules, 40 tests |
| The resting test (closing-price lab) | Leaves a lay at the best back price, or a back at the best lay price, 60 s before the scheduled off and at it. The order rests until the market really closes. Fills are replayed through the recorded queue. Three variants: every runner (no forecast), and only where the forecast agrees by 0 or 3% | 12 rules, 24 tests |
| The early lay to the real off | The forward test's lays left in the book until the market closes, not lapsed 5 s before the scheduled off | A second reading; the gate's figure is unchanged |

Both closing-price tests read gbm_small_plus's forecast, fitted on earlier days only.

### What re-timing did to the early lay (9 Nov, stream lab run 42787435666)

| | Lays matched | Matched liability | At the SP | To the result |
|---|---|---|---|---|
| Lapsed 5 s before the scheduled off (as before) | 296 | GBP5,812 | +1.85% | +2.62% |
| Resting until the market closes | 306 | GBP6,019 | +1.78% | +1.58% |

The clock barely matters for this rule: ten more lays matched and the SP return is level. Its orders go in at the
first traded book, so nearly all their fills come long before the off.

### Practice readings, 7 to 9 Nov (never counted; run 42786681970)

**Resting orders, closed at the SP, a unit of matched stake:**

| | Every runner (no forecast) | Forecast agrees | Forecast agrees by 3% |
|---|---|---|---|
| Back, 60 s before the scheduled off | +0.87% (0.43 to 1.09; 1,585 matched) | +1.94% (1.30 to 2.70; 843) | +3.64% (2.40 to 5.44; 251) |
| Lay, 60 s before | -0.26% (1,604) | +0.34% (-0.38 to 0.71; 729) | -0.53% (-2.75 to 1.11; 124) |
| Back, at the scheduled off | +0.14% (1,371) | +0.36% (813) | +0.20% (161) |
| Lay, at the scheduled off | -0.05% (1,418) | +0.10% (606) | -1.82% (53) |

- The backs resting 60 s out are the one clear pattern: the return grows with the forecast's margin, 1.1 and 2.8
  points above the no-forecast control.
- Nothing shows at the scheduled off, and nothing shows to the result (every interval there is wide).
- 75% to 90% of the orders matched at least in part.

**The band test:** lays 60 s out made +7.1% (3.7 to 10.6) on 26 trades at the 90% level and +3.9% (0.8 to 6.7) on 50
at the 80% level. Every other rule is inside its noise, with 5 to 50 trades each. The bands are a little too narrow:
the SP fell at or below the four quantiles 13.5%, 22.4%, 77.2% and 86.8% of the time, against 10, 20, 80 and 90.

**How far to trust this:**
- These are three practice days. By the rules fixed beforehand they count for nothing.
- The fills are replayed, not real. A high share of orders matching is exactly where a fill model flatters, and the
  ledger's row 8b is a rule whose whole edge was its fill assumption.
- The two no-forecast controls are not symmetric (backs +0.87%, lays -0.26%). A back closed at the SP gains more from
  a shortening than it loses from an equal drift, so read each forecast variant against its own control.
- Stakes are small (a lay risks GBP20), and what size would match is not tested here.

## 6. Checks

| Check | Result |
|---|---|
| Old specs reproduce exactly on the rebuilt frames | Pass: gbm_all_small on 9 Nov scores 0.1837 (last 60 s 0.1218, Win 1.5625) in runs 42726462862 (before) and 42768177695 (after); 7 and 8 Nov also match |
| Rows in against rows out | Pass: 22,980 runner-checkpoints and 136 races on 9 Nov in both runs; the late checkpoints are extra rows in their own window |
| Unit tests | Pass: 46 across the close lab, stream lab, stream replay, capacity and workflow guards |
| The three new tests run on real data | Pass: close-lab run 42786681970 (dry run, practice on 7-9 Nov) and stream-lab run 42787435666 (9 Nov) |
| Leakage: every new column read at or before its checkpoint | Pass by construction: books, traded ladders and SP pools as they stood; a late checkpoint exists only while the market was open then. The closing time and the SP itself are in the frame but in no feature family. The SP "ceiling" is look-ahead by design and is never a spec |
| Nothing written by the dry runs | Pass: `--dry-run` skips the registry and the champion file. The rebuilt frames are cached under a new prefix (`close_lab/frames_v2/`); the old cache is untouched |

Not verified: the meaning of the stream's `spb` and `spl` sizes (2.5), and anything about real fills.
