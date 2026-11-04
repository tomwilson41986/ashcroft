# racing² football on Betfair: model performance and every strategy type, scored on Betfair's prices

3 Nov 2026.

**Scope.** Betfair only (the owner: "ignore pinnacle, we only want to work on betfair"). The prices are football-data's
Betfair Exchange early (BFE) and closing (BFEC) prices, 2024/25 to 3 Nov 2026, with 2% commission on each winning bet.

**Sources.** Figures come from `reports/football_runs/market_scan.json` (`research/open_close/football_market_scan.py`
over the lab matrix) unless another source is named.

**How returns are counted.**
- Returns are per unit staked; for a lay, per unit of the backer's stake laid.
- Intervals are 90%, bootstrapped by match day.
- **"2024"** is 2024/25. **"2025+"** is 2025/26 to date; earlier pre-registered rules read those seasons, so they are
  development evidence. **"All"** is both.

## Bottom line

- **No strategy type makes money on Betfair on the evidence held.** That covers back to win, lay outright, back to
  lay, lay to back, over/under 2.5 and the Asian handicap, blind or model-led.
- **The model's value bets lose more than backing the same selections blind.**
- **The nearest to break-even:**
  - backing Over 2.5 at the close: -0.9% [-2.2, +0.4];
  - laying away sides at the close: -1.2% [-3.0, +0.7].
- **One pattern stands out: laying away sides at 10.0+ at the close.**
  - 1,361 matches.
  - About +1% of the money at risk.
  - It holds only if the real lay spread is at most about 5-10% of the price.
  - It was found by looking, so it is a candidate for a pre-registered forward test on the recorder's real lay prices,
    not an edge.

## 1. How the model is performing

| What | Result | Source |
|---|---|---|
| The model's value bets at Betfair's close (5% edge, all) | Home -6.6% (n 9,375), Draw -3.0%, Away -9.5%. Blind: -2.3%, -2.2%, -4.2% | `market_scan.json` |
| The model's value bets at Betfair's early price (5% edge) | Home -12.3%, Draw -5.1%, Away -11.9% | `market_scan.json` |
| The pre-registered move rule on Betfair's prices, holdout (Jul 2025-Nov 2026, 1,926 trades) | **Failed**: -1.11% a tick off fair (-1.67 to -0.61). +3.29% at the mid; -0.17% at a symmetric book | `reports/football_runs/team_move_holdout.json` |
| The stacked move rule | Pre-registered 3 Nov; Betfair 2024/25 development only: +1.58% a tick off fair, 416 bets. Never scored forward | `reports/preregistration_football_stack.md` |
| The paper trader (pooled model, 10 countries, Betfair) | 17 paper bets, 0 settled as of 3 Nov 05:42 UTC. Settlement was fixed in PR #184, merged today | football-model.yml run 41585231684 |
| Real money | None | - |

## 2. Hold to the result: back to win, lay outright

Every selection, blind, at Betfair's close (all).

**Match odds**, n 27,027:

| Selection | Back | Lay |
|---|---|---|
| Home | -2.3% [-3.7, -1.0] | -2.3% [-3.7, -1.0] |
| Draw | -2.2% [-3.8, -0.5] | -3.5% [-5.2, -1.8] |
| Away | -4.2% [-6.0, -2.5] | -1.2% [-3.0, +0.7] |

**Over/under 2.5**, n 17,085:

| Selection | Back | Lay |
|---|---|---|
| Over | -0.9% [-2.2, +0.4] | -3.4% [-4.8, -2.0] |
| Under | -3.0% [-4.4, -1.5] | -1.4% [-2.9, +0.0] |

**Asian handicap**, close, n 9,349 (2025+ only):

| Selection | Back | Lay |
|---|---|---|
| Home | -1.7% [-3.1, -0.0] | -2.1% [-3.7, -0.5] |
| Away | -1.6% [-3.3, -0.2] | -2.1% [-3.6, -0.4] |

**At Betfair's early price everything is worse:** in match odds and over/under, backs -5% to -9% and lays -6% to -12%; in the handicap, -1.6% to -4.9%. The early books are wide, so a
lay costed at the symmetric book there is expensive.

**The model's value bets** lose at every threshold in every market. The best cells sit near zero on small samples:
the Draw at 10% at the close is -0.7% [-6.4, +4.9], n 3,580; the Under lay at 10% is +0.0% [-3.8, +3.6], n 1,698.

**How the lay is costed.** Betfair's lay price is not in football-data. A lay is costed at a book symmetric about the
fair price (lay = the next tick at or above fair²/back). The recorder will replace this with real lay prices.

## 3. Back to lay and lay to back (early price to close, greened)

Each trade backs at the quoted back price and lays at the symmetric book's lay price at the other end. 2% commission
is taken on a greened profit.

| Market (all) | Back to lay | Lay to back | The model's direction, 10% gap | The close moves the model's way |
|---|---|---|---|---|
| Match odds, home | -5.7% [-6.1, -5.4] | -12.3% | Back to lay -4.7% | 51% |
| Match odds, draw | -6.5% | -9.3% | Back to lay -2.7% [-3.1, -2.3] | 55% |
| Match odds, away | -7.6% | -8.7% | Back to lay -5.6% | 49% |
| Over 2.5 | -6.3% | -8.0% | Back to lay -2.9% | **60%** |
| Under 2.5 | -6.4% | -8.0% | Back to lay -3.5% | 56% |

What this shows:
- **Crossing the spread twice costs 6-8%.**
- **The model calls the move on the goals market 56-60% of the time**, which is real information, but it doesn't
  cover the cost.
- **Lay to back is worse than back to lay** because the early book is wide on the lay side.
- **The team-feature move models do better** (64-66% the right way in the lab), but the registered rule still failed
  once costed.
- **What would change this:** a resting order earns the spread instead of paying it (the feed measures this), or the
  real early spread proving tighter than the symmetric book.

## 4. The one pattern worth testing: laying away longshots at the close

Away sides at 10.0+ at Betfair's close. Wins against the price's implied chance, and the return against the lay price
actually paid (as a multiple of the closing back price):

| Period | Bets | Wins | Implied | Lay at x1.00 | x1.05 | x1.10 | x1.20 | x1.30 |
|---|---|---|---|---|---|---|---|---|
| 2024 | 602 | 4.65% | 7.00% | +25.1% [-3.1, +52.8] | +21.4% | +17.8% [-13.2, +48.3] | +10.5% | +3.2% |
| 2025+ | 759 | 5.93% | 7.38% | +23.3% [+6.0, +39.3] | +19.6% [+1.4, +36.4] | +15.9% [-3.2, +33.4] | +8.4% | +0.9% |
| All | 1,361 | 5.36% | 7.21% | +24.1% [+7.7, +38.8] | +20.4% [+3.2, +35.9] | +16.7% [-1.3, +32.9] | +9.3% | +1.9% |

What this means in practice:

- **Return on the money at risk.** The liability is about 16.5 times the backer's stake. So +16.7% a stake (at a 10%
  lay spread) is about **+1.0% of the liability**, with long losing runs.
- **The lay spread decides it.** The lower bound clears zero only at lay prices within 5% of the back price. Spreads
  at 10.0+ are often wider.
- **Home and draw longshots show nothing:**
  - home at 10.0+: +1.2%, n 263;
  - draw at 10.0+: +5.7%, n 185;
  - both intervals are very wide.
- **Caveats.** It was found by scanning many cells; it is a known betting-market effect, which is the reason to test
  it, not to trust it.

## 5. What cannot be scored yet

| Strategy | Why not | What lets us |
|---|---|---|
| In-play trading | No in-play prices; the feed stops at the off | Extend `football/exchange_feed.py` to keep reading in play |
| Correct score, both teams to score, corners, cards, alternative totals | No prices held | Add the market types to the feed |
| Resting orders | No books between the early price and the close | The feed's resting-order lines, from its first day |
| Team-news window | No prices around the team sheets yet | The feed's 1-minute books and the line-ups collector |
| The real lay spread on any of the above | football-data carries Betfair back prices only | The feed records both sides with depth |

## 6. What to do before the runner starts

1. **Start the football runner on paper** (`FOOTBALL_LIVE` unset). It records Betfair's real back and lay prices,
   depth and traded money for every match: the spread that decides sections 2-4.
2. **Do not set `FOOTBALL_LIVE=yes` on the current model.** Section 1 shows no edge; the GBP2 stakes would only keep
   the live key in use.
3. **Pre-register the away-longshot lay before the first recorded book:**
   - the last pre-play book;
   - away 10.0+, only where the recorded best lay is within 5% of the best back;
   - a fixed liability per bet;
   - gate on the 90% lower bound of the return on liability.

## 7. Improvements tried (4 Nov 2026)

**Sources.** `research/open_close/football_improve.py`; figures from `reports/football_runs/improve.json`.

**Setup.**
- The stacked move model is fitted walk-forward on the moves of the seasons before each test season.
- It is traded on Betfair's own prices, in takeable books, greened, with 2% commission.
- **Selection is tuned on 2024/25 and read on 2025/26 to date.** That season was read by earlier rules, so the reads
  are development evidence.
- About 30 variants were read on 2025/26. Read every result against that count.

### Picking trades: weekly top-K instead of a fixed threshold (match odds)

| Rule | 2024/25 (tuned) | 2025/26 (read) |
|---|---|---|
| Threshold 0.04 (registered) | 416 bets: +1.58% tick, +3.82% sym | 531 bets: -0.53% tick [-1.8, +0.7], +1.59% sym [+0.4, +2.7] |
| Threshold 0.05 | 190 bets: +4.17% tick, +7.01% sym | 235 bets: +0.36% tick [-1.7, +2.5], +3.07% sym [+1.1, +5.2] |
| **Top 5 a week** | 225 bets: +4.35% tick, +6.88% sym | **285 bets: +1.20% tick [-0.3, +2.8], +3.64% sym [+2.1, +5.2]; 74% right way** |
| Top 10 a week | 450 bets: +2.21% tick, +4.13% sym | 570 bets: -0.88% tick, +1.20% sym |

("tick" = a tick off fair at each end; "sym" = a book symmetric about fair.)

- **Weekly top 5 is the best read on 2025/26.** It is the only rule within reach of break-even a tick off fair, and
  its bet count holds steady from season to season (about 6 a week).
- The gain over threshold 0.05 is within the noise. Its advantage is the steady volume.

### Daily volume: top K a match day (the owner, 4 Nov: "anything daily?")

**Rule A: forecast move.** The K biggest forecast price moves each match day, traded in the forecast's direction.

| Match odds, 2025/26 (316 match days) | Bets (a day) | Tick | Sym | Right way |
|---|---|---|---|---|
| Top 5 a day | 1,466 (4.6) | -1.92% [-2.5, -1.3] | -0.45% [-1.0, +0.1] | 61% |
| Top 10 a day | 2,718 (8.6) | -2.13% | -0.99% | 60% |
| Top 20 a day | 4,660 (14.7) | -2.20% | -1.27% | 59% |
| Top 50 a day | 8,928 (28.3) | -2.48% | -1.75% | 57% |
| *For scale: top 5 a week* | *285 (0.9)* | *+1.20%* | *+3.64%* | *74%* |

**Rule B: model-vs-market gap.** The K biggest gaps each day between the model's price and Betfair's early price,
traded toward the model.

| Top K a day, 2025/26 | Tick | Sym | Right way |
|---|---|---|---|
| 5 to 50 | -4.7% to -6.3% | -3.8% to -3.9% | 47% |

Rule B is worse than a coin flip: the market moves away from our model more often than toward it.

**Over/under, top 5 to 50 a day, forecast move:** -0.9% to -1.5% a tick off fair, -2.3% to -2.7% at a symmetric book.

2024/25 shows the same pattern. Top 5 a day is the only daily rule above zero there: +0.91% at a symmetric book,
-0.66% a tick off fair.

**Read: the edge sits in the strongest 1-2% of signals.** Below that, the forecast is right 57-61% of the time, too
little to beat the spread. More volume from this model means more losing trades.

**Where daily volume can come from instead:**
- more markets: every league on Betfair, which needs features that don't depend on football-data's 22 divisions;
- more decision moments in each match (re-entry as the book moves);
- order-book strategies on the feed.

All three need the football runner's live record.

### A cost filter (meta-label, match odds)

A small LightGBM model, fitted on 2024/25's candidate trades, predicts whether a trade clears its cost. Its inputs are
the forecast's size and direction, the price, the book's margin, the league, and home or away.

| On 2025/26 | Bets | Tick | Sym |
|---|---|---|---|
| Threshold 0.04 alone | 531 | -0.53% | +1.59% |
| Threshold 0.04 and filter > 0.5 | 300 | +0.26% [-1.7, +2.2] | +2.93% [+1.2, +4.7] |
| The filter's top 531 (same count as the threshold) | 531 | -0.65% | +1.73% |

**Read:** the filter adds about 1 point when combined with the threshold. At equal bet counts it adds little. Its
main inputs are league, book margin, forecast size and price. It's worth keeping, but it hasn't been shown to matter.

### An over/under 2.5 move model

The same stack, plus a goals block: the Dixon-Coles over chance against the early price, and the market's and the
model's goal rates.

| On 2025/26 | Bets | Tick | Sym |
|---|---|---|---|
| Top 5 a week, with the goals block | 285 | -0.69% | -2.76% |
| Top 5 a week, without it | 285 | -1.75% | -3.59% |
| Threshold 0.03, with the block | 384 | -0.34% | -2.26% |

**Read:** the goals block helps, by about a point. The over/under move still doesn't pay at Betfair's spreads.

### Built for the feed: order-book features (`football/feed_features.py`)

From the feed's record, at any decision moment, using reads at or before it only (tested):

| Group | Features |
|---|---|
| The book now | Best back and lay, spread in ticks, depth and imbalance on each side, the microprice, the book's overround |
| The flow | Money traded in the last 15 and 60 minutes, and the share of it at or above the mid |
| The path | The mid's change over 15, 60 and 240 minutes, its 60-minute volatility, minutes to the off |
| The label (fitting only) | The move to the last pre-play read, and the back-now, lay-at-close return |

This feeds the next stage:
- order-book move models;
- entry timing (enter when the book is tight);
- resting-order fills;
- high-volume strategies such as market making and short-horizon mean reversion.

None of these can be backtested from football-data, which holds two snapshots and no depth.

### What this leaves

- **Best rule so far on Betfair:** the stacked match-odds move model, its strongest 5 forecasts a week, back first
  where the chance is forecast up (lay first where it is forecast down), greened at the close. Optionally with the
  cost filter.
- **On 2025/26:** about break-even a tick off fair (+1.2%) and +3.6% at a symmetric book.
- **It is not registered.** The registered stack rule (threshold 0.04) is fixed. Weekly top 5 would need its own
  pre-registration, before the feed's first books, to be judged forward.
- **The deciding number is still the real early spread,** which only the feed records.
