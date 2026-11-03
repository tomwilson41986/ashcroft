# racing² football: model performance and every strategy type, scored on real prices

3 Nov 2026. Figures from `reports/football_runs/market_scan.json` (`research/open_close/football_market_scan.py`, the lab
matrix built from football-data.co.uk's gold tables) unless another source is named. Returns are per unit staked; for
a lay, per unit of the backer's stake laid. Intervals are 90%, bootstrapped by match day.

## Bottom line

- **No strategy type makes money on the evidence we hold.** That covers back to win, lay outright, back to lay, lay to
  back, over/under 2.5 and the Asian handicap, blind or model-led. Every cell with a real sample loses, or has an
  interval that spans zero.
- **The model is worse than the market.** Its value bets lose more than backing the same selections blind.
- **One pattern holds in two separate eras:** laying away sides priced 10.0 or longer, the favourite-longshot bias.
  - The return is about 1% of the money at risk.
  - It survives only if the real lay spread is at most about 10% of the price.
  - It was found by looking, so it needs a pre-registered forward test on recorded lay prices before any money goes on
    it.

## 1. How the model is performing

| What | Result | Source |
|---|---|---|
| Dixon-Coles team model against the open | Behind the open in every season, division and market; its pooled 1X2 weight is negative; value backing loses 4-7% | `reports/football_model.md` |
| The model's value bets, this scan (Pinnacle close, 2012/13-2023/24) | Home -5.5% (n 60,849), Draw -5.4%, Away -7.4%. Backing every selection blind loses less: -2.9%, -3.5%, -5.1% | `market_scan.json` |
| Team features | Nothing added to the result's price; they forecast the early-to-close move (R2 4.5%) | `reports/football_team_features.md` |
| The pre-registered move rule, holdout (Jul 2025-Nov 2026, 1,926 trades) | **Failed**: -1.11% a tick off fair (-1.67 to -0.61). +3.29% at the mid; -0.17% at a symmetric book | `reports/football_runs/team_move_holdout.json` |
| The stacked move rule | Pre-registered 3 Nov; development only (2024/25: +1.58% a tick off fair, 416 bets); never scored forward | `reports/preregistration_football_stack.md` |
| The paper trader (pooled model, 10 countries) | 17 paper bets, 0 settled as of 3 Nov 05:42 UTC. Settlement was fixed in PR #184, merged today | football-model.yml run 41585231684, settle step |
| Real money | None | - |

**Read:** the result's price in football is efficient against everything we have built. The only forecastable thing
is the price's path to the close, and the cost of trading it eats it.

## 2. Hold to the result: back to win, lay outright

Every selection, blind (`back_all`, `lay_all`). Pinnacle has no commission. Betfair takes 2% of a winning bet.

**Match odds:**

| Selection | Back, Pinnacle close 2012-23 | Back, Betfair close 2025+ | Lay, Pinnacle close 2012-23 | Lay, Betfair close 2025+ |
|---|---|---|---|---|
| Home | -2.9% [-3.5, -2.3] n 142,116 | -3.2% [-4.8, -1.5] n 16,356 | -2.9% | -1.4% [-3.2, +0.3] |
| Draw | -3.5% [-4.2, -2.8] | -2.8% [-4.9, -0.7] | -2.3% | -2.9% [-5.1, -0.7] |
| Away | -5.1% [-5.9, -4.3] | -3.4% [-5.7, -1.3] | -0.6% [-1.4, +0.2] | -2.0% [-4.3, +0.3] |

**Over/under 2.5:**

| Selection | Back, Pinnacle close 2012-23 | Back, Betfair close 2025+ | Lay, Pinnacle close 2012-23 | Lay, Betfair close 2025+ |
|---|---|---|---|---|
| Over | -3.3% [-4.2, -2.4] | -0.1% [-2.0, +1.7] | -3.7% | -4.2% |
| Under | -4.1% [-5.0, -3.3] | -3.8% [-5.8, -1.8] | -2.8% | -0.6% [-2.8, +1.4] |

**Asian handicap (main line):**

| Selection | Back, Pinnacle close 2012-23 | Back, Betfair close 2025+ | Lay, Pinnacle close 2012-23 | Lay, Betfair close 2025+ |
|---|---|---|---|---|
| Home | -3.2% | -1.7% [-3.1, -0.0] | -1.4% | -2.1% |
| Away | -1.4% [-2.1, -0.6] | -1.6% [-3.3, -0.2] | -3.4% | -2.1% |

**The model's value bets** (expected return above 0, 5% or 10% at the price) lose in all three markets, at the open
and at the close, on both venues. They lose more than backing everything blind. The best cells sit near zero with
small samples, for example Over 2.5 at Betfair's close 2025+ at 5% (+1.2% [-3.0, +5.3], n 1,762). Out of about 200
cells, a few touch zero by chance alone.

**How the lay is costed.** Betfair's lay price is not in the data, so a lay is costed at a book symmetric about the fair
price: lay = fair²/back, on the next Betfair tick. Pinnacle cannot be laid; there the same costing stands for an
exchange as tight as Pinnacle's margin.

## 3. Back to lay and lay to back (traded open to close, greened)

Each trade enters at the open and greens up at the close: back at the quoted back price and lay at the symmetric
book's lay price, or the reverse.

| Market, venue | Back to lay, all | Lay to back, all | The model's direction (gap 10%) |
|---|---|---|---|
| Match odds, Pinnacle 2012-23 | -5.6% to -5.8% | -6.1% to -6.3% | Back to lay -4.7% to -5.3%. The close moves toward the model 50% of the time: no information |
| Match odds, Betfair 2025+ | -4.3% to -5.6% | -4.1% to -5.1% | -3.4% to -6.2% |
| Over/under, Pinnacle 2012-23 | -6.7% to -7.4% | -7.0% to -7.8% | -5.8% to -6.7% |
| Over/under, Betfair 2025+ | -4.5% to -5.2% | -4.9% to -5.6% | Back the Over -2.3% (n 538); the close moves toward the model 58% of the time on the Over |

Crossing the spread twice costs about 5%. The Dixon-Coles model barely predicts the move. The team-feature models
predict it better (64-66% the right way), but the registered rule still failed once costed (section 1).

## 4. The one consistent pattern: the favourite-longshot bias

Laying away sides priced 10.0 or longer at the close. Win rate against the price's implied chance:

| Era | Bets | Wins | Implied | Lay at back x1.00 | x1.05 | x1.10 | x1.20 | x1.30 |
|---|---|---|---|---|---|---|---|---|
| Pinnacle close 2012-23 | 6,282 | 5.51% | 7.13% | +25.8% [+18.6, ...] | +22.1% | +18.4% [+10.5, ...] | +10.9% [+2.3, ...] | +3.5% [-5.8, ...] |
| Betfair close 2025+ | 759 | 5.93% | 7.38% | +23.3% [+6.0, ...] | +19.6% [+1.4, ...] | +15.9% [-3.2, ...] | +8.4% | +0.9% |

What this means in practice:

- **Return on capital at risk.** The return is per backer's stake. The liability is about 16 times the stake, so
  +18% a stake is about **+1.1% of the money at risk**, per bet, with long losing runs.
- **The lay spread decides it.** At a lay price 20-30% above the back, the edge is gone. Spreads at 10.0+ are wide on
  Betfair outside the big leagues.
- **Home longshots** point the same way but are not significant (n 1,252 and 136).
- **Draws at 5-10** were profitable to lay on Pinnacle's prices (+6.4% [+2.7, +9.9]) but not on Betfair's (-0.5%).
- **Caveats.** It was found by scanning many cells. The Betfair seasons (2025+) were read by earlier rules, but not
  for this question. It is a known effect in betting markets, so it is a reasonable prior, not a fresh find.

## 5. Strategies that cannot be scored yet

| Strategy | Why not | What would let us |
|---|---|---|
| In-play trading (lay the draw, back after a goal, cash-out timing) | No in-play prices in football-data; the new feed stops at the off | Extend `football/exchange_feed.py` to keep reading in play |
| Correct score, both teams to score, corners, cards | No prices held | Add the market types to the feed, or the Betfair Historic BASIC soccer plan |
| Resting orders (earn the spread instead of paying it) | No books between the open and the close | The feed's resting-order lines (`football/paper_book.py`) measure fill rate and adverse selection from its first day |
| Team-news window (after line-ups) | Line-ups collector built; no matching prices yet | The feed's 1-minute books in the last 15 minutes plus the collector's first-seen times |

## 6. What to do before the runner starts

1. **Start the football runner on paper** (`FOOTBALL_LIVE` unset).
   - It records the real back and lay prices, depth and traded money for every match. That is the missing number in
     sections 2-4.
   - Its paper lines price every entry: crossing and resting, at each checkpoint, sized.
2. **Do not set `FOOTBALL_LIVE=yes` on the current model.** Section 1 shows no edge. The GBP2 stakes would only keep the
   live key in use: an expected cost of about 4-7% of GBP20 a day (`reports/football_model.md`).
3. **Pre-register the longshot lay before reading the feed:**
   - lay away selections at 10.0+ at the last pre-play book, only where the recorded best lay is within 10% of the best
     back;
   - stake sized to a fixed liability;
   - gate on the 90% lower bound of the return on liability.
   I can write it now, so it is fixed before the first recorded book.
