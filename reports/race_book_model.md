# The race book: trading each market before the off

Written 30 Sep 2026 for the owner's brief:

- trade each market before the race;
- take positions against the horses the model rates as poor value;
- make a new book from a subset of the runners (three to seven in some races);
- include horses level with our price, or small individual underlays, as long as the race as a whole has
  positive price expectation;
- above all, get the sharpest closing-line value (CLV) possible.

Code: `model/race_book.py`, with tests in `tests/test_race_book.py`. Backtest:
`research/queries/done/race_book_backtest.py`, research query run 36676743958, on January–March 2026 only;
the locked holdout was not read.

## The model

### 1. The closing price: our forecast of the BSP book

The forecast moves as close to the final closing price as it can by using two sources at the moment of
trading:

- the market's price now;
- our own 06:00 price (the served gated three).

It is a conditional logit fitted to the Betfair SP book itself.

- **The target.** For each race the closing probabilities are `q_i = (1/BSP_i) / Σ 1/BSP_j`. The forecast
  `q̂ = softmax(X·β)` is fitted to them by cross-entropy, so the forecast is always a whole book of 1.
- **The inputs (X).** The log market probability now and the log model probability, each on its own and
  each times:
  - the runner's traded volume;
  - the race's total traded volume.

  There is also a volume term on its own.

What it learnt on January 2026 (and on January–February):

- **Market and model.** Weight 0.55–0.57 on the market and 0.37 on our price.
- **Volume.** The more money a runner has traded, the more the market's price counts (+0.14) and ours
  counts less (−0.16). A runner with more morning money also tends to shorten into the off (+0.11 to
  +0.16).
- **Spread of the miss.** From 0.51 for the thinnest-traded fifth of runners to 0.22 for the best-traded.
- **Time of day and curvature.** Adding them was tested and changed nothing (0.3018 against 0.3021). The
  model stays at seven inputs.

**How close each forecast gets to the BSP book** (mean |log q − log q_close| per runner; lower is closer):

| Forecast | February | March |
|---|---|---|
| Morning market alone | 0.350 | 0.358 |
| Our model alone | 0.377 | 0.400 |
| The two averaged | 0.334 | 0.345 |
| **Closing model** | **0.302** | **0.313** |

The closing model is 14% nearer the close than the morning market and 20% nearer than our model alone. That
is the "closer to the final closing price" step.

### 2. The value of each position at the close

The close-out values:

- **Backing** a runner at price m and laying the same money back at the off returns `m/BSP − 1` per unit,
  whatever the result.
- **Laying** at m returns `1 − m/BSP`.

Writing `1/BSP = O·q`, where O is the BSP book (median 1.003–1.005), each position's expected CLV is:

- backing: `m·O·E[q] − 1`;
- laying: `1 − m·O·E[q]`.

The price risk comes from draws of q around q̂: each runner is drawn by its own spread, then the race is
renormalised. A horse that shortens takes its share from the rest, which is exactly why a book's runners
hedge one another.

### 3. The book

`build_book` sets stakes on every runner of the race at once, backs and lays, for the most expected log
growth. It has two modes:

- **close**: the value at the off, with price risk only. This is the trade.
- **result**: Kelly on the result, where exactly one runner wins, using the closing model's probabilities.
  This is the book held to the result.

The owner's rules are its constraints:

| Owner's rule | Constraint |
|---|---|
| Positive price expectation on the race overall | The book's expected value ≥ θ × money at risk (θ = 3%) |
| Level or small individual underlays allowed, poor value never | No position below −δ expected edge (δ = 3%) |
| Take positions against poor value | Lays allowed, or backs on the rest of the field |
| Bank and liquidity | The race's exposure is capped; each runner's stake is capped |

Money at risk means stakes plus lay liabilities.

A horse level with our price, or a little under it, enters the book only when it pays in the outcomes where
the rest of the book loses. It must lower the book's risk by more than it costs in expectation. So the
owner's idea falls out of the optimisation rather than being a rule bolted on.

`dutch_book` is the owner's rule as written, for comparison:

- take runners in order of expected edge;
- add each one while its own edge is at least −3% and the subset's book keeps at least +3%;
- dutch the stakes so every horse in the subset returns the same.

### 4. In the day (paper only until the owner decides)

At each trading point (for example 06:00, T−60 min and T−10 min):

1. Take the exchange prices and traded volumes.
2. Update the closing model's forecast.
3. Rebuild each race's book.
4. Trade the difference from the position already held.
5. Close out (green up) at the off, or hold the result book.

The backtest has one entry point, Betfair's morning volume-weighted price, and one exit, the BSP.

## What the backtest says (February–March 2026, 1,665 whole races)

Setup:

- The closing model is fitted on the months before each test month.
- Positions are only taken where the runner had £100 or more matched in the morning.
- Commission is 5% on each race's net winnings.
- Returns are per unit of money at risk, with 90% race-bootstrap intervals.

| Strategy | Races | Horses a race | CLV per unit | Races with + CLV | Held to the result | Risk-adjusted CLV (per race) |
|---|---|---|---|---|---|---|
| Pre-registered rule (model ≥ 22% shorter) | 1,401 | 2.2 | +13.2% (+11.5 to +14.9) | 59% | +10.5% (−5.2 to +27.2) | 0.32 |
| **Single backs, closing model ≥ 5% edge** | 1,430 | 2.1 | **+15.1% (+13.6 to +16.8)** | 65% | +21.1% (+5.3 to +36.7) | 0.38 |
| Book at the close, backs | 1,560 | 2.6 | +13.9% (+12.5 to +15.5) | 65% | +16.2% (+1.7 to +31.7) | 0.38 |
| Book at the close, backs and lays | 1,580 | 2.7 | +13.3% (+12.0 to +14.8) | 65% | +14.5% (+1.2 to +29.0) | 0.39 |
| **Book on the result, backs** | 1,483 | 3.9 | +8.4% (+7.7 to +9.2) | 70% | **+8.2% (+3.1 to +13.7)** | 0.41 |
| Book on the result, backs and lays | 1,521 | 5.2 | +4.1% (+3.8 to +4.4) | 75% | +2.6% (+0.4 to +4.8) | **0.48** |
| Owner's dutch, race floor +3% | 1,560 | 3.6 | +4.7% (+4.0 to +5.5) | 66% | −0.5% (−6.3 to +5.5) | 0.25 |
| Owner's dutch, floor +6% | 1,347 | 3.0 | +8.6% (+7.5 to +9.8) | 66% | +4.0% (−4.4 to +12.9) | 0.34 |
| Owner's dutch, floor +10% | 1,047 | 2.4 | +12.5% (+10.7 to +14.3) | 65% | +13.3% (−4.4 to +33.7) | 0.35 |
| Single lays, ≥ 5% edge | 1,505 | 2.6 | +0.7% per unit of liability | 78% | 0.0% | 0.29 |

Each strategy against the pre-registered rule, races resampled together:

| Strategy | Difference in CLV per unit |
|---|---|
| Single backs by the closing model | **+1.9 points (+1.0 to +2.7)** |
| Close book, backs | +0.7 (−0.2 to +1.6) |
| Close book, backs and lays | +0.1 (−0.9 to +1.0) |
| Result book, backs | −4.8 (−6.3 to −3.3) |
| Dutch at +3% | −8.6 (−10.2 to −6.9) |

The ranking is the same in February and in March.

### Sharpest CLV per £ matched

Single backs chosen by the closing model are the best, +15.1%. They beat the current rule by 1.9 points, and
that difference is resolved: it is the gain from pricing the close better.

The book at the close matches the singles per unit and reaches more races.

### Most consistent race by race

The Kelly books on the result are the owner's structure:

- backs only, 3.9 horses a race;
- 70–75% of races positive at the close;
- the best risk-adjusted result of anything tested: +8.2% held to the result, with the narrowest interval.

They hold 3–7 horses in 1,126 of 1,483 races (76%): 1 horse in 51 races, 2 in 248, 3–4 in 772, 5–7 in 354,
8 or more in 58.

They make less per £ because some of their horses are hedges. In 719 of 1,483 races the book took a level or
small-underlay horse. Those horses cost 0.27 units of CLV in all, against the book's +6.73.

### The dutch rule as written dilutes the edge

At a +3% race floor it keeps adding horses until the book sits near +3%. It also stakes most on the short,
near-fair ones. It makes +4.7% at the close and loses held to the result.

Raising the floor to +10% brings it back to +12.5%, on 2.4 horses a race. The Kelly book does the owner's
idea better: it adds a near-fair horse only when that horse lowers the risk.

### Positions against poor value are best taken as lays of the short-priced

Laying the horses the model rates as poor value earns about +12% of the lay stake by the off. Most of them are
long-priced, though, so per £ of liability it is only +0.7%.

Oppose a poor-value horse by laying it when it is short in the betting, and by leaving it out of the book when
it is long. The optimiser makes that choice by itself: it laid 0.4 horses a race.

With the lay price 2% worse, the close book with lays still makes +13.5%.

## Staking to win £250 on every horse (the owner's staking)

The owner backs every selected horse to win the same £250, so a horse at price m gets £250 ÷ (m − 1). Which
horse wins does not matter; the portfolio does.

Same data and closing model: February–March 2026. Research query `to_win_staking.py`, run on the database as
well as locally.

| Horses backed | Bets | Turnover | CLV at the close | Held to the result (90%) | Days up | Worst drawdown | Longest losing run |
|---|---|---|---|---|---|---|---|
| Every horse with positive expected CLV | 5,237 | £270,127 | +£12,629 (+4.7%) | +£3,698, +1.4% (−2.7 to +5.4) | 46% | −£3,378 | 4 days |
| **Expected CLV ≥ +3%** | 3,816 | £164,478 | **+£11,855 (+7.2%)** | **+£11,364, +6.9% (+1.3 to +12.5)** | 54% | −£2,681 | 6 days |
| Expected CLV ≥ +5% | 3,004 | £111,780 | +£9,848 (+8.8%) | +£8,372, +7.5% (+0.3 to +14.3) | 51% | −£1,916 | 6 days |
| Expected CLV ≥ +10% | 1,720 | £43,805 | +£6,417 (+14.7%) | +£9,072, +20.7% (+8.1 to +32.8) | 56% | −£1,131 | 4 days |
| Pre-registered rule | 3,016 | £82,432 | +£6,096 (+7.4%) | +£1,145, +1.4% (−6.9 to +9.8) | 47% | −£3,411 | 3 days |
| Owner's book (by edge, each ≥ −3%, race ≥ +3%) | 5,515 | £240,256 | +£11,605 (+4.8%) | +£4,668, +1.9% (−2.1 to +6.2) | 47% | −£2,425 | 4 days |

### Staking to win puts the money on the short prices

Among every horse with positive expected CLV:

- Horses under 3.0 were 399 of the 5,237 bets but took £102,159 of the £270,127 staked.
- Their CLV is the smallest, +3.7%. Horses over 21.0 make +19.4%, but on £8,229.

So the same horses at level stakes make about twice the percentage: +9.3% at the close for every
positive-expected-CLV horse, against +4.7% to win. They also carry far more risk. Staking to win is the
steadier choice.

### The marginal horses are not worth the turnover

Going from "expected CLV ≥ +3%" down to "any positive expected CLV" adds:

- £105,649 of turnover;
- +£774 of CLV (+0.7%);
- −£7,666 held to the result.

After spread and slippage those horses are worth nothing. Under this staking the bar should be about +3%.

### Profit over time

Three of the selections were tracked week by week: every positive-expected-CLV horse, ≥ +5%, and the owner's
book. At the close, each one's cumulative CLV rose in every one of the ten weeks scored. For the
positive-expected-CLV selection it went +£215, +£1,609, +£3,104 … +£12,629. Held to the result the curve is
noisier, but every row of the table ends up.

Stakes above half of a runner's morning matched volume were 0.1% of bets or fewer.

## Caveats

1. **The backtest forecasts know race-day facts** (the going at the off, the jockey who rode, the horses that
   ran). That flatters every row equally, so the ranking is more reliable than the levels. The forward test
   on the 06:00 path, and a forward paper run of this book, measure the real levels.
2. **The entry price is the morning volume-weighted average**, not a price you could take at one moment.
   Live, the entry is the best back or lay price at the trading point, less the spread.
3. **Liquidity.** Stakes are per unit here. Live, each runner's stake must be capped at a share of the money
   traded on it. That cap, not Kelly, sets the size of a close-out book.
4. Two months of test races (1,665). The intervals above are the uncertainty.

## What it needs to run (paper)

- **Live exchange prices and volumes at the trading points.** That means `BETFAIR_APP_KEY` for the 06:00 and
  pre-race jobs (still not set).
- **The paper trader** records the book, the price taken and the BSP for every race, for the owner to review.
  No real-money orders are placed until the owner decides.

## Recommendation

- Price every runner with the closing model: it is the sharpest estimate of the close there is here.
- **Trade** single backs, and close-out books, where the expected CLV is at least 5%, closed out at the off.
- **Book** with the Kelly result book (backs only, θ = 3%, δ = 3%) where the owner wants a book of several
  horses to hold.
- **Staking to win £250 on each horse** (the owner's staking): back every horse whose expected CLV is at
  least +3%. That is where the pounds are: +£11,855 at the close and +£11,364 held to the result over the two
  months. Lower bars add turnover and no profit.
- Start both in paper mode alongside the pre-registered forward test. That test is left unchanged.
