# Automated trading on the Betfair win markets

The owner asked (28 Sep 2026) for betting to be automated: predictions each morning, then trading the
UK and Irish horse-racing win markets on our prices and the market's liquidity, backing overlays and,
where it helps a race as a whole, runners at fair odds or a little under; and for a judgement between
Kelly and staking to win a set amount. This is where that stands.

## What the evidence says, in one paragraph

The model is not a better forecast of the result than Betfair's SP: against the market its Brier skill
is negative, and backing every runner at BSP loses about 3.4%. What it does forecast is **where the price
is going**. Beside the morning price it carries about half the weight in the best forecast of the BSP, so
a runner the model makes much shorter than its morning price tends to shorten. The money is in that move:
back early at the longer price, then close the position at BSP. That edge is established only on
out-of-sample research forecasts. It **has not yet been confirmed at bet time**: those forecasts are built
from result rows and may know race-day facts a 06:00 bettor does not (the final field, the going, the
jockey who rode), and the one live test so far ran on a broken pipeline (reports/clv_betfair_2026q1.md).

## Which staking: the backtest

`research/queries/staking_backtest.py` (research query run 36404629070) runs every strategy on the best
model's out-of-sample forecasts (the complete-careers pair gated with race_xent) against Betfair's real
morning prices, morning volumes, BSPs and results, 31 Dec 2025 to 31 Mar 2026: 2,551 races. Fills at the
morning price, each stake capped at a tenth of the runner's morning volume and £50, 5% commission.

| strategy | held to the result: ROI (90% CI), worst drawdown | traded out at BSP: ROI (90% CI), worst drawdown |
|---|---|---|
| rule (model ≥ 22% shorter than the morning price), level £10 | +8.1% (−3.4 to +19.8), bank ruined at its worst | **+12.7% (+11.2 to +14.1), 2.2%** |
| rule, stake to win £20 | +1.0% (−6.9 to +9.2), 32% | +7.5% (+6.3 to +8.7), 1.0% |
| value (pooled price ≥ 2% edge), level | +4.5% (−4.9 to +13.5), ruined | +9.9% (+8.8 to +11.1), 2.7% |
| owner's rule (value + favourite to 10% under fair), to win £20 | −0.9% (−5.6 to +3.5), 48% | +4.4% (+3.7 to +5.1), 1.5% |
| single-runner Kelly, a quarter | +7.0% (−1.2 to +15.8), 47% | +9.8% (+8.7 to +10.9), 2.3% |
| race-level Kelly (hedging underlays), a quarter | +2.9% (−2.8 to +8.7), 56% | +7.2% (+6.4 to +8.0), 2.4% |
| race-level Kelly on the raw model (not pooled), a quarter | −1.6% (−7.4 to +4.4), bank down 98% | +5.1% (+4.5 to +5.8), 3.1% |

What it says:

- **Trade out, don't hold.** Every strategy makes money when each position is closed at BSP, with
  drawdowns of 1-3% and about 6% of days losing. Held to the result, none is reliably positive and most
  have deep drawdowns. The owner's aim of limiting losing runs is met by trading out, far better than by
  backing favourites at fair or under-fair odds (which, held, turns the value strategy from +4.5% to −0.9%).
- **Level stakes on the tested rule** give the best return on turnover. Stake-to-win puts most money on
  favourites, where the move is smallest. Kelly grows a bank fastest in absolute terms only by staking far
  more, and only on the pooled price: Kelly on the model's raw probabilities ruins the bank held to the result.
- **Race-level Kelly** is the exact form of "back a slight underlay when it helps the race": it does so only
  when the underlay raises the bank's growth. It is available as a strategy, but it adds nothing here once
  positions are traded out.
- **Scale is set by liquidity** (research query `trading_capacity.py`, run 36412850300). On the runners the
  rule backs, a median of £504 trades all morning (10% of them under £149), and the long shots, where the
  edge per £ is largest, are the thinnest (£208 at 30+). The day's money comes later (median £11,963
  pre-play, the morning 4.6% of it), but by then the edge has gone: backed at the pre-play average price
  the same runners lose 11-13% per £. Realistic scale is a tenth of what trades on the runner all morning,
  capped at £25-£50 a bet: about £140-£210 profit a day on £1,200-£1,900 turnover (+11%); £100 caps about
  £310. Fills of hundreds of pounds a runner are not there.

## Other options tested (28 Sep): any price above the forecast, all the liquidity, smaller stakes on long shots

The owner asked about backing at any price above our forecast, taking all the liquidity, and staking
less on long shots. Research queries `trading_options.py` (1-3; runs 36410545194, 36411192738,
36411631298) score them on the same data, traded out at BSP.

**Where the edge is.** At £1 level stakes, by how far the morning price sat above the forecast:

| above the forecast | under 4.0 | 4-8 | 8-16 | 16-30 | 30+ | all prices |
|---|---|---|---|---|---|---|
| 0-5% | −0.4% | +0.8% | −3.5% | −4.8% | −12.3% | −2.3% |
| 5-10% | +2.6% | +2.4% | −2.7% | −3.6% | −3.0% | −0.3% |
| 10-22% | +3.3% | +2.7% | +0.1% | −3.3% | −1.0% | +0.7% |
| 22-42% | +7.0% | +5.6% | +3.4% | +3.7% | +9.0% | +5.1% |
| 42%+ | +7.7% | +10.3% | +10.5% | +14.8% | +34.2% | +19.1% |

Our forecast is not the break-even price: the morning price carries about half of what the BSP will
be, so a small overlay is mostly noise and loses to commission. It pays only on short prices, where
the forecast is sharpest.

- **Any price above the forecast** doubles the bets (8,692 against 4,476) for the same profit: +6.3%
  per £ against +12.7% for the 22% rule, 551 units against 566.
- **A threshold by price** (about 5% under 8.0, 22% from 8.0), chosen on one half of the window and
  scored on the other: +8.8% on 6,799 bets, 599 units: some 6% more profit for half as many bets
  again.
- **All the liquidity**: take every offer at or above the threshold price (a limit order at the
  forecast × 1.22, filled down the ladder), not down to the forecast itself: the offers between the
  two are the slices that lose. The historic files hold no order book, so how much that is can only
  be measured forward; a backtest's profit grows with the money taken only because it cannot see the
  price move that money would cause.
- **Long shots are not where the bank goes when trading out**: a big overlay on a long shot is the
  best bet of all; a small one is the worst. The cure is a larger threshold there, not a smaller
  stake. Tapering the stake by price (at £500 a runner, the 22% rule) cuts the profit from £186k to
  £114k (1/√price past 4.0), £72k (bands) or £55k (to win a set amount) over the 88 days, while
  halving the daily swings at best (the square-root taper: the best profit per unit of risk). Held
  to the result instead, long shots are where the bank goes, and the taper matters far more.

## The paper trader (built)

`auto_trade.py` with the `trading/` package reads Betfair's live markets and prices (read-only), plans each
race from the model's prices, and simulates every order against the order book as it stands: a back
fills at the prices offered at or above its limit, the rest lapses, and each fill is closed with a
simulated lay at BSP. Nothing is placed on the exchange: the package has no order-placing code.

- Non-runners: each race is planned over the runners still active on Betfair, the model's probabilities
  re-normalised over them, so a withdrawal shortens the model's fair prices as it shortens the market's. In the
  races that lost runners on 28 Sep that left the 06:00 prices about 5% from a full re-run's, against 15% left
  stale (research ledger, non-runners-0928).
- Settings: `trading/config.json` (strategy `rule`, level £5 stakes, trade out at BSP on the fill, window
  08:00-11:00 UK, limits), overridable by `TRADING_*` variables.
- Limits on every simulated order: stake, race, daily turnover and bets, daily stop-loss on settled
  results, price range, the market's matched volume, half the size offered at the price.
- Kill switch: the S3 object `trading/STOP` in the predictions bucket.
- Ledger: every simulated order and settlement, `out/trading/` and `s3://<bucket>/trading/paper/<date>/`.
- `python auto_trade.py --until 11:05` in the morning; `python auto_trade.py --settle` in the evening
  settles the day from the closed markets (winner, non-runner, BSP) and emails a summary.
- Tests: `tests/test_trading_bot.py`, `tests/test_trading_staking.py`.
- Schedule: `.github/workflows/paper-trade.yml` trades from 06:50 UTC (the trader waits for its window) and
  settles at 21:45 UTC, and either can be run by hand. GitHub runs scheduled and hand-run workflows only
  from the default branch, so it starts once this branch is merged. The ledger is also kept as the run's
  artifact. It needs the secrets the other Betfair jobs use (`BETFAIR_USERNAME`, `BETFAIR_PASSWORD`,
  `BETFAIR_APP_KEY`; the certificate secrets `BETFAIR_CERT` and `BETFAIR_KEY` if present) and the AWS ones.
- A delayed application key serves the paper trader, with two caveats: its prices lag the exchange (by
  up to a few minutes), so simulated fills are against a slightly old book; and it may leave out a
  market's matched volume, in which case that market is not held to the volume floor (the share of the
  size on offer still caps each stake). If the first ledgers show every market skipped as below the
  floor, set the repository variable `TRADING_MIN_MARKET_MATCHED` to 0.
- GitHub's runners are in the US; Betfair restricts access from some countries. The first scheduled run
  shows whether a runner can log in and read the markets; if not, the trader runs from a UK machine.

**The paper trader is the forward test.** It measures the edge at the prices actually on offer at the
moment of decision, on the card as known that morning. The criterion to go further, fixed now: after at
least three weeks and 1,000 simulated trades, the stake-weighted CLV and the settled traded-out return
both above zero with their 90% intervals clear of it.

## Live execution (the owner's decision, 30 Sep 2026)

The owner chose to trade live from the start, without a paper period, and set the limits; the session's
permission mode was changed by the owner to allow the work. `auto_trade.py --live` with
`trading/config_live.json` and the workflow `live-trade.yml` place real orders on the owner's account:

- **The rule** (`closing_clv`, trading/strategy.py): 08:00-11:00 UK and never within 15 minutes of a race's off (the
  owner's limit), polled every minute, every runner of a race
  priced whole whose expected CLV at the best back price is at least +3% under the closing model
  (`model/race_book.py`; the model fitted without volume when the delayed key's feed carries none), with at least
  GBP100 matched on it when the feed reports matched money. Staked to win GBP250 before commission.
- **The owner's limits**: at most GBP300 a bet (the day's whole stake on a horse), no limit per race, at most
  GBP4,000 staked a day; when the day's limit binds, each poll's backs go in order of expected CLV.
- **Orders**: a back is a limit order at the price read, FILL_OR_KILL (at least GBP2), so nothing rests in the
  book; each matched back is laid at once at the Betfair SP for its winnings (MARKET_ON_CLOSE, liability stake x
  (price - 1)), so the price's move is kept whatever the result. A refused lay is sent again each minute, five
  times at most; nothing is ever cancelled (the open lays are the hedges).
- **Never twice**: every order is on record at once (the ledger, copied to S3 after each); a session that starts
  again takes up the day from the ledger and from Betfair's own list of the orders (listCurrentOrders), and a lost
  reply is settled by that list before anything else is sent. An order whose fate cannot be read stops the day.
- **The switches**: the repository variable `TRADING_LIVE` must be `yes` for any order; the job runs only on the UK
  runner named by `BETFAIR_RUNNER` (Betfair refuses GitHub's own runners). To stop at once: cancel the running
  "Live trading" run, or create the S3 object `trading/STOP` (no new bets within a minute).
- **Settlement**: each race is settled from Betfair's record of the settled bets (listClearedOrders): what Betfair
  paid on every back and every lay at SP, and the SP the lays matched at (the CLV), with commission at 5% of the
  race's net winnings. Betfair settles a race some minutes after it is run and the race waits until then: the
  delayed key's feed carries no SP, and on the first live evening (30 Sep) settling from the feed counted every
  lay as nothing. The evening job settles such a race again from Betfair's record; the old rows stay in the ledger
  as `unsettled`.
- **The record**: s3://$ULTRA_BETTING_S3_BUCKET/trading/live/<day>/ (ledger and summary), emailed after the
  morning session and after the evening settlement (each race's result, the BSP, the CLV, commission on the net).

**Why the morning only.** Entered near the off (Betfair's pre-play average price) the same rule loses, because the
closing model was fitted on morning prices and still trusts our price once the market is sharp. February-March 2026,
walk-forward, at a backer's price, the +3% bar, staked to win GBP250 (ledger `late-entry-0930`):

| Entry | Bets a day | CLV (90%) | GBP a day |
|---|---|---|---|
| Morning, with matched money (the tested case) | 56.6 | +6.8% (+5.8 to +7.8) | +173 |
| Morning, no matched money (the delayed key) | 63.9 | +5.3% (+4.3 to +6.2) | +131 |
| Near the off, the morning model | 21.9 | -1.8% (-2.6 to -1.2) | -19 |
| Near the off, the morning model, no matched money | 44.9 | -5.8% (-6.5 to -5.1) | -87 |
| Near the off, a model fitted on late prices | 4.8 | +2.9% (+2.0 to +3.8) | +16 |

**The delayed key's feed, seen live** (the first snapshot on the owner's UK runner, 30 Sep 15:10 UK, live-odds run
36726881232): the login from London works, and each market's matched money is reported (median GBP12,407) but each
runner's is not (0 on all 274 priced runners). The trader then reads the model fitted without volume, the best of
what that feed allows. Estimating each runner's volume as the race's total times its share of the book (as the
hourly list does) gives the same CLV for fewer pounds (ledger `delayed-key-volume-0930`):

| Runner volume read by the closing model | Bets a day | CLV (90%) | GBP a day |
|---|---|---|---|
| The runner's own (not on the delayed key) | 56.6 | +6.7% (+5.8 to +7.7) | +172 |
| None, the model fitted without volume (the trader) | 63.8 | +5.3% (+4.3 to +6.2) | +132 |
| Estimated from the race's total, fitted on the estimate | 52.2 | +5.3% (+4.3 to +6.3) | +115 |
| Estimated, read by the model fitted on the runner's own (the hourly list) | 58.9 | +5.3% (+4.3 to +6.3) | +110 |

Still the owner's to supply: a live application key restores the matched money the delayed key leaves out
(worth about a quarter of the backtested CLV), and a certificate (`BETFAIR_CERT`, `BETFAIR_KEY`) makes the login
non-interactive. The backtest behind the rule covers February-March 2026 only; the live record is the test now.

**The market record (the owner's ask, 30 Sep: keep all of it for the models).** Read-only, on the UK runner:

| What | How often | Kept in S3 | Loaded nightly into horse_racing.db |
|---|---|---|---|
| Every book the trader reads (`BETFAIR_RECORD=1`) | each minute, 08:00-11:00 UK | `betfair_live/<day>/books.csv.gz` | `betfair_live_marks` |
| The day's GB/IE win and place markets (live-record.yml) | every 5 minutes, each minute in the last hour, to 21:30 UK | the same | `betfair_live_marks` |
| The catalogue: cloth, stall, jockey, trainer, age, weight, rating, form, headgear, forecast price | once a market | `betfair_live/<day>/markets.csv.gz` | `betfair_live_markets` (matched to race_results) |
| The settled books: BSP, winners, removals and reduction factors | after racing and next morning | in `books.csv.gz` (source `final`) | `betfair_live_marks`, mark `final` |
| Betfair's daily price files (morning and pre-play prices and volumes, BSP, in-play range), which Betfair refuses to GitHub's runners | daily (live-record.yml), backfills by hand (betfair-prices.yml) | `betfair_prices_raw/` | `betfair_prices` (every file the database lacks) |
| The live trader's ledger | each order | `trading/live/<day>/ledger.csv` | `live_orders` |

`betfair_live_marks` keeps, for each runner, the book nearest to 08:00-12:00 UK and to 120, 60, 30, 15, 10, 5, 3
and 1 minutes before the off, the last book before the off, and the settled one. The full-resolution books stay in
S3. One runner serves both jobs, so the recorder gives way to the trader: its 06:55 UTC run records the whole day
only when the trader is not live, and its 10:10 UTC run takes the day from the end of the morning session.
daily-results.yml, the one job that writes the database, does the loading; a failed load never costs the night's
results.

The old `execute.yml` workflow (disabled by hand on 19 Aug) places live bets by default and has known
faults (real bets never settle, the stop-loss never counts them, the timing window is unused, the
certificate path is never expanded). It should stay disabled.
