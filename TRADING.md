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

- **The rule** (`closing_clv`, trading/strategy.py): from 08:00 UK until 15 minutes before each race's off (the owner's
  decision of 1 Oct; 08:00-11:00 UK until then, the tested window) and never within 15 minutes of a race's off (the
  owner's limit), polled every minute, every runner of a race
  priced whole whose expected CLV at the best back price is at least +3% under the closing model
  (`model/race_book.py`; the model fitted without volume when the delayed key's feed carries none), with at least
  GBP100 matched on it when the feed reports matched money. Staked to win GBP250 before commission.
- **The owner's limits**: at most GBP300 a bet (the day's whole stake on a horse), no limit per race, and no limit
  on the day's stakes (the owner, 7 Oct: the account's money is the limit; GBP4,000 a day until then, when each
  poll's backs went in order of expected CLV once it bound). No limit on the number of bets (the owner, 2 Oct; a cap of
  250 a day, set when the trader was built, stopped seven backs on 1 Oct).
  The to-win target stays at GBP250 (the owner asked, 2 Oct): 1 Oct replayed at GBP300-500, settled on every horse,
  was worth less with the account's money, since the funds bind before the book and a bigger target buys top-ups at
  a fraction of the CLV of the first fills (reports/other_markets_and_stakes_1002.md, ledger
  `stake-uplift-replay-1002`, `stake-uplift-replay-full-1002`).
  **Replayed on 1-5 Oct** (7 Oct, after the owner added funds and lifted the day limit; research-query run
  37588620637, ledger `stake-target-replay-1007`), the replay at each day's own balance came within GBP16 of the real
  days: GBP250 to win, +GBP482.51 after commission, against the real +GBP466.94. CLV x stake (the expected profit,
  before commission) over the five days:

  | GBP5,000 in the account | Staked | CLV x stake | Result after 2% | Ahead of GBP250 on |
  |---|---|---|---|---|
  | GBP250 to win, GBP4,000 a day | GBP18,981 | GBP899 | +GBP612 | |
  | GBP250, no day limit (GBP8,000 never reached) | GBP23,161 | GBP895 | +GBP535 | |
  | **GBP400, GBP4,000 a day** | GBP19,319 | **GBP1,314** | **+GBP1,061** | every day |
  | GBP400, GBP8,000 a day and GBP600 a bet | GBP34,302 | GBP1,011 | +GBP466 | |

  - Beyond GBP4,000 a day the stake goes on the afternoon's entries. At GBP250 the extra GBP4,180 earned nothing.
  - A bigger target under the limit spends the day's money on the morning's horses: the last back came at 10:34-14:19
    UK at GBP400, against 13:12-20:02 at GBP250.
  - At the account's own balance (the money binding), GBP250 was best: GBP670 against GBP563-620.
  - These are five days on the delayed key's books. 6 Oct (its afternoon at +4.7%) and the live key's days are added as
    their price files arrive. The owner decides.
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
  paid on every back and every lay at SP, and the SP the lays matched at (the CLV), with commission at 2% of the
  race's net winnings, the account's rate (the owner, 2 Oct; on 1 Oct the day's settled result at 2% met the account's
  balance to the penny). Each back is scored at the price Betfair settled it: less, where a horse withdrawn after
  the bet brought a reduction factor, and so the price to set against the smaller field's BSP (from 2 Oct; on 1 Oct
  the price as matched overstated the day's CLV by about half a point). Betfair settles a race some minutes after
  it is run and the race waits until then: the
  delayed key's feed carries no SP, and on the first live evening (30 Sep) settling from the feed counted every
  lay as nothing. The evening job settles such a race again from Betfair's record; the old rows stay in the ledger
  as `unsettled`.
- **The record**: s3://$ULTRA_BETTING_S3_BUCKET/trading/live/<day>/ (ledger and summary), emailed after the
  morning session and after the evening settlement (each race's result, the BSP, the CLV, commission on the net).
- **Short of funds** (the owner, 1 Oct: "hold and then continue trading later when balance is back up"): a race's
  stakes come back only once it is run, so a morning's backs can use up the account's balance (on 1 Oct Betfair
  refused new backs from 10:19 UK, after GBP1,455 staked; a lay at SP on a horse already backed added no exposure).
  Refused for want of funds, the trader holds new bets for five minutes, reads the account's funds
  (getAccountFunds, read-only), and trades again once they are back, staking no more than the account has; the lays
  at SP that hedge the backs already matched still go. Each refused order is kept with Betfair's own reason, and the
  summary and email carry the funds at the start and the end of the session.
- **To 15 minutes before each off** (the owner, 1 Oct: "You can trade right up until 15 minutes before race start. Just
  use what's in the balance"): the session trades every race from 08:00 UK until 15 minutes before its off, with the
  balance as the races are run and their stakes come back. It is a choice made knowing the Jan-Mar test: entered near
  the off with the morning's closing model the rule lost (late-entry-0930: CLV -5.8% on the delayed key's feed, about
  GBP87 a day), against +5.3% for the morning's entries. The summary and email score the bets entered by 11:00 UK
  apart from the later ones (`clv_entered_by_11_uk`, `clv_entered_after_11_uk`), so the live record answers it. One
  runner serves both, so on a trading day the trading job runs the market recorder beside the trader (read-only; the
  trader's own books go to `books_trader.csv`), and the recorder's own 10:10 UTC run takes the evening's part.
- **Every back can be laid**: Betfair takes no lay at SP under GBP10 of liability, so no back is sent whose winnings,
  with what the horse has unhedged already, would be smaller, and every fill is at least that (minFillSize); on
  1 Oct two such fills, GBP15.37 of winnings between them, were left without a lay (one was laid with the afternoon's
  top-up; the other's horse won).

**The first live session** (30 Sep, a test the owner chose; ledger `live-first-session-0930`). The session ran at
Kempton, 19:00-20:30 UK, and entered 28 to 118 minutes before each off, a window the backtest never tested.

| Race (UK) | Horses | Staked | Result |
|---|---|---|---|
| 19:00 | 11 | GBP121.94 | +25.70 |
| 19:30 | 3 | GBP180.63 | -1.36 |
| 20:00 | 5 | GBP59.91 | +15.21 |
| 20:30 | 9 | GBP186.51 | -2.08 |
| **Total** | **28** | **GBP548.99** | **+37.47** |

- 38 of the 45 backs sent were matched, and every matched back was laid at the SP.
- The result is after GBP2.15 commission, at 5%; at the account's 2% it was GBP0.86 and +38.76.
- Our prices beat the Betfair SP by +18.4%, stake-weighted.
- Four of our horses won. A winner laid at the SP nets nothing, so the night kept about 40% of the roughly GBP100
  the price moves were worth at the SP.

**The first full day** (1 Oct; ledger `live-day-1001`). The morning session (08:07-11:06 UK), then from 12:19 UK every
race to 15 minutes before its off (the owner's decision of that morning). 250 backs were matched on 124 horses in 41
races and each horse was laid at the SP for its winnings; two of the horses were withdrawn and their bets were void.

| Horse first backed | Horses | Races | Staked | Result before commission | CLV |
|---|---|---|---|---|---|
| By 11:00 UK (the tested window) | 57 | 26 | GBP1,605.92 | +153.58 | +9.9% |
| After 11:00 UK | 65 | 29 | GBP1,085.65 | +40.99 | +8.2% |
| **Day** | **122** | **41** | **GBP2,691.57** | **+194.57** | **+9.2%** |

- **+GBP188.19 after commission.** Betfair charged 2% of each race's net winnings (GBP6.38): the account went from
  GBP1,480.11 at 12:19 UK, before the first race, to GBP1,668.30 with nothing at risk, the settled result at 2% to the
  penny. The ledger and the emails of the day applied 5% (GBP15.95, +178.62); 2% from 2 Oct.
- **By time to the off** at the horse's first back: three hours or more +9.2% (61 horses, GBP1,684, +148.64 before
  commission), one to three hours +17.8% (29, GBP536, +44.20), 15 to 60 minutes -0.4% (32, GBP472, +1.73). One day;
  the Jan-Mar test had entries near the off at -5.8%.
- **Funds.** The morning's backs used up the balance: Betfair refused 113 backs on 9 horses from 10:19 UK (its reason
  then shown only as ERROR_IN_ORDER). The afternoon began with GBP2.95 to bet; the trader held 25 times and traded
  again each time races were run and their stakes came back. Of the 91 horses held for funds, 28 were never backed.
  From 16:35 UK the balance was enough.
- **The number of bets.** The cap of 250 a day was reached at 17:41 UK and stopped seven backs in Newcastle's last
  four races (expected CLV +3.0% to +3.7%); no cap from 2 Oct.
- **Non-runners.** The afternoon session took the reduced prices from Betfair at its start (8 horses in three races).
  For races that lost a runner later the day's CLV sets the price as matched against the smaller field's BSP; from
  what Betfair paid on the lays, that is about half a point: some +8.7% on the day, +9.6% by 11:00, +7.5% after.
- **The two small fills** left without a lay in the morning: Louiescall's (GBP8.27 of winnings) was laid with the
  afternoon's top-up (the horse lost, +20.39); Crafty Gael's (GBP7.09) stood alone and the horse won, +7.09.
- 47 backs were killed (FILL_OR_KILL, the price had gone): 35 in the morning, 12 in the afternoon. Every race was
  settled from Betfair's record, with the BSP. The recorder ran beside the trader all afternoon (88 markets, 494
  polls), the trader kept 89,615 books, and the night's load took 10,956 marks and the 928 ledger rows.

**The second full day** (2 Oct; ledger `live-day-1002`). One session from 08:05 UK to 15 minutes before each off, the
last back at 19:30 UK. 281 backs were matched on 125 horses in 43 races, each laid at the SP for its winnings; four
horses were withdrawn and their GBP206.71 was void (GBP182.48 of it on Cranachan, Ascot 16:45).

| Horse first backed | Horses | Races | Staked | Result before commission | CLV |
|---|---|---|---|---|---|
| By 11:00 UK | 49 | 27 | GBP1,805.06 | +121.77 | +4.1% |
| After 11:00 UK | 72 | 30 | GBP1,677.84 | -72.64 | -4.5% |
| **Day** | **121** | **43** | **GBP3,482.90** | **+49.13** | **-0.1%** |

- **+GBP44.30 after commission** (GBP4.83 at 2%): the account went from GBP1,668.30 to GBP1,712.60. Thirteen of the
  121 horses won. At -0.1% CLV the day was worth about nothing beforehand (about -GBP2); the +44 is the results.
- **By time to the off** at the horse's first back: three hours or more +3.3% (58 horses, GBP2,099, +119.25 before
  commission), one to three hours -6.7% (38, GBP946, -66.86), 15 to 60 minutes -2.0% (25, GBP438, -3.26). Every horse
  first backed by 11:00 was three hours or more from its off.
- **The two days together** (243 horses, GBP6,174.47, CLV +4.0%, +243.70 before commission): first backed by 11:00 UK
  +6.8% (106 horses, GBP3,411), after 11:00 +0.5% (137, GBP2,763); three hours or more before the off +5.9% (119,
  GBP3,783), one to three hours +2.2% (67, GBP1,482), the last hour -1.2% (57, GBP910). The backtest put entries near
  the off at -1.8% to -5.8% (below); the live record agrees for the last hour.
- **First fills and top-ups.** Each horse's first fill made +6.1% against its BSP; the top-ups (152 fills, GBP1,923,
  55% of the stake) made nothing, at prices 2.2% shorter. On 1 Oct the figures were +13.8% and +6.0%. Over the two
  days: first fills +9.5% (GBP2,821), top-ups +2.6% (GBP3,354), a gap of 7.1 points (90% interval 2.1 to 12.0,
  resampled by horse). The closing model expected the same of both (+5.4% and +5.3%), since it does not know a horse
  is being topped up. Where it expected 3-6%, first fills made +8.9% and top-ups +1.3%. But the first fills are
  mostly the morning's and the top-ups come later, so the gap is largely the time of day. The replay of 1 Oct,
  settled on every horse (research-query run 37072291665, ledger `topup-policy-replay-1001`), found the live rule
  worth the most at the day's balance: CLV x stake GBP214. The variants were worth less:
  - no top-ups GBP84;
  - top-ups only at 6% GBP169;
  - new horses first, with GBP300 kept free, GBP158;
  - no backs in the last hour GBP185;
  - morning only GBP149.
  The money a variant frees goes to later, weaker horses (with no top-ups the first fills made only +4.4%). With
  GBP5,000 in the account (the day limit binding), no backs in the last hour was worth GBP184 against GBP159. 2 Oct
  is replayed once its price file is published. Any change to the rule is the owner's.
- **Funds.** Betfair refused one back for funds. The trader held new backs each time the funds it read were spent,
  from 09:54 to 17:31 UK: 76 horses were held, 16 of them never backed. The day's settled summary showed
  `held_for_funds` 0, because the settling run never holds. From 3 Oct the summary also counts the horses held, from
  the ledger (`horses_held_for_funds`, `horses_held_never_backed`).
- 68 backs were killed (fill-or-kill: the price had gone), against 47 on 1 Oct. Every race was settled from Betfair's
  record, with the BSP. The day staked GBP3,689.61 including the void bets, short of the GBP4,000 day limit.

**The third full day** (3 Oct, a Saturday; ledger `live-day-1003`). One session from 08:09 UK to 15 minutes before
each off, the last back at 19:43 UK. 286 backs were matched on 128 horses in 41 races, each laid at the SP for its
winnings; no horse was withdrawn. The day's stakes reached the GBP4,000 day limit at the last race (GBP3,999.99;
one horse at Southwell 20:00 was left with less than the minimum stake).

| Horse first backed | Horses | Races | Staked | Result before commission | CLV |
|---|---|---|---|---|---|
| By 11:00 UK | 55 | 26 | GBP2,094.51 | +291.25 | +9.6% |
| After 11:00 UK | 73 | 30 | GBP1,905.48 | +25.85 | +1.8% |
| **Day** | **128** | **41** | **GBP3,999.99** | **+317.10** | **+5.9%** |

- **+GBP307.16 after commission** (GBP9.94 at 2%): the account went from GBP1,712.60 to GBP2,019.76. Eleven of the
  128 horses won. At +5.9% CLV the day was worth about +GBP236 beforehand (CLV x stake); the results did the rest.
- **By time to the off** at the horse's first back:

  | Time to the off | Horses | Staked | Result | CLV |
  |---|---|---|---|---|
  | Three hours or more | 62 | GBP2,216 | +315.97 | +9.9% |
  | One to three hours | 46 | GBP1,353 | -37.46 | -1.8% |
  | The last hour | 20 | GBP431 | +38.59 | +9.7% |

- **The three days together**: 371 horses, GBP10,174.46 staked, CLV +4.7%, +560.80 before commission. After
  commission the account went from GBP1,480.11 to GBP2,019.76 (+GBP539.65).

  | Horse first backed | Horses | Staked | CLV |
  |---|---|---|---|
  | By 11:00 UK | 161 | GBP5,505 | +7.9% |
  | After 11:00 UK | 210 | GBP4,669 | +1.0% |
  | Three hours or more before the off | 181 | GBP5,999 | +7.4% |
  | One to three hours | 113 | GBP2,835 | +0.3% |
  | The last hour | 77 | GBP1,341 | +2.3% |

- **First fills and top-ups.** On 3 Oct the top-ups made more than the first fills: +8.4% (158 fills, GBP2,198, 55%
  of the stake) against +5.2% (GBP1,802). Over the three days the first fills made +7.8% (GBP4,623) and the top-ups
  +4.9% (GBP5,552). The gap is 3.0 points (90% interval -0.6 to 7.0, resampled by horse), so it is no longer
  resolved.
- **The replay of 1-2 Oct** (research-query run 37156819718, ledger `topup-policy-replay-1002`): settled on every
  horse, scored on CLV x stake before commission. At each day's balance the live rule earned GBP261 over the two days
  (GBP214 and GBP47), and no variant beat it on both days:

  | Variant | 1 Oct | 2 Oct | Two days |
  |---|---|---|---|
  | The live rule | 214 | 47 | 261 |
  | Morning only | 149 | 125 | 274 |
  | No top-ups | 84 | 166 | 250 |
  | Top-ups only at 6% | 169 | 74 | 243 |
  | No backs in the last hour | 185 | 57 | 242 |
  | No backs in the last three hours | 139 | 91 | 230 |
  | New horses first | 158 | 60 | 219 |
  | Top-ups at 6%, not in the last hour | 124 | 81 | 206 |
  | A horse under 4.0 only at 6% expected CLV | 164 | 26 | 190 |
  | A horse under 4.0 only at 10% expected CLV | 165 | 20 | 185 |

  The short-price variants did worst, as the five-month backtest said they would. With GBP5,000 in the account (the
  day limit binding), no backs in the last hour beat the live rule on both days, narrowly: GBP184 against GBP159,
  and GBP227 against GBP225. The live rule stays; any change to it is the owner's.
- **Funds.** Betfair refused one back for funds. The trader held new backs each time the funds it read were spent,
  from 09:29 to 16:55 UK: 86 horses were held, 23 of them never backed. 55 backs were killed (fill-or-kill: the price
  had gone). Every race was settled from Betfair's record, with the BSP. The 21:45 UTC settling run never started, so
  it was run by hand at 22:10 UTC (run 37157529273).

**The fourth full day** (4 Oct, a Sunday; ledger `live-day-1004`). GitHub's scheduler did not start the 06:50 UTC
session (it created the run at 08:50 UTC, too late), so the session was started by hand and traded from 09:39 UK: the
08:00-09:39 window, the best of the morning so far, was missed. The last back went on at 17:08 UK. 212 backs were
matched on 88 horses in 27 races at four meetings (Uttoxeter, Kelso, Curragh, Killarney), each laid at the SP for its
winnings; 7 of the horses were withdrawn (GBP363.80 void). GBP3,203 was staked, under the day limit, and the account
was never short.

| Horse first backed | Horses | Races | Staked | Result before commission | CLV |
|---|---|---|---|---|---|
| By 11:00 UK | 27 | 16 | GBP1,175.58 | -4.74 | +4.3% |
| After 11:00 UK | 54 | 21 | GBP1,663.97 | -126.76 | -4.7% |
| **Day** | **81** | **27** | **GBP2,839.55** | **-131.50** | **-1.0%** |

- **-GBP135.12 after commission** (GBP3.62 at 2%): the account went from GBP2,019.76 to GBP1,884.64. Twelve of the 81
  horses won.
- **Non-runners took the day.** Later withdrawals cut the settled price of 28% of the stake (GBP793 in nine races at
  Kelso, Uttoxeter, Curragh and Killarney) by 4% to 41%: at the fill prices the day's CLV was +6.9%, at Betfair's
  settled prices -1.0%. Over 1-3 Oct the same cost was 1.5 points. The closing model does not price it.
- **By time to the off** at the horse's first back:

  | Time to the off | Horses | Staked | Result | CLV |
  |---|---|---|---|---|
  | Three hours or more | 36 | GBP1,498 | -44.64 | +2.5% |
  | One to three hours | 27 | GBP809 | -97.25 | -10.9% |
  | The last hour | 18 | GBP532 | +10.39 | +4.5% |

- **The four days together**: 452 horses, GBP13,014.01 staked, CLV +3.5%, +429.30 before commission. After
  commission the account went from GBP1,480.11 to GBP1,884.64 (+GBP404.53; the four days' results +GBP394.96).

  | Horse first backed | Horses | Staked | CLV |
  |---|---|---|---|
  | By 11:00 UK | 188 | GBP6,681 | +7.3% |
  | After 11:00 UK | 264 | GBP6,333 | -0.5% |
  | Three hours or more before the off | 217 | GBP7,497 | +6.4% |
  | One to three hours | 140 | GBP3,644 | -2.2% |
  | The last hour | 95 | GBP1,873 | +2.9% |

- **First fills and top-ups, at the settled price** (the three-day figures above were at the fill price): over the
  four days the first fills made +5.7% (GBP5,687) and the top-ups +1.8% (GBP7,327, 56% of the stake); the gap is 3.9
  points (90% interval 0.3 to 7.8, resampled by horse). On 4 Oct the top-ups lost 4.1% (63% of the day's stake),
  the first fills made +4.2%.
- **Against the closing model** (`fill_calibration.py`, 1-4 Oct, 992 fills in 152 races, at the settled price): all
  fills +3.5% staked (90% +0.9% to +6.2%), 1.2 points under the expected CLV they were backed on (-3.8 to +1.5); first
  fills +5.7% (+3.1% to +8.5%), 0.4 points over; top-ups +1.8% (-2.0% to +5.3%), 2.5 points under (-6.1 to +1.0).
- **The replay of 1-3 Oct** (research-query run 37238310484, ledger `topup-policy-replay-1003`): settled on every
  horse, scored on CLV x stake before commission, at each day's balance. The live rule earned GBP487 over the three
  days and was best on 3 Oct; no variant beat it on all three days, nor over the three together:

  | Variant | 1 Oct | 2 Oct | 3 Oct | Three days |
  |---|---|---|---|---|
  | The live rule | 214 | 47 | 226 | 487 |
  | New horses first | 158 | 60 | 213 | 432 |
  | Morning only | 149 | 125 | 149 | 423 |
  | Top-ups only at 6% | 169 | 74 | 155 | 398 |
  | No backs in the last three hours | 139 | 91 | 167 | 397 |
  | No backs in the last hour | 185 | 57 | 145 | 387 |
  | Top-ups at 6%, not in the last hour | 124 | 81 | 145 | 351 |
  | A horse under 4.0 only at 6% expected CLV | 164 | 26 | 133 | 322 |
  | No top-ups | 84 | 166 | 72 | 322 |
  | A horse under 4.0 only at 10% expected CLV | 165 | 20 | 114 | 299 |

  At the account's money the top-ups still add (GBP165 over no top-ups): their CLV is lower than the first fills',
  but the money they use would mostly have sat idle. With GBP5,000 in the account no backs in the last hour was ahead
  on all three days, narrowly (GBP770 against GBP742). The live rule stays; any change to it is the owner's.
- **Funds and the scheduler.** No hold for funds (the session started late with the whole balance); 91 backs were
  killed (fill-or-kill: the price had gone). Every race was settled from Betfair's record, with the BSP; the 21:45 UTC
  settling run started at 21:52 UTC.

**The fifth full day** (5 Oct, a Monday; ledger `live-day-1005`). GitHub's scheduler created the 06:50 UTC session at
07:20 UTC; the 07:05 UTC watchdog had already started it (07:07 UTC), and it traded from 08:08 UK to the last back at
20:11 UK. The late scheduled run waited behind it and ran for a minute at 21:30 UK without trading. 299 backs were
matched on 139 horses in 52 races at seven meetings (Kempton, Wolverhampton, Pontefract, Yarmouth, Killarney,
Fairyhouse, Stratford), each laid at the SP for its winnings; one horse was withdrawn (GBP12.88 void).

| Horse first backed | Horses | Races | Staked | Result before commission | CLV |
|---|---|---|---|---|---|
| By 11:00 UK | 71 | 36 | GBP2,404.79 | +93.58 | +5.1% |
| After 11:00 UK | 68 | 37 | GBP1,477.70 | -15.10 | -2.9% |
| **Day** | **139** | **52** | **GBP3,882.49** | **+78.48** | **+2.1%** |

- **+GBP71.98 after commission** (GBP6.50 at 2%): the account went from GBP1,884.64 to GBP1,956.62. Ten of the 139
  horses won.
- **Non-runners.** Later withdrawals cut the settled price of 17% of the stake (GBP643 in nine races) by 2% to 13%: at
  the fill prices the day's CLV was +2.9%, at Betfair's settled prices +2.0%.
- **By time to the off** at the horse's first back:

  | Time to the off | Horses | Staked | Result | CLV |
  |---|---|---|---|---|
  | Three hours or more | 77 | GBP2,511 | +45.17 | +3.3% |
  | One to three hours | 34 | GBP844 | +33.45 | +0.1% |
  | The last hour | 28 | GBP527 | -0.14 | -0.6% |

- **The five days together**: 591 horses, GBP16,896.50 staked, CLV +3.2%, +507.78 before commission. After commission
  the account went from GBP1,480.11 to GBP1,956.62 (+GBP476.51; the five days' results +GBP466.94).

  | Horse first backed | Horses | Staked | CLV |
  |---|---|---|---|
  | By 11:00 UK | 259 | GBP9,086 | +6.7% |
  | After 11:00 UK | 332 | GBP7,811 | -0.9% |
  | Three hours or more before the off | 294 | GBP10,008 | +5.6% |
  | One to three hours | 174 | GBP4,488 | -1.8% |
  | The last hour | 123 | GBP2,401 | +2.2% |

- **First fills and top-ups, at the settled price.** On 5 Oct the top-ups made more than the first fills: +2.9% (160
  fills, GBP2,241, 58% of the stake) against +0.9% (GBP1,641). Over the five days the first fills made +4.6% (GBP7,328)
  and the top-ups +2.0% (GBP9,569); the gap is 2.6 points (90% interval -0.8 to 5.9, resampled by horse), no longer
  resolved.
- **Against the closing model** (`fill_calibration.py`, 1-5 Oct, 1,291 fills in 204 races, at the settled price): all
  fills +3.2% staked (90% +0.7% to +5.6%), 1.9 points under the expected CLV they were backed on (-4.2 to +0.5); first
  fills +4.6% (+2.2% to +7.1%), 0.4 points under (-3.4 to +2.6); top-ups +2.0% (-1.4% to +5.5%), 3.1 points under
  (-6.0 to +0.1). On 5 Oct the fills were backed at +5.1% expected and closed +2.0% staked.
- **The replay of 1-4 Oct** (research-query run 37379755993, ledger `topup-policy-replay-1004`): settled on every
  horse, scored on CLV x stake before commission, at each day's balance (4 Oct opened with GBP2,019.76). New horses
  first came out GBP4 ahead of the live rule over the four days but was behind it on 1 and 3 Oct; no variant beat the
  live rule on every day:

  | Variant | 1 Oct | 2 Oct | 3 Oct | 4 Oct | Four days |
  |---|---|---|---|---|---|
  | New horses first | 158 | 60 | 213 | 229 | 661 |
  | The live rule | 214 | 47 | 226 | 169 | 656 |
  | Morning only | 149 | 125 | 149 | 200 | 623 |
  | No backs in the last three hours | 139 | 91 | 167 | 213 | 610 |
  | No backs in the last hour | 185 | 57 | 145 | 222 | 609 |
  | Top-ups only at 6% | 169 | 74 | 155 | 137 | 535 |
  | Top-ups at 6%, not in the last hour | 124 | 81 | 145 | 143 | 494 |
  | A horse under 4.0 only at 6% expected CLV | 164 | 26 | 133 | 163 | 485 |
  | A horse under 4.0 only at 10% expected CLV | 165 | 20 | 114 | 136 | 435 |
  | No top-ups | 84 | 166 | 72 | 91 | 413 |

  On 4 Oct, the day the late backs lost, new horses first, morning only and both variants without late backs were
  ahead of the live rule, and the variants that cut the top-ups were behind it. With GBP5,000 in the account no
  backs in the last hour was ahead on all four days, narrowly (GBP1,044 against GBP998), and new horses first was
  the live rule (the order matters only when the money runs out). The live rule stays; any change to it is the
  owner's.
- **Funds and the limit.** The money bound, not the GBP4,000 day limit (turnover GBP3,895): the trader held new backs
  each time the funds it read were spent, from 13:43 to 16:55 UK; 97 horses were held, 30 of them never backed. Betfair
  refused one back for funds, and 73 were killed (fill-or-kill: the price had gone). Every race was settled from
  Betfair's record, with the BSP; the 21:45 UTC settling run started at 21:50 UTC.
- **The evening record.** For the first time the trader's job recorded tomorrow's markets from 17:00 to 21:30 UK:
  6 Oct's 35 GB/IE win markets, 18 polls, 6,102 rows; the late look after the last race added 5 polls (1,695 rows).
  The recorder's evening part ran once: the 10:10 UTC run waited for the session and ran from 20:31 to 21:22 UTC,
  and the 06:55 UTC run, which waited behind the trader all day, stood down by itself at 20:30 UTC. The census of 6
  Oct's racing found no new market type (win 35, place 35, other place 52, each way 34).
- **The first evening read** (`evening_entry_check.py`, research-query run 37382868223, ledger `evening-before-1005`;
  the replication and the coverage only, as one evening decides nothing). The replication is exact on the books the
  trader read at its first step: 2-5 Oct, 16, 19, 14 and 11 horses, the same expected CLV. 5 Oct's evening record (4
  Oct, 17:39 to 19:10 UK) held 54 win markets and 547 runners priced at 18:00 and at 19:00, 9% of whom were taken out
  before the off; the model priced every runner in 49 of the races. The rule chose no horse at either mark because the
  books had not formed: the best lay was a median 227 times the best back at 18:00 and 83 times at 19:00, against 1.4
  to 1.8 times at the trader's first step each morning, so no runner reached the 3% bar. The query now reads the
  evening's book without matched money, as the trader's feed does, and says why the rule chose what it did on each
  book. The trader's own evening record (17:00 to 21:30 UK, from 6 Oct's racing) shows whether the books form later in
  the evening.

**The sixth full day** (6 Oct, a Tuesday; ledger `live-day-1006`). The 06:50 UTC session started at 07:06 UTC and
traded from 08:00 UK until a GitHub internal error ended it at 10:47 UK (run 37427640563). The owner restarted it at
13:55 UK (run 37466865912) and again at 15:25 UK, after PR #117 (run 37478841166); the 15:25 UK session took up the
day's ledger (561 rows, GBP2,937.90 matched) and from then on read the live key's feed. It traded to 15 minutes before
the last off and stopped at 21:35 UK.

| Horse first backed | Staked | CLV |
|---|---|---|
| By 11:00 UK | GBP1,874.07 | +1.7% |
| After 11:00 UK | GBP1,989.09 | +4.7% |
| **Day** | **GBP3,886.75** | **+3.2%** |

- **+GBP119.52 after commission** on 307 backs in 33 races, CLV +3.2% at Betfair's settled price (the 21:45 UTC
  settling run started at 21:48 UTC, run 37536497358). The account opened at GBP1,956.62 and closed at GBP2,577.13.
  The other GBP500.99 is not from trading: a deposit, by its size.
- **For the first time the backs after 11:00 UK closed better than the morning's** (+4.7% against +1.7%). Over the
  five days before, the morning held +6.7% against -0.9%.
- **Funds and the limit.** The money bound again, not the GBP4,000 limit (turnover GBP3,886.75): 31 horses were held
  for funds, 6 of them never backed. At the 15:25 UK restart GBP706.47 was free and GBP1,828.47 at risk. Betfair
  refused one back for funds, and 55 were killed (fill-or-kill: the price had gone).
- **The six days** (1-6 Oct): GBP20,783.25 staked, CLV about +3.2%, +GBP586.46 after commission.
- **The records beside it**, from 15:25 UK:
  - the win and place markets: 44 markets, 309 polls, 27,238 rows;
  - the other-place markets: 59 markets, 37,763 rows;
  - the greyhounds: 130 markets, 64,102 rows;
  - the Tote: 35 races, 521 pool reads and 50 result reads, no errors;
  - and, for the first time over the whole window, the evening record of 7 Oct's 38 win markets from 17:00 to 21:30
    UK: 18 polls, 7,308 rows.

  The day record held no BSP. Betfair sends none in a closed market's book, and the recorder reads it at the off from
  7 Oct (ledger `recorder-bsp-1006`). The trader's CLV is unaffected: it takes the BSP from Betfair's cleared orders.

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
non-interactive. The backtest behind the rule covered February-March 2026. Since 3 Oct it covers five months,
November 2025 to March 2026 (4,342 races, 2% commission). Research query runs 37115884197 and 37134803377; ledger
`to-win-five-months-1003` and `to-win-five-months-novol-1003`.

**The live key (the owner's, 6 Oct).** The owner bought Betfair's live application key on 6 Oct. It goes in the GitHub
secret `BETFAIR_LIVE_APP_KEY`, and only a trader's own process reads it (this trader in `live-trade.yml`, and from
PR #117 the greyhound trader). The market records beside the trader, and every other workflow, keep the delayed key
(`BETFAIR_APP_KEY`), because Betfair permits no read-only use of live data (`tests/test_workflow_guards.py` holds the
split). On the live key the feed reports each runner's matched money, so the trader reads the closing model fitted
with volume (`data/models/closing_model.json`) and holds each runner to the GBP100 matched floor, as the backtest
below did. Its log then says "The trader reads the live key's feed", and its backs stop saying "feed without volume".
A lost session is logged into again, once (`trading/exchange.py`); a trading day is far inside Betfair's 24-hour
session.

Until 15:25 UK on 6 Oct the trader read the delayed key's feed (every back on 1-2 Oct logged "feed without volume"),
so it read the closing model fitted without matched volume and held no runner to the GBP100 floor. Since the owner's
restart at 15:25 UK on 6 Oct (run 37478841166, with the secret set) it reads the live key's feed, and its log says so.
The live key's feed carries the volume.

| GBP of CLV a day, best expected CLV first (the session's order) | Delayed key (to 6 Oct) | Live key |
|---|---|---|
| CLV, money unconstrained | +6.6% (+6.1 to +7.1), 73 bets a day | +8.2% (+7.7 to +8.7), 63 bets a day |
| GBP a day, money unconstrained | 214 | 240 |
| GBP1,500 committed a day, nothing recycled | 144 | 162 |
| GBP3,000 a day | 196 | 220 |
| GBP5,000 a day | 212 | 240 |

- Every month was positive both ways: +4.6% to +8.5% now, +6.8% to +10.2% with the live key. Held to the result,
  the live-key rule made +9.2% (+5.6 to +12.7).
- Under 4.0 the backs took +5.6% to +8.2% CLV (with volume), so the short prices are not the leak that 1-2 Oct
  (-0.3%) suggests.
- Taking the day's races in order instead of best first, GBP1,500 earns GBP125 a day (with volume).

The live record is the test now.

**Does the expected CLV come true? (ledger `closing-model-forward-1004`).** On the 804 settled live fills of 1-3 Oct
(125 races), at Betfair's settled price (the fill price cut by the reduction factors of non-runners taken out after
it) against the BSP, the CLV was +4.7% staked (90% +2.1% to +7.6%), within a third of a point of the expected CLV
the fills were backed on (-0.3 points, -3.2 to +2.8): first fills +6.0% (+3.2% to +9.1%), top-ups +3.7% (0.0% to
+7.3%). The more expected, the more came true (first fills: 3-4% expected +2.4% level, 6-8% +7.6%, 8-10% +24%), so
nothing says the +3% bar is too low. Non-runners are a cost the closing model does not price: the reductions cut
about one fill in ten and took 1.5 points (+6.2% at the fill price). At the delayed feed's own book price the same
horses look worse (2 Oct: +1.2% staked): the book is stale, so a back sent at it fills better on a drifter and
misses a steamer; the fills are the result. `research/queries/done/fill_calibration.py` re-reads it as days come
in.

**Betting the evening before (the owner's question, 4 Oct: "even betting at 7pm the evening before?").** No history
holds an evening price (Betfair's price files start on the morning of the race), so it cannot be backtested; it is
being recorded instead, read-only: tomorrow's win markets after the last race from 4 Oct, and from 17:00 to 21:30 UK
from 5 Oct (table below). What argues for it: the live rule's earliest backs closed best (08:00 UK fills +11.4%
CLV, 09:00 +7.7%, 10:00 +4.8%, 1-3 Oct). What argues against: the books are thinner the earlier the hour (the best
back held a median of about GBP13 at 08:00), every non-runner taken out overnight cuts a back matched before it by
its reduction factor while the lay at SP is not cut, and the model would price the evening's card, before the
morning's non-runners and going. The non-runners' cost is measured (1-3 Oct, 135 races, `nonrunner_timing.py`):
most are taken out between 08:00 and 12:00 on the day, so a back already carries it in the morning (a mean cut of
4.0% of the price matched at 08:00, 2.2% at 11:00); one matched at 19:00 the evening before carries 4.75%, 0.8%
more than at 08:00. `evening_entry_check.py` reads each recorded evening: every runner's evening price
against its BSP, the trader's own plan at 17:00-21:00 on the evening's book (what the size on offer would have
matched, and its CLV before and after the later reductions), and the same day's morning as traded. Its replication
of the plan is exact: on the books the trader read at its first step it picks the same horses with the same expected
CLV (2 Oct 16 of 16, 3 Oct 19 of 19; run 37191766228). A few evenings decide nothing; the owner decides whether the
trader ever bets the evening before.

**The market record (the owner's ask, 30 Sep: keep all of it for the models).** Read-only, on the UK runner:

| What | How often | Kept in S3 | Loaded nightly into horse_racing.db |
|---|---|---|---|
| Every book the trader reads (`BETFAIR_RECORD=1`) | each minute, from 08:00 UK to 15 minutes before each off | `betfair_live/<day>/books_trader.csv.gz` (its own files: the recorder runs beside it) | `betfair_live_marks` |
| The day's GB/IE win and place markets (live-record.yml) | every 5 minutes, each minute in the last hour, to 21:30 UK | the same | `betfair_live_marks` |
| The catalogue: cloth, stall, jockey, trainer, age, weight, rating, form, headgear, forecast price | once a market | `betfair_live/<day>/markets.csv.gz` | `betfair_live_markets` (matched to race_results) |
| The settled books: BSP, winners, removals and reduction factors | after racing and next morning | in `books.csv.gz` (source `final`) | `betfair_live_marks`, mark `final` |
| Betfair's daily price files (morning and pre-play prices and volumes, BSP, in-play range), which Betfair refuses to GitHub's runners: every file it lists, all markets (72,443 on 1 Oct 2026), one at a time | nightly from 22:30 UTC, stopping by 06:15 (betfair-prices.yml: the last week, then UK/IE racing from 2018, then the other markets from 2018, then the older files, newest first, several nights for the backfill); the UK/IE files after the last race (live-record.yml) | `betfair_prices_raw/` | `betfair_prices`: UK/IE win and place from 2018, at most 2,500 files a night; other markets stay in S3 |
| The live trader's ledger | each order | `trading/live/<day>/ledger.csv` | `live_orders` |
| Tomorrow's GB/IE win markets, the evening before (the owner's question of 4 Oct) | every 15 minutes from 17:00 to 21:30 UK beside the trader, then every 10 minutes for 45 minutes after the last race (live-record.yml) | `betfair_live/<day>/books_evening.csv.gz` (its own files, under the racing day) | not yet: read by `research/queries/evening_entry_check.py` |

`betfair_live_marks` keeps, for each runner, the book nearest to 08:00-12:00 UK and to 120, 60, 30, 15, 10, 5, 3
and 1 minutes before the off, the last book before the off, and the settled one. The full-resolution books stay in
S3. One runner serves both jobs, so the recorder gives way to the trader: its 06:55 UTC run records the whole day
only when the trader is not live, and its 10:10 UTC run takes the day from the end of the morning session.
daily-results.yml, the one job that writes the database, does the loading; a failed load never costs the night's
results.

The old `execute.yml` workflow (disabled by hand on 19 Aug) places live bets by default and has known
faults (real bets never settle, the stop-loss never counts them, the timing window is unused, the
certificate path is never expanded). It should stay disabled.
