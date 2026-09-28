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
- **Scale is set by liquidity.** The median morning volume on a runner is £495; a tenth of it is under £50
  for half the runners. Stakes of a few pounds to a few tens of pounds per bet are what the morning market
  takes without moving.

## The paper trader (built)

`auto_trade.py` with the `trading/` package reads Betfair's live markets and prices (read-only), plans each
race from the model's prices, and simulates every order against the order book as it stands: a back
fills at the prices offered at or above its limit, the rest lapses, and each fill is closed with a
simulated lay at BSP. Nothing is placed on the exchange: the package has no order-placing code.

- Settings: `trading/config.json` (strategy `rule`, level £5 stakes, trade out at BSP on the fill, window
  08:00-11:00 UK, limits), overridable by `TRADING_*` variables.
- Limits on every simulated order: stake, race, daily turnover and bets, daily stop-loss on settled
  results, price range, the market's matched volume, half the size offered at the price.
- Kill switch: the S3 object `trading/STOP` in the predictions bucket.
- Ledger: every simulated order and settlement, `out/trading/` and `s3://<bucket>/trading/paper/<date>/`.
- `python auto_trade.py --until 11:05` in the morning; `python auto_trade.py --settle` in the evening
  settles the day from the closed markets (winner, non-runner, BSP) and emails a summary.
- Tests: `tests/test_trading_bot.py`, `tests/test_trading_staking.py`.

**The paper trader is the forward test.** It measures the edge at the prices actually on offer at the
moment of decision, on the card as known that morning. The criterion to go further, fixed now: after at
least three weeks and 1,000 simulated trades, the stake-weighted CLV and the settled traded-out return
both above zero with their 90% intervals clear of it.

## Live execution: not built, the owner's decision

Placing real orders was started and then stopped: the session's permission system classed automatic
order placement as a real-world transaction and blocked it, and that is the owner's call, not the
code's. Nothing here can place a bet. To go live later the owner would need to:

1. **Decide and grant it**: allow this work to build and run live order placement (a permission rule in
   Claude Code's settings, or an explicit instruction in a session where it is permitted).
2. **Betfair API access**: a live application key (requested) stored as the GitHub secret
   `BETFAIR_APP_KEY`, never in code or chat; the delayed key serves the paper trader meanwhile.
3. **Non-interactive login**: a self-signed certificate uploaded to the Betfair account (Security
   settings, automated betting program access) and its `.crt` and `.key` stored as the secrets
   `BETFAIR_CERT` and `BETFAIR_KEY`.
4. **A bank and limits** chosen by the owner: the money in the account, the largest stake, race and daily
   exposure, and the daily stop-loss.
5. **The forward test passed** (above), on the model that would trade.

The old `execute.yml` workflow (disabled by hand on 19 Aug) places live bets by default and has known
faults (real bets never settle, the stop-loss never counts them, the timing window is unused, the
certificate path is never expanded). It should stay disabled.
