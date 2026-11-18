# racing² - Ante-post trading on Betfair: the plan

18 Nov 2026. Status: plan only. Nothing here is built, backtested or traded yet, and every figure quoted is from a
file named beside it. Nothing places an order: a paper test comes first, and real stakes only on the owner's word.

## 1. The trade in one line

Take a position in a big race weeks or months ahead (back, or lay, on Betfair's `ANTEPOST_WIN` market). Re-price it
after each day's racing. Close it on race day: lay the horse at the Betfair SP in the day's WIN market (or at the
race-day morning price), so the result is the ante-post price against the SP.

For a back of stake S at ante-post price A, closed by a lay at the SP (B) in the day's WIN market:

- the horse runs: the position is greened at A against B, worth about `S * (A/B - 1)` before commission;
- the horse does not run: the WIN-market lay never exists and the ante-post stake is lost (`-S`), if Betfair's
  ante-post rule is all-in run or not (to be confirmed: section 3).

So the expected value of a back is

```
EV / S  =  p_run * E[A / B | runs]  -  1  -  costs
```

That splits into exactly the two models you named:

1. **Turn-up model:** `p_run`, the chance the horse runs in this race, given what is known today.
2. **Pricing model:** `E[B | runs]`, the SP the horse will start at if it runs. This is the standard BFSP model
   (`train_bfsp.py`, `predict_bfsp_today.py`), applied to a field that is not yet known.

The trade is a CLV trade (the ante-post price against the SP), not a bet on the result. That matches the live trader
(TRADING.md): a back laid at the SP. The difference is that the gap between entry and exit is weeks, not hours, and a
non-runner costs the whole stake.

## 2. What the repository already has

| Piece | Where | State |
|---|---|---|
| Daily snapshot of every GB/IE `ANTEPOST_WIN` market (catalogue + book) | `betfair_recorder.py antepost()`, run once a day by `live-record.yml` | Recording since 7 Oct 2026: `books_antepost.csv.gz` on 25 days to 31 Oct (`reports/open_close/data.md` 7.6) |
| Liquidity read | Census of next-24h markets | `ANTEPOST_WIN`: median 2 markets with a median GBP16,678 matched, max GBP182,765, on the 6 days it appeared (`reports/open_close/markets.md` 1, row 13) |
| Race-day BSP and books | `betfair_prices` (2018 on), `betfair_live_marks` (from 30 Sep 2026) | Held |
| The price model | `train_bfsp.py`, `model/` | Live, predicts log(BFSP) for a declared field |
| Card the evening before | HRB 48-hour declarations (`daily_predictions.py` "The night before") | Live |
| Lay at the SP | `trading/exchange.py` (`MARKET_ON_CLOSE`, sized by liability) | Live, in the horse trader |
| Strategy registry, risk path | `desk/strategies/horse_racing/`, `desk/risk.py` | Live; no ante-post entry yet |

What is **not** held (no data found in the repository):

- Big-race **entries** by stage (initial entry, forfeit stages, 5-day, final declarations) for past races. This is the
  main gap for the turn-up model.
- Ante-post price history before 7 Oct 2026. Betfair Historic Data (`betfair_historic.py`) may hold `ANTEPOST_WIN`
  markets for horse racing from Jan 2024 (the greyhound bundle carried one GB `ANTEPOST_WIN` market:
  `reports/open_close/data.md` 7.5); not checked.
- Betfair's ante-post rules as text. Their rules pages refused an automated read (HTTP 403, 18 Nov 2026).

## 3. Phase 0 - facts and data (first, about a week)

Nothing is modelled until these are answered from source.

1. **Betfair's rules for these markets.** Each market's own rules come with its catalogue
   (`MARKET_DESCRIPTION.rules`, `bspMarket`). Read them from the recorded markets or one read-only catalogue call, and
   write down: how a non-runner is settled (all-in run or not; whether reduction factors apply after final
   declarations: Betfair's rules page says no non-runner reductions are made to ante-post bets, per a search summary
   of https://www.betfair.com/aboutUs/Rules.and.Regulations/, unverified); when the market suspends (at a declaration
   stage, or the off); whether it carries a BSP; when it is voided (race abandoned or re-opened, conditions changed).
   These rules decide the exit (section 6), so they come first.
2. **Inventory of the recorded snapshots:** races, runners, days to the off, matched money, spread at each price
   band, and how many races have already been run so the ante-post price can be set against the realised SP and
   whether each horse ran.
3. **Historic Data:** whether the Horse Racing bundle holds `ANTEPOST_WIN` markets, with each runner's removal date
   (status `REMOVED`). If it does, that file gives two years of ante-post prices **and** the turn-up label (listed,
   then removed or ran) for free.
4. **Entries source:** whether HRB, the Racing Post or the BHA publish big-race entries and forfeit stages that can be
   read within their terms, and how far back.

Write-up: `reports/antepost_data.md`, with rows, races, date range, nulls in key fields, and the races dropped and why.
**Gate:** enough past races with an ante-post price, a race-day SP and a ran-or-not label to fit and test on held-out
seasons. If the count is too small, the plan goes to recording and paper only until it is not.

## 4. The turn-up model: `p_run`

**Target:** the horse runs in race R (1) or not (0), read at day t before the off. A horse that wins a trial and is
then pulled is a 0; a horse rerouted to another race at the meeting is a 0 for R.

**Universe:** every horse listed in the Betfair ante-post market on day t (and, once an entries source is held,
every entry at each stage).

**Features (known at t only):**

- Time: days to the off; stage passed (entry, each forfeit, 5-day, final declarations).
- Market: the horse's ante-post price and rank; its change over the last 7 and 28 days (a drifter is often a doubt).
- Horse: days since last run; run since the last snapshot and how it went (race-centred pounds beaten, `lbsc`);
  ground and trip record against the race's; rating against the race's band (in handicaps, below the cut it does not
  get in); age and any known target.
- Connections: the trainer's past ante-post-to-runner rate in big races; the number of the trainer's or owner's
  horses in the same race (one of several is less likely to go); entries in rival races the same week.
- Race: going forecast against the horse's ground record; field-size limit (safety factor).

**Model:** a discrete-time hazard (the chance of dropping out between snapshots), so `p_run` updates each day and a
horse's probability falls smoothly as stages pass. Gradient-boosted classifier as the comparison.

**Scored on:** held-out seasons, log loss and calibration by days to the off; the base rate (share of listed horses
that ran, by days to the off) is the baseline it must beat.

## 5. The pricing model: `E[B | runs]`

The live model needs a declared field, a draw, a jockey, a weight and the going. Weeks out these are unknown, so:

1. **Sample fields:** draw each listed horse in or out from `p_run` (Monte Carlo, a few hundred fields).
2. **Fill the unknowns** with their distribution: the going from the course's record for the date and any
   forecast; the draw uniform over the field size (flat); the jockey as the horse's usual rider; the weight from the
   race's conditions and the horse's rating (handicaps: the published weights once out).
3. **Price each field** with the served BFSP model, as `predict_bfsp_today.py` does, and keep each horse's predicted
   log(BFSP) conditional on it being in the field.
4. **Average** to `E[log B | runs]` and its spread, which feeds the EV and the stake.

Two cheaper starts to test first:

- **Market anchor:** the closing-price model in `model/race_book.py` already blends the market's price now with ours.
  The ante-post version: predict the race-day SP from the ante-post price, days to the off and the market's movement
  alone. That may carry most of the signal, and it is the baseline the full model must beat.
- **Form only:** the served model on the horse's current form, with the field held at the current ante-post list.

**Judged on** (the owner's rule, 18 Nov 2026): error against the realised BFSP (paired error, Brier skill,
concordance), on horses that ran, by days to the off. CLV is reported beside it, not used as a veto.

## 6. Entry, daily update and exit rules

**Entry (back):** `p_run * A / E[B | runs] - 1 - costs` above a threshold, with the threshold set on held-out seasons
and the uncertainty of `E[B]` taken into account (a lower quantile, not the mean). Lays the same way, the other
side: a lay at A is worth `1 - p_run * A / E[B|runs]`, and a non-runner is a full win for the layer, so the lay side
may be where the model's turn-up edge shows most.

**Costs:** the spread paid on entry (the ante-post book is wide: section 2), commission on the net winnings of each
market, and the stake held for weeks.

**After each day's racing** (a job after the nightly load, its own workflow, never the live trader's):

1. Re-score `p_run` for every listed horse (who ran today, how it went, entries or declarations that came out).
2. Re-price `E[B | runs]`.
3. For each open position: **hold** while the edge is above zero; **add** up to the stake cap if the edge has grown and
   the price is still there; **cut** (lay in the ante-post market, greened) if the edge has gone negative and the
   price allows it; **close** early if the price has come in past `E[B]` (the CLV is banked now, and the non-runner risk
   with it).
4. A dated ledger row for every decision with the inputs that made it.

**Exit on race day** (the default, to be confirmed against Phase 0's rules):

- Once the horse is a declared runner, the non-runner risk is mostly gone. Lay it in the day's WIN market at the SP
  (`MARKET_ON_CLOSE`, for its winnings, as the morning steamer is laid), or at the race-day morning price when the
  model says the SP will be shorter than the morning price.
- If the ante-post market stays open to the off and its rules settle like the WIN market, close inside it instead
  (one market, one commission).
- A horse taken out before declarations: the stake is lost. The model has priced that in; the record shows it.

## 7. Testing

1. **Register first:** each test written in `reports/research_ledger.jsonl` before its run, with its reading, and a
   pass needs the deflated Sharpe at least 0.95 with every trial counted (`research/overfit.py`) as well as an interval
   above zero (CLAUDE.md, "A new rule is read against every rule tried").
2. **Historic replay** (if Phase 0 finds the history): walk-forward by season; the replay sees only what was known on
   each day; fills at the recorded best price with its size, never the mid; non-runners as losses.
3. **Paper forward test** from the recorded snapshots, starting this jumps season: the King George (Kempton, 26 Dec),
   the Cheltenham Festival and the Grand National, then the 2027 flat (Guineas, Derby, Royal Ascot). A desk entry
   `desk/strategies/horse_racing/horse-antepost.json` with its own order-reference prefix and stop keys, stage paper.
4. **Small real stakes** only after the paper test passes its gate and the owner says so, through `desk/risk.py` with
   the `DESK_*` limits.

## 8. Where it lives (keeping the live horse trader untouched)

- A new package `antepost/`: `rules.py` (the market rules, from Phase 0), `data.py` (snapshots, historic, entries),
  `turnup.py`, `price.py` (the field sampler over the served model, imported read-only), `book.py` (positions, the
  daily update, exits), `track.py` (paper ledger and settlement).
- Its own workflow `antepost.yml`: an extra snapshot or two a day if the daily one proves too coarse, the daily
  update after the nightly load, settlement after each big race. Nothing added to `live-trade.yml`, `live-record.yml`,
  `auto_trade.py`, `trading/` or `model/` (CLAUDE.md, the 2 Nov 2026 rule; `tests/test_workflow_guards.py`).
- Tests beside it under `tests/test_antepost*.py`.

## 9. Risks

- **Liquidity:** a median two markets a day with matched money in the census (section 2). Stakes will be small and the
  spread large; the edge has to clear it.
- **Non-runners:** the whole stake on every horse that does not run; connections know more than any model.
- **Few races:** a few dozen big races a year, so a real test takes seasons, and the deflated Sharpe will be hard to
  pass on one.
- **Money held for months**, and correlated positions (several horses in one race; one trainer's string).
- **Rule surprises:** a race re-opened, a horse switched race, voids. Phase 0 settles what Betfair does.

## 10. Order of work

| Step | Output | Check |
|---|---|---|
| 0 | `reports/antepost_data.md`, `antepost/rules.py` | Rules quoted from the market description; row counts, races, date range, nulls |
| 1 | Turn-up model, held-out seasons | Log loss and calibration against the base rate, by days to the off |
| 2 | Price: market anchor, then form, then the field sampler | Error against the realised BFSP against the anchor |
| 3 | EV, entry and exit rules, replay | Registered test; CLV and P&L after spread, commission and non-runners |
| 4 | Paper forward test (desk entry, `antepost.yml`) | Ledger reconciles to the recorded books and the SP |
| 5 | Small real stakes | Owner's decision |
