# Autonomous trading on Betfair from the opening to the closing price: the programme (1 Nov 2026)

The owner asked (1 Nov 2026) for autonomous trading on Betfair that trades the price from its opening to its close,
hedged so that no result has to be right, with three agents: research (which markets), data (the best data for them),
modelling (prices by the best quantitative practice). Later the same day: "Evaluate all markets on Betfair." This note
ties the four reports together and says what happens next. Nothing in this change places a bet.

| Report | What it holds |
|---|---|
| `markets.md` | The ranked markets for a hedged pre-off trade: when a tradable open exists, the pre-off money, the evidence the move is predictable (from the market alone, only with a model, or not at all), the round-trip cost, the next tests |
| `all_markets.md` | Every one of Betfair's 35 event types and every market type inside each, a verdict a row, from the complete census of October 2026 |
| `data.md` | What open and close data is held, per market, with the real counts; the standard open-close panel (`sources/open_close.py`); the best data to obtain next, with prices and licences |
| `model.md` | The drift models: predict log(close / open) from the opening book, walk-forward by month, scored as a hedged trade after 2% commission and a tick each end, holdout read once |

## The result

1. **One market is proven and live: UK and Irish horse win.** The live rule is already this trade: back in the morning
   where the model's BSP forecast sits well under the price, lay at the BSP. GBP47,274 staked 19-31 Oct at +6.15%
   (research ledger `staking-1031`; TRADING.md). The edge is the model's: on the market alone the morning price
   carries no drift worth taking on the back side, and the drift that does exist (runners at 8-21 lengthen 2-5% from
   08:00 to the off) is about half the book's spread (`markets.md` line 6 of the summary; run 41291139483 sections A2,
   A3).
2. **Only racing has a close with no spread.** The Betfair SP is a single price; in every other sport the close is the
   last trade before the market turns in play and the exit crosses a tick or rests an order. That makes racing and
   greyhounds structurally the right place for a hedged trade (`markets.md` section 2).
3. **One market has a measured market-only drift to test: GB greyhound win.** On 1,863,552 runners (Betfair Historic
   Data, Dec 2025 to Sep 2026) the first traded price drifts out to the BSP by a median 2.5% at 3-5 up to 13% at 13-21,
   shortening on only 33-45% of dogs (run 41291139483 section D). But the book holds GBP0 until five minutes before
   the off: GBP34 at T-5, GBP380 at T-3, GBP1,993 at T-1, GBP2,377 at the last book (run 41294133067 section 3b). The
   open worth trading exists only in the last three minutes, the drift left at T-1 is 2-6% at 5-51, and a tick costs
   2-5% there. The test is a lay at T-3 or T-1 greened at the SP on the 27 recorded days (`all_markets.md` 2.2).
4. **Two markets need data first: football and cricket.** Football is the deepest non-racing market a day ahead
   (GBP437,775 matched on match odds and GBP175,058 on Asian handicap within the top 200, a median day), but its
   pre-play close is an unbiased forecast and the maker side earns about zero after fees (Whelan 2025), so the move
   is team news, not drift. Cricket carries GBP6.8m a day ahead in about 17 match-odds markets, with the toss as the
   information event. Neither has a first-traded price held: the free BASIC Historic Data plans unblock both.
5. **Everything else is parked or never**, each for a stated reason in `all_markets.md`: tennis (median next-day
   market GBP5), basketball, ice hockey, baseball, motor sport, rugby, boxing, snooker, darts (no money a day ahead);
   politics and specials (no close); golf outrights are research later (GBP507k on 3 markets, ticks 3-7%); 11 event
   types had no October markets at all.

## What was built

- `research/queries/done/open_close_inventory_1101.py`, run 41291139483: every open and close source held, with
  counts, date ranges and the first drift reads (horses from the recorder's marks, greyhounds from the Historic Data
  bundle, football from the gold benchmarks).
- `research/queries/done/census_all_markets_1101.py`, run 41294133067: the complete census (24 sports, 426 sport x
  market type rows, the busiest events, racing's money on the day at each mark).
- `sources/open_close.py` and `tests/test_open_close.py`: one standard open-close panel across markets (the Historic
  Data tables, the horse price files and the recorder's marks, the recorded greyhound books with the T-3 book as the
  open, football gold, the greyhound price files); idempotent, UTC, 8 tests.
- `research/open_close/` and `tests/test_open_close_model.py`: the drift models (no-change, band OLS, LightGBM, a
  conditional-logit book), the hedged-trade scorer (2% and 5% commission, a tick worse each end, bootstrap by day),
  the shuffle leak test, walk-forward by month, a holdout read once; 18 tests.
- `research/queries/done/open_close_horses_1101.py`, run 41292736133: the models on UK and IE win (market-only and
  with the forecast), GB greyhounds, and the pre-registered lay cell on the recorder's books. Its figures are in
  `model.md`.

## The owner's actions (nothing paid)

1. At historicdata.betfair.com, signed in with the trading account, "buy" the free BASIC plans for Soccer and
   Cricket (and Golf if wanted). `betfair_historic.py` fetches each sport nightly once the account holds it, and
   the panel's reader is ready (`data.md` section 6.3).
2. Keep the greyhound and horse recorders running (`live-record.yml`): the recorded books are the only takeable
   greyhound open and the only true horse open held.
3. Decide whether a GBP-paid odds feed is wanted for true opening lines in football and tennis (`data.md` 6.3-6.4
   lists the options and their published prices, to re-check before buying). Nothing proposed needs it.
4. The licences: Sackmann and TML tennis data are CC BY-NC-SA and football-data.co.uk is for private individuals,
   as the repository records them (`data.md` 6.5); check before any of them trains a model that stakes money.

## The path to a trader

The horse trader already trades this way and stays as it is. For any new market the order is the programme's
(`reports/market_research_programme.md`): the market-only test on recorded books, then the panel and a model, then a
paper record on the recorder's books, then small stakes, then the rule, each gate written to the research ledger
before the next starts. The greyhound T-3 lay test is first and needs no new data; football and cricket wait on the
free plans. No live step is taken without the owner's say.
