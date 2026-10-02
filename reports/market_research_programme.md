# Market research programme (the owner's brief, 2 Oct 2026)

The brief: for every market we can trade, (1) evaluate the market for edges, (2) source the data to model it,
(3) engineer features, (4) compare the models to CLV, (5) backtest, (6) trade. Either kind of edge will do: small
profits at high volume, or medium profits at medium volume.

## How a market moves through the stages

| Stage | What it means here | Passes when |
|---|---|---|
| 1. Edge scan | The market on its own, with no model: does its own price (SP, early price, in-running range) misprice anything, after 2% commission? | A cell chosen on earlier years is positive on later years with its 90% interval above zero, at enough bets a day to matter |
| 2. Data | Everything a model of that market needs, point-in-time, in the database or S3 | Coverage and joins checked, as for the UK/IE win data |
| 3. Features | The inputs, built as drop-in blocks on the cached matrix where they fit | Leak and parity tests pass |
| 4. Model vs CLV | The model's price against the market's closing price (the SP), walk-forward | Positive CLV out of sample, by month |
| 5. Backtest | The tradable rule at prices available when the order would go in, settled as Betfair would | Positive after commission over the development years; then one look at the holdout (from 1 Apr 2026) |
| 6. Trade | Forward record on the recorder's books, then small live stakes, then the full rule | The live CLV is in line with the backtest |

The holdout (1 Apr 2026 on) is read once, at stage 5, for the rule that is going to trade. Every result goes into
`reports/research_ledger.jsonl`.

## The markets

| Market | Liquidity (median traded a runner) | Data held | Model now | Stage | Next |
|---|---|---|---|---|---|
| UK/IE win | GBP11.8k (UK), GBP4.7k (IE) | Betfair files from 2017, HRB from 2010, live books from 1 Oct | BFSP model + closing model | 6, live | improve the model and the closing model |
| UK/IE place | GBP1.8k (UK), GBP0.7k (IE) | the same, place BSP in HRB from 2010 | none; the win-implied place chance | 1 passed (place pocket, 2 Oct) | the years before 2021 as a clean test (scan 2); a place model from our win probabilities (stage 3-4); the rule at T-15 from the recorded books (stage 6) |
| UK/IE in-play offsets (back at the SP and lay in running, or the reverse; orders resting from before the off) | in-running volume | IPMIN/IPMAX in every file from 2017 | none | 1 (scan 1, 2 Oct) | if a cell holds: a model of how far a horse trades in running, from run style and pace (our race-shape features) |
| SP bias (back or lay at the SP by segment) | the SP pool | HRB from 2010 | none | 1 (scan 2, 2 Oct) | |
| Greyhounds, UK and Australia (win, place) | GBP1.3k | files arriving from 2018 (the nightly archive, about four nights) | none | 1 waits for the files | edge scan first; form data (GBGB results) only if the market alone shows something |
| Australia racing (win, place) | GBP1.2k pre-play | files arriving | none | 1 waits | pre-play prices only (the files' "morning" includes post-race trading there); trading would run overnight UK time |
| AvB match bets | not known yet | none: no SP, so no files | from our win probabilities (who finishes ahead) | 1 waits for the census | record the markets once the census shows their code (live-record.yml, each evening) |
| France, USA, South Africa (win) | GBP250-600 | files arriving | none | low priority | |

## Which kind of edge suits which market

- High volume, small edge: strategies hedged at the SP or locked in running (offsets, SP bias, the live win rule) keep
  each day's result close to the edge, so a 1-3% edge on many bets is worth having.
- Medium volume, medium edge: model-driven selections (the win rule, a place model) where our form data adds what
  the market lacks.

## Betfair's Expert Fee and the programme

From 6 Jan 2025 Betfair charges an Expert Fee on top of commission. It is set each week by the account's gross
profit over its last 52 active weeks:

| Gross profit, last 52 active weeks | Fee rate |
|---|---|
| up to GBP25,000 | none |
| GBP25,000 to GBP100,000 | 20% |
| over GBP100,000 | 40% |

The commission already generated is credited against the fee, and past losses build a buffer. Betfair's own FAQ
says the rate is applied to the week's gross profit. One guide reads it as applying only to profit above each
threshold. The difference matters past GBP100,000 a year, so the account's Expert Fee page should be checked before
planning around it. Every strategy's profit adds to the same account total, so the fee is a programme-level cost:
the net figure for a new strategy is its profit less the extra fee it brings.

## This week

1. Scan 1 (`research/queries/edge_inplay_offsets.py`): in-play offsets, UK/IE win and place, 2018 to 2026 Q1.
2. Scan 2 (`research/queries/edge_bsp_bias.py`): the place pocket on 2010-20; backs and lays at the SP by segment;
   the place SP against the win-implied place chance.
3. Tonight: the market-type census (AvB), the 2 Oct settlement, the stake replay on every horse.
4. When the archive has them (about four nights): the same scans on the greyhound and Australian files.
