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
| UK/IE win | GBP11.8k (UK), GBP4.7k (IE) | Betfair files from 2017, HRB from 2010, live books from 1 Oct | BFSP model + closing model | 6, live | entry time and top-ups: 1 Oct replayed, the live rule best with the account's money; 2 Oct next. Then the model and the closing model |
| UK/IE place | GBP1.8k (UK), GBP0.7k (IE) | HRB place BSP from 2010; Betfair files from 2017 (place PPWAP sound; the other place columns placeholders) | none; the win-implied place chance | 1 passed at the SP (2 Oct); 5 failed at pre-play prices (ledger `place-rules-preplay-1002`): decided before the off only the long-shot lays keep an edge, +0.2% of the liability. 3 closed (3 Oct, ledger `place-model-stage3-1003`): our forecast adds nothing beyond the place SP (+0.00002 log loss) and no rule decided before the off finds value at the place SP | closed (3 Oct, ledger `place-clv-forward-1002`): stage 4 failed at morning prices. The morning book's best back sits a quarter to a third below the place SP and its best lay half again to twice above it; the win rule's horses lose 12.5% backed there (the field 17.0%). The recorder keeps the books |
| UK/IE in-play offsets | in-running volume | IPMIN/IPMAX in the win files (sound); the place files hold placeholders | none | closed for win (2 Oct): every cell negative, both directions, train and test | place only with a live in-play recording, not planned |
| SP bias (back or lay at the SP by segment) | the SP pool | HRB from 2010 | none | closed (2 Oct): the two cells chosen on 2010-17 did not hold on 2018-26Q1 | |
| Greyhounds, UK and Australia (win, place) | GBP1.3k | files arriving from 2018 (the nightly archive) | none | 1 waits for the files | edge scan first; form data (GBGB results) only if the market alone shows something |
| Australia racing (win, place) | GBP1.2k pre-play | files arriving | none | 1 waits | pre-play prices only (the files' "morning" includes post-race trading there); trading would run overnight UK time |
| AvB match bets | none offered | none | from our win probabilities (who finishes ahead) | dropped (3 Oct census: no match bets on GB/IE racing) | the census keeps running each evening |
| Other place markets (2/3/4 TBP, each way) | not known | none: no price files | the place model, once built | not started | recording first (the census lists 65 OTHER_PLACE and 39 EACH_WAY markets on 3 Oct) |
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

Done 2 Oct (ledger `edge-inplay-offsets-1002`, `edge-bsp-bias-1002`, `market-census-1003`):

1. Scan 1, in-play offsets: closed for the win markets. The place files' in-running fields are placeholders (field
   check, research-query run 37070286949), so the place cells were void.
2. Scan 2: the place pocket holds on the clean years, and the place SP misprices against the win market out of sample.
   The SP-bias segments did not hold.
3. The census: no AvB markets.

Next:

1. Done: the place files' pre-play WAP is sound, the other place columns are not (ledger `place-fields-check-1002`).
   At those pre-play prices the place rules lose their edge (ledger `place-rules-preplay-1002`).
2. Done 3 Oct: the place market's morning prices and the place price at each live win back, against the place SP
   (ledger `place-clv-forward-1002`). Neither side of the morning book closes in our favour; the place market is
   closed. The T-15 rules decide nothing on two days.
3. Done 3 Oct: the place model from our win probabilities (stage 3). It adds nothing beyond the place SP, and no rule
   decided before the off finds value there (ledger `place-model-stage3-1003`). Only a closing-line trade is left for
   place: the win rule's horses close shorter than the field there too, and item 2 reads it at the morning price.
4. Done 3 Oct: the win rule's entry time and top-ups on 1-2 Oct (ledger `topup-policy-replay-1002`). No variant beat
   the live rule on both days at the account's money; the short-price bars did worst. Replayed again as days accrue.
5. When the archive has them: the same scans on the greyhound and Australian files.
