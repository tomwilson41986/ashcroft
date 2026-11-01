# Open-to-close trading: the data (data agent, 1 Nov 2026)

The owner's ask (1 Nov 2026): autonomous trading on Betfair of the move from the opening price to the close, hedged
so the result does not matter. This note is the data half: what is held for that trade, market by market, in what
shape, with what gaps; one standard panel to read it through; and the best data to obtain next, with prices and
terms where they could be read. The ranking of the markets is the research agent's (`reports/open_close/markets.md`);
this note follows its order: 1 UK/IE horse win, 2 UK greyhounds, 3 football match odds, the rest parked.

Status (1 Nov 2026, evening): the inventory query is merged and running on GitHub's runner (research-query run
41291139483). Until its log is in, every count below that comes from that run is marked **CI pending**; what is
written as held comes from the code and the repository's own earlier notes, each cited. Nothing here is a model
result. Hyphens only; every number traces to a file, a query or a URL.

## 1. What the trade needs, and where each piece is

For a back early, lay at the close trade the data a market needs, a selection at a time, is: the earliest tradable
price (when it stood and how much was matched by then), the price at fixed marks before the off (T-24h, T-60, T-15,
T-1), the close (the BSP where there is one, else the last pre-play trade), the volumes, and the result for settlement
checks. The table says where each piece exists today.

| Piece | UK/IE horses | UK greyhounds | Football | Tennis, the rest |
|---|---|---|---|---|
| Earliest price with its time and money | the recorder's books from 1 Oct 2026 (06:15 UK, every 5 min, each minute in the last hour): `betfair_live/<day>/books.csv.gz`, loaded as `betfair_live_marks` with the 08:00-12:00 UK clock marks (`betfair_recorder.py`, `CLOCK_MARKS`) | the recorder's greyhound books from 6 Oct 2026 (`books_greyhound.csv.gz`; `reports/all_markets_priorities.md` section 4); the Historic Data bundle's first traded price has no time and no volume (`betfair_historic.py`, `_MarketState`: `ltp_first` only) | none held: football-data.co.uk's "open" is the price collected on Friday or Tuesday afternoon, not the first price (notes.txt: "Betting odds for weekend games are collected Friday afternoons, and on Tuesday afternoons for midweek games"); Betfair's own first trade needs the Historic Data Soccer plan the account does not hold yet | none |
| Morning price with volume | `betfair_prices` (the price files, 2017 on for UK/IE): MORNINGWAP and MORNINGTRADEDVOL, an average of what traded, not a takeable price (`betfair_prices.py`, `RAW_COLUMNS`) | the same files for greyhounds, 2018 on, built to `sources/greyhound_prices/prices_<year>.parquet` | n/a | n/a |
| T-24h / T-60 / T-15 / T-1 | the recorder's `off_120 .. off_1` marks (`OFF_MARKS = 120, 60, 30, 15, 10, 5, 3, 1`), Oct 2026 only | the Historic Data tables (`ltp_t_1440`, `ltp_t_60`, `ltp_t_15`, `ltp_t_1`; Jan-Sep 2026, GB/AU/NZ) and the recorder's greyhound books from 6 Oct | the Historic Data soccer tables once held (`football/data.py`, `betfair_closes`: first, T-60, T-1, close) | the Historic Data tables once each sport is held |
| Close | the BSP (price files; the recorder's `final` mark; `betfair_live_marks.bsp`) | the BSP (files, bundle, recorder) | no BSP: the last pre-play trade (`ltp_close`) or the bookmaker close (football-data's "C" columns: "with an additional "C" character following the bookmaker abbreviation/Max/Avg") | no BSP: `ltp_close` |
| Volumes | pre-play and in-play traded a runner (files); market total matched at each mark (recorder) | the same | none in football-data; `tv` only in the Historic Data PRO plan (unverified, section 6) | none |
| Result | WIN_LOSE (files); WINNER/LOSER on the settled book | the same; GBGB results 2018 on (`sources/gbgb/runs_<year>.parquet`) | scores in the gold table; `won` in the Historic Data tables | `won` |

## 2. The inventory (research/queries/open_close_inventory_1101.py)

One read-only script for `.github/workflows/research-query.yml`; each section guarded, a JSON block at the end
(`OPEN_CLOSE_INVENTORY_JSON`) for the modelling agent. What each section reports, and the figure to fill in:

| Section | What it prints | Figures |
|---|---|---|
| A0 | the tables in horse_racing.db with rows, first and last day, columns | CI pending |
| A1 | `betfair_prices` by country, market and year: runners, markets, how many carry a BSP, a morning WAP, in-play prices, a result, a join to a result; medians of the morning and pre-play money and the BSP; the morning WAP -> BSP move by morning-price band and period; the pre-play WAP -> BSP move | CI pending |
| A2 | `betfair_live_marks`: coverage a day (markets, runners, rows at each mark, settled rows with a BSP); the marks held; market money matched at each mark by market type; the top of the book at each mark (two-sided share, best-back size, relative spread) | CI pending |
| A3 | the 08:00, 10:00, 12:00, T-120 .. T-1 and last mids against the BSP by band: median log(BSP/mid), the share that shortened, the IQR and sd of the move; a back at the 08:00 mid laid at the BSP, mid/BSP - 1 a GBP1 stake | CI pending |
| A4 | `betfair_live_markets` by month, type and country | CI pending |
| B1 | `sources/betfair_historic/`: raw files by sport and year; each `markets_<sport>_<year>.parquet`: runners, markets, offs, non-null first/T-24h/T-60/T-15/T-1/close/BSP/result, markets by country and type, the share that went in play | CI pending. Held by the repository's own notes: the owner's bundle, BASIC, Jan-Sep 2026, GB/AU/NZ greyhounds, every market type, 261,008 files (`sources/betfair_imports.json`) |
| B2 | `sources/football/gold/matches.parquet`: coverage by season of the benchmark open and close for 1X2, O/U 2.5, AH and of Betfair Exchange's own; which bookmaker supplies each (last 12 seasons); the open -> close move on the home price and the over; `odds_long`, `betfair_market_map`, `dq/latest.json` | CI pending. By `reports/football_model.md` section 2: the open is Pinnacle (91,766 matches) or Betfair Exchange (15,414); the close Pinnacle to 2024/25 (153,336) and Betfair Exchange after (15,069) |
| B3 | `sources/tennis/`: Sackmann/TML matches; tennis-data.co.uk odds table with its columns, and whether any column names an opening price | CI pending. By `sources/tennis.py`: tennis-data.co.uk holds closing prices (Pinnacle, Bet365, the market's maximum and average; Betfair's own to 2014): **no opening price exists in any tennis table held** |
| B4 | `sources/gbgb/runs_<year>.parquet` row counts; `greyhound_prices/prices_<year>.parquet`: rows, win and place markets, countries, BSP and morning WAP coverage, the morning and pre-play money, the morning WAP -> BSP move by band and country | CI pending. By `reports/greyhound_model.md` section 4: the morning WAP prices 14.7% of runners; morning volume GBP0 to the 75th percentile |
| B5 | openfootball, API-Football, Punting Form tables | CI pending. None of the three carries a price (`sources/openfootball.py`, `sources/api_football.py` docstrings) |
| C1 | every file under `betfair_live/<day>/` by name: days, first and last day, total and median MB (the tags: `books`, `books_trader`, `books_greyhound`, `books_other_place`, `books_antepost`, `markets_*`, `census_sports.json`) | CI pending |
| C2 | the census of every sport, Oct 2026: by sport the markets a day and the money matched on the top 200 markets (total, median market, the concentration total/200 over median); sport x market type: markets a day and the money matched on the type within the top 200 | CI pending |
| D | the Historic Data tables: first traded -> close by first-price band, market type and country; T-24h/T-60/T-15/T-1 -> close; first -> BSP where there is one; the close against the result by band; how often a price stands 24 h out | CI pending |

Local checks (this session, the S3 keys refused): the DB sections ran against a synthetic SQLite database built with
the real DDL (`betfair_prices.ensure_table`, `betfair_recorder.ensure_tables`), and the S3 sections against a local
folder of synthetic tables in the real schemas with the module's S3 helpers monkeypatched; 0 failed sections. The
synthetic numbers are not reported anywhere.

## 3. The panel (sources/open_close.py)

One schema across markets, one row a selection a market:

| Column | Meaning |
|---|---|
| sport, country, market_type | Betfair's names where the source has them (WIN, PLACE, MATCH_ODDS, OVER_UNDER_25, ASIAN_HANDICAP); countries as ISO-2 (uk -> GB, ire -> IE, AUS -> AU) |
| market_id, event_id, event_name, selection_id, selection_name | the source's identifiers; a price file has no market id, so `market_id` is `<event id>:<type>`; a football match's `event_id` is the gold `match_id` |
| off_time_utc | the scheduled off or kick-off, tz-aware UTC (the price files' UK wall-clock times converted DST-aware) |
| n_runners | selections not removed |
| open_price, open_time_utc, open_matched | the earliest price the source holds, when it stood, the money matched by then (only the recorder's books give a time; the Historic Data BASIC files give neither time nor volume) |
| p_t24h, p_t60, p_t15, p_t1 | the price standing at each mark before the off, where the source has it |
| close_price | the source's close: the BSP where the market has one, else the last pre-play trade, else the benchmark bookmaker close |
| bsp | the Betfair SP where the market has one |
| pre_play_matched | the money matched before the off: a runner's in the price files, the market's in the recorder's books |
| result | 1 won or placed, 0 lost, empty where unknown or not settled here (Asian handicap) |
| source | which reader made the row |

Readers, each a function of a `Store` or a database path returning that frame, a source not held giving an empty
frame and never an error:

| Reader | Source | Open | Marks | Close | Notes |
|---|---|---|---|---|---|
| `read_betfair_historic(store, sport)` | `betfair_historic/markets_<sport>_<year>.parquet` | `ltp_first` (no time, no volume) | T-24h, T-60, T-15, T-1 | `ltp_close`; `bsp` where set | sports: greyhound_racing, football (the service's "soccer"), tennis, cricket, darts, snooker |
| `read_horse_prices(db_path)` | `betfair_prices` | MORNINGWAP with MORNINGTRADEDVOL | none | the BSP | result from WIN_LOSE; pre_play_matched = PPTRADEDVOL |
| `read_horse_marks(db_path)` | `betfair_live_marks` + `betfair_live_markets` | the 08:00 UK mid (back1/lay1), its poll time, the market's matched money then | the `off_60`, `off_15`, `off_1` mids | the last pre-off mid; `bsp` from the settled mark | the alternative open the brief asked for; a one-sided book gives no price |
| `read_football_gold(store)` | `football/gold/matches.parquet` | the benchmark open (`football.data.benchmarks`) | none | the benchmark close | one row an outcome (H/D/A, Over/Under 2.5, AH home/away with the line); `source` names the books, e.g. `football_gold:PS>PSC`; AH result left empty |
| `read_greyhound_prices(store)` | `greyhound_prices/prices_<year>.parquet` | MORNINGWAP with its volume | none | the BSP | as the horses' files |
| `read_recorded_books(store, sport, tag)` | `betfair_live/<day>/books_<tag>.csv.gz` + `markets_<tag>.csv.gz` (the greyhound record: tag `greyhound`) | the first two-sided pre-off book the record holds, its time, the market's matched money then | T-60, T-15, T-1 through `betfair_recorder.marks` (the nightly load's own picks) | the last pre-off mid; the BSP as the market reconciled it | the open for greyhounds the coordinator asked for; the same reader reads the horse record with tag `""` |

`build(store, sport, db_path=None)` concatenates the sport's readers and writes `sources/open_close/panel_<sport>_
<year>.parquet` (the year of the off), overwriting the same keys, so a rerun is safe. `horse_racing` takes the price
files and the recorder's marks from the database; `greyhound_racing` the Historic Data tables, the price-file tables
and the recorded books; `football` the gold benchmarks and the Historic Data soccer tables; the other sports the
Historic Data tables alone.

    python -m sources.open_close --build --sport greyhound_racing                      # the S3 store (CAPTURE_BUCKET)
    python -m sources.open_close --build --sport horse_racing --db horse_racing.db
    python -m sources.open_close --build --sport all --root /tmp/x --db horse_racing.db  # a local sources/ folder
    python -m sources.open_close --report

It is not wired into `source_data.py`: that nightly job builds the Historic Data tables after its source loop
(`.github/workflows/source-data.yml`, the "Betfair Historic Data tables" step), so a panel built inside the loop
would read them a night stale, and the job runs without horse_racing.db. The panel is a step of its own, run after
`betfair_historic.py --build` wherever the database and the store are both at hand (the research-query runner, or
the UK server).

Tests: `tests/test_open_close.py`, 8 tests, local folder store and a temporary SQLite database, no network: the
schema from each source's own layout, the UTC conversion (14:30 UK in July -> 13:30 UTC), the removed runner left
out of `n_runners`, the one-sided book giving no open, the first two-sided book as the greyhound open with its time
and money, the football results settled from the scores, the build writing a parquet a year and a rerun overwriting
it, a sport with nothing held building nothing, the CLI from a local root. `pytest tests/test_open_close.py -q`:
8 passed (1 Nov 2026); with `tests/test_sources.py` beside it, 35 passed.

## 4. Per-market data position (the research agent's order)

### 4.1 UK/IE horse win (live)

Held: the price files 2017 on (`betfair_prices`), the recorder's books from 1 Oct 2026 with the 08:00-12:00 UK marks
and T-120 .. T-1, the live key's feed from 6 Oct (`CLAUDE.md`, `BETFAIR_LIVE_APP_KEY`). The panel gives two opens
for the same horse: the morning WAP (an average, with volume, nine years deep) and the 08:00 mid (takeable in
principle, one month deep, with the market's matched money then). The gap the CI run measures: how many runners carry
a two-sided book at 08:00, and the matched money behind it (A2), and the 08:00 mid -> BSP move by band (A3).

Missing: the true first price with its time before 06:15 UK (the record starts with the session; the evening before
is unformed, ledger `evening-before-1007`), and the traded-volume ladder (the live key's `EX_TRADED` projection is
read by the trader but not recorded: `betfair_recorder.BOOK_FIELDS` carries `runner_total_matched` and the three best
prices a side, no ladder). Next data: section 6.1.

### 4.2 UK greyhounds (research next)

Held: the price files 2018 on (`greyhound_prices/prices_<year>.parquet`), GBGB results 2018 on, the owner's Historic
Data bundle Jan-Sep 2026 (first traded, T-24h/T-60/T-15/T-1, close, BSP, result; GB/AU/NZ, every market type), the
recorder's GB/IE greyhound books from 6 Oct 2026. The research agent's test (best back and lay at the first recorded
book and at T-60/T-10/T-1 against the BSP, on the recorded GB books from 6 Oct) reads straight off
`read_recorded_books(store, "greyhound_racing", "greyhound")`: `open_price` is the first two-sided book's mid (the
best back and lay are the recorder's `back1`/`lay1` at that poll; the panel keeps the mid, the day files keep both
sides), `p_t60`/`p_t15`/`p_t1` the mids at the marks (T-10 is a recorder mark, `off_10`, not a panel column; add it
to `MARK_PRICE` if the test wants it), `bsp` the close. Section D of the inventory gives the bundle's first -> close
move by band, country and type as the deeper read.

Missing: the first trade's time and volume in the bundle (BASIC files hold the last traded price each minute; the
time of the first `ltp` message is in the raw file and `betfair_historic._MarketState` could keep `pt` of the first
trade, a small change outside this note's paths); any greyhound data before 6 Oct with a takeable open. Next data:
section 6.2.

### 4.3 Football match odds (research next)

Held: football-data.co.uk 1993/94 on with the benchmark open and close a market (`football/gold/matches.parquet`,
`football.data.benchmarks`: the open Pinnacle or Betfair Exchange, the close Pinnacle to 2024/25 and Betfair Exchange
after, Pinnacle's public feed unreliable from 23 Jul 2025, `football/data.py`, `PINNACLE_UNRELIABLE`). The
football-data "open" is the Friday or Tuesday afternoon collection (notes.txt), so the panel's `open_price` for
football is a price a day or two before kick-off, not the market's first price, and `open_time_utc` is empty. No
Betfair soccer files are held: `betfair_market_map.parquet` is built only when `markets_soccer_<year>.parquet` tables
exist (`football/data.py`, `build`), and the account holds no Soccer plan on the Historic Data service yet
(`reports/all_markets_priorities.md` section 4: "Owner: buy the free BASIC plan for each sport").

Missing: Betfair's own first traded price, the marks and the pre-play close for football (the Soccer BASIC plan,
free), and any volume at the open. Next data: section 6.3.

### 4.4 Tennis and the rest (parked)

Tennis: Sackmann/TML statistics and tennis-data.co.uk closing odds (no opening price exists in any tennis table held;
`sources/tennis.py` docstring; B3 confirms from the columns). No Betfair tennis files (no Tennis plan held). Cricket,
darts, snooker: nothing but the census. Next data: section 6.4.

## 5. What the CI log will fill (the figures this note is missing)

From run 41291139483, into section 2's table and a new section 7:

1. `betfair_prices` rows and markets by country, market and year; the share with a morning WAP; the median morning
   and pre-play money; the morning WAP -> BSP move by band (A1).
2. `betfair_live_marks`: days held, runners a day, the marks present, the share of runners with a two-sided book at
   08:00 and the market money then, the median relative spread at each mark (A2); the 08:00 mid -> BSP by band and
   the greened trade's mid/BSP - 1 (A3).
3. The Historic Data tables held (sport, year, runners, markets, non-null marks and close) (B1); the first -> close
   move by band, country and type, and T-24h/T-60/T-15/T-1 -> close (D).
4. Football gold coverage by season and by book (B2); the tennis columns (B3); the greyhound tables' counts and the
   morning money by country and year (B4).
5. The recordings under `betfair_live/` by tag (C1) and the census table sport x market type (C2), which also fills
   the research agent's section 3.4 table.

## 6. The best data to obtain next, per market (sourced; every price to re-check before buying)

### 6.1 Horses

- **The traded-volume ladder from the live key.** The trader reads the live key (`live-trade.yml`); the recorder keeps
  `runner_total_matched` and the three best prices a side, not the traded ladder (`betfair_recorder.BOOK_FIELDS`).
  Betfair's `listMarketBook` price projection `EX_TRADED` returns the traded volume at each price; adding it to the
  recorder's read costs a projection weight, not money (Betfair developer docs, Application Keys page, read 1 Nov
  2026: https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687105/Application+Keys;
  the projection weights are on the Betting API's listMarketBook page, not fetched here).
- **A true opening timestamp: the Exchange Stream API.** The Application Keys page (read 1 Nov 2026) says the Stream
  API is available on both keys, the delayed key "Limited to 2 connections & 200 markets", and "Any Delayed App Keys
  created after April 8, 2020, have default access to the Stream API"; the same page puts the Live App Key activation
  fee at GBP499 ("Check that your Betfair account is funded to cover the GBP499 activation fee"; the repository's
  earlier notes said GBP299, so the fee has moved: re-check), and the account holds a live key since 6 Oct 2026
  (`CLAUDE.md`). A stream subscription made when a market is created delivers each change with its publish time
  (`pt`), so the first trade's time and size are recorded as they happen; the recorder's polling cannot see a trade
  between polls. The Stream API page itself was not readable anonymously (Confluence "Page Not Found" at the URL tried);
  the field list (marketDefinition, rc with ltp, tv, trd) is from the Historic Data file format the repository already
  parses (`betfair_historic.parse_market`), which is the stream's own recording.
- **The price files for other jurisdictions** are archived already (`betfair_prices.py --archive`; AU, FR, USA, RSA,
  UAE); their "morning" window is unusable for AU (`reports/other_markets_and_stakes_1002.md` section 4). No purchase.

### 6.2 Greyhounds

- **Betfair Historic Data, Greyhound Racing: BASIC is held (the bundle); ADVANCED/PRO add the traded volume and every
  price change.** The service's own help page refused this host (HTTP 403 at https://historicdata.betfair.com/#/help,
  1 Nov 2026) and the search found no published monthly prices; an aggregator describes the packages as "a free Basic
  tier plus paid Advanced/Pro tiers" bought per sport and month (https://apis.io/plans/betfair/betfair-plans-pricing/,
  third party). The repository's own reading: BASIC is "the last traded price each minute, the market's definition
  and its result", free, each sport "bought" once (`betfair_historic.py` docstring). **Prices: not readable here; the
  owner reads them signed in.** What ADVANCED/PRO would add for the trade: the traded volume behind the first price
  (so the first traded price can be told from a GBP2 bet), and the ladder.
- **Keep the GB/IE greyhound record running** (`betfair_recorder.py --record --event-type 4339 --tag greyhound`, from
  6 Oct): it is the only source of a takeable open with its time and money. No purchase.
- **The Betfair greyhound price files** (UK and AU, 2018 on) are archived and built; the morning WAP there prices 14.7%
  of runners (`reports/greyhound_model.md` section 4), so they size the close, not the open. No purchase.

### 6.3 Football

- **Betfair Historic Data, Soccer, BASIC (free): the one purchase that unblocks football.** Once the account holds it,
  `betfair_historic.py --fetch` (UK server, nightly) brings MATCH_ODDS, OVER_UNDER_25 and ASIAN_HANDICAP for GB, ES,
  DE, IT, FR, NL, PT, BE, TR, GR from 2016 (`PLAN_SCOPE`), the tables give first/T-24h/T-60/T-15/T-1/close/result, and
  `football/data.py` ties them to the gold matches (`betfair_market_map`). The owner "buys" it at
  historicdata.betfair.com signed in with the trading account (`reports/all_markets_priorities.md` section 4).
- **Pinnacle's opening and closing lines.** football-data.co.uk already gives both for the main leagues (notes.txt:
  "PSH and PH = Pinnacle home win odds"; the close "with an additional "C" character"), but its "open" is the
  Friday/Tuesday collection and its Pinnacle feed is stale from 23 Jul 2025 (`football/data.py`). Pinnacle's own API:
  "Access to Pinnacle API suite has been closed for the general public since July 23rd, 2025"; applications by email
  to api@pinnacle.com for "select high value bettors & commercial partnerships" and academic projects
  (https://github.com/pinnacleapi/pinnacleapi-documentation, read via search 1 Nov 2026). Not a purchase; an
  application, likely refused for a retail account.
- **A licensed odds API for true opening lines (football and tennis).** Published prices read 1 Nov 2026, to re-check:
  - The Odds API (https://the-odds-api.com/): Starter free, 500 credits a month; 20K credits USD30 a month; 100K USD59;
    5M USD119; 15M USD249; "Historical snapshots of sports odds data back to 2020" on every tier; tennis ("Grand Slams,
    ATP 1000, WTA 1000 and more") and Pinnacle covered; "Betfair" listed, the Exchange product not confirmed. A
    competitor's note says historical calls cost ten times a live call's credits
    (https://oddspapi.io/blog/the-odds-api-free-tier-limits, third party).
  - OddsJam and OpticOdds: quote-only; OpticOdds reported "from around USD5,000/month" by a competitor
    (https://sportsgameodds.com/blog/optic-odds-vs-sports-game-odds, unverified); OpticOdds' own blog says its
    historical endpoint carries Pinnacle opening and closing lines
    (https://opticodds.com/blog/the-power-of-historical-betting-odds-unlocking-insights-with-opticodds).
  - Sportmonks odds add-on: "Odds and Predictions bundle from EUR15/month", the Premium Odds Feed (TXOdds) EUR129 Lite
    and EUR199 Pro a month, history "7 days after kick-off" (https://oddspapi.io/blog/?p=3154, a competitor;
    Sportmonks' own docs confirm the add-on, not the price: https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/premium-odds-feed).
  - Verdict for the trade: none of these is Betfair's own book, and the trade is made on Betfair; a sharp-book opening
    line is a feature for forecasting Betfair's close, not the open the trade needs. The free Soccer BASIC plan comes
    first; The Odds API's free tier is enough to check whether a sharp open adds anything before paying.

### 6.4 Tennis and the rest

- Tennis: no opening price is held and no free source gives one. Betfair Historic Data Tennis BASIC (free, MATCH_ODDS,
  every country in `PLAN_SCOPE`) gives the exchange's own first trade and close; the census (C2) says first whether
  the UK exchange's pre-off tennis money exists at all (the Italian exchange's median is EUR0: `markets.md` section
  3.3). Buy nothing paid until the census says so.
- Cricket, darts, snooker: the free BASIC plans (`PLAN_SCOPE`) once the census shows pre-off depth; nothing else.

### 6.5 Licence constraints (as the repository records them, with what could be read today)

| Source | Terms | Where it says so |
|---|---|---|
| Sackmann ATP/WTA (archive mirror) and TML | CC BY-NC-SA 4.0, non-commercial: "check before the data trains a model that stakes money" | `sources/tennis.py` docstring; `reports/all_markets_priorities.md` section 4 |
| football-data.co.uk | "intended for private individuals only"; written permission for other use | `reports/football_model.md` line 187 (the site's page); notes.txt itself carries no terms statement (read 1 Nov 2026) |
| tennis-data.co.uk | not readable from here (Cloudflare refuses GitHub's runners and this host) | `sources/tennis.py` docstring; no terms read |
| openfootball | CC0, public domain | `sources/openfootball.py` docstring |
| GBGB API | a public API with no licence statement found by the repository | `sources/gbgb.py` (none recorded); **unknown: ask before commercial use** |
| Betfair price files, Historic Data, the API record | Betfair's Developer Programme terms (the API key agreement); the Historic Data plans are per account | `betfair_historic.py` docstring; the terms themselves not fetched |
| API-Football | the Pro plan, USD19 a month, 7,500 requests a day (as the repository recorded it) | `sources/api_football.py` docstring |

## 7. The CI figures

No data found yet: this section is written when the log of run 41291139483 is in (section 5 lists what goes here).
