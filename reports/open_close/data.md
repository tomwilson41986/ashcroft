# Open-to-close trading: the data (data agent, 1 Nov 2026)

The owner's ask (1 Nov 2026): autonomous trading on Betfair of the move from the opening price to the close, hedged
so the result does not matter. This note is the data half: what is held for that trade, market by market, in what
shape, with what gaps; one standard panel to read it through; and the best data to obtain next, with prices and
terms where they could be read. The ranking of the markets is the research agent's (`reports/open_close/markets.md`);
this note follows its order: 1 UK/IE horse win, 2 UK greyhounds, 3 football match odds, the rest parked.

Every figure below that is not a code fact comes from one run of the inventory query on GitHub's runner against the
S3 database and store: **research-query run 41291139483** (`research/queries/done/open_close_inventory_1101.py`,
run 2026-11-01T09:00Z, 128 s, every section ran). "Run 41291139483" after a number means that. Nothing here is a
model result. Hyphens only.

## 1. What the trade needs, and where each piece is

For a back early, lay at the close trade the data a market needs, a selection at a time, is: the earliest tradable
price (when it stood and how much was matched by then), the price at fixed marks before the off (T-24h, T-60, T-15,
T-1), the close (the BSP where there is one, else the last pre-play trade), the volumes, and the result for settlement
checks. The table says where each piece exists today, with the run's counts.

| Piece | UK/IE horses | UK greyhounds | Football | Tennis, the rest |
|---|---|---|---|---|
| Earliest price with its time and money | the recorder's books from 1 Oct 2026 (`betfair_live_marks`, 371,804 rows, 31 days, 26,245 runners; the 08:00 UK mark on 19 of the 31 days, 15,425 runners, 97.8% of win runners with a two-sided book there, median market matched GBP1,423: run 41291139483) | the recorder's greyhound books from 6 Oct 2026 (`books_greyhound.csv.gz`, 27 days, 76.7 MB: run 41291139483); the Historic Data bundle's first traded price has no time and no volume, and almost never stands early: a price at T-24h on 78 of 451,031 WIN runners, at T-60 on 37,413 of 3,054,761 rows, at T-1 on 1,154,352 (run 41291139483) | none held: football-data.co.uk's "open" is the price collected on Friday or Tuesday afternoon (notes.txt: "Betting odds for weekend games are collected Friday afternoons, and on Tuesday afternoons for midweek games"); Betfair's own first trade needs the Historic Data Soccer plan the account does not hold (`betfair_market_map.parquet`: no data found, run 41291139483) | none |
| Morning price with volume | `betfair_prices`: 2,136,686 rows, 226,863 markets, 2017-12-31 to 2026-10-30; UK win 2026: a morning WAP on 72,187 of 72,242 runners, median morning volume GBP598 a runner (90th pct GBP3,341) (run 41291139483) | the greyhound price-file tables 2012-2026 (15 tables, 970,848 rows in 2026); GB win: morning money GBP0 at the median and the 90th percentile in 2024-26, any morning trade on 9% of runners (run 41291139483) | n/a | n/a |
| T-24h / T-60 / T-15 / T-1 | the recorder's `off_120 .. off_1` marks, 25,659-25,733 runner rows each, 31 days (run 41291139483) | the Historic Data tables (above) and the recorder's greyhound books | the Historic Data soccer tables once held | the Historic Data tables once each sport is held |
| Close | the BSP: 19,068 settled runners with one in `betfair_live_marks` (25 days, from 7 Oct); every price-file row from 2018 (run 41291139483) | the BSP: 875,531 rows of the bundle (GB and some AU), every price-file row | no BSP: the benchmark close is held for 108,173 finished matches with an open too; Betfair Exchange's own close on 85.8% (2024), 93.3% (2025), 94.3% (2026) of matches (run 41291139483) | no BSP; nothing held |
| Volumes | morning, pre-play and in-play volume a runner (files); market total matched at each mark (recorder) | the same | none | none |
| Result | WIN_LOSE on every file row; WINNER/LOSER on the settled book | the same; GBGB 3.38m runs 2018-2026 (run 41291139483) | scores for 299,440 finished matches; `won` once the soccer tables exist | `won` |

## 2. The inventory (research/queries/done/open_close_inventory_1101.py)

One read-only script for `.github/workflows/research-query.yml`; each section guarded, a JSON block at the end
(`OPEN_CLOSE_INVENTORY_JSON`) for the modelling agent. Moved to `done/` after the run. What each section reports and
its headline figure (all run 41291139483; the tables are in section 7):

| Section | What it prints | Headline |
|---|---|---|
| A0 | the tables in horse_racing.db | race_results 2,018,679 rows (2010-01-01 to 2026-10-30); betfair_prices 2,136,686; betfair_live_marks 371,804 (1-31 Oct 2026); betfair_live_markets 26,245; live_orders 50,080; betfair_prices_files 12,035 |
| A1 | `betfair_prices` by country, market and year | uk_win 782,874 rows, uk_place 768,189, ire_win 294,142, ire_place 291,481; a BSP on every row from 2018; UK win 2026 median BSP 12.63, morning volume GBP598, pre-play GBP13,251 |
| A2 | `betfair_live_marks`: coverage, marks, money | 31 days; 08:00 mark on 19 days (15,425 rows), 10:00 on 30, 12:00 on 31; win market matched: GBP1,423 at 08:00, 6,680 at 12:00, 13,493 at T-60, 26,368 at T-15, 77,294 at T-1, 83,454 at the last book (medians) |
| A3 | the mids against the BSP by band | 13,992 runners with an 08:00 two-sided mid and a BSP; win: median log(BSP/mid) within +-0.04 from 1.5 to 21, -0.075 at 21-51, -0.476 at 51+ |
| A4 | `betfair_live_markets` | Oct 2026: 930 GB win, 332 IE win, 920 GB place, 332 IE place markets |
| B1 | `sources/betfair_historic/` | 261,008 raw files (0.40 GB), greyhound racing 2026 only; one table: 3,054,761 runners, 251,748 markets, offs 2025-12-10 to 2026-09-30 |
| B2 | `sources/football/gold/` | 299,646 matches 1993-2026; 108,173 finished with a 1X2 open and close; odds_long 10,149,855 rows; no Betfair map |
| B3 | `sources/tennis/` | sackmann_matches 1,268,349 rows (atp 704,194, wta 564,155; TML 78,081); odds_matches.parquet **not held** (the UK server has not built it): no tennis price of any kind in the store, so no open |
| B4 | GBGB and the greyhound price tables | GBGB runs 2018-2026: 438,671 / 442,709 / 335,982 / 377,516 / 371,982 / 386,392 / 384,406 / 358,132 / 279,630; prices_2026 970,848 rows, 77,285 GB and 75,160 AUS win markets |
| B5 | the other tables | openfootball 78,313 matches (no price); api_football none; puntingform none |
| C1 | `betfair_live/<day>/` | 367 objects over 32 days (1 Oct to 1 Nov); 13 file names (section 7.6) |
| C2 | the census, Oct 2026 | 25 days (7-31 Oct), 24 sports, 426 sport x market type rows (section 5) |
| D | the Historic Data first -> close move | 1,863,552 greyhound runners with a first and a closing price in 237,744 markets; GB WIN first -> BSP median log move +0.025 (3-5), +0.050 (5-8), +0.103 (8-13), +0.129 (13-21), +0.117 (21-51): the first traded price sits UNDER the BSP and only 34-45% of runners shorten |

## 3. The panel (sources/open_close.py)

One schema across markets, one row a selection a market:

| Column | Meaning |
|---|---|
| sport, country, market_type | Betfair's names where the source has them (WIN, PLACE, MATCH_ODDS, OVER_UNDER_25, ASIAN_HANDICAP); countries as ISO-2 (uk -> GB, ire -> IE, AUS -> AU) |
| market_id, event_id, event_name, selection_id, selection_name | the source's identifiers; a price file has no market id, so `market_id` is `<event id>:<type>`; a football match's `event_id` is the gold `match_id` |
| off_time_utc | the scheduled off or kick-off, tz-aware UTC (the price files' UK wall-clock times converted DST-aware) |
| n_runners | selections not removed |
| open_price, open_time_utc, open_matched | the earliest price the source holds, when it stood, the money matched by then (only the recorder's books give a time; the Historic Data BASIC files give neither time nor volume) |
| p_t24h, p_t60, p_t15, p_t3, p_t1 | the price standing at each mark before the off, where the source has it (T-3 from the recorder only: added 1 Nov for the greyhound open, section 4.2) |
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
| `read_recorded_books(store, sport, tag, open_mark=None)` | `betfair_live/<day>/books_<tag>.csv.gz` + `markets_<tag>.csv.gz` (the greyhound record: tag `greyhound`) | the book at `open_mark` (the build passes `off_3` for greyhounds: `GREYHOUND_OPEN_MARK`), its time, the market's matched money then; with no mark, the first two-sided pre-off book | T-60, T-15, T-3, T-1 through `betfair_recorder.marks` (the nightly load's own picks) | the last pre-off mid; the BSP as the market reconciled it | the greyhound open is T-3 because the money arrives only in the last three minutes (section 4.2); the same reader reads the horse record with tag `""` |

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
the UK server). It has not yet been run against S3: the panel files do not exist until it is.

Tests: `tests/test_open_close.py`, 8 tests, local folder store and a temporary SQLite database, no network: the
schema from each source's own layout, the UTC conversion (14:30 UK in July -> 13:30 UTC), the removed runner left
out of `n_runners`, the one-sided book giving no open, the first two-sided book as the greyhound open with its time
and money, the football results settled from the scores, the build writing a parquet a year and a rerun overwriting
it, a sport with nothing held building nothing, the CLI from a local root. `pytest tests/test_open_close.py -q`:
8 passed (1 Nov 2026); with `tests/test_sources.py` beside it, 35 passed.

## 4. Per-market data position (the research agent's order; figures run 41291139483)

### 4.1 UK/IE horse win (live)

Held: the price files 2017-12-31 to 2026-10-30 (2,136,686 rows; UK win 782,874; a morning WAP on 99.9% of UK win
runners from 2019 and on 90% of IE win runners), the recorder's books from 1 Oct 2026 (26,245 runners, 2,515
markets), the live key's feed from 6 Oct (`CLAUDE.md`). The panel gives two opens for the same horse: the morning
WAP (an average, with volume, nine years deep) and the 08:00 mid (takeable in principle, one month deep: 15,425
runner rows on 19 days, the record starting after 08:00 on the other 12).

What the run says about the 08:00 open (win markets): 97.8% of runners carry a two-sided book, the median best back
GBP6.17 and best lay GBP5.84, the relative spread 18.9% of the mid; the market has matched GBP1,423 at the median
(90th pct GBP3,946) against GBP83,454 at the last book. The 08:00 mid -> BSP by band: median log move -0.055 (1.5-2),
-0.027 (2-3), +0.010 (3-5), 0.000 (5-8), +0.024 (8-13), +0.037 (13-21), -0.075 (21-51), -0.476 (51+), with 46-55% of
runners shortening in the 3-51 bands: no market-only drift to catch on the back side in the middle of the book, and
the long-shot "shortening" is the wide 08:00 book's mid, not a price anyone could back (relative spread 18.9% at
08:00 against 5.4% at the last). The same at 10:00 and 12:00 (section 7.3). The place market's 08:00 mid is
unusable: relative spread 124% of the mid, median log move -0.3 to -2.9 in the bands above 8.

The on-the-day profile (census_all_markets_1101, research-query run 41294133067, section 3a: 31 days, 2,515
markets; the market's matched money, median a market, 90th percentile in brackets):

| Mark | GB WIN | IE WIN | GB PLACE | IE PLACE |
|---|---|---|---|---|
| 08:00 UK | 1,650 (4,735) | 641 (1,755) | 39 (166) | 30 (50) |
| 10:00 | 4,197 (10,101) | 1,556 (3,791) | 120 (446) | 43 (128) |
| 12:00 | 8,030 (22,832) | 3,126 (7,978) | 286 (1,282) | 80 (346) |
| T-120 | 12,064 (24,445) | 5,044 (9,141) | 468 (1,327) | 128 (428) |
| T-60 | 15,760 (31,922) | 6,703 (13,084) | 893 (2,269) | 231 (716) |
| T-15 | 30,690 (63,471) | 14,424 (27,105) | 2,524 (5,098) | 1,192 (2,493) |
| T-1 | 89,503 (162,759) | 47,406 (86,369) | 9,670 (18,009) | 4,899 (11,049) |
| last | 97,168 (170,561) | 52,095 (91,887) | 10,597 (19,488) | 5,555 (12,176) |

A day's GB+IE WIN markets hold GBP3,646,192 at the last book (median over 31 days; min 1,854,510, max 6,813,849),
against GBP23,155 on the next day's top 200 Horse Racing markets in the evening census (run 41291139483): the
evening read sees under 1% of the day's racing money, so the census compares the other sports' day-ahead money, not
racing's. A GB win market has 1.7% of its final money at 08:00, 8% at 12:00, 16% at T-60, 32% at T-15.

Missing: the true first price with its time before 06:15 UK, and the traded-volume ladder. Next data: section 6.1.

### 4.2 UK greyhounds (research next)

Held: the price files 2012-2026 (15 tables), GBGB results 2018 on (3.38m runs), the owner's Historic Data bundle
(3,054,761 runner rows, 251,748 markets, offs 2025-12-10 to 2026-09-30, GB/AU/NZ, every market type: GB WIN 35,306
markets, GB PLACE 35,306, GB FORECAST 35,385, AU WIN 34,394, AU MATCH_BET 34,221, NZ WIN 2,996), the recorder's GB/IE
greyhound books from 6 Oct (27 days to 1 Nov).

What the run says. (a) The bundle's "first traded price" is late: a price stands at T-24h for 78 of 451,031 WIN
runners, at T-60 for 37,413 of 3,054,761 rows, at T-15 for 101,619, at T-1 for 1,154,352; so the first trade is
typically inside the last quarter of an hour and the Historic Data BASIC files hold no early open for greyhounds.
(b) GB WIN first traded -> BSP: median log move +0.420 (1-1.5, n 698), +0.148 (1.5-2), +0.056 (2-3), +0.025 (3-5),
+0.050 (5-8), +0.103 (8-13), +0.129 (13-21), +0.117 (21-51), -0.262 (51+); shortened 17%, 33%, 41%, 45%, 42%, 36%, 34%,
37%, 61%. The first price sits under the BSP across the book (a lay-first, back-at-SP shape), the move's IQR about
+-0.13 to +-0.4 in log terms (section 7.8). AU and NZ show the same sign at a smaller size (AU +0.015 to +0.104 in
the 3-51 bands). (c) The close in these tables is the last price traded before the market closed, not an in-play
boundary: greyhound markets do not turn in play (0.0% went in play), so `ltp_close` is the last pre-off trade.
(d) The greyhound price files' GB morning money is GBP0 at the 90th percentile in 2024-26 (any morning trade: 9% of
runners), so the files size the close only. (e) The close is well calibrated: won 0.734 against implied 0.758 at
1-1.5, 0.259 against 0.260 at 3-5, 0.034 against 0.033 at 21-51 (section 7.8).

The on-the-day profile from the recorded books (census_all_markets_1101, research-query run 41294133067, section 3b:
27 days, 6 Oct to 1 Nov, 6,993 markets, through `betfair_recorder.marks`; the market's matched money, median a
market, 90th percentile in brackets):

| Mark | GB/IE WIN | GB/IE PLACE |
|---|---|---|
| 08:00 to 12:00 UK, T-120, T-60 | 0 (0) | 0 (0) |
| T-30 | 0 (7) | 0 (0) |
| T-15 | 0 (20) | 0 (0) |
| T-10 | 2 (50) | 0 (2) |
| T-5 | 34 (434) | 0 (2) |
| T-3 | 380 (1,735) | 0 (57) |
| T-1 | 1,993 (5,717) | 129 (1,155) |
| last | 2,377 (6,831) | 218 (1,528) |

The first book the record holds a market stands 612 minutes before the off at the median (90th pct 869) with GBP0
matched behind it (0% of markets over GBP100). A day's GB/IE WIN markets hold GBP427,681 at the last book (median
over 27 days, 137 markets a day) against GBP109 on the next day's top 200 in the evening census (run 41291139483).

The consequence for the panel: a greyhound market's open exists only in the last three minutes. The first two-sided
book is a hollow spread hours out, T-15 carries nothing, T-5 GBP34, T-3 GBP380 and T-1 GBP1,993 at the median; so the
panel's greyhound open is the T-3 book (`GREYHOUND_OPEN_MARK = "off_3"`, `p_t3` added to the schema from the
recorder's `off_3` mark, 1 Nov), with T-1 and the last book as the later marks and the BSP as the close. The
research agent's lay-early test is a T-3 (or T-1) to BSP test, with GBP380 (T-3) or GBP1,993 (T-1) matched a market
behind it: the size of any such trade is bounded by that money. The Historic Data bundle's first traded price (a
price at T-60 on 1.2% of rows) is the same late money seen from the other side. Missing: nothing a purchase would
mend; the record from 6 Oct is the data, and it grows a day a night. Next data: section 6.2.

### 4.3 Football match odds (research next)

Held: football-data.co.uk 1993/94 on in the gold table (299,646 matches, 299,440 finished); a 1X2 open on 62-67% of
finished matches each season 2012-2025 (the extra leagues carry a close only) and a close on 97-100%; both on
108,173 matches. The open's source: Pinnacle to 2023/24 (7,778 of 7,804 in 2023), Betfair Exchange from 2024/25
(7,181 in 2024, 6,896 in 2025); the close: Pinnacle to 2024/25 (12,430 in 2024), Betfair Exchange from 2025/26
(11,513 in 2025, 4,664 in 2026). Betfair Exchange's own open on 61.4% (2024), 57.8% (2025) of matches; its close on
85.8%, 93.3%, 94.3% (2024-26). The 1X2 home price's benchmark open -> close: median log move 0.000 to +0.013 in every
band, 46-47% shortening, sd 0.047 (1-1.5) to 0.211 (13-21); the over 2.5: +0.006 to +0.020, 36-47% shortening. No
drift from the Friday/Tuesday price to the close in the benchmark books. No Betfair soccer files: the account holds
no Soccer plan on the Historic Data service (`betfair_market_map.parquet`: no data found; dq `betfair_markets_linked`
0). The census (section 5) puts Soccer MATCH_ODDS at GBP437,775 matched a day on the type within the top 200 by the
evening before, 185 markets a day: the deepest pre-off market after cricket, and open a day ahead.

Missing: Betfair's own first traded price, the marks and the pre-play close for football (the Soccer BASIC plan,
free), and any volume at the open. Next data: section 6.3.

### 4.4 Tennis and the rest (parked)

Tennis: the Sackmann/TML table (1,268,349 matches) and nothing else: `tennis/odds_matches.parquet` is not held, so
there is no tennis price of any kind in the store and no open (B3). The census: Tennis MATCH_ODDS GBP107,574 a day on
105 markets, but the median market of the top 200 carries GBP5 by the evening before (section 5): the money sits in
a few main-tour matches. Cricket: GBP6.8m a day on 17 MATCH_ODDS markets by the evening before, the deepest market
Betfair lists (max GBP88m on one day), none of it held. Golf WINNER GBP507k a day. Darts GBP872, Snooker GBP3,344 a
day on the top 200: nothing to trade pre-off. Next data: section 6.4.

## 5. The census of every sport, October 2026 (section C2 of run 41291139483)

`betfair_live/<day>/census_sports.json`, 25 days (7-31 Oct), read once an evening: every sport with markets starting
in the next 24 hours, its market types with counts, and the money matched by then on its 200 busiest markets
(`betfair_recorder.census_sports`). Read the evening before, this is the pre-off money that exists a day ahead: the
"how early" measure. For horse racing and greyhounds the next day's cards have barely traded by the evening (the
morning is where their money starts: section 4.1), so their figures here are small by construction; for football,
cricket, golf and US sports the markets are open and traded days ahead.

### 5.1 By sport (median over the 25 days; GBP matched on the top 200 markets by the evening read)

| Sport | Days seen | Markets a day (next 24 h) | Top-200 matched, total GBP | Median market of the 200, GBP | Largest day's total GBP |
|---|---|---|---|---|---|
| Cricket | 25 | 135 | 6,799,296 | 0 | 88,244,909 |
| Soccer | 25 | 6,106 | 908,482 | 993 | 2,446,077 |
| Golf | 17 | 37 | 653,691 | 378 | 1,372,188 |
| American Football | 25 | 19 | 291,257 | 1,554 | 1,556,400 |
| Tennis | 25 | 2,781 | 115,954 | 5 | 6,188,991 |
| Basketball | 25 | 126 | 92,200 | 4 | 912,330 |
| Mixed Martial Arts | 8 | 24 | 42,997 | 6 | 142,089 |
| Ice Hockey | 25 | 80 | 30,179 | 0 | 119,632 |
| Baseball | 20 | 8 | 25,453 | 12 | 103,548 |
| Horse Racing | 25 | 413 | 23,155 | 4 | 793,657 |
| Motor Sport | 5 | 10 | 22,768 | 442 | 81,688 |
| Rugby League | 10 | 20 | 7,544 | 0 | 14,611 |
| Boxing | 8 | 15 | 6,746 | 0 | 238,809 |
| Rugby Union | 13 | 17 | 3,881 | 0 | 29,674 |
| Snooker | 8 | 8 | 3,344 | 131 | 6,597 |
| Politics | 7 | 1 | 2,040 | 1,993 | 205,093 |
| Darts | 24 | 15 | 872 | 18 | 31,662 |
| Special Bets | 18 | 2 | 617 | 291 | 139,824 |
| Australian Rules | 12 | 8 | 473 | 0 | 10,907 |
| Esports | 14 | 9 | 297 | 0 | 3,288 |
| Greyhound Racing | 25 | 481 | 109 | 0 | 4,380 |
| Cycling, Gaelic Games, Volleyball | 1, 1, 11 | 1, 2, 1 | 1,832 / 567 / 0 | | |

Concentration (the mean of the top 200 over its median): Soccer 4.6, Golf 8.6, American Football 0.9, Tennis 116,
Basketball 115, Horse Racing 28.9; a median market of GBP0 (Cricket, Ice Hockey, Greyhounds) means fewer than 100 of
the sport's next-day markets had traded at all by the evening.

### 5.2 Sport x market type (median over the days: markets a day, GBP matched on the type within the top 200)

The market types that matter for a pre-off trade, from the run's 60 best-matched types and the main sports in full:

| Sport | Market type | Days | Markets a day | Matched on the type, median GBP | Largest day GBP |
|---|---|---|---|---|---|
| Cricket | MATCH_ODDS | 25 | 17 | 1,669,718 | 88,088,108 |
| Cricket | A_1ST_INN_RUNS_LINE | 5 | 1 | 601,079 | 1,045,539 |
| Golf | WINNER | 17 | 3 | 506,658 | 1,242,569 |
| Soccer | MATCH_ODDS | 25 | 185 | 437,775 | 1,624,883 |
| Soccer | ASIAN_HANDICAP | 25 | 185 | 175,058 | 532,360 |
| American Football | HANDICAP | 25 | 3 | 143,448 | 905,078 |
| Tennis | MATCH_ODDS | 25 | 105 | 107,574 | 694,887 |
| Cricket | DRAW_NO_BET | 17 | 3 | 67,316 | 145,935 |
| American Football | COMBINED_TOTAL | 25 | 3 | 54,122 | 426,137 |
| Soccer | ALT_TOTAL_GOALS | 25 | 185 | 50,013 | 124,591 |
| Golf | TOP_10_FINISH | 13 | 1 | 44,235 | 64,067 |
| Mixed Martial Arts | MATCH_ODDS | 8 | 7 | 40,866 | 141,380 |
| American Football | MATCH_ODDS | 25 | 3 | 37,564 | 239,724 |
| Basketball | MATCH_ODDS | 25 | 42 | 34,533 | 429,385 |
| Soccer | OVER_UNDER_25 | 25 | 184 | 34,173 | 75,356 |
| Basketball | COMBINED_TOTAL | 25 | 42 | 31,603 | 222,682 |
| Basketball | HANDICAP | 25 | 42 | 26,065 | 412,865 |
| Horse Racing | WIN | 25 | 177 | 22,667 | 575,046 |
| Ice Hockey | MONEY_LINE | 25 | 20 | 18,817 | 97,345 |
| Horse Racing | ANTEPOST_WIN | 6 | 2 | 16,678 | 182,765 |
| Soccer | CORRECT_SCORE | 25 | 185 | 15,789 | 36,376 |
| Soccer | BOTH_TEAMS_TO_SCORE | 25 | 185 | 12,697 | 52,915 |
| Soccer | OVER_UNDER_35 | 25 | 185 | 8,761 | 93,581 |
| Tennis | TOURNAMENT_WINNER | 7 | 4 | 8,588 | 15,762 |
| Soccer | OVER_UNDER_15 | 25 | 185 | 8,570 | 49,581 |
| Baseball | MATCH_ODDS | 20 | 2 | 7,550 | 28,223 |
| Soccer | OVER_UNDER_05 | 25 | 185 | 7,060 | 23,065 |
| Boxing | MATCH_ODDS | 8 | 3 | 6,383 | 130,108 |
| Soccer | DRAW_NO_BET | 25 | 185 | 2,628 | 16,475 |
| Tennis | HANDICAP | 25 | 23 | 2,651 | 15,948 |
| Tennis | COMBINED_TOTAL | 25 | 23 | 1,894 | 10,220 |
| Tennis | SET_BETTING | 25 | 88 | 708 | 6,838 |
| Darts | OUTRIGHT_WINNER | 3 | 1 | 4,960 | 31,393 |
| Rugby League | MATCH_ODDS | 10 | 3 | 4,033 | 13,939 |
| Rugby Union | MATCH_ODDS | 13 | 4 | 3,175 | 17,196 |
| Cricket | TO_WIN_THE_TOSS | 23 | 5 | 335 | 7,517,656 |

Greyhound Racing WIN and PLACE (481 markets a day) never reached GBP109 in total by the evening before; Soccer's
player and card markets (BOOKING_ODDS, FIRST_GOAL_SCORER, the card and corner lines) never reached the top 200.
The whole table (426 rows) is in the run's log, section C2.

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
  between polls, and on 12 of October's 31 days it had not started by 08:00 (A2). The Stream API page itself was not
  readable anonymously (Confluence "Page Not Found" at the URL tried); the field list (marketDefinition, rc with ltp,
  tv, trd) is from the Historic Data file format the repository already parses (`betfair_historic.parse_market`),
  which is the stream's own recording.
- **The price files for other jurisdictions** are archived already (`betfair_prices.py --archive`; AU, FR, USA, RSA,
  UAE); their "morning" window is unusable for AU (`reports/other_markets_and_stakes_1002.md` section 4). No purchase.

### 6.2 Greyhounds

- **Keep the GB/IE greyhound record running** (`betfair_recorder.py --record --event-type 4339 --tag greyhound`, from
  6 Oct; 27 days held): it is the only source of a takeable open with its time and money, because the run shows the
  Historic Data first trade is itself late (a price at T-60 on 1.2% of rows). No purchase.
- **Betfair Historic Data, Greyhound Racing: BASIC is held (the bundle); ADVANCED/PRO add the traded volume and every
  price change.** The service's own help page refused this host (HTTP 403 at https://historicdata.betfair.com/#/help,
  1 Nov 2026) and the search found no published monthly prices; an aggregator describes the packages as "a free Basic
  tier plus paid Advanced/Pro tiers" bought per sport and month (https://apis.io/plans/betfair/betfair-plans-pricing/,
  third party). The repository's own reading: BASIC is "the last traded price each minute, the market's definition
  and its result", free, each sport "bought" once (`betfair_historic.py` docstring). **Prices: not readable here; the
  owner reads them signed in.** What ADVANCED/PRO would add: the volume behind the first price (a first traded price
  of GBP2 is not an open) and the ladder. Given (a) above, the recorder is the better buy and costs nothing.
- **The Betfair greyhound price files** (UK and AU, 2012 on) are archived and built; GB morning money is GBP0 at the
  90th percentile in 2024-26 (run 41291139483), so they size the close, not the open. No purchase.

### 6.3 Football

- **Betfair Historic Data, Soccer, BASIC (free): the one purchase that unblocks football.** Once the account holds it,
  `betfair_historic.py --fetch` (UK server, nightly) brings MATCH_ODDS, OVER_UNDER_25 and ASIAN_HANDICAP for GB, ES,
  DE, IT, FR, NL, PT, BE, TR, GR from 2016 (`PLAN_SCOPE`), the tables give first/T-24h/T-60/T-15/T-1/close/result, and
  `football/data.py` ties them to the gold matches (`betfair_market_map`). The owner "buys" it at
  historicdata.betfair.com signed in with the trading account (`reports/all_markets_priorities.md` section 4). The
  census says the money is there a day ahead (GBP437,775 a day on MATCH_ODDS, GBP175,058 on ASIAN_HANDICAP, run
  41291139483), which no other non-racing sport but cricket and golf shows.
- **Pinnacle's opening and closing lines.** football-data.co.uk already gives both for the main leagues (notes.txt:
  "PSH and PH = Pinnacle home win odds"; the close "with an additional "C" character"), but its "open" is the
  Friday/Tuesday collection and its Pinnacle feed is stale from 23 Jul 2025 (`football/data.py`); the run shows the
  benchmark open -> close move is centred on zero (section 4.3). Pinnacle's own API: "Access to Pinnacle API suite has
  been closed for the general public since July 23rd, 2025"; applications by email to api@pinnacle.com for "select
  high value bettors & commercial partnerships" and academic projects
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

- Tennis: no price of any kind is held (B3) and no free source gives an open. Betfair Historic Data Tennis BASIC (free,
  MATCH_ODDS) gives the exchange's own first trade and close; the census says the UK exchange's pre-off tennis money a
  day ahead is GBP107,574 on 105 MATCH_ODDS markets with a median market of GBP5 (section 5): a few main-tour matches
  carry it. The UK server's `tennis_odds` fetch (tennis-data.co.uk closing odds) has not produced its table; check
  `betfair-prices.yml`'s log before relying on it. Buy nothing paid.
- Cricket: GBP6.8m a day pre-off on MATCH_ODDS by the evening before (section 5), far the deepest; the brief excludes
  in-play and the toss, but a pre-off open-to-close read costs only the free Cricket BASIC plan. Golf WINNER
  GBP507k a day: outrights open a week ahead; the free plan again if the research agent wants the read. Darts,
  snooker: nothing to trade pre-off (section 5).

### 6.5 Licence constraints (as the repository records them, with what could be read today)

| Source | Terms | Where it says so |
|---|---|---|
| Sackmann ATP/WTA (archive mirror) and TML | CC BY-NC-SA 4.0, non-commercial: "check before the data trains a model that stakes money" | `sources/tennis.py` docstring; `reports/all_markets_priorities.md` section 4 |
| football-data.co.uk | "intended for private individuals only"; written permission for other use | `reports/football_model.md` line 187 (the site's page); notes.txt itself carries no terms statement (read 1 Nov 2026) |
| tennis-data.co.uk | not readable from here (Cloudflare refuses GitHub's runners and this host); its table is not held | `sources/tennis.py` docstring; run 41291139483 B3 |
| openfootball | CC0, public domain | `sources/openfootball.py` docstring |
| GBGB API | a public API with no licence statement found by the repository | `sources/gbgb.py` (none recorded); **unknown: ask before commercial use** |
| Betfair price files, Historic Data, the API record | Betfair's Developer Programme terms (the API key agreement); the Historic Data plans are per account | `betfair_historic.py` docstring; the terms themselves not fetched |
| API-Football | the Pro plan, USD19 a month, 7,500 requests a day (as the repository recorded it); nothing fetched yet (B5) | `sources/api_football.py` docstring |

## 7. The figures (research-query run 41291139483, job 124802487010)

### 7.1 horse_racing.db (A0)

| Table | Rows | First | Last |
|---|---|---|---|
| race_results | 2,018,679 | 2010-01-01 | 2026-10-30 |
| betfair_prices | 2,136,686 | 2017-12-31 | 2026-10-30 |
| betfair_prices_files | 12,035 | | |
| betfair_live_marks | 371,804 | 2026-10-01 | 2026-10-31 |
| betfair_live_markets | 26,245 | 2026-10-01 | 2026-10-31 |
| live_orders | 50,080 | 2026-09-30 | 2026-10-31 |
| betfair_odds | 0 | | |

### 7.2 betfair_prices (A1)

Rows by country and market: uk_win 782,874; uk_place 768,189; ire_win 294,142; ire_place 291,481 (226,863 markets).
2017 holds a few days only (838 UK win rows). Coverage: a BSP on every row from 2018 except 2018 UK win (93,016 of
104,004) and 2018 IE win (29,988 of 32,064); a morning WAP on 99.4-99.9% of UK win rows from 2019, 90-93% of IE win,
75-96% of UK place; a result on every row; joined to race_results on 98-99%.

UK win, by year: median BSP, morning WAP, morning volume (median / 90th pct), pre-play volume (median / 90th pct):

| Year | Rows | Median BSP | Median morning WAP | Morning GBP med / p90 | Pre-play GBP med / p90 | In-play GBP med |
|---|---|---|---|---|---|---|
| 2018 | 93,016 | 13.42 | 11.70 | 525 / 4,476 | 22,170 / 146,214 | 3,832 |
| 2019 | 91,504 | 13.50 | 12.07 | 627 / 5,085 | 20,444 / 138,460 | 3,518 |
| 2020 | 73,896 | 14.00 | 12.50 | 637 / 5,339 | 19,392 / 125,830 | 3,319 |
| 2021 | 91,267 | 13.00 | 11.65 | 647 / 4,523 | 18,933 / 111,773 | 3,386 |
| 2022 | 86,403 | 12.50 | 11.21 | 613 / 3,896 | 16,885 / 96,549 | 2,801 |
| 2023 | 87,656 | 12.22 | 11.27 | 536 / 3,291 | 15,414 / 80,339 | 4,042 |
| 2024 | 88,673 | 13.00 | 11.60 | 484 / 3,205 | 12,948 / 66,134 | 3,500 |
| 2025 | 86,391 | 12.50 | 11.22 | 452 / 2,932 | 13,214 / 66,484 | 3,196 |
| 2026 | 72,242 | 12.63 | 11.54 | 598 / 3,341 | 13,251 / 64,425 | 2,961 |

IE win 2026: 28,426 rows, median BSP 19.50, morning GBP106 / 884, pre-play GBP5,105 / 28,255. UK place 2026: morning
GBP17 / 142, pre-play GBP2,075 / 6,421. IE place 2026: morning GBP2 / 28, pre-play GBP948 / 3,861.

Morning WAP -> BSP, UK win, 2024-26 (median log(BSP/WAP); negative = shortened; share shortened; IQR):

| Morning WAP band | n | Median log move | Shortened | IQR |
|---|---|---|---|---|
| 1.0-1.5 | 756 | -0.026 | 63.2% | -0.071 to +0.029 |
| 1.5-2.0 | 2,682 | -0.022 | 56.0% | -0.112 to +0.087 |
| 2.0-3.0 | 9,529 | +0.002 | 49.5% | -0.124 to +0.159 |
| 3.0-5.0 | 29,024 | +0.023 | 45.9% | -0.135 to +0.203 |
| 5.0-8.0 | 43,482 | +0.035 | 45.0% | -0.151 to +0.244 |
| 8.0-13.0 | 49,406 | +0.074 | 41.5% | -0.155 to +0.322 |
| 13.0-21.0 | 40,169 | +0.130 | 37.2% | -0.136 to +0.420 |
| 21.0-51.0 | 42,686 | +0.206 | 33.4% | -0.116 to +0.564 |
| 51+ | 28,831 | +0.423 | 25.4% | -0.009 to +0.951 |

The years to 2023 are the same shape (UK: -0.017 at 1-1.5 to +0.482 at 51+); IE 2024-26: -0.021 at 1-1.5 to +0.461 at
51+, shortening 62.6% to 25.8%. Pre-play WAP -> BSP (UK win): +0.009 (1.5-2), +0.022 (2-3), +0.050 (3-5), +0.087 (5-8),
+0.121 (8-13), +0.150 (13-21), +0.171 (21-51), +0.236 (51+): the pre-play VWAP sits under the BSP for everything but
the shortest, because the money that forms the BSP comes after most of the pre-play volume.

### 7.3 betfair_live_marks (A2, A3)

Coverage: 31 days, 41-120 markets a day, 476-1,191 runners a day; the 08:00 mark on 19 days (none 1-9, 11-12, 23 Oct:
the record started later), 10:00 on 30, 12:00 on 31; a BSP on the settled mark from 7 Oct (768 of 811 runners that
day; 19,068 over 25 days). Rows a mark: clock_08:00 15,425 (15,173 two-sided), clock_09:00 23,786, clock_10:00 24,383,
clock_11:00 24,625, clock_12:00 25,681, off_120 25,733, off_60 25,707, off_30 25,679, off_15 / off_10 / off_5 / off_3 /
off_1 25,659, last 26,245, final 26,245.

Market money matched at each mark (median a market, GBP; 90th pct in brackets) and the top of the book (win):

| Mark | WIN matched | PLACE matched | WIN two-sided | WIN best back size | WIN relative spread |
|---|---|---|---|---|---|
| 08:00 | 1,423 (3,946) | 35 (135) | 97.8% | 6.17 | 18.9% |
| 09:00 | 2,295 (6,108) | 56 (261) | 96.2% | 4.76 | 16.0% |
| 10:00 | 3,524 (9,154) | 89 (385) | 95.2% | 5.60 | 13.3% |
| 11:00 | 5,028 (12,823) | 144 (622) | 94.4% | 6.84 | 11.2% |
| 12:00 | 6,680 (18,033) | 220 (1,009) | 93.0% | 10.91 | 10.9% |
| T-120 | 10,040 (21,513) | 373 (1,157) | 92.5% | 10.98 | 6.5% |
| T-60 | 13,493 (29,049) | 717 (1,965) | 91.9% | 13.62 | 6.2% |
| T-30 | 17,043 (35,767) | 1,067 (2,738) | 91.4% | 15.83 | 5.7% |
| T-15 | 26,368 (54,839) | 2,137 (4,708) | 91.1% | 22.19 | 4.9% |
| T-10 | 32,232 (65,610) | 3,017 (6,121) | 91.0% | 26.05 | 5.3% |
| T-5 | 45,714 (90,358) | 4,835 (9,429) | 90.9% | 35.46 | 4.7% |
| T-3 | 56,103 (112,462) | 5,905 (11,795) | 90.9% | 35.26 | 4.9% |
| T-1 | 77,294 (148,480) | 8,494 (16,729) | 90.8% | 32.54 | 5.3% |
| last | 83,454 (155,418) | 9,310 (18,265) | 90.9% | 31.80 | 5.4% |

The place book's relative spread: 124% at 08:00, 52% at 12:00, 18% at T-60, 6.5% at the last.

Mid -> BSP, WIN (median log(BSP/mid); share shortened; n), by mark and mid band:

| Band | 08:00 | 10:00 | 12:00 | T-120 | T-60 | T-15 | T-1 | last |
|---|---|---|---|---|---|---|---|---|
| 1.5-2 | -0.055 / 59% / 68 | -0.041 / 61% / 84 | -0.022 / 54% / 102 | -0.014 / 53% / 103 | +0.006 / 50% / 107 | +0.014 / 44% / 107 | -0.005 / 53% / 108 | +0.001 / 49% / 103 |
| 2-3 | -0.027 / 54% / 229 | -0.009 / 52% / 311 | -0.003 / 50% / 326 | 0.000 / 50% / 329 | +0.013 / 48% / 336 | +0.012 / 46% / 328 | 0.000 / 49% / 336 | +0.002 / 49% / 340 |
| 3-5 | +0.010 / 49% / 594 | +0.007 / 49% / 780 | +0.007 / 48% / 827 | +0.012 / 48% / 826 | +0.003 / 49% / 823 | +0.002 / 50% / 823 | +0.010 / 46% / 894 | +0.010 / 47% / 891 |
| 5-8 | 0.000 / 50% / 1,002 | +0.012 / 48% / 1,300 | +0.006 / 49% / 1,359 | 0.000 / 50% / 1,357 | +0.009 / 48% / 1,380 | +0.013 / 47% / 1,419 | +0.014 / 46% / 1,350 | +0.012 / 47% / 1,345 |
| 8-13 | +0.024 / 48% / 1,163 | +0.041 / 46% / 1,606 | +0.040 / 45% / 1,650 | +0.023 / 45% / 1,682 | +0.031 / 45% / 1,648 | +0.036 / 43% / 1,619 | +0.020 / 45% / 1,513 | +0.021 / 45% / 1,510 |
| 13-21 | +0.037 / 46% / 1,092 | +0.055 / 43% / 1,448 | +0.051 / 43% / 1,447 | +0.038 / 45% / 1,439 | +0.025 / 45% / 1,446 | +0.031 / 44% / 1,402 | +0.016 / 46% / 1,439 | +0.017 / 46% / 1,467 |
| 21-51 | -0.075 / 55% / 1,167 | +0.018 / 48% / 1,605 | +0.022 / 47% / 1,767 | +0.023 / 46% / 1,836 | +0.020 / 47% / 1,788 | -0.014 / 51% / 1,772 | -0.005 / 51% / 1,831 | -0.003 / 50% / 1,802 |
| 51+ | -0.476 / 69% / 1,670 | -0.049 / 53% / 1,900 | -0.058 / 53% / 1,939 | +0.119 / 42% / 1,862 | +0.045 / 47% / 1,889 | -0.088 / 57% / 1,906 | -0.095 / 60% / 1,878 | -0.090 / 60% / 1,887 |

The sd of the log move at 08:00 is 0.17 (1.5-2) to 0.49 (13-21) and 1.02 (51+); at T-1 0.07 to 0.25 and 0.45. By
country at 08:00 (win): GB +0.012 (3-5), +0.002 (5-8), +0.043 (8-13), +0.043 (13-21), -0.006 (21-51), -0.407 (51+);
IE +0.007, -0.014, -0.089, +0.012, -0.233, -0.670 on 115 to 700 runners. A back at the 08:00 mid laid off at the BSP
(win, mid/BSP - 1 a GBP1 stake, the mid as if takeable, before commission): mean +0.020 (3-5), +0.049 (5-8), +0.053
(8-13), +0.070 (13-21), +0.291 (21-51), +2.10 (51+); median -0.010, 0.000, -0.023, -0.037, +0.078, +0.609. The means
above 21 are the wide book's mids, not prices.

### 7.4 betfair_live_markets (A4)

October 2026: GB WIN 930 markets (8,971 runners, 6,314 joined to race_results), IE WIN 332 (4,204 / 2,751), GB PLACE
920 (8,868 / 6,287), IE PLACE 332 (4,197 / 2,751); one market without a type (5 runners).

### 7.5 Betfair Historic Data (B1) and football gold (B2)

`sources/betfair_historic/`: 261,008 raw files, 403 MB, greyhound racing 2026 only; `markets_greyhound_racing_2026.
parquet`: 3,054,761 runners, 251,748 markets, 9,279 events, offs 2025-12-10 to 2026-09-30; non-null ltp_first
1,863,552, ltp_t_1440 156, ltp_t_60 37,413, ltp_t_15 101,619, ltp_t_1 1,154,352, ltp_close 1,863,552, bsp 875,531, won
2,896,713. Markets by country and type: AU EXACTA 16,777, MATCH_BET 34,221, MONEY_BACK_2ND 3,296, PLACE 34,203,
QUINELLA 16,768, WIN 34,394; GB ANTEPOST_WIN 1, FORECAST 35,385, PLACE 35,306, WIN 35,306; IE PLACE 45, WIN 87; NZ
PLACE 2,963, WIN 2,996. Markets that went in play: 0.0%.

Football gold: 299,646 matches (finished 299,440, scheduled 202, no result 4), 1993-07-23 to 2026-11-02. Coverage of
finished matches by season (share with the benchmark price): 1X2 open 0.98-1.00 in 2002-2011 (the main leagues only),
0.62-0.67 in 2012-2025 (the extra leagues carry a close only), 0.41 in 2026 so far; 1X2 close 0.97-1.00 from 2012;
O/U 2.5 open 0.25-0.30 in 2002-04 then as 1X2, close from 2019; AH open from 2003, close from 2019; kickoff_utc from
2019; Betfair Exchange open 0.614 / 0.578 / 0.404 and close 0.858 / 0.933 / 0.943 in 2024 / 2025 / 2026. Sources: 1X2
open PS 7,547-7,813 a season to 2023, BFE 7,181 (2024), 6,896 (2025), 1,954 (2026); 1X2 close PSC 11,032-12,481 to 2024,
BFEC 11,513 (2025), 4,664 (2026); O/U open BbAv to 2018, P 2019-2023, BFE from 2024; AH the same pattern (BbAvAH, PAH,
BFEAH). 108,173 finished matches with a 1X2 open and close. 1X2 home open -> close: n 9,929 / 26,398 / 47,632 /
18,192 / 4,056 / 1,549 / 379 in the bands 1-1.5 to 13-21; median log move 0.000, 0.000, +0.005, +0.007, +0.013,
+0.009, -0.005; shortened 46.6%, 46.8%, 45.7%, 45.7%, 45.6%, 47.2%, 51.2%; sd 0.047 to 0.211. Over 2.5 open -> close:
0.000 (1-1.5, n 3,195), +0.006 (1.5-2, 28,366), +0.012 (2-3, 23,468), +0.020 (3-5, 267); shortened 47%, 42%, 39%, 36%.
odds_long 10,149,855 rows (26 bookmakers at the 1X2 open, 18 at the close; 9 / 5 for AH; 8 / 5 for O/U).
betfair_market_map: no data found. dq/latest.json (2026-11-01T0542Z): betfair_markets_linked 0, openfootball_linked
71,658, api_football_linked 0; issues: score_vs_openfootball high 188, season_match_count high 44, open_overround low
469, scheduled_in_past low 137.

### 7.6 Greyhound tables (B4), the other sources (B3, B5), the recordings (C1)

GBGB: 3,226 raw day files; runs 2018-2026 as in section 2 (races 107,417 / 110,298 / 88,404 / 98,693 / 96,315 /
98,024 / 94,897 / 87,389 / 68,780). Greyhound price tables, rows and GB / AUS win markets: 2012 187,716 (11,322 /
4,299, from 2012-09-18); 2013 1,441,906 (70,142 / 52,789); 2014 1,534,497; 2015 1,510,594; 2016 1,577,857; 2017
1,572,376; 2018 1,152,627 (110,137 / 45,751); 2019 893,653 (102,125 / 40,148); 2020 999,457; 2021 1,244,213; 2022
1,256,851; 2023 1,306,998; 2024 1,301,799 (104,493 / 93,763); 2025 1,232,102 (97,974 / 92,075); 2026 970,848 (77,285 /
75,160, to 2026-10-30). Before 2019 the files carry no volume (median morning and pre-play GBP0 for every country);
"FC" menu hints (2012-17) are forecast markets. GB win morning money by year: median GBP0 every year; 90th pct
GBP479 (2019), 598 (2020), 1,182 (2021), 1,016 (2022), 766 (2023), 0 (2024-26); share with any morning trade 44%
(2019), 23%, 20%, 20%, 17%, 9%, 9%, 9% (2026); GB pre-play median GBP1,781 (2019), 2,462, 2,755, 2,444, 2,769, 2,206,
1,717, 1,784 (2026). AUS "morning" money (a window that includes post-off trading for AU, `reports/other_markets_and_
stakes_1002.md` s.4): median GBP623 in 2026. GB win morning WAP -> BSP where a morning WAP exists: +0.137 (2-3, n
50,446), +0.146 (3-5, 169,623), +0.159 (5-8), +0.154 (8-13), +0.215 (13-21), +0.209 (21-51); shortened 20-38%.

Tennis: 166 raw files; sackmann_matches 1,268,349 rows, tourney_date 1999-12-27 to 2026-06-02 (atp 704,194, wta
564,155; sackmann 1,190,268, tml 78,081); odds_matches.parquet not held. openfootball: 211 raw files, 78,313 matches
2010-07-17 to 2027-05-30, no price. api_football and puntingform: nothing.

`betfair_live/<day>/`: 367 objects over 32 days (2026-10-01 to 2026-11-01):

| File | Days | First | Last | Total MB | Median MB a day |
|---|---|---|---|---|---|
| books.csv.gz | 32 | 2026-10-01 | 2026-11-01 | 144.0 | 4.79 |
| books_trader.csv.gz | 32 | 2026-10-01 | 2026-11-01 | 237.9 | 7.88 |
| books_other_place.csv.gz | 27 | 2026-10-06 | 2026-11-01 | 137.3 | 5.18 |
| books_greyhound.csv.gz | 27 | 2026-10-06 | 2026-11-01 | 76.7 | 2.85 |
| books_evening.csv.gz | 28 | 2026-10-05 | 2026-11-01 | 7.5 | 0.26 |
| books_antepost.csv.gz | 25 | 2026-10-07 | 2026-10-31 | 0.5 | 0.02 |
| census_sports.json | 25 | 2026-10-07 | 2026-10-31 | 0.4 | 0.02 |
| markets.csv.gz, markets_trader, markets_other_place, markets_greyhound, markets_evening, markets_antepost | 32 / 32 / 27 / 27 / 28 / 25 | | | 1.2 / 0.3 / 1.0 / 1.2 / 0.5 / 0.4 | |

### 7.7 The census (C2)

Section 5.

### 7.8 The Historic Data first -> close move, greyhounds (D)

1,863,552 runners with a first and a closing price in 237,744 markets. By market type and first-price band (median
log(close/first); shortened): WIN +0.023 (1.5-2, n 10,606), +0.046 (2-3, 37,408), +0.024 (3-5, 86,953), +0.032 (5-8,
94,315), +0.069 (8-13, 75,848), +0.080 (13-21, 48,887), +0.113 (21-51, 53,382), 0.000 (51+, 40,480); shortened 43%, 41%,
44%, 43%, 40%, 39%, 37%, 46%. PLACE: median 0.000 in every band (shortened 24-40%: most place runners trade once).
FORECAST, EXACTA, QUINELLA, MONEY_BACK_2ND, MATCH_BET: median 0.000 (one trade a selection is the norm).

WIN first -> close by country (median log move; shortened; n):

| Band | GB | AU | NZ |
|---|---|---|---|
| 1-1.5 | +0.418 / 17% / 705 | -0.022 / 57% / 2,205 | -0.014 / 52% / 233 |
| 1.5-2 | +0.161 / 32% / 2,420 | +0.010 / 46% / 7,549 | -0.003 / 50% / 616 |
| 2-3 | +0.062 / 40% / 18,026 | +0.037 / 42% / 17,962 | +0.013 / 46% / 1,378 |
| 3-5 | +0.028 / 43% / 51,528 | +0.016 / 45% / 32,834 | +0.013 / 48% / 2,512 |
| 5-8 | +0.051 / 41% / 56,841 | +0.017 / 46% / 34,171 | 0.000 / 50% / 3,223 |
| 8-13 | +0.105 / 36% / 35,951 | +0.041 / 43% / 36,002 | +0.043 / 43% / 3,758 |
| 13-21 | +0.129 / 34% / 13,894 | +0.059 / 41% / 31,811 | +0.118 / 37% / 3,149 |
| 21-51 | +0.118 / 37% / 7,905 | +0.100 / 37% / 41,896 | +0.266 / 27% / 3,525 |
| 51+ | -0.266 / 60% / 2,615 | +0.051 / 45% / 36,152 | +0.005 / 42% / 1,654 |

GB WIN first -> BSP (where the market has one; n 698 to 56,703): +0.420, +0.148, +0.056, +0.025, +0.050, +0.103, +0.129,
+0.117, -0.262 in the same bands; shortened 17%, 33%, 41%, 45%, 42%, 36%, 34%, 37%, 61%; IQR at 3-5: -0.133 to
+0.218; at 8-13: -0.107 to +0.370. IE: 137 runners (8-13 band only): +0.151.

WIN, the marks -> close (all countries; median log move; shortened; n): T-60: +0.030 (1-1.5, 944), +0.045 (1.5-2,
2,446), +0.061 (2-3, 5,068), +0.047 (3-5, 4,829), +0.056 (5-8, 3,216), +0.091 (8-13, 3,645), +0.229 (13-21, 2,463), +0.346
(21-51, 2,734), +0.095 (51+, 5,879); T-15: 0.000, +0.025, +0.045, +0.022, +0.056, +0.132, +0.163, +0.182, 0.000 (n 1,631 to
19,783); T-1: -0.016 (1-1.5, 3,535), -0.006 (1.5-2, 11,393), 0.000 (2-3, 33,876), 0.000 (3-5, 86,402), +0.017 (5-8,
85,391), +0.021 (8-13, 70,035), +0.030 (13-21, 50,394), +0.044 (21-51, 58,346), +0.057 (51+, 44,015); shortened at T-1
59%, 52%, 48%, 45%, 43%, 42%, 42%, 40%, 41%. A price standing 24 h before the off: 78 of 451,031 WIN runners (0.0%),
76.9% of those equal to the first traded price.

WIN close against the result (calibration): won 0.734 vs implied 0.758 (1-1.5, n 4,120); 0.562 vs 0.570 (1.5-2,
11,619); 0.399 vs 0.401 (2-3, 34,692); 0.259 vs 0.260 (3-5, 82,601); 0.163 vs 0.162 (5-8, 84,452); 0.104 vs 0.101
(8-13, 70,421); 0.065 vs 0.063 (13-21, 50,822); 0.034 vs 0.033 (21-51, 60,252); 0.009 vs 0.010 (51+, 49,664).

### 7.9 What the run changes in the reading above

1. Greyhounds: the Historic Data "first traded price" is not an open in the trade's sense (it stands minutes before
   the off and carries no volume); the drift from it to the BSP is real in direction (the first price under the BSP,
   GB +2.5% to +13% in the 3-51 bands) but the recorded books from 6 Oct are the only data with a takeable price, its
   time and its money. The research agent's lay-early test runs on those.
2. Horses: the 08:00 win mid is two-sided on 97.8% of runners with GBP6 at the best back and an 18.9% spread, and
   its median move to the BSP is within +-0.04 from 1.5 to 21: the open exists, the market-only drift does not; the
   long-shot shortening at 08:00 is the spread, not a trade.
3. Football: the benchmark open -> close move is centred on zero in every band; the data to measure Betfair's own
   open-to-close is one free plan away (Soccer BASIC), and the census shows the money is there a day ahead.
4. Tennis: no price held at all; the census shows a median next-day market of GBP5.
5. Cricket and golf: the deepest next-day money outside football (GBP6.8m and GBP654k a day on the top 200); both
   outside the brief as written, both a free BASIC plan away if the research agent wants the pre-off read.
