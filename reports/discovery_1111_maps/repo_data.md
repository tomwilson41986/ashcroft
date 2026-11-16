# MAP: repo_data

# MAP repo_data - what racing² holds, can fetch, lacks (11 Nov 2026)

## 1. HELD NOW (UK/IE horses unless said)
- **race_results** (HRB CSV; `scraper.py:121-193`) - date, time, track, race_name, age restriction, class, major, distance text/furlongs/yards, prize, going, runners, place, distbt, total_dst_bt, horse, stall, trainer, age, jockey, claim, pounds, odds (returned industry SP), fav, OR, comptime(+numeric), median/max OR, placing_numerical, race_code, bfsp, bfsp_place, plcs_paid, bf_plcs_paid, rail_move, race_type, comment (in-running), card_no, stall_positioning, track_direction, headgear, days_since_lr, career_runs, stallion, dam, dam_stallion, surface, horse_prizewin, sex. 1,938,313 rows 2010-2026 (`RESEARCH_FRAMEWORK.md:330`); complete from 2010, 115-129k runners/yr, comments from 2012; runners only, no non-runners (ledger `career-truncation`; `reports/hrb_system_builder_review.md:62`). Post-race; history known pre-06:00.
- **betfair_prices** (`betfair_prices.py:355-365`) - win+place per runner: bsp, ppwap, morningwap, ppmax, ppmin, ipmax, ipmin, morning_vol, pp_vol, ip_vol, win_lose, selection_id. No timestamps, no ladder. 2,136,686 rows 31 Dec 2017-30 Oct 2026 (ledger `open-close-inventory-1101`); place in-play columns are placeholders (`model/blocks/inrunning.py:5`). Post-race (next day). Raw S3 archive: 16 markets incl. AUS/FR/USA/RSA/UAE/greyhound (ledger `price-archive-inventory-1002`).
- **betfair_live_markets / _marks / live_orders** (`betfair_recorder.py:54-74,634-660`) - catalogue: cloth, stall, jockey, trainer, age, weight, OR, form string, days since run, wearing, Betfair forecast price, each-way divisor; books: best 3 back/lay + sizes, last traded, market matched (no runner volume on the delayed key), status, adjustment factor, removal time, sp_near/far; marks 08:00-12:00 hourly, T-120/60/30/15/10/5/3/1, last, settled BSP. From 30 Sep 2026; 371,804 marks, 31 days at 1 Nov (`open-close-inventory-1101`); polls 300 s, 60 s in the last hour. Also books_evening (17:00-21:30 the eve, from 4-5 Oct; ledger `evening-before-1004`), OTHER_PLACE/EACH_WAY, ante-post, sport census. live_orders = trader ledger (`trading/session.py:33`).
- **Exchange Stream, horse WIN** (`research/horse_stream_record.py`; `greyhound/stream_record.py:38`) - raw mcm: full ladder, traded ladder/volume, LTP, market definition, SP traded/projected, 500 ms conflation, S3 `betfair_stream/horse/<day>/`. First morning 12 Nov 2026; none held 10-11 Nov (ledger `stream-late-signal-1111`).
- **Betfair Historic Data BASIC, Horse Racing all countries from Jan 2024** (`au/historic.py:1-9`) - WIN: first traded, last traded at T-24h/60/15/1, close, BSP, result; no volume, no time of first trade (`sources/open_close.py`).
- **blandford_results** (Timeform feed; `blandford_sync.py:55-71`) - pre-race master/adjusted rating, performance rating, timefigure, finishing time, leader/winner/distance sectional, ISP, Betfair win/place SP, ip_min/max, bsp_advantage, xWR/xWM, draw, owner, sire/dam/damsire, foaling_date, horse_code, RP OR/speed/RPR, headgear, weight_kg, prize won, hot_race. 2021-01 on, bulk-loaded 23 Sep 2026, so historic revisions unknown; first_seen/revisions tables from then (ledger `blandford-revisions`). Same-day fields BANNED (results, rewritten 4 days; `reports/external_data_review.md:13,108`); lagged only.
- **Timeform archive extract** `data/external/timeform_pace.csv.gz` - race pace strength, contest, race finishing-speed %, pre-race pace-map ideal position; Flat/AW 2016-Feb 2026 (`scripts/build_timeform_pace.py`).
- **Sectionals** - `data/external/sectionals.csv.gz` (per-furlong, finishing speed, stride length/cadence; 67,723 runs 30 Jun 2024-29 Jun 2025; 42,783 from 18 Apr 2026; Flat/AW only; `external_data_review.md:36`); blandford_atr (finish %, early/mid/final/top mph, segment times; 107,744 runs with sectionals 12 Jan 2020-11 Dec 2025; ledger `blandford-sync-1023`); blandford_rtv (finishing speed, 0-20 mph, top speed, stride + ranks; 11,283 runs, fetch stopped; ledger `rtv-off-1031`); sectional_runs (fsp, fsp_rel, fsp_vs_pos, early_pct, late_rel, sl_rel; `blandford_sectionals.py:210`). Jumps: none.
- **Sales** `data/external/timeform_sales.csv.gz` - foal/yearling/breeze-up/store price, 3 Jul 2025 snapshot, unusable before it (`model/blocks/sales.py`).
- **Cards + forecasts archive** S3 `racecards/<day>_HHMM.csv`, `predictions/<day>.csv`; 06:00 files from 14 Jul 2026 (ledger `card-gaps-check`).
- **Oddschecker** - ~25 bookmakers' price per GB/IE runner, morning book + T-15, from 1 Nov 2026 only (`reports/open_close/parse_odds.md:13,58`; `.github/workflows/parse-odds.yml`).
- **Tote** S3 `tote/live/<day>/` - pool totals (win/place/exacta/trifecta/swinger/quinella/placepot), win/place approximate dividend per runner T-60..T+5, bookmakers' show price, declared dividends; from 6 Oct 2026; no pre-off exotic combination dividends (`tote_recorder.py`; `reports/tote_programme.md`).
- **Derived in code**: 86 course coordinates (`model/blocks/travel.py`); coarse track geometry (`model/abm/track.py`; `RESEARCH_FRAMEWORK.md:46`).
- **Raw fields already mined by blocks** (so not new): industry SP vs BSP spread and win/place SP ratio (`model/blocks/market_history.py`); in-running low/high vs BSP (`inrunning.py`); comment classes - trouble, keen, wide, eased (`comments.py`); jockey-trainer surname match, last race's winner's BSP and non-finishers (`hrb_extras.py`); yard base as centroid of its runners' courses, km travelled (`travel.py`); jockey bookings as the market rated them (`bookings.py`); stablemates and yard jockey share (`stable.py`); yard's debutants' prices (`debut_market.py`); clock speed figure from comptime (`time_figure.py`).
- **betfair_odds** snapshots table (`db_schema.py:20`); hrb_ratings table defined, absent at 23 Sep (`reports/qa_review_2026-09-23.md:227`).
- **Other sports/countries**: AUS BSP targets from Sep 2012 (`au/data.py`); JRA 47,596 runners (body weight, last 600 m, corner positions) and HK 10,810 runners + sectionals, Nov 2025-Oct 2026 (ledger `parse-jra-hk-backfill-1101`); GBGB 3.38m runs 2018-26 (`open-close-inventory-1101`); football, tennis; NOAA hourly weather at greyhound tracks only, 2014-Aug 2025 (`sources/noaa_isd.py`).

## 2. FETCHABLE WITH EXISTING CODE
- HRB one-day card and 48-hour declarations (`daily_predictions.py:203,579`); HRB Ratings Machine (15 pre-race rating sets, throttled; `hrb_ratings.py`).
- Betfair live: catalogue, books, BSP, ante-post, any country (`betfair_recorder.py`); stream (live key).
- Betfair price files; Historic Data BASIC, any sport (`betfair_historic.py`).
- Blandford backend tables: attheraces, racingtv (`blandford_sectionals.py`).
- Punting Form AUS form (key needed; personal-use terms; `sources/puntingform.py`, `reports/international_racing.md` pt 4).
- Parse: Oddschecker, netkeiba, HKJC (credits; `sources/parse_odds.py`, `sources/parse_racing.py`).

## 3. NOT HELD
- Weather at horse courses: none; ERA5 or Met Office DataHub named, each needs an account (`sources/noaa_isd.py`; `reports/model_review_1111.md:73,92`).
- GoingStick: none; BHA Racing Admin scrape overwrites daily (`external_data_review.md:40`).
- Full sectionals/GPS: partial Flat/AW only, gap Jul 2025-Apr 2026 in site files; no jumps; backend `sectionsparsed` not pulled (ledger `repos-review-1022`).
- Entries/declarations history, reserves order, non-runner history: none; dated copies possibly in the external scraper's S3 (`external_data_review.md:89`).
- Timeform pre-race tips, flags, analyst verdict, early-pace figure, smartStats: dropped by the scraper, never stored (`external_data_review.md:39`).
- Racing Post card (evening-before best odds, price history, tips, Spotlight, RPR/TS): replaced each run, no history (`external_data_review.md:41`).
- Bookmaker price history: only returned ISP; no forecast/early/opening show (`hrb_system_builder_review.md:94,199`); Oddschecker from 1 Nov.
- Wind operations (HRB holds, BHA declares since 2018), jockey sex/minimum weight, owners/colours: not in the HRB export (`hrb_system_builder_review.md:194-198`); owner and foaling date exist in blandford_results only.
- Gelding since last run: not knowable on the card (`model/card_enrich.py:25`).
- Horse body weight: UK none (JRA/HK only). Stewards' reports: UK none (Punting Form carries AUS). Vet records, gallops/workouts, video/tracking: no data found, no source named.
- Betfair: runner-level traded volume and full ladder before 12 Nov; evening prices before 4 Oct 2026; in-play tick path (only min/max); match bets not offered (ledger `market-census-1003`).
- "As of now" pedigree, sire, trainer-uplift tables: leak in a backtest (`external_data_review.md:42`).

## 4. TIMING
- **Evening before**: 48-hour declarations card; Betfair evening books; Racing Post/Timeform pre-race content exists externally, unstored.
- **06:00 (card fetched 04:00 UTC)**: time, track, class (blank 27%; `model_review_1111.md:43`), distance, morning going, prize, race name; per runner cloth/stall, name, colour/sex/sire/dam tooltip, age, weight before claim, jockey, trainer, OR, headgear, days, card odds (`daily_predictions.py:315-360`). Columns Form, Miles, HRB Total, CD, S unread; owner and foal date cut (`daily_predictions.py:580,654`). Filled from history: furlongs, race type, surface, direction, sex, pedigree, career runs, claim, max/median OR (`model/card_enrich.py:14-38`). Field includes later non-runners (351 of 5,932; ledger `morning-card-gap-1110`). All history to yesterday; Betfair files lag a day.
- **05:15-10:00 (trader's window; morning models at 10:00; `CLAUDE.md`)**: Betfair books (thin: median market matched GBP1,423 at 08:00; `open-close-inventory-1101`), Oddschecker morning book, non-runners, going changes (27% differ; ledger `card-vs-results-1110`), jockey changes.
- **Near the off**: T-120..T-1 marks, ladder/flow (stream), Tote dividends, SP projections, final field and reduction factors.
- **After**: result, distances, time, comment, BSP, in-play min/max, volumes, Timeform ratings/timefigure/sectionals (revised up to 4 days), tracking sectionals.

## Blind spots of this map
- No database or S3 was reachable: every row count and date range is quoted from a report or ledger entry as of that entry's date (mostly 26 Sep - 1 Nov 2026), not re-counted today. Current totals will be higher.
- Exchange Stream: the code says it records to 21:30 UK from 11-12 Nov, and the ledger (`stream-late-signal-1111`) found 0 parts on 10-11 Nov. Whether any stream data exists yet is unconfirmed.
- hrb_ratings: the table was absent on 23 Sep 2026 (`reports/qa_review_2026-09-23.md:227`); I found no later statement that it has been loaded. Coverage unknown.
- betfair_prices start: one ledger entry (`hrb-features-1006`, 6 Oct) says price-file records start 11 Apr 2019 with 84.8% of runs covered; a later one (`open-close-inventory-1101`) says 31 Dec 2017. I used the later; the fill of 2018-19 is not verified.
- The HRB CSV card path (`parse_racecard_csv`) was not read column by column; the card field list is from the HTML path and `model/card_enrich.py`. `predict_bfsp_today.py` and `scripts/fetch_data.py` were only skimmed (the latter just downloads the DB and model files).
- daily-horse-scraper was not opened (rule). What it holds (Timeform morning response, entries/declarations, Racing Post card, GoingStick) is taken from `reports/external_data_review.md` and ledger `repos-review-1022` only; whether its S3 `race_entries` files go back a season is unknown to the repo too.
- The Timeform yearly archives (2015 - 15 Feb 2026) live in another repository; only the extract `data/external/timeform_pace.csv.gz` and its build script were seen.
- Tote and Oddschecker record continuity (missed days) was not checked; start dates are from the reports.
- Vet records, gallops/workouts, video/tracking, stewards' reports for UK/IE: no file mentions a source; "no data found" means absent from the repo, not that no public source exists (no web search was run in this phase).
- Licence positions (Timeform/GlobalSportsAPI, Racing Post, ATR, RTV, BHA Racing Admin, Punting Form, Parse scrapers) are flagged as unchecked in the reports; not verified here.
- International and other-sport holdings (AUS, JRA, HK, GBGB, football, tennis) are listed in one line each, not field by field.
