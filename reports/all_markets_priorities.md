# Every market and sport: where to build next, and the data for each (the owner's ask, 3 Oct 2026)

The owner asked which other Betfair markets we can reach, across every sport. The ask was to rank them and to say
what beating the closing price (CLV) would take in each: sourcing the history, the features, the model, the trading,
and the monitoring. Then to make the changes and source the data for each candidate model. This note records the
ranking and what is now being sourced. The six stages each market goes through are those in
`reports/market_research_programme.md`.

## 1. What we can reach

| Route | What it covers | Held |
|---|---|---|
| Betfair's price files (promo.betfair.com) | Horse racing (UK, IE, AU, FR, USA, RSA, UAE) and greyhounds (UK, AU), win and place, with the BSP | Archived in S3 from 2017 (UK/IE) and 2018 (the rest): `betfair_prices.py --archive` |
| The Exchange API (our app key; the delayed key for now) | Every sport Betfair lists | Live books of GB/IE win and place from 1 Oct (`betfair_recorder.py`). From this change also: 2/3/4 TBP, each way, GB/IE greyhounds, ante-post, and a daily census of every sport |
| Betfair Historic Data (historicdata.betfair.com) | Every sport. The BASIC plan (last traded price each minute, definition, result) is free once "bought" per sport | From this change: `betfair_historic.py`, nightly on the UK server, once the account holds a sport |

Outside racing and greyhounds there is no BSP. So CLV there is measured against the last price traded before the
market turned in play (`ltp_close` in the Historic Data tables).

## 2. The ranking

| Priority | Market | Why | Kind of edge |
|---|---|---|---|
| 1 | UK/IE place, 2/3/4 TBP, each way | Same races and form as the win model. The place rule passed its gate at 2% (ledger `place-pocket-gate`). The cheapest to add | Medium volume, medium edge |
| 2 | Greyhounds (UK first) | The most races (about 259 a day with Australia). Has a BSP and the same file format. A softer market | High volume, small edge |
| 3 | Australian racing (win) | GBP1.2k a runner, 81 races a day. Needs a paid form feed and overnight running | Medium |
| 4 | Tennis | Very many matches, free data, a well-understood model; the edge is likely in the lower tiers and in play | Small to medium |
| 5 | Ante-post racing (the big races) | Our model prices big fields weeks ahead, in a softer market; few bets | Low volume, larger edge |
| 6 | Football | The deepest market and the sharpest close. Only secondary markets, lower leagues or in-play are realistic | Small; a long project |
| 7 | Darts, snooker | Elo-able, softer, small | Small |
| 8 | Cricket, golf, US sports | In-play only (cricket), paid data (golf), sharp markets (US) | Parked |
| - | France/USA/SA racing win; politics and specials | Too thin for a GBP250-to-win rule; specials do not repeat | Dropped |
| - | UK/IE in-play offsets; SP-bias segments | Failed out of sample (ledger, 2 Oct) | Closed |

Where sections 1 and 2 rest on general knowledge of these markets rather than our own figures (every sport outside
racing and greyhounds), the census of every sport (section 4) replaces that knowledge with numbers within two weeks.

## 3. What beating the close takes, by market

| Market | Source the history | Engineer features | Build a model | Bet and trade | Monitor |
|---|---|---|---|---|---|
| Place / TBP / each way | Place price files from 2017, HRB place BSP from 2010; **from now** the live 2/3/4 TBP and each-way books (no files exist) | Win-implied place chance (Harville-type, fitted exponents), place terms, place-book overround, the gap between the win-implied and the place price | Finishing-order model on our win probabilities; a closing model for the place SP like `model/race_book.py` | T-15: back at the place SP (MARKET_ON_CLOSE) where the expected return clears 5% after 2%; later whole-race win+place books | CLV vs place BSP; forward test on the recorded books before money |
| Greyhounds | Greyhound price files from 2018 (archive); **GBGB results from 2018** (trap, sectional, run and calculated times, comment, weight, SP, grade, breeding); **live GB/IE books from now** | Trap bias by track and distance, sectionals, calculated times, grade moves, freshness, trainer form; the form windows carry over (trap for draw) | LightGBM on log(BSP) + race-level conditional logit; the edge scan on the market alone first | Small edges at volume: SP backs/lays, early entry locked at the SP; whole-race books on 6-runner fields | Per-track, per-trap CLV: readable within weeks |
| Australian racing | AU price files (pre-play price only); **Punting Form** (meetings, fields, results) once a key is supplied | Port `custom_metrics` / `form_windows` to that schema; barrier bias, track condition, rail | Retrain the BFSP pipeline for the jurisdiction | 01:00-09:00 UK: a second runner or moved archive window | CLV vs AU BSP, separate from UK |
| Tennis | **Sackmann ATP/WTA** (stats; tour, Challenger, ITF), **tennis-data.co.uk** (closing odds incl. Pinnacle), **Betfair Historic Data** MATCH_ODDS | Surface Elo, serve/return points won (opponent-adjusted), fatigue, retirements, head to head | Point-based (serve/return -> match) blended with Elo; calibrated against Pinnacle's close first | Pre-off in the lower tiers; in-play later (Stream API, latency) | CLV vs Betfair's last pre-off price, by tour and tier |
| Ante-post | HRB form; **daily ante-post book snapshots from now** | The win model's features projected to the big race; the field's likely make-up | Our model over the expected field | Few, larger bets weeks ahead; non-runner risk priced | CLV vs the race's BSP |
| Football | **football-data.co.uk** (results, stats, opening/closing odds incl. Pinnacle and Betfair, 38 leagues); **Betfair Historic Data** match odds, O/U 2.5, Asian handicap | Team ratings (Dixon-Coles / Elo / xG), rest, line-ups | Poisson / bivariate team model; aim at O/U, AH, lower leagues | Early prices; a line-up model | CLV vs Betfair and Pinnacle closes; +1-2% is a good result |

## 4. What this change sources

| Source | What | Where it lands | How often | Needs |
|---|---|---|---|---|
| GBGB (`sources/gbgb.py`) | Every GB greyhound meeting from 2018: each dog each race | `s3://ashcroft/sources/gbgb/raw/<year>/<day>.json.gz`, `gbgb/runs_<year>.parquet` | `source-data.yml` nightly (GitHub's runners), backfill newest first over the first nights | Nothing |
| football-data.co.uk (`sources/football.py`) | 22 main divisions from 1993/94, 16 extra leagues from 2012 | `sources/football/raw/...`, `football/matches.parquet` | Nightly; the current season again each run | Nothing |
| Tennis stats (`sources/tennis.py`) | Sackmann's ATP/WTA matches from 2000 (tour, Futures, qualifying/ITF) and players, from the archive mirror (Sackmann's own repositories were taken down in 2026; snapshot June 2026); TML's ATP tour-level matches for the months since | `sources/tennis/raw/...`, `tennis/sackmann_matches.parquet` | Nightly; this and last year again | Nothing. Both are licensed non-commercial (CC BY-NC-SA): check before staking money on a model trained on them |
| Tennis odds (`sources/tennis_odds.py`) | tennis-data.co.uk results with closing odds (Pinnacle, Bet365, max, average) | `tennis/odds_matches.parquet` | `betfair-prices.yml` on the UK server, ten minutes a night: its Cloudflare refuses GitHub's runners (403 on every file, 3 Oct) | Nothing, if the UK server gets through; else Betfair Historic Data is the tennis close |
| Punting Form (`sources/puntingform.py`) | AU meetings, fields and results a day | `sources/puntingform/raw/...` | Nightly, once keyed | **Owner: a subscription and the secret `PUNTINGFORM_API_KEY`** |
| Betfair Historic Data (`betfair_historic.py`) | BASIC files for greyhounds, tennis, soccer, cricket, darts, snooker; tables of first / T-24h / T-60 / T-15 / T-1 / closing price and result | `sources/betfair_historic/raw/<sport>/...`, `betfair_historic/markets_<sport>_<year>.parquet` | `betfair-prices.yml` nightly on the UK server, an hour before the price-file archive; tables built in `source-data.yml` | **Owner: "buy" the free BASIC plan for each sport at historicdata.betfair.com with the trading account** |
| 2/3/4 TBP and each-way books | Every snapshot, as for win/place | `betfair_live/<day>/books_other_place.csv.gz` | Beside the trader (`live-trade.yml`) or the recorder (`live-record.yml`) | Nothing |
| GB/IE greyhound books | Win and place, every snapshot to 21:30 UK | `betfair_live/<day>/books_greyhound.csv.gz` | As above | Nothing |
| Ante-post books | Every GB/IE ANTEPOST_WIN market, once a day | `betfair_live/<day>/books_antepost.csv.gz` | `live-record.yml`, after the last race | Nothing |
| Census of every sport | Each sport's market types and matched money on its top 200 markets of the next 24 h | `betfair_live/<day>/census_sports.json` | `live-record.yml`, after the last race | Nothing |

None of these touch `horse_racing.db`. The new live records have their own tagged files, which the nightly load does not
read, so the existing tables and queries are unchanged. The greyhound record stops at 21:30 UK with the others, so the
late evening's dog races are left to the price files and GBGB.

### The first run (3 Oct, source-data.yml run 37106841528, 2 h 27 min)

| Source | Fetched | Table |
|---|---|---|
| GBGB | 3,197 days, 1 Jan 2018 to 2 Oct 2026 (8 days reached the API's 500-race list cap, so some meetings may be missing) | 853,412 races, 3.35m runs; `runs_2018` 107k races ... `runs_2026` 62k |
| football-data.co.uk | 732 files (32 division-seasons not published) | 300,254 matches, 38 leagues, 23 Jul 1993 to 2 Oct 2026; 25,679 with Betfair's closing price |
| Tennis | none: Sackmann's repositories 404 (taken down), tennis-data.co.uk 403 (Cloudflare) | moved to the mirror and TML, and the odds to the UK server (this change) |
| Punting Form | none: no key | |
| Betfair Historic Data | none yet: runs on the UK server after merge, once the account holds a sport | |

## 5. The owner's to supply

1. At historicdata.betfair.com, signed in with the trading account: the free BASIC plan for Greyhound Racing, Tennis,
   Soccer (and Cricket, Darts, Snooker if wanted). Until then `betfair_historic.py` fetches nothing and says so.
2. A Punting Form subscription and the secret `PUNTINGFORM_API_KEY`, only if Australian racing goes ahead.
3. The live application key (a one-off fee, about GBP299 when last published), which restores the traded volume
   the delayed key leaves out. It mattered for racing; it matters more where only the books show liquidity.
4. Licences: check the Sackmann, TML and football-data.co.uk terms before any of their data trains a model that stakes money.

## 6. Next

1. When the first fortnight of the census is in: the liquidity table by sport and market type. Then confirm or
   re-order ranks 4-8.
2. Greyhounds: the edge scan on the price files (2018 on). Then GBGB joined to them (track, time, trap and name) and
   the stage-2 coverage check.
3. Place: the T-15 rule on the recorded books, and the first look at the 2/3/4 TBP and each-way books.
4. Tennis: Sackmann joined to tennis-data and to the Betfair closes; a surface-Elo baseline against Pinnacle's close.
