# International racing on Betfair: the data to run our model there (research agent, 1 Nov 2026)

The owner's ask (1 Nov 2026): "Look at international horse racing markets
(https://www.betfair.com/exchange/plus/en/horse-racing-betting-7). How can we extend our data coverage so we can also
use our model on them?"

Conventions: hyphens only; every number traces to a repository file, a research-query run named in one, or a URL in
`reports/international_racing_sources.md` (cited here as [S1], [S2], ...). "Unverified" marks a claim with no source.
"ANZ data" means Betfair's own public Australian and New Zealand thoroughbred files ([S1]), which I downloaded and
read for 2023, 2024, 2025 and Sep 2026 (one script, run here, no repository code; figures in section 1.2). Their
volumes carry no currency label; they are presumably AUD (unverified), so they are not set against GBP figures as
if they were the same unit. The Betfair page the owner linked answers 403 to this host, so the list of what Betfair
shows comes from the price files and the census, not from that page. Nothing here was traded or touched S3.

## The result

1. **Betfair's SP exists in seven racing jurisdictions**, the seven it publishes price files for: UK, Ireland,
   Australia, France, USA, South Africa and UAE (`reports/other_markets_and_stakes_1002.md` s.1; `betfair_prices.py`
   lists every file). Hong Kong, Japan, Singapore, South America and India have no price files, so no BSP benchmark
   and no history to study; whether Betfair lists them at all is unverified.
2. **Only Australia is worth building for.** About 47 thoroughbred races a day (17,041 in 2025), a runner's pre-play
   money a median 1,781 (presumably AUD), and a third of each market matched at the BSP (ANZ data, section 1.2).
   France (GBP256 a runner), the USA (GBP300) and South Africa (GBP615) are an order of magnitude thinner than the
   UK's GBP11,821 (`other_markets_and_stakes_1002.md` s.1). New Zealand is an add-on to Australia: 11 races a day,
   325 a runner.
3. **horseracebase gives us nothing abroad.** Its CSV has no country column (`scraper.py` CSV_COLUMNS) and our
   history holds no French meeting (`scripts/build_sectionals.py` docstring). Every jurisdiction needs a new form source.
4. **For Australia the source is Punting Form**: its API returns each runner's last 10 runs with margin, barrier,
   weight, class, prize money, track condition on the 1-10 scale, in-running positions, stewards' reports, gear
   changes and the past **Betfair price (`priceBF`), SP and TAB price** ([S3]). That is close to one-to-one with the
   HRB columns our engine reads. Starter, with API access, is published at $59 a month; history and sectionals are
   Modeller options priced by quote ([S2], to re-check). Its terms say "personal use only" and forbid commercial use
   without written consent ([S4]): **the owner must get that consent in writing before money is staked on it**.
5. **The AU market alone shows no edge to catch.** Over 2023-2025 the pre-play average price sits within -0.010 to
   +0.027 (log) of the BSP for every band from 1 to 51, and a level back at the BSP after 2% commission is
   negative in every band except -0.1% to +1.7% cells that change sign from year to year (section 1.2). As in the UK,
   any edge must come from our model's forecast of the price, so the morning move is what the first study measures.
6. **The model ports as the greyhound model did: its own data module, the same lag-safe engine, its own walk-forward
   model per jurisdiction** (`reports/greyhound_model.md` s.1-3). The 19 custom metrics carry 67.3% of the served
   model's gain, and most of them need only positions, field sizes, prize money and past BSPs, which Punting Form
   carries (`reports/feature_inventory.md`); official ratings, going, class, draw and the comment-based pace features
   need Australian equivalents (section 3).
7. **Operations:** Australian racing runs about 02:00-07:45 UK in winter (5th to 95th percentile of off times, ANZ
   data) and the UK server's one runner is busy then with the price archive (22:30-06:15 UTC, `betfair-prices.yml`)
   and from 05:15 UTC with the UK trader (`live-trade.yml`). A second runner is the operational prerequisite. The
   recorder already takes `--countries AU,NZ` (`betfair_recorder.py` line 892); the trader's filter is a config value
   (`trading/config.py` line 73).
8. **The first step costs nothing:** the owner "buys" the free BASIC Horse Racing plan at historicdata.betfair.com
   (the greyhound bundle carried AU and NZ markets, `reports/all_markets_priorities.md` s.4, so the horse plan very
   likely does too: to confirm), so
   the first traded, T-60 and T-15 prices of Australian races can be set against the BSP; a fortnight of the
   recorder on AU and NZ books measures what a GBP15-200 back can fill. Punting Form comes second, and only if the
   fill is there.

## 1. What Betfair lists, and how much money it holds

### 1.1 The jurisdictions

| Jurisdiction | Price files (BSP) | Races a day | Pre-play money a runner, median | Source |
|---|---|---|---|---|
| UK | ukwin, ukplace from 2017 | 28 | GBP11,821 (morning GBP539) | `other_markets_and_stakes_1002.md` s.1 (run 36965830970) |
| Ireland | irewin, ireplace from 2017 | 11 | GBP4,743 (morning GBP100) | same |
| Australia | auswin, ausplace, archived from 2018 | 81 in the files (8 days, Oct 2026); 46.8 thoroughbred races in the ANZ data (2025) | GBP1,180 (files); 1,781 (ANZ data 2025, currency unstated) | same; section 1.2 |
| New Zealand | none of its own (in the ANZ data only) | 11.2 (2025) | 325 (2025, currency unstated) | section 1.2 |
| France | frwin, frplaced | 16 | GBP256 | `other_markets_and_stakes_1002.md` s.1 |
| USA | usawin, usaplace | 69 | GBP300 | same |
| South Africa | rsawin, rsaplace | 10 | GBP615 | same |
| UAE | uaewin, uaeplace | 0 in early Oct (out of season) | n/a | same |
| Hong Kong, Japan, Singapore, Chile/Argentina/Brazil, India | none | n/a | n/a | the file listing (`betfair_prices.list_published`; ledger `price-archive-inventory-1002`: 16 markets) |

The two Australian race counts differ: the price files' 81 a day is from eight early-October days and the ANZ data's
46.8 is thoroughbreds only. That the files' figure includes harness racing is likely but unverified; the ANZ data
publishes harness in a separate set ([S1]), and a Betfair Australia review counts 16,891 thoroughbred races in the
2025-26 season, about 46 a day ([S6]). The repository's figures for France, the USA and South Africa rest on eight
days of files; a full-year read of the archive (section 6, step 1) replaces them.

Betfair adds country markets around big events (the Japan Cup, Hong Kong's international races) on some days:
unverified. One API call answers it: `listCountries` for event type 7 returns every country with markets and how many.
The census (`betfair_recorder.census`, line 373) takes a country list but not that call; adding it is a few lines.

### 1.2 Australia and New Zealand in Betfair's own data (ANZ files, read 1 Nov 2026)

| Year | AU races (a day) | AU runner pre-play, median / p90 | AU market pre-play, median | AU BSP money, share of the market (median) | NZ races a day | NZ runner pre-play | In-play a runner (AU) |
|---|---|---|---|---|---|---|---|
| 2023 | 17,496 (48.1) | 2,306 / 15,827 | 45,713 | 27.4% | 11.1 | 602 | 126 |
| 2024 | 17,093 (46.8) | 1,974 / 13,267 | 39,290 | 32.3% | 10.8 | 436 | 101 |
| 2025 | 17,041 (46.8) | 1,781 / 12,201 | 36,167 | 33.6% | 11.2 | 325 | 119 |
| Sep 2026 | 1,278 (42.6) | 1,947 / 11,702 | 36,378 | | 10.5 | 399 | |

The money is falling (the runner median down 23% from 2023 to 2025), and the metropolitan tracks hold about 2.6 times
the provincial money (Sep 2026: 4,142 against 1,605 a runner; my list of 13 metropolitan tracks). Saturday markets
hold a median 50,266 pre-play against 26,319 on Mondays (Sep 2026). A third of each market is matched at the BSP,
so the lay at the SP that closes each back (TRADING.md) has depth. By the scheduled off the book is tight (median
back book 98%, lay book 103%, Sep 2026).

**The pre-play average price against the BSP** (AU, median log(BSP / pre-play WAP), share that shortened):

| Pre-play WAP band | 2023 | 2024 | 2025 | Shortened (2025) |
|---|---|---|---|---|
| 1-2 | -0.010 | -0.007 | -0.007 | 53% |
| 3-5 | +0.002 | +0.003 | +0.000 | 49% |
| 8-13 | +0.019 | +0.013 | +0.011 | 46% |
| 21-51 | +0.027 | +0.015 | +0.017 | 45% |
| 51+ | +0.103 | +0.067 | +0.077 | 35% |

In the UK the same measure runs from +0.009 at 1.5-2 to +0.236 at 51+ (`reports/open_close/data.md` s.7.2). In
Australia the pre-play money is matched so close to the off that its average price is the close; the files hold no
earlier price. So neither the price files ("morning" includes post-off trading: `other_markets_and_stakes_1002.md`
s.4) nor the ANZ data can show the early move; Betfair Historic Data can.

**A level back at the BSP** (AU, return a unit before and after 2% commission on winnings, by BSP band):

| BSP band | 2023 | 2024 | 2025 | 2025 after 2% |
|---|---|---|---|---|
| 1-2 | -2.1% | -2.0% | -0.2% | -1.0% |
| 3-5 | -0.6% | +1.7% | -0.5% | -2.0% |
| 8-13 | +0.7% | -1.9% | +3.5% | +1.7% |
| 21-51 | +2.1% | -5.5% | -1.1% | -3.0% |
| 51+ | -6.3% | -7.0% | -11.4% | -13.2% |

The favourite-longshot bias at 51+ is the only stable cell, and no cell is positive after commission in more than
one of the three years. Betfair's own Australian model ("Kash", published free with its results [S1]) does not beat
the BSP either: on 18,506 AU and NZ races of 2025 its race log-loss is 1.851 against the BSP's 1.793, and a level back
at the BSP wherever the BSP exceeds Kash's price loses 2.2-3.2% before commission at every threshold from 0 to 30%
(my reading of `Kash_Model_Results_2025.csv`). Kash is a benchmark for us, not a shortcut.

## 2. The form data each jurisdiction needs, and where it is

### 2.1 What the engine reads

`CustomMetricsEngine.calculate_all` (`model/custom_metrics.py` line 302) reads, per past run: date, track, time,
horse, finishing position, field size, beaten distance, the Betfair SP, official rating, prize money, going,
distance, class, comment, age, headgear, draw (`stall`), weight, jockey, trainer, sire and dam; the HRB CSV has 51
columns (`scraper.py` CSV_COLUMNS). Sectionals are a research block, not served (`model/blocks/sectionals.py`).

### 2.2 The sources, by jurisdiction

Costs are as published on the day and must be re-checked before buying. "Parse" is the parse.bot marketplace, which
resells scrapers of other sites by the credit (section 2.3; [S9]).

| Jurisdiction | Official body | Best commercial feed | Coverage and depth | Licence | Cost | Route |
|---|---|---|---|---|---|---|
| Australia | Racing Australia; since June 2025 it no longer wholesales, five authorised wholesalers do (BettorData, BetMakers, Mediality, News Perform, Racing and Sports) [S7] | **Punting Form** [S2][S3] | every TAB meeting; last 10 runs a horse with priceBF, priceSP, priceTAB, margin, barrier, weight, class, prize money, track condition 1-10, rail, in-run, stewards, gear; results with official and sectional time; sectionals AU/HK/SG; "10+ years" of history, sectionals "from 2014", point-in-time modelling data | personal use only; commercial use needs written consent; no automated access to the website without permission (the API is the sanctioned route) [S4] | Starter $59 a month (API and CSV); Professional $297 a month (sectionals on the site); Modeller and history by quote [S2]; currency not shown, presumably AUD (unverified) | REST API with key: `sources/puntingform.py` already fetches meetings, fields and results nightly once `PUNTINGFORM_API_KEY` is set |
| Australia, prices | Betfair | Betfair Data Scientists' ANZ files [S1] | every ANZ thoroughbred market 2020 to Sep 2026: BSP, pre-play WAP, max/min, volumes, BSP volume, the book at the scheduled off, win and place; Kash ratings 2021-2026 with speed categories | "use the data at your own risk"; no use restriction stated [S1] | free | CSV/zip download |
| Australia, alternatives | | Mediality (results "from January 1982") [S8]; Racing and Sports; The Racing API's Australia add-on, 329,974 results 2017-2026 [S10] | | The Racing API: not for betting operators; "data analysis" allowed [S10]; others by contract | by quote (prices not readable here) | API |
| New Zealand | NZTR (loveracing.nz) | Punting Form, if it covers NZ (unverified); the ANZ files for prices | | unverified | | |
| USA | Equibase (the industry's official charts) | Equibase products, DRF, Brisnet, TrackMaster (unverified) | charts with points of call, beaten lengths, fractions, odds, claiming prices, workouts; Parse lists 14 Equibase endpoints [S9] | Equibase: "personal use", "noncommercial", no "robot, spider, scraper or any other automated means" without written permission [S11] | consumer past performances about $1-3 a card (app listing, [S12], dated); bulk licensing by quote | licence, not scraping |
| France | France Galop (results, declarations, handicap values) | PMU programme JSON (unofficial, `online.turfinfo.api.pmu.fr`: unverified) | results with going, margins, weights, draw; France Galop publishes sectional timing [S13] | not read (France Galop's legal page 404 here) | free site; paid PMU feeds (unverified) | Parse lists 4 France Galop endpoints, 1 credit each [S9] |
| South Africa | National Horseracing Authority (results, merit ratings) [S14] | Timeform supplies SA form to bookmakers [S14] | the NHA database has had loading gaps (Dec 2023) [S14] | unverified | by quote | site or licence |
| UAE | Emirates Racing Authority | Racing Post international form | season Nov-Mar; many runners carry UK/IE form in HRB (unverified count) | | | |
| Hong Kong, Japan | HKJC, JRA | Punting Form (HK sectionals), netkeiba | deep official data, sectionals; Parse lists HKJC (8 endpoints) and netkeiba (3) [S9] | | | no Betfair SP: parked |

### 2.3 The Parse marketplace (queried 1 Nov 2026, [S9])

I searched 22 terms. Parse lists 16 racing scrapers; the useful ones and their per-call credits are in
`reports/international_racing_sources.md` table 2. The plans run from 1,000 credits for $30 a month to 100,000 for
$1,000 ([S9b]), so a credit costs $0.01-0.03. Australia through Parse: `form.tab.com.au` `get_race` (2 credits: barrier,
jockey, trainer, weight, last five starts, TAB fixed odds with the opening price and fluctuations) at about 47 races
a day is about 2,800 credits a month, the $100 plan. But every Parse listing is a scraper of another site, so the
underlying site's terms govern: Equibase forbids automated access outright [S11]; for Racing Post, the TAB and
Racenet the terms were not read here (unverified). Punting Form is not on Parse; its own API is the sanctioned route. **Parse is a way to
read a card, not a licensed history.** It gives no back history (the TAB card's "last five starts" is a string, not
past races), so a model would need a year of forward collection before its first fit.

### 2.4 What is already in hand

- The price-file archive in S3 holds AU, FR, USA, RSA and UAE win and place from 2018 (`reports/open_close/data.md`
  s.6.1); only UK/IE is loaded into `betfair_prices` (`betfair_prices.COUNTRIES`, line 56).
- The Blandford backend's At The Races results with sectionals are "worldwide, about 2.5m rows from 2018", and
  `blandford_sectionals.parse_atr` keeps only UK/IE rows (`blandford_sectionals.py` lines 9-12, 161-165). The rest
  are there to read; their licence is as unclear as the UK rows' (unverified).
- Betfair Historic Data: the greyhound bundle carried GB, AU, NZ and IE markets (`all_markets_priorities.md` s.4); the
  Horse Racing plan is not held.

## 3. How the model ports

### 3.1 What carries over, what needs an equivalent

The served model's gain (`reports/feature_inventory.md`, 24 Sep): within-race ranks 41%, prize money ranked in the
race (`rPMW3/5/10`) 26% of it, past BSP against field size (ORR2) 14%, official rating 5.9%, the race's median rating
1.3%, days since the last run 1.0%.

| Feature group | Needs | Australia (Punting Form) | USA | France |
|---|---|---|---|---|
| NFP, RB, WIV, WAX, WOA, CWO, FSS, DSLR, LRP, trainer-jockey, race strength, most form windows | position, field size, dates, connections | direct | direct (Equibase charts) | direct |
| WPMRF, PMW (prize money raced for x RB squared, `custom_metrics.py` line 760) | prize money a race | direct (`prizeMoney`, AUD); the ranks within the race are currency-free | purse | prize money |
| ORR2, PFD, OFS, expectation residuals, the form windows' market view and A-E | each past run's BSP | direct: `priceBF` on every past run, and the price files from 2018 and ANZ data from 2020 to check it | only the runs Betfair listed; else the tote odds, a different price | the runs Betfair listed; PMU odds |
| Official rating, FCS (race mean rating), OR trajectory | a rating per run | the benchmark rating (`handicapRating` on today's card); per past run it is not in the Form schema [S3]: to check | none: claiming price, purse and speed figures instead | handicap value (kg) |
| Going (`encode_going`, `custom_metrics.py` line 1282; `bfsp_features.py` line 749) | the going word | `trackConditionNumber` 1-10, a better input than the word | dirt/turf/synthetic x fast/good/sloppy/muddy/firm/yielding | penetrometer reading |
| Class movement (`_calc_class_movement`, line 1189) | class 1-7 | Maiden, Class 1-6, Benchmark 58-100, Listed, Group (`raceClass`) | maiden/claiming/allowance/stakes with the claiming price | conditions, handicap, Group |
| Draw (`_calc_draw_bias`, line 1322; `model/draw_curve.py`) | stall | barrier (and `originalBarrier` after scratchings), by track and rail position (`rail`) | post position | draw (corde) |
| EPF, pace, race shape (`model/race_shape.py`), from comments | the in-running comment | `inRun` positions and settling position: better than parsed text | points of call | rarely published (unverified) |
| Headgear changes | headgear | `gearChanges` | blinkers, Lasix | |
| Speed figures (`_calc_speed_figures`, line 1674) | comptime | official race time and Punting Form's own race time; sectionals | Equibase speed figure | sectional timing [S13] |

The ranks are ranks of these same metrics within the race, so they carry over as their inputs do; with ORR2 they
are 55% of the gain (41% + 14%), and their inputs are mostly the first three rows. Australia supplies all three; the
USA lacks the past BSP and the rating; France lacks the market depth.

### 3.2 One model or one per jurisdiction

**One per jurisdiction, AU and NZ pooled.** Reasons from the repository's own design:

- The greyhound model is its own package (`greyhound/data.py`, `metrics.py`, `model.py`, `track.py`) built "as the
  horse model is: results data -> lag-safe metrics -> a market-blind model -> scored against the market", reusing the
  engine's lag-safety rule (`model.lagsafe`) and its leak test, not the horse feature code (`greyhound_model.md` s.1-2).
  It found its first leak through that test before any result.
- The features are scale-dependent: prize money in AUD, ratings on another scale, tracks and going codes the UK
  vocabulary encodes as unseen (`bfsp_features.py` lines 760-767). A pooled LightGBM would learn the jurisdiction
  first and pay for it in fit.
- Horses rarely cross between the UK and Australia, so a pooled model gains no shared horse history; AU and NZ do
  share horses (unverified count), which is why they pool.
- The UK model is live and earning (+6.15% on GBP47,274, 19-31 Oct: TRADING.md); nothing here should touch it.

The engine itself should be shared, not copied: a jurisdiction config (going map, class parser, rating column,
distance units, track vocabulary) passed to `CustomMetricsEngine`, with the UK config the default and byte-identical
output on the UK matrix as the regression test.

### 3.3 Files to change (Australia)

| File | Change |
|---|---|
| `sources/puntingform.py` | `build()`: the raw days to a `runs` table in the `race_results` column names (one row a runner a race: date, track, time, horse, position, margin, field size, barrier, weight, jockey, trainer, class, prize money, distance in furlongs, track condition, in-run, gear, priceBF/SP/TAB) |
| new `au/data.py` (as `greyhound/data.py`) | read that table and the AU price files; match Betfair's selection names to Punting Form's (name, track, off time) |
| `model/custom_metrics.py` | a `jurisdiction` argument: going (line 1282), class (line 1189), rating columns (lines 701, 1962), distance units; comments to in-run positions for EPF (line 607) |
| `model/bfsp_features.py` | the going map (line 749) and the categorical vocabulary per jurisdiction |
| `model/race_shape.py`, `model/draw_curve.py` | in-run positions and barrier with rail |
| `train_bfsp.py`, `evaluate_oos.py` | `--jurisdiction au`: source table, feature cache directory, model path `data/models/bfsp_model_au.lgb` |
| `predict_bfsp_today.py` | an AU card from Punting Form's fields and scratchings endpoints, not HRB |
| `betfair_prices.py` | `parse_file_name` accepts only `uk|ire` (line 99) and `COUNTRIES` is UK/IE (line 56): add `aus`; an AU course map beside `BF_COURSE_MAP` (line 66); a load into its own table so UK queries are unchanged |
| `betfair_historic.py` | build Horse Racing tables (first, T-60, T-15, T-1, close, BSP) once the plan is held |
| `trading/config.py`, a new `trading/config_au.json` | `countries=("AU","NZ")`, the AU trading window, its own limits |

## 4. Operations

### 4.1 Race times in UK hours

| Jurisdiction | Off times, UK | Source |
|---|---|---|
| Australia | 5th-95th percentile 12:59-18:43 on the ANZ files' clock (Melbourne time, inferred: Perth races sit at 15:49-19:30 on it); so about 01:59-07:43 GMT from November to March and 03:59-09:43 BST from April to September; first race about 00:20 GMT, last about 10:30 GMT | ANZ data, Sep 2026 |
| New Zealand | 10:21-14:40 on the same clock: about 23:20-03:40 GMT in the UK winter (it crosses UK midnight), 01:20-05:40 BST | same |
| USA | about 17:00-04:00 UK | unverified |
| France | about 11:00-19:00 UK, later on summer evenings | unverified |
| South Africa | about 10:00-17:00 UK | unverified |
| UAE | Nov-Mar, about 12:00-18:00 UK | unverified |

France and South Africa sit inside the UK trading day; the USA, Australia and New Zealand need a night runner.

### 4.2 What has to change to record and trade them

- **The runner.** Every Betfair job runs on the UK server's self-hosted runner (`runs-on: ${{ vars.BETFAIR_RUNNER }}`
  in `live-trade.yml`, `live-record.yml`, `betfair-prices.yml`, `greyhound-trade.yml`). The price archive holds it
  from 22:30 to 06:15 UTC and the UK trader from 05:15 UTC; a runner runs one job at a time (the `runner-give-way-1007`
  defect in the ledger is that contention). An AU recorder or trader needs a second runner on the server (several can
  be registered on one machine: unverified for this host) or the archive moved. It must stay on the UK server: the
  account is British and promo.betfair.com geo-blocks other regions (`betfair_prices.py` line 20).
- **The recorder** already takes `--countries` and `--market-types` (`betfair_recorder.py` lines 891-892), so
  `--record --countries AU,NZ --tag au --until 11:00` records the AU day with no code change, as a tagged file the
  nightly load does not read (lines 879-883). `record()` defines the day as UK midnight to midnight
  (`trading.exchange.uk_day_window`, line 458), which holds Australia but splits New Zealand's winter cards across two
  UK days; the evening option (`--evening`) covers that.
- **The trader** filters on `cfg.countries` (`trading/config.py` line 73, `trading/session.py` line 143,
  `auto_trade.py` lines 337-343), so a separate config with AU and its own `trade_from`/`trade_until` in UK time, and a
  separate workflow and concurrency group, keep it apart from the UK session. The predictions must exist before the
  first AU race (about 00:20 GMT), so the AU run is the evening before, after Punting Form's final fields; late
  scratchings come from its scratchings endpoint.
- **The benchmark.** The BSP closes every back, as in the UK. Per jurisdiction the price files give the BSP and the
  pre-play price from 2018; the early price comes from Betfair Historic Data (horse racing plan) for history and from
  the recorder going forward.

## 5. The ranked plan

| Rank | Jurisdiction | Betfair liquidity | Form source and cost | Model port effort | Operations | Verdict |
|---|---|---|---|---|---|---|
| 1 | Australia | 46.8 races a day; runner pre-play 1,781 (2025, presumably AUD), a third at the BSP; metro 2.6x provincial | Punting Form: $59 a month Starter (API), history by quote; personal-use terms need written consent; ANZ prices free | Medium-large: a new data module and a jurisdiction config; Punting Form's fields map closely, including the past BSP | Night runner 00:00-11:00 UK; recorder ready; trader config | **First** (research; money only after a forward test) |
| 2 | New Zealand | 11 races a day, 325 a runner | Punting Form if covered (unverified); ANZ prices | Small once AU exists (pooled) | Crosses UK midnight | **Next**, with AU |
| 3 | South Africa | 10 races a day, GBP615 a runner | NHA site, Timeform by quote | Medium: merit ratings exist | UK daytime: fits today's runner | **Park**: thin, no licensed feed found |
| 4 | USA | 69 races a day, GBP300 a runner | Equibase licence by quote; scraping forbidden | Large: no official rating, no past BSP, dirt | Night runner | **Park** |
| 5 | France | 16 races a day, GBP256 a runner | France Galop and PMU open on the web | Medium | UK daytime | **Park**: the thinnest |
| 6 | UAE | seasonal (Nov-Mar); empty files in October | ERA site; UK/IE runners' HRB form | Small for UK-raced runners | UK daytime | **Park**; a first read of the files once the season starts |
| - | HK, Japan, Singapore, South America, India | no price files, no BSP | | | | **Never** while there is no SP |

## 6. The first concrete steps (Australia)

1. **The market-only scan on the archive (a research query, no purchase).** Load `auswin` 2018-2026 from the S3
   archive into its own table (the `betfair_prices.py` changes in 3.3) and repeat section 1.2 on it: the pre-play
   price to BSP by band and year, BSP returns, money by track and day, and the races a day split into thoroughbred
   and harness (the 81 against 47 question). The ANZ files cross-check it from 2020.
2. **The early price (owner: free).** Buy the BASIC Horse Racing plan at historicdata.betfair.com with the trading
   account; `betfair_historic.py --fetch` then pulls AU and NZ markets, and the first traded, T-60 and T-15 prices
   show whether an early AU price moves to the BSP as the UK's does. The basic plan keeps the last traded price only
   when it changes, so fixed marks are sparse (`all_markets_priorities.md` s.4).
3. **The fill (a second runner).** Run the recorder on AU and NZ for two weeks: what a back of GBP15-200 at the best
   price can fill at 00:00, T-120, T-60 and T-15 UK. The UK rule needs GBP15 a horse at least (CLAUDE.md); if the AU
   books cannot fill it before T-15, the model has nothing to trade and the plan stops here.
4. **The form (owner: one month of Starter, $59, and Punting Form's written consent).** Set `PUNTINGFORM_API_KEY`;
   `source_data.py --source puntingform` fetches from 2018 back. Write `build()` from the first files' layout, and ask
   Punting Form for the price of its history and point-in-time data and for a licence covering our own betting.
5. **The model.** A walk-forward AU model as the greyhound one was: market-blind, lag-safe with the leak test, scored
   against the early price and the BSP by CLV; a paper forward test with a pre-registered rule and gate before any
   stake (`greyhound_model.md` s.5).

What the owner supplies: the Horse Racing BASIC plan (free); a second runner on the UK server; a Punting Form Starter
month and their written consent to use the data for betting; later, the history quote.

## 7. Limits of this note

- The ANZ volumes have no currency label; the repository's AU GBP1,180 comes from eight days of price files. Neither
  says how much trades early in an AU market; steps 2 and 3 do.
- Punting Form's per-run fields are from its published API schema [S3], not from a response; the history depth
  Starter can query by date is unknown until a key exists.
- The prices of Modeller, Racing and Sports, Mediality, The Racing API and Equibase's bulk data were not published
  or not readable from here.
- Off times outside Australia and New Zealand are from general knowledge (unverified).

Sources: `reports/international_racing_sources.md`.
