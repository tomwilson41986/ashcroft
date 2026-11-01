# International racing: sources and the Parse listings (research agent, 1 Nov 2026)

The sources behind `reports/international_racing.md`, each read on 1 Nov 2026 from this host. Every page was read
as data; none was acted on. Prices are as published that day and are to be re-checked before any purchase.

## 1. The cited sources

| Id | Source | What was read | Used for |
|---|---|---|---|
| S1 | Betfair Data Scientists, data listing: https://betfair-datascientists.github.io/data/dataListing/ ; files `../assets/ANZ_Thoroughbreds_2023.zip`, `_2024.zip`, `_2025.zip`, `ANZ_Thoroughbreds_2026_09.csv`, `Kash_Model_Results_2022.csv`, `Kash_Model_Results_2025.csv` | the listing (ANZ thoroughbreds 2020-2026, Kash 2021-2026, harness, greyhounds; the disclaimer); the files downloaded and read with a scratch script (pandas): per market and runner, BSP, pre-play WAP and volume, BSP volume, the book at the scheduled off, results | the AU/NZ liquidity, the pre-play price against the BSP, BSP returns by band, the off-time clock, the Kash comparison (section 1.2 of the report) |
| S2 | Punting Form pricing: https://www.puntingform.com.au/pricing ; Modeller: https://www.puntingform.com.au/products/modeller | Free; Starter $59.00 a month (interactive form guide, API access); Professional $297.00 a month (sectionals AU/HK/SG, 200 m raw times, benchmarks); Modeller by e-mail (API to sectionals, analytics, history "from 2014" for AU/HK/SG sectionals, point-in-time data, "Personal use only", "historical Betfair prices and flucs data"); "10+ years of data"; commercial by quote | the AU form source and cost |
| S3 | Punting Form API reference: https://docs.puntingform.com.au/llms.txt ; `reference/form-1.md`, `reference/results-1.md` (OpenAPI schemas) | `/v2/form/form` (last 10 runs a horse; fields: position, starters, margin, raceClass, prizeMoney, distance, trackCondition, trackConditionNumber, jockey, weight, barrier, rail, flucs, priceSP, priceBF, priceTAB, inRun, officialRaceTime, pfRaceTime, stewardsReport, gearChanges, isBarrierTrial, originalBarrier, prepRuns); Runner (sire, dam, handicapRating, career records by going, A/E of trainer and jockey); `/v2/form/results` (position, margin, barrier, weight, inRun, flucs, price, gear, stewards; race time, sectional time, track condition, rail); Starter and up, sectionals Modeller only | the feature mapping (section 3.1) |
| S4 | Punting Form terms: https://www.puntingform.com.au/terms-and-conditions | "Access to all products and services on the Website, including API access and data, is for your personal use only. You must not use any part of the products, services or data on our Website for commercial purposes without Punting Form's prior written consent"; no automated access to the website without permission | the licence condition |
| S6 | SEN, "Betfair's 2025-2026 racing review" (3 Aug 2026): https://www.sen.com.au/news/2026/08/03/betfairs-2025-2026-racing-review | 16,891 races in the season; the Melbourne Cup over $10 million matched on the win market; 67 of 76 Group 1s paid better on the exchange than the corporate average | the AU race count cross-check |
| S7 | Racing Australia's wholesaler change (June 2025), via search results: https://pastthewire.com/racing-australia-announces-new-racing-materials/ and https://pamediagroup.com/pa-named-official-wholesaler-of-australian-racing-materials/ ; the agreement: https://t.racingaustralia.horse/uploadimg/media-releases/Racing-Materials-distribution-Wholesaler-Agreement.pdf (not opened) | five authorised wholesalers: BettorData, BetMakers, Mediality, News Perform (Punters Paradise), Racing and Sports, under one Wholesaler Agreement | the AU official route |
| S8 | Mediality data FAQ: https://medialityracing.com.au/data-faq/ (search snippet) | results "starting from January 1982"; fields, form, results, comment, markets, ratings | an AU alternative |
| S9 | parse.bot marketplace search API: `curl -sS -G https://api.parse.bot/marketplace/apis --data-urlencode q=<term> --data-urlencode limit=25`, and `https://api.parse.bot/marketplace/apis/<id>` | 22 searches (section 2 below); 16 listings read in full | the Parse table |
| S9b | parse.bot pricing: https://parse.bot/pricing | Free 200 credits; Hobby $30 a month 1,000; Developer $100 5,000; Team $300 20,000; Company $1,000 100,000; overage allowed; no per-credit price stated | the credit cost ($0.01-0.03, my division) |
| S10 | The Racing API: https://www.theracingapi.com/data-coverage ; https://www.theracingapi.com/terms-of-service | core 356,954 results (GB 209,193, IE 55,430, FR 41,550, USA 16,854, HK 8,428, AU 7,081, UAE 4,594, JP 2,984 ...); North America add-on 116,230; Australia add-on 329,974; 2017-2026; "not permitted for use in any commercial or operational capacity by betting operators and sportsbooks"; prices not readable (script-rendered) | an alternative feed |
| S11 | Equibase terms: https://www.equibase.com/about/terms.cfm | information "for your personal use and is not for resale"; "not to use any robot, spider, scraper or any other automated means ... without Equibase expressly providing, in an authorized writing, such permission"; "solely for your own noncommercial use" | the US licence position |
| S12 | Equibase app listing (search result): https://apps.apple.com/app/id453573402 | single-race PP $0.99, Basic $1.99, Premium $2.99 (undated) | consumer price only |
| S13 | France Galop, "Sectional timing now readily available" (2 Oct 2020): https://france-galop.com/en/content/sectional-timing-now-readily-available (search result) | sectional times integrated into an analysis tool for horsemen and the public | French sectionals |
| S14 | Search results on South African racing: https://sportingpost.co.za/nha-confirms-national-stats-integrity ; https://www.gamblinginsider.com/news/2482/betvictor-to-offer-timeform-south-african-racing-data | the NHA as custodian of results; a Dec 2023 loading gap; Timeform's SA data to BetVictor | the SA position |

Not readable from this host: the Betfair page the owner linked (HTTP 403); Equibase's product pages (403); France
Galop's legal notice (404); Punting Form's `/products/` page (500; the pricing page answered).

## 2. The Parse marketplace: racing listings (queried 1 Nov 2026)

Search terms: equibase, hkjc, netkeiba, france galop, punters, racing australia, racenet, tab, sporting life,
timeform, racing post, attheraces, bloodhorse, drf, horse racing, punting form, racing and sports, pmu, jra,
loveracing, geny, zeturf. No listing was found for Punting Form, Racing and Sports, Racing Australia, Punters, DRF,
PMU, Geny, ZEturf, LoveRacing or the JRA itself. Credits are per call; "verified" is the listing's
`last_verified_at`. "Access" is the listing's `access_requirements` (phone, card: a paid account needed).

| Listing | Source site | Endpoints (credits a call) | Verified | Access |
|---|---|---|---|---|
| equibase.com API | equibase.com | search_horse (1), get_horse_profile (2), get_race_results (2), get_track_entries (5), get_past_performances (1), get_card_results (2), get_race_dates (5), get_race_changes (5), get_horse_workouts (2), get_speed_figures (2), get_leaders (1), owners and foals (2-10); 14 in all | 2026-11-01 | none |
| form.tab.com.au API | form.tab.com.au | list_meetings (1), get_race (2: barrier, jockey, trainer, weight, last five starts, TAB fixed odds with the opening price and fluctuations) | 2026-10-30 | phone, card |
| tab.com.au API | tab.com.au | meetings by date, next to jump, race card, fixed odds, results (1 each) | not stated | phone, card |
| racenet.com.au API | racenet.com.au | list_meetings (10; AU and NZ, today only), get_race_fields (10) | 2026-10-26 | phone, card |
| racingpost.com API | racingpost.com | racecards today/by date (1), get_horse_profile (1; form with RPR/Topspeed), get_race_results_by_date (5), get_australian_fields (1), get_race_runners (1), get_race_detail (1), owners, sales; 11 in all | 2026-11-01 | none |
| france-galop.com API | france-galop.com | search_horse (1), get_race_results (1; going, prize, margins, weights, pedigree), get_race_entries (1; draw, weight, rating), list_races (1) | 2026-10-28 | none |
| racing.hkjc.com API | racing.hkjc.com | get_race_results (1), get_race_card (2), get_sectional_times (1), get_horse_profile (1), search, rankings, fixtures; 8 in all | 2026-10-29 | phone, card |
| hkjc.com API | hkjc.com | search_horses (1), get_horse_rating_history (1; official rating carried into each race), list_race_meetings (1), get_race_card (1) | 2026-10-26 | phone, card |
| race.netkeiba.com API | race.netkeiba.com | list_races (1; JRA), get_race_card (1), get_race_result (1) | 2026-10-28 | phone, card |
| attheraces.com API | attheraces.com | get_racecard (3), get_draw_data (5), get_pace_data (5), get_formscan (5), get_horse_form (10), tips | 2026-11-01 | none |
| timeform.com API | timeform.com | get_fixtures (1; courses grouped by region incl. USA), get_todays_runners (1), get_free_race_pass (1) | 2026-10-27 | phone, card |
| sportinglife.com API | sportinglife.com | get_racecards (10; UK and IE only) | 2026-10-27 | phone, card |
| bloodhorse.com API | bloodhorse.com | race results list and detail (1; North America), stakes entries, leaders, sales | 2026-11-01 | none |
| deutscher-galopp.de API | deutscher-galopp.de | get_race_results (1), search_horses (1), rankings (1) | 2026-10-29 | phone, card |
| equianalytix.com API | equianalytix.com | list_races (10), get_race_dashboard (2; TPR ratings, win probabilities, speed figures) | 2026-10-30 | phone, card |
| raceiq.com API | raceiq.com | metric guides and glossary only (1) | 2026-10-30 | none |

Every listing is a scraper of the named site. Equibase's own terms forbid automated access without its written
permission (S11); the other sites' terms were not read here. A Parse call returns what a page shows today; none of
these gives a back history except through each horse's form page, one call a horse.

## 3. Repository files read

`CLAUDE.md`; `TRADING.md` (opening sections); `reports/other_markets_and_stakes_1002.md` s.1 and 4;
`reports/all_markets_priorities.md`; `reports/open_close/all_markets.md` (result and s.2.1);
`reports/open_close/data.md` (s.1, 2, 5, 6.1, 6.5, 7.1-7.2); `reports/open_close/markets.md` (rank 7);
`reports/greyhound_model.md` s.1-5; `reports/feature_inventory.md` (summary); `reports/research_ledger.jsonl`
(`price-archive-inventory-1002`, `market-only-drift-1002`, `runner-give-way-1007`); `sources/puntingform.py`;
`betfair_prices.py`; `betfair_recorder.py`; `trading/exchange.py`, `trading/config.py`, `trading/session.py`,
`auto_trade.py` (the country filters); `scraper.py`; `scrape_cheltenham.py`; `scripts/build_sectionals.py`;
`blandford_sectionals.py`; `model/custom_metrics.py`; `model/bfsp_features.py`; `.github/workflows/live-trade.yml`,
`live-record.yml`, `betfair-prices.yml`, `greyhound-trade.yml` (runners and schedules).
