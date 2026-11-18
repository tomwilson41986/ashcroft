# Phase 0 data audit - horse racing (racing², 18 Nov 2026)

Source for every figure: `outputs/horse_v0_20261118/phase0_data_audit.json` and `phase0_data_audit_1118.txt`, from
`research/queries/done/phase0_data_audit_1118.py` (research-query run 44332233719) on `horse_racing.db` (S3
`horseracingresults`, 3.15 GB, as downloaded at that run).

## Tables

| Table | Rows | Dates | Role |
|---|---|---|---|
| `race_results` | 2,023,508 | 2010-01-01 to 2026-11-16 | results, one row a runner (HRB) |
| `betfair_prices` | 2,146,291 | 2017-12-31 to 2026-11-16 | Betfair SP and price files, UK/IE racing |
| `betfair_live_marks` | 522,762 | 2026-10-01 to 2026-11-17 | the live record's books at marks before the off |
| `betfair_live_markets` | 36,395 | 2026-10-01 to 2026-11-17 | live record catalogue |
| `live_orders` | 69,055 | 2026-09-30 to 2026-11-17 | the trader's orders |
| `blandford_results` | 678,768 | (no date column) | Blandford / Timeform-feed results |
| `blandford_atr` | 382,349 | 2018-01-01 to 2026-09-25 | At The Races sectionals |
| `sectional_runs` | 153,426 | 2018-01-01 to 2026-09-25 | sectional runs |
| `blandford_rtv` | 62,777 | 2023-01-01 to 2026-10-29 | Racing TV sectionals |
| `betfair_odds`, `bets`, `daily_pnl` | 0 | | empty (legacy) |

## race_results: rows and races by year and code

| Year | Flat turf | All-weather | Hurdle | Chase | NH flat | Rows | Races |
|---|---|---|---|---|---|---|---|
| 2018 | 46,375 | 28,336 | 28,756 | 14,876 | 4,581 | 122,924 | 13,052 |
| 2019 | 46,845 | 28,176 | 28,992 | 14,464 | 4,434 | 122,911 | 12,749 |
| 2020 | 36,051 | 27,702 | 24,257 | 11,906 | 3,656 | 103,572 | 10,365 |
| 2021 | 47,308 | 28,617 | 31,342 | 15,516 | 5,778 | 128,561 | 13,377 |
| 2022 | 45,250 | 27,803 | 29,555 | 14,184 | 4,423 | 121,215 | 13,097 |
| 2023 | 44,946 | 28,921 | 29,349 | 13,712 | 4,257 | 121,185 | 12,897 |
| 2024 | 45,130 | 29,814 | 29,802 | 13,633 | 3,919 | 122,298 | 12,887 |
| 2025 | 45,111 | 28,645 | 28,614 | 13,306 | 3,910 | 119,586 | 13,018 |
| 2026 (to 16 Nov) | 45,031 | 23,085 | 23,698 | 10,222 | 3,097 | 105,133 | 11,432 |

2010-2017 hold 115,021-124,491 rows a year (in the JSON); the model's history starts in 2018 and its training in 2021.

## Checks

| Check | Result |
|---|---|
| Duplicates, date + track + time + horse | 0 |
| Exact duplicate rows | 0 |
| Blank horse names | 0 |
| Race id | no `raceid` column: a race is date + track + time (`model/race_shape.race_key`) |
| Declared field size against rows held | 212,864 of 212,874 races agree; 6 declare more, 4 fewer; 24 one-runner races |
| Non-runners | not held: `place` has no NR code, so a withdrawn horse is absent, and Rule 4 is not in the results |
| Non-finishers (`placing_numerical` null) | 124,206 rows (6.1%): PU 85,637, F 20,205, UR 13,420, BD 1,625 and smaller codes |
| BFSP | 0 null; 891 at or below 1.0 and 274 above 1000 (both invalid, 1,165 rows, 0.06%); 2,022,343 valid |
| The BFSP book of a full race | median 1.002 (1st-99th percentile 0.964-1.042); 84 races outside 0.9-1.2 |
| Beaten lengths (`total_dst_bt`, text) | blank on 337,402 rows: mostly the 213,174 winners and 124,206 non-finishers (together 337,380); parsed on 1,686,106; finishers placed 2nd or worse with no margin: 89 |
| Official rating | 538,146 rows are 0 (unrated coded as 0, not null): to be read as missing |
| Prize money | 84 rows at 0 |
| Headgear | blank on 59.8-75.4% a year: blank means none worn (to confirm against the card) |
| Days since last run | null on 7.1-8.2% a year (debutants) |
| Other key fields (track, time, distance, going, class, type, surface, trainer, age, weight, stall, comptime) | no nulls |
| Last day held | 2026-11-16 (151 rows): HRB has paused downloads until 25 Nov |

## Rows usable for each target

| Target | Rows usable | Share | Dropped and why |
|---|---|---|---|
| The BFSP (`logit_norm_prob`) | 2,022,343 | 99.94% | 1,165 rows with an invalid BFSP |
| Normalised beaten lengths (finishers with a margin) | 1,899,213 | 93.86% | 124,206 non-finishers (no margin by definition), 89 finishers missing a margin |
| ... with a prize for the class adjustment | 1,899,129 | 93.85% | a further 84 with no prize |

By year the beaten-lengths target covers 93.2%-94.5% of rows (2010-2026), the BFSP 99.6%-100%.

## Locked holdout (counted, not evaluated)

From 2026-04-01: 82,545 rows, 8,864 races; from 2026-10-01: 16,289 rows.

## Gaps for later phases

| Gap | Effect |
|---|---|
| No non-runners or Rule 4 deductions in `race_results`, and no reduction-factor column in the live tables loaded (`betfair_live_*`, `live_orders`) | historic P&L ignores Rule 4; the reduction factors exist only in the raw live books in S3 (`betfair_live/<day>/`, from October 2026) |
| Order-book depth only in the live record, from 2026-10-01 (three levels a side, the delayed key) | execution realism (Phase 5) can be tested on about 7 weeks only; earlier backtests use the BSP or last traded and are optimistic |
| Sectionals end 2026-09-25 (ATR) and 2026-10-29 (RTV) | the sectional blocks go stale for recent races until the feeds catch up |
| HRB results stop at 2026-11-16 | the database and the live features lack the days to 25 Nov until the backfill |
