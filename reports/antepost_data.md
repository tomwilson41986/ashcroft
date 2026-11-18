# racing² - Ante-post Phase 0: what the data says (first read)

18 Nov 2026. Phase 0 of `docs/antepost_plan.md`. Every figure is from run 2 of `antepost-research.yml`
([run 44342882609](https://github.com/tomwilson41986/ashcroft/actions/runs/44342882609), artifact `antepost-phase0-2`:
`antepost_phase0.json`, `historic_markets.parquet`, `historic_runners.parquet`, `recorded_books.parquet`), made by
`python -m antepost.phase0`, unless a line names another source. Read-only: no order call, nothing written to S3.

## Status

| Question | Answer | Settled? |
|---|---|---|
| Betfair's ante-post rules as text | Not read: Betfair's login refused GitHub's runner (HTTP 403 on `identitysso.betfair.com/api/login`, runs 1 and 2) | **Open** |
| Does an ante-post market carry a BSP? | No: `bspMarket` false on 584 of 585 markets | Yes (from the data) |
| Does it go in play? | No: 0 of 585 went in play; `turnInPlayEnabled` false on 584 | Yes (from the data) |
| When does it close? | A median 55.6 hours before the scheduled off (section 2) | Yes (from the data) |
| How is a non-runner settled? | Mostly as a loser, some removed: section 3. Needs the rules text | **Open** |
| Turn-up labels from history | The race-day match is not reliable yet (section 4) | **Open** |
| Recorded books since 7 Oct | 44 markets over 42 days; thin and wide (section 5) | Yes |
| Entries by stage (forfeits, 5-day, declarations) | No data found | **Open** |

## 1. The historic ante-post markets

Source: the Historic Data bundle `s3://<capture bucket>/sources/betfair_historic/bundles/horse_racing_basic_2024.tar`
(4,404,215,808 bytes), streamed in full: 904,002 files, 0 unreadable, 907 holding an `ANTEPOST_WIN` market. The tar is
cut short ("unexpected end of data after 904,002 files"), as au/historic.py also found, so it may miss markets at the end.

| | Count |
|---|---|
| `ANTEPOST_WIN` markets, all countries | 585 |
| GB | 131 (2024), 140 (2025), 134 (2026) |
| IE | 18 (2024), 28 (2025), 18 (2026) |
| AU / US / FR / DE / GR | 97 / 14 / 3 / 1 / 1 |
| GB/IE scheduled offs | 20 Jan 2024 to 14 Nov 2026 |
| Runners (all countries) | 30,655 |
| Runners a market | median 44 (10th pct 16, 90th pct 101) |
| Days from first message to scheduled off | median 35.0 (25th pct 20.0, 75th 68.1, 90th 332.9) |

The bundle is BASIC: last traded prices only, no book and no volume. A last traded price is evidence of a price, not
of a fill.

Each ante-post market sits in its own event named for the meeting ("Newmarket 10th Oct", or "Cheltenham 2027" for a
festival). It is not the event of the race-day WIN market, which is why matching by event found 2 events in run 2.

## 2. The market's clock

- **No BSP, no in-play.** `bspMarket` false on 584 of 585 markets, `turnInPlayEnabled` false on 584, and none went in play.
  So the trade can't be closed at the SP inside the ante-post market. The SP exit has to be a lay in the race-day WIN
  market, as the plan assumed.
- **It closes about 2.3 days before the race.** The last time each market was open is a median **55.6 hours** before the
  scheduled off (5th pct 53.2, 25th 54.5, 75th 57.8, 95th 109.7; 578 markets). That timing fits the 48-hour declaration
  stage, but this is unverified until the rules text is read. The ante-post position can't be traded inside its own
  market after that point.
- **It suspends and reopens.** 86% of markets were suspended and reopened at least once (`status_path` in
  `historic_markets.parquet`), for example "OPEN > SUSPENDED > OPEN > SUSPENDED > ... > CLOSED". Why is not known yet:
  likely at each entry stage.
- **The scheduled time moves.** 30.8% of markets changed `marketTime` during their life, and some final times are not
  the race date. For example, the Celebration Mile market sits in event "Goodwood 29th Aug" with `marketTime`
  2026-07-29. Race dates taken from `marketTime` are therefore unreliable.
- Commission base rate (`marketBaseRate`): 5% on 490 markets, 10% on 40, 7% on 28, 8% on 16, mixed on 11. Each market
  is one winner (577 of 585).

**Consequence for the plan:** there's a gap of about 55 hours between the ante-post market's close and the race. Over
that gap the position can't be traded in its own market. It can only be hedged in the race-day WIN market, if that is
open. When the race-day WIN market opens for these races has not been measured yet.

## 3. How runners end (settlement, pending the rules)

| Final status in the ante-post market | Runners |
|---|---|
| LOSER | 28,361 |
| REMOVED | 1,692 |
| WINNER | 572 |
| HIDDEN | 30 |

- With a median of 44 runners a market and race-day fields far smaller, most horses that don't run must end as
  **LOSER**, not void. That fits all-in run or not for most non-runners. The rules text has to confirm it.
- **REMOVED** runners were removed a median 2.18 days before the off (25th pct 2.12, 75th 2.30), right after the market
  closes. Every one with an adjustment factor carried 0 (1,280 of 1,692). These come in clusters: Ayr Gold Cup 2026
  (175 of 247), Chester Cup 2026 (45 of 86), Royal Hunt Cup 2026 (44 of 104). Some markets have every runner removed:
  Grand National 2025 (43 of 43), Turners Novice Chase 2025 (40 of 41). Whether a REMOVED runner's bets are void is
  the most important open rule: if they are, some non-runners cost nothing.
- 9 runners carried a BSP in the ante-post market (`bsp_in_market` non-null): no use as an exit.

## 4. The race-day match: not reliable yet

The aim is to label each listed horse as ran in the race, ran elsewhere that day, or didn't run. That label is the
turn-up model's target and the source of the race-day BSP.

- **Run 1/2, by Betfair selection ID and race date against `betfair_prices`:** of 22,362 GB/IE runners, 3,783 ran,
  506 ran elsewhere, 11,925 no run, and 6,148 in races not found (356 of 469 markets matched).
- **The check that fails:** every market's WINNER ran, yet 55 winners are labelled "no run" in races that were found,
  and 107 more sit in races not found. The share not found grows by year: 13.3% in 2024, 31.5% in 2025, 36.8% in 2026.
- **Run 2, by event against the bundle's WIN markets:** 2 events matched, because ante-post markets sit in their own
  events (section 1).

Likely causes, in order:
1. The race date is taken from `marketTime`, which is wrong for some markets (section 2).
2. Festival events ("Cheltenham 2025") carry no day.
3. Some horses may have a different selection ID in the ante-post market than on race day; the 55 winners point that way.
4. `betfair_prices` may be missing some days.

**Run 4, the third attempt** ([run 44355120048](https://github.com/tomwilson41986/ashcroft/actions/runs/44355120048),
artifact `antepost-phase0-4`): the race date taken from the event name first, then the Betfair horse ID on the day or a
day either side, then the normalised name.
- 467 of 469 GB/IE markets are now joined.
- Of the 6,257 runners who ran, 3,803 were found by ID and 2,454 only by name. So the ante-post market's horse IDs often
  differ from race day's.
- Where both sources name a winner, they agree on 99.44% of the runners who ran.
- **The winner check still fails: 445 of 459 GB/IE winners are found running (96.9%).** The 14 misses are of two kinds:
  - **6 labelled "ran elsewhere"**: the wrong race was chosen because most of the market's listed horses ran in another
    race that day. These are the big handicaps with consolation races (Chester Cup 2024 and 2026, Ayr Gold Cup 2025,
    Lincoln 2026, Northumberland Plate 2026) and a festival day (Gallagher Novices' Hurdle 2024).
  - **8 labelled "no run"**: the winner isn't found by ID or name on the day: four Leopardstown markets on 31 Jan 2026,
    Fact To File (Cheltenham 2024), Kopek Des Bordes (Cheltenham 2025), Old Park Star (Cheltenham 2026), and Electric
    Moon (Haydock, 22 Nov 2025). Possibly days or races missing from `betfair_prices`; not yet checked.

**Until this is fixed, no turn-up rate or price-against-BSP figure is quoted.** They would be biased by whichever
horses failed to match. This is the second attempt at the match that hasn't worked, so the next step goes to the owner
before more changes (see the end).

## 5. The recorded snapshots (7 Oct to 17 Nov 2026)

Source: `s3://<capture bucket>/betfair_live/<day>/books_antepost.csv.gz` and `markets_antepost.csv.gz`, one snapshot a
day by `betfair_recorder.py --antepost` (live-record.yml).

| | Value |
|---|---|
| Days | 42 (7 Oct to 17 Nov 2026) |
| Markets | 44, from the Old Rowley Cup (9 Oct 2026) to the Ascot Gold Cup (15 Jun 2027) |
| Book rows / runners | 43,014 / 1,841 |
| Runner status at the last snapshot | ACTIVE 1,246, HIDDEN 567, REMOVED 27, LOSER 1 |
| Snapshots with the market open | 96.5% |
| Active runners with a real back and lay (lay under 1000) | 25.2% |
| Matched to date, per market a day | median GBP10,851 (10th pct 643, 90th 40,105) |

Largest matched totals at the last snapshot (cumulative over the market's life): Arkle Chase GBP91,462, Champion Hurdle
GBP64,731, Gold Cup GBP59,941, Grand National GBP41,389, King George Chase GBP39,427, Supreme Novice Hurdle GBP38,611.
Smallest: Ascot Gold Cup GBP646, Punchestown John Durkan GBP700, Fairyhouse Drinmore GBP756.

The spread (best lay / best back - 1) on active runners with a real lay, by best back price:

| Best back | n | Spread median | 25th-75th pct | Back size median | Lay size median |
|---|---|---|---|---|---|
| 1-3 | 231 | 35.9% | 5.9% - 110% | GBP26.90 | GBP15.00 |
| 3-5 | 458 | 14.3% | 8.3% - 42.1% | GBP25.18 | GBP14.42 |
| 5-10 | 1,142 | 21.2% | 11.1% - 43.2% | GBP25.35 | GBP13.50 |
| 10-20 | 2,459 | 41.2% | 20.0% - 119.5% | GBP21.71 | GBP11.00 |
| 20-50 | 3,879 | 84.6% | 36.4% - 233% | GBP21.11 | GBP9.00 |
| 50+ | 1,647 | 153% | 63.6% - 264% | GBP23.07 | GBP2.82 |

So the book is wide, and a typical best price holds about GBP10-25. Only a quarter of active runners have a real
two-sided book. Any edge has to clear spreads like these, and stakes would be small.

Of the 44 recorded markets, 15 had passed their off by this run. Their race-day match has the same problem as section
4: 238 runners in races not found, and 567 runners HIDDEN at the last snapshot.

## What's next

1. **The rules text.** It needs a Betfair login that works. GitHub's runner is refused (403), and no Betfair
   certificate was used: the run fell back to the password login. Options: one read-only catalogue call on the UK
   server runner, at a time the live jobs are idle, or a `BETFAIR_CERT`/`BETFAIR_KEY` secret if certificate login is
   allowed from GitHub. Owner's call.
2. **The race-day match (third attempt done, run 4; it fails the winner check: section 4). Previously planned:**
   - take the race date from the event name where it carries one, else `marketTime`;
   - match on selection ID within plus or minus 1 day, then on the normalised horse name where the ID fails;
   - check against `betfair_prices` day coverage (rows on the date);
   - report the winners found as the pass/fail check: every WINNER must be found running.
3. **When the race-day WIN market opens** for these races, against the ante-post close about 55 hours out. This sets
   how long a position has no exit.
4. **Entries by stage:** still no source. The ante-post market's own runner list and its suspensions may stand in.
