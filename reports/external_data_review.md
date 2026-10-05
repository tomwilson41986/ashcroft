# The Blandford repositories: what they hold, and what it is worth to the betting model

5 October 2026. The owner's question: review the data in Abiqb93/HorseRacesBackend,
Abiqb93/horses-website-deployed and Abiqb93/daily-horse-scraper, and say how it could feed
the betting model.

## In short

- **What they are.** The three repositories are the Blandford Bloodstock platform:
  - daily-horse-scraper collects the data;
  - HorseRacesBackend serves it from a MySQL database;
  - horses-website-deployed is the site, plus sectional and sales files built from it.
- **Already used.** Ashcroft has read one part of this every night since September: the Timeform results feed (`blandford_sync.py`). Its same-day fields leak the result and are banned; its lagged figures carried no edge in the last twelve development months.
- **New and testable here.** Two things are new and could be tested from the repositories' own files.

  | Data | Beyond Betfair SP on the winner | Cut in our model's BSP-forecast error |
  |---|---|---|
  | Sectionals (per-furlong times, finishing speed, stride) | Nothing | 1.2–1.4% in Flat/AW races; 1.7–1.8% with the same export's past ratings |
  | Auction sale prices | Nothing | 2.7–3.2% in maidens and novice races |

- **What that means.** Neither tells us who wins beyond the market. Both make our price forecast closer to the closing price, and the closing price is what the live trade is paid on.
- **The caveat.** Both gains are measured against the September out-of-sample file, an older model than the one serving today. The served model's gain will be smaller, and only an iteration on it, scored on CLV, can say how much.
- **Thrown away now.** Some of the most interesting data is discarded today:
  - Timeform's pre-race tips and flags, pace map and early-pace figure, which the paid API already returns;
  - the dated history of entries and declarations.

  Storing them costs almost nothing, and the trade's two weak points are steamers and withdrawals. These are the data that bear on both.
- **Security.** There are two problems the developer should fix now; see the last section.

## What the repositories hold

| Data | Where it lives | History | In Ashcroft? | Known before the off? |
|---|---|---|---|---|
| Timeform results feed (`APIData_Table2`): pre-race master/adjusted ratings, performance rating, timefigure, finishing and closing sectionals, Betfair win/place SP, in-play high/low, RP OR/TS/RPR | Backend MySQL, filled daily by the scraper from GlobalSportsAPI (Timeform) | 2006 on; about 2.4m rows | Yes, as `blandford_results`, 2021 on | No. Same-day rows are results, rewritten for 4 days after the race |
| Per-furlong sectionals, positions at each split, finishing speed, stride length and cadence, top speed | Site files `public/data/rtv/` | 67,723 runs from 30 Jun 2024 to 29 Jun 2025 (UK/IE/FR Flat and AW, TPD/Timeform/McLloyd); 42,783 UK/IE runners from 18 Apr 2026 to now, refreshed twice a day | No | Past runs only |
| ATR and Racing TV sectional/tracking tables (`attheraces`, `racingtv`, `sectionsparsed`) | Backend MySQL | ATR 1 Jan 2018 to 26 Nov 2025 (4.3m rows, worldwide); RTV Jul–Dec 2025 | No | Past runs only |
| Sale prices | Timeform horse files in the scraper (`Horses_Data_*.csv`, Feb and Jul 2025 snapshots: the price opens the production comment); HIT sale catalogues on the site (Tattersalls July 2026: 502 sold, prices and buyers); Goffs yearlings | Every horse foaled 2018 on in the July file | No | Yes: the sale comes before the racing |
| Timeform premium pre-race content: tips (Jury, TV Focus, Last Word, Long Ranger, Stat Selector), Top Rated, Horse in Focus, Warning Horse, hot/cold trainer, pace-map ideal position, draw comment, analyst verdict, early-pace figure, `bfMarketId` | Returned by the API the scraper calls every morning, then dropped | None kept (older full-frame S3 dumps may hold some) | No | Yes, the day before |
| Entries and declarations: BHA Racing Admin (entries, declarations, reserves order, GoingStick), HRI cards, Timeform entries for today +6 days | Scraper → MySQL tables, rewritten every run | None in MySQL; dated copies only in the scraper's S3 `race_entries` files | No | Yes |
| Racing Post cards, taken at 17:00 UTC the evening before: best odds, price history, newspaper tips, Spotlight, RPR/TS, trainer RTF | Scraper → MySQL `racingpost`, replaced each run | None | No | Yes |
| Pedigree (worldwide 463k-horse file), stallion ratings and fees, prospects index, trainer uplift, sire/dam/owner profiles, track pars | Backend and site | Current snapshots | No | Built from all data up to today: they leak in any backtest |
| French racing (PMU from 2013, France Galop) | Backend | 2013 on | No | Not a market we trade |

## What was tested

Both tests use Ashcroft's out-of-sample predictions (215,760 runners, 31 May 2024 to 21 Mar 2026), nothing from the locked holdout (1 April 2026 on), and the residual screen's method (`scripts/residual_screen.py`):

- **The winner beyond the market:** a conditional logit on each race's winner, with the Betfair SP alone against the SP plus the new readings. It is scored on races the fit never saw, in millinats per race.
- **The forecast:** the same new readings, used to correct our model's log-BSP forecast within each race.

Both are cross-fitted: by calendar-month parity, and forward.

**Sectionals** (`scripts/external_sectionals_screen.py`, `reports/external_sectionals_screen.json`)

- **Sample.** 65,298 runners in 6,905 Flat and all-weather races, August 2024 to June 2025. 82% had a sectional run behind them.
- **Matching.** The site's runs match our runners on date and name 99.9% of the time.
- **Coverage.** 99% of all-weather runners and 80% of turf Flat runners are covered; jumps are not covered at all.
- **Readings.** All are read only from the horse's runs on earlier days:
  - finishing speed, overall and against its field;
  - whether it finished faster than it placed;
  - early and late pace against the field;
  - final-furlong rank, stride length and cadence;
  - counts and days since.
- **Winner beyond the SP: nothing.** On the 4,893 races where at least 80% of the field had a sectional run, the readings together scored −1.7 ± 1.2 millinats per race (month parity) and +0.6 ± 1.5 (forward). With our model's probability added to the market: −1.5 ± 1.2 and +0.8 ± 1.5.
- **Forecast: a real cut.** The sectionals cut the within-race error of our log-BSP forecast by 1.36% (t 10.8, parity) and 1.22% (t 6.6, forward). The export's own past ratings (OR, RPR, Timeform, timefigure) add 0.6 points; the two together cut 1.84% and 1.74%.
- **A trap found on the way.** A "no sectional history" flag looked like a strong negative signal across the whole sample. It only marks where the export happens to start, and its meaning changes with the month. The clean reading above leaves it out.

**Sale prices** (`scripts/external_sales_screen.py`, `reports/external_sales_screen.json`)

- **Sample.** 14,477 runners in 1,575 Flat and all-weather maidens and novice races, August 2024 to March 2026. These are young horses (foaled 2022 on) with at most two earlier runs, where public form is thinnest and our model's error largest.
- **Coverage.** 98.7% were found in Timeform's horse file, and 43% had sold at public auction (median £40,000).
- **What is read.** Only the opening clause of the comment, the sales before the horse ran. The price is converted to pounds; a bare figure is guineas.
- **Readings:**
  - log price, and price against the race;
  - price rank in the race;
  - whether it sold;
  - whether it was a breeze-up sale;
  - debut × price.
- **Winner beyond the SP: nothing.** +0.3 ± 1.3 millinats per race (parity) and +1.3 ± 2.3 (forward). With our model added: +0.4 ± 1.3 and +1.5 ± 2.3.
- **Forecast: a real cut.** Sale price cuts the within-race error of our log-BSP forecast in these races by 3.18% (t 8.9, parity) and 2.74% (t 3.9, forward). The served model's debut-market block (the yard as the market rated it) may already hold part of this; the next step measures that.

## What could not be tested, and why

- **The backend's full sectional history** (ATR 2018–2025, Racing TV 2025) is in the MySQL database. I did not query the live backend: it was blocked from this session. I did not use the database credentials committed in the scraper, and the developer should export these tables to S3 instead.
- **Timeform's pre-race tips and flags** have never been stored, so there is nothing to test yet.
- **The history of entries and declarations** is overwritten every morning. Only the scraper's S3 `race_entries` files might hold dated copies.

## What to do, in order

1. **Sale prices into the model.** Build a sales block from the Timeform horse files; they cover the whole development window. Run it as an iteration on the served model, scored on CLV.
   - **Live source:** the horse files stop at July 2025, so this year's two-year-olds need a feed. The Timeform entries the scraper already pulls each morning return each horse's record, and its production comment carries the sale.
2. **Sectionals into the model.**
   - **For training:** ask the developer for a one-off export of `attheraces`, `racingtv` and `sectionsparsed` (UK/IE, one row per run) to S3.
   - **For serving:** use the site's twice-daily files, which hold every UK/IE runner since 18 April 2026.
   - **Then:** build the block and run it as an iteration.
3. **Keep what Timeform already sends.** One change in the scraper: write the API's full morning response to S3 each day instead of dropping 55 fields. That keeps the tips, Horse in Focus, Warning Horse, the pace map, the early-pace figure and `bfMarketId`.
   - **Why:** tips and flags are published the day before and move the Betfair morning market. After a few weeks, test whether they predict the move from the morning price to the SP, which is the move our CLV trade is paid on.
4. **Keep dated entries and declarations:**
   - Racing Admin's declared runners and reserve order, and GoingStick;
   - Timeform entries for the next six days.

   Together with Ashcroft's own Betfair record of withdrawals, they can price the non-runner cost the closing model misses: 1.5 points over 1–3 October, and 28% of the stake on 4 October.
5. **Do not use in a model:**
   - the Timeform feed's same-day fields (they are results, rewritten for four days);
   - anything built "as of now" in a backtest: prospects, profiles, pars, sire tables, trainer uplift, the worldwide pedigree file's careers;
   - Racing Post RPR/TS from the results table, which are post-race figures under the same names as the card's.

## Security and licensing (for the developer)

1. **The public backend serves its user accounts table to anyone.**
   - **Route:** `GET /api/:tableName` has no authentication, and its whitelist (`validTables` in `server.jsx`) includes `UserAccounts`, which holds emails, password hashes and mobile numbers. The same applies to the watch-list and tracking tables.
   - **Fix:** take the user tables off the whitelist, or put the route behind a login, then rotate whatever may have been read.
   - I read this from the code and did not call the route.
2. **The scraper commits live credentials.** The database, AWS keys, the GlobalSportsAPI (Timeform) key, the Racing Admin login and Gmail are in two committed `.env` files and in all 17 notebooks. The API key also appears in saved notebook outputs.
   - **Fix:** rotate every credential, move them to GitHub secrets, and purge them from the history.
3. **The backend republishes paid data with no login.** It serves Timeform, Racing Post, At The Races and Racing TV data to anyone. Before the betting model depends on any of these sources, check the GlobalSportsAPI/Timeform licence, the Racing Post, ATR and RTV terms, and the BHA Racing Admin terms.
