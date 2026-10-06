# Horseracebase's system builder against the model

The owner's list of 6 Oct 2026 is horseracebase's System Builder v4: about 800 criteria for building betting
systems, printed from horseracebase.com/v4builder.php. Each criterion is mapped here to what the model already
reads (reports/feature_inventory.xlsx, 1,532 features, 993 served) and to the data we hold (race_results from
horseracebase's export, and betfair_prices from Betfair's price files, 2018 on). The question asked was whether any
of these would be useful in our model. Ledger `hrb-features-1006`.

**In short:**
- Nearly all of the list is the model's already, often more finely: a horse's, rider's, yard's, sire's and dam's
  records by condition and window, the last four runs, weight against the mark, penalties, class and trip moves,
  draw, pace, prize money and travel.
- What the records could give that the model did not read is now in two drop-in blocks, screened on the winner and
  put in front of the served model (iteration 109):
  - the in-running low and high of each past run, from the price files. **It sharpens the served model's price
    forecast** (-0.0008, resolved; most at the top of the market), though it adds nothing to the BSP on the winner.
    Iteration 110 tries it in every member of the served blend before any training;
  - six small extras: nothing. Retired.
- What horseracebase holds and our export does not is listed at the end: wind operations, foal dates, owners,
  opening prices and others.

## Race data (the race's conditions)

| Criteria | In the model |
|---|---|
| UK/Ireland, race code, track, course speed, bends, track location and direction, surface, fence type, distance, going, runners, class (inc. Irish), group/grade/listed, prize money, festival, race type, handicap, NH race type, beginners/maiden/novice/nursery, juvenile, hunter/claimer/seller, age restrictions, jockey race type, season, dates, meeting time | Inputs where they vary a horse's chance: card fields, context and race-name conditions; track direction and rail movement in the draw block. A race-level constant cancels inside a race in a win model, so it matters only through the trees' interactions. |
| Rail movements | Read (draw block: `rail_move_yards`, `rail_moved`). |
| Ordered card number | Not read. The weights and marks it orders by are. |
| Places paid, SP of the favourite, second and third favourites, favourite against second | Today's prices: the market. The price forecast is market-blind by design; the trader's closing model combines our price with the market's. |
| Sexes in race, % males, max age, max weight, median/avg/max OR in race | OR against the race's max and median are served; the rest through the within-race readings. |
| Numbers (and %) of course, distance, going, track and surface winners; won/placed last time; ran within 30/90 days; claiming riders | The within-race readings (rr_, rw_) set each runner's form against the field's. |
| "Likely No" front runner, prominent, tracked/chased, mid-division, held up, poor start | The pace block (served, 91 features): run style from past comments, the race's shape. |
| Big race trend | Not built: horseracebase's trend tables. |

## Horse data

| Criteria | In the model |
|---|---|
| Sex, age, age position, age vs youngest/oldest | Served (card fields, within-race readings). |
| Origin of horse (IRE, FR...), colour, owner | Origin is not read (it is in the name's suffix). Colour and owner are not in our export. |
| Headgear, tongue tie | Headgear served, first-time headgear in the intent block. |
| Stall, position in stalls, draw segment, stalls analyser | Draw and stall block (served, 60). |
| Miles travelled, position by miles | Travel block (served). |
| Odds (SP, BFSP, forecast, early, opening), favouritism, market positions, odds moves | Today's market: not in a market-blind forecast. Past runs' market view and their surprise are served (form windows `fw_mkt_*`, A−E against the BSP); the morning-to-BSP move of past runs is in the research path (`model/market_features.py`); the SP against the BSP history was retired (market history, iteration 98). |
| Position vs position in the market (past races) | Served (A−E and the finish against the market's order). |
| Official rating, OR moves, OR vs race high/average/min, vs last race and last win, highest (winning) OR, OR vs C&D win | Served (custom metrics, handicap angles, Kalman rating). |
| Weight, pounds ahead, weight vs LR/max/avg/min, penalty carried, position by weight | Served (handicap angles: weight against the mark, + for a penalty; well in after a win). |
| Compare vs LR/2LR/3LR/4LR/last win, best in 3/5/10 runs | Form windows and form variants (served). |
| Highest class run/won, class move, distance move, max distance run/won, prize money vs LR | Served (custom metrics; PMW, performance-adjusted prize money). |
| Days since run, since distance/track/C&D/class win; position by days | Freshness block (served). |
| H-Run / H-Win / H-Places / H-Win% / H-Plc% / H-(P/L) by career, track, distance, going, season, surface, jockey, trainer, headgear, class, odds, C&D, direction, fences, bends, course speed, shorter/longer trips, grade, festival, month, 1 year, 90 days, ±1f, since a loss | Form windows (served) and the custom metrics' course, distance and going records. Condition form (cf_) measured them against the horse's overall record and added nothing (built, not served). |
| Earnings per start/win, total prize money, earnings rank | PMW (served). |
| H2H wins/losses/draws | Head-to-head block (served). |
| RC_Start (front runner ... poor start) | Pace block. |
| Runs since gelded | Intent block (first time gelded). |
| **BF In Play (Min), (Max); BF-Plc In Play; BF Placed** | **New: the in-running block (`ir_`)**. Win market only: the place files' in-play columns are placeholders on most rows (ledger `place-fields-check-1002`). |
| **H-% (Second)** | **New: `hx_second_car`, `hx_second_m10`.** |
| **Max field size won** | **New: `hx_maxfield_won`.** |
| **Placing in race last year** | **New: `hx_lastyear_nfp`** (the course and trip 335-395 days before). |
| Wind op, runs since wind op, foal month/quarter | Not in our export (horseracebase holds them). |
| No. times non-runner since LR | Not in our data: race_results holds runners only. |

## Jockey and trainer

| Criteria | In the model |
|---|---|
| Jockey, claim | Served (connection windows, custom metrics, the claim). |
| Jockey male/female, minimum weight | Not in our data. |
| **Same surname** | **New: `hx_same_surname`.** |
| Days/rides since a win; rides on the day, at the track, over 3/7/14/30 days and 1-2 years; wins, win%, place%, P/L by window, track, odds, distance and going; position by win% | Connection windows (served), bookings (served). |
| Rides the main trainer / not the trainer's main jockey | Bookings (served: the yard's main rider's choice). |
| Trainer, yard and primary location | Travel block. |
| Trainer runners on the day, at the track, by window, by jockey and odds; wins, win%, place%, P/L by window, jockey, odds, track, code, distance, horse age, handicap/non; string position; trainer+jockey runners on the day | Connection windows (served), stablemates (built, not served), bookings (served), the custom metrics' trainer-jockey records. |

## Sire and dam

| Criteria | In the model |
|---|---|
| Stallion, origin; runs, wins, win%, P/L by class, age, distance, track, going, surface, 1 and 5 years; horses in the race | Card pedigree served; sire aptitudes (built, not served); sire windows (retired: worse). |
| Dam, dam's stallion; runs and wins of her progeny by age, class, distance, track, going; her own racing career | Dam line (built, not served). |

## The last four runs, the last win and the first run

| Criteria | In the model |
|---|---|
| Each run's course, code, trip, going, runners, class, prize, type, handicap, age restriction, max/avg OR, placing, distance beaten, odds, favouritism, market position, stall, rating, weight, jockey, trainer, days between runs, headgear, distance moved | Form windows (the last run, the last 3/5/10, weighted), form variants, freshness, handicap angles, comments (built). |
| % of the field beaten | NFP (served). |
| Lengths beaten per furlong | Form variants (`fv_lbpf`). |
| Odds to runners ratio | OFS, odds × field size (served). |
| **(LR) BF In Play Min/Max** | **New (`ir_`).** |
| **(LR) Winners Odds** | **New: `hx_lr_winner_lbsp`** (the winner's BSP). |
| **(LR) Non Completers** | **New: `hx_lr_nonfin`.** |
| (LR) Odds Move (opening v SP), wind op | Not in our data (no opening show prices, no wind operations). |
| The first-ever run: code, track, trip, going, class, placing, odds, market position, age, trainer, month, race type | Within the career windows; the yard's debutants as the market priced them (served). |

The "Advanced" items (qualifiers per race, today's data, 48-hour declarations) are the builder's controls, not
features.

## What was built and how it scores

**The in-running block (`model/blocks/inrunning.py`, 13 features):**
- log of the in-running low; the low against the BSP; the high against the BSP. Each over the last run, the last 3,
  the last 5 weighted 5..1, and career;
- the share of beaten runs that traded at 2.0 or shorter;
- the number of runs held.

Each run joins its price-file record by day and normalised name.

**The extras (`model/blocks/hrb_extras.py`, 7 features):**
- same surname;
- the share of seconds (career, last 10);
- the last race's winner's BSP and its non-finishers;
- the biggest field won in;
- last year's run at the course and trip.

**The screen** (`research/queries/done/hrb_features_screen.py`, run 37474104432):
- The sample: 54,151 GB/IE races from 2022 to 31 Mar 2026 (the locked holdout from 1 Apr is not read), 505,032
  runners. Each race has a BSP for every runner and one winner.
- The method: a conditional logit on the winner, fitted on one period and scored on the other (2022-23; 2024 to
  Mar 2026). Each feature is scored on its own and beside the BSP (the market's log chance and its square). Scores
  are millinats a race.
- Coverage: 84.8% of runs have a price-file record (the files start on 11 Apr 2019). The winner's in-running low is
  1.05 or less in 83.0% of races, so the low marks the finish closely.

| | Alone | Beside the BSP |
|---|---|---|
| The in-running block, together | +137.4 (t 61) | −0.69 (t −1.6) |
| In-running low: the last 5 weighted, the last run | +112.0, +102.2 | −0.03, −0.17 |
| The low against the BSP (each window) | +3.9 to +9.2 | −0.03 to −0.05 |
| Share of beaten runs traded at 2.0 or shorter | +20 to +21 | +0.01 |
| The extras, together | +47.7 (t 34) | −0.65 (t −4.8) |
| Same surname | +1.2 | +0.01 (t 0.5) |
| Share of seconds (career, the last 10) | +28.4, +28.9 | −0.00, −0.02 |
| The last race's winner's BSP; its non-finishers | +2.5; +2.6 | −0.06; −0.04 |
| Biggest field won in | +16.6 | −0.33 (t −5.0) |
| Last year's run at the course and trip (known on 6.8%) | +0.5 | −0.18 (t −2.5) |

The market's A/E (winners over the winners its prices expected) by the in-running low of the last run is
0.97-1.01 in every band: 0.988 at 1.5 or shorter, 0.966 from 1.5 to 2.0, 1.014 over 10. A horse that traded short
and was beaten wins as often as its price says. The rider-trainer family bookings (1.9% of runners) won 14.1%
against the market's 13.7%: a little above the price, but well inside the noise.

What it means:
- The in-running record is a strong summary of form on its own: it carries the finish and how the race was run.
  But the market already prices it. Nothing here adds to the BSP on the winner.
- Two extras do worse than nothing beside the BSP out of sample (the biggest field won, last year's run): they fit
  noise.
- The served model forecasts the BSP rather than beating it on the winner, so the deciding test is iteration 109.
  It puts each block in front of the served model (main_ir, main_hx) and scores the price forecast, the rule and
  the owner's staking.

**Iteration 109** (research loop run 37474104512; the served recipe, 53,910 runners in 5,923 races, 27 Sep 2025 to
31 Mar 2026):

| | Price-forecast error (90% CI) | Rank 1 | Brier skill, concordance | The rule | The owner's staking at >= 3% |
|---|---|---|---|---|---|
| The served main | 0.3973 | | | +9.04% | CLV +6.50% (8,605 bets) |
| + the in-running block | **-0.0008 (-0.0014 to -0.0002)** | -0.0020 (-0.0030 to -0.0010) | level | +9.07% | CLV +6.88% (8,620 bets) |
| + the extras | +0.0001 (-0.0004 to +0.0007) | -0.0011 | level | +9.05% | CLV +6.66% |

- The in-running block is the first block to improve the served main since 28 Sep. The gain is small (0.2% of the
  error) and largest for the favourites (0.8%), where the money is.
- The served model is not the main alone. It is three models blended (data/models/bfsp_ensemble.json): the main and a
  partner averaged, then race_xent gated in by rank, so the three shortest prices in a race are race_xent's alone.
  A gain in the main reaches those prices only through the ranks. **Iteration 110** fits the block into all three
  members, compares the blend with it against the served blend refitted, and repeats the main at a second seed
  (seed 7), because refits alone move this measure by a few ten-thousandths.
- If it holds, the block is trained into the served models and goes through the usual verification, the
  matrix-against-live parity check and a dry run before it is served. The rule itself does not change.
- The extras are retired.

## What horseracebase holds that our export does not take

- Wind operations: the first run after one is a known angle, and the BHA has declared them since 2018.
- Foal dates: the relative-age effect in two-year-old races.
- Owners and colours.
- The jockey's sex and minimum riding weight.
- Non-runner history.
- Forecast, early and opening show prices.
- Big-race trend tables.

Of these, wind operations and foal months are the two worth adding to the scrape if horseracebase's export carries
them. The market sees both, so they would need testing like the rest. The owner decides whether to extend the
scrape.
