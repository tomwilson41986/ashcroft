# International racing: nine Parse sources tried (1 Nov 2026)

The owner's ask (1 Nov 2026): try nine Parse (parse.bot) racing APIs, so our model can run on international racing.
Each was subscribed (free) and probed with real past dates on 1 Nov 2026, 42 credits in all. Probe responses:
`data/parse/intl_probes/` (gzipped JSON). The plan they feed: `reports/international_racing.md`. Every figure below
comes from those responses.

## The result

1. **Two sources can build a form database with history: netkeiba (Japan) and HKJC (Hong Kong).** Both return a
   year-old race in full, for every runner. netkeiba gives the gate, weight carried, time, margin, odds and
   favouritism, the last 600 metres, running positions and body weight. HKJC gives the draw, weight, horse weight,
   lengths beaten, running positions and odds. HKJC adds each runner's sectional times, and a horse's whole
   Hong Kong record with its rating, back to 2019 for the horse tried. Each costs about 1 credit a race.
2. **Neither is known to trade on Betfair.** Betfair publishes no price files for Hong Kong or Japan
   (`reports/international_racing.md`), and October's census does not settle it. A by-country racing census now
   answers it (section 3). Nothing is built for either until it shows exchange money.
3. **Australia, the market with the money, has no history among the nine.** TAB crashes on every date. Racenet and
   Neds give today only. Racenet's field is rich: barrier, weight, a form summary, and 18 price sources including
   Betfair's back and lay. But it costs 10 credits a race and holds no past results.
4. **Germany has full results but no way to list a day's races.** Without race ids, Deutscher Galopp cannot be
   backfilled through Parse.
5. **The USA is out.** Equibase's card results name only the first three finishers, with payoffs and no margins.
   A six-month-old card (Churchill Downs, 2 May 2026) and a year-old date both return nothing. Its terms forbid
   automated access without written permission (`reports/international_racing.md`). Brisnet offers news, picks and a
   legacy results index, not form.

## 1. What each source gave

| Source | Jurisdiction | What a probe returned | History | Credits | Verdict |
|---|---|---|---|---|---|
| race.netkeiba.com | Japan (JRA) | `list_races` for 2 Nov 2025: 2 meetings, 24 races, each marked resulted. `get_race_result`: every runner with finish position, gate, horse number, sex, age, weight carried, jockey, trainer, time, margin, favouritism, win odds, last 600 metres, running positions at each corner, body weight and its change | At least a year | 1 a day + 1 a race | **Build, if Betfair trades JRA** |
| racing.hkjc.com | Hong Kong | `get_race_results` for 2 Nov 2025 race 1 (Happy Valley): 11 runners with placing, draw, actual and declared weight, lengths beaten, running positions, finish time, win odds. `get_sectional_times`: each runner's position, margin and time at every section | At least a year | 1 a race + 1 for sectionals | **Build, if Betfair trades Hong Kong** |
| hkjc.com | Hong Kong | `get_horse_rating_history` (Golden Sixty): 31 runs, 31 Mar 2019 to 28 Apr 2024, each with rating, class, distance, going, draw, jockey, trainer, lengths beaten, odds, weight, running positions, time, gear | A horse's whole local career | 1 a horse | With racing.hkjc.com |
| deutscher-galopp.de | Germany | `get_race_results` (München, 22 Aug 2026): position, box, lengths beaten, weight, jockey, trainer, owner, prize, odds, ground | Not tested further: no endpoint lists a day's race ids | 1 a race | **Park**: cannot enumerate races |
| racenet.com.au | Australia | `list_meetings`: 8 meetings, 78 races for 2 Nov 2026, with track condition and rail. `get_race_fields`: barrier, weight, jockey, trainer, form summary (career, last ten, days since the last run), 18 price sources including `betfair:exchange-win` and `exchange-win-lay` with their moves | Today only | 10 a call | **Park for history**; a possible live feed of form summaries, at about 470 credits a day for every Australian race |
| tab.com.au | Australia | Every call crashed (500), for 31 Oct 2026 and 1 Nov 2025; the listing has never been health-checked | None | 1 | **Broken** |
| neds.com.au | Australia | `get_racing_events`: today's meetings and race ids across thoroughbred, harness and greyhound racing (547 KB) | Today only | 1 | Park: a bookmaker's card, no form |
| equibase.com | USA | `get_race_dates`, 31 Oct 2026: 18 tracks. `get_card_results` (Belmont, 31 Oct 2026): 11 races, the first three with payoffs only. Churchill Downs, 2 May 2026: not found | Recent days only | 2-5 | **Out**: thin results, no history, terms |
| brisnet.com | USA | `get_results_index`: a list of legacy results files by track; the other endpoints are news, picks and an article search | Not form | 1 | **Out** |

## 2. What it would cost to backfill

| Source | Races a year (estimate) | Credits for a year | Note |
|---|---|---|---|
| netkeiba | about 3,450 JRA races (24 races a JRA day in the probe, about 144 race days; unverified count) | about 3,600 | The day lists are 1 credit each |
| racing.hkjc.com | about 800 races a season (unverified) | about 1,600 with sectionals | Plus 1 a horse for the form records |

Both fit the plan's spare credits (about 6,000 a month after the odds schedule, `parse_odds.md`). Neither is
started, because the model needs an exchange to trade on first (the owner's rule: exchange trades only).

## 3. The deciding check: does Betfair trade these races?

`betfair_recorder.py --census-racing` reads, for every country Betfair lists horse racing in: its win markets,
venues and the money matched so far. It makes one read-only call per country, on the delayed key. It runs each
evening in the recorder job on the UK server (`live-record.yml`, after the sports census), with a 7-day window, so
it lists every country's markets for the week, Hong Kong and Japan included. Readings are saved to
`betfair_live/<day>/census_racing_<HHMM>.json`.

A first attempt from GitHub's own runners failed: Betfair refused the login (403 at identitysso.betfair.com) from
that host. So the census runs only where the account's other Betfair reads run. The first reading comes the evening
after this branch merges, because the recorder job runs from the default branch. The evening reading lists the
week's Hong Kong and Japanese markets before they trade much. The money near the off would need a reading while
those countries race, about 01:00-09:30 UK. The UK server is busy then (the price archive to 06:15, then the
trader), so that is left until the listing shows the markets exist.

## 4. The backfill (1 Nov 2026): a year of Japan and Hong Kong

`sources/parse_racing.py` (2 tests): every race from 1 Nov 2025 to 31 Oct 2026, raw responses kept, tables in
`data/parse/` (from the schedule's store, `sources/parse/`).

| Table | Rows | Covers |
|---|---|---|
| `jra_races` | 3,438 races | All 10 JRA courses; 109 race days (52 Saturdays, 52 Sundays, 5 holiday Mondays); turf 1,690, dirt 1,629, jumps 119 |
| `jra_results` | 47,596 runners | 11,371 horses; median field 15. No duplicate race-horse pairs. 363 without a finish, all non-finishers: 195 did not finish (中止), 111 excluded (除外), 57 withdrawn (取消). 166 without odds. The going on every runner |
| `hk_results` | 10,810 runners | 88 meetings (Sha Tin and Happy Valley), 872 races, 1,473 horses; median field 12. No duplicates and no missing draw, odds, class or venue |
| `hk_sectionals` | 39,354 runner-sections | Every one of the 872 races |

Data checks: every Japanese call succeeded (157 day lists, 3,438 results). Of the Hong Kong calls, 28 returned an
error, all race-1 probes of days with no meeting (mostly Saturdays and the summer break). A first pass probed
Wednesdays, Saturdays and Sundays only. A second pass probed every day and found 6 more meetings on holiday weekdays
(3 Thursdays, 2 Mondays, 1 Tuesday) with 66 races. Cost: 5,537 credits in all (Japan 3,545, Hong Kong 1,920, the
trial weekend 72), leaving 7,892.

Names are as each source gives them: Japanese courses, horses and jockeys in Japanese, Hong Kong's in English. A
join to Betfair's runner names, if the census shows those markets, needs a name map for Japan.

## 5. Next

1. If the census shows Hong Kong or Japanese win markets with money: build that jurisdiction's metrics on the
   backfilled results, following the greyhound port (`reports/international_racing.md` section 3), extend the
   history further back (about 3,500 credits a year for Japan, about 1,900 for Hong Kong), and record its Betfair
   books for the market-only edge scan.
2. Australia still needs a history source: Punting Form (with its consent), or Betfair's own Australian data
   (`reports/international_racing.md` sections 1.2 and 2).
3. Unsubscribe the sources marked out or broken, if the owner wants the account kept tidy: TAB, Equibase, Brisnet,
   Neds. Subscriptions cost nothing.
