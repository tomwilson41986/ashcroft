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
   answers it (`betfair-racing-census.yml`, section 3). Nothing is built for either until it shows exchange money.
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
venues and the money matched so far. It makes one read-only call per country, on the delayed key. It runs three
times a day (`betfair-racing-census.yml`), so each region is read while it races: 05:30 UTC (Japan, Hong Kong,
Australia), 12:30 (Europe, South Africa) and 21:30 (the Americas). Readings are saved to
`betfair_live/<day>/census_racing_<HHMM>.json`. The first reading, from this branch over the week ahead, is below
when it lands.

## 4. Next

1. If the census shows Hong Kong or Japanese win markets with money: backfill that jurisdiction's results through
   Parse into its own data module, following the greyhound port (`reports/international_racing.md` section 3), and
   run the market-only edge scan once its Betfair books are recorded.
2. Australia still needs a history source: Punting Form (with its consent), or Betfair's own Australian data
   (`reports/international_racing.md` sections 1.2 and 2).
3. Unsubscribe the sources marked out or broken, if the owner wants the account kept tidy: TAB, Equibase, Brisnet,
   Neds. Subscriptions cost nothing.
