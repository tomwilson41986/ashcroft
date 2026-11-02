# Parse odds: Flashscore, Pinnacle and Oddschecker, saved and modelled (1 Nov 2026)

The owner's ask (1 Nov 2026): "Check what we can get. Save, database and use in modelling", for the owner's three Parse
(parse.bot) APIs: Flashscore, Pinnacle and Oddschecker. Code: `sources/parse_odds.py` (collectors, tables, panels),
`tests/test_parse_odds.py` (5 tests), `.github/workflows/parse-odds.yml` (the schedule). Tables:
`data/parse/*.parquet` (this session's capture); from the schedule, `s3://$CAPTURE_BUCKET/sources/parse/`. Every figure
below comes from those tables or from the response headers of the calls made on 1 Nov 2026.

## The result

1. **Each API gives something we did not hold.** Flashscore gives tennis opening odds, which no source we held had
   (`data.md` 6.4). Pinnacle gives the sharp line again, since its own API has been closed to the public since July 2025
   (`data.md` 6.3). Oddschecker gives 25 UK bookmakers' prices for every GB and IE runner, an early horse market deeper
   than Betfair's 08:00 book.
2. **Only Flashscore has history**: seven days of every sport from its daily feed, and per-match odds back months (the
   oldest held is 30 Jun 2026). Pinnacle and Oddschecker hold only what we record, so their value starts with the
   schedule.
3. **Flashscore's bookmakers are the US books** (BetMGM, bet365.us, FanDuel), because Parse's proxy is in the US.
   Neither Pinnacle nor Betfair appears there.
4. **Tennis, the first model: no edge yet.** On 3,403 bookmaker markets over 1,898 tour-level matches (Jun-Nov
   2026), the development months find nothing: no model beats no change, and every hedged trade loses after a tick
   at each end. The holdout (Oct-Nov, read once) shows small positives that the development months do not support
   (section 4). The prices are US bookmakers', and Betfair's pre-match tennis money is a median GBP5 a market
   (`all_markets.md`), so nothing here is a Betfair trade.
5. **Exchange trades only** (the owner, 1 Nov 2026): no bets with bookmakers. Bookmaker prices are information, never a
   venue. Every trade tested here has its legs on Betfair or another exchange.
6. **Horses are tomorrow's test.** Today's capture (62 GB/IE races, 661 runners, 25 price sources, 79 snapshots) is
   joined to the Betfair books and the BSP once tonight's load holds the day (`research/queries/parse_horses_1102.py`).
   It asks whether the bookmakers' morning price signals where the Betfair price goes, traded on Betfair: back at the
   08:00 best back or lay at the 08:00 best lay, then close at the BSP. It also prices a cross-exchange version, backing
   at Matchbook's morning price and laying at the Betfair BSP.
7. **Matchbook is in the capture.** Oddschecker's code `MA` sits a median 9.7% above the 25 sources' median morning
   price, and above it on 92% of runners. Every other code sits at the median, as an exchange's back price (no
   bookmaker margin) would. That points to Matchbook, but the name is unverified. Betfair Sportsbook (`BF`) is a
   bookmaker and is not an exchange leg.

## 1. What each API gives

| API, endpoint | What | History | Credits |
|---|---|---|---|
| Flashscore `get_fixtures_with_odds` | Every match of one sport on one day: teams, competition, start (UTC), score, status, the opening and final match-winner odds | 7 days (day offset -8 returns none) | 1 a sport-day |
| Flashscore `get_match_odds` | Every market of one match (1X2 or match winner, set and half markets, totals, handicaps, correct score): each bookmaker's opening and final price | Back to at least 30 Jun 2026 (a May 2026 match returned odds in the probe) | 1 a match |
| Flashscore `get_team_results` | A team's or player's past results, about 40 a page; the slug is not checked | Two pages reach Oct 2025 for a tour player | 1 a page |
| Pinnacle `get_matchups` + `get_odds` | Every pre-event line of a sport: American prices, the line's version, its maximum stake; names and starts from the matchups | None: polling makes the open and close | 3 + 2 a sport |
| Oddschecker `get_horse_racing_meetings` + `get_horse_race_odds` | The day's meetings (74 on 1 Nov, 9 of them GB/IE), then every runner's fractional price at each bookmaker | None | 1 + 3 a race |

The account: Team plan, 20,000 credits (the `x-credits-limit` header), 300 requests a minute. Whether the 20,000
renews monthly is unverified.

## 2. What was saved (1 Nov 2026)

| Table | Rows | Covers |
|---|---|---|
| `flashscore_matches` | 12,480 matches, 4,171 with opening odds | 14 sports, 25 Oct-1 Nov 2026 (golf events from 21 Oct). With opening odds: tennis 1,785, football 1,197, basketball 455, hockey 293, handball 256 |
| `flashscore_odds` | 87,094 prices | 1,987 matches, 30 Jun-1 Nov 2026, almost all tour-level tennis: BetMGM 43,727, bet365.us 41,557, FanDuel 1,810; 388 without an opening price, 433 without a final one |
| `flashscore_results` | 11,492 past results | 180 tour players' last two pages; 6,003 tour-level singles matches found, 2,000 of them fetched (newest first) |
| `pinnacle_prices` | 18,581 prices | The first snapshot of each sport: soccer 1,716 matchups, tennis 226, cricket 6; median moneyline maximum stake 375 (soccer), 350 (tennis), 250 (cricket), presumably USD (unverified) |
| `oddschecker_prices` | 16,452 prices | 62 GB/IE races, 661 runners, 25 bookmakers (Betfair Sportsbook `BF` and Matchbook `MA` among them), 79 snapshots: every race once from 11:53 UTC, then at T-120, T-60 and T-15 |

Data checks: no duplicate match ids; no runner price failed to convert. Each raw response is kept as it came, with
its fetch time (UTC), the call and the credits charged. The tables are rebuilt from raw on every build, so a rebuild
is safe to rerun. The key never reaches a file: it is read from `PARSE_API_KEY`. Spent on 1 Nov: about 2,700 credits
(the probes, the 7-day backfill, the tennis history, the recorders), leaving 13,480 after the tennis history.

### The day's capture, as it ended (1 Nov, 18:45 UTC)

The two forward recorders ran inside this working session, and stopped each time the session went idle. Both were
restarted at 15:23 UTC with network retries, and both had stopped by 15:59. What the day holds:

| Recorder | Held |
|---|---|
| Oddschecker | 62 GB/IE races, 661 runners, 103 snapshots: every race once from 11:53 UTC; 40 snapshots 130+ minutes before the off, 16 at 70-130, 25 at 20-70, 19 within 20 minutes (7 races) |
| Pinnacle | 6 snapshots (tennis, soccer and cricket at 11:53 and 15:23 UTC) |

So tomorrow's horse test reads the bookmakers' price from late morning (the first snapshot), not 08:00, and has no
T-15 reading worth scoring. Forward recording belongs to the scheduled job (`parse-odds.yml`), which runs once the
key is a repository secret, not to a working session. Raw responses for the day: `data/parse/raw_capture_odds_2026-11-01.tar`.

## 3. Into the models

- **The panel.** `panel_flashscore` turns the feed, and every bookmaker's winner market, into the standard open-close
  panel (`sources/open_close.py SCHEMA`): the opening price as the open, the final price as the close, and the result
  from the score. `panel_pinnacle` does the same for Pinnacle's full-match moneyline. Its open is the first price the
  record saw, its close the last before the start, and the maximum stake stands in for the money.
- **The drift models** (`research/open_close`) run on the panel as on every other market: walk-forward by month, the
  hedged trade after 2% and a tick each end, the leak test, the holdout read once. For bookmaker prices, each book is
  scaled to 100% at both ends first (`--normalise-book`), because the margin changes between the open and the close.
- **Horses.** The bookmakers' morning price is a candidate input to the closing model (`model/race_book.py`) and to
  the BFSP forecast's morning features: information for the Betfair trade, never a bet. It can be trained only once
  weeks of the schedule's capture are held. The first read is tomorrow's join.
- **Other exchanges.** Matchbook's price arrives through Oddschecker (`MA`). It gives one cross-exchange trade to price:
  back on Matchbook, lay at the Betfair BSP. Matchbook's own commission and its API are to check before any money.

### Horses: the first read (1 Nov, research-query run 41421619889)

One day, so descriptive only, not a test. 384 GB/IE runners joined to a Betfair win market with a BSP (of 661
captured; the join is by course, off time and name).

| Morning price | log(BSP / price), median (IQR) | Log loss of the winner (36 full races) |
|---|---|---|
| Bookmakers' median, first snapshot (median 171 minutes before the off) | +0.41 (+0.13 to +0.85) | 1.824 |
| Bookmakers' best | +0.13 (-0.07 to +0.41) | |
| Betfair 08:00 mid | +0.03 (-0.31 to +0.35) | 1.979 |
| BSP | | 1.756 |

The trade the bookmakers' price would signal, every leg on Betfair (back at the best back, or lay at the best lay,
closed at the BSP, 2% commission):

| Version | Bets | Net | Read |
|---|---|---|---|
| At Betfair's 08:00 book | 262 | -0.98 on 461.6 at risk; the 20%+ back cell +8.9% on 138 | **Look-ahead**: the bookmakers' prices were taken hours after 08:00 |
| At Betfair's 12:00 book (bookmakers' snapshot within 30 minutes, the race after it) | 314 backs | **-24.02 on 314, -7.6%**; every gap band negative | The tradable version |
| Matchbook's price backed, laid at the Betfair BSP (12:00) | 101 | -7.8% a GBP1 backed | Before Matchbook's commission |

Read:
1. The 08:00 cell's +8.9% was the look-ahead. At the time-matched book, the signal loses in every band.
2. The comparison is also unfair to the signal. The bookmakers' raw median carries their margin, so it sits
   shorter than Betfair's best back on almost every runner (314 of 326) and says little about a single horse. The
   next version compares chances with the margin removed, at matched times.
3. The bookmakers' morning book forecasts the result better than Betfair's 08:00 mid, but it was read about four
   hours later, so that is no evidence of an edge.

Nothing here supports an exchange trade. The fair test needs the scheduled recorder's 07:40 snapshot each day, a
normalised book, and weeks of days.

## 4. Tennis: the first result

`reports/open_close/model_tennis_flashscore.md` (figures `parse_runs/open_close_tennis_flashscore.json`): 6,806
selections in 3,403 bookmaker markets, 2026-06 to 2026-11 (34 markets whose results did not sum to one dropped, most
likely retirements). Development: Aug-Sep scored walk-forward after two months fit. Holdout: Oct-Nov, read once.

| | Development (Aug-Sep, 1,260 markets) | Holdout (Oct-Nov, 1,754 markets) |
|---|---|---|
| Best R2 against no change | under zero for every model | 0.02-0.07 (conditional logit highest) |
| Hedged trade at threshold 0, a tick worse each end | -3.1% to -3.6% | -0.6% to -2.0% |
| Band model at its chosen threshold (0.02), a tick worse | -1.95% (96 bets) | +7.5% (62 bets, 90% interval +0.1% to +15.9%) |

Read: the development months show no drift worth trading. The holdout's positives are small samples that the
development months contradict. They are worth re-reading once the other 4,003 tour matches are fetched (about 4,000
credits, back to late 2025). That re-read is pre-registered here: the band model, threshold 0.02, a tick worse each
end, passes only if its interval stays above zero on the longer development window. Even then the trade would be a
signal for Betfair's price on the tour, not a bookmaker trade.

## 5. The schedule (`parse-odds.yml`; runs once merged and keyed)

| Job | When (UTC) | Credits a month (estimate) |
|---|---|---|
| Flashscore yesterday (all sports) + yesterday's tour-level tennis per-match odds (up to 80) | 02:35 | about 1,900 |
| Pinnacle tennis, soccer, cricket | every 2 hours, 06:05-22:05 | about 4,050 |
| Oddschecker: every GB/IE race's morning book, then T-15 | 07:40 and 13:25 | about 8,100 |

About 14,000 of the 20,000 a month. A floor stops every collector when the account's credits fall under 2,000
(`PARSE_CREDIT_FLOOR`). The settings are repository variables: `PARSE_TENNIS_MATCHES`, `PARSE_PIN_SPORTS`,
`PARSE_OC_MARKS`.

## 6. The owner's actions

1. **Rotate the Parse key.** It was pasted into a chat. Then add the new one as the repository secret `PARSE_API_KEY`,
   so the schedule can run.
2. **Check the terms.** Flashscore, Pinnacle and Oddschecker are scraped through Parse, and racing²'s use is
   commercial. Their terms have not been read here.
3. Decide whether to spend about 4,000 credits on the rest of the tennis history (section 4).
4. Oddschecker is the largest cost in the schedule (about 8,100 credits a month) and is now a signal only. If
   tomorrow's join shows it adds nothing beyond the Betfair book, set `PARSE_OC_MARKS` to fewer marks or drop the job.
