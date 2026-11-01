# Every market on Betfair, evaluated for an open-to-close trade (research agent, 1 Nov 2026)

The owner's ask (1 Nov 2026): "Evaluate all markets on Betfair." The purpose is fixed by the earlier ask of the same
day: a hedged pre-off trade from the opening price to the closing price, automated, with no reliance on the result
(`reports/open_close/markets.md`). This note scores every sport Betfair lists, and every market type inside each,
on the same criteria, and gives every row a verdict. It is not a shortlist.

Conventions: hyphens only; every number traces to research-query run 41291139483 (job 124802487010; "C2" is its census
section, 25 days 7-31 Oct 2026, 24 sports; "A2", "A3", "B4", "D" its other sections), to a repository file, or to a
URL in section 5; "unverified" marks a claim with no source. Cells marked **census pending** wait for the data agent's
complete census dump (`census_all_markets_1101.py`), which lists every sport x market type row with its busiest
events; they are filled when its log lands. "A day ahead" means the money matched by the census's evening read on a
sport's 200 busiest markets starting in the next 24 hours (`betfair_recorder.py`, `census_sports`, lines 402-438).

## The result in six lines

1. **Of 35 event types Betfair's API lists, 24 had markets in October 2026 and 11 had none** (C2; section 1). Of the
   24, four carry pre-off money a day ahead in size: cricket (GBP6.8m on its top 200 markets, in about 17 match-odds
   markets), football (GBP0.9m, spread over 185 match-odds and 185 Asian-handicap markets), golf (GBP0.65m, in 3
   outrights) and American football (GBP0.29m, in 3 handicap markets). Horse racing (GBP23k) and greyhounds (GBP109)
   have almost nothing a day ahead because their money arrives on the day: 83% of a UK win market's pre-off money
   after T-60 (A2).
2. **Only two sports have a single-price close**: horse racing and greyhound racing (the Betfair SP), and only there
   does the exit cost no spread. Everywhere else the close is the last pre-play trade and the exit crosses a tick or
   rests an order (`betfair_historic.py`, lines 1-4; `markets.md` section 2).
3. **One market is proven and live:** UK/IE horse win, +6.15% CLV on GBP47,274 staked 19-31 Oct (research ledger
   `staking-1031`). **One has a measured market-only drift to test:** GB greyhound win, where the first traded price
   drifts out to the BSP by a median 2.5-13% at 3-51 (D). **Two need data first:** football (the free Historic Data
   Soccer plan) and cricket (the Cricket plan). **One is research later:** golf outrights.
4. **Every other sport and every secondary market type is parked or never**, each for a stated reason: no money a day
   ahead (tennis's median market GBP5; basketball GBP4; darts GBP872 across the top 200), no close (politics, specials),
   an information event that is the whole move (the toss, the weigh-in, the line-up), or no evidence anywhere that
   the move is predictable from the market (every sport outside racing and football: section 3).
5. **The literature's one general result** holds across sports: the close is a better forecast than the open, and a
   price's distance from the close predicts its return about 1:1 (Buchdahl on Pinnacle football; datagolf on golf;
   Gandar et al. 1998 on NBA spreads; Whelan 2025 on Betfair football). It does not hand over a drift one can trade
   from the market alone; where a drift is systematic (the favourite-longshot bias unwinding in racing), the spread
   is of the same order as the move, except where the exit is the SP.
6. **The owner's actions that unblock the most:** "buy" the free BASIC plans for Soccer, Cricket and Golf at
   historicdata.betfair.com (`reports/open_close/data.md` section 6); keep the greyhound and horse recorders running;
   nothing paid.

## 1. The master table: every sport, ranked

The universe is Betfair's event-type list as `listEventTypes` returns it. Betfair publishes no table of ids; the list
below is the union of three public client libraries that embed the API's response (jBetfair `EventTypes.java`, 31
types; the betfairlightweight tutorial's `listEventTypes` output, 29; less-stupid-betfair, 26: section 5), which agree
on every shared id, plus the 24 sports the census saw. Betfair adds and retires types; the complete census dump is
the live list (census pending for any type it adds).

Columns: census (days seen of 25 / markets a day / top-200 matched a day ahead, GBP / median market of the 200 /
concentration = mean over median); open = when a tradable price first exists; close = BSP (single price) or LPT (last
pre-play trade); IP = whether the market turns in play (Betfair aims to turn markets in play at the off for all but
soccer and Australian markets, per a copy of Exchange Rules Part B 1.2 in a forum post: unverified against the live
rulebook); events = information that arrives before the off; drift = (a) predictable from the market alone, (b) only
with a model or information, (c) unpredictable or untested, with the source; cost = one tick at the typical price
(`markets.md` section 5.2); verdict.

| # | Sport (event type id) | Census: days / markets a day / a-day-ahead GBP / median market / concentration | Open | Close | IP | Events before the off | Drift evidence | Data route | Cost | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Horse Racing (7) | 25 / 413 / 23,155 / 4 / 28.9 (the evening before; on the day a UK win market holds GBP1,423 at 08:00, GBP13,493 at T-60, GBP77,294 at T-1: A2) | 06:15 UK on the day; the evening before unformed (ledger `evening-before-1007`) | **BSP** | Yes, at the off (IPMIN/IPMAX in the files: ledger `edge-inplay-offsets-1002`) | Non-runners, going, jockey changes, the market's own money (80% of pre-off volume in the last 20 min: Brown & Yang 2017) | (b) proven: the model carries half the weight beside the morning price (TRADING.md); (a) back side none, lay side at 8-21 a 2-5% drift of the order of half the spread (A3, A2; `markets.md` 4.1) | Price files 2017 on, recorder from 1 Oct, `betfair_live_marks` | 2.6% at 8, 3.4% at 15; nil at the SP exit | **Trade now** (UK/IE win, live). **Test next:** the 8-21 lay with the model (`markets.md` 6.1). Place, TBP, each way: park (closed 3 Oct). Ante-post: research later. AU/FR/USA/RSA: park |
| 2 | Greyhound Racing (4339) | 25 / 481 / 109 / 0 / n/a | On the day, close to the off; the first trade thin (morning WAP on 14.7% of runners: `greyhound_model.md` s.4) | **BSP** | Yes, at the off | Trap draw is known days ahead; vacancies and reserves; the market's own money | (a) measured: GB first traded -> BSP median +0.025 (3-5) to +0.129 (13-21), +0.117 (21-51), shortened 33-45%; T-1 -> close still +0.017 to +0.057 at 5-51 (D); morning WAP -> BSP +0.137 to +0.215 at 2-51 since 2012 (B4). (b) form predicts which dogs shorten (`greyhound_model.md` s.4) | GBGB 2018 on, price files 2012 on, the 2026 bundle, GB/IE books from 6 Oct | 1.4-2.6% at 3.5-8; nil at the SP exit | **Test next:** lay at the first two-sided book where the best lay is 8-51 and GBP10+ is offered, closed at the BSP (`markets.md` 6.2, pre-registered) |
| 3 | Soccer (1) | 25 / 6,106 / 908,482 / 993 / 4.6 | Days ahead for the big leagues (unverified); the last hour adds 55-150% (betaminic: EPL GBP381k at T-60, GBP592k at T-1) | LPT | Yes; football is one of the two exceptions to the at-the-off rule (managed in-play reform after goals) | Line-ups about an hour before kick-off, injuries, weather, the week's results | (c) market alone: the pre-play LPT is an unbiased forecast; takers -2.5%, makers +0.6% on 902,568 bets 2022-24 (Whelan 2025); (b) team news: the close beats the open by 0.0034 log-loss and a team model cannot see why (`football_model.md` s.3) | football-data.co.uk 2012 on; Historic Data Soccer BASIC once held; gold tables | 0.8-1.4% at 2.5-3.5 | **Data first:** the free Soccer plan, then the first -> close read by tier and type (`markets.md` 6.3). MATCH_ODDS and ASIAN_HANDICAP first |
| 4 | Cricket (4) | 25 / 135 / 6,799,296 / 0 / n/a (17 match-odds markets carry GBP1.67m; 118 other markets hold nothing) | Days ahead for internationals and franchise leagues (the census sees it a day ahead) | LPT | Yes, at the start; the toss market settles before it | The toss and the teams about 30 min before the start (general knowledge, unverified); weather and pitch | (c) pre-match untested anywhere found; in play the market over-reacts in the first innings (Norton, Gray & Faff 2015), outside the brief | Historic Data Cricket BASIC once held; nothing held | 0.7-0.8% at 1.5-2.5 | **Data first:** the free Cricket plan; the first -> close and T-60 -> close read by format; a hedged trade must close before the toss window or price it |
| 5 | Golf (3) | 17 / 37 / 653,691 / 378 / 8.6 (WINNER GBP506,658 on 3 markets; TOP_10 GBP44,235; TOP_5 GBP27,513) | A week ahead for outrights; matchups a day ahead (datagolf) | LPT at the Thursday start | Yes, across four days | The field, withdrawals, the draw of tee times, weather | (b) Pinnacle's close prices the return about 1:1 and its open moves 54.7% towards Betcris's (datagolf, 106,204 matchups); Betfair untested | Historic Data Golf BASIC once held | 3.4-7.1% at 15-100 | **Research later:** depth on 3 markets a week, long prices (a tick is 3-7%), no model; data first if the owner wants the read |
| 6 | American Football (6423) | 25 / 19 / 291,257 / 1,554 / 0.9 (HANDICAP GBP143,448, COMBINED_TOTAL GBP54,122, MATCH_ODDS GBP37,564 on about 3 markets a day) | Days ahead (US books open the week before: general knowledge, unverified) | LPT | Yes | Injury reports, weather; US sharp books lead (unverified) | (c) opening and closing NFL lines not significantly different in accuracy 2002-11 (arXiv 1211.4000); (b) NBA spreads: line moves erase the opening bias by the close (Gandar et al. 1998) | Historic Data American Football BASIC (plan existence unverified) | 0.7-0.8% at 1.9 | **Park:** three markets a day, a weekly sport, the move made in US books; revisit only with the census's `busiest` list and a US-book feed |
| 7 | Tennis (2) | 25 / 2,781 / 115,954 / 5 / 116.0 (MATCH_ODDS GBP107,574 on 105 markets; HANDICAP GBP2,651) | A day ahead for a handful of main-tour matches; most markets empty (Italian exchange: median EUR0 across 6,477: theoddslab) | LPT | Yes | Withdrawals, retirements, scheduling, weather | (a) favourite-longshot bias on exchange tennis (Abinzano et al. 2016); pre-match drift on Betfair untested; about 90% of money in play (trading guide, unverified) | Sackmann, tennis-data.co.uk; Historic Data Tennis BASIC once held | 0.7-0.8% at 1.5-2.5 | **Park:** concentration 116 says a few matches carry the money; re-read the `busiest` field (census pending) before any test |
| 8 | Basketball (7522) | 25 / 126 / 92,200 / 4 / 115.2 (MATCH_ODDS GBP34,533 across 42 markets, about GBP800 each) | A day ahead, thin | LPT | Yes | Injury reports (late scratches are the move: Player absence and betting lines in the NBA, 2015) | (b) NBA line moves are informed (Gandar et al. 1998) | Historic Data Basketball BASIC (unverified) | 0.7% at 1.9 | **Park:** GBP800 a market a day ahead |
| 9 | Mixed Martial Arts (26420387) | 8 / 24 / 42,997 / 6 / 35.8 (MATCH_ODDS GBP40,866 on 7 markets, event nights only) | Weeks ahead for a card (unverified) | LPT | Yes | Weigh-in, late replacements | (c) untested | None | 0.7-1% at 1.5-3 | **Park:** one or two cards a week, event-driven |
| 10 | Ice Hockey (7524) | 25 / 80 / 30,179 / 0 / n/a (MONEY_LINE GBP18,817 across 20 markets) | A day ahead, thin | LPT | Yes | Goalie confirmations | (c) untested | None | 0.7-1% | **Park:** under GBP1,000 a market |
| 11 | Baseball (7511) | 20 / 8 / 25,453 / 12 / 10.6 (OUTRIGHT_WINNER GBP52,891; MATCH_ODDS GBP7,550 on 2 markets: the season's end) | A day ahead | LPT | Yes | Starting pitchers (the move: general knowledge, unverified) | (c) untested here | None | 0.7-1% | **Park:** season over in October; census pending for a summer read |
| 12 | Motor Sport (8) | 5 / 10 / 22,768 / 442 / 0.3 (WINNER GBP14,252) | Race week | LPT at lights out | Yes | Qualifying (the whole move: unverified), weather | (c) untested | None | 3-7% at long outright prices | **Park:** one event a week, qualifying is the information |
| 13 | Rugby League (1477) | 10 / 20 / 7,544 / 0 / n/a | A day ahead, thin | LPT | Yes | Team lists | (c) untested | None | 0.7-1% | **Park:** thin |
| 14 | Boxing (6) | 8 / 15 / 6,746 / 0 / n/a (MATCH_ODDS GBP6,383; max day GBP238,809) | Weeks ahead for a title fight | LPT | Yes | Weigh-in | (c) untested | None | 0.7-1% | **Park:** event-driven, thin except big nights |
| 15 | Rugby Union (5) | 13 / 17 / 3,881 / 0 / n/a (MATCH_ODDS GBP3,175) | A day ahead, thin | LPT | Yes | Team lists | (c) untested | None | 0.7-1% | **Park:** thin |
| 16 | Snooker (6422) | 8 / 8 / 3,344 / 131 / 0.1 | Tournament days | LPT | Yes | None of note | (c) nothing found | Historic Data Snooker BASIC (in `PLAN_SCOPE`) | 0.7-1% | **Park:** GBP3k a day ahead |
| 17 | Politics (2378961) | 7 / 1 / 2,040 / 1,993 / 0.0 (max day GBP205,093; GBP317m across the 2024 US election markets: oddschecker) | Months ahead | **No scheduled close**; settles on the result | Stays open through counting | News, polls | (b) markets beat polls and improve nearer the vote (Vaughan Williams & Reade 2016): the forecast converges on the result, not on a close | None | 0.7-3% | **Never** for this purpose: no close to trade to, and a hedged position is a bet on news |
| 18 | Cycling (11) | 1 / 1 / 1,832 / 1,832 / 0.0 | Race week | LPT | Yes | Start lists | (c) | None | 3-7% | **Park:** one market in a month |
| 19 | Darts (3503) | 24 / 15 / 872 / 18 / 0.2 (OUTRIGHT_WINNER GBP4,960 on 3 days; a BEAT_LUKE_LITTLER special GBP6,951 once) | Tournament days | LPT | Yes | None of note | (c) nothing found | Historic Data Darts BASIC (in `PLAN_SCOPE`) | 0.7-1% | **Park:** GBP872 a day ahead in October; census pending for the World Championship (Dec-Jan), when it may differ |
| 20 | Special Bets (10) | 18 / 2 / 617 / 291 / 0.0 (max day GBP139,824) | Varies | No scheduled close | Varies | News | Does not repeat (`all_markets_priorities.md` s.2) | None | n/a | **Never:** one-off markets, no close, no history |
| 21 | Gaelic Games (2152880) | 1 / 2 / 567 / 105 / 0.0 | Match week | LPT | Yes | Team lists | (c) | None | 0.7-1% | **Park:** season over; thin |
| 22 | Australian Rules (61420) | 12 / 8 / 473 / 0 / n/a | Match week | LPT | Yes (Australian markets are the other exception to the at-the-off rule: unverified) | Team lists | (c) | None | 0.7-1% | **Park:** off-season in October; thin |
| 23 | Esports (27454571) | 14 / 9 / 297 / 0 / n/a | A day ahead, thin | LPT | Yes | Roster changes | (c) on Betfair; bookmaker and parimutuel studies find both a favourite-longshot bias and its reverse by game (Sweeney et al. 2021; Dagaev & Stoyan 2019; JIFS 2024) | None | 0.7-1% | **Park:** GBP297 a day ahead |
| 24 | Volleyball (998917) | 11 / 1 / 0 / 0 / n/a | | LPT | Yes | | (c) | None | | **Never:** no money |
| 25 | Athletics (3988) | not seen in Oct 2026 | Championship weeks | LPT | Yes | | (c) | None | | **Never:** seasonal, no markets in the census; census pending for existence |
| 26 | Winter Sports (451485) | not seen | Season Nov-Mar | LPT | Yes | | (c) | None | | **Never:** no markets in the census; census pending |
| 27 | Handball (468328) | not seen | | LPT | Yes | | (c) | None | | **Never:** no markets in the census |
| 28 | Netball (606611) | not seen | | LPT | Yes | | (c) | None | | **Never:** no markets in the census |
| 29 | Badminton (627555) | not seen | | LPT | Yes | | (c) | None | | **Never:** no markets in the census |
| 30 | Bowls (998918) | not seen | | LPT | Yes | | (c) | None | | **Never:** no markets in the census |
| 31 | Chess (136332) | not seen | | LPT | Yes | | (c) | None | | **Never:** no markets in the census |
| 32 | Pool (72382) | not seen | | LPT | Yes | | (c) | None | | **Never:** no markets in the census |
| 33 | Poker (315220) | not seen | | | | | | None | | **Never:** not an exchange sport in practice (unverified) |
| 34 | Financial Bets (6231) | not seen | | No sporting close | | | | None | | **Never:** no markets in the census; the type exists in the API (jBetfair) but carried nothing in October |
| 35 | Current Affairs (27388198) | not seen | | No scheduled close | | News | | None | | **Never:** as Politics and Specials |

## 2. By sport: the market types

Each table lists the market types the census saw for the sport (C2: markets a day, the money matched on the type
within the sport's top 200 a day ahead, both medians over the days the type appeared; max day), with the criteria that
differ by type and a verdict for every row. Types with a blank matched column never reached the top 200. Where the
census printed only a sport's best types (the cross-sport top 60), the rest are census pending.

### 2.1 Horse Racing (7)

The census's day-ahead read misses the racing money (it arrives on the day: A2); the types are from the repository's
own racing census (ledger `market-census-1003`, 3 Oct: 44 WIN, 43 PLACE, 65 OTHER_PLACE, 39 EACH_WAY, 2 ANTEPOST_WIN
on one card) and C2.

| Market type | Markets a day (C2) | A day ahead, GBP (C2) | On the day | Close | Drift evidence | Cost | Verdict |
|---|---|---|---|---|---|---|---|
| WIN (GB/IE) | 177 | 22,667 (all WIN markets together) | GBP1,423 a market at 08:00, GBP26,368 at T-15, GBP83,454 last (A2); UK GBP11,821 a runner pre-off (`other_markets_and_stakes_1002.md`) | BSP | (b) proven, live; (a) 8-21 band drifts out 2-5% (A3) | nil at the SP exit; 2.6-4.2% a tick at 8-25 | **Trade now** (live); **test next** the 8-21 lay with the model |
| WIN (AU, FR, USA, RSA, UAE) | in the 177 | | AU GBP1,180 a runner pre-play; FR GBP256, USA GBP300, RSA GBP615 (`other_markets_and_stakes_1002.md` s.1) | BSP | (c) untested; the files' morning window unusable for AU | 2-4% | **Park:** no model, overnight (AU), thin (the rest) |
| PLACE (To Be Placed) | 43 on a card | census pending (not in the top 60) | GBP35 a market at 08:00, GBP2,137 at T-15, GBP9,310 last; relative spread 1.238 at 08:00, 0.098 at T-15 (A2); UK GBP1,825 a runner | place BSP | (c) closed: the 08:00 mid sits above the place SP in every band (A3); morning back a quarter to a third under the SP, lay 1.5-2x it (ledger `place-clv-forward-1002`) | 0.7-2% a tick at 1.3-5 | **Park (closed 3 Oct):** the morning book is a spread, not a price |
| OTHER_PLACE (2/3/4 TBP) | 65 on a card | census pending | recorded from 6 Oct (`books_other_place.csv.gz`, 27 days: C1); the daily scan finds arbitrage of GBP0.35-0.58 a horse-poll (TRADING.md, 15-23 Oct) | no SP: LPT | (c) untested; the scan's flagged backs negative at the BSP (TRADING.md, 21 Oct) | 1-3% | **Park:** no SP exit, thin, the scan finds pence |
| EACH_WAY | 39 on a card | census pending | recorded from 6 Oct | no SP: LPT | (c) as above | 2-4% | **Park** |
| ANTEPOST_WIN | 2 (6 days) | 16,678; max 182,765 | thin daily | the race's BSP weeks later; all-in run or not (Betfair ante-post rules: unverified) | (b) the model prices big fields weeks ahead | 3-7% at long prices | **Research later:** few bets, non-runner risk, the one racing market with money a day ahead |
| FORECAST, REVERSE_FC, TRICAST, match bets | none on GB/IE racing (ledger `market-census-1003`; `tote_programme.md` line 36) | | | | | | **Never:** not offered |

### 2.2 Greyhound Racing (4339)

Types from the owner's Historic Data bundle (GB: WIN, PLACE, FORECAST; AU: WIN, PLACE, MATCH_BET, EXACTA, QUINELLA,
MONEY_BACK_2ND: `all_markets_priorities.md` s.4) and section D's first -> close by type; the census saw GBP109 a day
ahead across the sport (C2: nothing before the day).

| Market type | Markets (bundle, Jan-Sep 2026) | First traded -> close, median log move by band (D) | Close | Drift evidence | Cost | Verdict |
|---|---|---|---|---|---|---|
| WIN (GB) | 35,306 | +0.025 (3-5) to +0.129 (13-21); 51+ -0.262 (first -> BSP) | BSP | (a) measured drift out at 2-51; (b) form picks the shorteners | nil at the SP exit; 1.4-2.6% a tick at 3.5-8 | **Test next:** the pre-registered lay at the first two-sided book (`markets.md` 6.2) |
| WIN (AU) | 34,394 | +0.034 (2-3) to +0.142 (21-51) first -> BSP | BSP | (a) the same shape as GB (D) | as GB | **Park** behind GB: overnight UK time, no recorder there yet; the same test runs on the Historic Data table alone |
| WIN (NZ, IE) | 2,996 / 87 | NZ +0.013 to +0.270 (D) | BSP | (a) as above, thin | | **Park:** thin |
| PLACE (GB, AU, NZ) | 35,306 / 34,203 / 2,963 | median 0.000 in every band; mean +0.016 to +0.047; shortened 20-40% (D) | place BSP | (c) the median first price does not move; the price files' place pre-play WAP is empty (`greyhound_model.md` s.4) | 0.7-2% | **Park:** no measured drift, thin |
| FORECAST (GB) | 35,385 | median 0.000 at 8+; +0.027 to +0.068 at 2-8; the 1.0-1.5 band +2.978 (a first trade at 1.12 on a forecast is a mistake that is corrected) (D) | LPT | (c) no systematic move; combination markets, no SP | 3-7% at 10-50 | **Never** for this trade: no SP, combination prices, the first trade is noise |
| MATCH_BET (AU) | 34,221 | 0.000 at 1-5; -0.123 at 5-8 (113 runners) (D) | LPT | (c) nothing at the median | 0.7-1% | **Park:** AU only, thin, no SP |
| EXACTA, QUINELLA (AU) | 16,777 / 16,768 | median 0.000 in every band; shortened 1-37% (D) | LPT | (c) the first price rarely moves at all: these markets barely trade | 3-7% | **Never:** combination markets, no SP, no trade |
| MONEY_BACK_2ND (AU) | 3,296 | median 0.000 (D) | LPT | (c) | | **Never:** a promotional market |

### 2.3 Soccer (1)

C2 lists every soccer type with five or more markets a day (the full list is in the run's log; this table keeps every
type that reached the top 200 and groups the rest). 6,106 markets a day; 185 matches a day priced across the types.

| Market type | Markets a day | A day ahead, GBP | Max day | Close | Events | Drift evidence | Cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| MATCH_ODDS | 185 | 437,775 | 1,624,883 | LPT | line-ups | (c) market alone: makers +0.6%, takers -2.5% (Whelan 2025); (b) team news (`football_model.md`) | 0.8-1.4% at 2.5-3.5 | **Data first:** the Soccer plan, then the first -> close read (`markets.md` 6.3) |
| ASIAN_HANDICAP | 185 | 175,058 | 532,360 | LPT | line-ups | as match odds; the Asian-handicap benchmark in the gold tables (`football_model.md` s.1) | 0.7% at 1.9-2.0 | **Data first, with MATCH_ODDS:** 40% of its money, two-way, a tick under 1% |
| ALT_TOTAL_GOALS | 185 | 50,013 | 124,591 | LPT | | (a) total-goals makers +2.0%, takers -2.2% on 2.09m bets (Whelan 2025): a maker return, not a drift | 0.7% | **Data first, third:** read the maker return at T-60 in the same query |
| OVER_UNDER_25 | 184 | 34,173 | 75,356 | LPT | | as ALT_TOTAL_GOALS; the O/U benchmark held | 0.7% | **Data first, with the above** |
| CORRECT_SCORE | 185 | 15,789 | 36,376 | LPT | | (c) untested; many selections, long prices | 3-7% | **Park:** a tick is 3-7% of the price |
| BOTH_TEAMS_TO_SCORE | 185 | 12,697 | 52,915 | LPT | | (c) | 0.7-1% | **Park:** thin; a derivative of the goals markets |
| OVER_UNDER_35 / 15 / 05 / 45 / 65 / 55 / 75 / 85 | 185 each | 8,761 / 8,570 / 7,060 / 4,044 / 3,258 / 2,278 / 2,178 / 357 | 93,581 at 3.5 | LPT | | as O/U 2.5, thinner | 0.7-3% (the tails are long prices) | **Park:** O/U 2.5 and ALT_TOTAL_GOALS carry the money |
| DOUBLE_CHANCE, DRAW_NO_BET, HALF_TIME, HALF_TIME_FULL_TIME, HALF_TIME_SCORE, ODD_OR_EVEN | 185 each | 4,086 / 2,628 / 1,029 / 1,372 / 191 / 254 | | LPT | | (c) derivatives of match odds | 0.7-7% | **Never** for this trade: priced off match odds, thinner, no own information |
| FIRST_HALF_GOALS_05 / 15 / 25 / 35 | 185 / 185 / 185 / 10 | 2,031 / 2,038 / 1,180 / 576 | 31,796 | LPT | | (c) | 1-3% | **Park:** thin |
| TEAM_A_1 / 2 / 3, TEAM_B_1 / 2 / 3, TEAM_A/B_WIN_TO_NIL, TEAM_A/B_OVER_UNDER_05 / 15 / 25 | 164-177 each | 333-1,652 | 9,420 | LPT | | (c) | 1-3% | **Never** for this trade: GBP300-1,700 a day across the whole type |
| MATCH_ODDS_AND_BTTS, MATCH_ODDS_AND_OU_25 / 35 | 164 / 164 / 8 | 1,443 / 1,024 / 782 | 29,430 | LPT | | (c) combinations | 1-3% | **Never:** combinations of thin markets |
| SHOTS_ON_TARGET_P1, CORNER_ODDS, OVER_UNDER_105_CORNR, OVER_UNDER_85_CORNR, TO_SCORE, PENALTY_TAKEN, FIRST_GOAL_SCORER, TO_SCORE_2_OR_MORE, TO_SCORE_HATTRICK, BOOKING_ODDS, OVER_UNDER_25/35_CARDS, SENDING_OFF, SHOWN_A_CARD, CORRECT_SCORE2 | 5-10 each (a few matches a day) | 2,840 / 1,350 / 1,245 / blank / 1,043 / 1,558 / blank / blank / blank / blank / blank / blank / blank / blank | 12,973 | LPT | line-ups (player props) | (c) untested; `football_model.md` s.5 named corners and cards as thin enough for a stats model to beat, which is an outcome edge, not a drift | 1-7% | **Park:** player and corner props exist on 5-10 matches a day and most never reach the top 200 |
| SPECIAL (2 days) | 2 | 327,424 | 646,448 | varies | | | | **Never:** one-off |
| TO_QUALIFY (4 days) | 10 | 5,134 | 9,525 | LPT | | (c) | 0.7-1% | **Park:** cup weeks only |
| Outrights (WINNER, TOP_N_FINISH, RELEGATION: not in C2's printed lists) | census pending | census pending | | end of season | | (c) | 3-7% | **Park:** season-long markets have no close before the result |

### 2.4 Cricket (4)

| Market type | Markets a day | A day ahead, GBP | Max day | Close | Events | Drift evidence | Cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| MATCH_ODDS | 17 | 1,669,718 | 88,088,108 | LPT at the start | the toss and teams about 30 min before (unverified timing) | (c) pre-match untested; in play over-reaction in the first innings (Norton, Gray & Faff 2015) | 0.7-0.8% at 1.5-2.5 | **Data first:** the Cricket plan; the first -> close and T-60 -> close read by format (`markets.md` 6.4) |
| A_1ST_INN_RUNS_LINE (5 days), B_1ST_INN_RUNS_LINE (7), A/B_2ND_INN_RUNS_LINE, TEAMA_1ST/2ND_INN_RUNS, A_1_INN_10_OVR_LINE, B_1_INN_5_OVR_LINE | 1 each | 601,079 / 3,888 / 38,754 / 10,901 / 7,457 / 3,716 / 8,973 / 5,778 | 1,045,539 | LPT; line markets | the toss decides who bats first: the market exists only after it (unverified) | (c) | line markets, one tick a run | **Never** for a pre-off trade: the market forms at the toss |
| DRAW_NO_BET (17 days) | 3 | 67,316 | 145,935 | LPT | the toss | (c) | 0.7% | **Park:** Tests only, a derivative of match odds |
| TIED_MATCH | 17 | 2,557 | 20,715 | LPT | | (c) | long prices | **Never:** a long-shot special |
| TO_WIN_THE_TOSS (23 days) | 5 | 335 | 7,517,656 | settles at the toss | | a coin | 0.7% | **Never:** a coin toss; the max day is one match's crowd |
| COMPLETED_MATCH, MATCH_ODDS_LO_TIE | 14 / 6 | 268 / 9 | | LPT | weather | (c) | | **Never:** thin |
| 1_INNING_10/15/20_OVR_LINE, 2_INNING_10/15/20_OVR_LINE, 1ST_INNINGS_RUNS | 5-6 each | 0 | 2,992 | in-play lines | | | | **Never:** in-play markets |

### 2.5 Golf (3)

| Market type | Markets a day | A day ahead, GBP | Max day | Close | Events | Drift evidence | Cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| WINNER | 3 | 506,658 | 1,242,569 | LPT at the Thursday start; in play four days | field, withdrawals, tee times, weather | (b) Pinnacle leads the move (datagolf); Betfair untested | 3.4-7.1% at 15-100 | **Research later / data first:** the Golf plan; the Monday -> Thursday move by price band |
| TOP_10_FINISH (13 days), TOP_5_FINISH, TOP_20_FINISH | 1 each | 44,235 / 27,513 / 26,863 | 64,067 | LPT | as WINNER | (c) | 1.4-3% | **Park:** derivatives of WINNER |
| EACH_WAY (14 days) | 1 | 6,304 | 10,313 | LPT | | (c) | 3-7% | **Park:** thin |
| ROUND_LEADER (5 days) | 1 | 10,243 | 14,871 | LPT at the round's start | tee times, weather | (c) | 3-7% | **Park:** thin |
| TOUR_MATCHBET_NTIE1 (13 days), TOUR_MATCHBET_NTIE3 (8) | 1-2 | 3,550 / 2,960 | 5,397 | LPT | | (b) Pinnacle vs Betcris convergence (datagolf) | 0.7-1% | **Park:** GBP3k a day across the type |
| Others (TOP_NATIONALITY, MAKE_THE_CUT, HOLE_IN_ONE, three-balls) | census pending | | | | | | | **Park / never:** not in the top 200 in October |

### 2.6 American Football (6423)

| Market type | Markets a day | A day ahead, GBP | Max day | Close | Events | Drift evidence | Cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| HANDICAP | 3 | 143,448 | 905,078 | LPT | injury reports (Wed-Fri), weather | (c) NFL open vs close no significant accuracy difference 2002-11 (arXiv 1211.4000) | 0.7% at 1.9 | **Park:** three markets a day (Sundays are many at once); the move is made in US books |
| COMBINED_TOTAL | 3 | 54,122 | 426,137 | LPT | weather | (c) | 0.7% | **Park** |
| MATCH_ODDS | 3 | 37,564 | 239,724 | LPT | | (c) | 0.7-1% | **Park** |
| Others (SUPER_BOWL outright, player props) | census pending | | | | | | | **Park / never** |

### 2.7 Tennis (2)

| Market type | Markets a day | A day ahead, GBP | Max day | Close | Events | Drift evidence | Cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| MATCH_ODDS | 105 | 107,574 | 694,887 | LPT | withdrawals, scheduling | (a) exchange FLB (Abinzano et al. 2016); drift untested; the Italian exchange's shorteners over-shoot (theoddslab, one month, pooled with football) | 0.7-0.8% at 1.5-2.5 | **Park:** median market GBP5; re-read the `busiest` matches (census pending) |
| HANDICAP, COMBINED_TOTAL, SET_CORRECT_SCORE, SET_BETTING, NUMBER_OF_SETS | 23 / 23 / 23 / 88 / 23 | 2,651 / 1,894 / 1,439 / 708 / 31 | 15,948 | LPT | | (c) | 1-3% | **Never** for this trade: GBP30-2,700 a day across each type |
| SET_WINNER (176), PLAYER_A/B_WIN_A_SET (23 each), GAME_BY_GAME_xx_yy (61 each, 30 types) | | 145 / 23 / 5 / 0 | 1,692,103 / 3,794,759 / 4,616,153 (single days: in-play crowds on one match) | in play | | in-play markets | | **Never:** in-play markets; the max days are one match's in-play volume |
| TOURNAMENT_WINNER (7 days) | 4 | 8,588 | 15,762 | end of the tournament | draws | (c) | 3-7% | **Park:** a week-long market, long prices |

### 2.8 Basketball (7522), Ice Hockey (7524), Baseball (7511), Mixed Martial Arts (26420387)

| Sport | Market type | Markets a day | A day ahead, GBP | Max day | Close | Drift evidence | Cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| Basketball | MATCH_ODDS / COMBINED_TOTAL / HANDICAP | 42 / 42 / 42 | 34,533 / 31,603 / 26,065 | 429,385 | LPT | (b) NBA line moves informed (Gandar et al. 1998); Betfair thin | 0.7% | **Park:** about GBP800 a market a day ahead; TOURNAMENT_WINNER once (GBP25,226) |
| Ice Hockey | MONEY_LINE / COMBINED_TOTAL / HANDICAP / RT_MATCH_ODDS | 20 each | 18,817 / 5,130 / 3,771 / 1,621 | 97,345 | LPT | (c) | 0.7-1% | **Park:** under GBP1,000 a market |
| Baseball | OUTRIGHT_WINNER (5 days) / MATCH_ODDS / COMBINED_TOTAL / HANDICAP | 1 / 2 / 2 / 2 | 52,891 / 7,550 / 5,811 / 3,096 | 65,239 | LPT | (c); starting pitchers (unverified) | 0.7-1% | **Park:** October is the play-offs; census pending for a regular-season read |
| Mixed Martial Arts | MATCH_ODDS / METHOD_OF_VICTORY / UNDIFFERENTIATED / ROUND_BETTING | 7 / 9 / 9 / 9 | 40,866 / 709 / 12 / 0 | 141,380 | LPT | (c) | 0.7-1% | **Park:** fight nights only; the props never trade |

### 2.9 Motor Sport (8), Rugby League (1477), Rugby Union (5), Boxing (6), Snooker (6422), Darts (3503)

| Sport | Market type | Markets a day | A day ahead, GBP | Max day | Close | Drift evidence | Cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| Motor Sport | WINNER (5 days); others census pending | 1 | 14,252 | 65,910 | LPT at lights out | (c); qualifying is the information (unverified) | 3-7% | **Park** |
| Rugby League | MATCH_ODDS (10 days) / OUTRIGHT_WINNER (2) | 3 / 1 | 4,033 / 2,918 | 13,939 | LPT | (c) | 0.7-1% | **Park:** thin, season's end |
| Rugby Union | MATCH_ODDS (13 days); handicaps census pending | 4 | 3,175 | 17,196 | LPT | (c) | 0.7-1% | **Park:** thin |
| Boxing | MATCH_ODDS (8 days); method and rounds census pending | 3 | 6,383 | 130,108 | LPT | (c) | 0.7-1% | **Park:** event-driven |
| Snooker | MATCH_ODDS and the rest: census pending (sport total GBP3,344, median market GBP131) | 8 | | 6,597 | LPT | (c) nothing found | 0.7-1% | **Park** |
| Darts | OUTRIGHT_WINNER (3 days) / BEAT_LUKE_LITTLER (1 day) / MATCH_ODDS census pending | 1 / 1 | 4,960 / 6,951 | 31,393 | LPT | (c) nothing found | 0.7-1% | **Park:** GBP872 a day in October; the December-January World Championship is census pending |

### 2.10 Politics (2378961), Special Bets (10), Current Affairs (27388198), Financial Bets (6231)

| Sport | Market type | Markets a day | A day ahead, GBP | Max day | Close | Verdict |
|---|---|---|---|---|---|---|
| Politics | NONSPORT (6 days) | 1 | 7,490 | 205,093 | none scheduled; the result | **Never** for this purpose: GBP317m traded on the 2024 US election (oddschecker) shows depth, but a hedged open-to-close trade needs a close, and the move between is news. Vaughan Williams & Reade (2016): the market's forecast improves towards the vote, i.e. it converges on the result |
| Special Bets | varies (18 days) | 2 | 617 | 139,824 | none scheduled | **Never:** one-off markets, no history, no close |
| Current Affairs | not seen in October | | | | none | **Never** |
| Financial Bets | not seen in October | | | | a fixing | **Never:** the type exists in the API; nothing listed in October |

### 2.11 Esports (27454571), Volleyball (998917), Cycling (11), Gaelic Games (2152880), Australian Rules (61420)

| Sport | Market types | Markets a day | A day ahead, GBP | Close | Drift evidence | Verdict |
|---|---|---|---|---|---|---|
| Esports | census pending (sport total GBP297; 9 markets a day; 14 days) | 9 | 297 | LPT | bookmaker and parimutuel studies find FLB and its reverse by game (Sweeney et al. 2021; Dagaev & Stoyan 2019; JIFS 2024); nothing on Betfair | **Park:** no money |
| Volleyball | census pending (total GBP0) | 1 | 0 | LPT | (c) | **Never** |
| Cycling | census pending (1 day, GBP1,832) | 1 | 1,832 | LPT | (c) | **Park:** one market in the month |
| Gaelic Games | census pending (1 day, GBP567) | 2 | 567 | LPT | (c) | **Park:** out of season |
| Australian Rules | census pending (12 days, GBP473) | 8 | 473 | LPT | (c) | **Park:** out of season (Grand Final late September) |

### 2.12 The eleven event types with no markets in October 2026

Athletics (3988), Winter Sports (451485), Handball (468328), Netball (606611), Badminton (627555), Bowls (998918),
Chess (136332), Pool (72382), Poker (315220), Financial Bets (6231), Current Affairs (27388198): listed by the API
(section 5, the three client libraries), none seen by the census on any of 25 days. Verdict for each: **never** for
this purpose until a census shows markets; the seasonal ones (Athletics, Winter Sports) are census pending for their
seasons, and a complete dump in January or July would settle them. None has a BSP, a model, a data route held or any
literature found on exchange pre-off moves.

## 3. What the evidence says across sports (the criteria applied)

1. **Pre-off money a day ahead (C2).** Four sports have it in size: cricket GBP6.8m, football GBP0.9m, golf GBP0.65m,
   American football GBP0.29m on their top 200 markets. Racing has it on the day instead (A2). Every other sport's
   median market a day ahead is under GBP15, and tennis's and basketball's concentration (116, 115) says the money
   sits in a handful of events.
2. **When a tradable open exists.** Racing: the morning of the day (06:15 UK; the evening before unformed, ledger
   `evening-before-1007`); greyhounds: on the day, late (a price 24 h out on 78 of 451,031 runners, D). Football,
   cricket, golf, the NFL: a day or more ahead, with the last hour adding 55-150% in football (betaminic). The rest:
   the open is thin whenever it comes.
3. **A single-price close.** Only the two racing sports have the BSP; the live rule's whole 6% survives because the
   exit costs nothing (`markets.md` section 2). Elsewhere the close is the last pre-play trade and the exit is a tick
   (0.7-1% at 1.5-3, 3-7% at outright prices) or a resting order.
4. **In play.** Betfair aims to turn markets in play at the off except soccer and Australian markets (Exchange Rules
   Part B 1.2 as copied in a forum post: unverified). A pre-off close is therefore well defined for every scheduled
   event; it is not defined for politics, specials and season-long outrights, which is why those are "never".
5. **Information events before the off.** Racing: non-runners, going, the market's own late money (80% of pre-off
   volume in the last 20 minutes: Brown & Yang 2017). Football: line-ups about an hour before. Cricket: the toss.
   Golf: the field and the weather. US sports: injury reports and the US books' lines. In each the event is the move;
   a hedged trade either closes before it or forecasts it.
6. **Drift predictability.** (a) from the market alone: measured only in racing and greyhounds (A3, D, B4), the
   favourite-longshot bias unwinding towards the off, 2-5% in horse 8-21 (half the spread) and 2.5-13% in greyhound
   3-51 (the lay side unmeasured). (b) with a model or information: proven in UK/IE horse win (TRADING.md); plausible
   in football (team news), golf and US sports (sharp books lead); greyhound form picks the shorteners. (c) nothing
   found for tennis pre-match, cricket pre-match, darts, snooker, MMA, boxing, esports, the rest. Flepp, Nuesch &
   Franck (2017): where exchange liquidity is low (under about GBP23,400 cumulative volume on a soccer match) the
   bookmaker's price is the better one, so a thin exchange market has no sharp close to trade towards.
7. **Data route.** Held: racing and greyhound price files, the recorder (any event type by `--event-type`), the 2026
   greyhound bundle, football-data.co.uk. Obtainable free: Betfair Historic Data BASIC per sport (first trade, T-24h,
   T-60, T-15, T-1, close, result: `betfair_historic.py`), once the owner "buys" each sport. Nothing paid is needed
   for any test proposed (`reports/open_close/data.md` section 6).
8. **Round-trip cost.** `stake x (B/L - 1)` less 2% on the market's net profit; one tick at 1.5-3 is 0.7-1.4%, at 5-8
   2-2.6%, at 15 3.4%, at 25 4.2%, at 40 5.3%, at 75+ 7.1% (`markets.md` 5.2); the Expert Fee at 20% of weekly gross
   profit above GBP25,000 over 52 active weeks applies to the account as a whole (`markets.md` 5.1).

## 4. What is pending, and what changes when it lands

- The complete census dump (`census_all_markets_1101.py`): every sport x market type row with days seen, markets a
  day, matched medians and the busiest events; the on-the-day money profile for racing. It fills the "census
  pending" cells (horse PLACE/OTHER_PLACE/EACH_WAY a day ahead; golf, NFL, motor sport, rugby, boxing, snooker, darts
  and esports type lists; the busiest tennis and basketball events) and may add event types the API lists that
  October's census did not reach.
- Seasonal re-reads: darts (December-January), winter sports, athletics, baseball's regular season, Australian
  rules, Gaelic games. A verdict of "never" on a seasonal sport is a verdict on October.
- None of the pending cells can move a sport into "trade now": that needs a measured drift and a close, which only
  racing has.

## 5. Sources

Repository:
- `reports/open_close/markets.md` (the ranking, the drift sections 4.1-4.5, costs 5.1-5.3, tests 6.1-6.5);
  `reports/open_close/data.md` (section 6, the data to obtain next; licences 6.5)
- research-query run 41291139483, job 124802487010 (`research/queries/done/open_close_inventory_1101.py`): sections
  A2, A3, B4, C1, C2, D as cited
- `TRADING.md`; `reports/other_markets_and_stakes_1002.md` (section 1, liquidity by jurisdiction; section 3, the racing
  census of 3 Oct); `reports/all_markets_priorities.md` (sections 2, 4, 5); `reports/greyhound_model.md` (sections
  4-5); `reports/football_model.md` (sections 3, 5); `reports/tote_programme.md` (line 36); `reports/research_ledger.jsonl`
  (`staking-1031`, `evening-before-1007`, `edge-inplay-offsets-1002`, `place-clv-forward-1002`, `market-census-1003`)
- `betfair_recorder.py` (lines 81-82: event type ids 7 and 4339; 402-438: `census_sports`); `betfair_historic.py`
  (lines 1-18; `PLAN_SCOPE`)

Betfair's event types (no official table is published; the API returns them):
- Betfair developer docs, listEventTypes: https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687448/listEventTypes ;
  Navigation Data For Applications (menu.json, EVENT_TYPE nodes): https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687802/Navigation+Data+For+Applications ;
  Betting Type Definitions (marketTypeCodes, listMarketTypes): https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687465/Betting+Type+Definitions
- jBetfair `EventTypes.java` (31 ids, fetched 1 Nov 2026): https://raw.githubusercontent.com/aaronSig/jBetfair/ea2a0635ed39f053f8955c0a27f318c1c9418258/src/main/java/com/jbetfair/api/entities/EventTypes.java
- betfairlightweight tutorial, `listEventTypes` output (29 ids): https://raw.githubusercontent.com/betfair-datascientists/betfair-datascientists.github.io/55719317474fb41344b50e9f1aa8252976992b06/src/api/apiPythontutorial.md
- less-stupid-betfair `BetfairExchangeTypes.scala` (26 ids): https://raw.githubusercontent.com/thinkmorestupidless/less-stupid-betfair/6f29726d046c3612677217064aa8ee6308eafc2b/lib/src/main/scala/com/thinkmorestupidless/betfair/exchange/domain/BetfairExchangeTypes.scala
- Exchange Rules Part B 1.2 (in-play at the off; soccer and Australian exceptions) as copied in a forum post, unverified: https://www.sportsbookreview.com/forum/sportsbooks-industry/1897775-betfair-the-tennis-match-a-kalinskaya-vs-s-rogers

Literature added for this note (the rest is in `markets.md` section 8):
- Gandar, J. M., Dare, W. H., Brown, C. R. & Zuber, R. A. (1998) Informed traders and price variations in the betting market for professional basketball games. Journal of Finance (abstract via aggregator): https://academicnewsletter.sufe.edu.cn/info/361852
- Szalkowski & Nelson (2012) The performance of betting lines for predicting the outcome of NFL games. arXiv 1211.4000: https://arxiv.org/pdf/1211.4000
- Norton, H., Gray, S. & Faff, R. (2015) Yes, one-day international cricket "in-play" trading strategies can be profitable! Journal of Banking & Finance 61(S2), 164-176: https://ideas.repec.org/a/eee/jbfina/v61y2015is2ps164-s176.html
- Flepp, R., Nuesch, S. & Franck, E. (2017) The liquidity advantage of the quote-driven market: evidence from the betting industry: https://unifr.ch/tim/en/assets/public/uploads/Publication%20list/2017/Flepp_N%C3%BCesch_Franck_2017.pdf
- Vaughan Williams, L. & Reade, J. J. (2016) Forecasting elections. Journal of Forecasting 35(4), 308-328: https://irep.ntu.ac.uk/id/eprint/29653
- Sweeney, K., Tuttle, M. H. & Berg, M. D. (2021) esports favourite-longshot bias (Games and Culture; via search); Dagaev, D. & Stoyan, E. (2019) HSE WP 216/EC: https://ideas.repec.org/p/hig/wpaper/216-ec-2019.html ; JIFS 46(3) 2024 esports biases: https://journals.sagepub.com/doi/10.3233/JIFS-232932
- Player absence and betting lines in the NBA (2015), Finance Research Letters 13, 130-136: https://ideas.repec.org/a/eee/finlet/v13y2015icp130-136.html
- datagolf, How sharp are bookmakers? (golf): https://datagolf.com/how-sharp-are-bookmakers
- betaminic, football liquidity on Betfair by league (Jul-Sep 2025): https://www.betaminic.com/betting-research/which-football-leagues-have-the-most-liquidity-on-betfair/
- theoddslab, tennis pre-match on the Italian exchange (Jul-Oct 2026): https://theoddslab.com/articoli/tennis-trading-betfair/
- oddschecker, 2024 US election totals on Betfair: https://www.oddschecker.com/insight/industry/20241106-over-300-million-matched-on-2024-us-election-markets-on-betfair-exchange
- Whelan, K. (2025) Agreeing to disagree: the economics of betting exchanges. MPRA 126351: https://mpra.ub.uni-muenchen.de/126351/1/MPRA_paper_126351.pdf
- Brown, A. & Yang, F. (2017) The role of speculative trade in market efficiency. Review of Finance 21(2): https://ueaeprints.uea.ac.uk/58626/1/Speculative_Trade_Revisedv5.pdf
