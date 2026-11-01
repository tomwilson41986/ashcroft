# Open-to-close trading on Betfair: which markets (research agent, 1 Nov 2026)

The owner's ask (1 Nov 2026): autonomous trading on Betfair, in markets where the price can be traded from its
opening to its close, hedged so the result does not matter. This note ranks Betfair's sports and market types for
that purpose. It builds on what the repository already knows and cites it; it does not repeat it. Every figure
traces to a source in section 8 or to a repository file. Where a claim rests on general market knowledge, it says
"unverified".

## The result in seven lines

1. **The UK/IE horse win market is the one market where this trade is proven, and it is already live.** The win rule is
   an open-to-close trade: back in the morning where the model's BSP forecast is well under the price, lay at the
   BSP. GBP47,274 staked 19-31 Oct at +6.15% CLV (research ledger `staking-1031`, line 525; TRADING.md). The edge is the
   model's, not the market's: with no model there is no drift to catch on the back side (ledger `market-only-drift-1002`).
2. **Racing and greyhounds are structurally better for a hedged trade than any other sport**, because the close is a
   single price with no spread: the Betfair SP. Everywhere else the exit must cross the spread or rest an order and
   hope, and the "close" is the last pre-play trade (`betfair_historic.py`, lines 1-4).
3. **UK greyhounds are next, and the inventory sharpened why.** On 1,863,552 runners (Betfair Historic Data, Jan-Sep
   2026) the GB first traded price drifts out to the BSP by a median 2.5% at 3-5 up to 13% at 13-21 and 12% at 21-51,
   shortening on only 33-45% of dogs; the price files say the same of the morning WAP since 2012 (research-query run
   41291139483, sections D and B4). That is a market-only drift of class (a), in the direction a lay-early trade
   needs. What is not known is whether the lay side is near that first price: early money is thin (the census sees
   GBP109 a day ahead across the top 200 greyhound markets) and the first trade is often the first bet of an
   unformed book. The test on the recorded books (section 6.2) decides it.
4. **Football is the deepest non-racing market a day ahead and at the off** (the census: GBP437,775 matched a day on
   match odds within the top 200, GBP175,058 on Asian handicap; betaminic: the Premier League about GBP0.4m at T-60
   and GBP0.6m at T-1). Its pre-play close is an unbiased forecast and the maker side earns about zero after fees
   (Whelan 2025), so a market-only drift is unlikely. The open-to-close move there is information (team news) arriving
   after the open. Research next, not trade.
5. **Cricket and golf move up from "park" to "research next, data first": the census shows real pre-off money a day
   ahead** (cricket match odds a median GBP1.67m a day across about 17 markets; golf outrights GBP507k across 3). The
   move in each is an information event (the toss and teams; the field and the weather) rather than a drift, and
   neither has a model or a close benchmark held, so they sit behind football. **Tennis, darts, snooker, US sports,
   politics stay parked:** tennis's money a day ahead is GBP116k across its top 200 markets with a median market of
   GBP5 (concentration 116: a handful of matches carry it); politics has no close.
6. **The horse market's own drift, read on the recorder's books for the first time** (run 41291139483, section A3,
   13,992-18,982 runners, Oct 2026): from the 08:00 mid to the BSP, win runners at 8-13 and 13-21 drift out a median
   2.4% and 3.7%, and still 3.6% and 3.1% from the T-15 mid; favourites move about nothing from 10:00 on; long shots
   (51+) sit a median 48% above their BSP at 08:00 because the mid of a wide book is not a price (the relative spread
   is 0.189 at 08:00 and 0.049 at T-15, section A2). So the lay-early cell on long shots that the first draft of
   this note proposed does not exist (section 6.1 is rewritten); the 8-21 band's 2-5% drift is real but of the order of half the spread, so a market-only lay there
   is about break-even before commission. The model remains the edge.
7. **Costs:** a greened back-then-lay earns `stake x (B/L - 1)` before 2% commission on the market's net profit. Taken at
   both ends with no move it loses one tick: 1% of the stake at 2.0, 2% at 5.0, 3.3% at 15, 5% at 40, 6.7% at 75+
   (Betfair's ladder). The Expert Fee adds 20% of weekly gross profit (less commission paid) once the account's gross
   profit over its last 52 active weeks passes GBP25,000, 40% past GBP100,000. At the live rule's run rate (+GBP4,532
   in 22 days to 31 Oct, TRADING.md) the account reaches GBP25,000 in about four months (arithmetic, not a forecast).

## 1. The ranked table

Ranked for the owner's purpose: a hedged pre-off trade, no in-play reliance, automated. "Open" means the first time a
price can be taken in size; "close" means the BSP where one exists, else the last pre-play trade. Predictability:
(a) from the market alone, (b) only with a model or an information edge, (c) known to be unpredictable or untested.
"A day ahead" figures are the census's (research-query run 41291139483, section C2; 25 days, 7-31 Oct 2026): the
money matched by the evening read on a sport's 200 busiest markets starting in the next 24 hours, a median over days.
Rows marked (changed) were re-ranked by the census or the inventory; section 1.1 says why.

| # | Market | When a tradable open exists | Typical pre-off matched | Evidence the move is predictable | Expected size of the move | Data held or obtainable | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | UK/IE horse win | From about 06:15 UK on the day; the evening before is unformed (lay/back 21-909, nothing matched: ledger `evening-before-1007`). A day ahead the census sees GBP22,667 across all 177 win markets. At 08:00 a market holds a median GBP1,423 matched, best-back size GBP6, relative spread 0.189; at 12:00 GBP6,680; T-60 GBP13,493; T-15 GBP26,368 (spread 0.049); T-1 GBP77,294 (run 41291139483, A2) | UK GBP11,821 a runner, IE GBP4,743 (`other_markets_and_stakes_1002.md` s.1); 80.4% of pre-race volume in the last 20 min (Brown & Yang 2017, 2013-14 data); 68% of a market's pre-off money arrives after T-15 (A2) | (b) strong: the model carries about half the weight beside the morning price in forecasting the BSP (TRADING.md); features explain the morning-to-off move (ledger `iter3-screen-production-repaired`). (a) back side: none (favourites +0.5%, long shots -24% CLV: ledger `market-only-drift-1002`). (a) lay side: the 08:00 mid at 8-21 drifts out 2.4-3.7% (median log, A3), about half the spread; long shots' mids sit 48% above the BSP at 08:00 (the spread, not a drift) | +6.15% CLV on GBP47k staked, 19-31 Oct (ledger `staking-1031`); first backs +10.7% on 31 Oct (TRADING.md) | Price files 2017 on, live books from 1 Oct, `betfair_live_marks` at 08:00-12:00 UK and T-120..T-1 (CLAUDE.md) | **Trade now (live).** The market-only lay on long shots is closed on sight (section 6.1); the 8-21 lay is a model question, the live rule's mirror |
| 2 | UK greyhound win (changed: the drift is now measured) | The first trade is early but thin: the morning WAP prices 14.7% of runners, morning volume GBP0 to the 75th percentile (`greyhound_model.md` s.4); a day ahead the census sees GBP109 across the top 200 markets, i.e. nothing | GBP1,274 a runner (UK+AU files); pre-play GBP2,213 at the median (GB) | (a) **measured:** GB first traded -> BSP, median log move +0.056 at 2-3, +0.025 at 3-5, +0.050 at 5-8, +0.103 at 8-13, +0.129 at 13-21, +0.117 at 21-51 (n 7,882-56,703 a band), shortened 33-45%; 51+ the reverse (-0.262); from T-1 still +0.017 to +0.057 (run 41291139483, D). The price files' morning WAP -> BSP, GB 2012-26: +0.137 to +0.215 at 2-51 (B4). (b) form predicts which dogs shorten: CLV +4.5-7.6% at the first price, 2026 bundle; a morning price traded in volume lost 2024-26 (`greyhound_model.md` s.4) | First price to BSP: 2.5-13% drift out by band (D), against a tick of 2.5-5% at 10-20 | GBGB 2018 on, price files 2012 on (B4), the owner's Historic Data bundle 2026, live GB/IE books from 6 Oct (27 days, 76.7 MB: run 41291139483, C1) | **Research next, promoted to the first test to run.** The lay at the first two-sided book against the BSP, by band, on the recorded books (section 6.2) |
| 3 | Football match odds, top leagues | A day ahead: 185 match-odds markets, GBP437,775 matched on the type within the top 200 (C2); real depth in the last hour: EPL GBP381k at T-60, GBP592k at T-1 (betaminic, Jul-Sep 2025) | EPL GBP0.6m a match at T-1; La Liga GBP359k; Ligue 1 GBP149k (betaminic) | (c) for the market alone: the pre-play close is an unbiased forecast; takers -2.5%, makers +0.6% on 902,568 bets 2022-24 (Whelan 2025). The close beats the open by 0.0034 log-loss and a team model cannot see why (`football_model.md` s.3). Italian-exchange sample: moves over 3% over-shoot (p 0.03-0.04, one month, football+tennis pooled: theoddslab). (b) team news about an hour before kick-off | Close vs open: 0.0034 log-loss on 1X2 (`football_model.md`) | football-data.co.uk open/close 2012 on; Betfair Historic Data soccer once the account holds it; `betfair_market_map` | **Research next.** The market-only drift test on the BASIC files (section 6.3); team-news trade needs API-Football and a fast feed |
| 4 | Football Asian handicap, O/U 2.5, correct score | A day ahead: Asian handicap GBP175,058, alternative total goals GBP50,013, O/U 2.5 GBP34,173, correct score GBP15,789 on the type within the top 200 (C2) | As match odds | (a) maker side: total-goals makers +2%, takers -2.2% on 2.09m bets (Whelan 2025): a market-making return, not a drift. (b) as match odds | As match odds | As match odds | **Research next, behind 3.** Asian handicap first: it carries 40% of the match-odds money a day ahead |
| 5 | Cricket match odds (changed: park -> research next, data first) | A day ahead: about 17 match-odds markets carry a median GBP1,669,718, the deepest pre-off money in the census (max GBP88m on one day); the median of all 135 cricket markets is GBP0 (C2) | GBP201m on one IPL match's match odds in 2025 (Betfair newsletter); the toss market itself a median GBP335 a day ahead (C2) | (c) untested; the toss and the teams are an information event about 30 min before the start (general knowledge, unverified) | Unknown | Free BASIC Cricket plan once "bought"; nothing held | **Research next, data first:** the first -> close move by band and format on the BASIC files, then whether a hedged trade that avoids the toss window exists |
| 6 | Golf outrights (changed: park -> research later) | A week ahead; a day ahead 3 WINNER markets carry GBP506,658, TOP_10 GBP44,235, TOP_5 GBP27,513 (C2, 17 days) | Not held beyond the census | (b) Pinnacle's close is the sharpest; its opening price moves 54.7% of the way to Betcris's where they differ by 5%+ (datagolf, 106,204 matchups). Betfair: untested | Unknown on Betfair | Free BASIC Golf plan; nothing held | **Research later:** depth exists, few markets a week, no model, the "close" is the Thursday start |
| 7 | Australian horse win | Overnight UK time; the files' "morning" includes post-off trading, so only the pre-play price is usable (`other_markets_and_stakes_1002.md` s.4) | GBP1,180 a runner pre-play, 81 races a day | (c) untested in horses; the AU greyhound first -> BSP drift is +0.034 to +0.142 at 2-51 (D) | Unknown | Price files 2018 on (pre-play only); Punting Form needs a key | **Park** behind greyhounds: no model, overnight, thin early |
| 8 | UK/IE horse place, 2/3/4 TBP, each way | Morning place book: best back a quarter to a third under the place SP, best lay 1.5-2x it (ledger `place-clv-forward-1002`); relative spread 1.238 at 08:00, 0.098 at T-15; GBP35 matched a market at 08:00 (A2) | UK GBP1,825 a runner, IE GBP696; morning GBP17 | (c) closed: neither side of the morning book closes in our favour; the win rule's horses lose 12.5% there (ledger `place-clv-forward-1002`). The 08:00 mid sits above the place SP in every band (A3): the spread | n/a | Recorded books continue | **Park (closed 3 Oct)** |
| 9 | Tennis match odds | A day ahead: 105 match-odds markets, GBP107,574 matched on the type within the top 200; the median tennis market GBP5; concentration 116 (C2). Italian exchange: median EUR0 across 6,477 pre-match markets (theoddslab) | About 90% of money arrives in play (trading guide, unverified) | (a) favourite-longshot bias on exchange tennis, larger in lower tiers and later rounds (Abinzano et al. 2016, AEL). Open-to-close drift pre-match: untested | Unknown | Sackmann, tennis-data.co.uk, Betfair Historic Data once held | **Park** for pre-off: the money a day ahead is in a handful of matches |
| 10 | US sports (NFL, NBA, NHL, MLB) | A day ahead: NFL handicap GBP143,448 and match odds GBP37,564 on about 3 markets; NBA match odds GBP34,533 across 42 markets; NHL money line GBP18,817 across 20 (C2) | Not held beyond the census | (b) sharp-book line moves lead soft books (general knowledge, unverified) | Unknown | None | **Park:** NFL has depth on 3 markets a day, the rest is GBP1,000 a market |
| 11 | Darts, snooker, MMA, boxing | A day ahead: darts GBP872 across the top 200, snooker GBP3,344, MMA match odds GBP40,866 on 8 days, boxing GBP6,746 (C2) | Thin | Untested | Unknown | None | **Park** |
| 12 | Politics, specials | Months ahead; a day ahead GBP2,040 median (max GBP205,093) (C2) | GBP317m on the 2024 US election across markets, GBP250m on the winner (oddschecker, 6 Nov 2024); Racing Post reported over GBP400m | No close, news-driven, does not repeat (`all_markets_priorities.md` s.2) | n/a | None | **Park** |
| 13 | Ante-post racing (big races) | Weeks ahead; a day ahead 2 ANTEPOST_WIN markets carry a median GBP16,678 (max GBP182,765) on 6 days (C2) | Daily snapshots recorded from 7 Oct (25 days: C1) | (b) the model prices big fields weeks ahead; non-runner risk prices in | Unknown | `betfair_live/<day>/books_antepost.csv.gz` | **Research later**: few bets, non-runner risk |

### 1.1 What the census and the inventory changed, and why

Run 41291139483 (job 124802487010) read 25 days of `census_sports.json` (7-31 Oct 2026, 24 sports), the recorder's
marks for October, the greyhound price files 2012-26 and the owner's Historic Data bundle. Four things changed:

1. **Greyhounds: the drift is measured and goes the right way.** The first traded price on GB dogs at 3-51 drifts out
   to the BSP by a median 2.5-13% (section D); the files' morning WAP by 14-24% since 2012 (B4). The market-only test
   is now the first thing to run, not a question of whether to run it. The verdict stays "research next" because the
   lay side's price and size at the first book are still unmeasured, and the files say a morning price that traded in
   volume lost in 2024-26 (`greyhound_model.md` s.4).
2. **Horses: the lay-early cell on long shots is closed.** The 08:00 mid at 51+ sits a median 48% above the BSP (A3)
   because the book is wide (relative spread 0.189 at 08:00, A2): the best lay is higher still, so a morning lay on a
   long shot cannot close at the BSP for a profit. Section 6.1 is rewritten around the 8-21 band, where a 2-5% drift
   out is real at every mark but of the order of half the spread.
3. **Cricket and golf: from "park" to "research next, data first" and "research later".** The census is the first
   measure of pre-off money a day ahead by sport, and it puts cricket match odds first (GBP1.67m a day on about 17
   markets) and golf outrights fourth (GBP507k on 3). Neither has a model, a close benchmark held or a drift
   measured, and both carry an information event before the off, so they sit behind football, not above it.
4. **Tennis, basketball, darts, snooker: parked with numbers.** Tennis's GBP116k a day ahead has a median market of
   GBP5 and a concentration of 116 (C2): a few matches carry it. Basketball's 42 match-odds markets share GBP34,533.
   Darts GBP872. These confirm the earlier reading from the Italian exchange and the trading guides.

Unchanged: football's rank and verdict (the census confirms the depth; the predictability evidence is unchanged),
UK/IE place (closed), politics (no close), AU racing (overnight; the AU greyhound drift, +3-14% at 2-51 in D, says
the same market shape exists there).

## 2. What "open to close" is on Betfair, and why racing is different

- **The trade.** Back at price B with stake S, lay at L later with stake S x B / L. The profit on either outcome is
  S x (B / L - 1) before commission. Lay first, back later: the mirror. The result of the event does not matter once
  both legs are matched. This is arithmetic, not a source.
- **The close is only a single price where there is a Betfair SP.** Racing and greyhounds have one; the trader's lay at
  the SP is `MARKET_ON_CLOSE` and crosses no spread (TRADING.md, "each back laid at the SP"). Outside racing there is no
  BSP; the close is the last price traded before the market turned in play (`betfair_historic.py`, lines 1-4), and
  the exit either takes the best price on offer (one tick worse) or rests an order that may not fill.
- **Where the edge comes from, in principle.** A price that sits away from the close is worth the gap: on Pinnacle
  closing odds 2012-16 (87,960 odds pairs) the ratio of the price taken to the closing price predicts the return to
  level stakes with a slope of "almost exactly 1.00" (football-data.co.uk, Buchdahl). The repository's own reading
  is the same: "a back matched at a price X% better than the BSP is worth about X% of the stake on average"
  (`reports/bsp_calibration.md`, line 100). So an open-to-close trade needs a forecast of the close that is better
  than the open. That is a forecast of information arriving, or of a systematic bias unwinding.
- **Information arrives late.** On UK racing 80.4% of pre-race volume traded in the last 20 minutes (Brown & Yang 2017,
  9,562 races, 2013-14). In the last 10 minutes volume rises about 4.2 times, liquidity 3.4 times and volatility
  1.6 times between the tenth minute and the off (Goncalves et al. 2025, 14,421 UK races, 2014-16). Betfair's own
  page says the "vast majority" of the last 15 minutes' volume is cold traders, over GBP1,500 a second in the last
  five minutes of a standard UK race (Betfair, "Cold trading", read via search snippet; the page refused the fetch).
  The morning is thin and the afternoon is where the price is made. The live rule earns its CLV by being early:
  horses first backed by 11:00 UK +6.8%, after 11:00 +0.5% (`other_markets_and_stakes_1002.md` s.5).

## 3. Question 1: when a liquid price exists, how much is matched pre-off, when the money arrives

### 3.1 Horse racing

| Jurisdiction | Races a day | Pre-off matched a runner (median) | Morning | Source |
|---|---|---|---|---|
| UK win | 28 | GBP11,821 | GBP539 | `other_markets_and_stakes_1002.md` s.1 (archive, one night, 2 Oct) |
| UK place | 28 | GBP1,825 | GBP17 | same |
| IE win | 11 | GBP4,743 | GBP100 | same |
| IE place | 11 | GBP696 | GBP2 | same |
| AU win | 81 | GBP1,180 pre-play | n/a (the window includes post-off trading) | same, s.4 |
| FR / USA / RSA win | 16 / 69 / 10 | GBP256 / GBP300 / GBP615 | | same |

- Volume has fallen. The average UK horse carried GBP53,600 of pre-race win volume in 2013-14 (Brown & Yang 2017,
  line 303 of the UEA manuscript) against GBP11,821 at the median in 2026 (above; a mean against a median, so the
  fall is overstated, but the direction is Racing Post's too: win-market volume "has fallen every year since
  2016"). Per-race pre-off volume is about 40% lower on days with over fifty races than on days with at most two
  meetings; pre-off volume in January 2026 was under 70% of 2022's (algorithmicsportsbetting, Mar 2022-Jan 2026).
- When the money arrives: 80.4% in the last 20 minutes (Brown & Yang 2017); the 10-minute profile above (Goncalves
  et al. 2025). The day's money "comes later (median GBP11,963 pre-play, the morning 4.6% of it)" (TRADING.md, the
  capacity study). The evening before is unformed (ledger `evening-before-1007`).
- The live key's feed shows matched money on 95.5-96.6% of runners from 06:15 UK, median GBP80-105 a runner (ledger
  `live-key-morning-1007`): a tradable open exists at the start of the session, in tens of pounds a horse.

### 3.2 Greyhounds

- 259 races a day in the UK and Australian files, GBP1,274 a runner pre-off (`other_markets_and_stakes_1002.md`);
  about 127 GB win markets a day (`greyhound_feasibility.md`). GB pre-play volume GBP2,213 a runner at the median;
  morning volume GBP0 to the 75th percentile and GBP259 at the 90th; the morning WAP prices 14.7% of runners
  (`greyhound_model.md` s.4). The first traded price is often the first matched bet of a thin market.
- The recorder's GB/IE greyhound books run from 06:55 or 10:10 UTC (`greyhound_model.md` s.5), so the depth behind
  the early price is now measurable.

### 3.3 Football, tennis and the rest (what is published)

- **Football, Betfair match odds, Jul 26-Sep 25 2025** (betaminic, 111 leagues): average matched EPL GBP381,467 at
  T-60, GBP591,539 at T-1, GBP1,928,091 at full time; La Liga GBP172,648 / GBP358,739 / GBP1,477,574; Serie A
  GBP136,500 / GBP289,941 / GBP970,824; Bundesliga GBP80,606 / GBP204,890 / GBP1,198,239; Ligue 1 GBP68,547 /
  GBP148,999 / GBP403,745. So the last hour before kick-off adds 55-150% to the pre-match total, and in-play then
  doubles to sextuples it. How early a price exists in size before T-60 is not published; the census answers it.
- **Tennis.** On Betfair's Italian exchange, 31 Jul-19 Oct 2026, 6,477 pre-match match-odds markets: median matched
  EUR0; 64.9% with nothing matched at the last snapshot; 14.5% over EUR100, 3.3% over EUR1,000, 1.0% over EUR3,000,
  0.3% over EUR10,000; the busiest about EUR25,000; the ATP/WTA main-tour median EUR6 (theoddslab). The UK exchange
  is deeper (unverified), and a 2019 trading guide puts about 90% of tennis money in play (sportstradinglife,
  unverified). The census gives the UK figure.
- **Cricket.** The 2023 World Cup matched GBP5.7bn across the tournament; a 2025 IPL match matched over GBP201m on
  match odds alone; a Feb 2025 Test GBP101m (Betfair Exchange newsletters). The pre-match share is not published; the
  census's evening read of the next day's matches gives it.
- **Politics.** GBP317,307,654 across the 2024 US election markets, GBP250,343,614 on the winner market (oddschecker,
  6 Nov 2024); the Racing Post reported "more than GBP400 million". Depth is not the problem; the absence of a close is.
- **Darts, snooker, golf, US sports.** Nothing published on Betfair's pre-off depth was found. Census.

### 3.4 The census: pre-off money a day ahead, by sport (run 41291139483, section C2)

The census reads, once a day in the evening, every sport with markets starting in the next 24 hours: its market types
with counts, and the money matched so far on its 200 busiest markets, so it compares sports like with like as "the
money matched by then, not each market's final total" (`betfair_recorder.py`, `census_sports`, lines 402-438). The
data agent's inventory (research-query run 41291139483, job 124802487010) read 25 days, 7-31 Oct 2026, 24 sports.
Each figure is a median over the days read; "top 200" is the sport's 200 busiest markets that evening.

| Sport | Days | Markets a day (median) | Top-200 matched, total GBP (median day) | Median market of the 200, GBP | Concentration (mean / median of the 200) | Max day, GBP |
|---|---|---|---|---|---|---|
| Cricket | 25 | 135 | 6,799,296 | 0 | n/a (median 0) | 88,244,909 |
| Soccer | 25 | 6,106 | 908,482 | 993 | 4.6 | 2,446,077 |
| Golf | 17 | 37 | 653,691 | 378 | 8.6 | 1,372,188 |
| American Football | 25 | 19 | 291,257 | 1,554 | 0.9 | 1,556,400 |
| Tennis | 25 | 2,781 | 115,954 | 5 | 116.0 | 6,188,991 |
| Basketball | 25 | 126 | 92,200 | 4 | 115.2 | 912,330 |
| Mixed Martial Arts | 8 | 24 | 42,997 | 6 | 35.8 | 142,089 |
| Ice Hockey | 25 | 80 | 30,179 | 0 | n/a | 119,632 |
| Baseball | 20 | 8 | 25,453 | 12 | 10.6 | 103,548 |
| Horse Racing | 25 | 413 | 23,155 | 4 | 28.9 | 793,657 |
| Motor Sport | 5 | 10 | 22,768 | 442 | 0.3 | 81,688 |
| Rugby League | 10 | 20 | 7,544 | 0 | n/a | 14,611 |
| Boxing | 8 | 15 | 6,746 | 0 | n/a | 238,809 |
| Rugby Union | 13 | 17 | 3,881 | 0 | n/a | 29,674 |
| Snooker | 8 | 8 | 3,344 | 131 | 0.1 | 6,597 |
| Politics | 7 | 1 | 2,040 | 1,993 | 0.0 | 205,093 |
| Darts | 24 | 15 | 872 | 18 | 0.2 | 31,662 |
| Special Bets | 18 | 2 | 617 | 291 | 0.0 | 139,824 |
| Australian Rules | 12 | 8 | 473 | 0 | n/a | 10,907 |
| Esports | 14 | 9 | 297 | 0 | n/a | 3,288 |
| Greyhound Racing | 25 | 481 | 109 | 0 | n/a | 4,380 |
| Volleyball, Cycling, Gaelic Games | 1-11 | 1-2 | 0-1,832 | | | |

The market types that carry the money a day ahead (GBP matched on the type within the top 200, median a day; C2):

| Sport | Market type | Markets a day | Matched a day ahead, GBP | Max day, GBP |
|---|---|---|---|---|
| Cricket | MATCH_ODDS | 17 | 1,669,718 | 88,088,108 |
| Cricket | A_1ST_INN_RUNS_LINE (5 days) | 1 | 601,079 | 1,045,539 |
| Golf | WINNER | 3 | 506,658 | 1,242,569 |
| Soccer | MATCH_ODDS | 185 | 437,775 | 1,624,883 |
| Soccer | ASIAN_HANDICAP | 185 | 175,058 | 532,360 |
| American Football | HANDICAP | 3 | 143,448 | 905,078 |
| Tennis | MATCH_ODDS | 105 | 107,574 | 694,887 |
| Cricket | DRAW_NO_BET (17 days) | 3 | 67,316 | 145,935 |
| American Football | COMBINED_TOTAL | 3 | 54,122 | 426,137 |
| Soccer | ALT_TOTAL_GOALS | 185 | 50,013 | 124,591 |
| Golf | TOP_10_FINISH (13 days) | 1 | 44,235 | 64,067 |
| Mixed Martial Arts | MATCH_ODDS (8 days) | 7 | 40,866 | 141,380 |
| American Football | MATCH_ODDS | 3 | 37,564 | 239,724 |
| Basketball | MATCH_ODDS | 42 | 34,533 | 429,385 |
| Soccer | OVER_UNDER_25 | 184 | 34,173 | 75,356 |
| Horse Racing | WIN | 177 | 22,667 | 575,046 |
| Ice Hockey | MONEY_LINE | 20 | 18,817 | 97,345 |
| Horse Racing | ANTEPOST_WIN (6 days) | 2 | 16,678 | 182,765 |
| Soccer | CORRECT_SCORE | 185 | 15,789 | 36,376 |
| Soccer | BOTH_TEAMS_TO_SCORE | 185 | 12,697 | 52,915 |
| Politics | NONSPORT (6 days) | 1 | 7,490 | 205,093 |
| Tennis | HANDICAP | 23 | 2,651 | 15,948 |
| Cricket | TO_WIN_THE_TOSS (23 days) | 5 | 335 | 7,517,656 |

How to read it:

- **Horse racing's GBP23,155 is small because of when the census reads**: the evening before, when the next day's win
  markets have barely traded (177 win markets share GBP22,667, about GBP128 each). The recorder's marks say the rest
  (A2): a win market holds a median GBP1,423 at 08:00, GBP6,680 at 12:00, GBP13,493 at T-60, GBP26,368 at T-15 and
  GBP77,294 at T-1, so about 68% of a market's pre-off money arrives in the last 15 minutes and 83% after T-60.
- **Greyhounds have no money a day ahead** (GBP109 across the top 200). Their open is on the day, close to the off.
- **The money that exists a day ahead is in cricket, football, golf and the NFL.** Cricket's is in about 17 match-odds
  markets; its other 118 markets a day hold nothing (median 0). Football's is spread (concentration 4.6) across 185
  match-odds and 185 Asian-handicap markets. Golf's is in 3 outrights. The NFL's is in 3 handicap markets.
- **Tennis and basketball are concentrated** (116 and 115): the median market holds GBP5 and GBP4; a handful of matches
  carry the money. Darts, snooker, rugby, boxing, MMA, esports and the specials are thin a day ahead.
- The census does not give the ratio of a day-ahead money to final money (it reads once); the Historic Data tables
  give it per sport once a plan is held, and the recorder's marks give it for horses (above).

## 4. Question 2: is the open-to-close move predictable?

### 4.1 Horse racing (UK/IE)

- **(b) With the model: yes, established and live.** The model is worse than the BSP at forecasting the winner (Brier
  skill negative) but "carries about half the weight in the best forecast of the BSP" beside the morning price
  (TRADING.md). The production features "predict the morning->off move (the CLV mechanism)" (ledger
  `iter3-screen-production-repaired`, line 6). The realised drift ln(morning/BSP) regressed on the forecast's move has
  R2 0.081 (t 21.9), more at longer prices, in jumps and handicaps (ledger `drift-stage2`, line 147). Live: +6.15% CLV
  on GBP47,274 (ledger `staking-1031`). The expected CLV is over-confident later in the day and near the off (ledger
  `closing-model-by-hour-1007`): the forecast of the close is good in the morning and decays as the market forms.
- **(a) Market alone, back side: no.** Placed by the morning price, UK win 2023-26Q1, 280,120 runners: favourites
  (1-2) +0.48% CLV, 3-5 -1.3%, 8-13 -2.4%, 21-51 -12.0%, 51+ -24.4%; runners with GBP2,000+ morning volume +1.0%
  (ledger `market-only-drift-1002`). Morning favourites drift on average (mean ln(BSP/morning) +0.036, shortened in
  47.3%: ledger `morning-favourite-clv`). The classic favourite-longshot bias is smaller on exchanges than with
  bookmakers (Smith, Paton & Vaughan Williams 2006: exchanges "exhibit evidence of significantly lower market
  biases"), and exchange odds carry more predictive value than bookmakers' in liquid markets (Smith, Paton &
  Vaughan Williams 2009, JEBO, abstract via search snippet).
- **(a) Market alone, lay side: read on the recorder's books (run 41291139483, section A3; Oct 2026).** The mid of
  the book at each mark against the BSP, win markets, median log(BSP / mid) (positive: drifted out):

  | Mid band | 08:00 (n) | 10:00 | 12:00 | T-120 | T-60 | T-15 | T-1 |
  |---|---|---|---|---|---|---|---|
  | 1.5-2 | -0.055 (68) | -0.041 | -0.022 | -0.014 | +0.006 | +0.014 | -0.005 |
  | 2-3 | -0.027 (229) | -0.009 | -0.003 | 0.000 | +0.013 | +0.012 | 0.000 |
  | 3-5 | +0.010 (594) | +0.007 | +0.007 | +0.012 | +0.003 | +0.002 | +0.010 |
  | 5-8 | 0.000 (1,002) | +0.012 | +0.006 | 0.000 | +0.009 | +0.013 | +0.014 |
  | 8-13 | +0.024 (1,163) | +0.041 | +0.040 | +0.023 | +0.031 | +0.036 | +0.020 |
  | 13-21 | +0.037 (1,092) | +0.055 | +0.051 | +0.038 | +0.025 | +0.031 | +0.016 |
  | 21-51 | -0.075 (1,167) | +0.018 | +0.022 | +0.023 | +0.020 | -0.014 | -0.005 |
  | 51+ | -0.476 (1,670) | -0.049 | -0.058 | +0.119 | +0.045 | -0.088 | -0.095 |

  Three readings. (i) The long shots' -24% CLV in the price files is a fact about traded prices, not about the book: at
  08:00 the 51+ mid sits a median 48% above the BSP (shortened 68.5%) because the book is wide (median relative spread
  0.189 at 08:00, best-back size GBP6: A2), and the best lay is above the mid. A lay there cannot close at the BSP
  for a profit; the cell is closed on sight. (ii) The 8-21 band drifts out 2-5% at every mark including T-15 and T-1,
  on 1,000-1,650 runners a cell, with means equal to the medians (A3): a real market-only drift, the favourite-
  longshot bias unwinding in the band where it is smallest. Against it: half the relative spread at 08:00 is about 9%,
  at T-120 3.3%, at T-15 2.5% (A2, medians over all runners; wider at these prices), and a tick at 10-20 is 2.5-5% of
  the price. A market-only lay at 8-21 is about break-even before commission. (iii) Favourites shorten a little from
  08:00 (-5.5% at 1.5-2) and not at all from 10:00. The model's job is to pick which of the 8-21 runners drift: that
  is the live rule's mirror, and `model/race_book.py` already prices lays. Whelan (2025) is the warning for any
  market-only lay: on Betfair football takers lose more the longer the price and makers earn about zero after fees,
  which is what a book wide enough to absorb the bias looks like. Section 6.1 is rewritten around this.
- **(c) Known not to work:** in-play offsets, every cell negative (ledger `edge-inplay-offsets-1002`); SP-bias
  segments (ledger `edge-bsp-bias-1002`); the evening-before entry (ledger `evening-before-1007`); entry near the
  off with the morning model, CLV -1.8% to -5.8% (ledger `late-entry-0930`).
- **Microstructure.** Pre-race volume is weakly informative: a horse with above-median pre-race win volume at the
  same price is 6.2% more likely to win; in the place market 1.1% less likely to place; in running 9.3% and 18.9%
  (Brown & Yang 2017). Betfair returns have light tails, autocorrelations that die quickly and no long memory
  (Tondapu 2024, arXiv 2402.02623): the price path itself carries little to forecast from. A 2025 deep-learning
  attempt at the last two minutes' direction from the order book reached 30.9% accuracy on five classes, "only 11
  percentage points above the baseline" (Goncalves et al. 2025). Informed late money is old and robust: plunges
  carry inside information (Schnytzer & Shilony 1995, Australian bookmakers); marked shortening signals insiders
  (Crafts 1985, 13.6% of 16,769 runners with a 1.5:1 probability change, UK 1982-83); late tote money is smart
  money (Gramm, McKinney & Parker; parimutuel). None of it is a drift one can front-run from the market alone.

### 4.2 Greyhounds (UK)

- **(a) Market alone: measured by band (run 41291139483, sections D and B4).** The owner's Historic Data bundle,
  1,863,552 runners with a first and a closing price in 237,744 GB/AU/NZ markets, Jan-Sep 2026; and the price files
  2012-26. Median log move, positive = drifted out; "short" = share that shortened:

  | GB win, first-price band | n | First traded -> BSP | short | T-1 -> close (all countries, n) | Morning WAP -> BSP, files 2012-26 (n) |
  |---|---|---|---|---|---|
  | 1.5-2 | 2,398 | +0.148 | 33% | -0.006 (11,393) | +0.145 (9,727) |
  | 2-3 | 17,933 | +0.056 | 41% | 0.000 (33,876) | +0.137 (50,446) |
  | 3-5 | 51,360 | +0.025 | 45% | 0.000 (86,402) | +0.146 (169,623) |
  | 5-8 | 56,703 | +0.050 | 42% | +0.017 (85,391) | +0.159 (161,392) |
  | 8-13 | 35,870 | +0.103 | 36% | +0.021 (70,035) | +0.154 (99,978) |
  | 13-21 | 13,858 | +0.129 | 34% | +0.030 (50,394) | +0.215 (29,532) |
  | 21-51 | 7,882 | +0.117 | 37% | +0.044 (58,346) | +0.209 (11,059) |
  | 51+ | 2,603 | -0.262 | 61% | +0.057 (44,015) | -2.951 (2,009) |

  The close is calibrated: won against implied within 0.003 in every band from 2.0 up (D), so the BSP is the right
  close. The first price on GB dogs at 2-51 is systematically too short, by 2.5-13% at the median and more at the
  mean (+0.084 to +0.182), and it is still too short at T-1 by 2-6% at 8-51. The files say the morning WAP has been
  too short by 14-24% at 2-51 since 2012, with only 20-38% of dogs shortening (B4). That is the favourite-longshot
  bias unwinding towards the off, the drift a lay-early trade needs, in the market with the most races. The 51+ band
  is the reverse (a first trade at 85 is usually a silly lay that gets taken, then shortens).
  What D cannot say: the first price's volume (BASIC has none), the lay side at that moment, and how often a price
  exists early at all (a price stood 24 h out on 78 of 451,031 runners; the census sees GBP109 a day ahead). The
  files add that the GB morning money is GBP0 at the median and GBP479-1,182 at the 90th percentile 2019-23 (B4),
  and `greyhound_model.md` s.4 that a morning price traded in volume was about 20% shorter than the BSP and lost
  15-19% backed. Read together: the drift is real, and the money that trades early is on the dogs that drift; the
  question for section 6.2 is whether the lay side of the first two-sided book is near enough to the first trade.
- **(b) With the model:** form predicts which dogs shorten. At the first price, edge > 0.2: +7.9% at that price, CLV
  +6.0%, Jan-Sep 2026; at T-1 about nothing (`greyhound_model.md` s.4). The morning WAP rules were positive 2021-23
  and negative 2024-26, and a morning price traded in volume lost 15-19% (`greyhound_model.md` s.4). The paper
  forward test runs from 5 Oct with its gate fixed (`greyhound_model.md` s.5). No academic study of greyhound
  open-to-close moves on the exchange was found; the one FLB study found is US parimutuel (Weinbach & Paul 2007).

### 4.3 Football

- **(c) Market alone, pre-match:** Betfair's last pre-play prices are unbiased forecasts (Whelan 2025, 200,622
  matches 2022-24, the F-test cannot reject unbiasedness). Takers of pre-play match-odds offers returned -2.5%,
  makers +0.6% (t 11.8 for the difference) on 902,568 bets; total goals: takers -2.2%, makers +2.0% on 2.09m bets
  (Whelan 2025, s.7.1). A maker-only strategy therefore earns about the commission back and no more on match odds;
  the total-goals maker return of 2% is a market-making return worth a look (section 6.3), not an open-to-close
  drift. The one open-to-close reading found is small: on 247 Italian-exchange markets over EUR3,000 in August 2026,
  outcomes that shortened by over 3% from the first to the last pre-match price won 85 times against 99 implied
  (z -2.03), drifters 75 against 61 (z +2.20): over-reaction to pre-match money, football and tennis pooled, one
  month, several thresholds tried (theoddslab). Suggestive of mean reversion, far from established.
- **The close is the forecast, the open is not.** Pinnacle's close prices the return 1:1 (Buchdahl; section 2).
  Franck, Verbeek & Nuesch (2010): the exchange forecasts the Big Five leagues (5,478 games) better than eight
  bookmakers, and backing where bookmakers' odds exceed the exchange's earns above-average and sometimes positive
  returns. Kaunitz, Zhong & Kreiner (2017): a strategy that backs the outlier against the consensus of 32 bookmakers'
  closing odds was profitable over a decade's simulation and five months of real money, until the books restricted
  the account. Angelini & De Angelis (2019): eight of eleven European markets efficient at the best available odds,
  three not. Hubacek, Sourek & Zelezny (2019) and Hubacek & Sir (2023): profit comes from de-correlating a model from
  the market's price, not from accuracy. Kuypers (2000): the English fixed-odds market held exploitable inefficiency.
  All of these are a model-or-outlier edge against bookmakers, i.e. class (b), and none measures a Betfair open.
- **(b) In the repository:** the close beats the open by 0.0034 log-loss on 1X2 and the Dixon-Coles model cannot
  see why; the pooling weight on the model is negative (`football_model.md` s.3). The information that moves the
  price is team news and money after the open. The route is line-ups about an hour before kick-off (API-Football)
  and a measurement of the price move around them (`football_model.md` s.5). In-play, goals are priced "swiftly and
  fully" (Croxson & Reade 2014): class (c) and outside the brief anyway.

### 4.4 Tennis

- **(a)** A favourite-longshot bias exists on tennis exchanges, larger between lower-ranked players, in later rounds
  and at high-profile tournaments (Abinzano, Muga & Santamaria 2016, Applied Economics Letters), and it rises with
  trading volume and falls with institutional bettors (same authors, 2017). On bookmakers' average closing odds
  2011-22 the bias is explained by disagreement among uninformed bettors (UCD WP23/16). A bias at the close is not an
  open-to-close move; nothing found measures the pre-match drift on Betfair.
- **(b)** In play, the market lags the effect of a break of serve (Easton & Uylangco 2010, Betfair): a speed edge,
  outside the brief.
- With pre-match depth as thin as section 3.3 reports, the question is moot until the census shows otherwise.

### 4.5 Cricket, golf, darts, snooker, US sports, politics

- Nothing found measures Betfair open-to-close moves in these. Golf: Pinnacle's opening line moves 54.7% of the way to
  Betcris's where they differ by 5%+, Betcris 15.6% towards Pinnacle, and Pinnacle's close prices the return about
  1:1 (datagolf, 106,204 matchups 2019-20): a sharp-book-leads-the-move story, class (b), and not on Betfair. US
  sports: the same story is general knowledge, unverified here. Politics: months-long, news-driven, no close.

## 5. Question 3: what trading out costs and constrains

### 5.1 Commission, the Expert Fee, the Premium Charge

- **Commission.** 2% on the account (TRADING.md; `other_markets_and_stakes_1002.md` s.2), charged on the net profit of
  each market. A greened trade's profit is the market's net profit, so 2% comes off the gap, never off turnover.
- **Expert Fee** (from 6 Jan 2025; Racing Post, 18 Dec 2024). Betfair's FAQ, read via search snippet on 1 Nov 2026
  because the support pages refused the fetch: an account "will only be considered for the Expert Fee if the last 52
  active week gross profit is greater than GBP25,000"; it "will need to be in lifetime profit"; "Expert Fee Due =
  (Weekly Gross P&L - Buffer) * Current Expert Fee Rate - Weekly Commission Generated"; the rate "increases from 20%
  to 40% as their rolling 52-week gross P&L increases from GBP25-100k to >GBP100k". A third-party reading adds that
  losing weeks build the buffer and the account must have bet in over 100 markets (botblog; not confirmed on
  Betfair's page). The repository's 2025 reading (`market_research_programme.md`, "Betfair's Expert Fee") stands;
  the open point there, whether 20% applies to all of the week's gross profit or only the excess over the band,
  remains open: the FAQ's formula applies the rate to the week's gross P&L less the buffer, the third-party guides
  read it as the excess. The account's own Expert Fee page settles it.
- **What it means for a trading programme.** The fee is on gross profit per week, net of the commission that week
  already paid. At 2% commission and a 20% rate, a week of GBP1,000 gross profit pays GBP20 commission and then
  GBP180 fee: an effective 20% of gross. Every strategy on the account adds to the same 52-week total. At the live
  rule's +GBP4,532 in 22 days (TRADING.md, 31 Oct), the account reaches GBP25,000 of gross profit in about 120 days
  if it continues (arithmetic). A second account is not an answer: the fee is per customer (general knowledge,
  unverified).
- **Premium Charge.** Withdrawn when the Expert Fee came in; it ran from 2008 on lifetime profit, with a higher charge
  from 2011 of up to 60% (Racing Post). Not a live cost.

### 5.2 Ticks, spread, minimum bets, the round trip

Betfair's price ladder (developer docs, "Betfair Price Increments", read via search snippet): 1.01-2 in 0.01; 2-3 in
0.02; 3-4 in 0.05; 4-6 in 0.1; 6-10 in 0.2; 10-20 in 0.5; 20-30 in 1; 30-50 in 2; 50-100 in 5; 100-1000 in 10. A bet
off the ladder is refused (`INVALID_ODDS`). Minimum bet GBP2 on most of the docs' pages (one page says GBP1), with
smaller stakes allowed where the payout would be at least GBP10 (the docs' "lower minimum stakes at larger prices").
The trader already meets the GBP2 floor and the SP-lay minimum (ledger `live-key-morning-1007`: 41 backs skipped
below the GBP2 minimum, 10 too small to lay at the SP).

**The round trip.** Back S at B, lay at L: profit S x (B/L - 1), less 2% if positive. Taken at both ends with no move,
the lay is one tick under the back, so the loss is about one tick as a share of the price:

| Price | Tick | One tick as a share of the stake (the cost of crossing once) | Move needed to clear it and 2% |
|---|---|---|---|
| 1.50 | 0.01 | 0.7% | 0.7% |
| 2.50 | 0.02 | 0.8% | 0.8% |
| 3.50 | 0.05 | 1.4% | 1.5% |
| 5.0 | 0.1 | 2.0% | 2.1% |
| 8.0 | 0.2 | 2.6% | 2.6% |
| 15 | 0.5 | 3.4% | 3.5% |
| 25 | 1 | 4.2% | 4.2% |
| 40 | 2 | 5.3% | 5.4% |
| 75 | 5 | 7.1% | 7.3% |
| 150 | 10 | 7.1% | 7.3% |

Arithmetic from the ladder: a back at B closed at L = B - tick gives B/L - 1. The spread is often more than one tick
in thin markets and in the morning (win books lay/back 1.16-1.84 at the trader's first step: ledger
`evening-before-1007`), and resting an order at the best price risks not filling. On racing and greyhounds the SP
exit costs nothing of this, which is why the live rule's whole 6% survives. In football match odds at 2-3, a
one-tick cost of under 1% is survivable; at long prices anywhere it is not. The first-fill-versus-top-up gap the
live record shows (first backs +10.7%, top-ups +5.9% on 31 Oct, TRADING.md) is the same cost in another form: the
top-ups pay for moving the price.

### 5.3 Keys, data limits, latency

- **Two application keys** (developer docs, read via search snippets). The delayed key: prices delayed "between 1-180
  second snapshots", no `totalMatched`, no `EX_ALL_OFFERS`, no order placement. The live key: a one-off GBP299
  activation (the repository's reading, `all_markets_priorities.md` s.5, agrees), the Stream API with it. The owner
  holds both; the live key serves only the live trader (CLAUDE.md, secrets table; `football_model.md` s.4: "Betfair
  permits no read-only use of live data"). Any new market's trader must run inside that one process or be approved
  for the key.
- **Market data request limits.** Each `listMarketBook` request carries a weight: the sum of its price projections'
  weights times the number of market ids, at most 200 points; `EX_BEST_OFFERS` weighs 5, `EX_BEST_OFFERS` with
  `EX_TRADED` 20; over the cap returns `TOO_MUCH_DATA` (developer docs, "Market Data Request Limits", current page via
  search snippet; an older revision said 150). Polling 40 markets with best offers in one call is the ceiling; more
  markets or more depth means more calls or the stream.
- **Transaction charges.** Betfair's current page was not reachable this session. Third-party and forum readings say
  a charge of about GBP0.01 a transaction applies past 1,000 (some say 5,000) order transactions an hour when the
  account's commission does not cover them, and that failed transactions may count (developer forum, 2020;
  tryfix.it.com guide): unverified. The live trader polls each minute and places fill-or-kill backs (TRADING.md),
  so it sits well under either figure; a ladder of three backs a horse across 40 races is still hundreds, not
  thousands, an hour.
- **Latency.** The trader runs on the UK server because GitHub's runners are in the US and Betfair restricts some
  countries (TRADING.md). The delayed key's 1-180 s lag is the main latency the paper trader lived with; on the live
  key "a fill-or-kill back at a delayed price fills better on a drifter and misses a steamer" ceased to bias the
  fills (ledger `closing-model-by-hour-1007`, real fills; ledger `live-key-morning-1007`: 3% killed against 19-25%).
  Betfair's "passive bet delay" test (May 2025 newsletter title) was not readable this session: unverified what it
  delays.
- **Bet delay and suspension.** Markets suspend for a second at the off, cancelling pre-race quotes (Brown & Yang 2017,
  footnote). An exit resting into the off is cancelled; the SP lay is not.

## 6. Question 5: next steps for the data and modelling agents

### 6.1 UK/IE horse win: the 8-21 lay, with the model (modelling agent; the long-shot cell is closed)

Section A3 of run 41291139483 closed the long-shot lay: the 08:00 mid at 51+ sits 48% above the BSP and the best lay
higher, so no morning lay on a long shot closes at the BSP for a profit. What is left is the 8-21 band, where the mid
drifts out a median 2.4-5.5% at every mark (A3) against a half-spread of the same order (A2).

- **Data needed:** `betfair_live_marks` (the book at 08:00-12:00 UK and T-120..T-1 with the BSP: CLAUDE.md) from
  1 Oct, 13,992-18,982 runners a mark with a BSP (A3); the trader's own books for the lay side's size at each poll.
  No new source.
- **The market-only read first (data agent, one query):** at each mark, for win runners whose best lay is 8-21, the
  CLV of a lay at the best lay closed by a back at the BSP, 1 - lay / BSP per pound of the backer's stake, by band
  and mark, with the size offered at the best lay; the share of runner-marks where the best lay is under the BSP.
  Expect about zero: it says how much of the 2-5% the spread takes at each hour, and where in the day a lay is
  cheapest (A2: the relative spread halves between 12:00 and T-120).
- **Then the model (modelling agent):** the live rule's mirror: lay where the model's expected BSP is well above the
  best lay, in 8-21 first, closed at the BSP. `model/race_book.py` prices the lays and the whole-race book already;
  it has never been run with the lay side live. Replay on the recorded books 1-31 Oct as `rule-candidates-replay`
  did for the backs, settled at Betfair's prices, 2% commission, with the day limit shared with the backs. Pass: CLV
  x stake positive on 20 of 31 days and the 90% interval above zero; then paper beside the live trader before money.

### 6.2 UK greyhounds: the lay at the first two-sided book against the BSP (data agent; the first test to run)

Section D made this the strongest market-only candidate in the programme: the GB first traded price at 2-51 drifts
out to the BSP by a median 2.5-13%, shortening on only 33-45% of dogs, on 7,882-56,703 runners a band; and T-1 still
drifts 2-6% at 8-51. What D lacks is the lay side and the money.

- **Data needed:** the recorded GB greyhound books from 6 Oct (`betfair_live/<day>/books_greyhound.csv.gz`, 27 days,
  76.7 MB: C1), read through `sources/open_close.py` with the first two-sided book as the open (the data agent's
  panel, `reports/open_close/data.md`); each race's BSP from the price files or the settled book.
- **The first test (market-only), pre-registered here:** at the first two-sided book and at T-60, T-10 and T-1, for
  every GB win runner: best back, best lay, their sizes, the market's matched money; against the BSP. Report by band
  of the best lay (2-3, 3-5, 5-8, 8-13, 13-21, 21-51): the CLV of a lay at the best lay closed at the BSP
  (1 - lay / BSP), the median and the mean, the share under the BSP, the size offered, and the minutes to the off at
  the first two-sided book. The cell to judge, chosen now from D's bands: **lay at the first two-sided book where the
  best lay is 8-51 and at least GBP10 is offered, closed at the BSP, 2% commission.** Pass: the mean CLV positive with
  its 90% interval above zero on 6-31 Oct, and again on November's first fortnight read once. Also report the mirror
  (back at the first book, lay at the BSP) so the direction is seen both ways, and the same cells at T-10 and T-1
  where the money is.
- **What would make it tradable at size:** the 90th-percentile morning money is GBP479-1,182 a runner (B4); the T-1
  drift of 2-6% at 8-51 (D) is where the money is but is about one tick (0.5 at 10-20). If the first-book cell passes
  and the T-1 cell does not, the trade is a lay in the first minutes of a book, small, across 127 races a day.
- **Then:** the paper track's primary rule (first recorded price, edge > 0.2) continues unchanged to its gate
  (`greyhound_model.md` s.5). Read together: if the market-only lay passes, the model's job is to pick which dogs
  not to lay, and the forward test's back side says which dogs the market is right to shorten.

### 6.3 Football: the open-to-close move itself (data agent; the owner's one action first)

- **The owner:** "buy" the free BASIC plan for Soccer at historicdata.betfair.com (outstanding since 3 Oct:
  `all_markets_priorities.md` s.5). Without it there is no Betfair open.
- **Data needed:** `betfair_historic.py --fetch --sport Soccer`, then `--build`: first traded, T-24h, T-60, T-15, T-1
  and the pre-play close per runner (`betfair_historic.py` docstring). `football/data.py` already maps markets to
  matches and holds Pinnacle's open and close from football-data.co.uk.
- **The first test (market-only):** by league tier and market type (match odds, O/U 2.5): the distribution of
  ln(close / first traded) and ln(close / T-24h); its autocorrelation with the move already made by T-60 (does a
  move continue or revert: the Italian sample says revert, p 0.03-0.04, one month); the CLV of a lay at T-24h closed
  at the pre-play close by band; the maker return by resting one tick inside the best price at T-60 (Whelan's +2%
  on total goals). Two seasons for choosing a cell, the latest for reading it.
- **Only if a cell passes:** a line-up trade needs API-Football (US$19 a month) and a feed inside the live trader's
  process. That is a project, not a test.

### 6.4 Cricket and golf (data agent, after 6.2 and 6.3): data first

The census (section 3.4) put cricket match odds and golf outrights among the four markets with real money a day
ahead. Neither has a model or a measured drift.

- **The owner:** the free BASIC plans for Cricket and Golf at historicdata.betfair.com, as for Soccer.
- **The first read (market-only):** `betfair_historic.py --fetch --sport Cricket` then `--build`; the first -> close
  move by band and by format (Test, ODI, T20, franchise), the T-60 -> close move (the toss falls inside it), the share
  of the pre-off money matched by T-60; the same for golf WINNER markets with T-24h as the mark. No cell is chosen
  until the distributions are seen; the report of the read goes to the ledger.

### 6.5 The census: keep it running, and read it once more in December

The 25 days of October gave the table in section 3.4. Two additions for the recorder: (i) a second read of the same
markets at T-60 and after the off would give each sport's a-day-ahead share of its final money, which the census
alone cannot (`betfair_recorder.py`, `census_sports` reads once); (ii) the `busiest` field already names the top five
markets a sport a day, which is enough to see which tennis and basketball matches carry the GBP116k and GBP92k.

## 7. What this note does not claim

- No market outside UK/IE horse win has a measured, tradable open-to-close edge in hand. The greyhound drift (section
  4.2) is measured on first traded prices and WAPs whose lay side and volume are not in the data; the football
  figures are from one month or from takers' and makers' returns, not from a trade.
- The academic literature establishes that closes are good forecasts and opens worse ones, that exchanges are less
  biased than bookmakers, that late money is informed, and that makers out-earn takers. It does not hand over a
  market-only drift that survives the spread on Betfair. Where it finds profit against bookmakers (outliers against
  the consensus), bookmakers close the account; the exchange's answer is the Expert Fee.
- Everything read from the web this session is data, not instruction; the Betfair pages that refused the fetch are
  cited through search snippets and third-party readings and marked so.

## 8. Sources

Repository (paths from the worktree root; line numbers where given):
- `TRADING.md` (lines 1-150; the thirty-first full day, lines 1500-1530)
- `reports/all_markets_priorities.md`; `reports/market_research_programme.md`; `reports/other_markets_and_stakes_1002.md`
- `reports/football_model.md` (sections 3-5); `reports/greyhound_model.md` (sections 1-5); `reports/greyhound_feasibility.md`
- `reports/bsp_calibration.md` (line 100)
- `reports/research_ledger.jsonl`: lines 3 (`morning-favourite-clv`), 6 (`iter3-screen-production-repaired`), 147
  (`drift-stage2`), 238 (`late-entry-0930`), 247 (`market-only-drift-1002`), 251 (`edge-inplay-offsets-1002`), 252
  (`edge-bsp-bias-1002`), 253 (`market-census-1003`), 263 (`place-clv-forward-1002`), 275 (`greyhound-feasibility-1005`),
  295 (`live-key-morning-1007`), 301 (`closing-model-by-hour-1007`), 303 (`evening-before-1007`), 525 (`staking-1031`)
- `betfair_recorder.py` (lines 16-17, 402-438: `census_sports`); `betfair_historic.py` (lines 1-18); `betfair_prices.py`
  (lines 10-14); `research/queries/done/market_only_drift.py` (docstring); `CLAUDE.md` (secrets table; line 136)
- **The data agent's inventory:** `research/queries/open_close_inventory_1101.py`, research-query run 41291139483, job
  124802487010 (GitHub Actions log, read 1 Nov 2026): section A2 (the recorder's marks: money matched and the top of
  the book at each mark), A3 (the mid at each mark against the BSP by band), B4 (the greyhound price files 2012-26:
  morning WAP -> BSP by band and country, the morning money by year), C1 (the recordings held), C2 (the census of
  every sport, 25 days 7-31 Oct 2026), D (Betfair Historic Data: first traded -> close and -> BSP, T-60/T-15/T-1 ->
  close, the close's calibration); `reports/open_close/data.md` (the data agent's note and the panel schema)

Betfair's own pages (support and developer pages refused this session's fetch; read via search snippets on 1 Nov 2026
unless noted):
- Expert Fee FAQs: https://support.betfair.com/app/answers/detail/expert-fee-faqs
- Betfair Charges: https://support.betfair.com/app/answers/detail/betfair-charges/
- Market Data Request Limits: https://betfair-developer-docs.atlassian.net/wiki/x/9gEp
- Application keys (delayed and live): https://betfair-developer-docs.atlassian.net/wiki/x/qxYp
- Price increments: https://betfair-developer-docs.atlassian.net/wiki/x/HIDJ ; currency minimums: https://betfair-developer-docs.atlassian.net/wiki/x/eRkp
- Exchange API FAQ (costs): https://developer.betfair.com/en/exchange-api/faq/ ; https://support.developer.betfair.com/hc/en-us/articles/115003864531-Are-there-any-costs-associated-with-API-access
- Transaction charges (forum, 2020, unverified): https://forum.developer.betfair.com/forum/sports-exchange-api/exchange-api/33242-transaction-charges
- Cold trading: https://betting.betfair.com/cold-trading-on-betfair.html
- Exchange newsletters: May 2025 (IPL GBP201m): https://betting.betfair.com/betfair-announcements/exchange-news/betfair-exchange-may-newsletter-ipl-breaks-record-and-passive-bet-delay-testing-scales-up-280425-204.html ;
  Nov 2023 (World Cup GBP5.7bn): https://betting.betfair.com/betfair-announcements/exchange-news/betfair-exchange-newsletter-cricket-world-cup-breaks-records-on-exchange-291123-6.html ;
  Feb 2025 (Test GBP101m): https://betting.betfair.com/betfair-announcements/exchange-news/betfair-exchange-february-newsletter-exciting-improvements-for-2025-and-cricket-test-matches-100m-280125-204.html
- Historic Data service: https://historicdata.betfair.com/ (help page returned 410; plan contents per `betfair_historic.py`)

Press and third-party readings of Betfair's charges:
- Racing Post, 18 Dec 2024, Expert Fee and the end of the Premium Charge (fetched): https://racingpost.com/news/britain/betfair-exchange-to-introduce-new-commission-system-for-2025-as-premium-charge-is-dropped-a7wbg0v4GCAJ
- botblog, Expert Fee explained (fetched; third party): https://botblog.co.uk/betfair-expert-fee-premium-charge/
- tryfix.it.com, API pricing guide (search snippet; third party, unverified): https://tryfix.it.com/how-much-does-betfair-api-cost-the-complete-2026-pricing-guide/

Academic:
- Brown, A. & Yang, F. (2017) The role of speculative trade in market efficiency: evidence from a betting exchange. Review of Finance 21(2), 583-603. Manuscript (fetched, text extracted): https://ueaeprints.uea.ac.uk/58626/1/Speculative_Trade_Revisedv5.pdf
- Whelan, K. (2025) Agreeing to disagree: the economics of betting exchanges. MPRA 126351 (fetched, text extracted): https://mpra.ub.uni-muenchen.de/126351/1/MPRA_paper_126351.pdf
- Goncalves, R., Ribeiro, V., Chertovskih, R. & Aguiar, A. P. (2025) Convolutional attention in betting exchange markets. arXiv 2510.16008 (fetched): https://arxiv.org/html/2510.16008
- Tondapu, S. (2024) Efficient market dynamics: ... Betfair's time series analysis. arXiv 2402.02623 (abstract fetched): https://arxiv.org/abs/2402.02623
- Smith, M. A., Paton, D. & Vaughan Williams, L. (2006) Market efficiency in person-to-person betting. Economica 73(292), 673-689 (abstract fetched): https://ideas.repec.org/a/bla/econom/v73y2006i292p673-689.html
- Smith, M. A., Paton, D. & Vaughan Williams, L. (2009) Do bookmakers possess superior skills to bettors in predicting outcomes? JEBO (abstract via search snippet): https://hal.archives-ouvertes.fr/hal-00684229
- Franck, E., Verbeek, E. & Nuesch, S. (2010) Prediction accuracy of different market structures: bookmakers versus a betting exchange. IJF 26(3), 448-459 (abstract fetched): https://ideas.repec.org/a/eee/intfor/v26yi3p448-459.html
- Angelini, G. & De Angelis, L. (2019) Efficiency of online football betting markets. IJF 35(2), 712-721 (abstract via search): https://cris.unibo.it/handle/11585/646980
- Croxson, K. & Reade, J. J. (2014) Information and efficiency: goal arrival in soccer betting. Economic Journal 124(575), 62-91 (abstract fetched, WP version): https://ideas.repec.org/p/bir/birmec/11-01.html
- Easton, S. & Uylangco, K. (2010) Forecasting outcomes in tennis matches using within-match betting markets. IJF 26(3), 564-575 (abstract fetched): https://ideas.repec.org/a/eee/intfor/v26yi3p564-575.html
- Hubacek, O., Sourek, G. & Zelezny, F. (2019) Exploiting sports-betting market using machine learning. IJF 35(2), 783-796: https://ideas.repec.org/a/eee/intfor/v35y2019i2p783-796.html ; Hubacek, O. & Sir, G. (2023) Beating the market with a bad predictive model. arXiv 2010.12508: https://arxiv.org/abs/2010.12508v1
- Kaunitz, L., Zhong, S. & Kreiner, J. (2017) Beating the bookies with their own numbers. arXiv 1710.02824 (abstract fetched): https://arxiv.org/abs/1710.02824
- Kuypers, T. (2000) Information and efficiency: an empirical study of a fixed odds betting market. Applied Economics 32(11), 1353-1363 (abstract via search): https://ideas.repec.org/a/taf/applec/v32y2000i11p1353-1363.html
- Strumbelj, E. (2014) On determining probability forecasts from betting odds. IJF 30(4), 934-943 (cited via secondary sources; not read)
- Crafts, N. F. R. (1985) Some evidence of insider knowledge in horse race betting in Britain. Economica 52, 295-304 (via Bangor/Kent working papers citing it): https://www.kent.ac.uk/economics/repec/0819.pdf
- Schnytzer, A. & Shilony, Y. (1995) Inside information in a betting market. Economic Journal 105(431), 963-971 (abstract via search): https://ideas.repec.org/a/ecj/econjl/v105y1995i431p963-71.html
- Gramm, M., McKinney, C. N. & Parker, D. (late money in parimutuel pools; via search snippet): https://economics.ecu.edu/wp-content/pv-uploads/sites/165/2019/07/ECU1202.pdf
- Abinzano, I., Muga, L. & Santamaria, R. (2016) Game, set and match: the favorite long-shot bias in tennis betting exchanges. Applied Economics Letters (abstract via search): https://academica-e.unavarra.es/handle/2454/18748 ; (2017) Hidden power of trading activity: https://academica-e.unavarra.es/handle/2454/25857
- UCD WP23/16 (2023), tennis favourite-longshot bias on average closing odds 2011-22 (via search snippet; PDF refused): https://www.ucd.ie/economics/t4media/WP23_16.pdf
- Weinbach, A. & Paul, R. (2007) greyhound favourite-longshot bias, US track (abstract via search): https://ideas.repec.org/a/spr/jecfin/v31y2007i1p49-58.html

Practitioner and data sources:
- Buchdahl, J., The efficiency of the Pinnacle.com closing line (fetched): https://football-data.co.uk/blog/pinnacle_efficiency.php
- betaminic, Which football leagues have the most liquidity on Betfair? (fetched; Jul 26-Sep 25 2025): https://www.betaminic.com/betting-research/which-football-leagues-have-the-most-liquidity-on-betfair/
- theoddslab, Tennis trading on Betfair: what the pre-match prices do (fetched; Italian exchange, Jul-Oct 2026): https://theoddslab.com/articoli/tennis-trading-betfair/
- algorithmicsportsbetting (Substack), Two markets, one race (fetched; partly paywalled; Mar 2022-Jan 2026): https://algorithmicsportsbetting.substack.com/p/two-markets-one-race-degeneracy-and
- datagolf, How sharp are bookmakers? (fetched; 106,204 matchups 2019-20): https://datagolf.com/how-sharp-are-bookmakers
- oddschecker, Over GBP300 million matched on 2024 US election markets (search snippet): https://www.oddschecker.com/insight/industry/20241106-over-300-million-matched-on-2024-us-election-markets-on-betfair-exchange ; Racing Post, over GBP400 million matched: https://racingpost.com/news/world-waits-on-next-president-with-over-400-million-matched-on-betfair-market-ax7aW1G6CmFB
- sportstradinglife, How to trade tennis (2019; the "about 90% in play" figure, unverified): https://sportstradinglife.com/2019/02/how-to-trade-tennis/
