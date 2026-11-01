# Open-to-close trading on Betfair: which markets (research agent, 1 Nov 2026)

The owner's ask (1 Nov 2026): autonomous trading on Betfair, in markets where the price can be traded from its
opening to its close, hedged so the result does not matter. This note ranks Betfair's sports and market types for
that purpose. It builds on what the repository already knows and cites it; it does not repeat it. Every figure
traces to a source in section 8 or to a repository file. Where a claim rests on general market knowledge, it says
"unverified".

## The result in six lines

1. **The UK/IE horse win market is the one market where this trade is proven, and it is already live.** The win rule is
   an open-to-close trade: back in the morning where the model's BSP forecast is well under the price, lay at the
   BSP. GBP47,274 staked 19-31 Oct at +6.15% CLV (research ledger `staking-1031`, line 525; TRADING.md). The edge is the
   model's, not the market's: with no model there is no drift to catch on the back side (ledger `market-only-drift-1002`).
2. **Racing and greyhounds are structurally better for a hedged trade than any other sport**, because the close is a
   single price with no spread: the Betfair SP. Everywhere else the exit must cross the spread or rest an order and
   hope, and the "close" is the last pre-play trade (`betfair_historic.py`, lines 1-4).
3. **UK greyhounds are next.** 127-259 races a day, a BSP, the first traded price a median 5.7% under the BSP
   (`reports/greyhound_feasibility.md`), form that predicts which dogs shorten (CLV +4.5-7.6% at the first price in
   2026, `reports/greyhound_model.md` section 4), but early money is thin and a traded morning price has not beaten
   the BSP since 2024. The market-only lay-early test (section 6) decides it.
4. **Football is the only non-racing market with the pre-off depth for size** (Premier League about GBP0.4m matched at
   T-60 and GBP0.6m at T-1, Jul-Sep 2025: betaminic). Its pre-play close is an unbiased forecast and the maker side
   earns about zero after fees (Whelan 2025), so a market-only drift is unlikely. The open-to-close move there is
   information (team news) arriving after the open. Research next, not trade.
5. **Tennis, cricket, darts, snooker, golf, US sports, politics: park for a pre-off hedged trade.** Tennis pre-match is
   near-empty on the Italian exchange (median matched EUR0 across 6,477 markets: theoddslab) and about 90% of its
   money arrives in play (trading guide, unverified). Cricket's record volumes are in-play. Politics has no close.
   The census of every sport (section 3.4) replaces these readings with numbers; the table is built to take it.
6. **Costs:** a greened back-then-lay earns `stake x (B/L - 1)` before 2% commission on the market's net profit. Taken at
   both ends with no move it loses one tick: 1% of the stake at 2.0, 2% at 5.0, 3.3% at 15, 5% at 40, 6.7% at 75+
   (Betfair's ladder). The Expert Fee adds 20% of weekly gross profit (less commission paid) once the account's gross
   profit over its last 52 active weeks passes GBP25,000, 40% past GBP100,000. At the live rule's run rate (+GBP4,532
   in 22 days to 31 Oct, TRADING.md) the account reaches GBP25,000 in about four months (arithmetic, not a forecast).

## 1. The ranked table

Ranked for the owner's purpose: a hedged pre-off trade, no in-play reliance, automated. "Open" means the first time a
price can be taken in size; "close" means the BSP where one exists, else the last pre-play trade. Predictability:
(a) from the market alone, (b) only with a model or an information edge, (c) known to be unpredictable or untested.

| # | Market | When a tradable open exists | Typical pre-off matched | Evidence the move is predictable | Expected size of the move | Data held or obtainable | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | UK/IE horse win | From about 06:15 UK on the day; the evening before is unformed (lay/back 21-909, nothing matched: ledger `evening-before-1007`). Morning median GBP539 a runner (UK) | UK GBP11,821 a runner, IE GBP4,743 (`other_markets_and_stakes_1002.md` s.1); 80.4% of pre-race volume in the last 20 min (Brown & Yang 2017, 2013-14 data) | (b) strong: the model carries about half the weight beside the morning price in forecasting the BSP (TRADING.md); features explain the morning-to-off move (ledger `iter3-screen-production-repaired`). (a) back side: none (favourites +0.5%, long shots -24% CLV: ledger `market-only-drift-1002`). (a) lay side on long shots: untested at the book's lay price | +6.15% CLV on GBP47k staked, 19-31 Oct (ledger `staking-1031`); first backs +10.7% on 31 Oct (TRADING.md) | Price files 2017 on, live books from 1 Oct, `betfair_live_marks` at 08:00-12:00 UK and T-120..T-1 (CLAUDE.md) | **Trade now (live).** The one addition to test: the lay-early-back-at-SP side on long shots (section 6.1) |
| 2 | UK greyhound win | The first trade is early but thin: the morning WAP prices 14.7% of runners, morning volume GBP0 to the 75th percentile (`greyhound_model.md` s.4) | GBP1,274 a runner (UK+AU files); pre-play GBP2,213 at the median (GB) | (a) the first traded price is a median 5.7% under the BSP (`greyhound_feasibility.md` s.1): a lay-early edge if the lay side is near it, untested. (b) form predicts which dogs shorten: CLV +4.5-7.6% at the first price, 2026 bundle; but a morning price traded in volume lost 2024-26 (`greyhound_model.md` s.4) | First price to BSP: 5.7% median (direction: drift) | GBGB 2018 on, price files 2018 on, the owner's Historic Data bundle 2026, live GB/IE books from 6 Oct | **Research next.** The market-only lay test on the recorded books, then the paper track's gate (`greyhound_model.md` s.5) |
| 3 | Football match odds, top leagues | Markets open days ahead (unverified); real depth in the last hour: EPL GBP381k at T-60, GBP592k at T-1 (betaminic, Jul-Sep 2025) | EPL GBP0.6m a match at T-1; La Liga GBP359k; Ligue 1 GBP149k (betaminic) | (c) for the market alone: the pre-play close is an unbiased forecast; takers -2.5%, makers +0.6% on 902,568 bets 2022-24 (Whelan 2025). The close beats the open by 0.0034 log-loss and a team model cannot see why (`football_model.md` s.3). Italian-exchange sample: moves over 3% over-shoot (p 0.03-0.04, one month, football+tennis pooled: theoddslab). (b) team news about an hour before kick-off | Close vs open: 0.0034 log-loss on 1X2 (`football_model.md`) | football-data.co.uk open/close 2012 on; Betfair Historic Data soccer once the account holds it; `betfair_market_map` | **Research next.** The market-only drift test on the BASIC files (section 6.3); team-news trade needs API-Football and a fast feed |
| 4 | Football O/U 2.5, Asian handicap | As match odds, thinner (census) | Not held; census | (a) maker side: total-goals makers +2%, takers -2.2% on 2.09m bets (Whelan 2025): a market-making return, not a drift. (b) as match odds | As match odds | As match odds | **Research next, behind 3** |
| 5 | Australian horse win | Overnight UK time; the files' "morning" includes post-off trading, so only the pre-play price is usable (`other_markets_and_stakes_1002.md` s.4) | GBP1,180 a runner pre-play, 81 races a day | (c) untested; Betfair Historic Data would give the first traded price | Unknown | Price files 2018 on (pre-play only); Punting Form needs a key | **Park** behind greyhounds: no model, overnight, thin early |
| 6 | UK/IE horse place, 2/3/4 TBP, each way | Morning place book: best back a quarter to a third under the place SP, best lay 1.5-2x it (ledger `place-clv-forward-1002`) | UK GBP1,825 a runner, IE GBP696; morning GBP17 | (c) closed: neither side of the morning book closes in our favour; the win rule's horses lose 12.5% there (ledger `place-clv-forward-1002`) | n/a | Recorded books continue | **Park (closed 3 Oct)** |
| 7 | Tennis match odds | Pre-match is near-empty: median EUR0 matched across 6,477 pre-match markets, 1.0% over EUR3,000 (Italian exchange, Aug-Oct 2026: theoddslab). UK exchange: census | About 90% of money arrives in play (trading guide, unverified) | (a) favourite-longshot bias on exchange tennis, larger in lower tiers and later rounds (Abinzano et al. 2016, AEL). Open-to-close drift pre-match: untested | Unknown | Sackmann, tennis-data.co.uk, Betfair Historic Data once held | **Park** for pre-off; revisit if the census shows ATP/WTA main-tour depth before the off |
| 8 | Cricket match odds | Toss and teams about 30 min before the start (general knowledge, unverified) | Record volumes: GBP201m matched on one IPL match's match odds, 2025 (Betfair newsletter); the pre-match share is not published | (c) untested; the toss is an information event, not a drift | Unknown | Historic Data once held | **Park**: the edge is in play or at the toss, both outside the brief |
| 9 | Golf outrights, matchups | Outrights open a week ahead; matchups a day ahead (datagolf) | Betfair: unknown (census) | (b) Pinnacle's close is the sharpest; its opening price moves 54.7% of the way to Betcris's where they differ by 5%+ (datagolf, 106,204 matchups) | Unknown on Betfair | None held | **Park** |
| 10 | Darts, snooker | Days ahead (unverified) | Unknown (census) | Untested | Unknown | None | **Park** |
| 11 | US sports | Days ahead | Betfair thin against US books (unverified) | (b) sharp-book line moves lead soft books (general knowledge, unverified) | Unknown | None | **Park** |
| 12 | Politics, specials | Months ahead | GBP317m on the 2024 US election across markets, GBP250m on the winner (oddschecker, 6 Nov 2024); Racing Post reported over GBP400m | No close, news-driven, does not repeat (`all_markets_priorities.md` s.2) | n/a | None | **Park** |
| 13 | Ante-post racing (big races) | Weeks ahead; 2 ANTEPOST_WIN markets on the 3 Oct card (ledger `market-census-1003`) | Thin (unverified); daily snapshots recorded from 3 Oct | (b) the model prices big fields weeks ahead; non-runner risk prices in | Unknown | `betfair_live/<day>/books_antepost.csv.gz` | **Research later**: few bets, non-runner risk |

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

### 3.4 The census: the table to drop in

The data agent is pulling `betfair_live/<day>/census_sports.json` for October 2026. The census reads, once a day at
the same time, every sport with markets starting in the next 24 hours: its market types with counts, and the money
matched so far on its 200 busiest markets, so it compares sports like with like as "the money matched by then, not
each market's final total" (`betfair_recorder.py`, `census_sports`, lines 402-438). Read the evening before, that is
close to the pre-off money that exists a day ahead: the "how early" measure this note needs. Fill this table from it:

| Sport | Market types (count) | Top-200 matched by the evening read, total GBP | Median market GBP | Busiest markets | Days read |
|---|---|---|---|---|---|
| Horse Racing | | | | | |
| Greyhound Racing | | | | | |
| Soccer | | | | | |
| Tennis | | | | | |
| Cricket | | | | | |
| Golf | | | | | |
| Darts | | | | | |
| Snooker | | | | | |
| Basketball, American Football, Baseball, Ice Hockey | | | | | |
| Politics, Special Bets | | | | | |

Two readings to take from it before re-ranking: (i) the ratio of the evening's matched money to the final matched
money, by sport, which is the share that exists a day ahead; (ii) for each sport the market types that carry it.

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
- **(a) Market alone, lay side: the one untested cell.** The long shots' -24% CLV is, read the other way, a lay at the
  morning WAP closed by a back at the BSP. Two things stand between that number and a trade: the WAP is not a price
  one could take (`market_only_drift.py`, docstring), and the morning book's lay side is wide (win books at the
  trader's first step: lay/back 1.16-1.84, ledger `evening-before-1007`; place books: best lay 1.46-2.01x the place
  SP, ledger `place-clv-forward-1002`). Whelan (2025) is the warning: on Betfair football, takers lose more the longer
  the price, and makers earn about zero after fees, which is what a market with a spread wide enough to absorb the
  bias looks like. The test is in section 6.1.
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

- **(a) Market alone:** the first traded price is a median 5.7% worse for a backer than the BSP, and every runner at
  the BSP returns within a few points of zero in every band (`greyhound_feasibility.md` s.1). The pre-off profile
  is a drift from the first price to the BSP on the median dog. Whether a lay near the first price can be had, and
  what the lay side's spread is at that moment, is not known: the bundle has no volume and the price files' pre-play
  WAP is empty (`greyhound_model.md` s.4). Untested; the test is in section 6.2.
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

### 6.1 UK/IE horse win: the lay side of the drift (data agent first, then modelling)

The question: does a lay at the morning book's best lay on long shots, closed by a back at the BSP, earn the drift
the WAPs show (-12% at 21-51, -24% at 51+: ledger `market-only-drift-1002`), after the spread?

- **Data needed:** `betfair_live_marks` (the book at 08:00-12:00 UK and T-120..T-1, with the BSP: CLAUDE.md) from
  1 Oct; the win price files for each race's BSP where the mark lacks it. No new source.
- **The first test (market-only):** for every runner at each morning mark, the best lay price and its size; CLV of a
  lay there closed at the BSP = 1 - lay / BSP (a pound of liability-adjusted stake); by price band of the best lay
  (8-13, 13-21, 21-51, 51-200), by hour, by size offered; settled as Betfair would at 2%. Also the best back for the
  mirror. Report the share of runner-marks where the lay side is within 10% of the BSP. Pre-register the cell
  before reading October: lay at 21-51 at 09:00-11:00 where at least GBP20 is offered. Pass: 90% interval above zero
  on the October days and again on November's first fortnight.
- **Then (modelling):** the model's forecast as the filter (the horses it makes longer than the market, the mirror of
  the live rule), which the closing model already prices (`model/race_book.py` lays). The race-level book is built;
  it has never been run with the lay side live.

### 6.2 UK greyhounds: the first price against the BSP, at the book (data agent)

- **Data needed:** the recorded GB greyhound books from 6 Oct (`betfair_live/<day>/books_greyhound.csv.gz`) with
  each race's BSP from the price files; the 2026 bundle for the first traded price and its time.
- **The first test (market-only):** at the first recorded book and at T-60, T-10 and T-1: best back and best lay
  against the BSP, by price band; the size offered; the lay-early-back-at-SP CLV and the back-early-lay-at-SP CLV.
  The feasibility study's 5.7% median says the first price is low; this says whether the lay side is near it or
  twice it. Pass as 6.1.
- **Then:** the paper track's primary rule (first recorded price, edge > 0.2) continues unchanged to its gate
  (`greyhound_model.md` s.5). The two are read together: if the market-only lay works, the model's job is to pick
  which dogs not to lay.

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

### 6.4 The census (data agent): re-rank 7-11

Fill section 3.4 from October's `census_sports.json` files. For each sport: the matched money a day ahead as a share
of the final, and the market types that carry it. Any sport with a median pre-off market over GBP10,000 a day ahead
and a market type with a natural close joins 6.3's design with its own Historic Data plan.

## 7. What this note does not claim

- No market outside UK/IE horse win has a measured, tradable open-to-close edge in hand. The greyhound and football
  figures above are either model-dependent, from one month, or from WAPs that cannot be taken.
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
