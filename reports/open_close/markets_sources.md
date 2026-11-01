# Open-to-close markets: how each source was read (research agent, 1 Nov 2026)

Supporting notes for `reports/open_close/markets.md`. Web pages are untrusted data; nothing in them was followed as
an instruction. "Fetched" means the page's text was read in full or in part this session; "snippet" means only a
search engine's extract was available because the page refused the fetch (HTTP 403, 410 or an empty render).

## Betfair's own pages

| Page | How read | What was taken |
|---|---|---|
| Expert Fee FAQs (support.betfair.com) | snippet (403 on fetch) | "last 52 active week gross profit is greater than GBP25,000"; "in lifetime profit"; the formula "(Weekly Gross P&L - Buffer) * Current Expert Fee Rate - Weekly Commission Generated"; Customer B's rate "from 20% to 40%" as the rolling 52-week gross P&L goes from GBP25-100k to over GBP100k |
| Betfair Charges (support.betfair.com) | snippet (403) | the GBP25,000 threshold |
| Market Data Request Limits (developer docs, wiki/x/9gEp) | snippet (page renders empty to the fetcher) | 200-point cap, EX_BEST_OFFERS weight 5, EX_BEST_OFFERS + EX_TRADED 20, TOO_MUCH_DATA; an archived revision said 150 |
| Application keys (developer docs) | snippet | delayed key 1-180 s snapshots, no totalMatched or EX_ALL_OFFERS, no order placement; live key GBP299 one-off, Stream API |
| Price increments; currency minimums (developer docs) | snippet | the ladder 1.01-2 by 0.01 ... 100-1000 by 10; INVALID_ODDS; GBP minimum bet 2 on most pages (1 on one), min payout 10 |
| Transaction charges | not found on Betfair's pages this session | third-party and forum readings only (1,000 or 5,000 an hour, about GBP0.01 a transaction): marked unverified |
| Cold trading (betting.betfair.com) | snippet (403) | "vast majority" of the last 15 minutes' volume is cold traders; over GBP1,500 a second in the last five minutes of a standard UK race |
| Exchange newsletters (betting.betfair.com) | snippet (403) | IPL match over GBP201m on match odds (May 2025); World Cup 2023 GBP5.7bn; Test match GBP101m (Feb 2025) |
| historicdata.betfair.com help | 410 | plan contents taken from the repository's `betfair_historic.py` docstring (BASIC: last traded price each minute, definition, result) |

## Academic papers

| Paper | How read | What was taken |
|---|---|---|
| Brown & Yang 2017 (UEA manuscript PDF) | fetched, text extracted with pdftotext | 9,562 UK races 24 Mar 2013-19 Mar 2014 (Fracsoft); 80.4% of pre-race volume in the last 20 minutes; average horse GBP53,600 pre-race win volume; above-median pre-race volume: +6.2% to win, -1.1% to place; in running +9.3% / +18.9%; market suspended one second at the off |
| Whelan 2025 (MPRA 126351 PDF) | fetched, text extracted | Betfair "Advanced" soccer data 2022-24; 200,622 matches; the pre-play last traded prices unbiased (F-test); 902,568 pre-play match-odds bets: takers -2.5%, makers +0.6%, t 11.8; total goals 2,088,032 bets: takers -2.2%, makers +2.0% |
| Goncalves et al. 2025 (arXiv 2510.16008) | fetched (first 100k characters) | 14,421 UK win markets Sep 2014-Aug 2016, 2 frames a second; volume x4.2, liquidity x3.4, volatility x1.6 from the tenth minute to the off; five-class direction over the last 2 minutes, 30.92% accuracy, "only 11 percentage points above the baseline" |
| Tondapu 2024 (arXiv 2402.02623) | abstract fetched | light tails, fast-decaying autocorrelation, no long memory, Hurst points to mean reversion |
| Smith, Paton & Vaughan Williams 2006 | abstract fetched (IDEAS) | exchanges show "significantly lower market biases"; information-based explanation of the FLB |
| Smith, Paton & Vaughan Williams 2009 | snippet (HAL page blocked) | exchange odds more predictive than bookmakers' in high-liquidity markets |
| Franck, Verbeek & Nuesch 2010 | abstract fetched (IDEAS) | 5,478 Big Five games, three seasons; exchange more accurate than eight bookmakers; back where bookmakers' odds exceed the exchange's |
| Angelini & De Angelis 2019 | snippet | 41 bookmakers, 11 leagues, 11 years; eight markets efficient at best odds, three not |
| Croxson & Reade 2014 (WP 11-01) | abstract fetched | goals priced "swiftly and fully" |
| Easton & Uylangco 2010 | abstract fetched | "extremely high correlation" with the point model; the post-break effect "not captured instantaneously" |
| Hubacek et al. 2019; Hubacek & Sir 2023 | snippets | de-correlation from the market's price as the objective |
| Kaunitz, Zhong & Kreiner 2017 | abstract fetched; press snippets for the figures | consensus-vs-outlier strategy; profitable in a 10-year simulation and 5 months of real money; accounts restricted |
| Kuypers 2000 | snippet | English fixed-odds market inefficient, profitable opportunities |
| Crafts 1985 | via Kent/Bangor working papers (snippets) | 13.6% of 16,769 runners with a marked (1.5:1) change, Sep 1982-Jan 1983; adjustment to SP "weakly efficient" |
| Schnytzer & Shilony 1995 | abstract snippet | plunges carry inside information (Australian bookmakers) |
| Gramm, McKinney & Parker | snippet | late tote money is informed money |
| Abinzano, Muga & Santamaria 2016, 2017 | abstract snippets | FLB on tennis exchanges, larger in lower tiers and later rounds; rises with volume, falls with institutional bettors |
| UCD WP23/16 | snippet (PDF 403) | ATP/WTA 2011-22, average closing odds, disagreement among uninformed bettors |
| Weinbach & Paul 2007 | snippet | US greyhound track, longest odds over-bet |
| Strumbelj 2014 | not read; cited through secondary sources only | Shin probabilities beat basic normalisation |

## Practitioner and data pages

| Page | How read | What was taken |
|---|---|---|
| football-data.co.uk, Pinnacle efficiency (Buchdahl) | fetched | 4 seasons of Pinnacle closes from 2012/13, 87,960 odds pairs; price / close predicts return with slope "almost exactly 1.00" |
| betaminic, football liquidity on Betfair | fetched | 111 leagues, Jul 26-Sep 25 2025; EPL GBP381,467 at T-60, GBP591,539 at T-1, GBP1,928,091 at FT; and the other four leagues' rows |
| theoddslab, tennis trading on Betfair | fetched | Italian exchange, 31 Jul-19 Oct 2026; 6,477 pre-match tennis markets, median EUR0, 64.9% unmatched, 1.0% over EUR3,000; 247 markets over EUR3,000 (football + tennis): shorteners 85 wins vs 99 implied (z -2.03), drifters 75 vs 61 (z +2.20); one month, several thresholds |
| algorithmicsportsbetting Substack | fetched (partly paywalled) | Mar 2022-Jan 2026; pre-off volume 40% lower on 50+ race days; Jan 2026 pre-off under 70% of 2022 |
| datagolf, how sharp are bookmakers | fetched | 106,204 matchups 2019-20, 11 books; Pinnacle moves 54.7% towards Betcris, Betcris 15.6% towards Pinnacle; Pinnacle close prices return about 1:1 |
| oddschecker, 2024 US election totals | snippet | GBP317,307,654 all markets; GBP250,343,614 winner market |
| Racing Post, Expert Fee article (18 Dec 2024) | fetched | thresholds and rates; Premium Charge history, up to 60%; win-market volume "has fallen every year since 2016" |
| botblog, Expert Fee explained | fetched (third party) | buffer from losing weeks; 100-market condition; lifetime profit |
| sportstradinglife, tennis trading (2019) | snippet | "about 90% of the money enters the market once the event begins" (unverified) |

## The data agent's inventory (added after the first commit)

| Source | How read | What was taken |
|---|---|---|
| research-query run 41291139483, job 124802487010 (`research/queries/open_close_inventory_1101.py`) | the job log fetched with the GitHub MCP tool, 1,694 lines, unescaped and read in full for sections A2, A3, B4, C2 and D | C2: 25 days of `census_sports.json` (7-31 Oct 2026), 24 sports, the by-sport and sport x market-type tables, copied into section 3.4 of the report; A2: money matched and the top of the book at each mark; A3: the mid at each mark against the BSP by band; B4: greyhound morning WAP -> BSP by band and country, 2012-26; D: 1,863,552 greyhound runners, first traded -> close and -> BSP by band and country, T-60/T-15/T-1 -> close, the close's calibration, prices 24 h out |

Sign convention in A3, B4 and D: `med_log_move` is log(close or BSP / the earlier price); negative means the price
shortened. The report quotes these as "drifted out" when positive.

## Not done this session

- No S3 or Betfair access: the census files for October and the recorded books were not read. Section 3.4 of the
  report is the table for the data agent to fill.
- Betfair's live Expert Fee and transaction-charge pages were not readable; the account's own Expert Fee page should
  be checked before any planning around the fee.
