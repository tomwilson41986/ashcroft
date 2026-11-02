# Football on Betfair: forecasting the early price's move to the close, and trading it hedged

The owner's ask (2 Nov 2026): "can we start modelling a sport and see if we can make a profitable model?" The sport is
football: Betfair MATCH_ODDS, with OVER_UNDER_25 as the second arm, exchange only (every leg on Betfair; bookmaker
prices are information only). The thesis, borrowed from the live horse rule (TRADING.md): forecast where Betfair's
early price is going and trade out at the close. The candidate information edge: Pinnacle leads the exchange, so
where Betfair's early price sits away from Pinnacle's fair price at the same moment, Betfair moves toward Pinnacle by
the close.

Code: `research/open_close/football_exchange.py` (the panel, the models, the scoring, the stale-quote split, the
holdout read), built on `research/open_close/` (features, drift, backtest, panels.football_panel, run.choose_threshold,
reused, not rewritten; `drift.GradientBoosting` gained an optional fixed feature list). Tests
`tests/test_football_exchange.py` (synthetic rows, mechanics only). Figures:
`reports/open_close/football_runs/football_exchange_development.json` and `football_exchange_holdout.json`. The
pre-registration: `reports/preregistration_football_exchange.md` (committed in 27753c6 before the holdout was read;
its SHA-256 is recorded in the holdout JSON). Every figure below is from those two JSON files unless another source is
named. Times are UTC.

## The verdict in five lines

1. **Not profitable at takeable prices.** The early-to-close move is strongly forecastable (R2 against no change up
   to 0.64 on match odds in development, 0.45 on the holdout) and the hedged trade "pays" +50% a bet a tick worse at
   |forecast move| 0.30, but 96-100% of those bets sit in Betfair early books whose back prices sum to over 1.05, and
   there the back side taken at Betfair's own recorded early back loses 8-23% a bet. It is the normalisation of a
   stale quote in an empty book, not a price anyone can take.
2. **In the takeable books (1.05 or under) nothing survives a tick each end.** The Pinnacle gap rule earns -1.6% to
   +1.0% at the mid on match odds and loses 3.6-11.6% a tick worse; LightGBM with every feature is under +2% at best,
   on two or three bets a week, with intervals spanning zero.
3. **Pinnacle adds nothing the market average does not.** On the months Pinnacle's prices were fresh (to 22 Jul
   2025), the market average's gap forecasts the move as well as Pinnacle's (R2 0.686 against 0.693) and LightGBM
   with Pinnacle scores 0.713 against 0.710 without. From 23 Jul 2025 football-data marks Pinnacle's odds stale, and
   from February 2026 they are gone; today's fixtures file has no Pinnacle column. The thesis cannot be traded from
   this source.
4. **The pre-registered rule failed on the holdout** (Aug-1 Nov 2026, read once): over/under 2.5, LightGBM without
   Pinnacle, |move| 0.05 or more, books of 1.05 or under: 39 bets, -1.40% a tick worse (90% -4.27 to +1.57), mean CLV
   +0.65%; the gate fails on bets, days and the lower bound. The secondary match-odds read was +9.21% (+3.72 to
   +13.99) on 19 bets, 17 of them lays: too few to mean anything.
5. **The top five leagues have no stale quotes and no edge.** Their 1X2 books are 95.8% tight (median 1.0077) and
   0.5% wide; the rest are 27.4% wide (median 1.0302). In the top five the models lose 6.4-7.3% a tick worse at
   |move| 0.05. The only test left is a paper trial on recorded books with sizes (section 9).

## 1. Data provenance

- **Source.** football-data.co.uk, fetched in this session on 2 Nov 2026 (`python source_data.py --source football
  --fetch --build --from 2012 --root <scratch>` then `python -m football.data --build --root <same>`), exactly as
  reports/open_close/model.md section 5 did on 1 Nov. Fetch: 348 files fetched, 0 missing, 0 skipped; silver table
  173,913 matches from 346 files, 38 leagues, 2 Mar 2012 to 1 Nov 2026, 27,030 with a Betfair close. Gold build run
  2026-11-02T1121Z: 173,929 matches (173,909 finished, 16 scheduled), 0 changed; data-quality flags: 4 played rows
  with no result (NO_RESULT), 432 opening books over 1.15, 1 scheduled match in the past, 8 season match-count gaps
  (6 low, the COVID-curtailed seasons; 2 medium). None of these touches the panel's rows beyond the NO_RESULT four,
  which are dropped as unfinished.
- **What the columns are** (football-data.co.uk notes.txt, fetched 2 Nov 2026): "BFEH = Betfair Exchange home win
  odds" (and BFED, BFEA); "PSH and PH = Pinnacle home win odds"; "P>2.5 = Pinnacle over 2.5 goals"; "MaxH = Market
  maximum home win odds", "AvgH = Market average home win odds"; "B365H = Bet365 home win odds"; the key "is for
  pre-closing odds. For the closing odds, as below but with an additional "C" character following the bookmaker
  abbreviation/Max/Avg". The notes do not list BFE>2.5 or BFEC*; they follow the same pattern and are in the files.
- **When the early odds are collected.** notes.txt: "Betting odds for weekend games are collected Friday afternoons,
  and on Tuesday afternoons for midweek games." matches.php: "the odds are collected for the downloadable weekend
  fixtures on Fridays afternoons generally not later than 17:00 British Standard Time. Odds for midweek fixtures are
  collected Tuesdays not later than 13:00 British Standard Time." data.php: "Since 2019/20 I have collected two sets
  of odds. The first set is collected after market opening at times specified on my fixtures page. The second set
  are the closing odds." No time stamp or volume is kept per match.
- **Pinnacle.** data.php: "Since 23/07/2025 Pinnacle's public API for odds delivery has become unreliable meaning
  their odds are systematically out of date relative to odds for other bookmakers, including both the pre-closing
  and closing odds. Consequently they should be used with caution when undertaking any betting analyses, and are no
  longer being included for the calculation of market average and maximum odds." In the panel, PSH is present on
  99-100% of matches to September 2025, 94% in October 2025, 20-48% from November 2025 to January 2026, and 0% from
  February 2026 on. Today's `fixtures.csv` (2 Nov 2026, 196 rows) has no PS* or P>2.5 column; BFEH, AvgH, MaxH and
  B365H are filled on every row and BFE>2.5 on 95%.
- **What BFE is.** Not stated by football-data. The early BFE books sum to more than one (match odds median 1.0202,
  10th-90th percentile 1.0054-1.0971, 99th 2.5595; the close BFEC median 1.0076, 1.0033-1.0166) and every price is on
  Betfair's ladder (100% of opens and closes): that is a book of best back prices, not lay prices (which sum under
  one) or last trades (which scatter about one). This report reads BFE as the best back at the collection.
- **The panel** (`exchange_panel`, from 1 Jul 2024): 28,499 gold matches (28,482 finished), of which 17,534 in the 22
  main divisions (the 10,948 in the 16 extra leagues carry Betfair's close only, no early price). 16,929 of those
  have every 1X2 leg priced at both ends and 16,409 every over/under side (`panels.football_panel` keeps only
  complete books). **83,605 selections in 33,338 markets**, 26 Jul 2024 to 1 Nov 2026: MATCH_ODDS 50,787 rows
  (16,929 markets), OVER_UNDER_25 32,818 (16,409). Duplicate (market, selection) rows: 0. Rows dropped by the
  cleaning (prices off 1.01-1000, markets of fewer than two rows): 0. Nulls: Pinnacle's early price 38.1% (absent
  from February 2026, as above), the market average 0.001%, Bet365 0.09%, the Dixon-Coles probability 0%. Pinnacle
  period of the rows: fresh (before 23 Jul 2025) 37,465, stale 14,316, absent 31,824. Book width (the Betfair early
  back book): match odds tight (under 1.02) 24,966, medium (1.02-1.05) 15,153, wide (over 1.05) 10,668; over/under
  11,964, 12,136 and 8,718. Markets with an early book over 1.25 (781 match odds, 697 over/under in development) are
  kept, in the wide group: the split is the test of them.
- **Windows.** Development: months 2024-07 to 2026-07, the first six fit only, scored 2025-01 to 2026-07 (match odds
  33,063 selections, 11,021 markets, 390 days; over/under 21,374, 10,687, 384 days). Holdout: 2026-08 to 2026-11 (1
  Nov), read once after the pre-registration was committed (match odds 6,531 selections, 2,177 markets, 77 days;
  over/under 4,274, 2,137, 74 days).

## 2. Method

- **Target.** y = log(close / early) per selection, on fair prices: each market's early and closing books scaled to
  sum to one (features.fair_prices; section 5 of model.md showed the margin's shrinkage otherwise reads as drift).
  The raw prices are kept for the as-quoted check.
- **Features, all from the early collection.** Betfair's own early book (log fair price, share, rank, gap to the
  favourite, the book's overround before normalising); Pinnacle's early price (PS* for 1X2, P>2.5 and P<2.5 for the
  goals; never PSC* or PC*, the closes) and its book; the market average and maximum and Bet365 at the same
  collection and their books; the key feature, the log gap between Betfair's early fair price and each bookmaker's
  early fair price, both normalised (`gap_ps` = log(Betfair fair / Pinnacle fair): positive where Betfair is longer,
  so the thesis says it shortens and the trade backs), and log(Betfair's back / the market maximum); the Dixon-Coles
  probability (`football.model.walk_forward`, each match priced from ratings fitted on the matches before its week
  only) and its gap to Betfair's fair price; league, top-five flag, month of the year, day of the week, kick-off
  hour, the days from the collection to the match implied by the collection rule (Friday for Friday to Monday,
  Tuesday for Tuesday to Thursday), and the season's stage (the share of the competition-season's matches on earlier
  days).
- **Models.** No change; the training mean (the floor a permuted target still earns); the pre-registered simple rule,
  y_hat = -gap (the close goes to the bookmaker's fair price; nothing fitted), for Pinnacle and for the market
  average; the gap-only least-squares line (`gap_ols_*`); LightGBM on Betfair's own book only (`gbm_base`, section 5's
  arm); LightGBM on everything except Pinnacle (`gbm_nopin`); LightGBM on everything (`gbm_all`). LightGBM's settings
  are section 5's, fixed in advance (400 trees, rate 0.03, 31 leaves, 100 rows a leaf, 80% sampling), no early
  stopping on scored data. Each market type is fitted on its own.
- **Walk-forward by month.** Every scored month is forecast by models fitted on every month before it only
  (drift.walk_forward, refit monthly); the holdout continues the walk (each holdout month fitted on everything
  before it, the earlier holdout months included), as a live trader would refit.
- **The trade.** GBP1 a selection: forecast shorter than the threshold, back at the early price and lay at the
  close; forecast longer, lay early and back at the close; greened, so the result does not matter. 2% commission on
  each market's net. **Headline: a tick worse at each end** on Betfair's ladder (trading/exchange.py), a back one tick
  under the fair price and a lay one tick over, at both ends. 90% bootstrap by day (2,000 draws), bets a day, worst
  drawdown, right-way share (the close moved the forecast way), mean CLV at the taken price (a back o/c - 1, a lay
  1 - o/c, o the tick-worse fill, c the fair close). Thresholds on |forecast move| 0, 0.02, 0.05, 0.10, 0.15, 0.20,
  0.30; each model's development threshold by `run.choose_threshold` (the best lower 90% bound a tick worse among
  thresholds trading at least once a betting day).
- **The as-quoted check.** For the back side only: back at Betfair's recorded early back price itself (no
  normalising), lay at one tick over its recorded closing back (the best a lay can be when the close spread is a
  tick). The lay side has no recorded lay price and cannot be checked this way.
- **The stale-quote split.** Every result by the Betfair early book's overround (tight under 1.02, medium 1.02-1.05,
  wide over 1.05), by price band, by league tier (top five: E0, D1, I1, SP1, F1; the rest) and by Pinnacle period.

## 3. Development: match odds (2025-01 to 2026-07, 33,063 selections, 390 days)

Accuracy (R2 of y against no change; the training mean's floor is 0.0041):

| model | R2 vs no change | direction hit |
|---|---|---|
| gap_rule_ps (Pinnacle gap, nothing fitted) | 0.4599 | 56.2% |
| gap_ols_ps | 0.4641 | 55.0% |
| gap_rule_avg (market average gap) | 0.5611 | 54.0% |
| gap_ols_avg | 0.5739 | 53.7% |
| gbm_base (Betfair's book only) | 0.2790 | 56.1% |
| gbm_nopin | 0.6392 | 59.5% |
| gbm_all | 0.6395 | 59.6% |

The hedged trade, a tick worse at each end (90%), with the mid beside it:

| threshold | gap_rule_ps: bets/day; a tick worse; mid | gbm_nopin: bets/day; a tick worse; mid | gbm_all: bets/day; a tick worse |
|---|---|---|---|
| 0 | 75.0; -0.98% (-1.51 to -0.39); +3.28% | 84.8; -0.74% (-1.15 to -0.30); +3.49% | 84.8; -0.71% (-1.14 to -0.27) |
| 0.05 | 16.4; +9.52% (+7.40 to +11.56); +15.22% | 14.5; +15.02% (+13.07 to +16.82); +20.77% | 14.1; +15.01% (+13.08 to +16.79) |
| 0.10 | 10.1; +22.76% (+19.83 to +25.31); +28.60% | 10.2; +26.87% (+24.71 to +28.76); +32.32% | 10.3; +27.40% (+25.35 to +29.25) |
| 0.30 (chosen) | 8.3; +51.22% (+49.35 to +52.87); +56.30% | 8.2; +51.71% (+50.11 to +53.22); +56.57% | 7.9; +52.26% (+50.77 to +53.66) |

At 0.30 the right-way share of the gap rules and the full models is 97.5-99.1% and their mean CLV at the taken price
+54.2% to +55.6%: numbers no real market gives. Section 5 says where they come from.

## 4. Development: over/under 2.5 (21,374 selections, 384 days)

R2 against no change: gap_rule_ps 0.1925, gap_rule_avg 0.3637, gap_ols_avg 0.3685, gbm_base 0.1584, gbm_nopin
0.3666, gbm_all 0.3683 (floor 0.0010). The trade a tick worse: gap_rule_ps at 0 -0.72% (-0.92 to -0.51), at 0.05
+11.14% (+10.08 to +12.24) on 6.8 a day, at 0.30 (chosen) +34.13% on 76 bets; gbm_nopin at 0 -0.57%, at 0.05 +11.71%
(+10.78 to +12.63), at 0.30 +30.73% on 144 bets, 100% right way. The same shape as match odds, smaller.

## 5. The decisive diagnostic: the stale-quote test

**Development, match odds, each model's chosen 0.30 by the early book's width (a tick worse each end):**

| model | tight <1.02 | medium 1.02-1.05 | wide >1.05 | back side as quoted (all widths) |
|---|---|---|---|---|
| gap_rule_ps | 19 bets, -17.34% | 9 bets, -4.26% | 782 bets, +53.52% (99.7% right) | 437 bets, -11.86% (-13.40 to -10.33) |
| gap_rule_avg | 29 bets, -6.51% | 9 bets, +3.12% | 1,027 bets, +52.78% | 568 bets, -12.29% |
| gbm_nopin | 9 bets, -2.42% | 14 bets, +20.38% | 1,033 bets, +52.61% | 518 bets, -13.18% |
| gbm_all | 9 bets, -4.27% | 12 bets, +24.82% | 1,018 bets, +53.08% | 512 bets, -12.87% |

**The whole sweep inside each width** (development, a tick worse each end, mid in brackets):

| match odds | tight <1.02 | medium 1.02-1.05 | wide >1.05 |
|---|---|---|---|
| gap_rule_ps, 0 | 10,560 bets: -4.22% (mid +0.16%) | 5,553: -3.64% (+0.48%) | 4,278: +10.48% (+14.60%); as quoted -9.95% |
| gap_rule_ps, 0.05 | 1,243: -7.25% (+0.22%) | 744: -4.84% (+0.96%) | 1,922: +25.92% (+30.44%); as quoted -13.99% |
| gap_rule_ps, 0.10 | 411: -10.26% (-0.99%) | 164: -6.43% (+0.63%) | 1,353: +36.33% (+40.98%); as quoted -14.71% |
| gap_rule_avg, 0.05 | 2,457: -7.29% (+0.09%) | 1,595: -6.41% (-0.27%) | 2,941: +22.02% (+26.58%); as quoted -13.74% |
| gbm_nopin, 0.05 | 814: -4.71% (+3.74%) | 740: +0.34% (+7.28%) | 2,733: +24.87% (+29.50%); as quoted -15.05% |
| gbm_nopin, 0.10 | 239: -3.89% (+5.66%) | 235: +2.95% (+0.55 to +5.46) (+10.79%) | 1,975: +33.43%; as quoted -15.99% |
| gbm_all, 0.10 | 220: -3.55% (+5.87%) | 229: +3.53% (+1.06 to +5.81) (+11.29%) | 1,960: +33.67%; as quoted -16.03% |

| over/under 2.5 | tight <1.02 | medium 1.02-1.05 | wide >1.05 |
|---|---|---|---|
| gap_rule_ps, 0 | 4,970: -1.57% (+0.61%) | 4,638: -1.84% (+0.33%) | 3,402: +2.07% (+4.21%); as quoted -7.70% |
| gap_rule_ps, 0.02 | 824: -1.02% (+1.41%) | 751: -1.46% (+0.91%) | 1,500: +6.22%; as quoted -9.37% |
| gap_rule_ps, 0.05 | 69: -1.84% | 65: +2.39% (-0.26 to +4.67) | 680: +13.30%; as quoted -12.12% |
| gbm_nopin, 0.05 | 79: -0.88% | 145: +1.21% (-0.19 to +2.51) | 1,314: +13.62%; as quoted -14.19% |

**The holdout says the same** (Aug-1 Nov 2026, a tick worse each end): match odds gbm_nopin at 0.30, 96 of 98 bets in
wide books (+46.20%), back side as quoted -11.05% (-16.54 to -6.94); inside the widths at 0.05: tight 116 bets
-5.79% (-8.36 to -2.98), medium 127 bets -1.98% (-4.39 to +0.01), wide 392 bets +15.74% with the back side as quoted
-10.19%. The market-average gap rule on match odds: tight 0.05 -8.19%, medium -7.94%, wide +13.31% (as quoted
-10.35%). Over/under gbm_nopin at 0.05: tight 16 bets -6.87%, medium 23 bets +2.40% (-1.35 to +6.34), wide 293 bets
+16.02%, as quoted -15.04%.

**Reading.** Every large return lives in the wide books, 95-100% "right", and there the back side at the price
Betfair actually showed loses 8-23% a bet. A wide back book is one where some selection's back side is empty or
stale, so its back price is far short of fair; normalising that book lengthens every other selection's "fair" price,
which then "shortens" to the close as the book fills. The gap to any bookmaker (Pinnacle, the average, Bet365)
detects the distortion because the bookmakers' books are coherent; that is why the average's gap works as well as
Pinnacle's. **The edge is a stale quote in an empty book and is not takeable.** In the tight and medium books the
fair-mid returns are 0-1% for the gap rules and 1-11% for LightGBM at its higher thresholds, which a tick each end
(1-3% of the price on 2-8 shots) takes away; the only cells left positive a tick worse (LightGBM in medium
match-odds books at 0.10-0.15, 1.4-2 bets a day) are 88-98% lays, whose lay price is not recorded, and their halves
disagree (section 8). Price band adds nothing the width does not: at 0.30, where 96-100% of the bets are in wide books, every band from
1-2 to 8-13 shows +47% to +60% a tick worse for gbm_nopin, and only the long shots of 21-51 lose.

## 6. What Pinnacle adds

On the development months with Pinnacle fresh (2025-01 to 2025-06, 11,697 match-odds selections): R2 gap_rule_ps
0.6932, gap_rule_avg 0.6863, gbm_nopin 0.7101, gbm_all 0.7128; at 0.05 a tick worse gbm_nopin +18.58% and gbm_all
+18.05% (all widths). With Pinnacle stale (2025-07 to 2026-01, 8,694): gap_rule_ps 0.4659, gap_rule_avg 0.4418,
gbm_all 0.5687, gbm_nopin 0.5668. Over/under, fresh: gap_rule_ps 0.2811, gap_rule_avg 0.2699, gbm_all 0.2441,
gbm_nopin 0.2343. LightGBM's gain on every development month (match odds): gap_avg 5,824, gap_b365 2,026, gap_ps 604,
dc_p 488, gap_max 379, the book's overround 255, league 212. Pinnacle's information is already in the market
average and Bet365: it adds 0.003 of R2 and nothing a tick worse. On the gap rule over every 2024/25 month with
Pinnacle fresh (22,863 rows, nothing fitted), in tight books: at 0 -4.31%, at 0.05 -7.79%, at 0.10 -10.21% a tick
worse; in wide books +9.48%, +25.03%, +35.63%. Where the book is takeable, Betfair does not move toward Pinnacle any
more than toward a tick of noise.

The Dixon-Coles probability carries some gain (dc_p 488) but its value is the same distortion read another way
(a coherent price beside an incoherent book); reports/football_model.md already found the team model behind the
opening price everywhere.

## 7. League tier

Development, the share of markets by the Betfair early book's width: match odds, top five 95.8% tight, 3.7% medium,
0.5% wide (median book 1.0077); the rest 34.0%, 38.7%, 27.4% (median 1.0302). Over/under: top five 84.5% tight,
3.8% wide (median 1.0128); the rest 21.7% tight, 33.3% wide (1.0387). The trade a tick worse, development, books of
1.05 or under:

| | top five | the rest |
|---|---|---|
| match odds, gap_rule_avg 0.05 | 1,172 bets, -7.31% (mid +0.52%) | 2,880 bets, -6.80% (mid -0.28%) |
| match odds, gbm_nopin 0.05 | 378 bets, -6.41% (mid +2.53%) | 1,176 bets, -0.99% (mid +6.36%) |
| match odds, gbm_nopin 0 | 7,683 bets, -3.97% | 18,405 bets, -2.90% |
| over/under, gbm_nopin 0.05 | 76 bets, +0.05% (-1.73 to +1.81) | 148 bets, +0.69% (-0.60 to +2.00) |

The wide books, and so the apparent edge, are a lower-league phenomenon (English League One and Two and the
National League, Scotland's lower three, the second divisions): 2,719 of the 2,733 wide-book gbm_nopin match-odds
bets at 0.05 were outside the top five. Where Betfair's early book is full, the top five, there is nothing.

## 8. The leak test, and the halves

The walk-forward with the training closes permuted (refit every 6 months): the LightGBMs fall to R2 against no change
-7.79 to -7.96 on match odds and -4.98 to -5.17 on over/under (direction hit 47.9-48.6%), far under no change: no
fitted model reads the answer. The gap rules fit nothing, so the permutation cannot touch them (they score their
unpermuted R2, 0.46 and 0.56), and the least-squares lines keep most of theirs (0.46 and 0.56) because, under a
permuted close, y still contains -log(early), which the gap tracks. Their power is a property of the same moment's
prices, not a leak; the timing question (is football-data's bookmaker snapshot taken later than its Betfair one?)
cannot be tested in these files, but the result does not need it: the edge sits in the books whose own back price
loses as quoted. No row is scored by a model that saw it (walk_forward asserts fit_month < month; tested), and the
Dixon-Coles probability of each match comes from ratings fitted only on matches before its week.

The halves (development, books of 1.05 or under, a tick worse): match odds gbm_nopin at 0.15, Jan-Oct 2025 +3.53%
(+0.22 to +6.79) on 136 bets, Nov 2025-Jul 2026 -1.81% (-5.86 to +3.02) on 73; at 0.10, +0.40% then -1.73%.
Over/under gbm_nopin at 0.05: +0.34% (139 bets) then +0.68% (85). Nothing stable.

## 9. The pre-registered holdout (read once)

The rule (reports/preregistration_football_exchange.md, committed 27753c6 before the holdout was computed): over/under
2.5, every league with Betfair early prices, books of 1.05 or under, `gbm_nopin`, |forecast move| over 0.05. It was
the best development lower bound among rules that need no Pinnacle price and have 200 or more development bets:
224 bets, +0.47% a tick worse (-0.60 to +1.56). The expectation written in advance: it would not pass.

| holdout, Aug-1 Nov 2026 | bets | a tick worse each end (90%) | at the mid | mean CLV | right way | gate |
|---|---|---|---|---|---|---|
| **primary: O/U 2.5, gbm_nopin, 0.05, books <=1.05** | 39 (1.9 a betting day) | **-1.40% (-4.27 to +1.57)** | +2.36% | +0.65% | 61.5% | fails: bets 39 < 300, days < 90, lower bound < 0 |
| secondary: match odds, gbm_nopin, 0.15, books <=1.05 | 19 | +9.21% (+3.72 to +13.99) | +17.91% | +13.74% | 94.7% | fails: bets and days |

The primary by width: tight 16 bets -6.87%, medium 23 bets +2.40% (-1.35 to +6.34); by tier: top five 14 bets
-4.61%, the rest 25 bets +0.39%; by month +7.98% (8 bets), -2.06% (12), -5.28% (14), -4.00% (5). The back side as
quoted: 9 bets, +0.68%. The secondary's 19 bets were 17 lays on 5-51 shots at 95% right way, 14 of them in medium
books: the pattern of section 5 one step under the 1.05 line, and too few to read. An erratum, not a change of rule:
the pre-registration describes the development rate as "about one bet a week a market"; it is about three (224 bets
in 19 months, the "about 12 rule trades a month" it also states). The rule and the gate are unaffected; the holdout
JSON records the pre-registration's SHA-256 (f8c7795e...), the same as the committed file. Holdout accuracy: R2 against no change
match odds gbm_nopin 0.4520, gap_rule_avg 0.2685; over/under gbm_nopin 0.4493, gap_rule_avg 0.4162; gap_rule_ps 0
(Pinnacle absent from every holdout row).

## 10. Verdict, and what a paper trial needs

**Not profitable at takeable prices, on development or on the holdout.** The forecastable move from football-data's
Betfair early snapshot to Betfair's close is the repair of stale quotes in empty lower-league books; at the back
price Betfair actually showed it loses 8-23% a bet, and where the book is full (the top five, any book of 1.05 or
under) a tick each end takes every bet's edge. The Pinnacle-leads thesis gives nothing beyond the market average,
and the source that held Pinnacle's early price no longer does. Kelly or fractional staking applies nowhere.

What a paper trial needs, if the owner wants the residual question closed (the pre-registration sets it out):

- **Prices from the coordinator's Betfair football book recorder, with sizes, never from football-data:** best back
  and best lay at football-data's collection moment (Friday by 17:00 UK, Tuesday by 13:00 UK), the fill only for the
  size shown, the close at the last two-sided book before the market turns in play (the opposite side's real price,
  not a tick estimate). That measures what this report cannot: the lay side's price and whether a medium book's
  quote is there to be taken.
- **Features from football-data's fixtures file of that day** (Avg, Max, B365, BFE are all present on 2 Nov 2026; no
  Pinnacle) and the week's Dixon-Coles fit; the book-width filter on the recorder's own book.
- **Gate:** at least 300 rule trades and 90 days, whichever later, capped at 12 months; pass only if the 90% lower
  bound of the return at the taken prices after 2% is above zero, the mean CLV at the taken price is above zero, and
  both halves are positive. At the development rate (about 12 rule trades a month a market) 300 trades is about two
  seasons; the expectation is that it fails.
- **More useful than the trial:** the recorder's own early books answer the stale-quote question directly (how often
  is a lower-league book over 1.05 at the collection, how much is offered at the best prices, how far the lay sits
  from the back). If wide books are nearly empty on the recorder, the question is closed without waiting for 300
  trades.

Hyphens throughout; every figure names its source above. Run on 4 cores, locally, 2 Nov 2026 (the development step's
model and scoring time 303 s, as its JSON records).
