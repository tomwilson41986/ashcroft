# Open-to-close drift: the model and what it is worth, traded hedged

The owner's ask (1 Nov 2026): trade out from opening prices to closing prices on Betfair, hedged, without relying
on the result. This note is the modelling agent's answer: a model of the move from the open to the close built on
leading practice, scored against the no-change baseline and as a hedged trade after commission and the spread, on
every market the repository reaches today. The research agent's survey of the markets is reports/open_close/markets.md,
the data agent's inventory reports/open_close/data.md; the data agent's standard panel is the input this code is
written against (its schema is research/open_close/features.py, PANEL_COLUMNS).

Code: `research/open_close/` (features, drift, backtest, panels, run), tests `tests/test_open_close_model.py`
(synthetic data for the mechanics only, never a figure here), the CI query `research/queries/done/open_close_horses_1101.py`
(research-query run 41292736133, 21 minutes). Hyphens throughout; every figure names its source.

## The verdict in five lines

1. **Nothing market-only is tradable on takeable prices.** On the horses the forecast of the move is real (R2 against
   no change 0.10-0.24 out of sample, 59-61% of selections move the forecast way) but the hedged trade only pays at
   the morning volume-weighted average price, which nobody is offered: a tick worse at each end turns +6% into -3% in
   development and +3% into -6.5% on the April-October 2026 holdout. On October's recorded books, the pre-registered
   lay of 21-51 shots at the best lay offered returned -0.62% on liability (90% interval -0.99 to -0.31): every one of
   the 24 cells is negative.
2. **The repository's earlier reading stands** (reports/other_markets_and_stakes_1002.md section 4: no drift to catch
   at the morning price). The apparent gain here comes from laying the long shots the band model expects to drift, at
   a WAP that sits under the offered lay price; the same -24% CLV the back side showed in 2023-26, read the other way
   and uncollectable for the same reason.
3. **The model's forecast adds to the market's** where it exists (UK R2 0.33 against 0.25 market-only on the rows with
   a forecast), which is the live rule's edge restated (TRADING.md); it is not a market-only drift and it has not been
   scored on the holdout months here (the out-of-sample forecasts end 21 Mar 2026).
4. **The greyhounds' first traded price is not a price.** The move from the first print to the BSP is forecastable
   (R2 0.21) and "pays" +6% at threshold 0 even a tick worse at each end, but the money sits in dogs first traded under
   2.0 that drift (lays, 93% right) and over 50 that shorten (backs, +350%): the first print in an empty book, with no
   volume in the BASIC files. The recorded books at T-60 and T-15, with sizes, are the test.
5. **Football: no market-only drift in match odds or over/under** on the bookmaker proxy (R2 against no change under
   0.01, -2% to -4% after a tick each end) once the bookmaker's margin is taken out; the handicap's apparent 0.3 is
   look-ahead by selection and is withdrawn. On Betfair's own early snapshot one cell is positive after a tick on both
   windows (LightGBM, |move| of 0.10 or more: +7.8% then +3.4%, 5-7 bets a day), on quotes too thin to call a price;
   it goes to the live football recorder's books for a test, not to trading.

Kelly or fractional sizing is applied nowhere: it waits on an edge surviving the spread, and none has.

## 1. Method

- **Target.** y = log(close / open) per selection; and y_share, the change in the selection's implied-probability
  share of its market, which a coherent book must forecast with the shares summing to zero (features.add_targets).
- **Features, opening book only.** log open, implied probability, the book's overround at the open, rank in the
  market, gap to the favourite, field size, hours to the off at the open (where the open's time is held), log open
  matched (where held), sport, country and market-type categoricals, day of week, hour of the off; in the "with
  model" arm the model's forecast against the open (log(open / model) and the share difference). Nothing from after
  the open (features.add_features).
- **Models** (drift.py): (a) no change, close = open, the efficient-market null; (b) OLS of y on log open within each
  price band of the open, the favourite-longshot unwind; (c) LightGBM on the features, fixed settings chosen in
  advance (400 trees, learning rate 0.03, 31 leaves, 100 rows a leaf, 80% row and column sampling), no early
  stopping on scored data; (d) a conditional logit on the closing shares within the market (the form of
  `model/race_book.py`'s ClosingModel, its helpers reused), the close's overround a log-linear function of the
  open's fitted on the training markets, so the forecast closes form a book; and (c) made coherent the same way
  ("gbm_book"). The "with model" arm adds the repository's out-of-sample BSP forecast (data/oos_predictions.csv) to
  (c) and (d) where a row has one.
- **Walk-forward by month.** Every scored month is forecast by models fitted on the months before it only; the first
  12 months (6 for the short Betfair-only football panel, 2 for the greyhounds' nine months) fit only. No row is
  ever scored by a model that saw it (drift.walk_forward; tested). The long football panel refits every third month
  (the scored month always after the fit's last), the rest every month.
- **Leak test.** The same walk-forward with the training targets permuted (refit every 6 months). A permutation keeps
  the marginal distribution, so a model still earns the constant-shift component (the mean of y, non-zero where the
  open's book and the close's differ in overround); the figure to read is each model's R2 above that floor
  (drift.shuffle_test; from commit 29f0b9d the closes themselves are permuted, which the CI run predates).
- **Scoring** (backtest.py): RMSE and R2 of y against no change (1 - SSE / SSE of the zero forecast) and the plain R2;
  the direction hit rate; the RMSE of y_share from the forecast prices made into shares.
- **The hedged trade.** For each selection whose forecast |move| clears a threshold: forecast shorter, back GBP1 at
  the open and lay open / close at the close; forecast longer, lay GBP1 at the open and back at the close; the
  profit is open / close - 1 (or 1 - open / close) whatever the result (race_book.close_payoffs). Commission as the
  exchange charges it, 2% on each market's net (the owner's rate; research/queries/done/staking_backtest.py's
  convention), 5% as the sensitivity. The spread: the same trades a tick worse at each end on Betfair's ladder
  (trading/exchange.py: 0.01 to 2, 0.02 to 3, 0.05 to 4, 0.1 to 6, 0.2 to 10, 0.5 to 20, 1 to 30, 2 to 50, 5 to
  100, 10 to 1000), a back one full tick under and a lay one full tick over, at the open and again at the close.
  One tick is a floor on the cost, not the cost: the morning win book's relative spread is 18.9% at 08:00
  (data.md, betfair_live_marks). Reported: ROI on the GBP1 units, a 90% bootstrap interval by day
  (race_book.ratio_interval on the days' sums, 2,000 draws), bets a day, the worst drawdown of the cumulative daily
  P&L, the losing-day share, the CLV-style share of trades whose close moved the forecast way; the threshold swept
  over 0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30; by price band of the open and by market.
- **Development and holdout.** The last three months of each panel (for the horses, April 2026 on: the repository's
  locked holdout) are read once, at the end, with models fitted on the development months and the threshold each
  model chose there (the best lower 90% bound of the ROI a tick worse at each end, among thresholds trading at
  least once a day). Nothing is tuned on the holdout.
- **Practice followed.** Walk-forward (temporal) validation and no evaluation on training rows
  (RESEARCH_FRAMEWORK.md section 7); block bootstrap by day for intervals (section 5, uncertainty.py); the
  closing line as the benchmark (section 11); a permutation test for leakage; a pre-registered holdout read once;
  the exchange's commission on the market's net; the tick ladder for the spread; fractional Kelly only after the
  edge (STAKING_REPORT.md); a pre-registered cell on the lay side (markets.md 6.1) scored exactly as registered.

## 2. UK and Irish horse win: the morning price to the BSP (research-query run 41292736133)

**Data.** betfair_prices, win markets, 1 Jan 2021 to 30 Oct 2026: 713,270 rows (UK 512,632, IE 200,638), 99.9%
joined to a result. The open is MORNINGWAP, Betfair's volume-weighted average of the morning's trades (no time
stamp; the morning volume is open_matched); the close is the BSP. Panels after cleaning: UK 510,698 selections in
59,150 markets; IE 184,592 in 17,024. The model's forecast (data/oos_predictions.csv, 215,760 rows, 31 May 2024 to
21 Mar 2026) joins 155,215 UK rows and 54,295 IE rows; no predictions table or feature cache holds an earlier one,
so the "with model" arm has no forecast in the holdout months. Development 2022-01 to 2026-03 (2021 fits only);
holdout 2026-04 to 2026-10, read once.

**The morning WAP is not a takeable price.** It averages trades on both sides of a book whose relative spread is
18.9% at 08:00 (data.md). A back at the WAP is optimistic by part of that spread, a lay by the rest. The columns
"a tick worse each end" are the minimum correction; the October record (section 4) is the real one.

### 2.1 Accuracy of the forecast (log BSP / morning WAP)

UK, development, 365,296 selections, 42,436 markets, 1,531 days:

| model | RMSE | R2 vs no-change | direction hit | the same on the 155,215 rows with a forecast: R2 vs no-change |
|---|---|---|---|---|
| no change | 0.5253 | 0 | - | 0 |
| band OLS | 0.4718 | 0.194 | 61.0% | 0.195 |
| LightGBM | 0.4573 | 0.242 | 61.2% | 0.250 |
| conditional logit | 0.4903 | 0.129 | 60.9% | 0.140 |
| LightGBM, coherent book | 0.4611 | 0.230 | 61.3% | 0.239 |
| LightGBM with the model's forecast | 0.4481 | 0.272 | 61.4% | 0.325 |
| conditional logit with the model's forecast | 0.4830 | 0.155 | 60.3% | 0.170 |
| leak test (targets permuted): band OLS / LightGBM | 0.4962 / 0.4966 | 0.108 / 0.107 | 61.0% | - |

The permuted floor of 0.11 is the constant shift (the WAP book and the BSP book differ); the models' 0.19-0.27 is
the per-selection signal above it. The model's forecast adds 0.03 overall and 0.08 where it exists, the live
rule's mechanism (TRADING.md: the forecast carries about half the weight in the best forecast of the BSP).

UK, holdout (April-October 2026, 54,536 selections, 6,360 markets, 213 days), read once: no change RMSE 0.4274;
band OLS R2 vs no-change 0.024; LightGBM 0.096; conditional logit 0.091; coherent LightGBM 0.111; with the model's
forecast (none in these months) 0.084 and 0.094; direction hit 57.8-59.1%. The signal is a third to a half of the
development window's: 2026's morning market is further from its past than the fit allows for.

IE, development (127,198 selections, 12,012 markets, 1,242 days): no change RMSE 0.7164; band OLS R2 0.223,
LightGBM 0.313, coherent 0.285, with the forecast 0.379 (0.482 on the 54,295 rows with one); permuted floor 0.15-0.16.
IE holdout (23,127 selections, 1,991 markets, 192 days): band OLS 0.015, LightGBM 0.156, coherent 0.166, with the
forecast 0.145, conditional logit with the forecast 0.083.

The top LightGBM feature by gain is the morning matched volume (UK 171,757 against 25,414 for the open's share of
the book); the drift is largest where the morning has traded least, which is also where the WAP is least like an
offered price.

### 2.2 The hedged trade at the WAP, and a tick worse at each end

UK development (GBP1 a selection, 2% on each market's net; the 90% interval by day):

| model, threshold on the forecast log move | bets a day | ROI at the WAP (90%) | right way | ROI a tick worse each end (90%) | worst drawdown, GBP on GBP1 units (a tick worse) |
|---|---|---|---|---|---|
| band OLS, 0 | 238.6 | +6.04% (+5.89 to +6.20) | 61.0% | -3.15% (-3.33 to -2.97) | 11,511 |
| band OLS, 0.30 | 41.4 | +18.75% (+18.08 to +19.39) | 73.9% | +4.62% (+3.81 to +5.40) | 553 |
| LightGBM, 0 | 238.6 | +6.39% (+6.24 to +6.54) | 61.2% | -2.78% (-2.96 to -2.61) | 10,171 |
| LightGBM, 0.10 | 138.5 | +10.69% (+10.43 to +10.94) | 66.5% | -0.14% (-0.43 to +0.16) | 1,351 |
| LightGBM, 0.30 | 36.7 | +23.89% (+23.26 to +24.49) | 76.7% | +10.96% (+10.24 to +11.67) | 125 |
| conditional logit, 0.30 | 6.7 | +26.14% (+24.42 to +27.85) | 83.8% | +18.14% (+16.18 to +20.11) | 43 |
| LightGBM coherent, 0.30 | 26.5 | +28.09% (+27.36 to +28.82) | 79.4% | +15.33% (+14.47 to +16.18) | 59 |
| LightGBM with the forecast, 0.30 | 37.4 | +26.60% (+25.99 to +27.18) | 78.6% | +14.63% (+13.92 to +15.31) | 94 |
| conditional logit with the forecast, 0.30 | 12.4 | +32.46% (+31.35 to +33.57) | 80.0% | +21.75% (+20.50 to +23.03) | 39 |

At 5% commission each ROI is 0.3-1.0 points lower. Where the bets are: at 0.30 the LightGBM's 56,243 bets are 32,492
at 51+ and 21,356 at 21-51 (the rest under 1,000 a band); the band OLS's 63,372 are all 21+. These are lays of the
long shots the models expect to drift, closed by a back at the BSP: the -24% CLV of a back at the WAP on 51+ shots
(other_markets_and_stakes_1002.md section 4, 2023-26Q1) read the other way. One tick at 51+ is 5 or 10 units, and the
real lay side sits far above the WAP (section 4), so even the "a tick worse" column overstates what a lay collects.

UK holdout, read once, the development thresholds fixed (0.30 for every model):

| model | bets a day | ROI at the WAP (90%) | right way | ROI a tick worse each end (90%) | bands |
|---|---|---|---|---|---|
| band OLS | 51.4 | +4.48% (+2.95 to +5.98) | 64.7% | -12.17% (-13.97 to -10.38) | 21-51 and 51+ |
| LightGBM | 34.5 | +8.40% (+6.56 to +10.17) | 67.6% | -7.61% (-9.73 to -5.50) | 51+ 5,330, 21-51 1,896 |
| conditional logit | 9.7 | +19.21% (+16.39 to +21.97) | 78.3% | +9.62% (+6.32 to +12.79) | all 2,050 at 51+ |
| LightGBM coherent | 25.3 | +11.98% (+9.88 to +14.04) | 70.5% | -3.66% (-6.16 to -1.28) | 51+ 4,585 |
| LightGBM with the forecast (none held) | 37.1 | +7.43% (+5.57 to +9.14) | 66.8% | -8.87% (-11.03 to -6.84) | 51+ 5,754 |
| conditional logit with the forecast (none held) | 7.4 | +21.56% (+17.71 to +25.21) | 76.3% | +9.22% (+4.79 to +13.59) | 51+ 1,284 |
| every model at threshold 0 | 256 | +3.0% to +3.4% | 57.8-59.1% | -6.2% to -6.7% | all |

The holdout halves the WAP ROI and turns every broad rule negative after one tick. The two that stay positive are
lays of 51+ shots at the WAP, 7-10 a day, where one tick is the least of the spread: section 4 measures that cell on
offered prices and finds it negative.

IE development: LightGBM at 0.30 +27.5% at the WAP, +16.8% a tick worse (30.8 bets a day); conditional logit with
the forecast at 0.30 +36.6% and +28.6% (11.9 a day); at threshold 0, +9.1% to +13.0% at the WAP and -0.3% to +3.7%
a tick worse. IE holdout (192 days): at threshold 0, +0.6% to +3.7% at the WAP and -7.4% to -10.6% a tick worse; at
the chosen 0.30, band OLS -18.1%, LightGBM -4.7%, coherent -0.7% (-4.07 to +2.61), with the forecast -7.7%,
conditional logit with the forecast +10.6% (+5.40 to +15.50) on 688 bets, 4.5 a day. The Irish books are thinner
than the UK's, the WAP further from an offered price.

## 3. Greyhounds, GB WIN: the first traded price to the BSP (research-query run 41292736133)

**Data.** Betfair Historic Data, BASIC plan (sources/betfair_historic/markets_greyhound_racing_2026.parquet):
188,660 GB WIN runners in 34,495 markets, 1 Jan to 29 Sep 2026; the open is the first traded price (no time stamp,
no volume: the files do not say how much traded at it, and data.md finds a price standing 24 hours before the off on
78 of 451,031 runners, so the first print is late and thin), the close the BSP (the coordinator's ask), with the last
pre-play traded price run beside it. Development March-June 2026 (January-February fit only), holdout July-September.

| window | model | R2 vs no-change | direction hit | at threshold 0: bets a day, ROI at the first price, a tick worse | at 0.30: bets a day, ROI, a tick worse |
|---|---|---|---|---|---|
| development (85,549 selections, 122 days) | band OLS | 0.103 | 57.2% | 701; +6.89% (+6.04 to +7.75); +1.41% (+0.58 to +2.24) | 15.2; +138.7%; +118.5% |
| development | LightGBM | 0.200 | 61.8% | 701; +12.21% (+11.44 to +13.03); +6.79% (+6.04 to +7.57) | 54.1; +64.6%; +56.3% |
| development | permuted floor, band / LightGBM | 0.064 / 0.058 | 57.1% | - | - |
| holdout (62,085 selections, 91 days) | band OLS | 0.121 | 57.4% | 682; +8.29% (+7.40 to +9.24); +2.77% (+1.91 to +3.68) | 16.5; +154.6%; +133.5% |
| holdout | LightGBM | 0.215 | 60.8% | 682; +11.50% (+10.75 to +12.31); +6.02% (+5.30 to +6.78) | 58.1; +63.4% (+56.6 to +70.9); +54.4% (+48.3 to +61.2) |
| holdout | conditional logit | - | 57.3% | 682; +11.03%; +5.67% | 50.0; +68.7%; +57.4% |

Closing at the last pre-play traded price instead of the BSP gives the same picture within a point (LightGBM holdout
R2 0.208, +11.35% and +6.37% at threshold 0).

Where the money is (LightGBM, holdout, threshold 0.30, by band of the first price): 1-2, 641 bets, 7 a day, +45.5%,
92.5% right, every one a lay; 2-3, 1,252 bets, +30.5%, 82% right, lays; 51+, 569 bets, +353%, 95% backs; 8-13,
920 bets, +14.6%, right 64%, a tick worse +8.8%. A dog whose first print is 1.5 and whose BSP is 2.5, or 200 and
20, is a first trade in an empty book, not a price anyone was offered for size. Returns of 50-350% per GBP1 with
no losing day are the signature of that, not of a market. The question that matters, whether a dog's early price
at the recorded book (T-60, T-15, with GBP offered each side) drifts to the BSP in a way that survives the spread,
cannot be answered from these files; the data agent's recorded greyhound books (sources/open_close.py
read_recorded_books, 2/3/4 TBP from 3 Oct) are the next run.

## 4. Pre-registered: the lay of long shots at the best lay offered, October 2026 (research-query run 41292736133)

The research agent's cell (markets.md 6.1), fixed before the data was read: lay at the best lay price at the
09:00, 10:00 and 11:00 marks on runners priced 21-51 there with at least GBP20 offered, closed by a back at the
BSP; pass if the 90% interval by day is above zero. Data: betfair_live_marks, WIN markets, 1-31 October 2026,
71,026 runner-marks; 22,153 runner-marks at the clock marks priced 21+ at the best lay with a BSP (GB 13,974, IE
8,179; 1,112 markets, 31 days); the BSP from the final mark (9,540) or, before 7 October, the price files through
race_results (2,312). Laying at the best lay takes the offer, so the spread is paid at the open; the BSP is the SP.
Each runner-mark is a GBP1 lay (liability price - 1); 2% commission on each market's net.

| cell | bets | liability | return on liability (90% by day) | return a GBP1 of stake | closed shorter than the lay |
|---|---|---|---|---|---|
| **pre-registered: 21-51, 09:00-11:00, GBP20+ offered** | 269 | GBP7,236 | **-0.62% (-0.99 to -0.31)** | -16.6% | 50.9% |
| the same, one lay a runner at its first mark | 58 | - | -1.16% (-1.74 to -0.43) | - | - |
| 21-51, 09:00-11:00, any size | 6,153 | GBP191,754 | -0.87% (-1.07 to -0.70) | -27.1% | 57.1% |
| 21-51, 12:00, GBP20+ | 184 | GBP5,293 | -0.51% (-0.76 to -0.28) | -14.6% | 53.8% |
| 51+, 09:00-11:00, GBP20+ | 85 | GBP19,610 | -0.79% (-1.21 to -0.47) | -183% | 67.1% |
| 51+, 09:00-11:00, any size | 8,009 | GBP2,866,261 | -0.64% (-0.76 to -0.51) | -230% | 69.6% |

All 24 cells (two bands, six mark windows, two size floors) are negative; the median size offered at the best lay
is GBP2-5 without the floor, GBP22-28 with it. The cell fails its pass rule. This is the takeable-price reading of
section 2's long-shot lays: at the WAP they look like +10-20%, at the price actually offered to a layer they lose
0.5-1% of the liability, which at 21-51 is 15-27% of the stake. The research agent's warning (Whelan 2025: takers of
pre-play offers lose, makers earn about zero after fees) holds on these books.

## 5. Football: the bookmaker open to the close, and Betfair's own early price (local run, 1 Nov 2026)

**Data and provenance.** football-data.co.uk fetched here on 1 Nov 2026 (`source_data.py --source football --fetch
--build --from 2012 --root <local>`, 348 files, 173,733 matches 2012/13 to 30 Oct 2026), the gold table built by
`football.data --build --root <local>` (173,929 rows; the same build the repository runs nightly, football_model.md
section 1), the panels by research/open_close/panels.football_panel, the runs by `python -m research.open_close.run
--split-by market_type --normalise-book` (refit every 3 months on the 14-season panel, every month on the Betfair-only
one; the first 12 and 6 months fit only). Outputs: reports/open_close/football_runs/open_close_football_benchmark.json and
_bfe.json, reports/open_close/model_football_benchmark.md and model_football_bfe.md (the raw-price runs kept as
*_raw). Two panels:

- *benchmark*: the open is football-data's benchmark open (Pinnacle on 91,767 matches, Betfair Exchange 16,222,
  B365 992, the average 973), the close Pinnacle's to 2024/25 and Betfair Exchange's from 2025/26 (football.data
  .benchmarks); each 1X2 outcome, each O/U 2.5 side and each Asian handicap side a selection: 503,934 selections in
  197,881 markets (match odds 324,516, over/under 110,590, handicap 68,828; the handicap from 2019/20, when the
  closing line is published, and only where it equals the opening line). **The open here is a bookmaker's line, a
  proxy for the exchange's early price, not a price one could lay or back on Betfair.**
- *bfe*: Betfair Exchange's own early and closing odds as football-data records them (BFE and BFEC columns, 2024/25
  on, match odds and over/under only): 82,707 selections in 32,979 markets, 26 Jul 2024 to 29 Oct 2026. Betfair's
  close exists in the gold table from 2024/25 on; before that there is no exchange close to score against.

**The bookmaker's margin had to come out first.** A first run on the raw prices (reports/open_close/football_runs/*_raw.json) gave
match odds R2 against no change 0.038 with the open's overround the top feature, the handicap 0.31 with the permuted
leak test scoring the same 0.305, and Betfair's early snapshot 0.69-0.70: all of it the book's overround shrinking
from the open to the close (benchmark match odds median 1.0293 to 1.0267, over/under 1.0356 to 1.0335; football-data's
Betfair early snapshot 1.0205 (10th-90th percentile 1.0056-1.0957) to 1.0076 (1.0033-1.0167)), which lengthens every
price a little and reads as a drift that no exchange trade can collect: a Betfair book sums to about one at both
ends. Every figure below is on prices scaled to a book of one at both ends (features.fair_prices), the comparison
that matches the exchange; on the thin early Betfair snapshot the normalised mid is still not a takeable price.

**Match odds, bookmaker benchmark** (development 2013-07 to 2026-07: 296,415 selections, 98,805 markets, 3,411
days): R2 against no change band OLS 0.002, LightGBM 0.004, conditional logit -0.002, coherent LightGBM 0.004;
direction hit 50.3-52.4%. Hedged at threshold 0: -0.0% to +0.4% at the mid, -3.9% to -4.4% a tick worse at each
end (86.9 bets a day); LightGBM at 0.05: 2.6 a day, +2.8% at the mid, -6.3% a tick worse. Holdout (August-October
2026, 6,078 selections, 2,026 markets, 74 days): LightGBM 0.044, coherent 0.037, band 0.010, conditional logit
-0.004; threshold 0 +0.4% to +1.1% at the mid, -3.1% to -3.9% a tick worse; LightGBM at 0.05: 50 bets in 74 days,
+14.0% at the mid, +6.3% (+1.6 to +11.2) a tick worse: too few to mean anything. **No market-only drift in match
odds**, as the research agent expected from Whelan (2025).

**Over/under 2.5, bookmaker benchmark** (development 2020-08 to 2026-07: 92,774 selections, 46,387 markets): R2
0.003-0.009; threshold 0 +0.3% at the mid, -1.9% a tick worse (57.6 a day). Holdout (4,052 selections, 74 days):
R2 0.008-0.009 (conditional logit -0.017); threshold 0 +0.2% to +0.4% at the mid, -1.7% to -2.1% a tick worse.
Nothing.

**Asian handicap, bookmaker benchmark: not evidence.** On fair prices the models still score R2 0.265 (development,
57,668 selections) and 0.315 (holdout, 2,446), 66-69% direction hit, +2.3% at threshold 0 at the mid (+0.1% a tick
worse) and +5.9% / +3.6% (+2.5 to +4.7) at 0.05 on 3-4 bets a day with 90-97% moving the forecast way; the permuted
floor is 0.15-0.16. The panel keeps only matches whose closing line equals the opening line, a fact known only at
the close: where the open's price sat far from evens the line usually moves, and the matches kept are those where
the price reverted instead. That is look-ahead by selection, so the handicap arm is withdrawn from the verdict. A
valid handicap panel needs the price at a fixed line at both ends (Betfair's handicap markets are one market a
line; the data agent's census).

**Match odds, Betfair's own early price** (development 2025-01 to 2026-07: 33,063 selections, 11,021 markets, 390
days): R2 LightGBM 0.069, coherent 0.067, band OLS 0.006, conditional logit -0.005; direction hit 53.4%.

| LightGBM, threshold | development: bets a day, ROI at the mid (90%), a tick worse (90%) | holdout Aug-Oct 2026 (5,991 selections, 74 days): the same |
|---|---|---|
| 0 | 84.8; +1.45% (+1.22 to +1.68); -2.80% (-3.03 to -2.57) | 81.0; +1.11% (+0.76 to +1.52); -3.14% (-3.47 to -2.74) |
| 0.05 | 13.6; +7.10% (+6.02 to +8.14); +1.82% (+0.74 to +2.86) | 10.2; +6.03% (+4.42 to +7.66); +0.45% (-1.19 to +2.09) |
| 0.10 | 6.6; +13.59% (+11.71 to +15.35); +7.79% (+5.92 to +9.49) | 4.9; +9.27% (+6.01 to +12.43); +3.42% (+0.12 to +6.60) |
| 0.20 | 2.6; +27.21% (+23.42 to +30.85); +20.66% (+17.04 to +24.10) | 1.8; +26.67% (+16.35 to +36.86); +20.44% (+10.65 to +30.27) |
| 0.30 (chosen) | 1.0; 99 bets | 10 bets |

Holdout R2 0.062 (LightGBM) and 0.073 (coherent). This is the one football cell positive a tick worse at each end
with an interval above zero on both windows (|forecast move| of 0.10 or more: +7.8% then +3.4%, 5-7 bets a day),
and it is not yet a trade: the early snapshot's book runs to 1.10 at the 90th percentile before normalising, so
these are thin quotes collected on a Friday or Tuesday afternoon, the moves of 0.10-0.30 in log price on a 5-13 shot
that then shortens (95-100% "right" at 0.30 in development) are what a stale quote in an empty book looks like, and
one tick on a 2-8 price is 1-3% where such a book's spread is more. The test is the live football recorder's books
with sizes (football/live.py records its own pre-play prices for CLV): the same rule on a takeable best back and
best lay. Over/under on Betfair's early price: R2 at or under zero in development (-0.006), 0.021 on the holdout;
threshold 0 +0.2% at the mid and -2.0% a tick worse. Nothing.

Football-data's Betfair early odds carry no time stamp and no volume (open_time_utc and open_matched are blank, so
hours_to_off and the matched-money feature are absent in this panel), which is the gap the data agent's census of
the live markets fills.

## 6. What next

- **Horses.** The market-only hedged trade is closed as a question: it does not survive the spread on offered prices
  (sections 2 and 4), as the 2 October reading said. What pays is the model's forecast of the move, already live.
  The one modelling step left in this direction is the lay side of the live rule (race_book.py's lays, the horses the
  model makes longer than the market), scored on the recorded books' best lay as section 4 did.
- **Greyhounds.** Re-run the panel on the recorded books (T-60, T-15, sizes) as the open, closed at the BSP; the
  first traded price is retired as an open.
- **Football.** Score the Betfair match-odds cell (LightGBM, |move| of 0.10 or more) on the live recorder's books
  with sizes, open at the first recorded two-sided book, closed at its pre-play close; rebuild the handicap panel at a
  fixed line from Betfair's own markets. The bookmaker-proxy question is closed: no drift once the margin is out.
- **The leak test** now permutes the closes (29f0b9d); a constant-shift baseline (the training mean of y) belongs in
  the model list so the permuted floor is read directly.
