# racing² - Ante-post Phase 0: what the data says (first read)

18 Nov 2026. Phase 0 of `docs/antepost_plan.md`. Every figure is from run 2 of `antepost-research.yml`
([run 44342882609](https://github.com/tomwilson41986/ashcroft/actions/runs/44342882609), artifact `antepost-phase0-2`:
`antepost_phase0.json`, `historic_markets.parquet`, `historic_runners.parquet`, `recorded_books.parquet`), made by
`python -m antepost.phase0`, unless a line names another source. Read-only: no order call, nothing written to S3.

## Status

| Question | Answer | Settled? |
|---|---|---|
| Betfair's ante-post rules as text | Read on the UK runner, 22:17 UTC 18 Nov 2026: one text on all 29 GB/IE markets open (section 0) | Yes |
| Does an ante-post market carry a BSP? | No: `bspMarket` false on 584 of 585 markets | Yes (from the data) |
| Does it go in play? | No: 0 of 585 went in play; `turnInPlayEnabled` false on 584 | Yes (from the data) |
| When does it close? | A median 55.6 hours before the scheduled off (section 2) | Yes (from the data) |
| How is a non-runner settled? | All bets stand run or not; a horse balloted out is void (REMOVED) (section 0) | Yes |
| Turn-up labels from history | 461 GB/IE markets labelled; every winner found running where `betfair_prices` holds the race (section 4) | Yes |
| Recorded books since 7 Oct | 44 markets over 42 days; thin and wide (section 5) | Yes |
| Turn-up model (ran / lost / void) | PASS on 2026: log loss 0.583 against 0.670 and 0.696 for the baselines (section 4.4) | Yes |
| Calibration of P(ran) | FAIL: error 0.023 to 0.015, but the log loss interval breaks the bar; not used (section 4.5) | Yes |
| Price stage: E[log SP given it runs] | PASS on 2026: mean abs. log error 0.545 against 0.570 and 0.564 (section 4.5) | Yes |
| EV entry rule (back, laid at SP) | FAIL on 2026: -32.3% on 216 bets, deflated Sharpe 0.000 (section 4.6) | Yes |
| Entries by stage (forfeits, 5-day, declarations) | No data found | **Open** |

## 0. Betfair's rules for these markets (read 18 Nov 2026)

Source: [run 44405657981](https://github.com/tomwilson41986/ashcroft/actions/runs/44405657981) (`antepost-rules.yml`,
`python -m antepost.rules`). It ran on the UK runner at 22:17 UTC and made one read-only `listMarketCatalogue` call with
`MARKET_DESCRIPTION`, on the delayed key; artifact `antepost-rules-1`. All 29 GB/IE `ANTEPOST_WIN` markets open
that night carry the same rules text, word for word:

> Who will win this race? Dead heat rules apply. This is an ante post market. All bets stand run or not unless void
> under the Horseracing Rules and Regs or under the Betfair Terms and Conditions. Please be aware that in some Antepost
> markets it is possible for more horses to be declared than the field safety limit allows and as such this market may
> have balloted out horses. Bets on such horses will be void.
> Betfair may suspend ante post markets at any stage to protect customers and carry out necessary administration,
> including in some cases the cancellation of unmatched bets. [...]
> Please note that an Antepost market is separate to a day of race market. Liabilities will not be carried between
> events however any winnings/losses in an ante-post market will be netted off against any winnings/losses in the
> relevant day of race market. You are only liable for commission on net winnings on both markets combined.

What this settles:
- **Non-runners lose: all bets stand, run or not.** The plan's `p_run` term is right: a horse that doesn't run costs
  the whole stake.
- **A horse balloted out is void.** That's the REMOVED status in section 3: in the big handicaps more horses are declared
  than the safety limit, and the ones balloted out are removed with their bets void (Ayr Gold Cup 2026: 175 of 247).
  So a horse eliminated by ballot costs nothing. That's a separate outcome from not running, and the turn-up model
  needs to predict it as a third class.
- **Commission is charged once on the net of both markets.** A back in the ante-post market hedged by a lay at the SP
  in the race-day WIN market is netted for commission. The SP exit costs one commission on the combined winnings, not
  two.
- **Betfair may cancel unmatched bets when it suspends the market**, which fits the suspensions and reopenings in section 2.

The descriptions also give:
- `bspMarket` false and `turnInPlayEnabled` false on all 29;
- `marketBaseRate` 2.0 and `discountAllowed` false on all 29 (the historic bundle showed 5% on most markets: section 2);
- a `suspendTime` per market. It looks like the next scheduled suspension rather than the close, for example:

  | Market | `suspendTime` | Race |
  |---|---|---|
  | Betfair Chase | 19 Nov 08:00 | 21 Nov, 54.5 h before the off |
  | Coral Gold Cup | 23 Nov 08:00 | 28 Nov |
  | King George Chase | 24 Nov 08:00 | 26 Dec |
  | Gold Cup | 12 Jan 08:00 | 19 Mar |
  | 2000 Guineas | 3 Mar 08:00 | 1 May |

  That these dates are entry or forfeit stages is unverified. If they are, the descriptions give each race's stage
  calendar, a feature for the turn-up model.

## 1. The historic ante-post markets

Source: the Historic Data bundle `s3://<capture bucket>/sources/betfair_historic/bundles/horse_racing_basic_2024.tar`
(4,404,215,808 bytes), streamed in full: 904,002 files, 0 unreadable, 907 holding an `ANTEPOST_WIN` market. The tar is
cut short ("unexpected end of data after 904,002 files"), as au/historic.py also found, so it may miss markets at the end.

| | Count |
|---|---|
| `ANTEPOST_WIN` markets, all countries | 585 |
| GB | 131 (2024), 140 (2025), 134 (2026) |
| IE | 18 (2024), 28 (2025), 18 (2026) |
| AU / US / FR / DE / GR | 97 / 14 / 3 / 1 / 1 |
| GB/IE scheduled offs | 20 Jan 2024 to 14 Nov 2026 |
| Runners (all countries) | 30,655 |
| Runners a market | median 44 (10th pct 16, 90th pct 101) |
| Days from first message to scheduled off | median 35.0 (25th pct 20.0, 75th 68.1, 90th 332.9) |

The bundle is BASIC: last traded prices only, no book and no volume. A last traded price is evidence of a price, not
of a fill.

Each ante-post market sits in its own event named for the meeting ("Newmarket 10th Oct", or "Cheltenham 2027" for a
festival). It is not the event of the race-day WIN market, which is why matching by event found 2 events in run 2.

## 2. The market's clock

- **No BSP, no in-play.** `bspMarket` false on 584 of 585 markets, `turnInPlayEnabled` false on 584, and none went in play.
  So the trade can't be closed at the SP inside the ante-post market. The SP exit has to be a lay in the race-day WIN
  market, as the plan assumed.
- **It closes about 2.3 days before the race.** The last time each market was open is a median **55.6 hours** before the
  scheduled off (5th pct 53.2, 25th 54.5, 75th 57.8, 95th 109.7; 578 markets). That timing fits the 48-hour declaration
  stage, but this is unverified until the rules text is read. The ante-post position can't be traded inside its own
  market after that point.
- **It suspends and reopens.** 86% of markets were suspended and reopened at least once (`status_path` in
  `historic_markets.parquet`), for example "OPEN > SUSPENDED > OPEN > SUSPENDED > ... > CLOSED". Why is not known yet:
  likely at each entry stage.
- **The scheduled time moves.** 30.8% of markets changed `marketTime` during their life, and some final times are not
  the race date. For example, the Celebration Mile market sits in event "Goodwood 29th Aug" with `marketTime`
  2026-07-29. Race dates taken from `marketTime` are therefore unreliable.
- Commission base rate (`marketBaseRate`): 5% on 490 markets, 10% on 40, 7% on 28, 8% on 16, mixed on 11. Each market
  is one winner (577 of 585).

**Consequence for the plan:** there's a gap of about 55 hours between the ante-post market's close and the race. Over
that gap the position can't be traded in its own market. It can only be hedged in the race-day WIN market, if that is
open. When the race-day WIN market opens for these races has not been measured yet.

## 3. How runners end (settlement, pending the rules)

| Final status in the ante-post market | Runners |
|---|---|
| LOSER | 28,361 |
| REMOVED | 1,692 |
| WINNER | 572 |
| HIDDEN | 30 |

- With a median of 44 runners a market and race-day fields far smaller, most horses that don't run must end as
  **LOSER**, not void. That fits all-in run or not for most non-runners. The rules text has to confirm it.
- **REMOVED** runners were removed a median 2.18 days before the off (25th pct 2.12, 75th 2.30), right after the market
  closes. Every one with an adjustment factor carried 0 (1,280 of 1,692). These come in clusters: Ayr Gold Cup 2026
  (175 of 247), Chester Cup 2026 (45 of 86), Royal Hunt Cup 2026 (44 of 104). Some markets have every runner removed:
  Grand National 2025 (43 of 43), Turners Novice Chase 2025 (40 of 41). Betfair's rules settle it (section 0): a horse
  balloted out is void, so these runners cost nothing. A market whose every runner was removed is unexplained by the
  rules text; it may have been replaced by a new market.
- 9 runners carried a BSP in the ante-post market (`bsp_in_market` non-null): no use as an exit.

## 4. The race-day match: not reliable yet

The aim is to label each listed horse as ran in the race, ran elsewhere that day, or didn't run. That label is the
turn-up model's target and the source of the race-day BSP.

- **Run 1/2, by Betfair selection ID and race date against `betfair_prices`:** of 22,362 GB/IE runners, 3,783 ran,
  506 ran elsewhere, 11,925 no run, and 6,148 in races not found (356 of 469 markets matched).
- **The check that fails:** every market's WINNER ran, yet 55 winners are labelled "no run" in races that were found,
  and 107 more sit in races not found. The share not found grows by year: 13.3% in 2024, 31.5% in 2025, 36.8% in 2026.
- **Run 2, by event against the bundle's WIN markets:** 2 events matched, because ante-post markets sit in their own
  events (section 1).

Likely causes, in order:
1. The race date is taken from `marketTime`, which is wrong for some markets (section 2).
2. Festival events ("Cheltenham 2025") carry no day.
3. Some horses may have a different selection ID in the ante-post market than on race day; the 55 winners point that way.
4. `betfair_prices` may be missing some days.

**Run 4, the third attempt** ([run 44355120048](https://github.com/tomwilson41986/ashcroft/actions/runs/44355120048),
artifact `antepost-phase0-4`): the race date taken from the event name first, then the Betfair horse ID on the day or a
day either side, then the normalised name.
- 467 of 469 GB/IE markets are now joined.
- Of the 6,257 runners who ran, 3,803 were found by ID and 2,454 only by name. So the ante-post market's horse IDs often
  differ from race day's.
- Where both sources name a winner, they agree on 99.44% of the runners who ran.
- **The winner check still fails: 445 of 459 GB/IE winners are found running (96.9%).** The 14 misses are of two kinds:
  - **6 labelled "ran elsewhere"**: the wrong race was chosen because most of the market's listed horses ran in another
    race that day. These are the big handicaps with consolation races (Chester Cup 2024 and 2026, Ayr Gold Cup 2025,
    Lincoln 2026, Northumberland Plate 2026) and a festival day (Gallagher Novices' Hurdle 2024).
  - **8 labelled "no run"**: the winner isn't found by ID or name on the day: four Leopardstown markets on 31 Jan 2026,
    Fact To File (Cheltenham 2024), Kopek Des Bordes (Cheltenham 2025), Old Park Star (Cheltenham 2026), and Electric
    Moon (Haydock, 22 Nov 2025). Possibly days or races missing from `betfair_prices`; not yet checked.

**Run 5, the fourth attempt: passes** ([run 44358928613](https://github.com/tomwilson41986/ashcroft/actions/runs/44358928613),
artifact `antepost-phase0-5`). The race is the one the market's settled winner ran in, where that winner is found
(446 markets). The result only names which race the market was on: it is no feature, and nothing is fitted on it.
- 454 of 459 GB/IE winners are found running. Where both sources name a winner, they agree on 99.65% of the runners who ran.
- **All 5 misses are two meetings missing from `betfair_prices`:**
  - Haydock, 22 Nov 2025 (the table holds only Ascot that day; Electric Moon's closest name there is "Electric Mason");
  - Leopardstown, 31 Jan 2026 (no Leopardstown runner in the table; no close name).
- Leaving out the 6 markets on those two meetings, **452 of 452 winners are found running: the check passes.**

### 4.1 The labelled set

GB/IE markets with their race found, the two missing meetings left out (computed from the run 5 artifact,
`historic_runners.parquet` and `historic_markets.parquet`):

| | Value |
|---|---|
| Markets | 461 |
| Race dates | 27 Jan 2024 to 14 Nov 2026 |
| Runners | 22,189: ran 6,218, ran elsewhere that day 979, did not run 14,992 |
| Duplicates (market, horse) | 0 |
| Runners who ran with no BSP | 0 of 6,218 |
| Found by ID / only by name | 3,804 / 2,441 (all GB/IE, before the two meetings were left out) |

### 4.2 Who turns up (descriptive)

The share of horses listed at each point before the off that ran in the race, by their last traded price then.
"Listed" means in the market and not removed at that time. The market closes about 55 hours out (section 2), so the
2-day and 1-day marks hold almost nothing.

| Days out | Listed | Races | Ran | 1-3 | 3-5 | 5-10 | 10-20 | 20-50 | 50+ |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 5,281 | 96 | 18.0% | 52.6% (19) | 79.5% (39) | 63.9% (122) | 47.4% (289) | 29.6% (712) | 11.6% (1,787) |
| 30 | 12,995 | 219 | 20.9% | 79.6% (54) | 75.4% (114) | 62.2% (312) | 49.2% (775) | 37.2% (1,438) | 19.9% (2,010) |
| 14 | 19,589 | 374 | 25.0% | 83.8% (105) | 84.4% (180) | 72.4% (583) | 55.8% (1,423) | 44.1% (2,989) | 27.5% (2,545) |
| 7 | 19,879 | 384 | 25.4% | 90.9% (143) | 83.8% (191) | 78.7% (717) | 59.1% (1,583) | 43.7% (3,386) | 27.3% (3,469) |
| 3 | 21,134 | 431 | 28.0% | 95.2% (167) | 91.5% (271) | 89.0% (893) | 76.6% (1,644) | 67.6% (2,613) | 47.2% (2,420) |

The "Ran" column counts every listed horse, with or without a price. The band columns count horses with a last
traded price at that point (n in brackets).

### 4.3 The price against the SP (descriptive, not a backtest)

For horses that ran, the ante-post last traded price at each point over the race's BSP.
- 30 days out: median log ratio 0.309 (price about 1.36 times the SP), above the SP for 66.3% (1,646 runners).
- 14 days: 0.152 (about 1.16 times), 59.6% (3,483).
- 7 days: 0.104 (about 1.11 times), 56.5% (4,222).
- 3 days: -0.001 (about level), 49.1% (5,375).

The horses that turn up shorten, and the longer out, the more.

Now the market as a whole: back every listed horse with a price at its last traded price, lay it at the SP if it ran,
lose the stake if it didn't. Mean return per GBP1, before commission, last traded price taken as if it filled:

| Days out | 1-3 | 3-5 | 5-10 | 10-20 | 20-50 | 50+ |
|---|---|---|---|---|---|---|
| 30 | -14.5% | -2.4% | -14.2% | -23.0% | -34.8% | -34.2% |
| 14 | -11.5% | -3.8% | -14.7% | -24.1% | -34.2% | -45.6% |
| 7 | -6.2% | -14.1% | -13.1% | -27.7% | -39.2% | -45.6% |
| 3 | -5.0% | -12.6% | -11.8% | -15.8% | -19.7% | -22.0% |

Over every listed horse it loses in every band. Horses that run gain on the SP, and horses that don't lose the stake,
which outweighs it. So the whole edge has to come from telling runners from non-runners better than the price does:
the turn-up model is the trade, not a side input. For example, 14 days out at 5-10, 72.4% ran and those runners'
median price was 1.07 times their SP. A back there pays only if the model can pick a subset whose chance of running,
times its price over the SP, clears 1 plus costs.

**Caveats:**
- A last traded price can be old: there is no book in BASIC.
- The spreads in section 5 aren't charged here.
- These are averages over all listed horses, not a rule, and nothing is selected. A test of any rule goes in the
  research ledger first (CLAUDE.md).

## 4.4 The turn-up model: PASS (ledger `antepost-turnup-1119`)

Registered 19 Nov 2026 09:39 UTC before its run, and read once:
[run 44500920492](https://github.com/tomwilson41986/ashcroft/actions/runs/44500920492), artifact `antepost-phase0-9`,
`antepost_turnup/antepost_turnup.json`. Two earlier runs, 44498425777 and 44499742662, stopped on code errors before
any score was produced; each is noted in the ledger. Code: `antepost/turnup.py`.

**What it predicts:** for each horse listed at each point from 180 to 3 days before the off, the chance of each
settlement outcome:
- **ran:** close the back at the SP;
- **lost:** didn't run, stake lost;
- **void:** balloted out or removed, stake back.

It uses only what was known at that point.

**Data:** 445 markets, 113,035 horse-points, no duplicates.

| Set | Races | Dates | Rows |
|---|---|---|---|
| Train | 234 | 27 Jan 2024 to 29 Jun 2025 | |
| Early stopping | 68 | Jul to Dec 2025 | |
| Test | 142 | 1 Jan to 24 Oct 2026 | 37,421: lost 25,941, ran 8,499, void 2,981 |

**Results on the 2026 test:**

| | Log loss | Brier, P(ran) | AUC, P(ran) | Calibration error, P(ran) |
|---|---|---|---|---|
| Model (LightGBM) | **0.583** | **0.121** | **0.856** | 0.025 |
| B1: outcome shares by days out and price band | 0.670 | 0.133 | 0.803 | 0.017 |
| B2: logistic on price and days out | 0.696 | 0.140 | 0.782 | 0.017 |

The model minus each baseline, per row (90% interval resampling whole races):
- against B1: -0.087 (-0.150 to -0.036);
- against B2: -0.113 (-0.177 to -0.060).

Log loss by days out (model / B1 / B2):

| Days out | 180 | 120 | 90 | 60 | 45 | 30 | 21 | 14 | 7 | 3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Model | 0.366 | 0.357 | 0.350 | 0.406 | 0.444 | 0.561 | 0.663 | 0.650 | 0.629 | 0.627 |
| B1 | 0.355 | 0.364 | 0.376 | 0.444 | 0.515 | 0.713 | 0.777 | 0.757 | 0.731 | 0.670 |
| B2 | 0.409 | 0.420 | 0.410 | 0.458 | 0.529 | 0.756 | 0.816 | 0.776 | 0.743 | 0.691 |

The registered reading holds on all three counts:
1. Both intervals are below zero.
2. The model is no worse than B2 at 7 of 7 points from 60 to 3 days out.
3. Its calibration error (0.0249) is within B2's (0.0169) plus 0.01.

The biggest gains are 45 to 7 days out. At 180 days the model is slightly worse than B1, but that point is outside
the registered ones.

**What carries it (gain):** in order, the last traded price, how old that price is, the number listed, the month,
days listed, days out, and the horse's runs in the last 365 days.

**Caveats:**
- **Calibration at the extremes.** By decile of P(ran), the model is too low at the top (predicted 0.691, observed
  0.770) and too high at the bottom (0.017 against 0.000). Both baselines are better calibrated, though much worse at
  ranking. Before its probabilities set stakes, it needs a calibration step, fitted on 2025 and checked on 2026.
- **Month** ranks fourth on only two training seasons. That may be a calendar effect that won't hold.
- **This is a model of the outcome, not a trading rule.** No stake or return was read. The edge still has to come from
  pricing these chances against the ante-post price and the expected SP (plan sections 5-6). Any entry rule is
  registered on its own, with the deflated Sharpe bar.

## 4.5 Calibration (FAIL) and the price stage (PASS) (ledger `antepost-price-1119`)

Registered before the run and read once:
[run 44503451335](https://github.com/tomwilson41986/ashcroft/actions/runs/44503451335), artifact `antepost-phase0-10`,
`antepost_price/antepost_price.json`. Code: `antepost/price.py`. On the same run the turn-up test reproduced its PASS.

### Calibration: FAIL

The turn-up model was fitted to Jun 2025. A correction for its probabilities (Dirichlet calibration) was fitted on its
Jul to Dec 2025 predictions, then applied to its 2026 predictions (37,421 rows).

| On 2026 | Uncalibrated | Calibrated |
|---|---|---|
| Log loss | 0.689 | 0.661 |
| Calibration error, P(ran) | 0.0233 | 0.0154 |
| Calibration error, P(void) | 0.063 | 0.043 |
| Top decile of P(ran): predicted / observed | 0.689 / 0.768 | 0.741 / 0.765 |

The calibration error falls, as required. But the paired log loss (calibrated minus uncalibrated) has a 90% interval of
-0.149 to +0.073. That breaks the registered bar, which required the upper end to be at most +0.002. On average the
correction helps; in some races it makes the forecast much worse. **So it fails, and the correction isn't used.** The
turn-up model's probabilities stay uncalibrated, with the known shortfall at the extremes (section 4.4).

### The price stage: PASS

For horses that ran: the race-day SP predicted from what was known at each point, on 6,508 rows in 142 races in 2026.
The model was fitted on 11,040 rows, with 2,453 for early stopping.

| On 2026 | Mean abs. log error | RMSE (log) | Bias (log) | Concordance within race |
|---|---|---|---|---|
| Model (LightGBM) | **0.545** | **0.700** | -0.081 | **0.734** |
| P0: the ante-post price | 0.607 | 0.796 | +0.171 | 0.733 |
| P1: price shifted by its typical move for the point and price band | 0.570 | 0.741 | -0.040 | 0.730 |
| P2: linear on price and days out | 0.564 | 0.725 | -0.025 | 0.733 |

The model's error minus each baseline's (90% race-bootstrap interval):

| Against | Mean difference | 90% interval |
|---|---|---|
| P0 | -0.062 | -0.094 to -0.031 |
| P1 | -0.025 | -0.041 to -0.011 |
| P2 | -0.019 | -0.032 to -0.007 |

Mean absolute log error by days out (model / P1 / P2):

| Days out | 180 | 120 | 90 | 60 | 45 | 30 | 21 | 14 | 7 | 3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Model | 0.746 | 0.804 | 0.792 | 0.658 | 0.638 | 0.593 | 0.532 | 0.536 | 0.533 | 0.466 |
| P1 | 0.745 | 0.786 | 0.752 | 0.657 | 0.649 | 0.620 | 0.558 | 0.575 | 0.567 | 0.490 |
| P2 | 0.736 | 0.786 | 0.755 | 0.652 | 0.644 | 0.607 | 0.547 | 0.570 | 0.559 | 0.488 |

What it shows:
- **The gain is 45 to 3 days out.** At 120 and 90 days the model is worse than both simple baselines.
- **It is biased short.** It forecasts the SP about 8% shorter than it comes (-0.081 in log). The EV must allow for it.
- **The SP stays uncertain from the ante-post market.** The spread of the error (log, Jul to Dec 2025) is 0.60 at 3
  days out, 0.66 to 0.73 at 7 to 30 days, and 0.76 to 0.81 further out. The expected price over the SP has to carry
  that spread, not just the central forecast.
- **What carries it (gain):** the price, its share of the market, the trades so far, the price rank, and the number
  listed.

Neither part is a trading rule; no stake or return was read.

## 4.6 The EV and entry rule: FAIL (ledger `antepost-ev-1119`)

Registered before the run and read once:
[run 44536430030](https://github.com/tomwilson41986/ashcroft/actions/runs/44536430030), artifact `antepost-phase0-11`,
`antepost_ev/antepost_ev.json`. Code: `antepost/ev.py`.

**The rule:**
- **When to bet:** at 60, 45, 30, 21, 14, 7 or 3 days out, back GBP1 the first time a horse's EV reaches the threshold.
  One bet per horse per race, held to race day.
- **The EV:**

  `chance it runs × (fill price ÷ expected SP − 1, less 2% commission on a win) − chance it doesn't run`

  The fill price is the last traded price less half the typical spread for its price band in the recorded books.
- **The exit:** a runner is laid at the SP, a non-runner loses the stake, and a balloted horse is void.
- **The threshold:** chosen on Jul to Dec 2025 as 0.05, the best return of the thresholds with at least 30 bets,
  though every threshold lost there too. Then scored once on 2026.

**The result on 2026:**

| | Value |
|---|---|
| Bets / races | 216 / 49 |
| Return per GBP1 | **-32.3%** (90% race-bootstrap interval -53.0% to -0.5%) |
| Deflated Sharpe | 0.000 (per-race Sharpe -0.228 against a bar of 0.626, 373 trials counted) |
| Bets on horses that didn't run | 125 of 216 |
| Predicted EV of the bets taken | +26.7% |

Every threshold, on 2026 (recorded for completeness; only 0.05 was registered to be read):

| Threshold | 0 | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| Bets | 307 | 216 | 154 | 86 | 56 |
| Predicted EV | +18.8% | +26.7% | +35.2% | +52.8% | +69.0% |
| Realised return | -29.8% | -32.3% | -50.8% | -64.3% | -71.7% |

**Why it fails:**
- **The higher the predicted EV, the worse the return, so the rule selects the models' errors.** Two errors stack:
  - The turn-up model's P(ran) is too high at the bottom, and its calibration failed (sections 4.4 and 4.5).
  - The price model's error spread (0.6 to 0.8 in log) gives long prices a large expected-value uplift that isn't there.
- **Most of the damage is at long prices and far out:**
  - 102 of the 216 bets were priced 50+, returning -61%.
  - 77 were taken 60 days out, returning -88%.
- **Some cells are positive** (5-10: +2.0% on 52 bets; 45 days out: +15.8% on 39). They are post-hoc and not a
  finding: a rule built on them needs its own registration and counts as more trials.

**No rule from this line goes forward.** The turn-up and price models stand as forecasts. Backing on the EV they imply
doesn't make money once a back pays the spread and loses its stake on non-runners.

## 5. The recorded snapshots (7 Oct to 17 Nov 2026)

Source: `s3://<capture bucket>/betfair_live/<day>/books_antepost.csv.gz` and `markets_antepost.csv.gz`, one snapshot a
day by `betfair_recorder.py --antepost` (live-record.yml).

| | Value |
|---|---|
| Days | 42 (7 Oct to 17 Nov 2026) |
| Markets | 44, from the Old Rowley Cup (9 Oct 2026) to the Ascot Gold Cup (15 Jun 2027) |
| Book rows / runners | 43,014 / 1,841 |
| Runner status at the last snapshot | ACTIVE 1,246, HIDDEN 567, REMOVED 27, LOSER 1 |
| Snapshots with the market open | 96.5% |
| Active runners with a real back and lay (lay under 1000) | 25.2% |
| Matched to date, per market a day | median GBP10,851 (10th pct 643, 90th 40,105) |

Largest matched totals at the last snapshot (cumulative over the market's life): Arkle Chase GBP91,462, Champion Hurdle
GBP64,731, Gold Cup GBP59,941, Grand National GBP41,389, King George Chase GBP39,427, Supreme Novice Hurdle GBP38,611.
Smallest: Ascot Gold Cup GBP646, Punchestown John Durkan GBP700, Fairyhouse Drinmore GBP756.

The spread (best lay / best back - 1) on active runners with a real lay, by best back price:

| Best back | n | Spread median | 25th-75th pct | Back size median | Lay size median |
|---|---|---|---|---|---|
| 1-3 | 231 | 35.9% | 5.9% - 110% | GBP26.90 | GBP15.00 |
| 3-5 | 458 | 14.3% | 8.3% - 42.1% | GBP25.18 | GBP14.42 |
| 5-10 | 1,142 | 21.2% | 11.1% - 43.2% | GBP25.35 | GBP13.50 |
| 10-20 | 2,459 | 41.2% | 20.0% - 119.5% | GBP21.71 | GBP11.00 |
| 20-50 | 3,879 | 84.6% | 36.4% - 233% | GBP21.11 | GBP9.00 |
| 50+ | 1,647 | 153% | 63.6% - 264% | GBP23.07 | GBP2.82 |

So the book is wide, and a typical best price holds about GBP10-25. Only a quarter of active runners have a real
two-sided book. Any edge has to clear spreads like these, and stakes would be small.

Of the 44 recorded markets, 15 had passed their off by this run. Their race-day match has the same problem as section
4: 238 runners in races not found, and 567 runners HIDDEN at the last snapshot.

## What's next

1. **The rules text: read 18 Nov 2026 (section 0).** Previously: GitHub's runner is refused (403), and no Betfair
   certificate was used: the run fell back to the password login. Options: one read-only catalogue call on the UK
   server runner, at a time the live jobs are idle, or a `BETFAIR_CERT`/`BETFAIR_KEY` secret if certificate login is
   allowed from GitHub. Owner's call.
2. **The race-day match: passed in run 5 (section 4).** Two meetings are missing from `betfair_prices` (Haydock,
   22 Nov 2025; Leopardstown, 31 Jan 2026): a gap in the nightly load to look at separately. Earlier plan, kept for
   the record:
   - take the race date from the event name where it carries one, else `marketTime`;
   - match on selection ID within plus or minus 1 day, then on the normalised horse name where the ID fails;
   - check against `betfair_prices` day coverage (rows on the date);
   - report the winners found as the pass/fail check: every WINNER must be found running.
3. **When the race-day WIN market opens** for these races, against the ante-post close about 55 hours out. This sets
   how long a position has no exit.
4. **Entries by stage:** still no source. The ante-post market's own runner list and its suspensions may stand in.
