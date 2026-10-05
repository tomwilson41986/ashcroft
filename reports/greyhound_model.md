# The greyhound model (the owner's ask, 5 Oct 2026)

Built as the horse model is: results data -> lag-safe metrics -> a market-blind model -> scored against the market.
Code: `greyhound/` (data, metrics, model); full-history run: `.github/workflows/greyhound-model.yml`; tests:
`tests/test_greyhound.py`. The feasibility study before it: `reports/greyhound_feasibility.md`.

## 1. Data

| Source | What | Held |
|---|---|---|
| GBGB results (`sources/gbgb.py`) | every GB run: trap, position, distances, sectional, run and calculated time, comment, weight, SP, grade, trainer, breeding | S3 from 2018 (853k races); locally Jul 2025-Sep 2026 for development |
| Betfair Historic Data (the owner's bundle) | GB/AU/NZ greyhound markets Jan-Sep 2026: first traded, T-1, closing price, BSP, result | S3 and locally |
| Betfair greyhound price files (`sources/greyhound_prices.py`) | BSP, morning and pre-play weighted average prices and traded volumes, 2018 on | built into tables in S3 by the CI run |

## 2. The metrics (`greyhound/metrics.py`, 135 features)

Each run gets its figures first; a dog's form is then those figures over its earlier DAYS (one row a dog a day,
lagged a day), and every group statistic comes from earlier days through `model.lagsafe`, the horse engine's rule.

| Metric | What it is |
|---|---|
| GSR, speed rating | the calculated time (GBGB's going-corrected) against the track and distance's standard winning time, in lengths (0.08 s), with our own meeting variant on top; a figure 40+ lengths off is a bad time in the source and dropped |
| ESR, early speed | the first sectional against the track and distance's standard |
| FSR, run-in speed | the time after the sectional against its standard |
| LBW | lengths behind the winner |
| From the comment | the break (very quick away +2 .. missed the break -2), led early, trouble (crowded, bumped, baulked, checked), running line (rails 1, middle 2, wide 3), finish (ran on +1, faded -1) |
| Trouble-adjusted and clear-run ratings | GSR credited 1.5 lengths a bump; GSR from clear runs only |
| Windows | last 1, 3, 6 runs (mean, best), recency-weighted, trend, consistency; at the track, the track and distance, from this trap |
| RvP | the dog's recent best against the par (winners' GSR) of today's track and grade |
| Trap | today's trap's bias at the track and distance (shrunk), the dog's running line against its trap |
| Trainer, sire | strike rates (shrunk), the trainer's A/E against the SP |
| Market history | the dog's past SP and its A/E against it |
| Freshness, condition, class | days since the last run, trials, age, sex, weight change, grade and distance change |
| Against the field | rank, gap to the best and z-score of the key figures; the pace map (early-speed order, squeezed between faster dogs, a faster dog drawn inside) |

The lag-safety test rewrites the last day's results (orders, times, comments) and requires every feature of that
day's races to be unchanged. It caught a real leak in the first version (a dog running twice in a day read its own
earlier result) before any number below was produced.

## 3. The model and the first result (development data, Jul 2025-Sep 2026)

LightGBM binary on "won", market-blind (no price of the race itself), normalised within the race; walk-forward:
fit from Oct 2025, test Jan-Mar, Apr-Jun, Jul-Sep 2026 each on everything before it. The fit window here is short
(3-9 months); the CI run fits on 2019 on.

| Jan-Sep 2026, 189,673 runners in 34,663 races | Log-loss |
|---|---|
| Uniform (1 / field) | 0.47435 |
| **The model** | **0.45357** |
| Betfair first traded price | 0.44111 |
| Betfair T-1 price | 0.43307 |
| Bookmakers' SP | 0.42963 |
| BSP | 0.42752 |

Calibrated by decile (predicted against actual win rate: 7.8% / 6.4% at the bottom, 33.7% / 35.0% at the top).
Blended with the BSP it takes almost no weight (0.03 against 0.98): **it does not add to the final price**.

Against the early market (Betfair's first traded price; the bundle gives no volume):

| Back at the first price where the model's edge there is | Bets/day | CLV vs BSP | Return at the first price (90%) |
|---|---|---|---|
| > 0 | 298 | +1.7% | +1.3% (-0.3 to +2.9) |
| > 0.1 | 240 | +2.6% | +2.9% (+1.1 to +4.7) |
| > 0.2 | 193 | +3.7% | +3.6% (+1.5 to +5.7) |
| > 0.3 | 154 | +4.9% | +4.7% (+2.2 to +7.2) |

The same picture as the feasibility study, with a better model: form predicts which dogs the market shortens, so the
first prices on them are worth more than their BSPs. Every threshold is shown, none chosen; the full-history run is
the test. A rank objective over the race (LightGBM xendcg, softmax with a fitted temperature) did worse (0.455-0.468
by fold) and is not used.

## 4. The full history (CI run 37308130136: GBGB 2018-2026, fit from 2019, test a year at a time 2021-2026)

**The model against the BSP, by year (price-file markets)**

| Year | Model | BSP |
|---|---|---|
| 2021 | 0.4357 | 0.4219 |
| 2022 | 0.4364 | 0.4218 |
| 2023 | 0.4414 | 0.4244 |
| 2024 | 0.4462 | 0.4265 |
| 2025 | 0.4481 | 0.4265 |
| 2026 | 0.4500 | 0.4275 |

Over the whole period it scores 0.44348 against the BSP's 0.42499, across 1.39m runners and 244,933 markets. In a
blend it takes a weight of 0.115 against the market's 0.946, and the blend scores the same as the market alone.
**The model does not add to the BSP**, and it falls further behind it every year. Whether that is the market
getting sharper or the model's inputs changing (GBGB's grading, sectionals, the number of meetings) is still to
be checked.

The strongest features are: the dog's last three SPs, the recency-weighted GSR against the field (z-score, gap and
rank), runs before, grade change, weight against its mean, A/E against the SP, GSR at the course and distance, and
age.

**The early market: the 2026 Betfair bundle**

The rule is to back at the first traded price where the model's edge there exceeds the threshold.

| Edge | Bets/day | Return at the first price (90%) | CLV vs BSP | The same bets at BSP |
|---|---|---|---|---|
| > 0.1 | 236 | +5.2% (+3.4 to +7.1) | +4.5% | +1.4% |
| > 0.2 | 186 | +7.9% (+5.7 to +10.0) | +6.0% | |
| > 0.3 | 147 | +10.0% (+7.5 to +12.6) | +7.6% | +3.1% (+0.4 to +5.9) |

At T-1 the same rule makes about nothing at the T-1 price, and +1.8% to +2.5% at BSP, with lower bounds just
above zero. The first traded price can be thin: the bundle shows no volume behind it, so these fills are unverified.

**The price files (2021-2026): morning WAP with its traded volume**

Morning volume per runner is £0 at the median and £263 at the 90th percentile. Pre-play volume is £2,213 at the
median.

| Morning rule | Bets/day | At the morning WAP | CLV | The same bets at BSP |
|---|---|---|---|---|
| edge > 0.3, any volume | 13.5 | +8.2% (+5.2 to +11.2) | +4.7% | +7.5% |
| edge > 0.x, volume >= £20 or £100 | | -14% to -20% | about -20% | +4% to +12.6% |

The morning figures in this run are not to be trusted:
- The morning WAP's log-loss came out at 1.78, and the pre-play WAP's could not be computed. Both were normalised
  over races where only some runners had traded, and a WAP on two dogs of six says nothing about the race.
- Morning prices shorter than BSP by 20% on dogs the model calls value is not a market reading of the same race.

The scoring now:
- reads each price only on the markets where it prices every runner;
- reports how many runners and markets each price covers;
- drops any market whose result in the file disagrees with GBGB's, because a join to the wrong race would score
  one race's prices against another's result;
- reports the main morning rule year by year.

The next CI run gives the corrected figures.

**What the full history says**

- The model is not a better price than the BSP. Its value, if any, is in the early market: form predicts which
  dogs the market will shorten. In 2026 the first prices on those dogs beat the BSP by 4.5-7.6%, and the at-BSP
  returns are positive with lower bounds near zero.
- None of these thresholds was chosen out of sample. They are all reported, and none has yet been tested on a
  period it was not seen on.
- The at-BSP returns of a market-blind selection (+1.4% to +3% on the bundle) are worth a leak and robustness
  check before anything else. The lag-safety test covers the features, but it does not cover the joins.

## 5. Next

1. The corrected price-file scores (whole-field prices, result-checked joins, by year) from the next CI run.
2. Why the model falls behind the BSP by year: refit on recent years only, and check the feature drift.
3. If the early rule holds: the recorded GB greyhound books (from 5 Oct) for the depth at the time a bet would go
   in, then a paper forward test with thresholds fixed in advance, then small stakes, with the same gates as the
   horse rule.
