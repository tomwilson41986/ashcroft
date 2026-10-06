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

**The price files (2021-2026): morning WAP with its traded volume** (CI run 37316328027, the corrected scoring;
`reports/greyhound_model_ci_1005.json`)

The joins are sound: 244,933 markets matched GBGB's races, and the file's result disagrees with GBGB's in only 219
of them, which are dropped.

The prices are sparser than they looked:
- **The pre-play WAP is empty** in every greyhound file (0% of runners priced), so it cannot be scored.
- **The morning WAP prices 14.7% of runners**, and every runner in only 11.1% of markets. Morning volume per runner
  is £0 up to the 75th percentile and £259 at the 90th. Pre-play volume is £2,213 at the median.

Where the morning WAP prices the whole field, it is a real price. It scores 0.4315 against the model's 0.4438 on
the same markets, and the BSP's 0.4248 across all of them. The broken 1.78 in the first run came from normalising it
over part of a field.

The morning rules: back a dog at its morning WAP where the model's edge there exceeds the threshold.

| Morning rule | Bets/day | At the morning WAP (90%) | CLV vs BSP | The same bets at BSP (90%) |
|---|---|---|---|---|
| edge > 0.2, any volume | 17.7 | +5.4% (+2.9 to +7.9) | +2.3% | +6.4% (+1.0 to +11.8) |
| edge > 0.3, any volume | 13.5 | +8.1% (+5.1 to +11.1) | +4.7% | +7.4% (+0.5 to +14.3) |
| edge > 0.2, volume >= £20 | 11.2 | -15.1% | -19.6% | +10.4% (+2.0 to +18.8) |
| edge > 0.2, volume >= £100 | 10.7 | -18.6% | -23.1% | +6.2% (+2.0 to +10.5) |

The traded morning prices are about 20% shorter than the BSP on these dogs: they are dogs that drift, and backing
them in the morning loses. The positive at-BSP returns come with very wide intervals, because the BSP is uncapped and
a few long-priced winners carry them.

By year, the morning rule (edge > 0.2, any volume):

| Year | Bets/day | At the morning WAP (90%) | CLV | At BSP (90%) |
|---|---|---|---|---|
| 2021 | 24.1 | +3.4% (-2.3 to +9.1) | +5.1% | +0.5% (-5.9 to +6.8) |
| 2022 | 24.2 | +12.1% (+7.0 to +17.1) | +9.3% | +1.3% (-3.9 to +6.6) |
| 2023 | 22.4 | +14.5% (+9.1 to +19.8) | +7.3% | +8.6% (+2.7 to +14.5) |
| 2024 | 12.8 | -1.6% (-8.4 to +5.2) | -6.0% | +28.0% (-7.8 to +63.7) |
| 2025 | 11.6 | -8.3% (-15.4 to -1.2) | -5.4% | -5.9% (-13.9 to +2.2) |
| 2026 | 11.2 | -3.8% (-12.4 to +4.8) | -13.5% | +12.3% (+1.2 to +23.4) |

**The morning edge was there in 2021-2023 and has gone since 2024.** CLV turns negative, the bets fall to half as
many, and this is the same period in which the model falls further behind the BSP. In 2026 the first-traded-price
rule on the Betfair bundle still shows +7.9% at that price. It is a different price, though, often the first
matched bet of a thin market, and the price files say that a morning price which traded in volume did not beat the
BSP in 2024-2026.

**What the full history says**

- The model is not a better price than the BSP. Its value, if any, is in the early market: form predicts which
  dogs the market will shorten. In 2026 the first prices on those dogs beat the BSP by 4.5-7.6%, and the at-BSP
  returns are positive with lower bounds near zero.
- None of these thresholds was chosen out of sample. They are all reported, and none has yet been tested on a
  period it was not seen on.
- The at-BSP returns of a market-blind selection (+1.4% to +3% on the bundle) are worth a leak and robustness
  check before anything else. The lag-safety test covers the features, but it does not cover the joins.

## 5. The forward test (from 5 Oct 2026)

The first day with recorded greyhound books is 6 Oct, from about 10:25 UTC. On 5 Oct the trading day ran a version of
`live-trade.yml` that had no greyhound recorder yet, so the tracking run on 6 Oct priced 5 Oct's 136 races and found
no books to set them against.

Paper only (`greyhound/track.py`, `.github/workflows/greyhound-track.yml`, daily at 11:41 UTC once merged to the
default branch). What is fixed in advance:
- **The model:** fitted on every GBGB race before 5 Oct 2026 and frozen in S3 (`sources/greyhound/track/model.txt`)
  with its features and rules (`meta.json`). It is never refitted during the test.
- **The prices:** each day's GB races are priced from the dogs' earlier days only. The model is set against the
  recorded Betfair greyhound books (live-record.yml), at the first book recorded and at 60, 10 and 1 minutes before
  the off.
- **The bets:** a paper back of GBP2 on every dog whose edge at the best back price clears 0.1, 0.2 or 0.3, with no
  bet above 20. A bet counts as filled only if the book offered at least the stake. It is settled at that price and
  at the BSP, with CLV against the BSP.
- **The primary rule**, fixed before any tracked day: the first recorded price, edge > 0.2. The other marks and
  thresholds are reported, not judged.
- **The gate:** after 2,000 primary bets or 30 days, whichever is later, move to small real stakes only if the
  return at the taken price has a 90% lower bound above 0 and the mean CLV is above 0.
- **The ledger:** `sources/greyhound/track/days/<day>.parquet`; the priced runners and their marks,
  `pred/<day>.parquet`; the running summary, `summary.json` and each run's summary.

The limits of the test:
- The books come from the delayed key, so a price may be a few seconds old.
- A race's field is the one that ran, so a late non-runner is not seen as the market saw it.
- The first recorded book is the recorder's first snapshot (from 06:55 or 10:10 UTC), not the market's first trade.

## 6. The owner's normalised finishing position (6 Oct 2026)

The owner's formula, (N + 1 − 2P) / (3·√((N + 1) / (3(N − 1)))·(N − 1)), centres the finishing position on the middle
of the field and scales it so that every field size has the same spread (standard deviation 1/3). Ours runs from 1 for
the winner to 0 for last. Each was windowed as the parity block windows every measure, and fitted on the same features
and folds (local real GBGB, three 2026 folds; `reports/greyhound_nfpz_local_1006.json`):

| Finishing position | Log-loss | Jan / Apr / Jul 2026 | First price, edge > 0.2 | Edge > 0.3 |
|---|---|---|---|---|
| Ours (1 .. 0) | 0.45138 | 0.45402 / 0.44860 / 0.45145 | +5.2% | +7.4% |
| Both | 0.45113 | 0.45331 / 0.44842 / 0.45162 | +5.8% | +7.7% |
| **The owner's, in place of ours** | **0.45104** | 0.45352 / 0.44808 / 0.45145 | **+6.0%** | **+7.8%** |

It is better in two folds and level in the third. The gain is small, and the returns' ranges overlap, but it is the
best of the three on every summary, so the parity block now uses it in place of ours. Ours is still computed
beneath it, as the base of the shape and market-order measures.

## 7. Through the day

`greyhound-track.yml` runs every hour from 11:11 to 21:11 UTC and again at 22:41 (`greyhound/track.py --intraday`).
Each run sets the races GBGB has already resulted against the greyhound books recorded so far. It reports:
- the paper bets of every rule, settled at the price taken on GBGB's result;
- the price taken against the last book before the off (the BSP arrives only the next morning);
- the primary rule's profit hour by hour, cumulative through the day.

The output is the run's summary and `sources/greyhound/track/intraday/<day>.json`. The next morning's run settles the
day properly, at BSP.

## 8. Next

1. **Why the model falls behind the BSP from 2023.** Check whether GBGB's data changed (the coverage of sectionals and
   calculated times, comments, grading, the meetings held). Refit on a rolling recent window rather than everything
   since 2019. Check the drift of the features that matter most.
2. **The first traded price, with its depth.** The recorded GB greyhound books (from 6 Oct) show what was on offer
   when a bet would have gone in. Only that can say whether the bundle's first-price returns could have been taken.
3. **The at-BSP returns:** cap the BSP, then check by price band and track before reading anything into them.
4. Only after all three, a paper forward test with thresholds fixed in advance.
