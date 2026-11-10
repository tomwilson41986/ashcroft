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

## 6. The parity metrics on the full history (CI run 37458951248; `reports/greyhound_parity_ci_1006.json`)

The base 135 features and base + the parity block (267: the horse model's families on GBGB, `greyhound/parity.py`)
were fitted on the same 2.6m runs and the same folds (fit from 2019, test a year at a time 2021-2026).

| | Base | + parity |
|---|---|---|
| Log-loss against the SP (SP 0.42612) | 0.44210 | **0.43874** |
| By fold 2021 / 22 / 23 / 24 / 25 / 26 | .4355 / .4360 / .4408 / .4457 / .4478 / .4496 | **.4324 / .4324 / .4377 / .4424 / .4443 / .4459** |
| Blend weight beside the SP | 0.181 | 0.249 |
| Blend weight beside the BSP (2026 bundle) | 0.063 | 0.085 |
| 2026 first price, edge > 0.1: return (90%), CLV | +5.6% (+3.7 to +7.4), +4.6% | **+8.7% (+6.8 to +10.6), +7.1%** |
| 2026 first price, edge > 0.2 | +7.5% (+5.3 to +9.7), +6.0% | **+10.1% (+7.9 to +12.3), +9.0%** |
| 2026 first price, edge > 0.3 | +9.7% (+7.1 to +12.2), +7.5% | **+12.1% (+9.5 to +14.7), +11.1%** |
| Morning rule (edge > 0.2), 2022 / 23 / 24 / 25 / 26 | +13.8 / +14.5 / −0.1 / −5.1 / −3.3% | +17.6 / +17.1 / +2.4 / +2.6 / −3.9% |

The parity block is better in every year, by about 0.003 of log-loss, and the model now takes a quarter of the weight
in a blend with the SP. It is still well short of the BSP, and it still falls further behind it each year. The early
price is where it gains most: the first-price CLV rises from +6.0% to +9.0%. The morning price in volume still did not
beat the BSP from 2024. The top features now include the race-strength change (`gp_rs_vs_past`) and the dog's market
history against the field (`gp_mkt_w3`, `gp_mkt_w5`, `gp_mkt_l1`).

The paper forward test still runs on the base model frozen on 5 Oct; it is the pre-registered test and is not changed
mid-way. A second frozen model with the parity block would be a second track, run beside it.

## 7. The owner's normalised finishing position (6 Oct 2026)

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

## 8. HorseRaceBase's System Builder categories, converted (6 Oct 2026)

The owner's list (HRB's v4 System Builder) was converted to greyhounds where a greyhound version exists and the model
did not already have it: `greyhound/hrb.py`, the `hrb` block, 64 features.
- **Head to head** with today's field.
- **The field's make-up:** course-and-distance and track winners, won or placed last time, ran in the last 30 days.
- **The dog's record** at today's grade, trap, trip, going and with its trainer.
- **Class, trip and wins:** best grade raced and won, longest trip won, days and runs since a win, best of ten.
- **Weight** against its heaviest and lightest.
- **Trainer:** the last 7 and 14 days, at the track, its home-track share.
- **Breeding and owner:** sire by trip, dam, owner.
- **The race:** hour, card number, weekday, month, winner's prize, handicap.
- **Age** against the field, birth month, and days since a bitch's season.

Jockey, headgear, official ratings, fences, surface and the day's odds have no greyhound counterpart, or are left to
the market-blind design.

On real GBGB data, with three 2026 folds and the same features (`reports/greyhound_hrb_local_1006.json`):

| | Log-loss | Jan / Apr / Jul | First price, edge > 0.2 (CLV) |
|---|---|---|---|
| Base + parity | 0.45104 | .45352 / .44808 / .45145 | +6.0% (+5.7%) |
| + the HRB block | **0.44993** | **.45222 / .44781 / .44972** | **+6.8% (+6.2%)** |

Dropping one group at a time, the log-loss change (positive: the group helps):

| Group | Change | Verdict |
|---|---|---|
| Trainer | +0.00019 | helps |
| Head to head | +0.00015 | helps |
| Weight | +0.00013 | helps |
| Field | +0.00010 | helps |
| Record by condition | +0.00009 | helps |
| Age and season | +0.00008 | helps |
| Race | −0.00003 | neutral |
| Class, trip and wins | −0.00010 | slightly hurts |
| Breeding and owner | −0.00010 | slightly hurts |

The block now serves all but the last two groups. Those are still computed but are not features (`hrb.LEFT_OUT`).
`greyhound-model.yml --compare` scores base, base + parity and base + parity + HRB on the full history.

### 8.1 On the full history (CI run 37476323556; `reports/greyhound_hrb_ci_1006.json`)

GBGB 2018-2026, fitted from 2019, tested a year at a time 2021-2026, 1.64m runners. "All" is base + parity + HRB
(316 features).

| Log-loss | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | All years |
|---|---|---|---|---|---|---|---|
| Base | .43551 | .43596 | .44076 | .44572 | .44783 | .44955 | 0.44210 |
| Base + parity | .43240 | .43235 | .43775 | .44256 | .44430 | .44581 | 0.43876 |
| All | **.43163** | **.43155** | **.43664** | **.44153** | **.44311** | **.44429** | **0.43772** |
| SP | .42247 | .42258 | .42617 | .42831 | .42889 | .42957 | 0.42612 |

The HRB block is better in every year, by about a third as much as the parity block was. The model still falls behind
the market from 2023.

| | Base | Base + parity | All |
|---|---|---|---|
| Blend weight on the model (with SP) | 0.18 | 0.25 | **0.26** |
| Blend weight on the model (with BSP, 2026 Betfair bundle) | 0.06 | 0.09 | **0.12** |
| 2026 bundle, first traded price, edge > 0.2: ROI (CLV vs BSP) | +7.5% (+6.0%) | +10.2% (+9.0%) | **+13.2% (+10.0%)** |
| 2026 bundle, T-1 minute, edge > 0.2: ROI (CLV) | +1.2% (−1.3%) | +1.9% (−0.8%) | **+3.4% (−0.6%)** |
| 2026 bundle, the same bets at BSP | +2.3% | +2.2% | **+4.7%** (90%: +2.3 to +7.0) |
| Price files, morning price, edge > 0.2, all years: ROI (CLV) | +6.3% (+2.4%) | +10.8% (+7.3%) | **+13.0% (+8.4%)** |

Morning price, edge > 0.2, by year (all features): 2021 +8.9%, 2022 +20.7%, 2023 +23.2%, 2024 +2.0%, 2025 +3.4%,
2026 +1.1% (CLV −9.0%). The morning books with volume (≥ GBP20 matched) lose 11-16% at the morning price, as in §6.

**Verdict:** the HRB block (less class/trip/wins and breeding/owner) belongs in the live model (the card check, §8.4). Before it goes in,
its two weight features (`hb_weight_vs_max`, `hb_weight_vs_min`) must join `live.CARD_UNSAFE`: they read the day's
weigh-in, which the morning card does not have. Then the live training's engine takes `blocks=("parity", "hrb")`.

### 8.2 The model overrates the dogs it picks

On the first live paper day (§9), the edge > 0.2 bets at the first traded price had 3 winners where the prices
expected 6.9 and the model 10.5. The full history says the second half of that is not bad luck. On every test year,
for the dogs the model backs at the SP with edge > 0.2 (price up to 20, all features):

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | All |
|---|---|---|---|---|---|---|---|
| Bets | 29,209 | 27,160 | 30,315 | 31,589 | 31,104 | 23,800 | 173,177 |
| Winners | 3,079 | 2,865 | 3,130 | 3,063 | 2,928 | 2,321 | 17,386 |
| Model's expected | 5,470 | 5,137 | 5,805 | 5,926 | 5,916 | 4,522 | 32,775 |
| SP's expected (overround removed) | 2,975 | 2,813 | 3,164 | 3,204 | 3,169 | 2,380 | 17,705 |
| Winners / model | 0.56 | 0.56 | 0.54 | 0.52 | 0.50 | 0.51 | **0.53** |
| Winners / SP | 1.04 | 1.02 | 0.99 | 0.96 | 0.92 | 0.98 | **0.98** |

The dogs it picks win about half as often as it says, and about as often as the SP says. Some of that is built into
picking on the largest edges. But the model's probabilities, used raw, overstate every edge they select on.

The blend with the market (0.26 on the model) is calibrated on the same bets: 0.98 of its expected winners. Selecting
on the blended edge instead leaves few bets at the SP (481 over six years at edge > 0.02), but they made +4.8% at the
SP.

So the edge a live rule acts on should be the blended probability against the price, the blend fitted on Betfair
prices rather than the SP. The live trader's rule (raw model, edge > 0.2) is unchanged until that is tested on the
recorded books.

### 8.3 The edge on the blended chance, out of sample (CI run 37516339956; `reports/greyhound_blend_ci_1006.json`)

These use the 2026 Betfair bundle. The blend (both log-odds, normalised in the race) is fitted on 1 Jan to 16 May and
every rule is scored on 17 May to 29 Sep only. The model is the full-history "all" model, with weigh-in features.

| | First traded price | T-1 minute |
|---|---|---|
| Blend weights (price / model) | 0.66 / 0.55 | 0.82 / 0.26 |
| Test log-loss: price / model / blend | 0.4427 / 0.4453 / **0.4381** | 0.4315 / 0.4464 / 0.4316 |

The first traded price is a weak price, and the model adds a lot to it. A minute before the off, the price has taken
in almost everything the model knows.

| Rule (test half) | Bets | ROI at the price (90%) | CLV | At BSP | Winners: actual / price's / model's |
|---|---|---|---|---|---|
| First price, raw edge > 0.2 | 23,192 | +14.6% (+11.4, +17.8) | +11.6% | +4.2% | 3,611 / 3,043 / 4,784 |
| First price, blend edge > 0.05 | 23,980 | +16.0% (+13.2, +18.8) | +15.3% | +1.5% | 5,109 / 4,480 / 5,914 |
| First price, blend edge > 0.10 | 17,380 | +19.9% (+16.5, +23.3) | +19.7% | +0.9% | 3,628 / 3,133 / 4,378 |
| T-1, raw edge > 0.2 | 25,212 | **+4.6%** (+1.4, +7.7) | +0.3% | +4.6% | 3,272 / 3,059 / 4,793 |
| T-1, raw edge > 0.3 | 19,616 | +5.9% (+2.2, +9.6) | +0.5% | +6.0% | 2,384 / 2,198 / 3,696 |
| T-1, blend edge > 0.05 | 15,836 | +3.5% (−0.2, +7.1) | +0.9% | +2.8% | 2,493 / 2,397 / 3,686 |

Three readings:

- **Against Betfair's prices, the model's picks beat the price.** At T-1 they win 1.07 times the winners the price
  expects, and at the first traded price 1.19 times. The model still expects 1.5 times what they win. So the edge is
  real but about a third of the size the model says. Against the SP (§8.2) the picks only matched the price, because
  the SP is a bookmaker's price.
- **The blend does not beat the raw rule a minute before the off.** At the first traded price it buys more return at
  that price but less at BSP. It is finding the first price's mistakes (the price moves its way) more than the dog's.
  That is only worth having where the first traded price can really be taken.
- **The live rule is unchanged:** raw model, edge > 0.2. The blend is not wired into the trader. Today's recorded books
  (§9) showed the first-traded bets beating the last book by only 1-2%, against the bundle's 11.6%. Until the
  recorded books show the first traded price as takeable as the bundle says, the raw T-1 result (+4.6%, the same at
  BSP) is the one to trust.

### 8.4 The morning card against the results (6 Oct 2026; `greyhound.live --check-card`)

The live model prices the morning's card: Betfair's catalogue, each dog matched to its GBGB history by name. The
check prices a past day twice, once from its card and once from GBGB's results, and compares every feature for every
dog with a GBGB history. 6 Oct, 544 dogs:

| | Features agreeing (≥ 99% of dogs) | Field size agrees |
|---|---|---|
| First check | 236 of 316 | 87% |
| Removed runners left off the card | 255 | 91% |
| Betfair's HC read as GBGB's HP; every runner priced | **300** | **100%** |

Of the 16 that still differ:
- **Nine cannot be known in the morning** and are left out of the live model (`live.CARD_UNSAFE`):
  - the four weigh-in features;
  - the three going features;
  - the prize and the handicap mark.
- **Seven are race-relative and differ only where a dog in the race has no GBGB history in the morning:**
  - the trainer's z, gap and rank (89-96% agree);
  - age against the youngest (96%);
  - three of the lead features (99%).

  On 6 Oct that dog was a GBGB debutant or a dog its first run named differently, and the results know it while the
  card does not. That gap is the card's, not a bug.

The removed-runner and grade fixes help the current live model as much as the HorseRaceBase one. Their features were
the ones off. The HorseRaceBase live model is ready to fit (`greyhound-live.yml` task `train`) once these fixes reach
the default branch, where the morning prices run.

## 9. Through the day

`greyhound-track.yml` runs every hour from 11:11 to 21:11 UTC and again at 22:41 (`greyhound/track.py --intraday`).
Each run sets the races GBGB has already resulted against the greyhound books recorded so far. It reports:
- the paper bets of every rule, settled at the price taken on GBGB's result;
- the price taken against the last book before the off (the BSP arrives only the next morning);
- the primary rule's profit hour by hour, cumulative through the day.

The output is the run's summary and `sources/greyhound/track/intraday/<day>.json`. The next morning's run settles the
day properly, at BSP.

## 10. Next

0. **The first traded price, recorded against the bundle's** (§8.3): over the coming days, compare what the recorded books
   offered at the first match with the bundle's first traded price. The first-price rule and the blend only matter if
   the two agree.
1. **Why the model falls behind the BSP from 2023.** Check whether GBGB's data changed (the coverage of sectionals and
   calculated times, comments, grading, the meetings held). Refit on a rolling recent window rather than everything
   since 2019. Check the drift of the features that matter most.
2. **The first traded price, with its depth.** The recorded GB greyhound books (from 6 Oct) show what was on offer
   when a bet would have gone in. Only that can say whether the bundle's first-price returns could have been taken.
3. **The at-BSP returns:** cap the BSP, then check by price band and track before reading anything into them.
4. Only after all three, a paper forward test with thresholds fixed in advance.

## 11. The feature library and the race's own loss (CI run 37665090403; `reports/greyhound_lib_ci_1007.json`)

Three models were fitted on the same GBGB rows and folds (fit from 2019, tested a year at a time 2021-2026, 1.64m
runners), all with the weigh-in features, so none of them is the card-safe live model:
- **all:** base, parity and HRB, 316 features, fitted runner by runner (binary).
- **all+lib:** the same plus the `lib` block (Elo, Kalman rating, form lines, first-bend crossings, time since a
  break, the class-adjusted speed figure), 342 features.
- **all+lib:softmax:** the same features, fitted on the race's conditional logit (each race's runners in a softmax).

A run with five variants (the two above plus all:softmax and all+lib:pl2) reached the job's 340-minute limit on its
fourth and was re-run with these three.

**Log-loss (lower is better; SP 0.42613)**

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | All years |
|---|---|---|---|---|---|---|---|
| all | .43167 | .43162 | .43672 | .44141 | .44299 | .44435 | 0.43772 |
| all+lib | .43059 | .43087 | .43511 | .43948 | .44145 | .44262 | 0.43630 |
| all+lib:softmax | **.42992** | **.42960** | **.43398** | **.43848** | **.43986** | **.44121** | **0.43514** |
| SP | .42247 | .42258 | .42617 | .42831 | .42889 | .42964 | 0.42613 |

Both changes help in every year. The library is worth about 0.0014 and the race's loss another 0.0012, together
closing about a quarter of the gap to the SP. The model is still well behind the market and still falls further
behind it each year.

**On the 2026 Betfair bundle**

| | all | all+lib | all+lib:softmax |
|---|---|---|---|
| Log-loss (BSP 0.42752) | 0.44466 | 0.44292 | **0.44150** |
| Blend weight on the model, with the BSP | 0.119 | 0.123 | **0.130** |
| T-1, edge > 0.2: ROI at the price (90%) | +3.4% (+1.2, +5.5) | +4.1% (+1.9, +6.3) | **+4.6% (+2.4, +6.8)** |
| T-1, edge > 0.2: the same bets at BSP | +4.5% | +5.2% | **+5.3%** |
| T-1, edge > 0.3: ROI at the price (90%) | +3.7% (+1.2, +6.3) | +4.4% (+1.8, +7.0) | **+6.4% (+3.7, +9.1)** |
| T-1, edge > 0.2: CLV vs BSP | −0.6% | −0.5% | 0.0% |
| Test half only (17 May-29 Sep), T-1 raw edge > 0.2 | +3.6% (+0.4, +6.7) | +3.2% (+0.1, +6.4) | **+4.1% (+0.9, +7.3)** |
| Test half, T-1 blend edge > 0.02 | +2.5% (−0.3, +5.3) | +1.8% (−1.0, +4.5) | **+3.9% (+1.2, +6.6)** |
| Test half, T-1 raw edge > 0.2: winners / model's / price's | 3,214 / 4,776 / 3,044 | 3,147 / 4,615 / 2,986 | 3,273 / 4,681 / 3,071 |

- **The race's loss is the better model on every measure a minute before the off.** Its picks still win 1.07 times what
  the price expects and 0.70 of what the model expects: the softmax is better calibrated (the binary models' 0.67) but
  still overconfident.
- **The first traded price shows the largest numbers again** (+17% at that price for the softmax), but the live day
  of 7 Oct is why those are not to be trusted: early books are thin, and what fills there is not what the bundle
  records.
- **The poverty blend adds nothing.** The model's fitted weight runs from 0.23 for a debutant to 0.27 at 30 runs, and
  its test log-loss is the plain blend's.
- **The library's strongest features:** the class-adjusted speed figure's recency-weighted z in the race
  (`lb_cv_gsr_ewm_z`, now the model's first feature), the Kalman rating's z and gap in the race (`lb_kr_z`, `lb_kr_gap`).
  Elo, the form lines, the first-bend crossings and time since a break do not reach the top 30.

**Verdict.** all+lib:softmax is the research model from here. None of this goes live: every number above is the
model with the weigh-in, which the morning card does not have. The decisive check is still to come: the card-safe
model (no `CARD_UNSAFE` features) with the library and the race's loss, scored on the same Betfair test at T-1. Then
the library's card check (`greyhound.live --check-card`), and only then a paper test of the near-the-off rule with its
thresholds fixed in advance. Greyhound trading has been paper-only since 14:53 UTC on 7 Oct (the owner).

## 12. The forward test of 5 Oct retired; the trade-out (the owner, 2 Nov 2026)

**Why the 5 Oct test cannot pass.** Its primary rule (the first recorded price, edge > 0.2) placed no bet in 27 days
(6 Oct to 1 Nov; track run 41439030252). The first recorded book is an empty placeholder book: on 1 Nov it stood a
median 504.9 minutes before the off with a best back of 1.10 and the lay a median 533 times the back, no runner tight
(track run 41358499330). At 1.10 no dog clears 0.2, so the gate (2,000 primary bets) is never reached. The rules that
did fire chose by the chance to win and lost at the price taken: entry 1 minute before the off, edge > 0.1, 6,066 bets,
-4.86% (90% -10.87 to +1.16), the price taken 6.85% under the SP on average; first traded price, edge > 0.1, 4,255 bets,
-10.56% (-17.49 to -3.64), 12.21% under the SP. The model's log loss on 18,942 runners is 0.4513 against 0.4264 for the
BSP and 0.4307 for the book a minute out (lower is better). The paper trader (`greyhound.live`, a separate rule):
27 days, 5,091 bets, GBP24,740.31 staked, -GBP751.11 (-3.04%, 90% -10.05 to +3.98), CLV -1.15% (settle run
41440971389). The 5 Oct test keeps running (its priced runners feed the new test) but is no longer judged.

**The trade-out** (`greyhound/tradeout.py`). Take a price before the off and close it at the SP, as the horses' live
rule does: a back at the best back is worth `back / BSP - 1` a unit closed out, a lay at the best lay `1 - lay / BSP`,
whatever wins. The forecast of the SP is the horses' closing model (`model/race_book.py` ClosingModel, unchanged): a
conditional logit on the SP book from the market's price now, our price and the money matched. This is Benter's (1994)
second stage with the SP book, not the result, as the target (`reports/greyhound_literature.md` 3.3).

**The new forward test, fixed before its first day** (`greyhound/tradeout.py` RULES; `greyhound-track.yml`, the
daily run after the 5 Oct test's): frozen once on the recorded days from 6 Oct to the day before the freeze; first test
day the later of 4 Nov 2026 and the day after the freeze (amended the same day, section 13: the trader's live books,
frozen after 10 days of them); primary rule: entry at the book 1 minute before the off,
the horses' closing model, back where the expected CLV clears 0.03, a price up to 20 with the stake offered, GBP2 paper,
closed at the SP, 2% commission on each market's net. Reported, not judged: the first traded book and 3 and 5 minutes
out, the lay side, bars 0 and 0.05, the microstructure closing model, and the old edge-to-win selection closed at the
SP. Gate: after 30 test days and 1,000 primary trades, a pass only if the return per GBP staked has a 90% lower bound
above 0 (bootstrap by day); a pass goes to the owner. The win model is the 5 Oct one, never refitted, so every price
the closing model is fitted on is out of sample.

**The study** (`research/greyhound/tradeout_study.py`, `greyhound-research.yml`, read-only): the same panel, every
SP forecast walk-forward by day, the trade scored at every bar, the last 7 days a holdout read once, the learning
curve, a leak test (SPs shuffled within the race), the data audit. Results: section 13, from its CI run.

**The data.** The greyhound trader reads the books every 5 seconds in a race's last 3 minutes, where the money is
(T-3 GBP380, T-1 GBP1,993 matched a market at the median; reports/open_close/data.md 4.2), but kept none of them; the
recorder keeps a book a minute in the last hour (delayed key). From this change the trader's books are kept with
BETFAIR_RECORD=1 (`books_greyhound_trader.csv` beside the recorder's), for order-flow features the 60-second books
cannot give (literature tests 10 and 11).

## 13. The trade-out study: the first reading (greyhound-research.yml run 41447610128, 2 Nov 2026)

Panel: 70,014 runner-marks, 3,212 GB races, 26 days with priced runners and books (6 Oct to 1 Nov). Walk-forward by
day from the 8th day; development 14-25 Oct (12 days), holdout 26 Oct-1 Nov (7 days), read once. GBP2 a position,
2% commission on each market's net. Figures from the run's report (its artifact greyhound-research-1).

**1. The recorder's books are about a minute and a half old.** In a race's last 3 minutes the poll's time less the
market's last match is 77-92 seconds at the median and 162-215 seconds at the 90th percentile, on every one of the 27
days. Money matches every few seconds there (T-3 GBP380, T-1 GBP1,993 a market at the median; open_close data.md 4.2),
so this is the delayed key's delay, not a quiet market: the book stamped "1 minute before the off" shows the market
of about 2.5 minutes before it. The recorder polls a market every 60 seconds (5 polls in the last 5 minutes). The
earlier reading that a price "may be a few seconds old" (section 5) understated it. Every paper result read off these
books (the 5 Oct test's and this study's) prices a fill at a book that was already gone. No trader books exist yet
(the 5-second, live-key books are kept from this PR on). The runners' matched money appears to be empty in these books
(its features carried zero weight in every fit), so the horses' closing model runs here without its volume terms.

**2. How far each price is from the SP** (median absolute % error, all panel days): 1 minute out, best back 11.59%,
last traded 10.63%; 3 minutes out, best back 21.47%, last traded 14.16%; 5 minutes out, best back 41.43% (the book is
still hollow: the SP sits a mean 0.69 in log terms above the best back).

**3. Forecasting the SP.** Cross-entropy gain over the market's price, a race, holdout (Benter's measure, the SP book
as the target): 1 minute out, market alone 0.00073, the horses' closing model 0.00101, the market's microstructure
0.00433, with our price 0.00435, LightGBM 0.00544; 3 minutes out, LightGBM 0.02303; first traded, LightGBM 0.04116.
The leak test (SPs shuffled within the race) loses everywhere (-0.08 to -0.18). Our form model adds almost nothing
once the book is in: 1 minute out 0.00435 with it against 0.00433 without. The forecast's gain is in the spread and
the last traded price against a stale, wide book, which is not a price anyone can take.

**4. The trade closed at the SP fails.** Holdout, each rule at the bar its development days chose: all 52 rules
negative. The nearest to level: 3 minutes out, market, back, -0.19% (90% -1.01 to +0.90, 24.6 a day); 1 minute out,
market, back, -0.56% (-1.58 to +0.38, 40.6 a day). The three development rules whose interval cleared zero were
lays 1 minute out on 15-16 trades (about 2 a day); in the holdout the best of them made -2.13% (-4.82 to +0.61). The
old selection (edge to win over 0.2) closed at the SP: 1 minute out -10.17% (-11.63 to -8.75, 1,188 trades), first
traded -14.10%. The pre-registered primary rule (1 minute out, the horses' closing model, back, 0.03) in development:
34 trades, -0.70% (-5.55 to +3.62).

**5. More of the same data will not help** (the learning curve): fitted on the last 3, 7, 14 days or all of them,
the 1-minute microstructure model's gain is 0.00462, 0.00468, 0.00459, 0.00457 a race on the same scored days. The
limit is the books' age and cadence, not their number.

**The owner's decisions (2 Nov 2026).** Gather the live books first; the paper trader keeps the live key. So the
forward test of section 12 reads the trader's own books (live key, 5 seconds in the last 3 minutes, kept from this
change) at its entry marks, with the SP from the recorder's settled books, and freezes only once 10 days of them are
held (``MIN_FREEZE_DAYS``); until then the daily run reports how many days it has and writes nothing. Each test day
reports its books' feed lag, so a stale day shows. Its first test day is the later of 4 Nov and the day after that
freeze (mid-November at the earliest, if the trader runs every day).

**Next.** Once the trader's books exist, the study again on them (`greyhound-research.yml`, source `trader`): the
same tables on books that are current, then the literature's order-flow tests (reports/greyhound_literature.md
tests 10, 11, 14, 17), which only 5-second books can carry.

## 14. The card-safe full model and the log-BSP model in the trade-out study (greyhound-full-score.yml run 41450988359)

The owner (2 Nov 2026): score the card-safe full model, rerun the study, and add the horse model's log-BSP target.
`research/greyhound/score_full.py`: every GBGB run to 1 Nov (2,624,775 runs), base, parity, HRB and library blocks
(342 features built, 333 card-safe: the weigh-in features left out). One fit on every modelling race from 2019 to
5 Oct (2,018,463 + 224,278 early-stopping rows): the win model on the race's softmax (1,859 rounds), and beside it the
horse model's target, LightGBM L2 on log BSP (1,282,488 + 142,505 priced rows; it ran the full 3,000 rounds without
stopping, so it is not yet converged). Every race from 6 Oct to 1 Nov predicted: 19,038 runners, 3,440 races, all
out of sample. Then the trade-out study three times on the same days and books (the recorder's), our price from each.

**1. Against the result** (log-loss, the same 18,240 runners in 3,295 races, 26 days; lower is better):

| 5 Oct base model | Full win model | Log-BSP model | Betfair SP |
|---|---|---|---|
| 0.45140 | 0.44498 | 0.44458 | 0.42612 |

Both new models beat the base by about 0.0065, a quarter of its gap to the SP; both are still 0.018-0.019 behind it. On
the run's own SP comparison (GBGB's SP, 19,038 runners) the full win model's blend weight beside the SP is -0.172:
on these four weeks it adds nothing to the SP. The log-BSP model's predicted SP misses the BSP by 0.456 in log terms
(RMSE; 0.754 knowing only each race's average), a median 27.0%; the best back 1 minute out misses it by 11.6%
(section 13). Top features of both: the Kalman rating's z-score in the race (`lb_cv_gsr_ewm_z`), the race-strength
change, the dog's market history (`gp_mkt_w3`, and for the log-BSP model its BSP history `gp_bsp_mkt_w3`).

**2. Forecasting the SP from the book** (holdout, cross-entropy gain over the market a race; our price from each model):

| Mark | Model | Base | Full win | Log-BSP |
|---|---|---|---|---|
| 1 minute out | the horses' closing model | 0.00101 | 0.00147 | 0.00142 |
| 1 minute out | microstructure + our price | 0.00435 | 0.00451 | 0.00446 |
| 1 minute out | (microstructure alone: 0.00433) | | | |
| 3 minutes out | microstructure + our price | 0.01698 | 0.02040 | 0.02080 |
| 5 minutes out | microstructure + our price | 0.04191 | 0.05529 | 0.05645 |
| first traded | LightGBM | 0.04116 | 0.04428 | 0.04442 |

The full models forecast the SP better at every mark, most where the book is thinnest (5 minutes out, +32%). A minute
out, where the money is, our price still adds almost nothing to the book: 0.00451 with the full model against 0.00433
from the microstructure alone.

**3. The trade still fails.** Holdout (26 Oct-1 Nov), each rule at its development bar: with the full win model, one
rule of 44 has a positive point estimate, the first traded book, LightGBM, lay, bar 0.05: +2.06% (90% -1.77 to +5.45,
79 trades, 11.3 a day), its interval across zero and one of 44 tried; with the log-BSP model none of 43. The
pre-registered primary rule (1 minute out, the horses' closing model, back, 0.03) in development: base -0.70%
(34 trades), full -3.38% (90% -5.55 to -1.55, 164 trades), log-BSP -3.05% (-5.46 to -0.94, 131 trades). A better
price trades more often and loses more: its extra selections are dogs the market then lengthens.

**Reading.** The full feature set and the log-BSP target are real improvements to the model and to the SP forecast,
and they are what any greyhound trade should use from here. They do not make the trade-out pay on these books. The
limit remains the books themselves (about 90 seconds stale, a minute apart, section 13); the next test is the same
three models on the trader's live 5-second books once 10 days of them are held. The log-BSP model should be refitted
with more rounds (or a higher learning rate) before it is frozen, since it stopped at the 3,000-round cap.

## 15. The log-BSP model refitted with more rounds (greyhound-full-score.yml run 41460911545)

The owner (2 Nov 2026): refit the log-BSP model with more rounds. Same features (333 card-safe), rows (1,282,488 +
142,505 priced) and cutoff (5 Oct) as section 14; learning rate 0.03 unchanged; the cap raised from 3,000 to 20,000,
early stopping after 200 rounds without gain. The fit took about two hours on GitHub's runner.

**It ran to the cap again.** Best round 19,966 of 20,000; early stopping never fired. The early-stopping slice's RMSE of
log BSP by round: 1,000 0.4759; 3,000 0.4574; 5,000 0.4532; 10,000 0.4492; 15,000 0.4476; 20,000 0.4469. The last
5,000 rounds gained 0.0007, so the fit is close to flat; more rounds at this rate would add little, a higher learning
rate would get there faster.

**Against the result** (the same 18,240 runners, 3,295 races, 26 days; lower is better): the log-BSP model 0.44368
(0.44458 at 3,000 rounds); the full win model 0.44498, the 5 Oct base 0.45140, the Betfair SP 0.42612 (unchanged: the
same fits). The refit is now the best of the three, 0.0013 ahead of the win model and 0.0176 behind the SP. Its
predicted SP misses the BSP by 0.4451 in log terms (RMSE; 0.4560 before), a median 26.6% (27.0% before).

**Forecasting the SP from the book** (holdout, cross-entropy gain over the market a race, our price from the log-BSP
model, 3,000 then 20,000 rounds): 1 minute out, the horses' closing model 0.00142 then 0.00158, microstructure + our
price 0.00446 then 0.00450; 3 minutes out 0.02080 then 0.02164; 5 minutes out 0.05645 then 0.05874; first traded,
LightGBM 0.04442 then 0.04487. Better at every mark, a little.

**The trade still fails.** Holdout, each rule at its development bar: one rule of 43 positive, 1 minute out,
microstructure, lay, bar 0.05: +2.46% (90% -3.41 to +6.49) on 15 trades, its interval across zero; none with a lower
bound above zero. The primary rule in development: -3.11% (90% -5.85 to -0.53, 169 trades; -3.05% at 3,000 rounds).
The market-only models, which do not read our price, also moved a little between the two runs, so the recorder's day
files changed in between (the latest day's settled books); not every small difference here is the refit's.

**Reading.** The refit is the better log-BSP model and the one to freeze when the live books allow (section 13). It
does not change the trade-out's verdict on the recorder's books.

## 16. QA of sections 12-15 and the roadmap (2 Nov 2026)

A full QA and review (an independent read of the code, `reports/greyhound_roadmap.md` section 2) found no look-ahead
in the models, and these that bear on the sections above:
- **The forward test of section 12 cannot start as built**: since 2 Nov the greyhound trader runs in
  `paper-models.yml` on the delayed key without `BETFAIR_RECORD`, so the trader books it waits for are never written
  (and would be stale). The roadmap's Phase 0 replaces them.
- **The holdout of sections 13-15 was not read once**: it was recomputed on every run, three price models each. The
  verdicts stand, because every rule failed and selection on the holdout can only flatter.
- **"Closed at the SP" assumed a perfect green-up**; a BSP lay is sized by liability, so the realisable return has the
  same expectation only if the BSP is calibrated, and wider intervals than reported.
- **Trades a day counted active days only**, so the bar filter "at least once a day" did nothing.
The roadmap (`reports/greyhound_roadmap.md`) sets out the fixes, the feature and model work, and the gated path from
here.

## 17. Re-timed books and the horse model's settlement (greyhound-research.yml run 41508653282)

The study on the recorder's books placed at each snapshot's last match (`source="retimed"`, roadmap 0.1) with
positions taken and settled as the horse trader's (to win GBP100 down the three best prices, each back's winnings laid
at the SP, settled by the result; roadmap section 9). Panel: 53,460 runner-marks, 3,205 races, 26 days (fewer marks
than section 13's 70,014: a 60-second poll does not fall in every true one-minute window). Development 14-25 Oct,
holdout 26 Oct-1 Nov.

**At the true last minute the book is the SP.** One minute before the off the geometric mid of the best back and lay
sits a mean 0.14% from the BSP in log terms (-0.0014; section 13's stale stamps: -0.104), the best back a median 8.58%
from it (11.59% before). The forecast of the move to the SP gains nothing there: 0.0001 a race over the market for the
microstructure models, -0.00001 for the horses' closing model. Three and five minutes out a move remains (gain 0.0078
and 0.0308, LightGBM, holdout).

**The trade, holdout, each rule at its development bar** (trades a day and the bootstrap over every scored day):
3 of 44 rules positive, none with a 90% lower bound above zero. The nearest to level: the first traded book,
LightGBM, lay, +0.67% (90% -1.02 to +2.03, 40.6 a day); 1 minute out, microstructure, back, +0.03% (-1.17 to +1.31,
14.6 a day). The old selection (edge to win over 0.2) closed at the SP: 1 minute out -3.60% (-4.30 to -2.82),
first traded -15.40%. The pre-registered primary rule took 4 positions in 12 development days: at the true last
minute almost no dog's price clears 3% over the forecast SP.

**Reading.** Read on books that are current, the market a minute before the off has already found the SP, and
earlier, where a move is left to forecast, it does not pay after the spread and commission on these models. The
live-key books (roadmap 0.2) will say the same or not with 5-second books and volume; the levers left are the late
information (the weigh-in, roadmap L1) and the cross-market signals (M5).

## 18. The fast loop's first batch (greyhound-fastloop.yml run 41595386223, 3 Nov 2026)

**The loop's timings** (measured on the run): the feature cache built once in 20 minutes 24 seconds (every GBGB run to
2 Nov, every block, the weigh-in features and the BSP, from June 2020); then ten experiments side by side, each fitted
and scored in 46 to 167 seconds (the job, with the cache's download, 1 to 6 minutes); the whole batch from the cache to
the table in about 7 minutes. A batch on a cache already built: about 7 minutes, against two to three hours before.

**The batch** (quick recipe: fit January 2024 to May 2026 at a learning rate of 0.15; test every race from 1 June to
2 November 2026, out of sample; log-loss against the result, lower is better):

| Experiment | Features | Objective | Log-loss | vs baseline | On BSP races: model / BSP | Model's weight beside the BSP |
|---|---|---|---|---|---|---|
| base_card_safe (baseline) | 333 | softmax | 0.44687 | | 0.44749 / 0.42838 | -0.009 |
| with_weighin | 342 | softmax | 0.44581 | -0.00106 | 0.44640 / 0.42838 | 0.062 |
| plus_race_rel2 | 423 | softmax | 0.44736 | +0.00049 | 0.44803 / 0.42838 | -0.002 |
| plus_first_bend | 339 | softmax | 0.44685 | -0.00002 | 0.44745 / 0.42838 | 0.010 |
| plus_style_draw | 337 | softmax | 0.44692 | +0.00005 | 0.44752 / 0.42838 | 0.013 |
| plus_all_blocks | 433 | softmax | 0.44650 | -0.00037 | 0.44709 / 0.42838 | -0.005 |
| weighin_all_blocks | 442 | softmax | 0.44651 | -0.00036 | 0.44716 / 0.42838 | 0.024 |
| objective_pl2 | 333 | Plackett-Luce | 0.44647 | -0.00040 | 0.44712 / 0.42838 | -0.006 |
| objective_binary | 333 | binary | 0.44740 | +0.00053 | 0.44803 / 0.42838 | -0.008 |
| **objective_logbsp** | 333 | log BSP | **0.44343** | **-0.00344** | 0.44400 / 0.42838 | -0.018 |

**Reading.** The horse model's target, log BSP, is the best by a distance (0.0034, ten times the next). The weigh-in
helps (0.0011). The three new blocks add little on their own; together 0.0004; race_rel2 alone is worse. The
Plackett-Luce objective helps a little, the binary one hurts. No experiment adds to the BSP: every model's weight in a
blend with it is about zero, and the blend's log-loss on the second half of the test is the BSP's within 0.0001. These
differences were not yet given intervals; from the next batch the table carries a paired 90% interval against the
baseline by day.

**The trade-out study on three of them** (re-timed books, the horse model's settlement): on the log-BSP model's prices
one rule of 44 in the holdout has a 90% lower bound above zero, a lay at the first traded book at bar 0.08, +9.78%
(0.03 to 18.56) on 44 trades; one in 44 is about what chance gives at that level, so it is a lead to test on new days,
not a finding.

## 19. The fast loop's second batch (greyhound-fastloop.yml run 41598761809, 3 Nov 2026)

**The batch** (the cache of section 18, reused in 2 seconds; quick recipe unless marked; every race 1 June to
2 November 2026, out of sample; log-loss against the result, lower is better; the interval is the paired difference
from the baseline on the same runners, 90%, a bootstrap over days):

| Experiment | Features | Objective | Rounds | Log-loss | vs logbsp (90%) | On BSP races: model / BSP | Model's weight beside the BSP | Seconds |
|---|---|---|---|---|---|---|---|---|
| logbsp (baseline) | 333 | log BSP | 1,262 | 0.44343 | | 0.44400 / 0.42838 | -0.018 | 269 |
| logbsp_weighin | 342 | log BSP | 897 | 0.44310 | -0.00033 (-0.00071 to +0.00005) | 0.44364 / 0.42838 | -0.010 | 207 |
| logbsp_blocks (race_rel2, first_bend, style_draw) | 433 | log BSP | 1,336 | 0.44296 | -0.00047 (-0.00088 to -0.00007) | 0.44351 / 0.42838 | -0.000 | 431 |
| logbsp_fb_sd (first_bend, style_draw) | 343 | log BSP | 945 | 0.44301 | -0.00042 (-0.00081 to -0.00007) | 0.44362 / 0.42838 | 0.009 | 161 |
| **logbsp_weighin_fb_sd** | 352 | log BSP | 954 | **0.44286** | **-0.00057 (-0.00098 to -0.00020)** | 0.44341 / 0.42838 | 0.006 | 298 |
| softmax_ref | 333 | softmax | 101 | 0.44687 | +0.00343 (+0.00289 to +0.00401) | 0.44749 / 0.42838 | -0.009 | 47 |
| pl2_weighin | 342 | Plackett-Luce | 140 | 0.44553 | +0.00210 (+0.00150 to +0.00268) | 0.44610 / 0.42838 | 0.072 | 54 |
| logbsp_confirm (2021 on, 0.05) | 333 | log BSP | 8,000 (the cap) | **0.44247** | **-0.00096 (-0.00127 to -0.00066)** | 0.44306 / 0.42838 | -0.005 | 3,064 |

**Reading.**
- The log-BSP target's lead over softmax holds with an interval: 0.0034 (0.0029 to 0.0040).
- The first-bend and style-draw blocks help it (0.0004, the interval clear of zero). The weigh-in on its own is not
  clear of zero (-0.0003). Together they give 0.0006 (0.0002 to 0.0010). The weigh-in is published live at only 12
  tracks (greyhounds.today), so an experiment with it overstates what a live model would have elsewhere.
- The confirm recipe beats the quick one by 0.0010 (0.0007 to 0.0013). It ran into its 8,000-round cap, so a longer
  fit might gain more. It took 51 minutes and held the table back that long, which is fixed below.
- No experiment adds to the BSP. The Benter gain is between 0 and +0.00009 in every row, and the model's weight beside
  the BSP is between -0.018 and +0.072. The best model's log-loss on BSP races (0.44306) is 0.0147 behind the BSP's
  (0.42838).

**The trade-out study on the confirm model's prices** (re-timed books, the horse model's settlement; 27 panel days from
6 October to 2 November 2026, 20 scored: 13 development, 7 holdout):
- One minute before the off the book is the SP. Every closing model's cross-entropy gain over the market at off_1 is
  0.00013 a race or less, on development and holdout days alike.
- At the first traded book the forecast is strong: the GBM's gain is 0.044 a race on development and holdout days
  alike (R² against no change 0.77 and 0.79). The shuffled null model is negative everywhere (-0.12 at the first
  traded book), so the gain is not a leak.
- Of 52 rules read once on the holdout (40 without the shuffled null and the win-edge rule), one has its 90% lower
  bound above zero: a lay at the first traded book on the GBM's forecast at bar 0.03, +3.14% (+0.22 to +5.79) on 169
  trades over 7 days.
- Its mean CLV is negative (-1.37%) and its held return is -3.04%, so the profit came from the results, not from the
  price moving. One rule in 40 at the 90% level is what chance gives.
- Section 18's lead was the same kind of rule (a first traded lay, there at bar 0.08 on the quick log-BSP prices). Two
  readings on overlapping days are not two tests. Neither is a finding; both go to the forward test on new days.

**What changed in the loop** (greyhound/fastloop.py, greyhound-fastloop.yml):
- **Two lanes.** The quick experiments' table is written as soon as they finish. The slow lane (the confirm recipe and
  the study) gets its own table of every experiment after, so a slow job no longer holds up the quick answer.
- **A cheaper confirm recipe.** It now fits from 2022 at 0.08 with a 6,000-round cap and a 25-minute budget, stopping
  at the best round so far. Each result says what stopped it: early stopping, the cap or the time.
- **Batch 3** (research/greyhound/experiments/current.json) asks where the confirm recipe's gain comes from (the
  slower rate on the quick window, or the longer window at the quick rate), tests two tree sizes, and runs the best
  features on the new confirm recipe with the study.

## 20. The fast loop's third batch (greyhound-fastloop.yml run 41606065187, 3 Nov 2026)

**The two lanes worked.** The quick lane's six experiments and its table were done at 08:56 UTC, 7 minutes after the
push. The slow lane (best_confirm) was still fitting. In batch 2 the table waited 56 minutes on one slow job.

**The quick lane** (fitted on January 2024 to May 2026 unless marked; tested on every race from 1 June to 2 November
2026; log-loss against the result, lower is better; paired against the baseline, 90%, by day):

| Experiment | What changes | Rounds | Log-loss | vs logbsp (90%) | On BSP races: model / BSP | Seconds |
|---|---|---|---|---|---|---|
| logbsp (baseline) | quick recipe (learning rate 0.15, 63 leaves) | 1,262 | 0.44343 | | 0.44400 / 0.42838 | 262 |
| logbsp_weighin_fb_sd | + the weigh-in, first_bend, style_draw | 954 | 0.44286 | -0.00057 (-0.00096 to -0.00018) | 0.44341 / 0.42838 | 290 |
| **logbsp_lr08** | learning rate 0.08 | 2,675 | **0.44274** | **-0.00069 (-0.00100 to -0.00038)** | 0.44332 / 0.42838 | 324 |
| logbsp_from2021 | fitted from 2021 at 0.15 | 1,340 | 0.44323 | -0.00020 (-0.00055 to +0.00014) | 0.44381 / 0.42838 | 325 |
| logbsp_leaves127 | 127 leaves, 100 a leaf | 685 | 0.44361 | +0.00018 (-0.00018 to +0.00054) | 0.44417 / 0.42838 | 198 |
| logbsp_leaves31 | 31 leaves, 500 a leaf | 1,484 | 0.44318 | -0.00025 (-0.00055 to +0.00004) | 0.44376 / 0.42838 | 264 |

**Reading.**
- **Most of the confirm recipe's gain is the learning rate.** Batch 2's confirm recipe (2021 on, 0.05) gained 0.00096
  in 51 minutes. The quick window at 0.08 gains 0.00069 (72% of it) in 324 seconds. The longer window at the quick
  rate gains 0.00020, and its interval crosses zero.
- **Tree size:** neither change is clear of zero. Larger trees lean worse and smaller ones lean better.
- **The feature gain is repeatable.** The weigh-in plus first_bend and style_draw give the same 0.00057 as in batch 2,
  and the interval is still clear of zero.
- **No model adds to the BSP.** The Benter gain is between -0.00001 and +0.00002 in every row, and the model's weight
  beside the BSP is between -0.020 and +0.006.

**The slow lane: best_confirm** (the weigh-in, first_bend and style_draw on the new confirm recipe: fitted on 2022 to
May 2026 at 0.08 with a 25-minute budget). The table came at 09:20 UTC, 31 minutes after the push. In batch 2 the slow
job alone took 55 minutes.

| Experiment | Rounds | Log-loss | vs logbsp (90%) | On BSP races: model / BSP | Seconds |
|---|---|---|---|---|---|
| best_confirm | 5,438 | **0.44225** | **-0.00119 (-0.00153 to -0.00086)** | 0.44283 / 0.42838 | 1,534 |

- It is the best model so far.
- It took 1,534 seconds against a 1,500-second fit budget and stopped under the 6,000-round cap, so the budget most
  likely ended it. The result file records which.
- Its study (the same 27 days and the same development/holdout split as section 19):
  - Its price adds to the SP forecast much as batch 2's confirm price did: at first traded 0.0079 a race in
    development and 0.0049 in the holdout; at 5 minutes out 0.0131 and 0.0146.
  - No holdout rule of 52 has its 90% lower bound above zero.
  - Section 19's lead (a lay at first traded) comes in at bar 0.02 here: +2.07% (-0.02 to +3.85) on 213 trades, with a
    mean CLV of -1.69%. So it fades, as a result driven by the results rather than the prices would.

**The measure changes from batch 4** (the owner, 3 Nov 2026: "we are just trying to trade an earlier price vs. Betfair
SP ... get money on at higher prices that are expected to shorten at BFSP"):
- The SP is the market's best price. Whether a model beats it, the Benter column of sections 18-20, is not the
  trade-out's question, so the fast loop no longer ranks by it (the results JSON keeps it).
- What the trade-out needs from our price is how much it adds to the forecast of the SP from the book on offer. At
  each entry mark (first traded, 5 and 3 minutes out), the same conditional logit is fitted without and with our
  price, walk-forward by day on the recorded books from 6 October (re-timed). The table leads with the difference, a
  race, and its paired 90% interval against the baseline. Log-loss against the result comes second.
- **What it showed in batch 2's study** (the confirm log-BSP prices; the SP book's cross-entropy a race; "our price
  adds" is the conditional logit with our price less the same one without it):

| Mark | Days | Our price adds | Gain over the market price |
|---|---|---|---|
| First traded, development | 13 | 0.0076 | 0.0391 |
| First traded, holdout | 7 | 0.0048 | 0.0374 |
| 5 minutes out, development | 13 | 0.0127 | 0.0288 |
| 5 minutes out, holdout | 7 | 0.0141 | 0.0325 |
| 3 minutes out, development | 13 | 0.0002 | 0.0041 |
| 3 minutes out, holdout | 7 | 0.0003 | 0.0069 |
| 1 minute out, development and holdout | 13 and 7 | -0.00001 and 0.00000 | 0.0001 (the book is the SP) |

- So our price does help call which runners will shorten, at first traded and 5 minutes out. The trades have not yet
  made it pay. Of the 40 holdout rules, 37 have a negative mean CLV. The other three are lays 1 minute out (+0.29% to
  +0.62%), and all three still returned less than zero.
- The books before 1 minute out are the recorder's, about 90 seconds stale. The live-key recorder's 5-second books
  wait on the paper-models runner being registered.
- **Why calling the move has not paid: the price on offer to back is usually shorter than the SP.** From the studies'
  section 1 (all 27 panel days, re-timed books, best back price against the SP):

| Mark | Runners | Mean log(SP / best back price) | Median gap |
|---|---|---|---|
| First traded | 18,360 | +0.363 | 21.6% |
| 5 minutes out | 12,322 | +0.452 | 27.1% |
| 3 minutes out | 12,744 | +0.141 | 13.0% |
| 1 minute out | 12,290 | +0.045 | 8.6% |

  - A greyhound book before the last minutes is thin: most of its money is matched in the last three minutes:
    a median GBP380 matched at 3 minutes out and GBP1,993 at 1 minute out (reports/open_close/data.md 4.2). The best back price on offer is mostly well under the SP.
  - Taking it gives away more than our forecast of the move gains. The backs taken 5 minutes out on the micro forecast
    have a mean CLV of -6.18% in the holdout.
  - The horse trader takes the price because horse books are deep by the morning. For greyhounds, the test this
    points to is asking for the price: a back offered at our forecast SP plus a margin, left in the book to be
    matched, or not. It can be simulated on the live recorder's 5-second books with traded volume, and only roughly
    on the recorder's.

## 21. The fast loop's fourth batch: ranked by calling the move (greyhound-fastloop.yml run 41610771793, 3 Nov 2026)

**The first batch ranked by what the trade-out needs** (section 20's change of measure). Every experiment was fitted
on January 2024 to May 2026 on the new quick recipe (learning rate 0.08). Columns:
- **Our price adds:** how much the experiment's price adds to the forecast of the SP from the book at first traded,
  5 and 3 minutes out. It is the SP book's cross-entropy a race: the conditional logit without our price less the same
  with it. It is walk-forward by day on the recorded books from 6 October, re-timed. Higher is better.
- **vs logbsp:** the paired difference from the baseline over those three marks, 90% by day. Negative is better.
- **Log-loss:** against the result, every race from 1 June to 2 November 2026, paired the same way.

| Experiment | Rounds | Our price adds: first traded / 5 min / 3 min | vs logbsp (90%) | Log-loss | vs logbsp (90%) | Seconds |
|---|---|---|---|---|---|---|
| logbsp (baseline: card-safe, 63 leaves) | 2,675 | 0.00665 / 0.01304 / 0.00024 | | 0.44274 | | 521 |
| **logbsp_lr05** (learning rate 0.05) | 8,000 (the cap) | **0.00694 / 0.01363 / 0.00028** | **-0.00029 (-0.00040 to -0.00017)** | 0.44241 | -0.00032 (-0.00056 to -0.00009) | 1,241 |
| **logbsp_weighin_fb_sd** (+ the weigh-in, first_bend, style_draw) | 3,432 | **0.00691 / 0.01354 / 0.00028** | **-0.00026 (-0.00037 to -0.00015)** | 0.44240 | -0.00034 (-0.00063 to -0.00006) | 736 |
| logbsp_fb_sd_leaves31 | 5,270 | 0.00663 / 0.01332 / 0.00025 | -0.00008 (-0.00018 to +0.00002) | 0.44242 | -0.00032 (-0.00058 to -0.00005) | 878 |
| logbsp_fb_sd (+ first_bend, style_draw) | 2,944 | 0.00660 / 0.01330 / 0.00024 | -0.00005 (-0.00018 to +0.00008) | 0.44284 | +0.00011 (-0.00015 to +0.00036) | 701 |
| logbsp_ff05 (half the features a tree) | 4,080 | 0.00661 / 0.01324 / 0.00024 | -0.00003 (-0.00013 to +0.00006) | 0.44275 | +0.00002 (-0.00026 to +0.00030) | 514 |
| logbsp_ff09 (more features and rows a tree) | 2,613 | 0.00658 / 0.01322 / 0.00023 | -0.00002 (-0.00013 to +0.00009) | 0.44285 | +0.00012 (-0.00012 to +0.00036) | 740 |
| softmax_fb_sd (the win objective) | 170 | 0.00502 / 0.00981 / 0.00014 | +0.00163 (+0.00132 to +0.00192) | 0.44562 | +0.00289 (+0.00237 to +0.00338) | 137 |

**Reading.**
- **The two measures rank the experiments alike.** What forecasts the result better also calls the move better.
- **The log-BSP target matters most for the move.** The softmax model's price adds a quarter less at first traded
  (0.0050 against 0.0067) and the interval is well clear.
- **A slower learning rate helps.** At 0.05 the fit ran to its 8,000-round cap, so more rounds may help further. The
  weigh-in helps too.
- **The first-bend and style-draw blocks no longer help on their own** at 0.08. They did at 0.15 in batches 2 and 3,
  so their gain was partly making up for the fast rate.
- **Neither tree size nor subsampling changes anything.**
- **The run took 26 minutes.** The shortening score adds about 8 to 10 minutes a job (S3 reads and the walk-forward).
  The longest job was 24 minutes against a 30-minute limit, so the limit goes to 60 (batch 5 fits at 0.05 to 10,000 rounds).

## 22. The limit-order study (greyhound-limit-study.yml run 41613570934, 3 Nov 2026)

The owner (3 Nov 2026): "build the limit-order study ... take prices based on the behaviour of the greyhound market.
I am open to any strategy." (research/greyhound/limit_study.py)

**The setup.**
- **Orders:** at first traded, 5 and 3 minutes out, every runner gets each order. It rests in the book at its limit
  and lapses at the off unless matched. The orders are:
  - our forecast SP plus or minus 0/5/10/20% (`ask` backs above it, `bid` lays below it);
  - the head of the queue (`join`);
  - 1/3/5 ticks off the best price (`ticks`).
- **Gates:** all runners, those forecast to shorten, or those forecast to drift.
- **Closing:** at the SP, as the horse trader closes, with 2% commission on each market's net.
- **Books:** the recorder's, re-timed; our price is the 5 October model's.
- **Days:** 53,460 runner-marks. Scored days 14 October to 1 November: 12 development, 7 holdout. 2 November had no
  model prices.
- **Selection:** each family's best rule on the development days was read once on the holdout, 36 rules in all.
- **Two match rules:**
  - `through`: a later book offers the other side at the limit.
  - `traded`: a later last-traded price reaches the limit.

  For a lay resting at P, a trade at or below P means backers would have matched our offer first. So `traded` is
  sound for the price, but it says nothing about the size. The full stake is assumed matched, and the books are 60
  seconds apart.

**1. The market's behaviour: greyhound prices drift from the first book to the SP.**

| From | Runners drifting (SP above the mid then), by price band | The favourite | 6th and beyond |
|---|---|---|---|
| First traded | 66.3% to 93.9% (mean log SP/mid +0.24 to +0.80) | 81.7% | 76.2% |
| 3 minutes out | 49.2% to 76.5% (+0.01 to +0.29) | 53.3% | 75.3% |

- The shortest runners drift most from first traded: 93.9% of those under 2.0 drift, by a mean log of 0.80.
- By 3 minutes out the favourites are near their SP. Outsiders still drift: the 6th and beyond go off about 35% longer
  on average (mean log +0.30).
- The 5-minute books look thin: 67% to 91% drift, and the best lay is 3 to 11 times the SP (mean log 1.06 to
  2.36). They are often older than
  their time says, so read them as early books.

**2. Backing at higher prices loses. Every back rule is negative on the holdout.**
- The resting backs are matched when the dog drifts. Matched backs had a mean CLV between -1.7% and -8.8%. The same
  orders unmatched would have had +6% to +59% at their limit. That gap is the adverse selection.
- Asking for a price above the market gets matched only by the runners that keep drifting.

**3. Laying the early book: two rules hold their development result on the holdout.**

| Rule (first traded, runners forecast to drift, matched by the traded price) | Days | Trades a day | Return (90%) | Mean CLV | Held to the result |
|---|---|---|---|---|---|
| Lay at 80% of our forecast SP: development | 12 | 160 | +1.24% (+0.44 to +1.93) | +4.59% | +1.20% |
| Same: holdout | 7 | 150 | **+1.92% (+0.75 to +3.01)** | **+6.00%** | +1.20% |
| Lay at the best back price now (`join`): development | 12 | 120 | +0.92% (-0.21 to +2.80) | +0.63% | -0.25% |
| Same: holdout | 7 | 124 | +0.71% (+0.16 to +1.29) | +2.95% | +1.95% |

- These are the first trade-out rules whose CLV is positive as well as their return. Earlier leads came from the
  results, with a negative CLV.
- They fit the behaviour in section 1. Early greyhound backers take prices shorter than the SP. A layer resting
  slightly short of our forecast is matched by them and closes at a longer SP.
- **Not yet a finding:**
  - 2 of 36 holdout rules clear zero, and chance gives about 2. The development selection and the positive CLV make
    them more than that, but not proof.
  - Both rely on the `traded` match. With `through`, neither is in the development top 40.
  - The size matched is unknown. The early books are thin, and a GBP2 trade at our price does not mean our GBP25 stake
    (GBP100 liability) would have been matched.

**What it needs next:**
- **The traded volume at each price.** The live-key recorder (greyhound/book_record.py) asks Betfair for each
  runner's traded volume, and can keep the volume by price in a file of its own. It waits on the paper-models runner.
- **The best model's prices.** The study can run on them through the fast loop, as the trade-out study does.
- **A forward test fixed before its first day.** The two lay rules, frozen as written above, scored on each new day's
  recorded books, read only after enough days.

## 23. Every strategy: the full grid (greyhound-limit-study.yml run 41617988279, 3 Nov 2026)

The owner (3 Nov 2026): "I am also fine with a lay to back strategy. Evaluate all trading strategies, keep an open
mind."

**The setup.**
- **Entries:** take the price now; ask/bid at our forecast SP plus or minus 0/5/10/20%; join the queue; 1/3/5 ticks
  off the best price. Each as a back or a lay, at first traded, 5 or 3 minutes out.
- **Exits:** close at the SP, or green up at the market in the runner's last book before the off (`last`: lay then
  back, or back then lay).
- **Gates:** all runners; forecast to shorten or to drift; the favourite now; short-priced (mid under 3.0);
  long-priced (8.0 or more).
- **Fills:** `through` or `traded`.
- **Data:** the same 27 days, books, model (the 5 October one), development and holdout days as section 22.
- **Selection:** the best rule of each of 96 families (mark x order x side x fill x exit) on the development days,
  read once on the holdout.

**The verdict on the grid as a whole: nothing beyond chance.**
- 3 of the 96 holdout rules have their 90% lower bound above zero; chance alone gives about 4.8.
- Every back strategy but one loses: 47 of the 48 back families are at or below zero on the holdout. Whatever way in, the backs
  matched are the drifters.
- Lays do better: 10 of 48 above zero.

**One family is consistent: lay the short-priced runners in the first book, at 80% of our forecast SP.** It closes
the same whether it is greened up before the off (lay to back) or at the SP. Matched by the traded price:

| Rule (first traded, lay at 80% of forecast SP, traded fill) | Days | Trades a day | Return (90%) | Mean CLV | Held to the result |
|---|---|---|---|---|---|
| Short-priced, lay to back: development | 12 | 52.3 | +3.34% (+2.46 to +4.32) | +7.09% | +2.60% |
| **Short-priced, lay to back: holdout** | 7 | 37.9 | **+3.31% (+1.75 to +4.85)** | **+7.62%** | +7.10% |
| Short-priced, lay to SP: development | 12 | 52.3 | +2.76% (+1.61 to +3.81) | +7.09% | +2.60% |
| The favourite, lay to SP: development | 12 | 33.0 | +3.31% (+1.45 to +5.30) | +6.83% | +4.89% |
| The favourite, lay to SP: holdout | 7 | 28.7 | +2.19% (-1.43 to +7.13) | +5.56% | +8.21% |
| All runners, lay to back: development | 12 | 162.4 | +1.12% (+0.85 to +1.38) | +4.68% | +1.39% |

**Reading.**
- **It matches the market's behaviour** (section 22). From the first book, 93.9% of runners priced under 2.0 drift to
  the SP, by a mean log of 0.80. The shortest runners open shortest against their SP. A layer resting a little under
  our forecast is matched by the early backers, and the price runs out by the off.
- **Lay to back is as good as lay to SP, with less variance.** The greened-up return does not depend on the result,
  so its interval is about a third as wide on all runners (+0.85 to +1.38 against +0.42 to +1.90).
- **It still rests on the `traded` fill and on unknown size.**
  - Under `through` (a later book offering the price) this family is not among the best.
  - With most of a greyhound market's money arriving in its last three minutes, a trade at our price in the first
    book may be a few pounds against our GBP25 stake (GBP100 liability) to win GBP100.
  - The grid as a whole is at chance. This is a lead to test forward, not an edge.

**What decides it:**
1. **Size.** The live-key recorder's traded volume a runner, every 5 seconds from 15 minutes out, shows how much was
   matched at or below our price after we would have posted it. It waits on the paper-models runner (the Vultr
   server, being set up).
2. **A forward test fixed before its first day.**
   - The rule: at first traded, lay every runner priced under 3.0 at 80% of the forecast SP.
   - Exits: green up in the last book, and separately close at the SP.
   - Fills: matched by the traded price, and separately by the book, recorded both ways.
   - Scored on each new day's recorded books, read only after 20 days.
3. **The same grid on the best fast-loop model's prices** (batch 5's slow lane).

## 24. The fast loop's fifth batch, and the early lays on a better price (greyhound-fastloop.yml run 41614579036, 3 Nov 2026)

**The quick lane.** Fitted on January 2024 to May 2026; the measures as in section 21. Runs at 0.05 stopped by early
stopping, under the 10,000-round cap.

| Experiment | Rounds | Our price adds: first traded / 5 min / 3 min | vs logbsp (90%) | Log-loss | vs logbsp (90%) | Seconds |
|---|---|---|---|---|---|---|
| logbsp (baseline: card-safe, 0.08) | 2,675 | 0.00665 / 0.01304 / 0.00024 | | 0.44274 | | 531 |
| **weighin_lr05** (+ the weigh-in, 0.05) | 7,515 | **0.00708 / 0.01393 / 0.00029** | **-0.00044 (-0.00054 to -0.00034)** | **0.44207** | -0.00067 (-0.00094 to -0.00039) | 1,193 |
| weighin_fb_sd_lr05 (+ first_bend, style_draw) | 7,650 | 0.00703 / 0.01381 / 0.00029 | -0.00038 (-0.00049 to -0.00028) | 0.44207 | -0.00066 (-0.00094 to -0.00039) | 1,031 |
| card_lr05 (card-safe, 0.05) | 8,917 | 0.00694 / 0.01360 / 0.00028 | -0.00029 (-0.00042 to -0.00017) | 0.44238 | -0.00035 (-0.00059 to -0.00012) | 1,604 |

- **The best price yet** at calling the move to the SP is the weigh-in at 0.05. Its log-loss (0.44207) is the best of
  any fit so far, 0.0002 better than section 20's slow best_confirm (0.44225), in about 20 minutes.
- The first-bend and style-draw blocks add nothing on top of it.
- The weigh-in is published live at 12 tracks only. The card-safe model at 0.05 is the one to serve elsewhere.

**The slow lane: the weigh-in at 0.05 through both studies** (its prices on the same 27 days, re-timed books; this time
2 November is scored too: 13 development days, 7 holdout):
- **Trade-out study (taking the price):** no holdout rule of 52 has its 90% lower bound above zero.
- **Limit-order study:** run on the section 22 version (exit at the SP; gates all, shorten, drift). 2 of 36 holdout
  rules clear zero, against about 1.8 expected by chance. They are the same two as section 22, now on a better price:

| Rule (first traded, forecast-drift runners, traded fill, closed at the SP) | Days | Trades a day | Return (90%) | Mean CLV | Held |
|---|---|---|---|---|---|
| Lay at 80% of the forecast SP: development | 13 | 169.6 | +1.90% (+1.09 to +2.69) | +6.83% | +1.37% |
| Lay at 80% of the forecast SP: **holdout** | 7 | 158.7 | **+1.42% (+0.34 to +2.50)** | **+9.07%** | +0.80% |
| Lay at the best back price (join): development | 13 | 115.8 | +1.06% (-0.10 to +2.91) | +0.97% | -0.71% |
| Lay at the best back price (join): **holdout** | 7 | 119.9 | +0.76% (+0.19 to +1.40) | +3.29% | +0.32% |

- Compared with section 22 on the 5 October model (whose development days stop a day earlier, 12 against 13), the 80% lay's development return rises from +1.24% to +1.90%. Its
  holdout CLV rises from +6.0% to +9.1%, and its holdout return falls from +1.92% to +1.42%, both intervals clear of
  zero.
- Better prices make the early lay's prices better. The study is still at chance overall, and the size matched is
  still unknown (sections 22 and 23).
- The live-key recorder started on the new server on 3 November at about 10:41 UTC (`ashcroft-paper`,
  78.141.242.224). Its books, with each runner's traded volume, are what will measure the size.

## 25. Trading the move: green up, or take profit early (greyhound-limit-study.yml runs 41633779822 and 41635047467, 3 Nov 2026)

The owner, 3 Nov: "we want to trade out and take profit whenever we get the movement correct. I dont care if the dog
wins or not. We are trading the price move." The study now has three kinds of exit for every matched entry:

- `sp`: closed at the SP as the horse trader closes, settled by the result. In expectation this is the CLV, but the
  P&L of each race depends on the result.
- `last`: greened up in the runner's last recorded book before the off (a lay backed at the best back price there, a
  back laid at the best lay price). The P&L is the same whatever the result.
- `tp2`, `tp5`, `tp10` (take profit): once the entry is matched, a closing order 2, 5 or 10 ticks better than the
  entry price, matched where a later book reaches it by the same fill rule. Where the move never comes, it greens up
  as `last`. Each entry's fill time is read from the books' path: at the mark if taken, else the first later book
  that reaches the limit.

Panel: 55,635 runner-marks, 6 Oct to 2 Nov; development days 14 to 26 Oct (13), holdout 27 Oct to 2 Nov (7). That
holdout has now been read by several runs of this study, so it describes; it is no longer a clean test.

**The exits on the same entries** (run 41635047467, section 2b): 643 entry rules with a trade a day on the development
days, each exit's return against `last` on the same entries.

| exit | dev: median vs last (points) | dev: better than last | holdout: median vs last | holdout: better than last |
|---|---|---|---|---|
| sp | +0.76 | 89.9% | +0.98 | 88.2% |
| tp10 | -0.03 | 6.1% | -0.03 | 10.1% |
| tp5 | -0.10 | 6.2% | -0.11 | 7.9% |
| tp2 | -0.19 | 7.2% | -0.19 | 7.8% |

The early lay (first traded, lay the forecast drifters at 80% of the forecast SP, traded fill), return on the money at
risk:

| exit | development | holdout |
|---|---|---|
| sp | +1.44% | +1.62% |
| last | +1.06% | +1.25% |
| tp10 | -0.56% | -0.19% |
| tp5 | -1.12% | -0.82% |
| tp2 | -1.53% | -1.26% |

What this says:

- **Taking profit early loses.** It cuts the winners short and leaves the losers as they were. A drift that reaches 2
  ticks usually goes further, so closing there gives up the rest of the move, and the trades that go the wrong way
  still green up at the last book for a loss. Take profit did worse than `last` on about 90% of entry rules, in both
  halves, and the shorter the target the worse it did. It turned the early lay from a gain into a loss.
- **Greening up at the last book costs about 0.4 points against closing at the SP** on the early lay, and about 0.8
  to 1.0 points across all rules. Part of the move comes after the last book. One cause is the recorder's books, about
  90 seconds stale (section 13): live, the green-up would be placed in the last seconds and capture more of the move.
  The forward test's live-key books will show how much.
- The trade-off is return against variance. The SP close keeps the higher expectation, but each race's P&L swings
  with the result. The green-up locks the P&L at the off. The forward test runs both: green-up as the primary (the
  owner's way of trading), the SP close as `lay80_drift_sp`.
- Overall the grid is still at chance on this holdout: 3 of 240 chosen rules had the 90% interval's lower bound above
  zero, where about 12 are expected by chance alone (run 41633779822).

## 26. The BFSP model: the new features, the closing price, and the two together (3 Nov 2026)

The owner, 3 Nov: "we need to focus on improving our BFSP model. The final price / log of final price is key. Once
we can get the most accurate BFSP model then we can trade around it." Two models make the forecast of the SP:

- **The pre-race model** (the fast loop's `logbsp` objective): log(BSP) from form alone, before the market forms.
- **The closing model** (`research/greyhound/bfsp_close.py`): log(BSP) from the book at a mark, with the pre-race
  price among its inputs.

Accuracy is the RMSE of log(BSP): 0.30 is a typical miss of about 35% on the price.

### 26.1 Batch 6: the new feature blocks (greyhound-fastloop.yml run 41659861848)

Each block on its own on top of `core` (every block before 3 Nov, the weigh-in included), learning rate 0.05,
log-BSP objective. Test window 1 Jun to 2 Nov 2026; differences against core with 90% intervals by day.

| Experiment | Log-loss vs core | Adds to the SP forecast at first traded |
|---|---|---|
| core | 0 (0.44231) | 0.00704 |
| + F3 times (track-normalised, meeting variant) | -0.00082 (-0.00107 to -0.00057) | 0.00771 |
| + expected final time | -0.00048 (-0.00074 to -0.00022) | 0.00718 |
| + F6 ability with uncertainty | -0.00019 (-0.00045 to +0.00006) | 0.00725 |
| + F5 trainer intent | +0.00004 (-0.00021 to +0.00028) | 0.00711 |
| all four | -0.00135 (-0.00167 to -0.00102) | 0.00784 |
| all four + F2 race simulator | -0.00139 (-0.00171 to -0.00107) | 0.00812 |
| all four + F2 + L2 forecast price | -0.00142 (-0.00172 to -0.00111) | 0.00806 |

F3 and the expected time carry the gain. F5 adds nothing. L2 is almost absent from the 2024-on training window: it
was first recorded on 6 Oct 2026.

### 26.2 The closing price on the recorder's books (greyhound-bfsp.yml runs 41668293745, 41669591841)

At each mark, walk-forward by day over 20 scored days, 14 Oct to 2 Nov 2026 (2,464 races at first traded). Our price
is the 5 Oct forward test's model. RMSE of log(BSP):

| Mark | Market mid | Mid scaled to the SP book | Live trader's logit | Trees, per mark | Trees, pooled over marks |
|---|---|---|---|---|---|
| First traded | 0.635 | 0.492 | 0.356 | 0.321 | 0.320 |
| 5 min out | 0.709 | 0.529 | 0.414 | 0.369 | 0.370 |
| 3 min out | 0.337 | 0.299 | 0.244 | 0.231 | 0.225 |
| 1 min out | 0.158 | 0.157 | 0.146 | 0.149 | 0.144 |

- **The mid is biased early:** it sits about 35% short of the BSP at first traded (bias -0.35), because the early book
  carries a large overround and greyhound prices drift.
- **The trees lean on:** the last trade against the mid (22% of gain), the micro-price (17%), our price (20% in two
  forms) and the spread (10%).
- **Traded volume scores 0%:** the recorder's books never had it. The live-key books do, from 3 Nov.

### 26.3 Batch 7: the pre-race model's accuracy, and the closing model on its prices (run 41671443774)

The pre-race model, RMSE of log(BSP) on the test window. The paired intervals were not saved for this batch; they
are from batch 8.

| Pre-race model | RMSE log(BSP) |
|---|---|
| core | 0.4418 |
| best (all four new blocks and the simulator) | 0.4282 |
| best, fewer features a tree (0.6) | 0.4280 |
| best, 63 leaves | 0.4295 |
| best, rescaled to the SP book in each race | **0.4086** |

The closing model with each pre-race model as our price, trees pooled over marks, RMSE of log(BSP):

| Mark | With the 5 Oct model | With core | With best |
|---|---|---|---|
| First traded | 0.320 | 0.292 | **0.287** |
| 5 min out | 0.370 | 0.330 | **0.321** |
| 3 min out | 0.225 | 0.217 | **0.215** |
| 1 min out | 0.144 | 0.144 | 0.144 |

What this says:

- **The two models multiply.** A better pre-race price makes the closing forecast more accurate wherever the book
  still has something to learn.
- **The size of the gain:** at first traded, 0.320 with the old model's price, 0.287 with the best. The error is 55%
  below the market mid's (0.635).
- **At 1 minute out it makes no difference:** the market has found the SP.
- **Rescaling the pre-race prices** to the SP book's usual total cuts their error by 0.020 (0.4282 to 0.4086).

### 26.4 What comes next

1. **The live pre-race model:** retrain it on the best features (all four new blocks and the simulator, log-BSP,
   rescaled).
2. **The closing model on the live-key books:** refit it as the days build up, with traded volume and the order-flow
   features (M1 to M3) the dense books carry. This is the step that should close the gap between the recorded and
   the live forecasts (section 25 and the live trades of 3 Nov).
3. **Then put the pooled trees in the live trader,** in place of the logit, once the live-books evaluation confirms
   them on live books.

## 27. The corrected simulator, the staking grid, the retrain path and the PLACE book (4 Nov 2026)

The owner, 4 Nov: "implement all recommendations in order" (the closing model on the live-key books and a bookmaker
feed dropped), "I want to live trade today so let's get to the most profitable model", "Evaluating best staking
strategy; type and size".

**A look-ahead in the queue fills, fixed** (`greyhound/queue_fill.py`). The live-key recorder stamps a read's books
to the second and its traded ladder to the microsecond. An order priced on a book at second t read that read's
ladder row (t.xxx) as traded after it, so money matched before the order existed was credited to it. Found when a
1.5 s order latency turned quotes 5% around the mid on 3 Nov from +GBP154 to -GBP74; the ladder times are now floored
to the second. Every two-sided figure of 3 Nov's report (section 26 of the running day) was inflated by it.

**3 Nov's live-key books, corrected** (greyhound-profit-sim.yml runs 41791725404, 41801730008; 71 races):

| Strategy | Net after 2% | Races won / lost |
|---|---|---|
| Early lay as live (rest at 80% of the forecast, first traded) | -3.93 | 12 / 16 |
| Early lay, 10% margin | +3.82 (26 fills) | 16 / 8 |
| Early lay, 5%, 15%, 25%, 30% | 0.00, -1.50, -0.82, -1.85 | |
| Early lay decided 5 min out, resting or taking | 3-8 races only: the fields have not all traded by then | |
| Taking at the first traded books, lays / lays and backs | -7.08 / -9.51 | |
| Quotes around the mid 5% / 10% / 20% | -76.20 / -31.67 / -9.48 | |
| Quotes around the closing model 10% / 20% / 30% | -101.69 / -37.78 / -1.23 | |

The calibration against the live ledger holds (159 lays: simulated -GBP5.08 against -GBP3.59 live, matched or not the
same in 147). **The staking grid** (run 41788897608, quotes 5% around the mid, hard fills): every type and size loses
1.5% to 2.4% of matched money, GBP10 to GBP100 liability, fixed stakes, a share of the money shown, raised late; the
loss scales with size. Sizing follows a measured edge; none is measured.

**Today's live rule:** the early lay at a 10% margin, GBP20 liability a lay, the guard on: the best measured, within
noise. The quotes around the mid (`greyhound/quote_mid.py`, built for the owner's "go live") run on paper.

**The retrain path** (`greyhound/live.py`): the live model's train and serve path now fits every engine block plus
the race simulator on log(BSP) from 2024, rescaled to the SP book (section 26.3's best_rescaled). Gated on the card
check of a past day, and on refitting the early-lay forecast on the new model's prices.

**The WIN book against the PLACE book** (roadmap M5; `research/greyhound/cross_market.py`, greyhound-cross-market.yml
run 41803082690; 50,886 runner-marks over 28 days, walk-forward): the dislocation between the PLACE book's chance and
the Harville place chance from the WIN book correlates -0.03 to +0.05 with the move to the BSP; beside the last trade
it cuts the RMSE of log(BSP) by at most 0.001 (first traded, interval -0.0011 to -0.0001) and nothing at 3 or 1 minute;
the result's log-loss is not improved at any mark. **Nothing to trade: the two books agree.**

## 28. The BFSP plan, batch 11: the race fitted as a book (greyhound-fastloop.yml run 42792051707, 10 Nov 2026)

The plan of 10 Nov (reports/greyhound_bfsp_plan.md; the owner: "Fundamental is to improve our model to predict BFSP /
Closing Prices"), steps 0 and 1, registered before any result (`research/greyhound/experiments/done/batch11_bfsp.json`).
Test: every race from 1 June to 5 November 2026 (106,719 runners with a BSP), one fit on the data before 1 June, the
live recipe's card-safe features (425). Paired differences by day; z is the difference over its bootstrap standard
error; the bar is 3.

**The result: fitting the race as a book beats fitting each runner's price.** The `share` objective (a softmax over the
race fitted to each runner's share of the Betfair SP book) against the live recipe (squared error on log(BSP)):

| Experiment | RMSE log(BSP) | vs live_ref | z | Log-loss | vs live_ref | z |
|---|---|---|---|---|---|---|
| live_ref (the live recipe, learning rate 0.05) | 0.4090 | - | - | 0.44112 | - | - |
| live_ref + per-track stretch (what the live model serves) | 0.4017 | -0.0072 | - | 0.44124 | +0.00012 | - |
| **share** | **0.3849** | **-0.0241 (90% -0.0252 to -0.0229)** | **-33.8** | **0.43956** | **-0.00156 (-0.00197 to -0.00116)** | **-6.3** |
| mix (70% the SP shares, 30% the result) | 0.4092 | +0.0002 | +0.3 | 0.44084 | -0.00028 | -1.1 |
| history_hl365 (every year from June 2020, one-year half-life) | 0.4123 | +0.0034 | +8.1 | 0.44119 | +0.00007 | +0.5 |
| huber | 0.4096 | +0.0007 | +1.8 | 0.44113 | +0.00001 | 0.0 |

- `share` clears the bar on both targets, by far the largest single gain the greyhound model has had (earlier blocks
  moved the RMSE by 0.001 to 0.007). Against the stretched live price it is still 0.0168 better, unpaired.
- More old history hurts: the market of 2020-23 is not today's.
- The mixed target and the Huber loss add nothing.
- Not finished: three seeds, and the live and share models refitted through the window (each ran into the job's
  60-minute limit at this learning rate). Batch 12 refits share and runs at 0.08.

**Baselines** (the same test; each fitted on named features only):

| Knows only | RMSE log(BSP) | Share of the within-race spread explained |
|---|---|---|
| each race's average log(BSP) (no model) | 0.7649 | 0% |
| field size (base_field) | 0.8311 | 0% |
| the dog's past market (SP and BSP history, 42 columns) | 0.7089 | 26.9% |
| speed figures (118 columns) | 0.5491 | 55.1% |
| everything (live_ref) | 0.4090 | 75.1% |
| everything, the race as a book (share) | 0.3849 | 77.8% |

The race's level alone (whether a race's prices sit short or long overall) carries 0.1358 of share's error; the rest
is the order within the race.

**What the model relies on** (live_ref; each family shuffled within the race on 4,000 test races, the rise in RMSE):

| Family | Features | RMSE rise | SHAP (sum) |
|---|---|---|---|
| Library (Elo, Kalman rating, class-adjusted figure) | 26 | +0.178 | 0.46 |
| Base engine | 133 | +0.138 | 1.16 |
| Parity windows | 132 | +0.138 | 0.97 |
| Past market (a theme across families) | 55 | +0.076 | 0.39 |
| Record by condition | 42 | +0.075 | 0.45 |
| Expected time | 21 | +0.044 | 0.26 |
| Track-normalised times | 20 | +0.038 | 0.21 |
| Simulator | 6 | +0.015 | 0.08 |
| Ability with uncertainty | 13 | +0.012 | 0.12 |
| Weight forecast | 7 | +0.006 | 0.05 |
| Trainer intent | 25 | +0.006 | 0.06 |

Top features by mean absolute SHAP value: `lb_cv_gsr_ewm_z` 0.200 (the class-adjusted speed figure against the
field), `et_p_fastest` 0.072, `hb_age_vs_oldest` 0.065, `gp_rs_vs_past` 0.058, `hb_age_vs_youngest` 0.057,
`open_race` 0.056, `gsr_ewm_z` 0.053, `tn_var_last` 0.053, `gp_bsp_mkt_w3` 0.049, `gp_bsp_ae_w10_z` 0.047. For share
the leaders are the same family: `lb_cv_gsr_ewm_z` 0.119, `tn_var_last` 0.104, `gp_rs_vs_past` 0.102.

**Where the error sits** (share; bias is the mean of log forecast less log BSP, below 0 priced too short):

| BSP | Runners | RMSE | Bias |
|---|---|---|---|
| under 2 | 2,357 | 0.335 | +0.232 |
| 2 to 3 | 9,403 | 0.306 | +0.181 |
| 3 to 5 | 27,369 | 0.274 | +0.087 |
| 5 to 8 | 27,234 | 0.289 | -0.017 |
| 8 to 15 | 23,581 | 0.364 | -0.160 |
| 15 and over | 16,775 | 0.656 | -0.473 |

- The forecast is still compressed (favourites priced too long, outsiders too short), less than live_ref's (+0.303 and
  -0.506). Batch 12 tests the per-track stretch on top of share.
- Debutants (1,934 runners) carry an RMSE of 0.715 against 0.371 for dogs with 20 or more runs.

## 29. The BFSP plan, batch 12: the market-implied rating and the corrected figures (greyhound-fastloop.yml run 42802035088, 10 Nov 2026)

Registered after batch 11, before its own scores (`research/greyhound/experiments/batch12_figs.json`). The same test
(1 June to 5 November 2026, 106,719 runners), the race fitted as a book (`share`) at a learning rate of 0.08, and the
new figs table of the metrics database (`greyhound/pit/figs.py`, built over 3,966,248 GBGB runs from 2014: run
42794389155): corrected figures on their own standards and shrunk meeting variants, levels decayed in days and shrunk
by evidence, class in lengths, and a market-implied rating (each dog's race-relative log SP share, opponent-adjusted,
through a Kalman filter fitted on 2014-17).

| Experiment | RMSE log(BSP) | vs share08 | z | Log-loss | vs share08 | z | Within-race R2 |
|---|---|---|---|---|---|---|---|
| live08 (the live recipe at 0.08) | 0.4127 | +0.0244 | +32.4 | 0.44155 | +0.00164 | +6.4 | 74.7% |
| live08 + per-track stretch | 0.4062 | | | 0.44168 | | | |
| share08 | 0.3883 | - | - | 0.43990 | - | - | 77.4% |
| share08 + per-track stretch | 0.3815 | -0.0067 (calibration, same fit) | | 0.44021 | +0.00031 | | |
| share08 refitted on 1 Sep (two segments) | 0.3865 | -0.0018 | -5.0 | 0.43968 | -0.00022 | -1.6 | 77.6% |
| share08 + figs, figures only (no rating) | 0.3819 | -0.0064 | -13.4 | 0.43938 | -0.00052 | -2.8 | 78.1% |
| share08 + figs, the market rating only | 0.3612 | -0.0270 | -34.7 | 0.43839 | -0.00151 | -6.0 | 80.3% |
| **share08 + the whole figs table** | **0.3593** | **-0.0290 (90% -0.0302 to -0.0278)** | **-40.1** | **0.43806** | **-0.00184 (-0.00225 to -0.00147)** | **-7.8** | **80.6%** |

- **Every figs variant clears the bar on both targets.** The market-implied rating carries most of it; the corrected
  figures add 0.0064 on their own and a little more beside the rating.
- **The race as a book is confirmed at the faster learning rate:** share beats the live recipe by 0.0244 (z 32.4).
- **The gain survives a refit:** refitting on 1 September improves the score slightly (-0.0018, z -5.0).
- From the live price as served (0.4017 at 0.05, batch 11; 0.4062 at 0.08 here) to share with the figs table (0.3593)
  the error falls by about 11% at the same learning rate. The stretch on top of share and figs is not yet scored.
- With the figs table the model leans on the rating first: shuffled within the race, the figs family raises the RMSE
  by 0.347 (base 0.109, parity 0.082, library 0.046, the old market-history columns 0.043). Top features by mean
  absolute SHAP value: `fg_mr_vs_field` 0.159, `tn_var_last` 0.092, `fg_mr_share` 0.085, `fg_mr_lshare` 0.079,
  `gp_mkt_w10` 0.071, `lb_cv_gsr_ewm_z` 0.065.
- The compression is smaller still (short prices +0.21 in log, outsiders -0.40, against +0.30 and -0.51 for the live
  recipe). Debutants are no better (RMSE 0.72): they have no rating and no figures.

**Not adopted yet.** The test window has been read by earlier batches. The forward test, registered now
(`research/greyhound/experiments/bfsp_forward.json`), scores the same recipes on days after 10 November only.

## 30. The BFSP plan, batch 13: what adds on top (greyhound-fastloop.yml run 42808526186, 10 Nov 2026)

On share with the figs table, rebuilt with the market rating's wider drift grid (run 42794389155): RMSE of log(BSP)
0.3579 (batch 12's build: 0.3593). Paired against it:

| Experiment | RMSE log(BSP) | vs share08_figs | z | Log-loss | vs share08_figs | z |
|---|---|---|---|---|---|---|
| share08_figs | 0.3579 | - | - | 0.43789 | - | - |
| + per-track stretch (same fit) | 0.3533 | -0.0047 (-0.0053 to -0.0041) | | 0.43805 | +0.00017 | |
| three seeds averaged | 0.3542 | -0.0037 (-0.0042 to -0.0033) | -14.1 | 0.43770 | -0.00019 | -2.0 |
| refitted on 1 September | 0.3576 | -0.0003 | -0.9 | 0.43796 | +0.00007 | +0.8 |
| + every other metric table (ability, trouble, connections, market, breeding, life) | 0.3542 | -0.0038 (-0.0044 to -0.0032) | -9.8 | 0.43769 | -0.00019 | -1.2 |

- Seeds and the other metric tables each clear the bar; the stretch improves the price again at a small cost to the
  win chance (as on the live recipe), so it belongs on the BFSP forecast only.
- The refit changes nothing: the single fit at 1 June holds through November.
- With every table, the figs family still carries the model (shuffled within the race: +0.347 RMSE); of the other
  tables life (+0.010) and ability (+0.006) matter most.

The forward test (bfsp_forward.json) gains `share08_full`, these together, before any of its days exist.

## 31. The BFSP plan, batch 14: the Betfair-SP rating and the known-wrong fields (greyhound-fastloop.yml run 42818559765, 10 Nov 2026)

On share with the figs table (the figs table rebuilt with the rating read on the Betfair SP as well, run 42816936434:
259,857 races rated on the BSP). Paired against share08_figs:

| Experiment | RMSE log(BSP) | vs share08_figs | z | Log-loss | vs share08_figs | z |
|---|---|---|---|---|---|---|
| share08_figs | 0.3593 | - | - | 0.43806 | - | - |
| + the Betfair-SP rating (fg_bmr) | 0.3570 | -0.0023 (-0.0028 to -0.0017) | -6.5 | 0.43786 | -0.00020 | -1.4 |
| without hb_days_since_season | 0.3589 | -0.0004 | -1.1 | 0.43830 | +0.00025 | +1.6 |
| without ti_new_trainer | 0.3596 | +0.0003 | +1.0 | 0.43834 | +0.00028 | +2.2 |
| without every trainer column | 0.3625 | +0.0032 (+0.0025 to +0.0039) | +7.4 | 0.43840 | +0.00034 | +1.9 |

- The Betfair-SP rating adds a little beside the SP one.
- The two fields shown wrong in the review carry nothing; they are dropped from batch 15 on.
- The trainer columns stay: removing all of them (the 6 Nov test missed some) costs 0.0032, as the earlier test found.
  The trainer is the current one on every past run, so this is the yard's identity, not its form at the time.

**Run-to-run noise.** The same recipe scored 0.3593 here and in batch 12 (3,163 rounds) but 0.3579 in batch 13 (4,372
rounds): GitHub's runners differ and LightGBM's multithreaded fits were not deterministic. The day-level bootstrap does
not see that, so gains near 0.001-0.002 (the BSP rating, seeds, the other tables) are replicated in batch 15 with every
fit deterministic on four threads. The large gains (the race as a book, 0.024; the figs table, 0.029) are far beyond it.

## 32. The BFSP plan, batch 15: the replication and the best recipe (greyhound-fastloop.yml run 42824328816, 10 Nov 2026)

Every fit deterministic on four threads; the two wrong fields dropped. Paired against share08_figs (0.3598 here):

| Experiment | RMSE log(BSP) | vs share08_figs | z | Log-loss | vs share08_figs | z |
|---|---|---|---|---|---|---|
| share08_figs | 0.3598 | - | - | 0.43836 | - | - |
| + the Betfair-SP rating | 0.3586 | -0.0013 (-0.0019 to -0.0007) | -3.4 | 0.43816 | -0.00020 | -1.4 |
| three seeds | 0.3553 | -0.0045 (-0.0049 to -0.0041) | -19.2 | 0.43800 | -0.00036 | -4.3 |
| + every other metric table | 0.3561 | -0.0037 (-0.0044 to -0.0031) | -9.2 | 0.43809 | -0.00027 | -1.6 |
| **share08_best: all of these together** | **0.3510** | **-0.0088 (-0.0094 to -0.0083)** | **-25.6** | **0.43777** | **-0.00059 (-0.00084 to -0.00034)** | **-3.9** |
| share08_best + per-track stretch (same fit) | 0.3448 | -0.0061 more | | 0.43795 | +0.00017 | |

All three small gains replicate (the BSP rating only just). Together they add up.

**Where the plan's first day leaves the BFSP forecast** (the same test throughout, 106,719 runners):

| Step | RMSE log(BSP) | Within-race spread explained |
|---|---|---|
| Knowing only each race's average | 0.7649 | 0% |
| The live recipe as served (per-track stretch; batch 11) | 0.4017 | 75.1% before the stretch |
| The race fitted as a book (batch 11) | 0.3849 | 77.8% |
| + the figs table (batch 12) | 0.3593 | 80.6% |
| + the BSP rating, every other table, three seeds (batch 15) | 0.3510 | 81.5% |
| + per-track stretch (batch 15) | **0.3448** | |

From 0.4017 to 0.3448: 14% less error in log(BSP), and a better winner's log-loss at every step bar the stretch.

What the best recipe relies on (shuffled within the race): the figs table +0.339 (31 columns), base +0.094, parity
+0.078, library +0.046, track-normalised times +0.043. Top features by mean absolute SHAP value: `fg_bmr_vs_field`
0.186 (the Betfair-SP rating against the field), `tn_var_last` 0.092, `lb_cv_gsr_ewm_z` 0.068, `fg_mr_vs_field` 0.053.

Still compressed: short prices +0.20 in log, outsiders -0.40. Debutants unchanged (0.70).

The forward test (`bfsp_forward.json`, from 25 Nov 2026) now carries `share08_best` exactly as here, beside the live
recipe and share08_figs, on days after 10 November only. Adoption into the live pricing waits on it and on the owner.

## 33. The BFSP plan, batch 16: the better price down the clock (greyhound-fastloop.yml run 42842099498, 10 Nov 2026)

Step 5 of the plan: does share08_best's pre-race price (one seed here: 0.3555 against the live recipe's 0.4120 on the
June-November test, z -54.1; log-loss -0.00332, z -10.7) sharpen the forecasts nearer the off? Each price fed into the
closing-price study (`research/greyhound/bfsp_close.py`, recorded books re-timed, 22 scored days 14 Oct to 5 Nov,
walk-forward) and the limit-order study (`research/greyhound/limit_study.py`). share08_best's out-of-sample prices are
published to `greyhound/research/oos_prices_share_best`.

**The closing forecast, RMSE of log(BSP)** (the best model at each mark, pooled trees; the market mid beside it):

| Mark | Races | The mid | With the live price | With share08_best's price | Change |
|---|---|---|---|---|---|
| First traded | 2,685 | 0.6286 | 0.2852 | 0.2648 | -0.0204 |
| 5 minutes out | 1,792 | 0.7048 | 0.3214 | 0.2926 | -0.0288 |
| 3 minutes out | 1,869 | 0.3388 | 0.2155 | 0.2092 | -0.0063 |
| 1 minute out | 1,807 | 0.1566 | 0.1435 | 0.1434 | -0.0001 |

The better pre-race price carries through to 3 minutes out and is gone by the last minute, where the book knows what
we know. Our price is the model's largest input at first traded (59% of the boosted model's gain, `log_model_vs_mid`).
The two runs are not paired here, so no interval is given for the change.

**The early lay's rule** (lay the runners forecast to drift at 80% of the forecast SP, first traded book): the
forecast refitted on each price, development days (15):

| Exit | With the live price | With share08_best's price |
|---|---|---|
| Greened up at the last book | +1.49% a unit (90% 1.24 to 1.77), CLV +7.32% | +1.86% (1.63 to 2.11), CLV +10.03% |
| Closed at the SP | +1.88% (1.14 to 2.64) | +2.31% (1.62 to 3.03) |

2,536 and 2,453 matched lays, on paper with the study's fill rule. The better price makes the early lay's forecast
better: the lays beat the SP by 10.0% against 7.3%. The live early lay and its forward-test gate are unchanged; this
is a reading for the owner, who decides whether the early lay should move to the new price once the BFSP forward
test (`bfsp_forward.json`) has run.
