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
