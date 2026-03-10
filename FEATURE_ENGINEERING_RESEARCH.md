# Feature Engineering Research: Discovery & Testing Plan

## Overview

This document audits the current feature set of the Ultra Betting prediction system, identifies untapped data columns in the database, and proposes **42 new features** grouped into 12 categories. Each proposal includes the rationale, formula, implementation notes, and expected predictive value.

---

## 1. Current Feature Inventory

### Database Schema (50 columns in `race_results`)

| Column | Currently Used By | Status |
|--------|------------------|--------|
| `race_date` | CustomMetrics, PreRaceBuilder | Core key |
| `race_time` | CustomMetrics | Core key |
| `track` | PreRaceBuilder (course stats) | Partially used |
| `horse_name` | CustomMetrics, PreRaceBuilder | Core key |
| `jockey_name` | CustomMetrics, PreRaceBuilder | Core key |
| `trainer` | CustomMetrics, PreRaceBuilder | Core key |
| `placing_numerical` | CustomMetrics (NFP, RB, won/placed) | Fully used |
| `bfsp` | CustomMetrics (ORR2, PFD, OFS, WIV) | Fully used |
| `number_of_runners` | CustomMetrics, PreRaceBuilder | Fully used |
| `official_rating` | CustomMetrics (FCS, ORR2) | Partially used |
| `prize_money` | CustomMetrics (WPMRF, PMW) | Fully used |
| `going_description` | PreRaceBuilder (going_preference) | Partially used |
| `dist_furlongs` | PreRaceBuilder (distance_preference) | Partially used |
| `race_class` | PreRaceBuilder (class_change) | Partially used |
| `comment` | CustomMetrics (EPF) | Fully used |
| `horse_age` | PreRaceBuilder | Minimal use |
| `pounds` | PreRaceBuilder (weight_change) | Minimal use |
| `stall` | PreRaceBuilder (draw_position) | Minimal use |
| **`comptime_numeric`** | **UNUSED** | Speed figures |
| **`total_dst_bt`** | **UNUSED** | Actual lengths beaten |
| **`distbt`** | **UNUSED** | Per-runner distance beaten |
| **`stallion`** | **UNUSED** | Sire breeding data |
| **`dam`** | **UNUSED** | Dam breeding data |
| **`dam_stallion`** | **UNUSED** | Damsire breeding data |
| **`horse_sex`** | **UNUSED** (in custom metrics) | Sex of horse |
| **`headgear`** | **UNUSED** (in custom metrics) | Equipment changes |
| **`surface_type`** | **UNUSED** (in custom metrics) | Flat/AW |
| **`race_type`** | **UNUSED** (in custom metrics) | Handicap/Maiden/etc |
| **`track_direction`** | **UNUSED** | Left/right-handed |
| **`stall_positioning`** | **UNUSED** | Draw bias data |
| **`rail_move`** | **UNUSED** | Rail movement |
| **`odds`** | **UNUSED** | Traditional SP odds |
| **`fav`** | **UNUSED** | Favourite flag |
| **`jockeys_claim`** | **UNUSED** (in custom metrics) | Apprentice allowance |
| **`horse_prizewin`** | **UNUSED** | Career prize earnings |
| **`career_runs`** | **UNUSED** (in custom metrics) | Raw career run count |
| **`days_since_lr`** | **UNUSED** (in custom metrics) | Raw days since last run |
| **`median_or`** | **UNUSED** | Median OR in race |
| **`max_or_in_race`** | **UNUSED** | Max OR in race |
| **`race_restrictions_age`** | **UNUSED** | Age restrictions |
| **`race_name`** | **UNUSED** | Race title |
| **`major`** | **UNUSED** | Major race flag |
| **`race_distance`** | **UNUSED** | Text distance |
| **`yards`** | **UNUSED** | Distance in yards |
| **`card_no`** | **UNUSED** | Card number |
| **`bfsp_place`** | **UNUSED** | Place market BFSP |
| **`plcs_paid`** | **UNUSED** | Places paid |
| **`bf_plcs_paid`** | **UNUSED** | Betfair places paid |
| **`race_code`** | **UNUSED** | Race code |

### Currently Computed Features (~70+ per runner)

The CustomMetricsEngine produces 19 metric families, and PreRaceBuilder extracts ~70 features per runner. See `model/custom_metrics.py` and `model/prerace_builder.py` for full details.

**Key gap**: 17 database columns with potentially high predictive value are completely unused.

---

## 2. Proposed New Features

### Category A: Speed Figures (from `comptime_numeric`)

**Rationale**: Completion time is arguably the single most predictive raw variable in racing after odds. Speed figures — time-adjusted for distance, going, and track — are the backbone of systems like Timeform, Racing Post Ratings, and Beyer figures. Our database has `comptime_numeric` and we are not using it at all.

#### A1. Raw Speed Rating (RSR)

```
standard_time = median(comptime_numeric) per (track, dist_furlongs, going_description)
RSR = (standard_time - comptime_numeric) / standard_time * 100
```

Positive RSR = faster than standard. Compute lag-safe career and rolling averages:
- `preracehorsecareerRSR` — expanding mean, lagged
- `LR3RSR`, `LR5RSR`, `LR10RSR` — recency-weighted rolling RSR
- `LRRSR` — last run RSR

**Expected value**: HIGH. Speed figures consistently rank among the top predictors in academic literature (Benter 1994, Bolton & Chapman 1986).

#### A2. Speed Figure Improvement (SFI)

```
SFI = LR1_RSR - LR2_RSR
```

Captures improving/declining speed form. Horses showing speed improvement are significantly more likely to win next time.

- `SFI_3` — average improvement across last 3 runs
- `SFI_trend` — linear regression slope of RSR over last 5 runs

**Expected value**: MEDIUM-HIGH. Trend features complement level features.

#### A3. Best Speed Rating

```
preracehorsecareerBestRSR = max(RSR) across all prior runs, lagged
LR5BestRSR = max(RSR) in last 5 runs, lagged
```

Peak ability matters — a horse that once ran fast can do so again. The gap between best and recent RSR indicates unrealised potential.

- `RSR_gap = preracehorsecareerBestRSR - LR3RSR`

**Expected value**: MEDIUM. Best speed figures are strong predictors for class droppers.

#### A4. Going-Adjusted Speed (GAS)

```
going_standard = median(comptime_numeric) per (track, dist_furlongs, going_description)
track_standard = median(comptime_numeric) per (track, dist_furlongs)
GAS = RSR + (going_standard - track_standard)
```

Normalises speed across different going conditions, isolating the horse's true ability from ground conditions.

**Expected value**: HIGH. Going adjustment is critical for UK racing where ground varies enormously.

---

### Category B: Actual Lengths Beaten (from `total_dst_bt` and `distbt`)

**Rationale**: Our current RB metric approximates beaten distances using finishing position / field size. But we have ACTUAL beaten-lengths data in `total_dst_bt` (cumulative from winner) and `distbt` (beaten by horse ahead). These are far more informative — a horse beaten a neck in 2nd is vastly different from one beaten 20 lengths in 2nd.

#### B1. Lengths Beaten (LB)

```
LB = parse_numeric(total_dst_bt)  # convert "2.5" or "nk" to numeric lengths
```

Special values: `nk` = 0.2, `shd` = 0.1, `hd` = 0.15, `nse` = 0.05, `dht` = 0

- `preracehorsecareerLB` — expanding mean, lagged
- `LR3LB`, `LR5LB` — recency-weighted averages
- `LRLB` — last run lengths beaten

**Expected value**: VERY HIGH. Actual margins are the gold standard for form assessment. This directly replaces the approximated RB with ground truth.

#### B2. Field-Size Adjusted Lengths Beaten (FSALB)

```
FSALB = LB * (number_of_runners / median_field_size)
```

Analogous to FSARB but using actual lengths. Being beaten 3 lengths in a 20-runner race is more forgivable than in a 5-runner race.

**Expected value**: HIGH.

#### B3. Lengths Per Position (LPP)

```
LPP = LB / (placing_numerical - 1)   # for non-winners
```

Captures race compression — in tight finishes LPP is small (competitive race), in strung-out races LPP is large. Career average LPP indicates whether a horse typically races in competitive affairs.

**Expected value**: MEDIUM.

#### B4. Closing Sectional Proxy

```
If we have both EPF (early position) and LB (final margin):
closing_gain = expected_LB_from_EPF - actual_LB
```

Where `expected_LB_from_EPF` is the average lengths beaten for horses with that EPF score. A horse that started at the back (EPF=1) but was only beaten 2 lengths showed strong closing speed.

**Expected value**: MEDIUM-HIGH. Sectional data is extremely valuable and this is a reasonable proxy.

---

### Category C: Breeding / Pedigree Features (from `stallion`, `dam`, `dam_stallion`)

**Rationale**: Pedigree is a major factor especially for: (1) debut/lightly-raced horses with limited form, (2) first-time going/distance changes, (3) age group trends. We have sire, dam, and damsire data sitting unused.

#### C1. Sire Strike Rate

```
preraceStallionWinRate = expanding mean of won, grouped by stallion, lagged
preraceStallionPlaceRate = expanding mean of placed, grouped by stallion, lagged
preraceStallionAvgNFP = expanding mean NFP, grouped by stallion, lagged
```

**Expected value**: MEDIUM-HIGH for younger/lightly-raced horses. Sire influence is well-documented.

#### C2. Sire Going Aptitude

```
stallion_going_win_rate = win rate of stallion's progeny on today's going
stallion_going_nfp = avg NFP of stallion's progeny on today's going
```

Computed per (stallion, going_description) combination. Critical for predicting how untested horses will handle ground conditions.

**Expected value**: HIGH for debut/lightly-raced horses on unfamiliar going.

#### C3. Sire Distance Aptitude

```
stallion_dist_win_rate = win rate of stallion's progeny at today's distance bucket
stallion_dist_nfp = avg NFP of stallion's progeny at today's distance bucket
```

Distance aptitude is strongly inherited. First-time-at-distance horses can be assessed via their sire's distance profile.

**Expected value**: HIGH for trip changes and debuts.

#### C4. Damsire Going/Distance Profiles

Same as C2-C3 but grouped by `dam_stallion`. Damsire influence is well-established, especially for stamina and going preference.

- `damsire_going_nfp`
- `damsire_dist_nfp`

**Expected value**: MEDIUM. Damsire is a secondary but meaningful influence.

#### C5. Sire WIV (Win Index Value by Sire)

```
stallion_WIV = cumulative progeny wins / cumulative expected wins, lagged
```

A sire-level WIV analogous to the horse/jockey/trainer WIV. Identifies sires whose progeny consistently outperform market expectations.

**Expected value**: MEDIUM.

---

### Category D: Equipment Changes (from `headgear`)

**Rationale**: Equipment changes (blinkers, visors, tongue-ties, cheekpieces) are a well-known and exploitable signal. First-time blinkers in particular show statistically significant improvement in UK racing data.

#### D1. Headgear Change Flag

```
headgear_change = 1 if current headgear != last run headgear else 0
first_time_headgear = 1 if headgear is not None/empty AND no prior run had headgear
headgear_removed = 1 if last run had headgear AND today has none
```

**Expected value**: MEDIUM-HIGH. First-time blinkers/visor is one of the most well-known "system" angles.

#### D2. Headgear Type Encoding

```
Encode headgear types: b=blinkers, v=visor, t=tongue-tie, p=cheekpieces, h=hood, e/s=eye-shield
headgear_blinkers = 1 if 'b' in headgear
headgear_visor = 1 if 'v' in headgear
headgear_tongue_tie = 1 if 't' in headgear
headgear_cheekpieces = 1 if 'p' or 'cp' in headgear
```

**Expected value**: LOW-MEDIUM. Categorical interaction with other features.

#### D3. Performance With/Without Headgear

```
horse_nfp_with_headgear = avg NFP when wearing headgear
horse_nfp_without_headgear = avg NFP when not wearing headgear
headgear_impact = horse_nfp_with_headgear - horse_nfp_without_headgear
```

For horses with sufficient data, this reveals whether equipment actually helps them.

**Expected value**: MEDIUM.

---

### Category E: Draw Bias Features (from `stall`, `stall_positioning`, `track_direction`)

**Rationale**: Draw bias varies by track, distance, and going. Low draws are advantageous at Chester, high draws at Beverley on soft ground, etc. We use `stall` as a raw feature but don't model the bias.

#### E1. Track-Distance Draw Bias

```
For each (track, dist_bucket, going_category):
  draw_win_rate[stall_quartile] = historical win rate by stall quartile

stall_quartile = stall / number_of_runners  # 0-1 scale
draw_bias = draw_win_rate[this_quartile] - (1 / number_of_runners)
```

Positive draw_bias means this draw position is historically advantageous at this track/distance/going combination.

**Expected value**: HIGH at specific courses (Chester, Beverley, Epsom, Windsor). Medium overall.

#### E2. Draw Position Relative to Field

```
draw_normalised = stall / number_of_runners  # 0-1 scale
draw_vs_median = stall - (number_of_runners / 2)
```

Better than raw stall number because it adjusts for field size.

**Expected value**: LOW-MEDIUM. Normalisation improves raw stall.

#### E3. Track Direction Interaction

```
track_direction_encoded = 1 if left-handed, 0 if right-handed
draw_x_direction = draw_normalised * track_direction_encoded
```

Draw bias interacts with track handedness — low draws may favour on left-handed courses but not right-handed.

**Expected value**: LOW-MEDIUM.

---

### Category F: Market Intelligence Features (from `odds`, `fav`, `bfsp_place`)

**Rationale**: We use BFSP extensively, but ignore traditional SP odds, the favourite flag, and place market prices. The relationship between win and place prices contains information about perceived volatility.

#### F1. SP-to-BFSP Spread

```
sp_bfsp_ratio = odds / bfsp
```

This captures market disagreement between traditional bookmaker SP and Betfair SP. Large divergences may indicate informed money on one side.

- `preracehorsecareerSPBFSP` — expanding mean, lagged (does this horse consistently attract more/less exchange money?)

**Expected value**: MEDIUM. Market microstructure signal.

#### F2. Win-Place Price Ratio

```
implied_place_prob = 1 / bfsp_place
implied_win_prob = 1 / bfsp
wp_ratio = implied_win_prob / implied_place_prob
```

Horses with a higher wp_ratio are perceived as "all or nothing" — they either win or fail to place. Horses with a lower wp_ratio are considered consistent placers.

**Expected value**: MEDIUM. Captures market view of horse consistency.

#### F3. Market Favourite Flag Features

```
is_favourite = 1 if fav contains 'F'
is_joint_favourite = 1 if fav contains 'JF'
favourite_position_shift = was_favourite_last_time - is_favourite_today
```

Being favourite is already captured by BFSP, but the binary flag and changes in favourite status add information about market confidence shifts.

**Expected value**: LOW. Mostly redundant with BFSP-derived features.

---

### Category G: Race Type & Conditions Features

**Rationale**: The `race_type` column (Handicap, Maiden, Stakes, Novice, etc.) and `race_restrictions_age` provide important context. Handicaps behave very differently from non-handicaps.

#### G1. Race Type Encoding

```
is_handicap = 1 if 'handicap' in race_type.lower()
is_maiden = 1 if 'maiden' in race_type.lower()
is_novice = 1 if 'novice' in race_type.lower()
is_stakes = 1 if 'stakes' in race_type.lower() or 'group' in race_type.lower()
is_chase = 1 if race_code in ['C', 'Ch']
is_hurdle = 1 if race_code in ['H', 'Hu']
is_flat = 1 if race_code in ['F']
```

**Expected value**: MEDIUM. Different race types have different dynamics.

#### G2. Horse Race-Type Performance

```
horse_handicap_nfp = avg NFP in handicaps, lagged
horse_nonhandicap_nfp = avg NFP in non-handicaps, lagged
horse_racetype_fit = NFP in today's race type vs career NFP
```

Some horses systematically perform better in handicaps (where they get weight advantages) vs conditions races.

**Expected value**: MEDIUM-HIGH.

#### G3. Surface Type Performance

```
horse_surface_nfp = avg NFP on today's surface type, lagged
horse_surface_runs = runs on today's surface type, lagged
horse_surface_win_rate = win rate on today's surface, lagged
```

All-weather (AW) vs turf is a major factor. Some horses clearly prefer one surface.

**Expected value**: HIGH. Surface preference is a strong and well-documented factor.

#### G4. Age Restriction Context

```
is_age_restricted = 1 if race is for specific age group (e.g., "3yo only")
horse_age_vs_restriction = horse_age - min_age_in_race
```

In age-restricted races, the relative maturity of the horse matters more.

**Expected value**: LOW-MEDIUM.

---

### Category H: Jockey/Trainer Advanced Features

**Rationale**: We compute career-level WIV/NFP/WAX for jockeys and trainers but miss several important conditional performance metrics.

#### H1. Jockey-at-Track Performance

```
jockey_track_win_rate = jockey win rate at this track, expanding, lagged
jockey_track_runs = number of rides at this track, lagged
```

Some jockeys have notable track preferences and specialisations (e.g., course specialists).

**Expected value**: MEDIUM-HIGH. Course specialists are well-documented.

#### H2. Trainer-at-Track Performance

```
trainer_track_win_rate = trainer win rate at this track, expanding, lagged
trainer_track_runs = number of runners at this track, lagged
```

**Expected value**: MEDIUM-HIGH.

#### H3. Jockey Claim Interaction

```
jockeys_claim_numeric = parse numeric value from jockeys_claim
has_claim = 1 if jockeys_claim > 0
claim_x_weight = jockeys_claim * weight_lbs  # effective weight reduction
```

Apprentice jockeys get weight allowances. A talented claimer on a well-handicapped horse is a strong angle.

**Expected value**: MEDIUM.

#### H4. Trainer Recent Form (Hot/Cold)

```
trainer_LR14_win_rate = trainer win rate in last 14 days, lagged
trainer_LR30_win_rate = trainer win rate in last 30 days, lagged
trainer_form_delta = trainer_LR14_win_rate - preracetrainercareerWIV
```

Trainers go through hot and cold streaks due to yard illness, horse fitness, travel patterns, etc.

**Expected value**: MEDIUM-HIGH. Trainer form is one of the strongest non-horse factors.

#### H5. Jockey Recent Form (Hot/Cold)

```
jockey_LR14_win_rate = jockey win rate in last 14 days, lagged
jockey_form_delta = jockey_LR14_win_rate - preracejockeycareerWIV
```

**Expected value**: MEDIUM. Jockey confidence and booking patterns matter.

---

### Category I: OR (Official Rating) Features

**Rationale**: We store `official_rating`, `median_or`, and `max_or_in_race` but barely use them beyond FCS. OR is one of the most predictive features for handicaps.

#### I1. OR Relative to Field

```
or_vs_max = official_rating - max_or_in_race
or_vs_median = official_rating - median_or
or_percentile = rank(official_rating within race) / number_of_runners
```

**Expected value**: HIGH in handicaps. The horse's rating relative to the field is critical for assessing weight burden.

#### I2. OR Trajectory

```
or_change = current OR - OR at last run, lagged
or_change_3 = current OR - OR 3 runs ago, lagged
or_trend = linear regression slope of OR over last 5 runs
```

Rising OR indicates the handicapper has caught up with improvement. Dropping OR may indicate the horse is now "well-handicapped".

**Expected value**: HIGH. OR changes are among the most predictive features in handicaps.

#### I3. OR vs Career Best

```
career_best_or = max(official_rating) across career, lagged
or_vs_best = official_rating - career_best_or
or_off_peak = 1 if (career_best_or - official_rating) > 5
```

Horses racing off their peak OR are often value bets — they've dropped in the handicap and may bounce back.

**Expected value**: MEDIUM-HIGH.

---

### Category J: Seasonal & Temporal Features

**Rationale**: Horse racing has strong seasonal patterns — ground conditions, horse fitness cycles, trainer patterns, and specific race calendar effects.

#### J1. Month/Season Encoding

```
race_month = month of race_date
race_quarter = quarter of year
is_winter_nh = 1 if month in [11, 12, 1, 2, 3] (National Hunt core season)
is_flat_season = 1 if month in [4, 5, 6, 7, 8, 9, 10]
```

**Expected value**: LOW-MEDIUM. Seasonal effects exist but may be captured by going/race-type.

#### J2. Day of Week

```
day_of_week = race_date.dayofweek
is_weekend = 1 if Saturday or Sunday
is_midweek = 1 if Tuesday/Wednesday/Thursday
```

Weekend cards tend to be stronger quality. Midweek racing is often weaker.

**Expected value**: LOW.

#### J3. Horse Seasonal Pattern

```
horse_month_nfp = avg NFP in this calendar month, grouped by horse, lagged
horse_season_win_rate = win rate in this quarter, lagged
```

Some horses have strong seasonal preferences (e.g., better in summer on faster ground).

**Expected value**: MEDIUM for horses with enough data.

#### J4. Campaign Stage

```
runs_this_season = count of runs in current flat/NH season, lagged
is_seasonal_debut = 1 if first run of the season
days_since_season_start = days since April 1 (flat) or November 1 (NH)
```

Horses returning from a break vs those deep into their campaign behave differently.

**Expected value**: MEDIUM.

---

### Category K: Major Race / Class Context Features

**Rationale**: The `major` column flags significant races (Group 1/2/3, Listed, etc.). Performance in major races indicates true class.

#### K1. Major Race Experience

```
major_race_runs = count of runs in major races, lagged
major_race_win_rate = win rate in major races, lagged
major_race_nfp = avg NFP in major races, lagged
```

**Expected value**: MEDIUM. Indicates class ceiling.

#### K2. Class Ladder Position

```
typical_class = median(race_class) across career, lagged
class_deviation = today's race_class - typical_class
is_class_drop = 1 if today's class > typical_class (lower class number = higher quality)
```

Horses dropping in class are consistently profitable angles.

**Expected value**: HIGH. Class drops are one of the most reliable handicapping angles.

---

### Category L: Interaction & Composite Features

**Rationale**: Non-linear interactions between existing features can capture conditional effects that individual features miss.

#### L1. Fitness-Form Composite

```
fitness_form = FinalDSLR * LR3NFPtotal
```

A horse in good recent form AND getting fitter (shorter gaps between runs) is a strong signal.

**Expected value**: MEDIUM.

#### L2. Class-Speed Composite

```
class_speed = RSR * race_class_numeric
```

Speed in context of class — a high speed figure in a high-class race is more valuable than in a low-class race.

**Expected value**: MEDIUM.

#### L3. Experience-Debut Interaction

```
debut_x_sire_wiv = is_debut * stallion_WIV
debut_x_trainer_wiv = is_debut * preracetrainercareerWIV
```

For debut runners, sire and trainer quality become the primary predictors.

**Expected value**: HIGH for debut runners specifically.

#### L4. Weight-OR Efficiency

```
weight_per_or_point = pounds / official_rating
weight_efficiency = weight_per_or_point - median(weight_per_or_point in race)
```

In handicaps, the relationship between weight carried and OR defines value. Horses carrying less weight per OR point have an advantage.

**Expected value**: MEDIUM-HIGH in handicaps.

---

## 3. Priority Ranking

Features ordered by expected impact on model performance:

| Priority | Feature Group | Expected Value | Difficulty | Data Requirement |
|----------|--------------|---------------|------------|-----------------|
| 1 | **B: Actual Lengths Beaten** | VERY HIGH | LOW | `total_dst_bt`, `distbt` |
| 2 | **A: Speed Figures** | HIGH | MEDIUM | `comptime_numeric` |
| 3 | **C: Breeding/Pedigree** | HIGH (for debuts) | MEDIUM | `stallion`, `dam`, `dam_stallion` |
| 4 | **I: OR Features** | HIGH (handicaps) | LOW | `official_rating`, `median_or`, `max_or_in_race` |
| 5 | **D: Equipment Changes** | MEDIUM-HIGH | LOW | `headgear` |
| 6 | **H: Jockey/Trainer Advanced** | MEDIUM-HIGH | MEDIUM | Existing + `jockeys_claim` |
| 7 | **G: Race Type Features** | MEDIUM | LOW | `race_type`, `surface_type`, `race_code` |
| 8 | **E: Draw Bias** | MEDIUM (course-specific) | MEDIUM | `stall`, `track_direction`, `stall_positioning` |
| 9 | **K: Major Race/Class** | MEDIUM | LOW | `major`, `race_class` |
| 10 | **F: Market Intelligence** | MEDIUM | LOW | `odds`, `fav`, `bfsp_place` |
| 11 | **L: Interaction Features** | MEDIUM | LOW | Computed from above |
| 12 | **J: Seasonal/Temporal** | LOW-MEDIUM | LOW | `race_date` |

---

## 4. Implementation Plan

### Phase 1: Quick Wins (LOW difficulty, HIGH impact)

1. **Parse `total_dst_bt` to numeric lengths beaten** — replace approximated RB with actual margins
2. **OR relative features** — `or_vs_max`, `or_vs_median`, `or_change`
3. **Equipment change flags** — `headgear_change`, `first_time_headgear`
4. **Race type encoding** — `is_handicap`, `is_maiden`, surface performance

### Phase 2: Speed Figures (MEDIUM difficulty, HIGHEST impact)

5. **Build speed rating infrastructure** — standard times per (track, distance, going)
6. **Calculate RSR, SFI, GAS** — core speed figure family
7. **Integrate into CustomMetricsEngine** — rolling/career averages with recency weighting

### Phase 3: Pedigree Features (MEDIUM difficulty, HIGH impact for thin-form)

8. **Sire/Damsire performance stats** — win rates, NFP, WIV by stallion
9. **Conditional sire stats** — per going, per distance
10. **Debut interaction features** — sire/trainer quality for first-time runners

### Phase 4: Advanced Features (MEDIUM-HIGH difficulty)

11. **Draw bias modelling** — track/distance/going specific draw advantages
12. **Jockey/Trainer at-track stats** — course specialist identification
13. **Trainer/Jockey hot form** — 14/30 day rolling form
14. **Composite interaction features** — non-linear feature crosses

---

## 5. Testing Strategy

For each new feature:

1. **Lookahead check**: Verify `.shift(1)` is applied — no future data leaks
2. **Distribution check**: Plot feature distribution, check for extreme outliers
3. **Univariate predictive power**: Compute AUC-ROC of feature alone vs `won`
4. **Marginal improvement test** (Benter method): Add feature to existing model, measure log-loss change
5. **Walk-forward validation**: Test on 3+ temporal folds to ensure stability
6. **Correlation check**: Ensure feature isn't >0.95 correlated with existing features (redundancy)
7. **Coverage check**: What % of runners have non-null values for this feature?

### Key Metrics for Feature Evaluation

| Metric | Threshold for Inclusion |
|--------|------------------------|
| Single-feature AUC-ROC | > 0.52 (better than random) |
| Marginal log-loss improvement | > 0.001 (statistically significant) |
| Walk-forward consistency | Improves in >60% of folds |
| Feature coverage | > 70% non-null for established horses |
| Correlation with existing | < 0.90 (not redundant) |

---

## 6. Special Considerations

### Parsing `total_dst_bt` and `distbt`

These columns contain mixed formats that need careful parsing:
- Numeric: `"2.5"`, `"10"`, `"0.75"`
- Text abbreviations: `"nk"` (neck=0.2L), `"shd"` (short head=0.1L), `"hd"` (head=0.15L), `"nse"` (nose=0.05L), `"dht"` (dead heat=0)
- Combined: `"2nk"` (2 and a neck), `"shd"`
- Winner: `""` or `"0"` (winner was 0 lengths beaten)
- Symbols: may include `"dist"` (distanced, > 30 lengths)

A robust parser is needed before feature computation.

### Sire Data Sparsity

Many sires have few runners in the dataset. Use Bayesian shrinkage:
```
adjusted_sire_stat = (n * sire_stat + k * population_stat) / (n + k)
```
Where `n` = sire's sample size, `k` = shrinkage constant (e.g., 20).

### Speed Figure Reliability

`comptime_numeric` may be null or unreliable for:
- National Hunt races (no sectional timing at most courses)
- Abandoned races
- Walk-over/void races

Handle gracefully with null propagation and confidence flags.

---

## 7. Summary

**Current state**: 70+ features from 19 metric families, using ~60% of available data columns.

**Proposed additions**: 42 new features from 12 categories, exploiting the remaining ~40% of unused database columns.

**Expected impact**: The highest-priority additions (actual lengths beaten, speed figures, pedigree features, OR-relative features) could significantly improve model discrimination, particularly for:
- **Thin-form horses** (debuts, lightly-raced) via pedigree features
- **Handicaps** via OR-relative and weight-efficiency features
- **All races** via speed figures and actual margins

The implementation follows the existing lag-safe architecture and integrates with `CustomMetricsEngine` and `PreRaceBuilder`.
