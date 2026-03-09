# PROMPT: Build Custom Racing Metrics Module for Ultra Betting

You are working on the Ultra Betting project — an autonomous thoroughbred horse racing prediction system. Your task is to build a Python module `src/data/custom_metrics.py` that calculates all the custom performance metrics we've developed from years of research. These metrics are the core IP of the system and must be implemented exactly as specified.

## Context

We have a matched dataset of UK/IRE horse racing results joined to Betfair SP price data, stored in a SQLite database. Each row is one runner in one race. The data includes: `date`, `course`, `race_time`, `raceid` (unique race identifier), `horse_name`, `finish_position` (placing_numerical), `jockey`, `trainer`, `sp`, `weight_lbs`, `bsp` (Betfair Starting Price), `number_of_runners` (field_size), `official_rating`, `prize_money`, `going`, `distance`, `race_class`, `comment` (in-running race comment text from results), `age`, `headgear`.

The module must be a single Python file using pandas. All calculations must be **lag-safe** — every feature for a given race must use ONLY data available BEFORE that race (use `.shift(1)` or `.lag` equivalents). No lookahead bias.

## Required Output

Create `src/data/custom_metrics.py` with a class `CustomMetricsEngine` that has a method `calculate_all(df: pd.DataFrame) -> pd.DataFrame` which takes the raw matched data and returns it with all custom metric columns appended.

Also create comprehensive tests in `tests/test_custom_metrics.py`.

---

## METRIC SPECIFICATIONS

### 1. Recency Weight System (foundation for all rolling metrics)

For each horse, look back at their last 10 runs. Create binary indicators for whether each lagged run exists, and harmonic recency weights:

```
LR1weight = 1.0    (most recent run)
LR2weight = 0.5
LR3weight = 0.333
LR4weight = 0.25
LR5weight = 0.2
LR6weight = 0.167
LR7weight = 0.143
LR8weight = 0.125
LR9weight = 0.111
LR10weight = 0.1
```

Only assign weight if the run exists (horse actually had that many prior runs). Compute weight sums for 3/5/10 run windows:
- `LR3wsum` = sum of weights for last 3 runs that exist
- `LR5wsum` = sum of weights for last 5 runs that exist  
- `LR10wsum` = sum of weights for last 10 runs that exist

Also compute run counts: `LR3COUNT`, `LR5COUNT`, `LR10COUNT` — how many of the last 3/5/10 runs actually exist.

### 2. Confidence Intervals (data completeness)

Per race, per horse:
```
CIL3  = (LR3COUNT / (number_of_runners * 3)) * 100
CIL5  = (LR5COUNT / (number_of_runners * 5)) * 100
CIL10 = (LR10COUNT / (number_of_runners * 10)) * 100
```

This measures what proportion of the theoretical maximum data we actually have. Low CI means the rolling metrics are unreliable.

### 3. NFP — Normalised Finishing Position

If the raw data contains an `NFP` column, use it directly. If not, calculate:
```
NFP = (number_of_runners - placing_numerical) / (number_of_runners - 1)
```
This gives 1.0 for a winner and 0.0 for last place, normalised by field size.

Then compute:
- `preracehorsecareerNFP` = expanding mean NFP for this horse, lagged by 1 (pre-race)
- `LRNFP` = NFP from last run (lag 1)
- `LR3NFPtotal` = mean NFP over last 3 runs
- `LR5NFPtotal` = mean NFP over last 5 runs  
- `LR10NFPtotal` = mean NFP over last 10 runs

### 4. RB — Race Beaten (lengths-behind proxy)

If `RB` column exists in raw data, use it. If not, approximate:
```
RB = 1 - (placing_numerical - 1) / (number_of_runners - 1)
```
This gives 1.0 for a winner, 0.0 for last.

Then compute:
- `preracehorsecareerRB` = expanding mean RB, lagged
- `FSARB` = RB * (number_of_runners / median_field_size) — field-size adjusted RB
- `preracehorsecareerFSARB` = expanding mean FSARB, lagged
- `preracehorsecareerFSARB2` = expanding mean of FSARB², lagged (captures consistency)

### 5. xWINRAND and WIV — Expected Wins and Win Index Value

```
xWINRAND = 1 / number_of_runners   (expected win probability under random chance)
```

**WIV (Win Index Value)** — the key market-adjusted strike rate:
```
WIV = cumulative_wins / cumulative_xWINRAND
```
A WIV of 1.0 means winning exactly as often as random chance predicts. WIV > 1.0 means outperforming. Compute pre-race (lagged) versions for horse, trainer, jockey, and trainer-jockey combinations:

- `preracehorsecareerWIV = lag(cumsum(wins) / cumsum(xWINRAND))`
- `preracetrainercareerWIV = lag(cumsum(wins) / cumsum(xWINRAND))` grouped by trainer
- `preracejockeycareerWIV = lag(cumsum(wins) / cumsum(xWINRAND))` grouped by jockey
- `trainerjockeycareerWIV = lag(cumsum(wins) / cumsum(xWINRAND))` grouped by (trainer, jockey)

### 6. WAX, WOA, CWO — Wins Above Expected, Wins Over Average, Cumulative Wins Over

```
WAX (per run) = win_flag - xWINRAND     (did they beat random expectation?)
WOA (per run) = win_flag - field_avg_win_rate   (did they beat the field average?)
CWO = cumulative WAX
```

Compute expanding cumulative sums, lagged, for horse/trainer/jockey:
- `preracehorsecareerWAX`, `preracehorsecareerWOA`, `preracehorsecareerCWO`
- Same for trainer and jockey

### 7. ORR2 — Odds-to-Runner Ratio

```
bf_prob = 1 / BFSP
fs_prob = 1 / number_of_runners
ORR2 = bf_prob / fs_prob
```

This normalises market assessment by field size. An ORR2 > 1 means the market rates this horse above average for the field.

Then compute **recency-weighted ORR2 scores** using the harmonic weights:
```
LR3_ORR2 = (lag1_ORR2 * 1.0 + lag2_ORR2 * 0.5 + lag3_ORR2 * 0.333)
LR3_RWO = LR3_ORR2 / (sum of applicable weights)
```
Same pattern for LR5_RWO and LR10_RWO.

Also compute:
- `preracehorsecareerORR2` = expanding mean ORR2, lagged
- `LR_ORR2` = last run's ORR2

### 8. EPF — Early Position Figure (NLP from race comments)

Parse the `comment` field using regex to classify the horse's early race position:

```python
def calculate_epf(comment: str) -> float:
    if not comment or not isinstance(comment, str):
        return 3.0  # default midfield
    c = comment.lower()
    
    # Leaders (score 6)
    if re.search(r'made virtually all|made all|made most|led to\b|led,|led early|led after|led before|led until|led over|soon led', c):
        return 6.0
    # Disputed lead (5.5)
    if re.search(r'disputed|disputed lead|with leader', c):
        return 5.5
    # Chased leader (5)
    if re.search(r'chased leader|tracked leader|chased winner', c):
        return 5.0
    # Prominent (4)
    if re.search(r'pressed leader|tracked leaders|chased leaders|prominent|close up|in touch|in-touch|pressing leaders|chasing leaders|tracked front pair|tracked leading pair|chased leading|tracked\s|tracking leaders', c):
        return 4.0
    # Front of midfield (3)
    if re.search(r'front of mid-division|front of mid division|front of midfield', c):
        return 3.0
    # Held up midfield (3)
    if re.search(r'held up in midfield|held up in mid-division|towards rear of midfield|held up in touch', c):
        return 3.0
    # Behind/rear (1)
    if re.search(r'towards rear|held up behind|behind|held up|held up,|last pair', c):
        return 1.0
    if re.search(r'in rear|always rear', c):
        return 1.0
    
    return 3.0  # default
```

Then compute derived EPF metrics:
```
EPF2 = -0.74 + (EPF * 0.8637) + (number_of_runners * 0.09375)   # field-size adjusted
EPF3 = EPF * (number_of_runners - placing_numerical) / (number_of_runners - 1)  # adjusted for result
```

Per race:
```
RPS = sum(EPF) across all runners in race  (Race Pace Score)
pace_pressure = (count of EPF > 4) / number_of_runners * 100  (% prominent runners)
prom_runner = 1 if EPF > 4 else 0
```

Lagged rolling versions for horse (LR_EPF, LR2_EPF through LR5_EPF, and same for EPF2, EPF3).

Career averages (pre-race, lagged) for:
- `Horse_Career_EPF` = expanding mean EPF2 for the horse
- `Jockey_Career_EPF` = expanding mean EPF2 for the jockey
- `trainer_Career_EPF` = expanding mean EPF2 for the trainer

### 9. FSS — Field Size Stability

For each of the last 10 runs, compute the field size delta squared:
```
LRn_FSDelta = today's_runners - lag_n_runners
LRn_FSDelta2 = LRn_FSDelta^2
```

Then:
```
FSS = sqrt(sum of all FSDelta2 values) / count_of_runs_that_exist
```

High FSS means the horse has been racing in wildly varying field sizes. Low FSS means consistent competitive environments. This is a RMSD-style measure of field size volatility.

### 10. FCS — Field Class Strength

Same architecture as FSS but using race-level mean official rating:
```
Race_avgOR = mean(official_rating) per race

For last 10 runs, compute:
LRn_FCSDELTA = today's Race_avgOR - lag_n Race_avgOR
LRn_FCSDELTA2 = LRn_FCSDELTA^2

FCS = sqrt(sum of FCSDELTA2) / LR10COUNT
```

High FCS means the horse has been bouncing between wildly different class levels.

### 11. PFD — Probability-Field Difference

```
PFD = bf_prob - fs_prob
```

Positive PFD means the market rates this horse above random chance. Compute recency-weighted rolling averages:
```
PFD3 = weighted_sum(lag1_PFD..lag3_PFD, weights) / LR3wsum
PFD5 = weighted_sum(lag1_PFD..lag5_PFD, weights) / LR5wsum
PFD10 = weighted_sum(lag1_PFD..lag10_PFD, weights) / LR10wsum
```

### 12. WPMRF / PMW — Prize Money Metrics

**WPMRF** (Win Prize Money Raced For):
Recency-weighted average of prize money from last 3/5/10 runs:
```
WPMRF3 = weighted_sum(lag1_prize..lag3_prize, weights) / LR3wsum
WPMRF5 = weighted_sum(lag1_prize..lag5_prize, weights) / LR5wsum
WPMRF10 = weighted_sum(lag1_prize..lag10_prize, weights) / LR10wsum
```

**PMW** (Prize Money Won — performance-adjusted):
```
PMW = (prize_money / 100) * RB^2
```
Then compute recency-weighted PMW3, PMW5, PMW10 using same pattern.

Also compute race-level:
```
RACE_WPMRF = sum of all runners' prize_money in this race
```

### 13. OFS — Odds × Field Size

```
OFS = (1/BFSP) * number_of_runners
```
Same as ORR2 conceptually. Compute recency-weighted rolling OFS1, OFS3, OFS5, OFS10.

### 14. DSLR — Days Since Last Run (enhanced)

Beyond simple days_since_run, compute the RATE OF CHANGE of gaps between runs:
```
DSLR1 = days between this run and lag 1
DSLR2 = days between this run and lag 2
DSLR3 = days between this run and lag 3
DSLR4 = days between this run and lag 4

DSLR12diff = (DSLR2 - DSLR1) * 1.0
DSLR23diff = (DSLR3 - DSLR2) * 0.5
DSLR34diff = (DSLR4 - DSLR3) * 0.33

WgtDSLR = DSLR12diff + DSLR23diff + DSLR34diff
FinalDSLR = WgtDSLR / LR3COUNT
```

This captures whether the horse's campaign is accelerating (being run more frequently) or decelerating.

### 15. LRP Index — Last Run Placed Score

Position-weighted momentum score for jockeys:
```
LR_Placed = did the horse place last time? (lag of places flag)
LR_Pos = finishing position last time (lag of placing_numerical)

LRP1Score = 17.41 if (LR_Placed == 1 AND LR_Pos == 1) else 0
LRP2Score = 14.77 if (LR_Placed == 1 AND LR_Pos == 2) else 0
LRP3Score = 12.62 if (LR_Placed == 1 AND LR_Pos == 3) else 0
LRPTotalScore = LRP1Score + LRP2Score + LRP3Score
```

Then per jockey:
```
totaljockeyLRPscore = cumsum(LRPTotalScore) per jockey
totaljockeyrides = cumsum(Runs) per jockey
totalLRPjockeyindex = (totaljockeyLRPscore / totaljockeyrides) * 10
```

### 16. Race Strength Metrics

Per-race averages of pre-race horse career metrics:
```
RACE_RB = mean(preracehorsecareerRB) across all runners
RACE_WIV = mean(preracehorsecareerWIV) across all runners
RACE_NFP = mean(preracehorsecareerNFP) across all runners
RACE_Wins = mean(preracehorsecareerWins) across all runners
RACE_WOA = mean(preracehorsecareerWOA) across all runners
```

Then lag each per horse: `LR_RACE_RB`, `LR_RACE_WIV`, etc. — so for each horse we know the average quality of opposition it faced in its last race.

### 17. Pace Metrics (horse/trainer/jockey level)

Using a `RunStyle2` column if available, or EPF as proxy:
```
racepacescore = sum(RunStyle) per race
racepaceindex = racepacescore / number_of_runners

horsepaceindex = cumulative sum of RunStyle / cumulative runs per horse (pre-race, lagged)
trainerpaceindex = same per trainer
jockeypaceindex = same per jockey
```

### 18. Trainer-Jockey Combination Metrics

Grouped by (trainer, jockey), compute pre-race lagged versions of:
- `trainerjockeycareerWIV`
- `trainerjockeycareerNFP`
- `trainerjockeyWAX`, `trainerjockeyWOA`, `trainerjockeyCWO`

### 19. Within-Race Rankings

After computing ALL the above metrics, rank every horse within its race on every continuous metric. Use `rank(ascending=False, method='min')` with NaN handling. Prefix with `r` for rank:

Horse-level ranks: `rNFP`, `rNFPLR3`, `rNFPLR5`, `rNFPLR10`, `horseRBrank`, `horseFSARBrank`, `horseFSARB2rank`, `horsexRBMARrank`, `horseNFPrank`, `horseWAXrank`, `horseWIVrank`, `horseWOArank`, `horseCWOrank`, `horseRunsrank`, `horseWinsrank`, `horsePlacesrank`

ORR2/RWO ranks: `rORR2LR`, `rRWOLR3`, `rRWOLR5`, `rRWOLR10`

EPF ranks: `rEPF_LR`, `rEPF2_LR`, `rEPF3_LR`, `rJockeyEPF`, `rTrainerEPF`, `rHorseCareerEPF`

Other ranks: `rDSLR`, `rFSS`, `rFCS`, `rPFD3`, `rPFD5`, `rPFD10`, `rWPMRF3`, `rWPMRF5`, `rWPMRF10`, `rPMW3`, `rPMW5`, `rPMW10`, `rOFS3`, `rOFS5`, `rOFS10`, `rTJWIV`, `rTJNFP`

Trainer ranks: `trainerRBrank`, `trainerNFPrank`, `trainerWIVrank`, `trainerWAXrank`, `trainerWOArank`, `trainerCWOrank`

Jockey ranks: `jockeyRBrank`, `jockeyNFPrank`, `jockeyWIVrank`, `jockeyWAXrank`, `jockeyWOArank`, `jockeyCWOrank`, `jockeyLRIrank`

---

## IMPLEMENTATION REQUIREMENTS

1. **Group-then-lag pattern**: All cumulative/rolling features must be grouped by the appropriate entity (horse_name, trainer, jockey, or (trainer, jockey)), sorted by date, and lagged by 1 to avoid lookahead.

2. **Handle NaN gracefully**: First-time runners will have NaN for all lagged features. Fill with 0 or population mean as appropriate. Document which strategy you use.

3. **Performance**: The dataset may have 500K+ rows. Use vectorised pandas operations, avoid iterating row-by-row. Use `.transform()` for group-level calculations.

4. **Race ID**: If `raceid` doesn't exist, create it as `f"{date}_{course}_{race_time}"`.

5. **Configurable windows**: Make the lookback windows (3, 5, 10) configurable but default to these values.

6. **Return all original columns plus all new metric columns.**

7. **Write comprehensive tests** that verify:
   - No lookahead bias (a feature for race N uses only data from races < N)
   - Correct handling of first-time runners
   - Correct recency weighting arithmetic
   - Within-race ranks sum correctly
   - EPF NLP parsing handles edge cases

8. **Docstrings**: Every method should have a clear docstring explaining the metric's purpose and citing whether it's a standard metric or our custom IP.

The file should integrate cleanly with our existing `src/data/features.py` — either as a replacement or as an additional module that `features.py` calls.
