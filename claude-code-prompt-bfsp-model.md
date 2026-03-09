# PROMPT: Build BFSP Prediction & Race Probability Model

You are working on the Ultra Betting project — an autonomous thoroughbred horse racing prediction system. Your task is to build the core predictive model in `src/models/` that predicts Betfair Starting Prices (BFSP) and true win probabilities for each runner in a race.

## The Critical Problem

We have **compiled race results** — historical data where we know the outcome. But at prediction time, we only know the **declared runners** for an upcoming race. We do NOT know finishing positions, BFSP, or any post-race data.

The model must therefore:
1. Take a list of declared runners for a future race (horse, jockey, trainer, course, distance, going, class, weight, draw, field size)
2. Look up each entity's FULL HISTORY from the results database
3. Build a **pre-race feature vector** for each runner using ONLY historical data
4. Predict the probability of each runner winning
5. Convert those probabilities into predicted BFSP
6. Where actual BFSP is available (pre-race from Betfair markets), blend model probability with market probability using Benter's two-stage method to identify overlays

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PREDICTION PIPELINE                       │
│                                                             │
│  Declared Runners ──→ History Lookup ──→ Pre-Race Features  │
│                                                             │
│  Pre-Race Features ──→ Stage 1: Fundamental Model ────────→ P_model(i)
│                                                             │
│  P_model(i) + BFSP ──→ Stage 2: Benter Blending ─────────→ P_combined(i)
│                                                             │
│  P_combined(i) vs P_market(i) ──→ Stage 3: Overlay ──────→ Bet/No Bet
│                                                             │
│  Bet sizing ──→ Fractional Kelly on edge ─────────────────→ Stake
└─────────────────────────────────────────────────────────────┘
```

---

## Files to Create

```
src/models/
├── prerace_builder.py     # Builds pre-race feature vectors from history
├── probability_model.py   # Stage 1: Fundamental win probability model  
├── benter_blend.py        # Stage 2: Log-linear blending with market prices
├── overlay_detector.py    # Stage 3: Identifies value bets and sizes stakes
├── trainer.py             # Model training with walk-forward validation
├── evaluator.py           # Backtesting and performance metrics
└── __init__.py

tests/
├── test_prerace_builder.py
├── test_probability_model.py
└── test_benter_blend.py
```

---

## DETAILED SPECIFICATIONS

### 1. `prerace_builder.py` — Pre-Race Feature Vector Construction

This is the most important file. It takes declared runners and builds feature vectors from historical data.

#### Class: `PreRaceBuilder`

```python
class PreRaceBuilder:
    """Build pre-race feature vectors by looking up each entity's history.
    
    At prediction time we know ONLY:
    - horse_name, age, sex
    - jockey, trainer
    - course, distance, going, race_class
    - weight_carried, draw_position
    - number_of_runners (field size)
    - (optionally) current Betfair price / morning price
    
    We must look up everything else from the historical results database.
    """
    
    def __init__(self, history_df: pd.DataFrame):
        """
        Args:
            history_df: Complete historical results with custom metrics
                        already calculated (from CustomMetricsEngine).
                        This is our "database" of past performances.
        """
```

#### Method: `build_features(declared_runners: pd.DataFrame, race_date: str) -> pd.DataFrame`

For each declared runner, look up the horse's history from `history_df` where `date < race_date` and extract the MOST RECENT pre-race state. This means taking the LAST ROW of each horse's history (which already has all the lagged cumulative metrics from CustomMetricsEngine).

**Horse features to extract from last historical appearance:**
```
- preracehorsecareerRuns (how many career runs)
- preracehorsecareerWins
- preracehorsecareerPlaces
- preracehorsecareerNFP (career avg normalised finishing position)
- preracehorsecareerRB (career avg race beaten)
- preracehorsecareerFSARB (field-size adjusted RB)
- preracehorsecareerWIV (win index value — wins vs expected)
- preracehorsecareerWAX (wins above expected)
- preracehorsecareerWOA (wins over average)
- preracehorsecareerCWO (cumulative wins over)
- Horse_Career_EPF (career early position figure)
- horsepaceindex (cumulative pace tendency)
- last_finish_pos (most recent finishing position)
- last_bsp (most recent BFSP)
- last_3_avg_position
- last_5_avg_position
- LR3NFPtotal, LR5NFPtotal, LR10NFPtotal (recent NFP averages)
- LR3_RWO, LR5_RWO, LR10_RWO (recency-weighted odds-runner ratio)
- FSS (field size stability)
- FCS (field class strength)  
- PFD3, PFD5, PFD10 (probability-field difference)
- WPMRF3, WPMRF5, WPMRF10 (prize money raced for)
- PMW3, PMW5, PMW10 (prize money won, performance-adjusted)
- OFS3, OFS5, OFS10 (odds-field-size scores)
- FinalDSLR (weighted days-since-last-run trend)
- days_since_last_run (computed fresh: race_date - last appearance date)
- CIL3, CIL5, CIL10 (data confidence intervals)
```

**Jockey features (from jockey's most recent ride in history):**
```
- preracejockeycareerWins
- preracejockeycareerRuns
- preracejockeycareerWIV
- preracejockeycareerNFP
- preracejockeycareerRB
- preracejockeycareerWAX
- preracejockeycareerWOA
- Jockey_Career_EPF
- jockeypaceindex
- totalLRPjockeyindex (last-run-placed momentum score)
```

**Trainer features (from trainer's most recent runner in history):**
```
- preracetrainercareerWins
- preracetrainercareerRuns
- preracetrainercareerWIV
- preracetrainercareerNFP
- preracetrainercareerRB
- preracetrainercareerWAX
- preracetrainercareerWOA
- trainer_Career_EPF
- trainerpaceindex
```

**Trainer-Jockey combination features:**
```
- trainerjockeycareerWIV
- trainerjockeycareerNFP
```

**Today's race features (known from declarations):**
```
- number_of_runners (field_size)
- distance_furlongs
- going_numeric (encoded)
- race_class_numeric (encoded)
- weight_lbs
- draw_position
- age
```

**Derived features to compute at prediction time:**
```
- going_preference: deviation of today's going from horse's historical avg going
- distance_preference: deviation of today's distance from horse's historical avg distance
- class_change: today's class vs horse's last race class
- weight_change: today's weight vs horse's last race weight
- course_win_pct: horse's win rate at this specific course
- course_runs: horse's total runs at this specific course
```

**Handling unknown entities:**
- First-time runners (no history): fill all horse features with population medians from the training set. Flag with `is_debut = 1`.
- Unknown jockey/trainer: fill with population medians. This should be rare.
- Known horse, new jockey/trainer combo: fill combination features with NaN/0, individual features available.

**Handling field-level features at prediction time:**
Once all individual runners have features, compute within-race metrics:
```
- field_avg_career_nfp = mean of all runners' preracehorsecareerNFP
- field_avg_career_wiv = mean of all runners' preracehorsecareerWIV
- field_avg_career_rb = mean of all runners' preracehorsecareerRB
- rank every runner within the race on every continuous metric (same as in custom_metrics)
```

---

### 2. `probability_model.py` — Stage 1: Fundamental Probability Model

#### Class: `FundamentalModel`

The model predicts P(win) for each runner. This is a CLASSIFICATION problem (win = 1, lose = 0) where we want calibrated probabilities, not just rankings.

**Model options (implement both, make configurable):**

**Option A: Conditional Logit (Benter's original)**
```python
from sklearn.linear_model import LogisticRegression
# Fit on individual runners, but probabilities must be normalised within each race
# After predicting raw P(win) for each runner, normalise:
# P_norm(i) = P_raw(i) / sum(P_raw(j) for j in race)
```

**Option B: XGBoost (modern extension)**
```python
import xgboost as xgb
# XGBClassifier with careful hyperparameter tuning
# Same normalisation step after prediction
```

**Option C: LambdaMART / Learning-to-Rank**
Consider using XGBoost's `rank:pairwise` objective which directly optimises for ranking runners within a race. This is more natural than binary classification for this problem.

**Feature selection approach (per Benter):**
Not all features should go into the model. Apply Benter's variable selection test:
1. First, include log(BFSP_implied_prob) as a feature (the public odds)
2. Add candidate features one at a time
3. Keep only features that improve log-loss BEYOND what the public odds already capture
4. Use permutation importance or SHAP values to identify which features add genuine signal vs noise

Implement a `select_features()` method that performs this analysis and returns the optimal feature set.

**Feature list (starting set — to be pruned by selection):**

Core horse form:
```
preracehorsecareerNFP, preracehorsecareerRB, preracehorsecareerFSARB,
preracehorsecareerWIV, preracehorsecareerWAX, preracehorsecareerCWO,
LR3NFPtotal, LR5NFPtotal, LR10NFPtotal,
last_finish_pos, last_bsp, days_since_last_run
```

Market-derived:
```
LR3_RWO, LR5_RWO, LR10_RWO,
PFD3, PFD5, PFD10,
OFS3, OFS5, OFS10
```

Stability/environment:
```
FSS, FCS, FinalDSLR, CIL3, CIL5, CIL10,
going_preference, distance_preference, class_change, weight_change,
course_win_pct, course_runs
```

Prize money / class proxy:
```
WPMRF3, WPMRF5, WPMRF10,
PMW3, PMW5, PMW10
```

Jockey/trainer:
```
preracejockeycareerWIV, preracejockeycareerNFP,
preracetrainercareerWIV, preracetrainercareerNFP,
trainerjockeycareerWIV, trainerjockeycareerNFP,
Jockey_Career_EPF, trainer_Career_EPF, Horse_Career_EPF
```

Race shape:
```
horsepaceindex, jockeypaceindex, trainerpaceindex
```

Today's race:
```
number_of_runners, distance_furlongs, going_numeric, race_class_numeric,
weight_lbs, draw_position, age, is_debut
```

Within-race ranks (selected subset):
```
rNFP, rNFPLR5, horseWIVrank, horseRBrank,
rRWOLR5, rRWOLR10, rPFD5, rFSS, rFCS
```

---

### 3. `benter_blend.py` — Stage 2: Log-Linear Probability Blending

This is Benter's key innovation. The fundamental model alone is not enough. The public odds (BFSP) encode enormous information. The optimal strategy is to COMBINE them.

#### Class: `BenterBlender`

```python
class BenterBlender:
    """Benter (1994) two-stage log-linear probability blending.
    
    P_combined(i) = [P_model(i)^(1-λ) * P_public(i)^λ] / Z
    
    Where:
        P_model(i) = fundamental model win probability for horse i
        P_public(i) = 1/BFSP for horse i (market implied probability)
        λ (lambda) = blending weight, typically ~0.80
        Z = normalisation constant so probabilities sum to 1 within race
    
    λ = 0.80 means the public odds provide ~80% of the signal.
    The model's value is in the remaining 20% where it systematically 
    disagrees with the market.
    """
```

**Methods:**

```python
def blend(self, race_df: pd.DataFrame, lambda_: float = 0.80) -> pd.DataFrame:
    """Blend model and market probabilities for a single race.
    
    Args:
        race_df: DataFrame with columns 'p_model' and 'bfsp' for runners in ONE race
        lambda_: blending weight (0 = pure model, 1 = pure market)
    
    Returns:
        DataFrame with added columns:
        - p_public: market implied probability (1/BFSP)  
        - p_combined: blended probability
        - model_fair_bfsp: 1/p_combined (what BFSP "should" be according to blended model)
        - edge: (p_combined / p_public) - 1 (positive = overlay, horse is underpriced)
        - edge_pct: edge * 100
    """
```

```python
def optimise_lambda(self, val_df: pd.DataFrame) -> float:
    """Find optimal lambda by minimising log-loss on validation data.
    
    Try lambda values from 0.0 to 1.0 in steps of 0.05.
    For each lambda, blend probabilities and compute log-loss against actual outcomes.
    Return the lambda that minimises log-loss.
    
    This should be run on a held-out validation set, NOT training data.
    """
```

**When BFSP is not yet available (pre-race prediction before market forms):**
If we're predicting before the Betfair market has enough liquidity, skip blending and use P_model directly. Flag these predictions as `blend_available = False`.

**When BFSP IS available (typical case — morning of race day):**
Use the morning WAP or current Betfair price as P_public for blending.

---

### 4. `overlay_detector.py` — Stage 3: Identify Value and Size Stakes

#### Class: `OverlayDetector`

```python
class OverlayDetector:
    """Identify overlays (underpriced runners) and size bets using fractional Kelly.
    
    An overlay exists when P_combined > P_market, meaning our blended model
    thinks the horse has a higher chance of winning than the market implies.
    
    Edge = P_combined / P_market - 1
    
    We only bet when edge exceeds a minimum threshold (default 5%).
    """
```

**Kelly staking:**
```python
def kelly_stake(self, p_combined: float, bfsp: float, 
                kelly_fraction: float = 0.25, bankroll: float = 1000) -> float:
    """Calculate stake using fractional Kelly criterion.
    
    f* = (p * b - q) / b
    where p = p_combined, b = bfsp - 1, q = 1 - p
    
    Actual stake = kelly_fraction * f* * bankroll
    
    Apply constraints:
    - min_edge: don't bet if edge < 5%
    - max_stake_pct: no single bet > 2% of bankroll  
    - min_stake: £2 minimum
    - max_stake: £500 maximum
    """
```

**Output: a bet card**
```python
def generate_bet_card(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Generate the day's bet card from predictions.
    
    Returns DataFrame with columns:
    - date, course, race_time, horse_name
    - p_model, p_public, p_combined
    - bfsp (actual/current), model_fair_bfsp
    - edge, edge_pct
    - kelly_f, adjusted_kelly_f
    - stake_gbp
    - expected_value_gbp
    - bet_type: 'BSP' or 'LIMIT'
    
    Sorted by expected value descending.
    Only includes runners meeting minimum edge threshold.
    """
```

---

### 5. `trainer.py` — Model Training with Walk-Forward Validation

#### Class: `ModelTrainer`

**CRITICAL: Walk-forward validation only. No random splits. No shuffling.**

```python
class ModelTrainer:
    """Train and validate the prediction model using strict temporal ordering.
    
    Walk-forward approach:
    1. Sort all data by date
    2. Define an expanding or rolling training window
    3. For each validation fold, train on all data BEFORE the fold date
    4. Predict on the fold period
    5. Evaluate predictions against actual results
    
    This simulates real-world deployment where we can only use past data.
    """
```

**Training pipeline:**
```python
def train(self, full_history_df: pd.DataFrame, config: dict) -> dict:
    """Full training pipeline.
    
    Steps:
    1. Load history with custom metrics already calculated
    2. Sort by date
    3. Split into walk-forward folds (e.g., train on years 1-2, validate on month 3)
    4. For each fold:
       a. Build PreRaceBuilder from training portion
       b. Build pre-race features for validation runners
       c. Train FundamentalModel on training portion
       d. Predict on validation portion
       e. Blend with actual BFSP using BenterBlender
       f. Evaluate predictions
    5. After all folds: retrain final model on ALL data
    6. Save model artifacts
    
    Returns dict of metrics across all folds.
    """
```

**Walk-forward fold structure:**
```python
def create_folds(self, df: pd.DataFrame, 
                 min_train_days: int = 365,
                 val_window_days: int = 30,
                 step_days: int = 30) -> list[tuple]:
    """Create temporal train/validation folds.
    
    Example with 3 years of data:
    Fold 1: Train on months 1-12,  Validate on month 13
    Fold 2: Train on months 1-13,  Validate on month 14
    Fold 3: Train on months 1-14,  Validate on month 15
    ...etc
    
    Returns list of (train_end_date, val_start_date, val_end_date) tuples.
    """
```

**Hyperparameter tuning:**
```python
def tune_hyperparameters(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Tune model hyperparameters on the most recent validation fold.
    
    For XGBoost, tune:
    - n_estimators: [200, 500, 1000]
    - max_depth: [4, 6, 8]
    - learning_rate: [0.01, 0.05, 0.1]
    - min_child_weight: [3, 5, 10]
    - subsample: [0.7, 0.8, 0.9]
    - colsample_bytree: [0.7, 0.8, 0.9]
    
    Use log-loss as the objective. Use early stopping on validation set.
    
    Also tune lambda (blending weight) on this fold.
    """
```

---

### 6. `evaluator.py` — Backtesting and Performance Metrics

#### Class: `ModelEvaluator`

```python
class ModelEvaluator:
    """Evaluate model predictions against actual race outcomes.
    
    Metrics are computed at multiple levels:
    - Per-race: calibration within individual races
    - Per-day: daily P&L simulation
    - Overall: aggregate performance across the test period
    """
```

**Metrics to compute:**

```python
def evaluate(self, predictions_df: pd.DataFrame) -> dict:
    """Comprehensive evaluation of model predictions.
    
    Probability calibration:
    - log_loss: primary metric — are probabilities well-calibrated?
    - brier_score: mean squared error of probability estimates
    - f2_statistic: Bolton & Chapman goodness-of-fit
      f2 = 1 - [sum(-outcome * ln(p_predicted))] / [sum(-outcome * ln(1/N))]
      f2 > 0 means model beats uniform prediction
    
    Discrimination:
    - auc_roc: can the model rank winners above losers?
    - top_n_accuracy: what % of winners are in the model's top 1/2/3 picks?
    - favourite_accuracy: when model picks a favourite, how often does it win?
    
    Betting performance:
    - simulated_roi: backtest P&L using actual BFSP
    - simulated_roi_by_month: monthly breakdown
    - strike_rate: what % of bets win?
    - avg_winner_bfsp: average BFSP of winning bets
    - avg_edge_of_bets: average edge of bets placed
    - max_drawdown: worst peak-to-trough decline in cumulative P&L
    - sharpe_ratio: daily P&L mean / daily P&L std
    - longest_losing_streak
    
    Calibration analysis:
    - calibration_curve: group predictions into deciles, compare predicted vs actual win rate
    - overround_analysis: do our probability sums exceed 1.0? By how much?
    
    Feature importance:
    - permutation_importance: which features matter most?
    - shap_values: SHAP analysis of feature contributions (if XGBoost)
    
    Comparison baselines:
    - vs_market: is the model better than just using BFSP implied probabilities?
    - vs_favourite: is the model better than always backing the favourite?
    - vs_random: is the model better than random selection?
    """
```

**P&L simulation:**
```python
def simulate_betting(self, predictions_df: pd.DataFrame, 
                     config: dict) -> pd.DataFrame:
    """Simulate actual betting performance over the test period.
    
    For each predicted race:
    1. Identify overlays (edge > min_threshold)
    2. Calculate Kelly stakes based on predicted probabilities and actual BFSP
    3. Settle bets using actual outcomes
    4. Track running P&L, bankroll, ROI
    
    Apply Betfair commission (5% on net winnings per market).
    
    Returns daily summary DataFrame with:
    - date, n_bets, n_winners, total_staked, total_returns
    - daily_pl, cumulative_pl, bankroll, roi_pct
    """
```

**Calibration plot:**
```python
def plot_calibration(self, predictions_df: pd.DataFrame, output_path: str):
    """Generate calibration plot: predicted probability vs actual win rate.
    
    Bin predictions into 20 equal-frequency buckets.
    For each bucket, plot mean predicted probability (x) vs actual win rate (y).
    Perfect calibration = 45-degree line.
    
    Also plot the market's calibration (BFSP implied prob vs actual) for comparison.
    
    Save as PNG to output_path.
    """
```

**Monthly P&L chart:**
```python
def plot_monthly_pnl(self, daily_df: pd.DataFrame, output_path: str):
    """Generate monthly P&L bar chart and cumulative P&L line chart.
    Save as PNG to output_path.
    """
```

---

## CRITICAL IMPLEMENTATION NOTES

### No lookahead — the cardinal sin
At every point in the pipeline, verify that NO future data leaks into predictions. The PreRaceBuilder must filter history to `date < race_date`. The walk-forward trainer must never train on data that overlaps the validation window. The evaluator must use actual BFSP, not post-race adjusted prices.

### Race-level probability normalisation
After predicting raw P(win) for each runner, probabilities MUST sum to 1.0 within each race:
```python
race_probs = raw_probs / raw_probs.sum()  # per race group
```
This is because exactly one horse wins each race — the model must respect this constraint.

### Commission modelling
Betfair charges commission on net winnings per market (typically 5% for standard accounts, can be reduced with discount rate). The P&L formula is:
```
If win:  PL = stake * (BFSP - 1) * (1 - commission_rate) - but only if net positive on market
If lose: PL = -stake
```
For simplicity, approximate as: `PL_win = stake * (BFSP - 1) * 0.95`

### Handling non-runners
Horses can be withdrawn after declarations. The model should gracefully handle field size changes. If a horse in our predictions is withdrawn, remove it and re-normalise remaining probabilities.

### Model persistence
Save trained models using joblib/pickle. Save alongside:
- Feature list used
- Training date range
- Validation metrics
- Optimal lambda
- Hyperparameters
- Feature importance rankings

### Integration with existing code
This module should work with:
- `src/data/custom_metrics.py` (CustomMetricsEngine) for computing historical metrics
- `src/data/matcher.py` (DataMatcher) for loading matched data from SQLite
- `src/betting/executor.py` (BetfairExecutor) for live bet placement
- `src/reporting/email_report.py` (DailyReporter) for results reporting

The main entry point for daily operation:
```python
# Morning: predict today's races
builder = PreRaceBuilder(history_df)
todays_features = builder.build_features(declared_runners, today)
model = FundamentalModel.load('data/models/latest/')
predictions = model.predict(todays_features)
blended = BenterBlender().blend(predictions, lambda_=0.80)
bet_card = OverlayDetector(config).generate_bet_card(blended)

# Evening: evaluate results
evaluator = ModelEvaluator()
results = evaluator.evaluate(todays_predictions_with_outcomes)
```

### Testing requirements
Write tests that verify:
- PreRaceBuilder produces correct feature vectors for a known horse
- Probability normalisation sums to 1.0 per race
- Benter blending with lambda=0 gives pure model, lambda=1 gives pure market
- Kelly stakes are 0 when there's no edge
- Walk-forward folds have no temporal overlap
- Simulated P&L matches hand-calculated examples
- Model can handle races with all debut runners (no history)
- Feature importance runs without error
