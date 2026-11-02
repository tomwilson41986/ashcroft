# Football: QA review, research ideas and the roadmap to profit (2 Nov 2026)

**Bottom line: the football work is sound, and it has found a real, repeatable signal: the move from the early price
to the close is forecastable. It has not found a profit. Every test so far loses once we pay to cross Betfair's spread
at both ends. The road to profit is therefore not a better result model. It is (1) a stronger move forecast from
stacking models and new data we already hold (xG, injuries, the recorded books), (2) cheaper execution (resting orders
instead of crossing the spread, and the team-sheet window), and (3) checking capacity, how much money the books actually
take. All three are blocked on one thing: the paper-models runner, which has never run.**

Every figure below names its source file. "Development" means seasons the model may be tuned on. "Holdout" means data
read once against a rule written down beforehand.

## 1. Where we are (results to date)

| Test | Data | Result | Source |
|---|---|---|---|
| Dixon-Coles team model vs the open | 92,766 matches, 2012-2026 | Behind the open every season; pooled weight negative | `reports/football_model.md` s.3 |
| Paper trader (pooled model, 5% edge) | Live, since Oct 2026 | 15 bets, 0 settled (settlement fault, see 2.2) | football-model.yml settle logs |
| Exchange gap rule, pre-registered | Holdout Aug-Nov 2026 | Failed: -1.40% a tick worse (39 bets) | `reports/open_close/model_football_exchange.md` |
| Team features vs the result's price | 68,535 matches, development | Nothing adds; best +0.12 per thousand (-0.01 to +0.26) | `reports/football_team_features.md` |
| Team features vs the early-to-close move | 60,807 Pinnacle matches, development | R2 4.6% (floor -0.07%); top decile catches 4.0% before costs | same |
| Team-feature move rule on Betfair, pre-registered | Holdout Jul 2025-Nov 2026 | Failed: -1.10% a tick worse (-1.64 to -0.54), 1,827 bets; +3.27% at the mid, 64% right way | `reports/football_runs/team_move_holdout.json` |
| Player ratings, line-up surprise | 46,869 matches, development | 0.7% of the move; adds nothing to the close | `reports/football_player_ratings.md` |
| Stacked move model (QA experiment, 2.4) | 60,807 Pinnacle matches, development | R2 5.2% vs 4.1% (linear); top decile catches 4.9% vs 4.0% | `reports/football_runs/move_model_comparison_dev.json` |

**The pattern:** results are priced efficiently. The path of the price is not, but its predictable part is about the
size of the cost of trading it.

## 2. QA review

### 2.1 Code review (two independent reviewers, plus my own checks)

Two reviewers read the new code without being told what I believed about it. One covered the features, the
evaluation, the player ratings and the Betfair trade; the other covered the collectors, the backfill, the paper-trial
scorer, settlement and the workflows. I verified each material finding on the data or with a test that fails on the
old code before fixing it. Fixes: commit aa6f196. Findings and status: section 2.5.

### 2.2 Operational faults (P0)

1. **The paper-models runner has never run** (0 runs of `paper-models.yml`). Without it there is no Betfair football
   book record, no paper trial, no line-up timestamps and no greyhound or football paper trading. Every forward test in
   this roadmap waits on it.
2. **The football paper ledger does not settle.** There have been 12-15 matched paper bets since at least 25 Oct and 0
   settled 8 days later (football-model.yml logs, runs 40166604556, 41009678251, 41395334749). Bets older than 7 days
   are never re-settled (`football/live.py` settle loops the last `days` only). The cause of the first failure needs
   the ledger in S3: see 2.5.
3. **Research sessions cannot read S3** (the session's AWS keys are refused). Today that cost a 2,700-request re-fetch
   of data already in S3. Read-only keys in the cloud environment would remove it.

### 2.3 Methodological checks

| Check | Finding | Action |
|---|---|---|
| Holdouts | football-data's 2025/26-on data has now been read twice (exchange rule, team-move rule). It is no longer clean for selecting models | Further selection on data to 2024/25 only. The final test of anything new is forward: the recorder and paper |
| Time of the "early" price | football-data's open is an untimed snapshot from Friday or Tuesday afternoon. Features must be as of that moment, not the match day | See 2.5: refits aligned to the collection day |
| Cost model | "A tick worse at each end on the fair price" is a stand-in. The true cost is the recorded best back and lay, with sizes | The recorder answers it |
| Training target ages | Pinnacle's early prices are stale from 23 Jul 2025, so the move forecaster trains on data to 2024/25 only | Retrain on Betfair moves from the recorder |
| Multiple testing | 18 width x threshold cells examined in development; the one positive cell was registered as a secondary read and failed | Keep registering the chooser, not the best cell |
| Data quality | API-Football: 34 of 49,206 matches have a repeated player id (source errors); 715 substitutions unresolved; 207 players without an id | Negligible; documented |

### 2.4 The model comparison (QA experiment, development only)

Early-to-close home move, Pinnacle to Pinnacle, 2016/17-2024/25, walk-forward by season
(`research/queries/football_move_model_comparison.py`, `reports/football_runs/move_model_comparison_dev.json`):

| Model | R2 vs mean | Top-decile move caught | Right way |
|---|---|---|---|
| Linear on team features (the current model) | 4.08% | 4.03% | 65.5% |
| LightGBM on team features | 3.19% | 3.48% | 61.8% |
| LightGBM on team features + market state + league | 4.58% | 4.48% | 64.6% |
| **Stack: mean of linear and LightGBM (market)** | **5.24%** | **4.91%** | **66.4%** |
| LightGBM on a permuted target (floor) | -0.81% | 0.01% | 49.4% |

A simple two-model stack beats either model alone by a clear margin. That is the strongest evidence today that model
work (ensembles, more market-state features) can still raise the captured move.

### 2.5 Reviewer findings and fixes

**Fixed, results affected (all re-run: section 2.6)**

| # | Finding | Evidence | Fix |
|---|---|---|---|
| 1 | **Look-ahead.** Market ratings and Dixon-Coles prices were refitted in 7-day blocks anchored to the first match day, so a block starting on a Sunday let Saturday's closes and results into Sunday and Monday matches whose early price was taken on Friday. The windows were lagged a day, not to the collection | Reviewer, on synthetic data; a new test (a Saturday rewrite must not move Sunday's features) fails on the old code | Every fit on the matches before the match's collection day (Friday for Fri-Mon, Tuesday for Tue-Thu); windows lagged to it |
| 2 | **Own goals credited to the wrong side** in the goals plus-minus | Crediting the listed side reproduces the final score in 562 of 562 own-goal matches; my convention in 9 | Credit the listed side |
| 3 | **Stoppage-time goals lost** after a late substitution (minutes capped at 90) | Stint goals reconciled with the final score in 95.8% of 3,000 matches after fix 2; 124 of the 126 misses were this | Real minutes; now 99.9% reconcile |
| 4 | **The season-to-date window included last season's final match** | Reviewer, by running; a new test fails on the old code | Shift within the season |
| 5 | **The model-market disagreement shared an error term with the move** it was tested against | Reviewer, from reading (small) | Built from probabilities, not the implied-rate grid |
| 6 | **Optimistic costing:** "a tick worse than the fair price" can sit above the real back in a wide book, and there is no recorded lay | Reviewer, from the arithmetic | Both sides now also costed at the recorded quotes (a back at the quoted back, a lay a tick over it); the gate read again at those prices |
| 7 | **Player ratings' monthly boundary:** a fit made on the 1st holds the 30th-31st, after a Friday-30th collection | Reviewer | Each match takes the fit of its collection month |

**Fixed, operational**

| # | Finding | Fix |
|---|---|---|
| 8 | One per-minute rate-limit reply stopped the line-up collector for the day (reviewer, simulated) | Wait and carry on; stop after 60 failed steps instead |
| 9 | Backfill marked a match done before its line-ups or events arrived | Done only when both line-ups and events are present; the current season's last 14 days are fetched again; done ids saved in a `finally` |
| 10 | Paper-trial scorer: women's and youth markets pair with the senior fixture by name, so the wrong market could be scored (reviewer, by running) | Excluded by name, kick-off within 3 h, one market a match and type |
| 11 | Scorer: no time check on the close, so a missing day's record could turn the evening-before book into the close | The close must come after the fill and within 30 minutes of the off |
| 12 | Scorer: training on months before the match let a Monday match be fitted with Saturday's closes; season stage moved with the scoring date | Train only on matches before the decision; season stage as known at the decision |
| 13 | The recorder never has a book at the collection moment for Sun/Mon/Wed/Thu matches | `paper-models.yml`: Friday 16:55 and Tuesday 12:55 records of those days' markets (unmodified recorder) |
| 14 | Settlement looked at the last 7 days only, and the settle step was skipped whenever an earlier step failed | Every day still holding an unsettled bet is re-settled; old unsettled bets are explained in the summary (what gold holds for that day and those teams); the step runs after failures |
| 15 | The line-up tables step ran even when the runner guard failed; a collector crash was silent | Gated on the guard; a warning on a non-zero exit |

**Not fixed, noted**

- The paper ledger's first-cause failure (why 15 bets never matched a result) needs the S3 ledger. The new settle
  summary (`unsettled_old`) will show it on the next run.
- The exchange study's costing (`research/open_close/football_exchange.py`, the earlier pre-registered rule) uses the
  same tick-off-fair method as finding 6. Its rule failed anyway; the paper trial scores at recorded quotes.
- No multiple-comparison adjustment across families. Every family's result is reported, none selected.
- Market ratings have a free level (attack + c, defence + c), so `own_r_d5` moves slightly with the pool's make-up.
- The line-up tables build reads every raw poll each night and will slow as files accumulate (incremental build later).

## 3. Feature engineering: what to build next

Ordered by expected value for the move (the target with a signal) and by cost.

| # | Family | Features | Data | Status |
|---|---|---|---|---|
| F1 | **Market microstructure** | Spread, depth imbalance (back size minus lay size over the sum), matched money, price velocity and acceleration, time to kick-off, the path since the first book | Betfair recorder (2-minute books with sizes) | Data waits on the runner. The horse model's best predictors were of this kind |
| F2 | **Cross-book gaps** | Betfair fair price vs the market average, the maximum, Bet365; Pinnacle via Parse; dispersion across books; steam (several books moving together) | football-data Avg/Max/B365 (held); Parse Pinnacle (subscribed) | Held. Combine with the team features (stacking) |
| F3 | **xG form** | xG for and against windows, xG difference above the market's expectation, finishing luck (goals minus xG) and its regression | API-Football statistics (held in the backfill): 11 top flights and the Championship from 2023/24, more from 2025/26 | **New find: held, not yet used** |
| F4 | **Availability** | Expected eleven from the injury list (not last-5 starters), missing players' rating sum by position, goalkeeper change, number of changes from the last match | API-Football injuries (collector), line-ups, player ratings | Collector built; waits on the runner |
| F5 | **Schedule** | Rest days difference, matches in 14 and 30 days, a European or cup match within 3 days either side, travel | API-Football fixtures (all competitions; held nightly once merged) | Cheap |
| F6 | **Table state** | Points to the title, Europe and safety with matches left; dead rubbers; last-day effects | Gold table | Cheap |
| F7 | **Manager** | Matches since a manager change; new-manager bounce | API-Football coaches endpoint | Cheap, needs a small collector |
| F8 | **Cross-market consistency** | Goal rates implied by match odds vs over/under vs Asian handicap disagreeing | football-data AH and O/U (held) | Cheap |
| F9 | **Player-level value gaps** | Market rating vs goals rating of a team's eleven (the market under- or over-rating a side's players) | Player ratings (built) | Cheap |
| F10 | **Referee** | Cards and penalty tendencies (for the cards and totals markets) | football-data referee (England), API-Football | For second-tier markets |

## 4. Algorithms and models to try

**For the move (the trade):**

1. **Stacking with out-of-fold predictions** (2.4 says it works). Level 1: linear team model, LightGBM with market state,
   the exchange study's gap model (`gbm_nopin`), xG model, player-surprise model. Level 2: non-negative least squares or
   a monotone LightGBM. Folds: by season, never shuffled.
2. **Direction classifiers with calibrated probabilities**, plus **meta-labelling**: a second model that decides
   whether the expected move clears the cost. It trades fewer, better signals.
3. **Quantile regression** of the move, to size by the chance of clearing the spread, not the mean.
4. **A closing-price model** like the horse race book (`model/race_book.py` `ClosingModel`): a conditional logit of the
   closing book on the current book, the sharp book and our price, interacted with depth. It worked for horses.
5. **Execution models** (the cost is the problem): fill probability for a resting order one tick inside the spread
   (survival model on the recorded queue), adverse selection after fills, and the best time in the week to enter.

**For the team and player models (inputs to the stack):**

6. Dynamic ratings: score-driven or Kalman Dixon-Coles with time-varying strength (Koopman and Lit), pi-ratings, Elo.
7. Bayesian hierarchical goals model (partial pooling across leagues and seasons) with xG as an input.
8. Player ratings with priors: plus-minus regularised towards a box-score prior (minutes, ratings, position), and
   position-group strength of the eleven.
9. Decorrelation training (Hubacek and Sir): train the model to disagree with the market where the market is wrong,
   not to copy it.

**Ensembles beyond stacking:** bag the move models across seasons and leagues; blend models trained on different
targets (Pinnacle move, Betfair move, result); and keep the Benter stage-2 as the referee of whether anything adds to
the market.

## 5. Roadmap to profit

Each phase has a gate. Nothing moves to money without passing the gate in the phase before.

| Phase | When | Work | Gate to the next |
|---|---|---|---|
| **0. Unblock** | This week | Register the paper-models runner; fix settlement (re-settle every unsettled day; log why a bet cannot settle); read-only S3 keys for research sessions; apply the QA fixes (2.5); recorder snapshots at the Friday and Tuesday collection for Sun/Mon/Wed/Thu matches | The recorder, collector and paper traders write files every day |
| **1. Build the stack** | Weeks 1-4 | F2, F3, F5, F6, F8 into the team features; the stacked move model (4.1), the meta-label (4.2); all on development data to 2024/25 with season folds | Development: top-decile move caught at least 1.5x the measured round-trip cost at Betfair prices, interval above it |
| **2. Measure costs and capacity** | Weeks 2-6, from the recorder | Real spreads, sizes and queue fills on the recorded books; the resting-order simulator (4.5); the team-sheet window using the collector's first-seen times against the 2-minute books | Known cost per trade and money available per signal. If capacity is under about GBP 50 a signal, football cannot carry meaningful profit: stop here |
| **3. Pre-register and paper-trade** | From week 4 | Write the stacked rule (and a resting-order variant) into pre-registration files before forward data; run on paper on the recorder's books with sizes | The pre-registered gate: at least 300 trades and 90 days, 90% lower bound of return at taken prices above zero, CLV above zero, both halves positive |
| **4. Second-tier markets** | Months 2-4, in parallel | Asian handicap and alternative totals (same engine); corners and cards (F10), where a stats model is more plausible, after the census shows liquidity; Betfair Historic BASIC files for 2016-on paths (owner's action) | Same gate per market |
| **5. Live, small** | After a paper pass | Live key, minimum stakes, match and day caps, the STOP switch; written permission from football-data.co.uk first (its terms: private use only) | Live CLV inside the paper interval for a month before any increase |

**Kill criteria.** If by the end of Phase 3 no rule passes at takeable prices, football stops as a trading line. The
recorder and data collection continue at no marginal cost as research inputs. The decision is then whether to redirect
the effort to the markets already trading.

**Economics to check early (Phase 2).** At the development rate of about 1,800 signals a year in takeable books
(1,827 in 16 holdout months), a GBP 10 stake and a 2% net edge make about GBP 360 a year. Profit at a level that matters
needs stakes in the hundreds a signal, which only the recorder's sizes can show the books will take. This is the first
number to establish, because it decides whether the rest is worth doing.

## 6. Owner's actions

1. Register the paper-models self-hosted runner on the UK server (label `paper-models`, not the live trader's runner).
2. Add read-only AWS keys to the cloud environment for research sessions (S3 `ashcroft` bucket, read only).
3. Request the free Betfair Historic Data BASIC plan for Soccer.
4. Write to football-data.co.uk for permission before any money is staked on models trained on its data (I can draft
   it; I will not send it).
