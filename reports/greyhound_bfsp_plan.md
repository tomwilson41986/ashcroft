# The greyhound BFSP plan: features, statistics and modelling reviewed

racing² - greyhound research, 10 Nov 2026. Greyhounds only; the horse model and its jobs are not touched.

## Status (10 Nov 2026, every step run)

Every step of the plan has been run and recorded (reports/greyhound_model.md sections 28 to 36). On the same test
(1 June to 5 November 2026, 106,719 runners) the RMSE of log(BSP):

| Recipe | RMSE log(BSP) | With the per-track stretch | Source |
|---|---|---|---|
| The live recipe as served | 0.4017 | - | section 29 |
| share08_best (share objective, figs and the metric tables, 3 seeds) | 0.3510 | 0.3448 | batch 18, run 42853398485 |
| stack3_all3 (+ form, draw, bend tables; trees stacked with 3 neural set models) | **0.3396** | **0.3366** | same |

0.3366 is 16% less error than the live recipe; the winner's log-loss is better too (0.43732 against 0.43777, z -3.5).
- Step 5: the better price carries into the first-book forecast (0.2852 to 0.2648) and to 3 minutes out, not to the
  last minute (section 33).
- Steps 2.2-2.6: the form, draw and bend tables add 0.0013 at three seeds (z -5.5; section 34-36). Fitted
  constants replace guessed ones: form's half-life is 21 days, not 90; the trap bias the market misses is static.
- Step 4: the neural set model alone matches the trees; stacked with them it is the largest step since the share
  objective (-0.0090 at three seeds without the new tables, -0.0114 with them).
- Still open: the debutants (RMSE about 0.70, unchanged), the compression (outsiders still -0.34 in log), the trend
  filter's noise on its grid's edge, the market's memory against our own forecasts (needs out-of-fold forecasts).

Not adopted. Two forward tests on days from 11 November, both run from 25 November: bfsp_forward.json (share08_best)
and bfsp_forward_stack.json (stack3_all3), each against the live recipe with the bar fixed before its days.
Adoption into the live pricing (greyhound/live.py) is the owner's decision.

## Context

The owner: "Fundamental is to improve our model to predict BFSP / Closing Prices." The ask is a step back from
rule-testing to the forecast itself: review the features, the statistics behind them, the modelling, borrow from
other sports, use Bayesian methods where they help, and show the feature set and its importance.

Two code reviews were run for this plan (features; models and results). Their findings are below, then the plan.
Every number is quoted from the repo's reports with its source. The plan was approved by the owner on 10 Nov 2026; its results are recorded in reports/greyhound_model.md as they come.

## Where the forecast stands

RMSE of log(BSP), lower is better. The forecast is three models at three moments.

| Moment | Market alone (the mid) | Our best | Source |
|---|---|---|---|
| Card time (before any price) | no price exists; 0.754 knowing only the race's average | 0.4053 | fast-loop batch 8, run 41827648070; 0.754 from reports/greyhound_model.md section 14 |
| First traded book | 0.635 | 0.287 with our best card-time price fed in | section 26.2-26.3, runs 41668293745, 41671443774 |
| 5 minutes out | 0.709 | 0.321 | same |
| 1 minute out | 0.158 | 0.144 | same |
| Last 60 s (stream) | 0.1235 | 0.1218 | close lab run 42726462862, 9 Nov |

What that says:
- The card-time model matters most early. Fed into the first-book forecast it takes the error from 0.320 to 0.287.
- Card-time gains have been small and slowing: each new block moved 0.4130 by 0.001 to 0.003.
- No naive baseline other than 0.754 has ever been scored, so we do not know how much of 0.405 is beatable.

## The feature set we have

The live card-time recipe reads about 425 features (counted from the code; no report states the total).

| Family | Features | What it measures | File |
|---|---|---|---|
| Base engine | 133 | Speed, early-pace and finishing figures over the last 1, 3 and 6 runs; trend; track, distance and trap record; grade and age; each ranked against the field; a pace map | greyhound/metrics.py |
| Parity | 132 | 15 per-run measures (normalised finish, beaten lengths, early position, market view, result against the SP and BSP, race strength) over 7 windows; race shape | greyhound/parity.py |
| Record by condition | 42 | Win and place rates by grade, trap, trip, trainer; trainer form; head to head; the field's record | greyhound/hrb.py |
| Library | 26 | Elo, a Kalman rating, form lines, first-bend crossings, class-adjusted speed | greyhound/lib.py |
| Trainer intent | 25 | Class and distance moves, trial returns, each with the trainer's record against the SP | greyhound/intent.py |
| Expected time | 21 | An empirical-Bayes estimate of today's time, its spread, the chance of being fastest | greyhound/etime.py |
| Track-normalised times | 20 | Figures against 28, 91 and 365-day rolling standards, with a shrunk meeting variant | greyhound/times.py |
| Ability with uncertainty | 13 | The Kalman rating's spread; a dog-by-track level | greyhound/ability.py |
| Weight forecast | 7 | A predicted weigh-in from past weights | greyhound/blocks/wpred.py |
| Race simulator | 6 | 300 simulated runs: chance to lead, win, be checked | greyhound/blocks/sim.py |

Not live: the metrics database (greyhound/pit, 82 columns: improves BFSP by 0.0029, forward test from 7 Dec).

### Feature importance: what exists, and the gap

Only LightGBM gain share has ever been recorded, and never for the live log-BSP model. The latest numbered list is
the win model of 7 Oct (reports/greyhound_lib_ci_1007.json, all+lib softmax):

| Feature | Share of gain | What it is |
|---|---|---|
| lb_cv_gsr_ewm_z | 9.4% | Class-adjusted speed figure, recency-weighted, against the field |
| gp_rs_vs_past | 4.1% | Today's race strength against the dog's past races |
| gp_mkt_w3 | 3.6% | The market's view of the dog over its last 3 runs |
| lb_kr_z | 1.6% | Kalman rating against the field |
| gp_mkt_w5, gp_mkt_l1 | 1.5%, 1.4% | The market's view, other windows |
| grade_change | 1.4% | Up or down in class |
| weight_vs_mean | 1.3% | Weight against the dog's usual (not card-safe) |
| trap_bias, gsr_trend, days_since | 1.1% each | Trap bias, form trend, days off |

For the live log-BSP model, section 14 names the same leaders plus the dog's own BSP history (gp_bsp_mkt_w3), with
no numbers. Gain share is a weak measure: it is read off the last fold only, splits credit arbitrarily among
correlated features, and says nothing per family. **Step 1 of the plan produces the real table.**

## Is the modelling best in the world? No

What is good: a large, careful feature set; walk-forward fitting; an out-of-sample bootstrap by day; recent
pre-registration.

What falls short of the best practice in this field:

| Weakness | Evidence |
|---|---|
| Each runner's price is fitted alone (squared error on log BSP), then patched | The forecast is compressed: a 1.59 favourite priced 2.27, a 14.0 outsider 9.25 (batch 8). Two patches follow (rescale to the book, a per-track stretch), and the stretch makes the win chance worse |
| The fit is starved | The 20,000-round fit was still improving at its cap, best round 19,966 (section 15) |
| No ensemble of any kind | One model, one seed |
| Training throws away history | Fitted from 2024 only; ten years are held |
| One test window, reused about 52 times | Audit B1: every batch read 1 Jun onward, with no correction |
| Speed figures are heuristics | Standard time is a running mean of winners; the meeting variant a median of winners; early pace gets no going correction; the run-in subtracts a raw sectional from a corrected total |
| The simulator's settings are guesses | Pace weights 0.75 / 0.25, noise 1.0, 2 lengths lost: none fitted |
| Shrinkage constants are guesses | Trainer k = 30, sire k = 50, trap +5 |
| No forecast of uncertainty at card time | A point price only |
| Importance never measured properly | Above |
| Known bad inputs are live | `hb_days_since_season` reads a field where 58% of values postdate the race; the trainer field is the current trainer on every past run |

## Chapter: every feature reviewed, with recommendations

How this was done: I read the code that builds each feature. Base, parity, library, times, ability, expected time,
simulator and intent were read line by line; record-by-condition in part; the weight forecast from its description
only. Every verdict is from reading the code. **None has been measured yet**: each recommendation is an experiment
for the fast loop, and the review may be wrong where the data disagrees.

Verdicts: **Sound** (correct and well made), **Weak** (correct but statistically poor), **Wrong** (a calculation
flaw as I read it).

### The ten things that matter most

Calculation flaws to fix first:

1. **The track-normalised times double-count a track's drift** (`greyhound/times.py`). The meeting variant is
   measured against the all-history standard, then applied to figures built on a 28, 91 or 365-day standard. If a
   track has sped up, the rolling standard already holds that, and the variant removes it a second time. A winner
   running exactly to the recent standard should score 0 and does not. This is the block that gave the largest gain
   so far (-0.00082), so a correct version may give more.
2. **Early pace has no going or meeting correction**, and its standard is the all-history mean of every runner.
   A slow night reads as slow dogs. Early pace feeds about 40 features and the simulator.
3. **The run-in is charged the whole going allowance.** It is the corrected total minus a raw sectional.
4. **Grade moves compare grade numbers across tracks and grade types.** `grade_change`, the class flags in trainer
   intent and `hb_grade_vs_*` treat an A3 at one track as an A3 at another, and an A grade as comparable with a D
   or S grade.
5. **Two fields are not what they claim:** the trainer on every past run is the dog's current trainer (so
   `ti_new_trainer` can never fire), and `hb_days_since_season` reads a season date that postdates the race 58% of
   the time.

Statistical upgrades that apply across many features:

6. **No window allows for how many runs are behind it.** A dog with one run has the same "mean of 6" as a dog with
   six. Every windowed mean should be shrunk by its count, and the count offered beside it.
7. **Seven hand-picked windows a measure, none fitted.** Parity alone makes 105 near-duplicate columns. Replace
   with one decayed mean a measure, its half-life in days fitted by how well it predicts the next run.
8. **"Best of the last 6" grows with the number of runs,** so it favours dogs that have simply run more.
9. **Race-relative z-scores divide by the spread of six dogs.** A race where everyone is within half a length
   gets the same z-scores as one spread over five lengths. Use lengths against the rest of the field, scaled by a
   population spread, and a share (a softmax with a fitted temperature).
10. **Win rates carry little in graded racing.** Dogs are graded until they win about one race in six, so trainer,
    sire, track and trap win rates mostly measure grading, not ability. Use ratings and results against the price.

### Base engine (greyhound/metrics.py)

| Features | How it is calculated now | Verdict | Recommendation |
|---|---|---|---|
| Speed figure `gsr` | Standard = all-history mean of winners' times at the track and distance. Variant = median deviation of that meeting's winners. Figure in lengths at 0.08 s | Weak | The standard depends on which grades raced; the variant is a median of about 12 winners, unshrunk, and reads a high-class night as fast going. Replace with the joint model (plan step 2.1), which uses every runner, not only winners |
| Early pace `esr` | All-history mean sectional of every runner, no variant | Wrong | Give it its own shrunk meeting variant and a rolling standard |
| Finishing `fsr` | Corrected total minus raw sectional, less the variant | Wrong | Split the going allowance between sectional and run-in by their share of the trip |
| Trouble credit `gsr_trouble_adj` | Adds a flat 1.5 lengths a bump | Weak | Use the learnt cost of each comment code (already built: `greyhound/pit/trouble.py`) |
| Means over last 1, 3, 6 runs (51 columns) | Plain rolling mean, at least 1 run | Weak | Shrink by count toward the dog's rating; add the count (points 6, 7) |
| Maximum over last 3, 6 | Raw maximum | Weak | Replace with a high quantile adjusted for the number of runs (point 8) |
| Decayed means `*_ewm` | Half-life 3 runs, whatever the dates | Weak | Decay in days, half-life fitted for each measure |
| `gsr_trend` | Last run minus the mean of 6 that includes it | Weak | A fitted slope, or the trend state of the rating (step 2.2) |
| `gsr_std6` | Raw spread of up to 6 runs | Weak | Shrink toward the population's within-dog spread, as `etime` already does |
| `sp_p_mean3` (the base set's top feature) | Mean of the last 3 SP chances | Weak | Average on the log-odds scale, scaled by field size, and adjust for the strength of each field: a market-implied rating (below) |
| `dog_ae6` | Mean of (won minus SP chance) over 6 | Weak | Very noisy. Use a standardised form: the sum of (won minus chance) over the root of its variance |
| Track, course-and-distance and trap records | Raw career means from 1 run upward | Weak | Shrunk versions exist in `ability.py`; drop the raw ones. Express each preference in standard errors, as Benter did for distance |
| `grade_change`, `dist_change` | Difference from the last run | Wrong (grade), Sound (distance) | Measure class in lengths: today's grade par less the last race's |
| `grade_par` | All-history mean of winners' figures by track and class | Weak | A rolling year, shrunk toward the track's trend across grades |
| `trap_bias` | Wins over fair share, all history, 5 prior wins | Weak | Wins against the SP's expectation, recent, and moving by meeting (step 2.3) |
| `trainer_win`, `trainer_ae`, `sire_win` | All-history rates, fixed priors of 30, 20, 50 | Weak | Recent-weighted results against the price; fitted priors; progeny ratings in place of sire win rate (built in `greyhound/pit/breeding.py`) |
| Rank, gap and z of 13 ratings | Within-race mean and spread | Weak | Point 9 |
| Pace map: `early_rank`, `squeeze`, `faster_inside` | Counts of faster neighbours, yes or no | Weak | The chance each neighbour reaches the bend first, from the gap in early pace and its noise (step 2.4) |
| `path_vs_trap`, `path_conflict` | Fixed linear mapping of running line to trap | Weak | Learn it by track |
| Freshness, age, sex, today's trap, field, distance | Dates and card fields | Sound | Age gets a population curve (step 2.2) |

### Parity (greyhound/parity.py)

| Features | How it is calculated now | Verdict | Recommendation |
|---|---|---|---|
| Normalised finish `nfpz` | The finishing rank centred and divided by its spread for the field size | Sound | Checked: it is the z-score of the rank, over 3 |
| `wax`, `lbw`, `lbsc` | Won minus fair share; lengths behind the winner; against the field's mean | Sound | None |
| Early position `ep`, `lead`, `gain` | Rank of the sectional in the race | Sound | These are also the labels for the fitted bend model |
| `shape` | Finish against the average for that early position and pace bucket at the track | Sound | The pace buckets are fixed cuts; a smooth fit is a small gain |
| Market view `mkt`, `bsp_mkt` | Log of (field size x chance) | Sound scale, Weak use | Not adjusted for the opposition. See the market-implied rating |
| `ae`, `bsp_ae`, `nres` | Won minus chance; finish against market rank | Weak | Standardise by the variance; use the expected finishing position from the prices, not the market rank |
| Race strength `rs`, `gp_rs_vs_past` (second top feature) | Mean of the other runners' decayed speed figure | Weak | Build it on the better rating, and weight the strongest rivals more (a log-sum, not a mean) |
| 7 windows x 15 measures | Career, last, mean 3 and 5, weighted 3, 5, 10 | Weak | Point 7: one fitted decay a measure. Cuts about 75 columns |
| `gp_wiv` | Career wins over fair wins | Weak | Point 10 |
| Projected pace, `gp_lead_share` | Early position filled with 0.5 when unknown; lead rates normalised | Weak | Ignores the trap and the run to the bend. Step 2.4 |
| `gp_lead_value`, `gp_pos_value` | Leaders' win rate at the track | Sound | Use a rolling window |
| Trainer's last 30 days | Fixed priors | Weak | Fitted priors; and it reads the current-trainer field |

### Library (greyhound/lib.py)

| Features | How it is calculated now | Verdict | Recommendation |
|---|---|---|---|
| Elo | Every dog starts level; fixed K of 24; finish rank only; no time decay | Weak | It does not reach the top 30. Start from the grade par, shrink K with runs, and score margins. Better: the market-implied rating below |
| Kalman rating `lb_kr` (a top feature) | Noise, drift and start variance set by hand; reads the raw figure; a troubled run counts as much as a clean one | Weak | Fit the constants (done in `pit/ability.py`); read the class-adjusted figure; trust a troubled run or a trial less; heavy-tailed noise; add trend and age (step 2.2) |
| Class-adjusted figure `lb_cv_gsr_*` (the top feature, 9.4%) | The night's winners measured against their own grade's usual time, then the median | Sound idea, Weak method | This is a rough form of the joint model. Its rank says step 2.1 is the right bet |
| Form lines | Rivals' later win rate | Weak | Point 10. Use how the rivals' later figures compared with their ratings |
| First-bend crossings | Counts of crossing paths, weighted by closeness in early pace | Weak | Step 2.4 |
| Since a break | Days and runs since 42 days off | Sound | None |

### Times, ability, expected time

| Features | How it is calculated now | Verdict | Recommendation |
|---|---|---|---|
| `tn_` figures on rolling standards | Rolling median of winners; variant from the all-history standard | Wrong | Flaw 1. Measure the variant against the same standard the figure uses |
| The shrunk meeting variant | Night's mean shrunk by n / (n + k), k from earlier meetings | Sound | The right method. One k for every track; fit it by track |
| `ab_p_best` | Chance of the best rating, from rating uncertainty alone | Wrong | It leaves out the noise of the race, so it is overconfident. The database's version fits that noise (`kf_gsr_pbest`) |
| Dog and dog-by-track levels | Career means with fixed priors | Weak | No decay: it is the career, not the present. Express the track preference in standard errors |
| Expected time `et_eb` | Career mean shrunk toward today's grade at today's track, prior of 3 runs | Sound idea, Weak | Add decay, today's trap and the distance moved |
| `et_p_fastest` | Simulated from each dog's shrunk spread | Sound | Dogs are treated as independent; the fitted bend adds the crowding |

### Simulator, intent, record by condition, weight

| Features | How it is calculated now | Verdict | Recommendation |
|---|---|---|---|
| Race simulator `sim_` | Pace weights, noise, the cost of a check and who counts as a neighbour are all set by hand; one random stream for every race | Weak | Fit the settings to the recorded bend order and results, or replace with step 2.4. Give each race its own random seed so training and live agree |
| Trainer intent `ti_` | Flags for class and distance moves; the trainer's wins over the SP's expectation in each, prior of 5 | Weak | Standardise by variance; decay; the class flags have flaw 4; `ti_new_trainer` cannot fire |
| Record by condition `hb_win_*` | Win and place rates by grade, trap, trip, shrunk toward the dog's own rate, prior of 5 | Weak | Point 10: use figures by condition, in standard errors |
| Head to head | Meetings and wins against today's rivals | Sound | Small samples; shrink |
| `hb_days_since_season` | Current-season field | Wrong | Drop |
| Weight forecast `wp_` | A model of the weigh-in from past weights, refitted yearly on earlier years | Sound (not read line by line) | None |

### One new feature this review points to

**A market-implied rating.** The target is the market's price, and the best single input in the base set is the
dog's past SP. But a past SP only says how the dog was rated against that field. Fit a rating for each dog so that
a softmax of the ratings in each past race matches the SP's shares, updated run by run with its own uncertainty.
It is Elo's update with the SP share as the result in place of win or lose, so it learns from every runner in
every race and carries far less noise. It gives what `sp_p_mean3`, the `gp_mkt` windows and race strength each
approximate.

## The plan

Target: RMSE of log(BSP) at card time, with the first-book forecast scored beside it (that is where a better
card-time price pays). Win log loss is reported for every step, as the two-target rule requires.

### Step 0. Measure before changing anything

1. **A proper importance table for the live model.** Three readings on the test window: permutation importance by
   family (shuffle a family within each race, read the rise in RMSE), SHAP by feature (LightGBM `pred_contrib`),
   and drop-one-family refits. Add to `greyhound/fastloop.py` as an `"importance": true` option on an experiment;
   the hook is where `top_features` is built in `run_experiment`.
2. **Baselines and where the error sits.** Score: field size only; the dog's own past BSP only; Betfair's forecast
   price only (held from 6 Oct). Split the 0.405 by price band, track, grade, and runs before (debutants against
   established dogs). This says which families of work can pay.
3. **An honest test.** Replace the single window with monthly walk-forward refits over 2025-26, and lock days
   after each registration as the final judge (the pattern `pit_forward.json` already uses). A change is kept on a
   paired gain 3 standard errors above zero by day, the bar the closing-price lab adopted on 10 Nov.
4. **Fix the calculation flaws the feature chapter found** (its points 1 to 5): the double-counted drift in the
   track-normalised times, early pace and run-in corrections, class moves measured in lengths, and the two bad
   fields. Rerun the trainer test dropping every trainer column (the last one missed `gp_trainer30_*` and
   `hb_*trainer*`). Each fix is built as a corrected column beside the old one and scored against it, so the live
   recipe changes only when a fix is shown to help.

### Step 1. The objective and the fit (cheap, likely the largest single gain)

1. **Fit the race, not the runner.** A new objective, `share`: a softmax over the race fitted to each runner's
   share of the SP book (soft labels). It is `race_objective` in `greyhound/model.py` with the SP share as the
   target instead of the winner, the same loss the first-book forecast already uses (`tradeout.fit_clogit`). The
   forecast is coherent across the race by construction, so the rescale and the stretch should become unnecessary.
2. **A mixed target:** the SP share plus the winner, weighted, for a model that serves both targets.
3. **Robust loss and price bands:** Huber on the log price, and scores by price band, since long shots dominate
   squared error.
4. **All the history, weighted by recency:** fit from 2014 with a half-life chosen on earlier years, instead of
   cutting at 2024.
5. **Ensembles:** five seeds averaged; then a second-stage conditional logit stacking the log-BSP, share and
   winner models on out-of-fold predictions (Benter's structure).
6. **Uncertainty:** quantile fits of the BSP at card time, for the trader's use.

Files: `greyhound/fastloop.py` (OBJECTIVES, the fit block in `run_experiment`, sample weights, seeds),
`greyhound/model.py` (`race_objective`), new specs under `research/greyhound/experiments/`.

### Step 2. Better statistics under the features (the Bayesian work)

Each is a new point-in-time table in `greyhound/pit/`, plugged into the fast loop as a `pit_<table>` block, with the
existing leak test (`tests/test_greyhound_pit.py`: rewriting the last day's results must move no row).

1. **Speed figures from one joint model.** Run time as the sum of a track-and-distance standard, a meeting-day
   variant, a trap effect, the cost of each trouble code and the dog's ability, with every level shrunk by how much
   evidence it has. Today the standard comes from winners and the variant from that night's winners, which is
   circular: a fast time may be a fast track or a good dog. Fitting them together separates the two and puts every
   track on one scale, so a dog changing tracks is priced correctly. Early pace and run-in get the same treatment.
2. **Ability that knows about age and trend.** Extend the Kalman rating (`greyhound/pit/ability.py`) with a trend
   and a population age curve (young dogs improve, old dogs fade, by sex), and heavy-tailed noise so one troubled
   run does not move the rating. `gsr_trend` is already a top feature, which says trend matters.
3. **Trap bias that moves.** A state-space estimate of each track's trap bias by meeting, replacing the static
   shrunk average.
4. **A fitted first bend.** Learn the chance of each bend position from early pace, trap and neighbours, using the
   recorded sectional order as the label; then the finish given the bend. This replaces the simulator's guessed
   settings.
5. **The market's memory.** How far the SP has sat from our own forecast for this dog, trainer and track in the
   past, shrunk by evidence. The target is the market's price, so its habits are signal. Needs out-of-fold
   forecasts, refitted yearly as `blocks/wpred.py` already does.
6. **Shrinkage constants estimated, not guessed,** by fitting each rate's spread (beta-binomial).
7. **The market-implied rating** from the feature chapter: a rating fitted to past SP shares, with uncertainty.
8. **The cross-cutting upgrades from the feature chapter** (its points 6 to 10), each as one experiment:
   count-aware shrinkage of every window; one fitted decay a measure in place of seven windows; a run-count-adjusted
   "best"; race-relative features in lengths and as shares; ratings and results against the price in place of win
   rates.

New feature versions go in as drop-in blocks (`greyhound/blocks/`, computed on the cached matrix) wherever they can,
so an experiment costs a model fit, not a rebuild of the feature cache.

### Step 3. Ideas from other sports (inside steps 1 and 2)

| Idea | From | Where it lands |
|---|---|---|
| Park factors: a venue's effect estimated jointly with the players' | Baseball | The joint speed-figure model (2.1) |
| Aging curves and projections that regress to the mean by reliability | Baseball | Ability with age and trend (2.2) |
| Time-decayed likelihood, the decay chosen by prediction | Football (Dixon-Coles) | Recency weights (1.4) |
| Ratings with their own uncertainty and volatility | Chess, tennis (Glicko) | Already held as the Kalman spread; extended in 2.2 |
| Performance split into phases against expectation | Golf (strokes gained) | The fitted bend and finish-given-bend (2.4) |
| Model plus market in a second stage | Horse racing (Benter) | The stacked ensemble (1.5) |

### Step 4. A neural challenger (last, and only if steps 1-2 leave a gap)

A small set model over the field (each runner's features attend to the others'), trained on the SP share, the
finishing order and the bend order together. About 680,000 races are held. It is scored as one more member of the
stack, not as a replacement.

### Step 5. Carry the gain down the clock

Refit the first-book forecast and the closing-price lab's `model` feature on the improved card-time price, and
score both. This is where the gain is spent.

## Order and effort

| Step | Why this order | Run cost |
|---|---|---|
| 0 | Tells us what to work on, gives a test we can trust, and fixes the five calculation flaws | Fast-loop jobs, minutes each |
| 1 | One function and a few specs; attacks the compression at its source | Fast-loop jobs |
| 2.1, 2.2 | The figures feed 200+ features | A nightly table build each; then fast-loop jobs |
| 2.3-2.6 | Smaller, independent | The same |
| 5 | Needs a better price to exist first | Existing jobs |
| 4 | Largest build, least certain | A longer job; decided after step 2 |

Every run stays inside the fast loop (the owner, 3 Nov: no 2 to 3 hour runs). Each experiment is registered as a
JSON spec before it runs, and every one is counted.

## What I cannot promise

No gain is claimed in advance. Past blocks moved the score by 0.001 to 0.007. The objective change may do more
because it removes a known distortion, but that is a hypothesis until step 1 is scored. Part of the 0.405 is news
we do not hold at card time, and step 0.2 is what sizes it.

## Verification

- **Tests:** each new objective and table gets unit tests; every pit table passes the existing point-in-time test.
  `python -m pytest tests/test_greyhound_pit.py tests/test_greyhound_fastloop*.py`
- **Reproduction first:** the unchanged live recipe must still score 0.4095 (batch 9) under the new harness before
  any change is read.
- **Each change:** a fast-loop run against the live recipe on the walk-forward months, paired by day, both targets,
  by price band; kept only at 3 standard errors; then confirmed on days after its registration.
- **Data audit on every table:** rows in and out, rows dropped and why, duplicates, nulls in key fields, date range.
- **Record:** results into `reports/greyhound_model.md` (new sections) and the ledger, with run ids.
