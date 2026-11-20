# racing² - results, features, code and model review (20 Nov 2026)

The owner, 20 Nov: "Results seemed to have dropped this week. Is it due to a model change? I need a deep and
intensive review of our code, our features, feature engineering, the statistical approach for each type of feature,
and our models: are they best fit, can we build separate models that feed into an ensemble. Ideate how to find
errors, gaps, improvements."

Horse racing only. Nothing here changes the live trader, the served model or any limit. Every proposal is a test to
register in `reports/research_ledger.jsonl` before its run. Sources are ledger tags (in backticks), files and run ids.

## Verdict

1. **The drop is not from the model change.** It began on 9 Nov, on the old model, five days before the first
   change. Since the stack blend went live (15 Nov) the edge per pound is no worse than the old model's on the same
   winter card. What fell is the card (half the money a day) and the jumps edge.
2. **The model is sound but narrow.** Three LightGBM members with the same 1,181 features and the same settings,
   differing only in the target. It is a strong forecaster of the Betfair SP from the market's own memory of each
   horse. Its ceiling is the morning price, which is already closer to the SP than the model in every code.
3. **Seven defects or risks found in the code**, none of them a leak. The two that matter most: an unrated horse's
   official rating of 0 is read as a real number in the older feature families (27% of rows), and nothing checks
   the closing model or keeps a shadow when the served model changes.
4. **The largest gaps** are (a) no market-aware member, (b) a hand-set gate where a learned stack belongs,
   (c) no second kind of learner, (d) no standing monitor that would have answered the owner's question on day one.

## Corrections and results since (20 Nov, later the same day)

- **The morning price against the model (section 2 and M7).** "The morning price is already closer to the SP than
  the model in every code" was the old blend, live, against a later morning price (`model-qa-1113`). For the served
  stack blend against the morning WAP on the development window it is true of chases only: flat 0.354 against the
  morning's 0.372, all-weather 0.359 / 0.395, hurdle 0.399 / 0.448, bumper 0.497 / 0.504, chase 0.313 / 0.306
  (`chase-error-1120`).
- **Chases.** The chase gap is small (0.0074, 0.0015 to 0.0134) and sits in Class 1-3 and non-handicap chases of
  exposed horses at short prices with money in the market (Class 1 +0.074, Class 2 +0.060; BSP under 3 +0.054).
  The model is ahead of the morning price on novices, fallers and thin markets. Two candidates follow: the official
  rating at the weights for every race (the model has it for handicaps only) and a chase-only market rating.
- **Race types to drop: none** (`race-types-1120`). Every code, kind and race type is above zero in research at the
  live 3% bar; live, no type has its interval below zero. Bumpers are level live (-0.97%, 160 horses): watch.
- **The lost fits (C7): cause found and fixed** (`lost-fits-1120`). Every fit read the cached matrix twice over
  (RAM full, 9 GB in swap ten minutes in); a runner with a slow disk died there. The matrix is now read a row group
  at a time and held in single precision where no block reads it: a fit's peak 14.6 GB, no swap, values unchanged.
- **Iteration 131's first run is void** (`iter131-figure-member`, amendment): the figure member predicted 0.0 for
  every runner (its label was built after its inputs were dropped). Fixed, with tests; re-running whole.

## 1. Did the model change cause the drop?

Source: `week-drop-1120` (research-query run 44709119985; `outputs/review_20261120/week_drop_1120.txt`). The live
record (`live_orders`, mode live), CLV at Betfair's settled price, stake-weighted, 90% intervals by race day.

| Period | Served model | Days | Horses a day | Matched a day | Settled a day | CLV settled |
|---|---|---|---|---|---|---|
| A: 10 Oct - 8 Nov | old blend | 30 | 129 | GBP3,679 | +GBP209.55 | +4.97% (+3.46 to +6.34) |
| B: 9 - 13 Nov | old blend | 5 | 61 | GBP2,020 | +GBP13.36 | +0.95% (-0.06 to +2.14) |
| C: 14 Nov | market-rating blend | 1 | 116 | GBP3,375 | +GBP16.48 | +2.45% |
| D: 15 - 19 Nov | stack blend | 5 | 72 | GBP1,887 | +GBP62.89 | +4.31% (+0.40 to +9.26) |

| Comparison | Difference in CLV | Reading |
|---|---|---|
| B less A (same model, the card changed) | -4.02 points (-5.72 to -2.19) | the drop, resolved, on the old model |
| D less B (same card, the model changed) | +3.36 points (-0.75 to +8.43) | the stack no worse; not resolved better |
| D less A | -0.66 points (-4.96 to +4.44) | no difference resolved |

What the record shows:

- **Timing.** The fall is on 9 Nov (CLV -0.37%, -0.15%, +0.24% on 9-11 Nov). The served model changed for 14 Nov's
  card (`serve-market-rating-1113`) and again for 15 Nov's (`serve-stack-1114`).
- **The trader behaved the same after the change.** Horses backed a race 3.60 (A), 3.13 (B), 3.47 (D); the gap
  between the model's price and the market's at the first back 0.36, 0.33, 0.35; expected CLV 6.6%, 6.1%, 6.6%.
- **One day carries period D.** 15 Nov was +14.84%. The four days of this week (16-19 Nov) are +1.20%, -3.64%,
  +2.51%, +1.06%: about +1.1% together, the same as the old model's winter days (+0.95%). Five days cannot say the
  stack is better live. They do say it is not the cause.
- **What did fall.** (1) Money a day halved with the card: the turf flat ended, 129 horses a day became 61-72
  (`fall-diagnosis-1118`). (2) Jumps was already the thinnest code in the autumn (+3.69% against all-weather +6.73%
  and flat turf +5.74%) and the card is now about 85% jumps. On 9-13 Nov jumps made -0.16% (-1.96 to +1.94).
  (3) First backs: realised +12.39% in A, +4.81% in B, +8.97% in D.
- **Top-ups under-deliver in every period**, the autumn included: realised less expected -2.79 points (A), -5.40
  (B), -2.75 (D). This is the same finding as `phase5-reductions-1119` (the cap of 3 proposed to the owner).
- **To watch:** all-weather since 15 Nov is -2.02% (-4.92 to +3.02) on 152 horses. Not resolved; too few days.
- **Ruled out:** the pause in HRB's results (to 25 Nov). Only 0 to 1 runner a day (0.3%) had run since the last
  result day on 17-19 Nov. It grows: 4.8% of runners ran within 7 days and 8.0% within 9, so by 24-25 Nov about one
  runner in twelve will have its latest run missing from every form feature.

What could not be tested: the old model and the stack blend never priced the same live card (the shadow run was
stopped on 13 Nov, `serve-market-rating-1113` "not done"). Proposal M1 below fixes that for every future change.

## 2. What the model is

| Part | What it does | Source |
|---|---|---|
| First-stage models | market rating (a Kalman filter on past Betfair SPs), debut forecaster, expected time, pedigree (Bayesian), connections rating, improvement | `model/blocks/` |
| Three members | LightGBM, 1,181 features each, 12,000 rounds, learning rate 0.02, 127 leaves, extra trees; targets: logit of the race-normalised price (Huber), race-demeaned log price, within-race cross-entropy | `data/models/**/bfsp_model_meta.json` |
| The gate | the cross-entropy member takes the first three runners by rank in full, the next four at half, the rest not at all: weights 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0 | `data/models/bfsp_ensemble.json` |
| The closing model | at trade time: a conditional logit on the market's price now (0.774) and ours (0.263), fitted January-March 2026 | `data/models/closing_model_novol.json` |

Where the gain sits (`phase2-feature-audit-1119`): debut 15.5%, market rating 14.9%, within-race ranks 11.7%,
race-relative (wide) 9.6%, race-relative 8.1%, form windows 6.0%. Six families carry 64%. 386 of 1,181 features carry
90%; the bottom half carry 5.1%.

Where it is weak:

| Segment | Model error vs BSP | Reference | Source |
|---|---|---|---|
| All (development window) | 0.3747 single member, 0.3699 served blend | | `phase0-baseline-v0-1118`, `iter130-members-new-blocks` |
| BSP 50+ | 0.561 | 0.268 at 3-6 | `phase0-baseline-v0-1118` |
| 16+ runners | 0.428 | 0.336 at 2-7 | same |
| Maidens, novices, bumpers | 0.50-0.57 | 0.34-0.37 in handicaps | `model/blocks/debut.py` |
| Chase, live 6 Oct - 11 Nov | 0.360 | the morning price 0.249 | `model-qa-1113` |
| Hurdle, live | 0.443 | the morning price 0.379 | same |
| Flat and all-weather, live | 0.392 | the morning price 0.360 | same |

Two facts frame everything else:

- **Against the SP at the result the model adds nothing** (Benter weight about 0.00-0.01 against the market's 0.98,
  `phase0-baseline-v0-1118`). Against the *morning* price it earns about half the weight (0.50,
  `iter125-stack-final`). The edge is the morning price against the close, not the result.
- **Live, the edge is the price taken early**, not the picks: the trader's horses at the morning WAP -0.30%, at
  the matched price +8.29%, settled +3.64% (`phase5-execution-1119`).

## 3. Code review - defects and risks

No leak was found. The point-in-time tests cover every served feature (`tests/test_point_in_time.py`), a planted
leak is caught, and the baseline reproduces to 0.0002.

| # | Finding | Evidence | Effect | Fix and its test |
|---|---|---|---|---|
| C1 | **An unrated horse's rating of 0 is read as a number** in the older families. Newer code maps 0 to missing; older code does not | `model/custom_metrics.py:703` (the race's mean OR, which feeds `FCS`), `:1979-2001` (`or_change`, `or_change_3`, `career_best_or`, `or_vs_best`, `or_off_peak`, `or_vs_last_win`); `model/bfsp_features.py:726,738-739` (`or_num`, `or_vs_max`, `or_vs_median`, `median_or_num`): 11 served features and their within-race versions. Correct in `model/form_windows.py:129`, `model/primitives.py:711` | 538,146 rows (27%) have OR 0 (`phase0-baseline-v0-1118`). A horse unrated then rated 70 reads as "+70 lb"; a race mean falls with each unrated runner | A block that rebuilds these 11 features with 0 as missing; one loop arm against the served recipe |
| C2 | **No check on the closing model, and no shadow, when the served model changes** | `serve-market-rating-1113`, `serve-stack-1114`: both list "no shadow week"; `data/models/closing_model_novol.json` fitted Jan-Mar 2026, last changed 4 Nov | The rule's expected CLV is computed from weights fitted to an older model's prices. Period D shows no harm, but the owner's question could not be answered directly | M1 and M2 below |
| C3 | **No freshness check on results before pricing** | `predict_bfsp_today.py`: no comparison of the last result day with the card's day (only a fallback when no runners, `:976-984`) | The HRB pause went unflagged by the pipeline. 0.3% of runners affected so far, about 8% by 25 Nov | A guard that logs and reports the gap in days and the runners affected; Betfair's card gives days since the last run |
| C4 | **First-stage forecasters refit once a year**, on folds counted from the first day of the history | `model/blocks/expected_time.py:140-152` (`range(d0 + FOLD_DAYS, ...)`, 365 days); `debut.py` uses the same | With a history from 1 Jan 2018 the fold in force began at the turn of the year, so the debut and expected-time models serving today were fitted on data to about the end of December 2025. The fold dates move if the history's first day moves | Fold length as a registered arm (365 against 91 days); pin the anchor date |
| C5 | **Settings changed by overwriting another module's globals** | `model/blocks/debut.py:84-89` sets `expected_time.FOLD_DAYS` and three more, restored in `finally` | Works today; wrong results if two blocks ever build at once | Pass the settings as arguments. A refactor with the block's existing tests |
| C6 | **Dead and near-dead features are served** | `track_draw_bias` empty in every year; 5 sectional features over 90% missing, 12 empty 2021-2023; 2 with no gain (`phase2-feature-audit-1119`) | Noise, training time, and a leak test that passes without looking (`EMPTY_ON_THE_CARD`) | Drop them in the pruning arm (F1) |
| C7 | **Research fits die without a log about 47 minutes in**, 3-5 fits every iteration | iterations 127-131 (`iter128-stack-future-form`, `iter129`, `iter130`, run 44549603086) | Each iteration costs a re-run of several hours; the cause is not known | Log memory and the runner's CPU model at the start and every 10 minutes of a fit; compare lost against kept |

Also open, already with the owner: dependency versions are not pinned and the UK runner (pandas 3.0.6, numpy 2.5.3)
differs from GitHub's (3.1.1, 2.5.5), run 44494791431.

Checked and found sound: the walk-forward split (30-day purge, no meeting split), the price normalisation
(`model/bfsp_model.py:50-70`), the ensemble's serving rule (each member inverted by its own target,
`predict_bfsp_today.py:110-155`), settlement (`tests/test_pnl_by_hand.py`), deterministic fits.

## 4. Features by type - what we do and the leading approach

"Tried" cites the ledger so nothing is proposed twice. Each "next" is a hypothesis, not a claim.

| Feature type | What we do now | Leading approach | Status and next |
|---|---|---|---|
| The market's memory of a horse (past SPs) | Kalman local-level filter on multinomial-logit readings, two pools, uncertainty kept (`market_rating.py`) | this is the leading approach | Done, 14.9% of gain. Next: use `mr_sd` to scale the bar for uncertain horses (M5) |
| Ability from results (pounds beaten, figures) | race-centred pounds beaten windows; a figure model (`nbl_rating`), Kalman on the performance figure | a state-space or mixed model with horse and race effects fitted together, so each race's strength is estimated jointly | Partly: collateral and future form approximate it (`iter128`, `iter130` pass). Next: the figure member (`iter131`, running) |
| Rates for trainers, jockeys, sires | old families: raw 14 and 30 day win rates, no shrinkage (`custom_metrics.py:2037-2046`); new: a rating of each yard's pull on the price (`connections_rating.py`) | hierarchical (partial-pooling) Bayes, measured against the market's expectation, time-decayed | Shrunk rates did not improve the price, but did improve the win probability (`iter38`). Next: retire the raw-rate families in pruning (F1) |
| Official rating | as a number, with 0 for unrated in the older families | 0 as missing, plus an "unrated" flag; the rating's change only between rated runs | Defect C1. Next: F2 |
| Categories (course, race type, going) | integer codes in the trees; 86 courses, 37 race types | time-ordered target statistics with shrinkage, or a learner with native ordered categories (CatBoost) | Not tried. Next: inside the second learner (M3) |
| Horse, trainer, jockey, sire identity | not given to the model; only aggregates | learned embeddings or ordered target statistics | Not tried. Next: M3 |
| Within-race standing | hand-built ranks and z-scores, 29% of gain; a within-race cross-entropy member | a model that reads the whole field at once (attention over runners) | The hand-built route is mature. Next: only as M3's second variant |
| Counts that grow with history | raw counts; 43 features drift (PSI over 0.25) | rates, logs, or within-race ranks | Small: they carry 1.65% of gain. Next: fold into F1 |
| Draw | course and trip curves with effective sample sizes (`draw_curve.py`) | hierarchical curves pooled across course, trip, going and field size | 2.5% of gain. No test proposed: low value in a jumps winter |
| Pace and race shape | run style from comments, pace blocks; the simulator is not served | simulation or learned pace maps | 0.2-0.4% of gain. No test proposed |
| Times and sectionals | expected-time model (2.5%); sectionals empty before 2024 | standard times with a per-meeting going allowance; drop features with no history | Next: drop sectionals in F1 |
| Pedigree | Bayesian sire, dam, cross with uncertainty (`pedigree_bayes.py`) | this is close to the leading approach | Done; it feeds the debut forecaster |
| Unexposed horses | a first-stage forecaster of the market's price for horses with at most 2 runs (`debut.py`) | the same, refitted more often, with sales price and stable-mates' debuts | 15.5% of gain. Next: C4's refit arm; sales data (`iter107` tried a sales block) |
| Recency | fixed windows and linear recency weights; exponential decay features carry 0.09% | filters (Kalman) over windows | Recency weighting of the fit was worse (`iter72`). Nothing proposed |
| The model's own bias | none applied: the price is shorter than the BSP by about 5% on short prices and 17% at 40+ | a monotone calibration of the long end | A long-end correction cut the error 0.0029 but raised the winner's log loss (`price-bias-long-end-1113`). Next: read it under the owner's rule of 18 Nov (M6) |

## 5. Models - are they the best fit, and what should feed an ensemble

What is right: the target choice (three views of the same price), the gate, deterministic fits, purged
walk-forward folds, and a long list of things tried and closed: separate jumps and flat models are worse
(`iter117`), recency weighting worse (`iter72`), 28-day refits no better than 91-day (`retrain-cadence-1115`),
ratings from finishing order add nothing (`iter55`), rounds, leaves, regularisation and Huber all tuned
(`iter31`, `iter63`, `iter76-95`).

What is narrow: every member is the same learner on the same features with the same settings. Their errors are
highly correlated, so the blend gains little from averaging. The fits never stop early: all end at the round cap
(`best_iteration` 12,000).

| # | Candidate | Why | Evidence it could pay | Test and pass bar |
|---|---|---|---|---|
| M1 | **Shadow the previous blend for 14 days after every change** and replay the trader's rule on both | answers "is it the model?" from data, the day after | the question could not be answered directly today | Not a trial: a standing job. Reports CLV x stake a day, new less old |
| M2 | **Refit the closing model on the served blend's own out-of-sample prices**, replayed through the trader's code | the live weights (0.774 market, 0.263 model) were fitted to an older model; research gives the model about 0.50 beside the morning price | research CLV for the stack (+8.88% at the 3% bar, `iter125`) assumes a refit; two earlier refits on live books failed (`closing-refit-replay-1109`), so this one is fitted on the price files, not the live books | Replay on every recorded day: CLV x stake after reductions, refit less served, 90% interval above zero and deflated Sharpe at least 0.95 |
| M3 | **A second kind of learner as a member**: CatBoost with ordered categories for course, trainer, jockey, sire | different errors from the same data; identities are not in the model today | three seeds of the same learner already gave -0.0027 (`iter46`); a different learner should give more | One loop arm, then blended: paired error at least 0.0007 below the served blend, interval excluding zero, Brier and concordance not worse |
| M4 | **A learned stack in place of the hand-set gate**: weights by rank, code and field size, fitted on earlier folds only | the gate's weights were set once by hand, on 29 Sep, for an older model | a learned gate priced 0.0014 better than the step gate and traded no better (`iter105`); the owner's rule of 18 Nov now reads the price error first | Fold-by-fold weights from the folds before; same bar as M3 |
| M5 | **A price with its uncertainty**: quantile members, or the market rating's own standard deviation, to scale the bar | the closing model's spread varies only by price band (5 values) | first backs realise more than expected and top-ups less, in every period (section 1) | Replay: the bar in units of predicted spread against the flat 3% bar; same bar as M2 |
| M6 | **Read the long-end price correction under the owner's rule of 18 Nov** | it cut the error by 0.0029 (-0.0033 to -0.0025) and the 40+ bias from -17% to -3%; it failed on the winner's log loss (+0.0006), a criterion the rule of 18 Nov replaced with Brier skill and concordance, which were not read | `price-bias-1113`, `price-bias-long-end-1113`; the long end is where the error is (0.561 at 50+) | A new registration on the served blend's predictions: error, Brier skill and concordance as the rule of 18 Nov reads them. It may not pass |
| M7 | **A market-aware forecast at 05:15**: the blind price plus the overnight book | the morning price beats the model in every code (chase 0.249 against 0.360) | `model-qa-1113`; overnight books are recorded each night (22:36-04:46 UTC on 19 Nov, `night-before-1119`) | Needs about 60 nights of overnight books; the count held is not checked here. Register when 60 are held; until then M2 is the proxy |

Not proposed: more feature blocks on the same learner. About 130 iterations have been scored on one 15-month
window (28 Dec 2024 - 31 Mar 2026). The recent blocks are worth 0.001-0.003 each (`iter128`, `iter129`, `iter130`),
and the locked holdout has been opened at least twice with a third reading announced (`iter116`, `iter125`,
`iter132-final`). E7 below checks the gains on a window never used to choose them.

## 6. How to find errors and gaps - the programme

| # | Check | What it finds | Cost |
|---|---|---|---|
| E1 | **The period table of section 1, every Monday** (`week_drop_1120.py` as a standing query) | a fall in edge, by code and by first back against top-up, within a week | minutes |
| E2 | **Served features against the table's, every night** once results arrive (the one-off `serving-feature-qa-1111` made routine) | train-serve differences: it found the jumps stall and the weight before the claim | minutes |
| E3 | **A model of the model's own error**: a small tree model on the blend's out-of-sample residual, read by segment | where the error is predictable, so where a feature or member is missing | one query |
| E4 | **Train period against recent period classifier** over the 1,181 features | features whose meaning has drifted (the 43 already flagged, and any the audit missed) | one query |
| E5 | **Drop one family at a time** on the served recipe, for the six families with 64% of gain and the forty under 0.3% | what each is worth out of sample; what can be pruned | one loop iteration |
| E6 | **Rebuild a past day's features with the results cut off at that day**, on real data, for random days | leaks the synthetic card cannot see (5 served features are empty on it) | one query |
| E7 | **A second development window never used for selection** (Oct 2023 - Dec 2024, as `retrain-cadence-1115` scored) | whether 130 iterations' gains hold outside the window they were chosen on | one loop iteration |
| E8 | **Data monitors before pricing**: last result day, rows a day, nulls in key fields, card fields against yesterday's | pauses and feed changes on the day, not a week later | small code |
| E9 | **The morning price as the yardstick, by segment** | where the model is furthest from what is already knowable (chase, bumpers) | in E1 |

## 7. Order of work proposed

| Order | Item | Why first |
|---|---|---|
| 1 | M1 shadow, E1 weekly table, E8 monitors, C3 freshness guard | cheap; they answer the owner's question permanently |
| 2 | F2 (= C1, the rating of 0) and F1 pruning with E5 | a defect in 27% of rows; a smaller, cleaner model |
| 3 | M2 closing model refit, M6 long-end correction | both act on money directly and reuse existing code |
| 4 | M4 learned stack, then M3 second learner | the ensemble gains; M3 is the larger build |
| 5 | E7 second window, E3, E4, E6 | they audit everything above |
| Later | M7 market-aware forecast | waits for 60 days of overnight books |

F1: prune to the features that carry the gain (drop the dead, the sectionals, the raw-rate and drifting-count
families), judged by the paired error not rising by more than 0.0007. F2: the eleven official-rating features rebuilt
with 0 as missing.

Decisions for the owner: the order above; whether M1's shadow runs on GitHub's runners after each change; and the
open items from Phase 5 (a cap of 3 top-ups).
