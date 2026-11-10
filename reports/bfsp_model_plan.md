# Plan: a better BFSP / closing-price model

## Context

The owner asked to step back from the daily trading and improve the thing everything rests on: the forecast of
the Betfair SP and of the closing price. This plan reviews the models and features we have, audits every feature's
calculation, says honestly where the model stands, and sets out the order of work. All of it is research on the
branch. Nothing live changes until a candidate passes the decision rule, a final read and a week priced beside
the served model.

## 1. What we have today

### The models

| Model | What it is | Where |
|---|---|---|
| BFSP forecast | Three LightGBM boosters on the same 1,006 features, averaged by a hand-set gate, each race normalised to a book of 1 | `data/models/bfsp_ensemble.json`, `predict_bfsp_today.py` `AveragedBooster` |
| – main | Target: logit of the market's share of the race. Huber loss | `model/bfsp_model.py` |
| – dml | Target: log price less the race's mean. Squared error | same |
| – rx | Custom loss: cross-entropy against the market's shares within the race. Sets the top three prices | `race_xent_objective` |
| Closing model | A two-weight conditional logit: 0.774 on the morning Betfair price, 0.263 on our forecast. This is what the trader bets on | `model/race_book.py`, `data/models/closing_model_novol.json` |

The BFSP forecast never sees today's market. Today's price enters only in the closing model.

Settings are fixed by hand: learning rate 0.02, 127 leaves, 12,000 rounds, one seed, no early stopping, no
usable hyperparameter search. Retraining is manual. Trained through 13 Oct 2026.

### How good it is

| Forecast of the closing book | Error (mean abs log price) |
|---|---|
| Morning market alone | 0.350 |
| Our model alone | 0.377–0.400 |
| Closing model (both) | 0.302 |

Rank agreement with the BSP is about 0.91. Our model alone is behind the morning market; the value is in the blend.

Weak spots: prices over 21 (0.46) and over 51 (0.63); chases and Class 1–2 (15–28% behind the morning
market); forecasts are compressed (slope 1.03–1.11); live error 0.407 against 0.397 in testing.

### The feature set (1,006)

| Family | Features | Share of gain |
|---|---|---|
| Within-race comparisons (ranks, z-scores, gap to the best) | 289 | 57% |
| Form windows and variants (NFP, lengths beaten, pounds, performance figure) | 157 | 10% |
| Proprietary horse metrics and race strength (NFP, ORR2, EPF, PFD, OFS, prize money, DSLR) | ~87 | 6% |
| Connections (trainer, jockey, pair, bookings, windows) | ~64 | 6% |
| Intent, freshness, handicap angles, exposure, travel | 64 | 7% |
| Draw and pace | 151 | 4% |
| Debutants (how the market prices the yard's newcomers) | 10 | 2.4% |
| Ratings and times (Kalman rating, time figure, sectionals, RSR) | 45 | 2% |
| Pedigree | 15 | 0.5% |
| Card, context and other | ~98 | 5% |

Cutting across families: prize-money features carry 14.6% of gain and past market prices 18.9%.

### Feature importance (main model, top 15 by gain)

| # | Feature | What it is | Gain |
|---|---|---|---|
| 1 | rw_fv_nfp_s8_gap | Shrunk finishing-position form, gap to the best in the race | 6.4% |
| 2 | rPMW3 | Rank in race: prize-money figure, last 3 | 3.3% |
| 3 | rw_fv_nfp_e3_gap | Recency-weighted form, gap to the best | 3.2% |
| 4 | rPMW5 | Rank: prize-money figure, last 5 | 2.9% |
| 5 | rr_fw_mkt_w5_gap | The market's past view of the horse, gap to the best | 2.4% |
| 6 | rPMW10 | Rank: prize-money figure, last 10 | 2.2% |
| 7 | hca_wt_vs_mark | Weight carried against the handicap mark | 2.2% |
| 8 | rr_PMW3_z | Prize-money figure last 3, z-score in race | 1.9% |
| 9 | rr_fw_nfp_w5_gap | Form last 5, gap to the best | 1.6% |
| 10 | rw_PMW5_z | Prize-money figure last 5, z-score | 1.5% |
| 11 | OFS1 | Odds-based figure, last run | 1.1% |
| 12 | fw_mkt_l1 | The market's view at the last run | 1.0% |
| 13 | rOFS3 | Rank: odds-based figure, last 3 | 1.0% |
| 14 | fv_lbpf_l1 | Lengths beaten per furlong, last run | 1.0% |
| 15 | rORR2LR | Rank: odds-to-runners ratio, last run | 0.9% |

The top 10 features carry 28% of gain, the top 100 carry 67%. Full table: `data/models/bfsp_feature_importance.csv`.
Gaps: no importance for the dml and rx members, and no permutation or SHAP importance at all.

### Is it best in the world?

No. It is a well-run model with honest testing, but it falls short of the best in six ways:

1. One model family (boosted trees), one seed, hand-set settings, a hand-set gate.
2. The model does not look at the race as a whole. 57% of its gain comes from hand-made within-race comparisons.
3. Past prices are used raw. A 4/1 shot in a Group race and in a seller read alike, so prize money stands in for class.
4. 110+ experiments were all judged on one six-month window (two folds). Some recorded gains will be luck.
5. Data: no weather, no GoingStick, partial sectionals, no Racing Post figures. The top syndicates have all of these.
6. The features hold about 35 calculation faults and about 250 duplicate or dead columns (section 2).

Already tried and failed, so not in this plan: extra closing-model terms, closing models by race type, price-move
models, result-based Elo, pedigree blocks, pace and draw rebuilds, comments, recency and profit weighting, other
losses and learning rates. A learned gate and three-seed averaging each gained a little (−0.0014, −0.0026) and
were not served.

## 2. Feature-by-feature review

Every function that builds a served feature was read in full: the engine (`model/custom_metrics.py`,
`financial_features.py`, `bfsp_features.py`), the form, intent, freshness, pace and draw modules, and the 14
blocks in `model/blocks/`. Each calculation was checked for its formula, leakage, and statistical method.
This was a code read. There is no database on this machine, so sizes are not measured and a few items are marked
"to confirm".

### Headline findings

- **No leak of the result or of today's price** in any served feature.
- **The live field is not the training field.** Training builds within-race features on the horses that ran.
  Live builds them on the 06:00 card, which still holds later non-runners, and the non-runner update does not
  rebuild them. This touches 57% of gain. The serving test never exercises the case
  (`tests/test_serving_blocks.py:36`).
- **About 35 confirmed faults**, listed below.
- **About 250 columns are duplicates, dead, or rescalings of the same few signals.**
- **Statistics are hand-set throughout.** Every shrinkage constant is typed in, none is fitted
  (`model/shrinkage.py method_of_moments_k` is used by no served feature). Windows count runs, not days.

One caution. A tree model does not care how a single feature is scaled, so a fault that only flips or stretches
a column changes little: fixing the direction of four ranks was tested twice (iterations 41 and 54) and did
nothing. The recommendations below are ordered by whether they change the information the model gets.

### Verdict by family

| Family | Gain | Calculation valid? | Recommendation | Priority |
|---|---|---|---|---|
| Within-race z-score and gap (`rr_`, `rw_`; 206) | 37% | Arithmetic right. Gap is to the single best runner, so one outlier or one late non-runner moves the whole race. z has no minimum field: with two known values it is always ±0.71. Money, counts and days go in raw and skewed. For ten lower-is-better inputs the "gap to the best" is the gap to the worst | Gap to the best rival (leave the horse itself out). Log skewed inputs first. At least 3 known values, z shrunk by n/(n+3). A softmax share for past-market inputs: the old price renormalised to today's field | High |
| Within-race ranks (`r*`; 83) | 20% | Raw ranks, so rank 5 is last of 5 and top quarter of 20. Missing values sit beside the worst. About 10 exact duplicates | Rank as a share of the field, missing kept missing, duplicates removed | High |
| Prize money (`PMW`, `WPMRF`, ranks) | 14.6% | Not prize won: it is the race's prize × finishing position squared. No adjustment for year, class or currency. Missing prize counts as 0. `RACE_WPMRF` is prize × field size | Deflate by the year × code × class median and take logs. Add prize actually won (`horse_prizewin` is in the table, unused). Missing kept missing | High |
| Past prices (`ORR2`, `RWO`, `OFS`, `PFD`, `fw_mkt`) | ~19% | Runners ÷ BSP is averaged as it stands, so one short-priced run dominates. `OFS` is an exact copy of `ORR2`. No adjustment for the strength of the field | Average the log share of the book. Decay by days. The market-implied rating (Step 4) | High |
| Finishing-position form (`NFP`, `fw_`, `fv_`) | ~10% | Non-finishers are dropped, so a horse pulled up three times in five keeps the form of its two finishes. No decay by days. No count or spread beside the mean | Non-finisher scores 0, plus a non-finish rate. One window decayed by days. Carry the count. Fit the shrinkage in `fv_nfp_s3/s8`, prior by trainer × code | High |
| Lengths and pounds beaten (`LB`, `fw_lbs`, `fw_lbsc`) | Largest loss when withheld | Hard cap of 20 lb is 6.7 lengths at 5f but 20 lengths at 2m. Pounds per length by trip only. Centred on a mean pulled by strung-out fields | Conversion by going and code (`primitives.lps_scale` exists). Soft cap. Centre on the median. Pulled-up counts as the cap | High |
| Performance figure (`fw_perf`, exposure, Kalman) | ~3% | Anchored on each horse's own rating, so two horses finishing together but rated 20 lb apart get figures 20 lb apart. A non-finisher is rated as running to its mark (fault). Kalman noise settings typed in | One standard per race, then pounds beaten and weight carried. Non-finisher missing. Kalman settings fitted (`state_space.fit_kalman_params`) | High |
| Official rating (`or_num`, `or_vs_*`, `or_change`) | ~2.5% | Unrated horses are stored as 0 and never masked. A first rated run reads as +70 | Mask 0 as unrated everywhere; add the share of the race that is rated | High |
| Going (`going_numeric`, preferred going) | — | Good to Soft is read as Soft, Good to Firm as Good, Standard to Slow as Standard | Exact matching; turf and all-weather on separate scales | Medium |
| Speed (`RSR`, `fw_rsr`, `tf_`) | ~1% | `RSR` is the race's winning time, the same for every runner. Trips rounded to the furlong (about 3% of time). Mean, not median. Going allowance is a plain mean of the day | The horse's own time. Standards on exact yards, rolling median. Going allowance shrunk as in `greyhound/times.py meeting_variant` | Medium |
| Trainer and jockey rates (`WIV`, `WAX`, `CWO`, 14/30-day, track) | ~3% | No shrinkage: one win in a 20-runner field gives WIV 20. Lifetime, no decay. Track rates divide by the wrong count. `form_delta` subtracts a ratio from a rate | (Wins − expected) ÷ its standard deviation. Shrink with a fitted constant. 365-day decay. Split by code | Medium |
| Handicap angles (9) | 2.6% | `hca_wt_vs_mark` is anchored on the top weight, so a penalised top weight shifts every other runner | Anchor on the race median of weight + rating | Medium |
| Connection windows, bookings (20) | 3.1% | Right, but shrunk toward 0.5 and toward 0, both of which are off the true average | Prior set to the code's average; fitted constant | Medium |
| Debut market (10) | 2.4% | Right. The z mixes debutants, second and third runs on different baselines | Centre each stage first; prior by code × race type | Medium |
| Intent (20) | 1.3% | `it_ae_sum`, `it_ae_best`, `it_n_angles` include the gelding angle, which the 06:00 card cannot know. Layoffs read from dates with missing months | Seven card-safe angles only; layoff from the card's days-since-run | Medium |
| Race strength (10) | 0.6% | Includes the horse itself; live uses the declared card | Mean of the rivals only | Medium |
| Freshness (14) | 1.7% | Valid | Add spread of the usual gap; trainer gap by code | Low |
| Pedigree (15) | 0.5% | Missing sires pooled as one sire called "unknown" | Leave missing; prior by code | Low |
| Head-to-head, form lines, travel, sectionals (39) | 1.2% | Valid. Sectionals cannot tell "jumps run" from "no data" | Weight-adjusted margins; a coverage flag for sectionals | Low |
| Pace (91) | 1.4% | Comment parser faults ("chased leaders" scored as "chased leader"; a missing comment scored as midfield). About 86 are rescalings of three signals | Fix the parser; cut to about 5 | Low |
| Draw (60) | 2.7% | One column blank, several duplicates, some built on the final field | Cut to about 4 (`draw_pct_adj`, `draw_bias_ev_course`, `_adj`, `_n_eff`) | Low |
| Financial extras (RSI, MACD, Sharpe, skew) | ~1% | Valid but noise on 3–5 runs | Drop skew and Sharpe | Low |

### Confirmed faults (the ones that change information)

| # | Fault | Where |
|---|---|---|
| 1 | Within-race features built on a different field live than in training | `model/blocks/__init__.py:169`, `predict_bfsp_today.py:463, 559` |
| 2 | Unrated official rating (0) never masked | `bfsp_features.py:726-739`, `custom_metrics.py:703, 1969-2002` |
| 3 | Going read wrongly by substring order | `custom_metrics.py:1275-1289`, `bfsp_features.py:742-756` |
| 4 | Non-finisher rated as running to its mark; feeds the Kalman rating | `perf_figures.py:172-173` |
| 5 | Non-finishers dropped from win, A−E and NFP means | `form_windows.py:110, 146`, `custom_metrics.py:456` |
| 6 | Gelding angle (not on the 06:00 card) inside three served intent features | `intent_features.py:222-226` |
| 7 | Trainer/jockey track rates, pace index, jockey last-run index: wrong denominator | `custom_metrics.py:896-899, 927-929, 1938-1953` |
| 8 | `trainer/jockey_form_delta` subtracts a ratio from a strike rate | `custom_metrics.py:2055-2063` |
| 9 | Prize: missing counted as 0; `RACE_WPMRF` is prize × field size; `prize_momentum` reads the wrong column | `custom_metrics.py:762, 765`, `financial_features.py:289-292` |
| 10 | `hca_wt_vs_mark` anchored on one horse | `handicap_angles.py:67-69` |
| 11 | `form_slope` sign inverted, so `is_improving` flags decliners | `custom_metrics.py:1364-1389` |
| 12 | Pace comment parser order; missing comment scored as midfield | `pace_metrics.py:72-106` |
| 13 | Bookings shrunk toward 0, which is above the true average | `bookings.py:60, 92` |
| 14 | Layoff from held dates despite missing months | `intent_features.py:166-168` |
| 15 | Speed standard pools trips by rounding to the furlong | `custom_metrics.py:1701` |

To confirm against the data: Irish prizes in euros; time-figure meetings mixing turf and all-weather; Southwell's
surface change inside one standard time.

### Columns to remove (about 1,006 → about 750)

- Exact copies: `WOA` (= `WAX`), `RB` (= `NFP`), `OFS` (= `ORR2`), `racepacescore` (= `RPS`),
  `course_experience` (= `horse_track_runs`), `class_change` (= `it_class_change`), `it_debut` (= `is_debut`),
  `DSLR1`, `hca_days_since_lr`, `rw_preracehorsecareerRB_*`, the duplicate ranks.
- Dead or mis-defined: `track_draw_bias` (blank), `CIL*`, `LR*COUNT`, `LR*wsum`, `LR*_ORR2` (unnormalised sum),
  the DSLR "diffs" and `FinalDSLR` (the freshness block replaces them).
- Pace 91 → 5 and draw 60 → 4.

### How the repairs are made and tested

The served model must keep getting the features it was trained on, so the engine and the served blocks are not
edited. Each repair is a new column in a new drop-in block, tested with the old column withheld
(`--drop-feature-list`). The old code is retired only at a model swap. Six batches, each one loop run with a
placebo arm:

| Batch | Contents | Why this order |
|---|---|---|
| A | Within-race transforms: gap to best rival, logs before z, minimum field, softmax share, rank as share | 57% of gain |
| B | Data faults 2–14 above | Wrong information today |
| C | Prize deflated and logged; past prices as log share, decayed by days | 34% of gain between them |
| D | Performance figure on a race standard; pounds by going and code; soft cap; own-time speed figure; fitted Kalman | The form inputs to batch A |
| E | Shrinkage with fitted constants and decay by days: trainer/jockey, hot form, track, connection windows, debut market, pedigree | Small samples |
| F | Remove the ~250 columns | Cheaper fits; last, so it is measured against the repaired set |

## 3. The work, in order

Gains are changes in the log-price error. The noise from one refit to the next is 0.0015–0.0028, so a change
has to beat that to count.

| Step | What | Expected gain | Stop rule |
|---|---|---|---|
| 1 | A wider measuring stick | None directly; halves the noise bar | — |
| 2 | Show what we have; save the full audit | — | One page plus the audit report |
| 3 | Measure the morning-card gap (fault 1) | Possibly the largest; one day read 0.412 against 0.369 | Build nothing before the measurement |
| 4 | Market-implied rating (the main new feature) | −0.002 to −0.006 | One screen, one tuning pass |
| 5 | Feature repairs, batches A–F (section 2) | −0.0005 to −0.003 a batch | Keep only what clears the bar |
| 6 | One learned combiner in place of the gate | −0.001 to −0.002; slope toward 1 | Serve only if it clears the bar |
| 7 | Closing model: same form, ten times the races | Small; may fix the overstated edge | Must win the trader's replay |

### Step 1. A wider measuring stick
- Screen on 11 folds (30 Sep 2023–31 Mar 2026) instead of two: `"bfsp_from": "2023-09-30"` in `research/loop.json`,
  base set to the 14 served blocks, with a `placebo` arm and a second seed to re-measure the noise.
- A guarded final read of Apr–Sep 2026, once per stage: a `bfsp_until` key in `scripts/research_loop.py`
  (replacing `EVAL_UNTIL` at :110 and :491), honoured only with `"final": true`, refusing a second look and
  writing a "HOLDOUT LOOK" ledger entry. Copy the guard from `scripts/outcome_model.py:444-458`.
- A second score for every arm: the closing model's error built on that arm (`scripts/clv_betfair.py`, reusing
  `rb.whole_races`, `rb.closing_inputs`, `rb.fit_closing_model` in `model/race_book.py`).
- A segment table in `scripts/compare_oos_runs.py` (price band, code, class, country, maiden) beside `by_rank`.
- Cost: about 15 runner-hours for the first screen, 13 after; about 80 for a confirm on the served settings;
  about 26 for a final read.

### Step 2. Show what we have
- Save the full audit (every calculation, formula as coded, verdict) as `reports/feature_audit.md`.
- Rerun `scripts/feature_workbook.py` for the 1,006-feature model (the 24 Sep inventory is for 501 features).
- Gain for the dml and rx members (one read-only query).
- Withhold importance by family on Step 1's cached base (`--drop-feature-list`, new lists in `research/lists/`).

### Step 3. The morning-card gap
- One read-only query over the live record: the 06:00 file's error on races with and without later non-runners.
- If the gap is real: rebuild the within-race features when non-runners come out, in place of renormalising
  (`predict_bfsp_today.py:549`), and add a serving test where a card runner has no price.
- Also a model-age curve from the wide folds, to set a retrain schedule.

### Step 4. Market-implied rating block (`model/blocks/market_rating.py`, prefix `mr_`)
- One rating per horse and code pool. Each past race observes the horse's log BSP share against the field's
  ratings, so the rating is on one scale across classes. Kalman update with drift by days since the last run;
  drift and noise fitted (`model/state_space.py fit_kalman_params`).
- Features: rating, its uncertainty, today's implied share of the race, z, gap, rank, field strength, trend,
  rating against the official mark.
- Follow-up `market_plus_minus.py`: trainer and jockey effects on what the horse rating leaves over, by
  `CrossClassifiedRidge` (`model/effects.py`) with time decay. This is football's market plus-minus.
- Pattern to copy: `model/blocks/elo.py` (pools), `model/blocks/kalman.py` (filter). Create with
  `python scripts/new_block.py market_rating --prefix mr_`; `tests/test_feature_blocks.py` checks leakage.

### Step 5. Feature repairs
Batches A–F of section 2, one loop run each. Reuse: `model/shrinkage.py` (`method_of_moments_k`, `shrink_chain`),
`race_shape.asof_decayed_mean`, `draw_metrics.race_lagged_decayed_mean`, `primitives.lps_scale`,
`greyhound/times.py meeting_variant`, `greyhound/ability.py p_best`, `state_space.fit_kalman_params`.

### Step 6. One combiner (`model/stack.py`, new; `race_book.py` is not edited)
- A conditional logit on the three members' race-normalised log prices against BSP shares, with rank-band terms,
  fitted on earlier folds only. The sum of the weights corrects the compression and keeps the book at 1.
- Costs no fits: it runs on the per-arm `oos_*.csv` files Step 1 writes. Extra seeds are tested second.
- Serving later: per-member weights in `AveragedBooster.predict` and the manifest; `tests/test_averaged_serving.py`.

### Step 7. Closing model
- Refit the same two-weight form on every race with a walk-forward forecast (about 25,000 after Step 1, against
  2,504 today), and separately on the recorded 05:15–10:00 books (about 40 days).
- No new terms: five were tested on 30 Sep and added nothing.
- The gate is the trader's daily replay, not the error. Two earlier closing-side wins died there.

### Parked
A race-level neural network. It needs a new library, and the rx member already fits a within-race loss. Open it
only if Steps 4–5 show within-race features still gaining.

## 4. Critical files

| File | Change |
|---|---|
| `research/loop.json` | Wider folds, arms for each experiment |
| `scripts/research_loop.py` | Guarded final read; `"stack"` ensemble type |
| `scripts/clv_betfair.py`, `scripts/compare_oos_runs.py` | Closing score; segment table |
| `reports/feature_audit.md` (new) | The full audit |
| `model/blocks/market_rating.py`, `market_plus_minus.py` (new) | Step 4 |
| `model/blocks/` new blocks for batches A–E; `research/lists/` drop lists | Step 5 |
| `model/stack.py` (new) | Step 6 |
| `predict_bfsp_today.py`, `data/models/`, the engine | Only at a swap, after the gates |

Not touched: `trading/`, `auto_trade.py`, `live-trade.yml`, `model/race_book.py`, the served blocks and engine.
New blocks keep to numpy, pandas and scipy and a unique prefix, because the morning run imports every block file.

## 5. Verification

1. Each block: `pytest tests/test_feature_blocks.py -k <block>` and a hand-worked value test.
2. Screen: edit `research/loop.json`, push; `research-loop.yml` runs; read `compare.md` in the artifact.
3. Decision: `compare_oos_runs.py decide` on 11 folds, with a placebo arm, on both scores.
4. Record every run in `reports/research_ledger.jsonl`, kept or not.
5. Confirm the survivors together on the served settings (six folds).
6. Final read on Apr–Sep 2026, once per stage.
7. Swap: `train-bfsp.yml` per member, `scripts/verify_model.py` passes, `pytest tests/test_averaged_serving.py
   tests/test_verify_model.py tests/test_workflow_guards.py`, a week priced beside the served model, then merge.
   The merge is the deploy; the owner is told before it.
