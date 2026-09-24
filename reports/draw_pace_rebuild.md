# Draw and position metrics, rebuilt: 24 September 2026

The brief:
- **Draw:** normalised finishing position and normalised lengths beaten, by course, distance and field size.
- **Position:**
  - read each horse's past early positions from the race comments, and project its running style for today;
  - define the race's shape from the field's projected styles;
  - measure what each expected position is worth within that shape, in lengths beaten.

Two research blocks now do this:
- `model/race_shape.py` (24 features);
- `model/draw_curve.py` (8 features).

Both sit outside the served feature path. Nothing the 06:00 job or the feature cache imports has changed.

Every number below is from the development data: races to 31 March 2026. The locked holdout (1 April 2026 on) was not read.

## 1. What was built

### The two outcomes

Every past run is scored twice, each centred within its race:

| Outcome | Definition | Scale |
|---|---|---|
| `rs_nfp_c` | Normalised finishing position, (runners − place) / (runners − 1), less the race mean | Winner 1, last 0; + = beat the average runner |
| `rs_lbs_c` | The race's mean pounds beaten less this runner's | Lengths × pounds per length at the trip, capped at 20 lb; + = beat the average |

Pounds rather than lengths, because a length is worth about 3 lb over five furlongs and under 1 lb over two miles. An average across trips has to be on the weight scale.

### Draw: `model/draw_curve.py`

- **Where the horse is drawn:** the stall's rank among the race's runners from stalls, 0 = lowest to 1 = highest. It is binned into fifths of the field.
- **Cells:** course × code × trip × field size (2–7, 8–11, 12–15, 16+) × fifth of the field.
- **Pooling:** each cell is shrunk toward the level above, in proportion to its effective sample:
  - code × field size;
  - course × code;
  - course × code × trip;
  - course × code × trip × field size.
- **Recency:** each past race is weighted by a two-year half-life, because rails, watering and resurfacing move biases.
- **Features:** the draw's worth in finishing position and in pounds (`dc_edge_nfp`, `dc_edge_lbs`). Also:
  - the edge relative to this field (`dc_edge_rel_lbs`);
  - how much the draw matters in this race (`dc_race_spread_lbs`, best stall less worst);
  - the evidence behind the cell (`dc_n_eff`).
- **Draw by running style:** the same draw is not worth the same to every runner. An inside stall at a turning sprint helps a horse that races handily most. The finest cell is therefore split by the runner's projected style, forward or not, and shrunk toward the draw-only cell (`dc_edge_style_lbs`, `dc_edge_style_rel_lbs`).

Stalls are drawn at random in Britain and Ireland, so a raw average by draw position is an unbiased estimate of the draw's effect. The cells need pooling because they are thin, not because anything is confounded.

### Position: `model/race_shape.py`

1. **Run style from the comment.** The comment's *first* positional phrase is read on word boundaries: led, disputed, prominent, tracked, mid-division, held up, in rear, and about 110 phrasings of them.
   - A named place ("raced in 4th", "close 3rd") gives the early position exactly: (place − 1) / (runners − 1).
   - A lead taken in the first furlong or two ("prominent, led after 1f", "soon led" in the opening clauses) counts as leading.
   - A slow start yields to a position given right after it.
   - A comment with no position is *unknown*, not midfield.
   - The parser reads 99.6% of comments.
2. **Projected style.** For each horse, recency-weighted chances of leading, racing prominently, racing mid-division and racing in rear (`p_lead`, `p_prom`, `p_mid`, `p_rear`), plus a projected early position (`pred_epf`).
   - Recency uses a 4-run half-life.
   - The chances are shrunk toward the population, weighted as 4 runs, for the code, trip band and debut status.
   - Tuned on 383,693 runs: `research/queries/done/style_projection_tuning.py`.
3. **Race shape**, from the field's projected styles:
   - expected number of leaders;
   - chance that nobody leads;
   - chance the lead is contested;
   - share of the field on or near the pace;
   - the two strongest claims on the lead, and how clear the first is;
   - a three-way shape: *no natural leader* (nobody's chance of leading reaches 0.30), *one natural leader*, or *contested lead* (a second runner's does).
4. **What the expected position is worth in that shape**, in pounds and in finishing position:
   - Cells are course × code × trip × field size × shape × projected position.
   - They are pooled down code × shape → + trip band × field band → course × code × trip × shape → + field band.
   - `pv_exp_lbs` is the value of the most likely position.
   - `pv_act_lbs` weights every position's value by the runner's chance of getting it. For each position, it uses the history of horses that actually raced there.

### Leakage

Every statistic of a race uses races on **earlier days** only, through exact per-key prefix sums. `tests/test_race_shape.py` has 42 tests, including:
- **Day-flip test:** today's results and comments are scrambled or blanked. Every feature for today must stay bit-identical. The test was mutation-checked: it fails when a same-day row is let in.
- **Planted effects:** a draw bias, a lone-leader advantage and a draw penalty that hits only forward runners drawn high.
  - Each is found in the right direction and at most of its planted size.
  - No draw bias is found at the courses where none was planted.
- A brute-force as-of check of the pooled cells.

## 2. What the served features get wrong

These were measured on the same rows (`research/queries/done/position_draw_survey.py` and `draw_pace_validity.py`).

**Run style:**
- The served parser (`pace_metrics.parse_run_style`) matches "led," inside "pulled," and "travelled,".
- 19% of its "leaders" are not leaders on the comment's first phrase. Most are "prominent, led after 1f" type comments.

**Draw:**
- `td_draw_bias` and `tdg_draw_bias` explain nothing: within-race R² 0.00 (§3.4).
- `track_draw_bias` is blank on every row (feature inventory §5).
- The served draw feature that does work is `draw_bias_ev_course` (R² 2.37).

**Pace:** the served pace set's strength is mostly form, not pace.
- Its best single feature is `front_sustainability`, with R² 20.6 on pounds beaten. That is the horse's average late movement in its past runs from prominent positions.
- That is a measure of how good the horse is. It is a useful feature, but it is not the shape of the race.

## 3. Results

**How the sets are scored:**
- Within-race R² × 1000: the share of within-race variance a feature or set explains out of sample.
- Sets are fitted on 2023–24 and scored on 2025 to March 2026.
- Single features use race-clustered t.
- Source: `research/queries/done/draw_pace_validity.py`, research-query run 21 on the tuned blocks.

### 3.1 Running style

Scored on 360,391 runs from 2023:

| | Correlation with the comment's early position | Log loss for leading (base rate 0.3805) |
|---|---|---|
| New projection (`pred_epf`, `p_lead`) | **0.370** | **0.3455** |
| Served (`pred_epf_norm`, `predicted_lead_prob`) | 0.305 | 0.3584 |

`p_lead` is calibrated decile by decile: from 0.055 to 0.324 predicted, against 0.040 to 0.363 realised.

### 3.2 Race shape and what each expected position is worth in it

Mean pounds beaten relative to the race average (+ = beat it), by projected position within projected shape, 2023 to March 2026:

**Flat and all-weather** (24,597 races)

| Shape | Lead | Prominent | Mid-division | Rear |
|---|---|---|---|---|
| No natural leader (16,021 races) | −0.36 (177 runs) | **+0.43** | +0.18 | **−0.59** |
| One natural leader (6,485) | +0.06 | +0.13 | +0.07 | −0.29 |
| Contested lead (2,091) | **−0.62** | +0.08 | +0.47 (990 runs) | **0.00** |

**Jumps** (16,761 races)

| Shape | Lead | Prominent | Mid-division | Rear |
|---|---|---|---|---|
| No natural leader (10,268) | +1.27 (128 runs) | +0.59 | −0.09 | −0.68 |
| One natural leader (4,724) | **+0.97** | +0.21 | −0.37 | −0.52 |
| Contested lead (1,769) | +0.08 | +0.10 | +0.16 | −0.36 |

The shape changes what a position is worth, in the direction a race reader would expect:
- **On the flat, a hold-up horse** loses 0.59 lb when nobody wants to lead. It loses 0.29 lb behind one natural leader, and nothing when the lead is contested.
- **A projected leader in a contested lead** loses 0.62 lb, against +0.06 lb when it is the only one.
- **Over jumps, an uncontested leader** gains about 1 lb; in a contested lead the gain is gone.

These are raw averages. A projected position also carries some ability: prominent runners are a little better on average. So compare *across* shapes within a column, not across columns within a row. The model features (`pv_*`) are measured at course × trip × field size, not only by code as here.

### 3.3 Position value

| | Out-of-sample within-race R² × 1000, pounds beaten | Finishing position | Win |
|---|---|---|---|
| New position set | 6.9 | 7.8 | 2.5 |
| Served pace set | 24.7 | 24.2 | 9.3 |
| **Both** | **27.5** | **27.0** | **10.0** |

Scored on 133,337 runners. By code, on pounds beaten:
- flat and all-weather: new 6.7, served 21.1, both 23.9;
- jumps: new 7.9, served 33.6, both 35.6.

The new set adds 2.8 to the served one. It does not replace it: the served set carries form (§2), and the new set is about the race.

Calibration of `pv_exp_lbs`, deciles out of sample: −0.84 to +0.77 lb predicted, −0.75 to +1.06 lb realised. The top decile is a little conservative.

### 3.4 Draw

Single features, flat and all-weather races from stalls, 2023 on:

| Feature | Within-race R² × 1000, pounds | t |
|---|---|---|
| `dc_edge_style_lbs` (new, draw by projected style) | **3.11** | 24.4 |
| `dc_edge_lbs` (new) | 2.79 | 23.1 |
| `draw_bias_ev_course` (served, best) | 2.37 | 20.9 |
| `draw_bias_ev_stall` (served) | 2.01 | 19.8 |
| `td_draw_bias`, `tdg_draw_bias` (served) | 0.00 | 2.5, −0.9 |

Sets out of sample (83,442 runners):

| | Pounds | Finishing position | Win |
|---|---|---|---|
| New draw set | **2.76** | **2.75** | **0.61** |
| Served draw set (12 features) | 2.51 | 2.44 | 0.58 |
| Both | 3.18 | 3.10 | 0.81 |

**Calibration.** `dc_edge_lbs` by decile is −0.64 to +0.56 lb predicted and −0.74 to +0.55 lb realised. A pound of predicted draw advantage is worth a pound.

**Tuning.** It took eight times the first guess at the shrinkage (`research/queries/done/draw_pace_tuning.py`). Before that, the finest cells were twice too extreme and the set trailed the served one (2.0 against 2.5).

**Do the course-and-trip biases hold up in races they have not seen?** Across 48 course × trip cells with 40+ races in 2025 to March 2026:
- predicted and realised low-minus-high edges correlate 0.80;
- the weighted slope is 1.19.

The strongest:

| Course | Trip | Races | Predicted low − high (lb) | Realised |
|---|---|---|---|---|
| Kempton | 6f | 107 | +2.17 | +2.85 |
| Wolverhampton | 5f | 128 | +1.65 | +1.87 |
| Kempton | 7f | 141 | +1.43 | +1.53 |
| Newcastle | 10f | 77 | −1.26 | −2.64 |
| Kempton | 1m | 146 | +1.23 | +1.43 |
| Kempton | 11f | 50 | +1.18 | +1.84 |
| Southwell | 6f | 134 | +1.14 | +0.53 |
| Wolverhampton | 7f | 184 | +1.07 | +0.26 |
| Lingfield | 1m | 106 | −0.91 | −1.49 |
| Chelmsford City | 5f | 47 | +0.90 | +1.25 |

(+ = low draws favoured.)

**Splits that did not help:**
- stall placement (+0.05);
- going (−0.01);
- measuring the cells on outcomes less their rating-implied part (+0.01).

## 4. Against the market

The blocks describe the race better than the served features. Whether they make money depends on whether BSP already knows it.

**Iteration 17** (research-loop run 28, blocks at their pre-tuning settings):
- **Set-up:** every production feature plus both blocks; the market as an offset; walk-forward over 40,932 development races.
- **Full model:** −0.457 mnats per race against the market (t −2.55).
- **Residual screen** (each block's joint information beyond BSP and the production features):
  - shape +0.136 mnats (t +1.51);
  - draw −0.009 (t −0.25).
- **Verdict:** nothing beyond BSP. The market prices draw and pace, as it priced every fundamental block before them.

**Iteration 19** (research-loop run 34): the same test on the tuned blocks, 533 features, 40,932 races.

| | Iteration 17 (first settings) | Iteration 19 (tuned) |
|---|---|---|
| Full model against the market, mnats/race | −0.457 (t −2.55) | **−0.220 (t −1.24)** |
| Shape block over BSP and production | +0.136 (t +1.51) | **+0.157 (t +1.57)** |
| Draw block over BSP and production | −0.009 (t −0.25) | −0.033 (t −0.57) |
| Rank-1 where the model sees value (EV > 0) | −5.66% on 3,952 bets | −0.45% on 3,976 bets (90% CI −4.4% to +3.8%) |

- **Tuning helped.** The full model's loss to the market halved, and the shape block is the best fundamental block tested against BSP so far.
- **It is still not significant.** No block clears t = 2.
- **One strategy looks positive, but it is not a finding.** Rank-1 bets with EV > 0.02 returned +4.1% on 1,538 bets (90% CI −3.1% to +11.4%; halves +5.3% and +1.3%). It is one of seven thresholds tried, and its interval includes zero. The in-day rule's EV > 0.05 subset looked like this too, and failed the locked holdout.
- **Verdict:** the draw is priced, and so is most of the race shape.

**Iteration 18** (research-loop run 33): does the served price model forecast BSP better with the blocks? That bears on the early-price trade, which buys the morning price where the model says BSP will be longer or shorter.
- **Set-up:** the served recipe was fitted walk-forward with and without the blocks, and paired runner by runner (53,910 runners, 5,923 races, Sep 2025 to Mar 2026).
- **Error:** the mean absolute log error goes from 0.4566 to 0.4564. The paired difference is −0.0002 (90% CI −0.0010 to +0.0005), which the data cannot resolve.
- **Concordance:** winner-versus-loser concordance gets slightly worse (−0.0023, CI −0.0037 to −0.0010).
- **Verdict:** the default stands. **The blocks do not sharpen the forecast of BSP**, so they bring nothing to the early-price trade either.

## 5. Verdict, and what serving them would take

**Not worth serving.**
- The blocks measure the draw and the race shape better than the served features do (§3).
- But the closing market already prices what they measure: no information beyond BSP (iterations 17 and 19).
- The price model cannot use them to forecast BSP either (iteration 18).
- They stay as research blocks. They are the right tools for questions about the race itself, such as which courses and trips have a draw bias and how much a contested lead costs a front-runner. They are not model features.

The notes below stand in case a later test says otherwise.


Neither block is served, and the served model should not change during the forward CLV test. That test opens with 25 September's 06:00 run on the current model (`reports/preregistration_clv_forward.md`). Swapping the model mid-window would break it.

There are two ways to serve them:
- **After the window:** retrain with the blocks once the forward test has read out, if §4 finds they help.
- **In shadow now:** run the blocks in the 06:00 job and store their predictions next to the served ones, without changing what is served. The forward test stays clean, and the blocks build a forward record of their own.

Either needs the live path to build the blocks from the card plus history. The block functions already handle card rows: they take no comment and no result, get features from earlier days, and contribute nothing. What is missing is the wiring into `predict_bfsp_today.py` and a parity check.

## 6. Promoted to production features (24 September, later the same day)

At the owner's request, both blocks became standard inputs to the price model and so to its win probability (the served `predicted_win_prob_norm` is that model's normalised price). §5's verdict is left as written, because iteration 18's numbers have not changed. The promotion is measured afresh, as it will be served.

**What changed.**
- **Built on every row.** `CustomMetricsEngine.calculate_all` builds the shape block and then the draw curves (with the by-style cells) on every row of the history. Iteration 18 built them on the priced rows of the cached matrix only, which left non-runners without a price out of the field.
- **Served feature list.** `ALL_FEATURE_COLS` carries all 32 (`SHAPE_DRAW_FEATURES`), going from 501 to 533. The legacy win model's list (`PreRaceBuilder.get_feature_columns`) carries them too.
- **Guards.** The per-run columns (`RACE_SHAPE_POST_RACE`: a run's own style, finishing position and margin) join the post-race guard, so neither training nor serving accepts them.
- **The live path builds the block only for a model that reads it** (`needs_race_shape`). The model served today does not read it, so its 06:00 runs are untouched.
- **Research tooling.** `evaluate_oos.py --withhold shape_draw` fits without the block by exact name; prefix matching would also drop the older `pred_epf_norm`. The `shape` and `drawcurve` research blocks are retired and refuse to run, because rebuilding them there would replace the production values with priced-rows-only ones.

**Tests.** `tests/test_shape_draw_production.py`:
- A card with no results, margins, comments or prices, priced through the whole engine, gets bit-identical values to the same day with its results in.
- The control: the same blanking a day earlier moves more than ten of the card day's features.
- `tests/test_leakage.py`'s reversed-result test now covers all 32 features, because they are deployed.

**Measurement.** Iteration 25 is the served price model walk-forward without the block (the feature set served today) against with it:
- same quarterly folds from 1 Jul 2025 to the holdout;
- paired by runner;
- the early-price trade scored on both.

**Deployment.** Serving a model trained with the block is a separate decision. It would replace the model the forward CLV test opens on (25 September, 06:00), so it waits for the owner.

**Result (iteration 25, research-loop run 40).** Same 53,910 runners, 5,923 races and folds as iterations 18 to 24, and the early-price trade scored on both.

| | without the block (served today) | with the block | difference (90% CI) |
|---|---|---|---|
| Mean absolute log error of BSP | 0.4566 | 0.4566 | −0.0000 (−0.0008 to +0.0007) |
| Brier skill against the market | −0.0431 | −0.0438 | −0.0007 (−0.0014 to +0.0001) |
| Winner-vs-loser concordance | 0.7322 | 0.7305 | **−0.0017 (−0.0031 to −0.0003)** |
| Early-price rule, net CLV | +5.74% | +5.69% | |

- **No gain at any rank.** Rank 1 +0.0002, rank 2 −0.0012, rank 3 −0.0011, rank 8 and below +0.0001, none resolved.
- **Worse ordering.** The model ranks winners against losers slightly worse with the block, and that loss is resolved.
- **Decision rule:** the default stands.

**Training run** (train-bfsp run 30, not published):
- 533 features, and it passes the serving guard;
- the 32 new features carry 1.07% of the model's gain;
- the best is `pv_front_bias_lbs` at rank 130, and the median rank is 273.

The model uses them a little and gains nothing from them.

**Verdict:** built as served, on every row, the blocks still add nothing to the price forecast. §5's reading stands: the closing market already prices the draw and the race shape.

The production code can build them for any model that reads them. Whether the served feature list keeps them is the owner's decision; the evidence says a model without them is at least as good.
