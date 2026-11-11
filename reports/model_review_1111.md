# racing² - BFSP model review and QA, 11 Nov 2026

Scope: the served BFSP forecast (three LightGBM members, 1,006 features, trained to 13 Oct 2026) and the path that
serves it at 06:00. Sources are named on each line; the full feature-by-feature audit is `reports/bfsp_model_plan.md`
(10 Nov). A feature-level QA of the served path on real cards is running (`serving_feature_qa_1111`); its result will
be added below.

## 1. Verdict

The model is sound but improvable. Two changes are already proven and in hand; the biggest remaining gains are in the
serving path and in repairing features the model leans on most.

| Question | Answer |
|---|---|
| Can we improve it? | Yes. The market rating is worth -0.0067 on the locked holdout (ledger `iter116-market-rating-final`); the jumps stall fix -0.0118 over jumps (`card-as-table-check-1111`). |
| Any errors? | Yes: section 3. Four were fixed today; nine known faults remain in the features. |
| Any gaps? | Yes: section 4. The largest is that the 06:00 card is not what the model was trained on. |

## 2. Where it stands

| Measure | Value | Source |
|---|---|---|
| Error vs BSP, model alone (test) | 0.377-0.400 | `reports/race_book_model.md` |
| Morning market alone | 0.350 | same |
| Closing model (market + ours) | 0.302 | same |
| Live 06:00 error, 4-9 Nov | 0.3910 | run 42980435893 |
| Candidate with market rating, holdout Apr-Sep 2026 | -0.0067 vs served (CI -0.0074 to -0.0059) | ledger `iter116` |

## 3. Errors

### Fixed on 11 Nov
| Error | Effect | Fix |
|---|---|---|
| Betfair "Bangor-on-Dee" vs card "Bangor" | No Bangor runner ever traded | PR #307: alias, plus a fallback match by time and horse |
| Jumps stall filled from the saddlecloth number | 06:00 jumps error +0.0118 | PR #311: `CARD_AS_TABLE=stall`, live 12 Nov |
| Live record not re-linked to late results | 24% of runners unlinked in research queries | PR #308: nightly re-match (76.0% to 90.6%) |
| Stream recorder stopped at 10:30 | No ladder data near the off | PR #305: records to 21:30 |

### Open, serving path
| Error | Evidence | Next |
|---|---|---|
| Card weight is before the jockey's claim | Differs for 22% of runners; fixing it made the Flat +0.0013 worse (se 0.0008) | Unresolved: test the weight alone on more days before any change |
| Card lacks race class for 27% of runners (pattern and Irish races) | `card-vs-results-1110` | Derive the class on the card as the table has it; test as the stall was |
| Morning going differs from the final going on 27% | same; swap test -0.0003 | No action: no measurable cost |
| Evening-before job does not use the stall fix | `predict-evening.yml` | Match it to the morning job |
| Served model is 4 weeks old (trained to 13 Oct) | `data/models` meta | The candidate is trained to 9 Nov; set a retrain schedule from the model-age curve |

### Open, feature faults (from the 10 Nov audit; none leaks the result)
1. Within-race features are built on a different field live than in training (57% of gain reads them).
2. Unrated official rating (0) is never masked.
3. Going read wrongly by substring order (Good to Soft read as Soft).
4. A non-finisher is rated as running to its mark; this feeds the Kalman rating.
5. Non-finishers dropped from win, A-E and NFP means.
6. The gelding angle (not on the 06:00 card) sits inside three served intent features.
7. Trainer/jockey track rates, pace index: wrong denominator.
8. `form_delta` subtracts a ratio from a strike rate.
9. Prize: missing counted as 0; `RACE_WPMRF` is prize x field size.

### Cosmetic
- `verify_model.py` prints "objectives l2 + l2 + l2" whatever the members are (run 60-62 logs).
- The race-xent member's training summary prints R² -128.8: its raw output is not a log price, so the line is
  meaningless, not a fault. Its verify error (0.3606 in the blend) is the real figure.

## 4. Gaps

| Gap | Why it matters |
|---|---|
| One model family, one seed, a hand-set gate | Three-seed averaging (-0.0026) and a learned gate (-0.0014) each gained and were not served |
| No importance for the dml and rx members; no permutation or SHAP importance | We rank features on one member's gain only |
| About 250 duplicate or dead columns | Slower fits, noisier importance |
| Shrinkage constants typed in, none fitted | Small-sample rates (trainer, jockey, track) are noisier than they need be |
| Windows count runs, not days | A run 400 days ago weighs the same as one 14 days ago |
| No weather, GoingStick or full sectionals in the served features | The top syndicates have these |
| 18 courses have not raced since 1 Oct | Their Betfair names are unchecked; the fallback match covers them |

## 5. New feature blocks to work on

Ordered by expected value. Each is a drop-in block (`model/blocks/`), screened on 11 folds with a placebo arm, ledger
entry first.

| # | Block | What it adds | Why it should work |
|---|---|---|---|
| 1 | Within-race repairs (batch A) | Gap to the best rival, logs before z, rank as share of field, softmax share | Repairs the family carrying 57% of gain; robust to non-runners |
| 2 | Market plus-minus | Trainer and jockey effects on what the horse's market rating leaves over | Builds on the market rating, the largest gain on record; football's plus-minus idea |
| 3 | Franked form | How the rivals a horse met have rated since (change in their market rating) | Form lines today read results; the market's later view of beaten rivals is new information |
| 4 | Handicapper lag | Market rating against the official mark, and its trend | Finds horses the market rates above their mark before the handicapper reacts |
| 5 | Suitability | Horse x going x trip x course record, shrunk hierarchically with fitted constants | Replaces raw counts on small samples |
| 6 | Days-decayed form | One window weighted by days since each run, with count and spread | Windows by run count ignore layoffs |
| 7 | Race-standard performance figure (batch D) | One standard per race, pounds by going and code, soft cap | Removes the anchor on each horse's own mark |
| 8 | Prize and class, deflated (batch C) | Prize by year x code x class in logs; prize actually won | Prize features carry 14.6% of gain and are raw today |
| 9 | Non-runner risk | Trainer and going-change withdrawal rates for today's field | Lets the forecast price a field that will shrink; within-race features move with it |
| 10 | Weather and going change | Rain since declarations, forecast going against preferred going | New data, already in the metrics rebuild; needs a card-time source |

Not proposed again (tested, no gain): pedigree blocks, pace and draw rebuilds, result-based Elo, comments, recency
or profit weighting, other losses.

## 6. Order of work

1. Shadow week of the market-rating candidate (12-20 Nov), then the serving decision.
2. Serving QA result (this run): fix each card-side error behind a flag, tested as the stall was.
3. Screens: batch A, then market plus-minus, franked form and handicapper lag together with the market rating in
   the base.
4. Data faults 2-9 as one batch.
5. Combiner (learned gate, seeds) on the saved out-of-sample files: no new fits.

## 7. Serving QA result (added 17:30 UTC)

Source: run 43050875949 (`research/queries/done/serving_feature_qa_1111.py`, ledger `serving-feature-qa-1111`): the
served model on the real 06:00 cards of 7 and 9 Nov (stall fix on) against the same code on a perfect card; 658
runners, 68 races.

| Measure | Value |
|---|---|
| Error vs BSP, served card | 0.3648 |
| Error vs BSP, perfect card | 0.3613 |
| Gap | +0.0031 (se 0.0032): not resolved |
| Features identical on every runner | 493 of 1,006 |
| Features differing on over half the runners | 8 |

With the stall fixed, the 06:00 card prices within noise of a perfect card (the gap was about 0.008 before).

What still differs, by impact (difference x share of gain):

| Feature or family | Runners differing | Cause |
|---|---|---|
| `hca_wt_vs_mark` (weight against the mark) | 31%, by 0.43 sd; nine times the next | Card weight is before the claim |
| Prize-money ranks and z-scores (`rPMW*`, `rr_PMW*_z`) | 14-27% | The field still holds later non-runners |
| Going features | 18-44% | Morning going, not the final going |
| `weight_vs_avg` | 86% | Race average moves with the field and the claims |
| `race_class_num` | missing for 5.6% | Card has no class for pattern and Irish races |

Form windows, form variants, form lines and the draw-by-track family are identical: the form the model reads is
served exactly.

Next: the weight after the claim tested apart from the stall (`card-weight-check-1111`).
