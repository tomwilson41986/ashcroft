# MAP: repo_blocks

# MAP repo_blocks - what is already built and tried (racing², 11 Nov 2026)

Key: figures are the paired change in mean absolute log-BSP error vs the base (negative = better). "L:" = tag in /home/user/ashcroft/reports/research_ledger.jsonl ("iterNN" is the tag's prefix, e.g. iter48 = iter48-served-bookings). Block code = /home/user/ashcroft/model/blocks/<name>.py. Served list = data/models/bfsp_model_meta.json `drop_in_blocks` (14 blocks, 1,006 features, 3 members). Noise bar: +-0.0025 quick recipe (docs/FAST_LOOP.md:114), 0.0007 on 11 folds (L:iter114).

## 1. BLOCKS (46)

### Served (14)
- bookings (bk_) - jockey's/yard's past runners as the market rated them (BSP share, decayed, shrunk); today's jockey vs the horse's last 3 riders - L:iter48 -0.0041.
- connection_windows (cw_) - trainer/jockey last 20/100 runners: NFP, lb beaten, wins minus BSP chances, race rank - L:iter68 -0.0040.
- debut_market (dm_) - how the market priced the yard's debutants, 2nd and 3rd-4th runs, by code; field's debut share - L:iter80 -0.0032 (maidens -0.0067).
- exposure (ex_) - best/slope of performance figures; expected next-run gain by runs x age x code cell, trainer, sire; headroom vs mark - with time_figure + race_relative_wide ("all3") L:iter42 -0.0045.
- form_lines (fl_) - rivals from last 1/3 races: next-3-run win/place rate, wins minus BSP chances - L:iter59 -0.0038.
- form_variants (fv_) - 77 NFP/lengths variants: decay in runs and days, field-size weights, EB shrink to trainer, DNF-as-last, log/per-furlong/relative lengths - L:iter56 -0.0025.
- handicap_angles (hca_) - weight vs mark, last-win margin in lb, unchanged mark after a win, mark vs last winning mark - nothing alone; with kalman + travel -0.0017 (L:iter96).
- head_to_head (h2h_) - earlier direct meetings among today's runners: share ahead, lb margins - L:iter41 -0.0019; all3+h2h -0.0053 (L:iter42).
- kalman (kr_) - local-level Kalman rating on the performance figure, its sd, vs mark, race z/rank - nothing alone (L:iter96).
- race_relative (rr_) - within-race z-score and gap-to-best of 30 top inputs - L:iter35 -0.0074; L:iter36 -0.0078.
- race_relative_wide (rw_) - same for 62 more ranked inputs; field top-two gap and sd - alone -0.0012 unresolved (L:iter37).
- sectionals (sc_) - past finishing speed vs field, faster-than-placed, early/late pace, stride length - L:iter113 -0.0008 in the blend; zero once files stale (L:iter111).
- time_figure (tf_) - clock figure: course standard, day going allowance, lb beaten, weight - alone -0.0005, lifts Brier/concordance (L:iter37).
- travel (tv_) - km yard-base to course vs yard's usual, cross-sea raid, track share - L:iter96 -0.0012.

### Passed, waiting
- market_rating (mr_) - every past BSP as a multinomial-logit strength, race level from rivals' ratings, Kalman-filtered; softmax share today - L:iter114 -0.0080, L:iter115 -0.0064, holdout L:iter116 -0.0067; retrained, shadow from 12 Nov (L:market-rating-retrain-1111).
- jumps_form (jf_) - fall/pulled-up rates, first time over the obstacle, market view by obstacle, 3-mile completion - registered, unread (L:iter117-nh-vs-flat).

### Built, not served
- inrunning (ir_) - log in-play low/high vs BSP of past runs; beaten runs traded <= 2.0 - main -0.0008 (L:iter109), blend -0.0002 unresolved (L:iter110).
- tf_pace (tp_) - Timeform pace strength/contest, finishing-speed %, pace-map ideal position; finish by pace type - L:iter111 -0.0004; no live source.
- sales (sl_) - log pre-racing auction price by sale type - L:iter107 cancelled (snapshot look-ahead); screen 1.6% SSE cut, t 2.1 (L:sales-served-screen-1005).
- collateral (cl_) - implied lb margin through common opponents - -0.0015 beside form lines (L:iter66); nothing beside connection windows (L:iter68).
- shrunk_rates (sr_) - empirical-Bayes cell rates, trainer/jockey/sire/horse - 0.0000 (L:iter38), -0.0013 (L:iter54).
- cond_form (cf_) - figure/NFP/market view in today's trip, going, course, handicap, headgear vs overall - L:iter57 -0.0002.
- future_form (ff_) - grid: last 1/3/5 races x rivals' next 1/2/3/5 runs (win rate, A-E, NFP, figures) - L:iter90 -0.0003.

### Retired
- career_before (cb_) pre-2021 runs joined - L:iter83 -0.0002.
- comments (cm_) trouble/slow start/keen/wide/kind-ride classes - L:iter45 +0.0019.
- connection_grains (cg_) 14-day, code, course, pair - L:iter69 -0.0008 beside cw.
- connection_relative (cwr_) cw as z/gap - L:iter71 +0.0003.
- dam_line (dam_/sib_/family_) - L:iter49 +0.0008; L:iter52 -0.0001.
- draw_v2 (d2_) draw by course x trip x going x stall placement on NFP/lb/A-E - L:iter35 -0.0004; L:iter54 -0.0009.
- elo - multi-runner Elo, two pools - L:iter55 +0.0001.
- form_lines_ab (fa_) rivals split ahead/behind - L:iter60 -0.0007.
- form_lines_close (flc_) rivals weighted exp(-lengths/3) - L:iter65 -0.0012 unresolved.
- form_uplift (fu_) rivals' later figure/mark/market change - -0.0046 on price but no outcome information, a resolved loss vs the market beside form lines (L:iter62); figures only -0.0003 (L:iter64).
- hrb_extras (hx_) same surname, seconds share, last-race winner's BSP, non-completers, max field won, same race last year - L:iter109 +0.0001.
- market_history (mh_) Betfair-vs-industry SP spread; win/place SP ratio - L:iter98 +0.0006.
- nfp_z (nz_) field-size z-scored placing - L:iter108 +0.0002.
- pace_v2 (p2_) early-position windows, course/trip EB, leader gap, pace heat - L:iter35 +0.0009.
- pedigree_market (pdm_) market's view of sire's debutants, damsire, dam's foals - L:iter84 -0.0006.
- placebo (pb_) noise control - +0.0022 (L:iter50), +0.0007 (L:iter114).
- quant (qm_) Sharpe, signed chi, A/E vs BSP; form volatility/drawdown - L:iter106 +0.0001.
- race_relative_new (rn_) - L:iter44 +0.0007.
- rank_fix (rfix_) - L:iter41 0.0000.
- seasonal (sn_) same months, earlier years - L:iter98 0.0000.
- sire_apt (aptitude residuals, nick) - L:iter52 -0.0006.
- sire_windows (sw_) - L:iter73 +0.0017.
- stable (st_) stablemates, yard's first jockey - L:iter47 +0.0018; L:iter52 -0.0007.

## 2. IDEA FAMILIES TRIED AND REJECTED
- Any information beyond BSP on the winner: 726 features, six fundamental blocks, dLL -0.119 mnats (L:iter8-fundamental-no-blandford); Benter two-stage worse (L:iter6); again L:iter36-served-6000-candidate-outcome, L:benter-blend-1020, L:small-book-late-1111.
- Pedigree, four readings: sire aptitudes, dam line, sire windows, pedigree-as-priced (L:iter49, iter52, iter73, iter84).
- Pace and draw rebuilds: run-style projection, race shape, position value, draw curves, draw-by-style -0.0002/-0.0000 (L:iter18, iter25; reports/draw_pace_rebuild.md:244); draw_v2, pace_v2; trained leader model AUC 0.78, A/E ~1.0 (L:leader-forecast).
- Result-based ratings: Elo (L:iter55); nfp_z; quant metrics.
- Comments (L:iter8, iter45, iter52).
- Entity mispricing vs BSP: trainer/jockey/sire/horse/course x draw residuals do not persist (L:iter11-ae).
- Same-day connection form: dev +20.2%, holdout -18.7%, rejected (L:confirm-holdout).
- Intent angles vs BSP (headgear, gelding, new yard, class drop): none (L:intent-survey); calendar/context (L:context-survey); race-name conditions (L:race-name-conditions); A-E cells, black type (L:travel-residuals-0927).
- Other markets' prices: place BSP, industry SP, FLB (L:iter5-markets); market_history; late steam beyond BSP (L:steam-beyond-bsp); market-only drift (L:market-only-drift-1002, open-close-horses-1101).
- Weighting: recency +0.0012/+0.0059 (L:iter72); exp(-days/365) +0.0011, profit sqrt(1/BFSP) +0.0147 (reports/h2h_summary.md:21-23); share weighting (L:iter104).
- Losses/targets: L1, Fair (L:iter74); Huber alone -0.0003 at served recipe (L:iter76); log-BSP target (L:iter85-served-968-partners); native categoricals (L:iter92); within-race stretch (L:within-race-stretch); bigger/faster trees (L:iter40).
- Field-relative re-readings of newer blocks (rn_, cwr_, fa_); deeper form-line weighting (flc_, ff_).
- Whole careers as features (cb_); more training rows 2018-20 -0.0005 unresolved (L:iter100).
- Timeform same-day ratings: leak, banned (L:blandford-value-vs-missing); lagged figures no recent edge (L:blandford-revisions).
- Place market: outcome model, morning CLV (L:place-model-stage3-1003, place-clv-forward-1002); in-play offsets (L:edge-inplay-offsets-1002).
- Residual screens as a ranking tool (L:market-season-residuals-0928).
- Closing-model variants and refits (L:closing-model-v2-0930, closing-refit-replay-1109); removal tilt (L:removal-tilt-replay-1109); neural net on the move (L:move-model-1021).
- Recorded as equivalent, not built: Cox = Plackett-Luce = exploded logit; Elo/Glicko = Kalman; credibility = EB; kriging = GP; copulas, Hawkes, RL staking skipped (RESEARCH_FRAMEWORK.md:174-181, 288).

## 3. IDEAS THAT GAINED
- Target logit of race-normalised probability -0.0016 (reports/h2h_summary.md:24).
- Intent, card-safe -0.0052 (L:iter24); freshness -0.0049 (L:iter26); form windows -0.0073 (L:iter29).
- Within-race z/gap (Lessmann-Sung-Johnson) -0.0074; form lines; connection windows; bookings; debut market; travel (section 1).
- Recipe: more rounds -0.0050 (L:iter31); lr 0.02, leaves of 500, extra trees + Huber, best single booster 0.4001 (L:iter91).
- Diversity: L2 + Huber average -0.0026 (L:two-loss-average); within-race demeaned-log partner, pair 0.3981 (L:iter94).
- Complete careers (features from 2018) -0.0012 (L:iter99-served-968s5xh-all3-history-2018).
- race_xent member gated by rank: -0.0039, rule +12.23% vs +10.38% (L:iter105) - served.
- Sectionals served 24 Oct (L:serve-sectionals-1023); jumps stall card fix -0.0118 on jumps (L:card-as-table-check-1111).
- Waiting: market rating; three-seed average -0.0026 (L:iter81) and learned gate -0.0014 (L:iter105), not served (reports/model_review_1111.md:68).
- Trade: closing model 0.302 vs market 0.350 (L:race-book-0930); edge is before 10:00 UK (L:live-by-hour-1104).

## 4. OPEN LEADS THE RECORD NAMES
- reports/model_review_1111.md:83-92: within-race repairs; market plus-minus (trainer/jockey on market-rating residual); franked form (rivals' market-rating change); handicapper lag; hierarchical suitability; days-decayed form; race-standard figure; deflated prize; non-runner risk; weather and going change.
- L:iter117-nh-vs-flat; L:card-weight-check-1111; L:stream-late-signal-1111 (no data yet); L:desk-lay-book-1113.
- Untested because never stored: Timeform tips/flags, pace map, dated entries, GoingStick (reports/external_data_review.md:88-104).
- Sales from a point-in-time source; tf_pace live source; inrunning in the blend.
- Betfair's catalogue forecast price as a veto, +20% CLV x stake on 7 days (L:move-model-1021); expected-reduction term (L:reductions-1021); late over-confidence of expected CLV (L:closing-model-by-hour-1007).
- Low draws, soft-ground Flat sprints, A/E 1.095 z 3.0, untested (L:pace-draw-bias).
- Race ABM built, never calibrated or run at scale (RESEARCH_FRAMEWORK.md:551-553); mixed/nested logit, exploded-logit depth, change-points "next" (RESEARCH_FRAMEWORK.md:171).

## Blind spots of this map
- Block docstrings were read to 1,500 characters each (tails read for 12 of the longer ones); FEATURES lists were not enumerated feature by feature, and no build() code was read, so a block's line describes its documented computation, not a code audit.
- Ledger: all 676 tags were listed, but result/design fields were read truncated (roughly 260-650 characters) for most entries; only about ten entries were read in full. Daily operational rows (live-day, replay, strategy-replay, model-quality, marks-load, tote-compare, all-markets-scan, blandford-sync, changes, staking, night-before) and validation/verify/dry-run/parity/deploy rows were not read; any research finding recorded only inside them is missing.
- "Served" status is taken from data/models/bfsp_model_meta.json (drop_in_blocks and feature prefixes of the main member); the dml and rx members' feature lists were not checked separately. Whether the market-rating candidate has since been served could not be confirmed beyond L:market-rating-retrain-1111 (shadow from 12 Nov, not published).
- Engine-level (non-drop-in) features in model/custom_metrics.py, form_windows.py, intent_features.py, freshness_features.py, shape_form.py, race_shape.py, draw_curve.py, financial_features.py, market_features.py, primitives.py, pedigree.py, the ABM and the research toolkit modules were not read; they appear only where the ledger or reports describe them. The 19 proprietary metrics and the ~187 engine features are therefore not itemised here.
- reports/draw_pace_rebuild.md sections 3.1-3.4 and 6 (tests) were skimmed, not read; RESEARCH_FRAMEWORK.md sections 8-9 and 13 were not read. reports/bfsp_model_plan.md, reports/hrb_system_builder_review.md, reports/metric_audit.md, reports/race_book_model.md and STAKING_REPORT.md were not opened.
- No ledger entry mentions the race ABM (0 matches for "abm"); its status (built, uncalibrated, never run at scale) rests on RESEARCH_FRAMEWORK.md:551-553 alone and may be out of date.
- Greyhound, football, tennis and Australian work appears in the ledger but is outside this map except where it bears on horses; the morning models (steamers, quick returners) and the paper desk arms were not mapped.
- Nothing was run against the database or S3 (unreachable); no figure was recomputed - every number is quoted from the cited file or ledger tag. No web search was used.
