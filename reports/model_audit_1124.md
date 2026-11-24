# racing² - Horse model and trading audit, 24 Nov 2026

The owner, 24 Nov: "i am concerned about the performance of our model. Do a thorough review and audit."

## Result

1. **The model has not got worse.** Its error against the BSP is flat through the autumn, the winter card and the three
   model changes (0.4015, 0.4050, 0.3791, 0.4136 by period). The 08:00 market is closer to the BSP than the model in
   every period and code (0.317 against 0.363 in the autumn). Source: ledger `audit-1124-model-vs-market`,
   run 45403610721, `outputs/audit_20261124/model_vs_market_1124.txt`.
2. **What fell is the edge the trader turns into money, and most of the fall is in the top-ups.** Top-ups are 73-78%
   of the stakes and realise 2.8 to 5.4 points less CLV than the closing model expects, in every period; first backs do
   not fall short (ledger `week-drop-1120`, `week-drop-1123`, `phase5-reductions-1119`). The audit found the mechanism:
   a top-up reads the trader's own take from the book as the market shortening (MC-1, EL-04), and the closing model has
   no clock, so it is over-confident after 07:00 (DP4).
3. **Since 15 Nov the weak spot is all-weather, not jumps.** All-weather -1.43% on GBP7,413 against +6.73% in the
   autumn; jumps +5.07% on GBP13,966 (diag_performance, from `week_drop_1123.txt`). It starts with the stack blend.
4. **Crossing the spread costs about 10% against the SP on an average runner at 08:00**, and only the horses the model
   rates far shorter than the market (gap over 0.5 in log terms) clear it: +8.0% (+3.9 to +12.0), +2.1% (-4.6 to
   +7.9) from 9 Nov (`audit-1124-model-vs-market`). A resting (maker) order has never been tested on the horses.
5. **Two high-severity execution faults and a kill switch that fails open** were found and are fixed in this pull
   request (section 4). None has a recorded incident.
6. **No leak was found anywhere.** 28 readers covered every feature module, the training, evaluation, serving, blend,
   closing model, execution code and the recorded evidence; every claimed defect was then checked by a second reader
   trying to refute it: 96 confirmed (2 high, 34 medium, 60 low), 2 uncertain.

## 1. How the audit ran

| Part | Readers | What each returned |
|---|---|---|
| Features | 18 (custom_metrics in three parts, bfsp_features, primitives, pace, draw, shape, connections, pedigree, intent/freshness/windows, market memory, debut/time, ratings/lines, relative form, angles, other blocks, machinery) | every family: what it measures, the method, logic check on toy frames, the leading alternative, what the ledger already tried, priority |
| Models | 5 (training, evaluation, serving, blend, closing model) | method, defects, improvements |
| Execution | 3 (decision logic, exchange layer, recorded evidence) | defects, improvements |
| Diagnosis and operations | 2 | the performance fall, the job schedule |

Every non-low defect went to a sceptic who tried to refute it and rated its severity, whether it touches the served
model, and whether a fix is safe to merge now. No real data was on the review machine; every toy figure is a logic
check, not a result. Full output: workflow run wf_89d8e2c8-e68 (126 agents).

## 2. Features

The served price is three LightGBM members on 1,263 features. Gain is concentrated: the market rating (mr_, 14.9%,
mr_share alone 13.2%), the debut forecaster (db_, 15.5%), relative form (23.9%), the legacy engine (about 30% with its
ranks). Source: `outputs/horse_v0_20261118/phase2_feature_audit.csv`.

| Family | Served | Finding | Better approach | Priority |
|---|---|---|---|---|
| Market rating (mr_) | yes, 14.9% | Update gain too large by about N/(N-1); fixed origin for all-newcomer races; one reading noise for a 1.5 shot and a 300/1 shot; Flat and jumps share nothing; q and R re-chosen on a 3x3 grid each build, not pinned (FMM-1 to FMM-3). Live field includes late non-runners, training does not (FMM-12, medium) | exact update, price-dependent noise, robust update, pinned q/R; measure FMM-12 with a serving parity check | high (largest single input) |
| Handicap angles (hca_wt_vs_mark, rank 7, 1.72%) | yes | Top weight and top rating taken from different runners, so one claimer shifts the field (FA-01); trained on the weight after the claim, served before it (FA-02) | race mode of (weight - rating) as the reference; train on the card weight | high |
| Pedigree Bayes (pb_, feeds db_x 11.9%) | yes | Prior strength K biased to its upper bound with unequal run counts (PED-1); first foals' damsire missing on the 06:00 card (PED-4) | one-way ANOVA moment estimator; dam to damsire table from all years | medium |
| Category codes (track_cat etc.) | yes | Trained on the 2018+ numbering, served on the 2021+ numbering: 11 courses Tramore-York priced as the course before (BF-1/MT-1) | store the vocabulary the codes were made with; check it in verify_model | medium (small gain, certain fault, needs retrain) |
| Connections rating (cr_jk_delta) | yes, 0.34% | The decayed-mean helper caps decay at 1,000 half-lives, so "the old jockey's latest rating" is a long-run average (FS-1/CONN-1) | exact last reading | low |
| DSLR (12 features) | yes, 0.08% | Under pandas 3 the date is microseconds, read as nanoseconds: the value counts 1,000-day boundaries (CMB-1). Same in training and serving, so no skew | fix in a retrain only; the freshness block carries the real signal | low |
| Going code | yes | Substring match in dictionary order: Good to Soft reads Soft (CMB-3) | exact map | low |
| Draw (60 columns) | yes, 2.46% | Jumps stall 0 read as a real stall in about 40 columns; card stall against post-withdrawal field (FD-02, FD-04); priors 60/40/30/20 runners against the tuned 2400/1200/640/320 | prune to the tuned curves | low (every draw rebuild added nothing) |
| Pace (91 legacy) | yes, 1.3% | Legacy parser takes the most forward phrase anywhere (PACE-1, known since 24 Sep) | the repaired parser (already in pace_v2) | low |
| Legacy engine (NFP, RB, WIV, WOA, OFS...) | yes | 37 served columns are exact copies of others (11 RB, 13 WOA, 13 OFS); no shrinkage; non-finishers dropped | prune; the leading method is already served in a block | low |
| Time figure (tf_) | yes, 0.25% | Going allowance clipped at 1 s/f on a uniformly slow card (FDT-4) | median of unclipped deviations | low |

The full table of about 350 families, with method, logic and the alternative for each, is in the workflow output.
The ledger has already tested most alternatives; where it has, the result is cited there.

## 3. Models and blend

| Finding | Severity | Status |
|---|---|---|
| The served price is short of the BSP by construction (mean ln(pred/BSP) -0.1075): forcing the book to 1 makes it 1/E[1/BSP], while the error metric rewards the median. A constant shift moves the error by 0.0075 with no probability changed; the owner's rule cannot see a price level (MT-2, B1) | medium | reporting added (this PR); a level test to register |
| The 0.0007 bar holds no fit noise: marginal passes (iter126 -0.00072, iter129 -0.0009) are within seed noise (ME-01) | medium | fold-clustered interval added (this PR) |
| The deflated Sharpe took its spread from whatever Sharpes were passed while N came from the ledger, so it could deflate by almost nothing (ME-02, B7) | medium | floored at the null variance (this PR) |
| The pre-publish gate verifies a hybrid and silently skips the comparison with the served model since boosters became .lgb.xz (MT-3, MS-2) | medium | the comparison now runs and an unloadable served model fails the gate (this PR); pricing each member on its own still to do |
| The served recipe exists only in dispatch arguments (MT-5) | medium | noted |
| Blend weights 1,1,1,0.5,0.5,0.5,0.5,0 set by hand on 28 Sep for members that no longer exist; a cross-fitted gate priced 0.0014 better then (iter105) | known (M4) | the members correlate 0.992-0.995: at most 0.001-0.003 to gain |
| The "never under our predicted BFSP" guard reads the 03:00 price, not the field that is left after non-runners (B3) | medium | lost bets only; registered test needed |

The recipe itself (targets, gradients, inversion, float32) was checked and is correct. It has been searched hard
already (ledger iter40-105).

## 4. Execution

| Finding | Severity | Status |
|---|---|---|
| A placeOrders reply with status TIMEOUT is booked as "not placed": a back or an SP lay can be placed twice, or a placed lay forgotten (EX-01, EL-01) | high | fixed (this PR): any reply not a clean SUCCESS or FAILURE goes to Betfair's own order record (re-read for 15 s); unconfirmed means no second order and new bets stop for the day |
| The kill switch reads any S3 error as "not stopped", with no log (EX-04, EL-07, OPS-3) | medium | fixed (this PR): only "no such object" clears; three unreadable checks in a row stop new bets; the GBP1,000,000 settled stop-loss is still inert (the owner's number) |
| Betfair login and logout have no timeout (EX-05) | medium | fixed (this PR) |
| A transient S3 error at restart is read as "no ledger" (EX-03) | low | fixed (this PR): retried; if still unreadable, no new backs that day and the day's ledger is never overwritten |
| No in-day alert when the trader stops itself or leaves a back unhedged (OPS-2) | medium | fixed in the watchdog (this PR) |
| A trader that dies inside the live job is invisible to the watchdog until 21:30 (OPS-1) | medium | fixed (this PR): a non-zero exit before the end of the session ends the run as failed, so the watchdog restarts it |
| The old order path (execute.yml every 30 minutes with DRY_RUN false, settle.yml nightly) was still scheduled in the file, disabled only by hand (OPS-9) | medium | fixed (this PR): schedules removed, manual runs dry by default |
| Predictions fail silently; serving fills missing features with NaN with no guard; results freshness is a log line; an unseen category is read as a real value (OPS-6, MS-3, MS-4, BF-3) | medium | fixed as alerts (this PR): a serving-checks line and sidecar, an alert on the day's issue; no served value changes |
| The scheduled settle takes its day from the wall clock (OPS-4) | low | fixed (this PR) |
| No reconciliation with Betfair during the session after a lost reply (EL-02) | medium | to do: periodic sync |
| A top-up reads the trader's own take as the market shortening (MC-1, EL-04); the expected CLV is over-confident after 07:00 and on later fills (DP4) | medium | registered replay needed: hold the pre-take read for top-ups; a clock term in the closing model |
| The cap of 3 top-ups was adopted on a floor-biased table; the only replay of fewer top-ups lost (EE-3) | medium | scored in the 100-strategy test (live-rule variants) |

The normal order path is clean: matched +0.17% over the best back read, 152 EXPIRED of 7,565 backs (ledger
`execution-audit-1021`). The non-runner reductions (-4.66 points) are not an avoidable leak: three tests show
Betfair's factor is fair.

## 5. Strategies: 100 tested

The owner asked whether we use the best strategy and for up to 100 others, led by the data. 189 were proposed by eight
designers, 100 kept and registered before the run (ledger `strat100-1124`), built with tests and two adversarial
reviews, and scored on three data sets. Full table: `outputs/audit_20261124/strat100_summary.md`.

**No strategy passed the owner's rule**: none beat the live rule in pounds a day with a deflated Sharpe of 0.95 or more.

| | CONFIRM return a pound | 90% interval | Net GBP, 1-23 Nov |
|---|---|---|---|
| The live rule, replayed (S001) | +4.69% | +3.37 to +6.09 | +2,184.50 |
| First backs only, no top-ups (S013) | +7.57% | +5.88 to +9.26 | +1,144.95 |
| Top-up bar 7% on the real fills (S015) | +7.47% | +5.41 to +9.56 | +1,891.43 |
| Bar 8% (S003) | +8.43% | +5.93 to +10.96 | +1,364.72 |
| Target scaled by hours to the off (S019) | +4.76% | +3.35 to +6.25 | +2,356.18 |

Tighter rules earn more a pound but less a day. This agrees with the audit's finding that top-ups dilute the edge.
The next step is a registered replay of a top-up bar and of S019's staking, not a live change. Lay books, place
markets, momentum, in-running exits and model-free controls all failed; in-running exits lost 25-33%.

## 6. What to do next, in order

1. **Top-ups** (the largest share of the fall): replay through the trader's code, on the winter card, (a) top-ups
   tested against the read before our own take, (b) a closing model with time to the off, (c) caps 0-3. Partly in the
   100-strategy test (ledger `strat100-1124`).
2. **Resting orders on the horses**: a back-only maker order scored on the horse stream record (from 10 Nov).
3. **All-weather under the stack blend**: a retro-shadow of the old blend on 15-23 Nov and settled CLV by code, first
   back or top-up, and hour.
4. **Feature correctness as one gated arm**: hca_wt_vs_mark reference and card weight, pedigree K, the category
   vocabulary, cr_jk_delta, going code, DSLR. Each changes served values, so they go through the research loop under
   the owner's rule, not straight into the served model.
5. **Price level**: a registered test of a median-unbiased price level (B1, MT-2), judged with the level now printed.
6. **Prune**: the 37 duplicate columns and the legacy families whose leading method is already served.

Every claim above traces to the workflow output, the ledger tag or the file named beside it. The locked holdout (from
28 Mar 2026) was not read.
