# MAP: repo_models

# MAP repo_models - what this repository has already built or tested

Status: S served, T tested, B built but never measured on real data, P pending. `L:` = tag in reports/research_ledger.jsonl. "Error" = paired mean |log(forecast/BSP)| (negative is better). RF = RESEARCH_FRAMEWORK.md.

## 1. Target and objectives (LightGBM GBDT, one row a runner)
- log(BFSP), L2 - T, original target; retired (reports/h2h_summary.md; L:iter85-served-968-partners).
- Logit of race-normalised implied probability - S main member; -0.0016 vs log(BFSP) (model/bfsp_model.py:450-459).
- Within-race demeaned log price - S partner; alone +0.0004 n.s., pair 0.3981 (h2h_summary.md; L:iter94-served-968-slow-dml-partners).
- race_xent: within-race softmax fitted by cross-entropy to BSP book shares - S gated member; price +0.0104 worse alone, trade rule best +12.60% (model/bfsp_model.py:392-413; L:iter105-served-gated-race-xent).
- Profit-weighted asymmetric loss (weight sqrt(1/BFSP), 1.5x under-prediction) - T, +0.0147 (h2h_summary.md).
- Exponential recency weights - T, +0.0011 (h2h_summary.md; L:iter72-screen-recency-weight). Book-share row weights - T, retired (L:iter104-served-share-weight-race-xent-seeds).
- Huber, L1, Fair losses - Huber S inside the extra-trees main member: -0.0029 on the screen, no candidate at learning rate 0.03, best single booster at the slow recipe; L1 +0.0112 and Fair +0.0064, T (L:iter74-screen-loss, L:iter76-served-958-huber, L:iter91-served-968-slow-xt).
- Binary win classifier, race-normalised, isotonic calibration - legacy B (model/probability_model.py:33; model/calibration.py).
- Stage F: market-free conditional logit / LightGBM race-softmax on the winner - T: R2 0.0740 vs market 0.1717 (RF 10.5).
- Outcome model: race-softmax boosted from a market-only conditional-logit offset (ln p, ln p squared) - T: -0.087 mnats/race on 752 features (L:iter36-served-6000-candidate-outcome). Linear version on in-day connection features: dev +20.2%, holdout -18.7%, rejected (L:confirm-holdout).
- Exploded logit depth 2 inside it - T, -0.621 mnats, dropped (same tag).
- Move models, target ln(morning/BSP) or CLV: walk-forward ridge (model/clv.py:62) - B; linear stage 2 with interactions - T, no gain on the fixed rule (L:drift-stage2); ridge, LightGBM, MLP on 42 live-book features - T, MLP worst, vetoes failed the replay (L:move-model-1021, L:combo-replay-1021).
- Steamer model: LightGBM regression of log(morning/BSP) on card and history - S (half stakes); top 5% A/E 1.155 vs morning price (reports/strategy_research_2026-11-04.md:124-143).
- Leader classifier (LightGBM) - T: AUC 0.795, backing it A/E 1.006 (same report:180-211).

## 2. Members and combination
- Recipe grid (learning rate, leaves, min leaf 50-1,000, rounds 3,000-16,000, feature fraction, path smoothing, native categoricals, L2) - T; slow recipe served (L:iter40-served-recipe, L:iter87-served-968-slow-regularisation-complete, L:iter92-served-968-slow-xt-settings).
- Extra-trees (randomised split) boosters - S in main and partner; best single booster 0.4001 (L:iter91-served-968-slow-xt).
- Three-seed average - T, -0.0026, not served (L:iter81-served-958-seeds). Two-loss average - T, -0.0048 (L:averages-on-the-window).
- Equal-weight pair and three members; optimal weights gain a fifth of refit noise (L:pair-weight-0928, L:three-member-clv-0927).
- Rank-gated blend: main + partner, race_xent weighted by rank (1,1,1,0.5x4,0) - S; -0.0039 (L:iter105-served-gated-race-xent). Cross-fitted gate weights - T, -0.0014 more, not served (same).
- Within-race power stretch q ~ p^alpha (alpha 1.05) - T, fails Brier (L:within-race-stretch).
- Complete-careers history from 2018: features carry, extra rows do not (L:iter100-served-968s5xh-all3-history-2018-float32).
- Jumps-only and Flat-only fits - P (L:iter117-nh-vs-flat).

## 3. Market blending
- Benter log-linear blend with lambda grid - B/T; stored lambda 1.0 (RF 10.1; model/benter_blend.py).
- Stage C second logit on out-of-fold fundamentals, with Benter's unratable-runner rule - T: delta R2 +0.00001 (RF 10.5, 15.1).
- Closing model: conditional logit to the BSP book from ln market-now, ln model, and volume interactions - S; 0.302 vs morning market 0.350 (reports/race_book_model.md:47-56). No-volume variant S (L:owner-rule-1009).
- Extra terms (curvature, hours, field, code) - T, nothing (L:closing-model-v2-0930). Refit on live books with spread/liquidity weights - T, failed replay twice (L:benter-blend-1020, L:closing-refit-replay-1109). Ridge bias correction by time and price - T, not robust (L:closing-correction-1009).
- Benter to the winner: +0.004 (se 0.039) beyond BSP; model weight 0.00-0.08 at T-5 to T-1 (L:benter-blend-1020; reports/small_book_study.md:25-33).
- Market-only scans (steam beyond BSP, SP bias cells, in-play offsets, morning drift) - T, closed (L:steam-beyond-bsp, L:edge-bsp-bias-1002, L:edge-inplay-offsets-1002, L:market-only-drift-1002).
- BSP calibration - T: calibrated (L:bsp-calibration-1007).

## 4. Ordering and multi-place
- Harville vs Benter/Lo-Bacon-Shone exponents (0.746, 0.665; by field band) - T, Harville rejected (RF 10.5; L:place-market-vs-win).
- Place chance from win probabilities vs place SP - T, +0.00002, closed; win-implied place value loses 4.4% (L:place-model-stage3-1003, L:place-market-vs-win).
- Plackett-Luce order simulation, exact exacta/quinella/trifecta, Shin probabilities - B (model/ordering.py; RF 15.5). Tote exotics pay 0.87-0.91 of fair (L:tote-exotics-1009).

## 5. Simulation
- Race ABM: critical-power energy, pace policy, IDM blocking, drafting, lane ground loss; ABC-SMC on seven patterns - B, never calibrated or run at scale (RF 3, 14.3).
- Monte Carlo of the closing book (log-normal draws by volume band, renormalised) - S (model/race_book.py:164).

## 6. Ratings, effects, causal, selection
- Kalman local-level rating, ML q and r - T: next-run MSE 220 vs EWM 233 (RF 10.5); block adds nothing alone (L:iter96-served-968s5xh-kalman-handicap-travel).
- Multi-runner Elo - T, retired (L:iter55-screen-elo).
- Cross-classified ridge (RAPM) - B (model/effects.py); entity residuals vs BSP do not persist (L:iter11-ae).
- Propensity, IPW, AIPW, DiD, RD, E-values - B, no ledger result (model/causal.py).
- GP draw surface, Moran's I - B (model/spatial.py). BH, model-X knockoffs, mRMR, ALE - B (RF 5.1). Empirical-Bayes shrink chains - S, constants hand-set (reports/model_review_1111.md:71).

## 7. Uncertainty and calibration
- Split conformal on log BFSP, conformal Kelly, Baker-McHale shrinkage - B (model/uncertainty.py:49-127).
- Quantile price calibrator; within-race slope 1.0275 - T (model/price_calibration.py; STAKING_REPORT.md:210).
- Murphy decomposition, ECE, CRPS, concordance, JS drift - diagnostics in use (RF 2). Temperature scaling - B, never called (RF 14.1).
- The trader's expected value quoted only after calibration through its own record - S in reports (L:forecast-1022).

## 8. Staking and execution
- Kelly on model probabilities at BSP - T: 100% drawdown at every fraction (RF 12).
- Joint race Kelly, portfolio Kelly with block covariance, Ledoit-Wolf, Grinold breadth, alpha decay - B (model/portfolio.py).
- Race book (Kelly over backs and lays at the close or result; dutch) - T: single backs at 5% best, +15.1% CLV (reports/race_book_model.md:127-148).
- Served rule: to-win stakes at expected CLV 3%+, three-level ladder, capped top-ups, day limit, lay at SP (CLAUDE.md; L:blend-ladder-replay-1020).
- Hedge ratios, exits, two-sided quotes, lays, stake floors, removal tilt, entry timing, late dutch books (0 of 54 pass) - T (L:hedge-ratio-1021, L:arms-full-1103, L:paper-desk-1104, L:removal-tilt-replay-1109, L:small-book-late-1111).
- Quick-return rule (1-4 days) at the morning price - S (reports/strategy_research_2026-11-04.md:28-48). 197 + 166 system-rule tests and an 8,000-level profile scan - T; only the quick return, the Timeform improver at BSP (+2.0%) and the steamer model pass (same report:25-60, 109-143).
- Entry the evening or night before - recording, unread (L:evening-before-1004, L:night-before-1105).

## 9. Validation
- Walk-forward, 30-day purge, tail holdout for early stopping; paired race bootstrap; decision rule on error, Brier, concordance (scripts/compare_oos_runs.py).
- Placebo block and seed floors; 11-fold screen, noise bar 0.0007 (L:iter50-quick-placebo, L:iter46-served-seeds, L:iter114-wide-screen-market-rating).
- Residual screens on walk-forward errors - T, cannot rank candidates (L:market-season-residuals-0928). Refit noise across CPU types 0.0001 (L:fit-hardware-noise).
- Pre-registration, locked holdout read once (L:iter116-market-rating-final: -0.0067); deflated Sharpe 0.95, PSR, PBO (research/overfit.py; model/uncertainty.py:212-286); chi-square A/E gates (research/strat/strat_lib.py).

## 10. Other sports (forms only)
- Greyhounds: softmax, binary, Plackett-Luce depth 2, log-BSP, share, DeepSets set network, tree-plus-network stack - T, none beats the close (greyhound/fastloop.py:72-77; reports/greyhound_audit_1105.md:10-20).
- Football: pooled Dixon-Coles with log pooling; ridge + LightGBM move stack - paper trials (football/ratings.py; research/open_close/football_stack_trial.py).

## 11. Named as future work, never tried on the horses
- Mixed/nested logit, probit; exploded logit depth 3-4 with eased runners censored; CUSUM/BOCPD (RF 5.1, 10.4; reports/syndicate_papers.md:36-40).
- Conditional-logit combiner over members (model/stack.py); race-level neural network, parked (reports/bfsp_model_plan.md:251-265).
- Optimal stopping for entry, stochastic-programming Kelly, Thompson sampling (RF 5.1); Almgren-Chriss, CVaR, GARCH on price paths, regime switching (RF 10.4, 15.8).
- LambdaRank, CatBoost, Bayesian hierarchical ability, Hawkes order flow (reports/greyhound_roadmap.md:113-150).
- ABM at scale, ABM exotics vs Harville (RF 3.5); Benter's DP6A preference residuals (reports/syndicate_papers.md:41-43); a late signal from the stream ladder (reports/small_book_study.md:63-83).

## Blind spots of this map
- Ledger: all 676 tags listed, about 230 entries read, most truncated to 300-1,100 characters. The daily live-day, replay, model-quality, tote-compare and all-markets-scan entries (about 300) were not read, so a rule variant tested only inside a daily replay could be missing.
- Not read in full: STAKING_REPORT.md (headings and line 210 only), reports/bfsp_model_plan.md (lines 85-100 and 245-270 only), FEATURE_ENGINEERING_RESEARCH.md, docs/, TRADING.md, the trading/ code, evaluate_oos.py and scripts/research_loop.py bodies (docstrings only), the model/abm/ code (RESEARCH_FRAMEWORK.md sections 3 and 14.3 used instead), and model/portfolio.py, staking.py, effects.py, causal.py, selection.py, spatial.py (docstrings and function lists only).
- "Built but never measured" (ABM, conformal, knockoffs, causal designs, RAPM, portfolio Kelly, GP draw surface) rests on three things: RESEARCH_FRAMEWORK.md sections 14-15 (September), no matching keyword in the ledger, and no data/abm* files. A run recorded only in a report I did not open would not show.
- Legacy win classifier (model/trainer.py, probability_model.py, overlay_detector.py): no ledger result found; status taken from the code and RESEARCH_FRAMEWORK.md section 10.1.
- Steamer model: form taken from the strategy report; its hyperparameters and the code in research/paper/morning_features.py were not read.
- Greyhound and football: model forms only, from docstrings and the audit's verdict; their per-experiment results were not mapped.
- No number was re-derived: no database or S3 access, so every figure is as the cited file or ledger entry states it. No web search was made; this map is repository-only.
- Exact line ranges for RESEARCH_FRAMEWORK.md are given as section numbers, not line numbers.
- Scratchpad collision: my draft was first written to scratchpad/map.md, a name another agent in this run also used (its "world_adjacent" map is what that file now holds). I may have briefly overwritten it; my copy is now at /tmp/claude-0/-home-user-ashcroft/20b9529a-9fd3-5b98-8cce-f095c5ce099c/scratchpad/repo_models_agent/repo_models_map.md. Nothing inside the repository was created or changed (git status clean).
