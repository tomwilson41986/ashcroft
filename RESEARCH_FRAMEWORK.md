# Ashcroft Research Framework

**Scope.** This document turns three inputs into one build plan for the Ashcroft BFSP model:

1. the agent-based-modelling (ABM) feasibility report (Cliff/BBE, Mercier–Aftalion, IDM, RMIT drafting, Monte-Carlo);
2. the adjacent-disciplines review (Murphy decomposition, RAPM, causal designs, GP/kriging, conformal Kelly, knockoffs, and the "log as equivalence, don't build" list);
3. the Betfair historic price files (`promo.betfair.com/betfairsp/prices`: morning WAP → pre-off WAP → BSP, volumes, in-play highs/lows).

Everything in it is grounded in what `horse_racing.db` actually holds, and every recommendation carries a module, a command and an acceptance test. Sections 3, 5 and 6 describe code that now exists in the repo; the roadmap in §8 says what to run, in what order, and what result promotes or kills each block.

---

## 0. TL;DR

| Finding | Consequence |
|---|---|
| On the real 2024-01 → 2026-02 walk-forward output (253,532 runners, 27,223 races) the BFSP model is **almost perfectly calibrated** (reliability 0.0000, ECE 0.0013) but has **less resolution than the Betfair market** (0.0087 vs 0.0119). Brier skill vs BSP **−4.5 %**, log-loss skill **−5.3 %**, within-race concordance 0.654 vs 0.681. | The overlay tiers lose ~4 % at level stakes because "overlay" is mostly *our* error, not the market's. Re-estimating what the market already prices cannot produce an edge; the edge must come from signals the market does not have, and from blending with the market rather than betting against it. |
| Neither research report's data assumptions hold here: there are **no GPS sectionals, no biometrics**. We have finishing times, beaten lengths, in-running comments, draw, weight, ratings, BSPs. | The full Mercier–Aftalion calibration is **gated** (§1). What is buildable is a reduced pace-energy ABM whose per-horse parameters come from performance figures and comment-derived run styles, calibrated by pattern-matching against comment/result patterns. Built: `model/abm/`. |
| A cluster of "new" methods are the same estimator as the race-grouped logit we already use. | Recorded in the equivalence ledger (§5.2); nothing built for them. |
| The Betfair price files carry three things the results database does not: price *movement* (steam/drift), *confidence* (volume) and **in-play lows** — how close a beaten horse came. | Built: `betfair_prices.py` + `model/market_features.py` (lag-safe), nightly workflow, `--market-features` in training. Blocked from this container by Cloudflare (403); runs from GitHub Actions or a local machine. |

**Three signal layers the market may lack** (the only places an edge can live): tactical/pace and traffic (ABM), market microstructure and hidden in-running form (Betfair files), and causally-estimated intervention effects (headgear, gelding, trainer switch, class moves). Everything else in the toolkit is *evaluation hygiene* so that we stop fooling ourselves.

---

## 1. Ground truth: data we have vs data the reports assume

**Update (15 Sep 2026).** Two sources changed this table after the first draft: the Blandford Bloodstock backend (`blandford_sync.py`) turned out to be an open Timeform-style results feed, and the HRB Ratings Machine (`hrb_ratings.py`) exposes the user's own pre-race rating sets. Rows marked **open (Blandford)** were previously "closed".

| Needed by | Report assumes | Ashcroft has | Gate |
|---|---|---|---|
| Ability ratings | Timeform/RPR | `official_rating`; **Blandford feed: `preRaceMasterRating`, `preRaceAdjustedRating` (pre-race), `performanceRating` per run** (77–90 % / 96–98 % coverage, 2022→) | **open (Blandford)** — `--blandford-features` |
| Speed figures | Timeform timefigures | `RSR` from `comptime_numeric`; **Blandford `timefigure` per run** (55–90 %) | open (Blandford) |
| Sectionals | 10 Hz GPS | **Blandford: race `finishingTime`, `leaderSectional`, `winnerSectional` over the last `distanceSectional` furlongs** (55–72 % of runners; race-level, not per horse) → `LR_race_fsp_pct` race-shape feature and an ABM calibration target | partly open; per-horse sectionals still closed |
| Stride / biometrics | Equimetre | Blandford `/api/stride/percentiles/{course}` (course-level stride length/frequency percentiles); per-meeting stride data sparse | mostly closed |
| In-play prices | — | **Blandford `ipMin`/`ipMax`, `bSPAdvantage` per run** (77–98 %) and the Betfair price files (§6) | open |
| Run style, trouble, keenness | in-running positions | `comment` parsed by `model/pace_metrics.parse_run_style` | open (ABM behavioural data and calibration target) |
| User ratings | — | **HRB Ratings Machine sets** (HRB Standard, Speed HRB, jockey2, Recency 2, Ability, Conex New, SpeedRatingsLR, …), downloadable per day as CSV; HRB rate-limits downloads per account (shared with the nightly results scrape) | open, throttled — `hrb_ratings.py`, `research_lab.py ratings-eval` |
| Draw / geometry | course geometry, wind | `stall`, `stall_positioning`, `track_direction`, `rail_move`; coarse table in `model/abm/track.py` | partly open |
| Market | Betfair SP | `bfsp`, `bfsp_place`, live snapshots; Betfair historic files (§6); Blandford `betfairWinSP/PlaceSP`, `ispDecimal` | open |
| Interventions | gear / wind-op flags | `headgear` (first-time derivable), sex, trainer/jockey changes, class moves; Blandford `headGear` | open for headgear etc.; wind surgery closed |
| Pedigree | Blandford bloodstock DB | `stallion`, `dam`, `dam_stallion`; Blandford `sireName/damName/damsireName`, `foalingDate`, stable `horseCode` | open |

Matching: the Blandford feed linked **1,484 / 1,484** runners of a real February-2026 week to `race_results` on the full (date, course, time, horse) key.

**Rule kept:** nothing is trained or evaluated on generated data (CLAUDE.md). The unit tests use small synthetic frames only to exercise code paths.

---

## 2. Where the model stands (real data, `python research_lab.py score`)

```
n_runners 253532   n_races 27223   base_rate 0.1075
model_log_loss 0.29588   market_log_loss 0.28107   log_loss_skill_vs_market -0.0527
model_brier    0.08615   market_brier    0.08247   brier_skill_vs_market    -0.0446
reliability   model 0.0000  market 0.0000        (both calibrated)
resolution    model 0.0087  market 0.0119        (market discriminates better)
concordance   model 0.654   market 0.681
mean per-race JS(model, market) 0.030
```

Reading: Murphy's decomposition says the gap to the market is **entirely resolution**, not calibration. Isotonic recalibration cannot help; only new information can. The reliability table is flat in every decile (max |gap| 0.003), so the top-decile "value" bets are priced correctly *on average* — the loss comes from the model ranking runners within a race slightly worse than the market does.

Two immediate consequences, both already supported by the code base:

* **Blend, don't oppose.** `model/benter_blend.py` (log-linear blend with the market) is the correct default; the Betfair *morning* price (§6) lets the blend happen at decision time rather than against BSP after the fact.
* **Look for segments, not averages.** Concordance and Brier skill should be re-computed by race type / field size / class / market maturity (`scoring_report` on subsets, block-bootstrapped by race). Any segment with positive skill vs BSP is a candidate strategy; the whole-population number says the unsegmented overlay strategy should not be bet.

---

## 3. Layer A — the race ABM (`model/abm/`)

### 3.1 Positioning
The ABM is a **feature generator and counterfactual tool**, not a replacement model (report §4 and the F1 / Terawong–Cliff precedent). Its win probability is dominated by the same ability signal LightGBM already has, so the useful outputs are the *decomposition*:

* `abm_win_prob_solo` — interaction-free run: every horse runs the same even-pace race; result = ability + form noise. A Harville-like reference.
* `abm_win_prob` — full run: pace policy, following, drafting, blocking, lane changes, interference, draw ground-loss.
* `abm_pace_delta = abm_win_prob − abm_win_prob_solo` — **what tactics and traffic add or remove.** This is the only column the statistical model cannot already see.
* Plus `abm_place_prob` (UK each-way terms), `abm_exp_beaten_l`, `abm_p_led_early`, `abm_p_led_2f`, `abm_p_trouble`, `abm_p_blocked_final`, `abm_pace_contest`, `abm_pace_collapse_p`, `abm_race_entropy`, `abm_win_sd` (Monte-Carlo error), `rAbmWin`.

### 3.2 Agent parameters from DB columns (`agents.py`)

| Parameter | Source column(s) | Mapping |
|---|---|---|
| ability (lbs) | `horse_perf_lbs_ewm` (lag-safe recency-weighted performance figure, `model/perf_figures.py`), fallback `official_rating`/`median_or`; `pounds` | `perf − (pounds − field mean)`; in a handicap the weights offset the marks by construction, so what remains is "form vs mark". Converted to speed via lbs → lengths (`lbs_per_length`, 3.0 at 5f … 0.75 at 20f) → 2.4 m per length over the trip. |
| sustainable speed `v_cp` | distance par table (17.3 m/s at 5f … 14.8 at 20f) × going factor × 0.97 × ability factor | |
| surge `dv`, reserve `W'` | trip; `horse_career_late_move`, `horse_keen_rate`, age | `W'` ≈ 80 surge-metres ± 10 % |
| early effort / desired gap | `pred_early_pos` → `horse_early_pos_3r` → `horse_career_early_pos` → `Horse_Career_EPF` → `EPF` | front-runners (≥ 4.5) try to lead; others sit `(6 − style) × 2.5 m × √(N/10)` off the pace |
| kick distance | trip, `jockey_front_rate` (aggression), style | 300 m at 5f … 600 m at 16f |
| trouble proneness | `horse_trouble_rate` | hazard multiplier 0.5–2 |
| form noise | `career_runs` | **7 lbs** per run (Timeform-scale figures vary ~7 lbs run to run); ×1.3 for horses with < 3 runs |
| draw | `stall` | initial lane 0.35 × (stall − 1), fast convergence on the rail before the first bend |

`abilities="market"` instead reads abilities off the BSP: lbs = ln(p/mean p)/β. **Self-consistency result:** with 7 lbs form noise, the solo ABM reproduces the market distribution almost exactly at **β ≈ 0.22 log-odds per lb** (KL 0.003 at 0.25, 0.013 at 0.20, 0.06 at 0.16 on a 10-runner test field). That pins the model's implicit logit slope to a Benter-type value and is the first thing to re-check on real races (`research_lab.py abm --abilities market`).

### 3.3 Dynamics (`energy.py`, `simulate.py`)
1.5-D track: rail coordinate + lane. Fixed Δt = 0.25 s, vectorised over (sims × runners); a 10-runner mile at 1 000 sims takes ~1.7 s per mode on one core.

* **Physiology** — reduced Aftalion/critical-power model: first-order speed lag (τ 1.5 s accelerating, 2.5 s easing); W'-balance drains ∝ (v − v_cp) above cruise and refills below; surge fades as W' empties and sustainable speed sags up to 8 % ("weakened"). Going scales speed and energy cost.
* **Policy** — front-runners push for 150 m to establish the lead, then set fractions (effort 0.30 uncontested, 0.55 when disputed); everyone else tracks the leader's speed and settles into their desired gap; final kick at full surge from the kick distance. Keen horses waste W' early.
* **Traffic** — IDM-style: blocked if the gap to the horse ahead in the same lane < s₀ + v·T; pass by switching out when the outer lane is free; drift to the rail when free; stochastic interference while boxed in among ≥ 2 others (speed −8 %, W' cost); "denied a run" = seconds blocked *while wanting to pass* in the last 600 m.
* **Drafting** — RMIT-derived: anaerobic cost −15 % behind one horse, −25 % behind two or more (the wind-tunnel drag reductions scaled by the aerodynamic share of cost).
* **Bends** — small speed loss ∝ curvature; outer lanes cover more ground (≈ 1.3 lengths per lane per half-circle at R = 180 m — matches the rule of thumb). Coarse geometry per course in `track.py`.

### 3.4 Calibration (`calibrate.py`) — pattern-oriented, likelihood-free
Behavioural parameters (blocking gap, headway, lane-change rate, interference hazard, drafting saving, lead-effort policy, stall spread, process noise) have no tractable likelihood. `abc_smc` is a population-Monte-Carlo ABC: sample the prior box, keep the 30 % of particles whose simulated patterns sit closest to the observed ones, perturb, repeat. Following Grimm & Railsback, **seven patterns are matched at once**, all computable from `race_results`:

`trouble_rate`, `front_win_share`, `early_finish_corr`, `median_win_margin`, `p90_win_margin`, `contested_lead_penalty`, `low_draw_win_share`.

Run: `python research_lab.py abm-calibrate --db horse_racing.db --from 2025-01-01 --races 200`. Store the result (`data/abm_calibration.json`) and pass it as `SimConfig(**params)` when generating features. Until this has been run on real races the defaults are deliberately moderate; **do not read the feature values as calibrated.**

### 3.5 Acceptance tests and kill criteria (report Stage 2 thresholds, made concrete)
1. `abm-calibrate` loss < 1.0 (all seven patterns within ~1 scale unit) on 200 held-out races — otherwise the mechanics are wrong; fix before generating features.
2. Market self-consistency on real cards: KL(market ‖ solo) < 0.02 at β ≈ 0.22 — otherwise the ability mapping is off.
3. Feature value: train with `--abm-features` on the 2024+ window; **promote** only if walk-forward log-loss improves *and* `brier_skill_vs_market` rises by ≥ 0.005 with a block-bootstrap (by race) 90 % interval excluding zero; and `abm_pace_delta` survives BH/knockoff selection (§5). **Kill** the physics-heavy ABM if after calibration it fails (1) or (3); fall back to a statistical pace simulation (sampling comment-derived position profiles).
4. Exotics: compare ABM forecast/tricast probabilities against Harville on the same abilities (the ABM's correlated finishing orders are its structural advantage — report §4).

Compute: ~1 s/race at 500 sims (both modes). 27k races ≈ 7.5 CPU-hours; `--jobs N` fans out across cores. A JAX port is the Stage-2 speed-up if the feature block is promoted.

---

## 4. Layer C — causal intervention effects (`model/causal.py`)

Why the market mis-prices these: entry into headgear, gelding, a trainer switch or a class drop is *chosen* in response to form, so naive before/after deltas are confounded (Allen & Franklin's wind-surgery result is the cautionary tale). Built: propensity scores, Hajek IPW (ATE/ATT with effective sample sizes), **AIPW doubly-robust with 2-fold cross-fitting and influence-function SEs**, 2×2 and two-way-FE DiD with unit-clustered SEs, sharp RD (handicap-mark thresholds), E-values, a lag-safe `first_time_flag`.

```
python research_lab.py causal --db horse_racing.db --treatment first_time_headgear --outcome won
```
Use the *lag-safe* form features as covariates (the defaults in the CLI). Report ATE, ATT, CI and E-value. The feature stays `first_time_headgear`; what changes is our *prior* on it (and whether to interact it with the covariates the propensity model found decisive). Next treatments in order: gelding (sex change between runs), trainer change, class drop, apprentice claim.

---

## 5. Adjacent-discipline imports

### 5.1 What was built and where

| Tier | Method | Module / function | Status |
|---|---|---|---|
| 1 | Murphy decomposition (bias-corrected, Stephenson within-bin terms), Brier/log-loss skill vs BSP, ECE, reliability tables | `model/diagnostics.py` | built, run on real OOS (§2) |
| 1 | RPS (ordinal win/place), CRPS (ensemble + Gaussian) | `diagnostics.py` | built; CRPS is for ABM margin/time distributions |
| 1 | Concordance (Cox C-index within race), Kendall τ by race | `diagnostics.py` | built |
| 1 | KL/JS drift monitor (model vs market by month) | `diagnostics.divergence_by_period` | built |
| 1 | Cross-classified ridge effects (RAPM) with per-block empirical-Bayes λ; lag-safe walk-forward scoring | `model/effects.py` | built; the single horse/jockey/trainer/sire heterogeneity route (frailty = panel RE = ridge) |
| 1 | Propensity / IPW / AIPW / DiD / RD / E-value | `model/causal.py` | built |
| 1 | GP (kriging) smoothing of draw bias by stall (count-weighted noise), going drift within a meeting, Moran's I | `model/spatial.py` | built |
| 1 | Benjamini–Hochberg, race-demeaned univariate screens, **model-X Gaussian knockoffs** (lasso coefficient-difference, knockoff+), mRMR | `model/selection.py` | built; recovered exactly the 5 planted signals among 30 in test |
| 1 | Split conformal intervals on log-BFSP → probability intervals; conformal Kelly (lower-bound or width-shrunk); Baker–McHale-style shrinkage | `model/uncertainty.py` | built |
| 2 | Block (cluster-by-race) bootstrap, stationary bootstrap for P&L series | `uncertainty.py` | built |
| 2 | ALE, conditional permutation importance (Strobl) | `model/interpret.py` | built |
| 2 | Performance figures (lbs) + lag-safe EWM/best-3/sd/"well-in" features | `model/perf_figures.py`, `--perf-features` | built |
| 2 | Mixed/nested logit (IIA relaxation), exploded logit to depth 3–4, CUSUM/BOCPD change-points, stacking | — | **next**; the exploded-logit depth experiment is cheapest (extend the race-grouped softmax) |
| 3 | Optimal stopping for pre-off timing, stochastic-programming Kelly, Thompson sampling for strategy allocation | — | later; needs the Betfair pre-off price model (§6.4) |

### 5.2 Equivalence ledger — recorded, not built
* Cox partial likelihood ≡ conditional logit ≡ Plackett–Luce ≡ rank-ordered (exploded) logit (Allison & Christakis 1994). Our race-grouped softmax is a Cox partial likelihood with race as stratum. Buys the `coxph` toolchain (frailty, robust SEs) for free.
* Rasch and Bradley–Terry are special cases of the same logit family. Elo is an online approximation of the state-space ratings we already have (Glicko ≈ Kalman).
* Bühlmann–Straub credibility ≡ empirical-Bayes shrinkage (our trainer/jockey/sire suites). Keep the exposure-weighting nuance only.
* Kriging ≡ Gaussian-process regression (built once as GP).
* Gumbel error (EVT) ≡ the logit assumption the softmax rests on; justifies mixed/nested logit as the IIA relaxation.
* Frailty ≡ panel random effect ≡ ridge cross-classified effect: one build (`effects.py`).
* MDL ≡ BIC; ordered logit is dominated by exploded logit; competing risks, loss reserving, ruin theory (Kelly wealth is multiplicative), Pythagorean expectation, transfer entropy, SAX/shapelets, ABC-for-tractable-likelihoods: **skip**. (ABC is used only where it belongs — the ABM's intractable behavioural likelihood.)

### 5.3 Interpretation hygiene
Racing features are inter-correlated (ratings, speed, class, pace). SHAP and plain permutation importance split credit arbitrarily; use `conditional_permutation_importance` and `accumulated_local_effects`, and let BH → knockoffs decide inclusion of any new block (`univariate_pvalues` race-demeaned first, `knockoff_filter` on the survivors).

---

## 6. Layer B — Betfair historic price files

### 6.1 Ingestion (`betfair_prices.py`, table `betfair_prices`)
Files `dwbfprices{uk|ire}{win|place}{DDMMYYYY}.csv`, one row per runner-market: `BSP, PPWAP, MORNINGWAP, PPMAX, PPMIN, IPMAX, IPMIN, MORNINGTRADEDVOL, PPTRADEDVOL, IPTRADEDVOL`. The script downloads (with clear 403/Cloudflare reporting), parses, upserts on `(event_id, selection_id, market_type)`, and links rows to `race_results` by `(date, course, time, horse)` with `(date, course, horse)` and `(date, horse)` fallbacks; `--report` prints monthly coverage and unmatched course hints to add to `BF_COURSE_MAP`.

```
python betfair_prices.py --fetch --from 2024-01-01 --to 2024-12-31   # or --dir <folder of files fetched elsewhere>
python betfair_prices.py --load --match --report
```
Nightly: `.github/workflows/betfair-prices.yml` (06:15 UTC, last 3 days, S3 backup). **This container is refused (HTTP 403)**; the workflow runner or a home machine is the fetch point.

### 6.2 Features (`model/market_features.py`, `--market-features`) — all lag-safe by default
* `LR_mkt_ip_low_ratio`, `horse_ip_low_3r`, `horse_ip_hit_low_5r` — **in-play low relative to BSP on previous runs**: a horse that traded at 1.2 in running and lost "should have won"; this is hidden form invisible to finishing position and comment parsing.
* `LR_mkt_steam`, `horse_steam_3r`, `horse_steam_rate_5r`, `trainer_steam_rate`, `jockey_steam_rate` — morning → BSP moves ("stable money").
* `LR_mkt_vol_share`, `horse_vol_share_3r`, `LR_mkt_pp_range` — confidence and pre-off volatility.
* `LR_mkt_outperform`, `horse_outperform_3r`, `horse_outperform_career`, `trainer_outperform` — finished better than the market rank implied.
* `LR_mkt_place_ratio`, `LR_mkt_implied_p`, `horse_mkt_runs`.

### 6.3 Leakage rules (decision-time gating)
* `PPWAP`/`BSP` of *today's* race are never features for a BSP-target model.
* `MORNINGWAP` of today's race (`include_same_day_morning=True` → `today_morning_*`) is legitimate **only** for a decision taken after the morning market forms — which is when `daily_predictions.py` runs. Use it for the Benter blend and for a *price-movement* model, never silently in the BSP regression.

### 6.4 The model this enables: predict the move, not the price
With morning prices in hand, the actionable target is `ln(BSP / MORNINGWAP)` given fundamentals + morning market: which horses will be backed in. That is the pre-off edge in exchange betting (back early what will shorten; lay what will drift), it pairs with the optimal-stopping/timing work (Tier 3), and its evaluation is the same scoring suite with the morning price as the reference forecast. `market_movement_summary` (steam deciles → win rate and ROI at BSP) is the first diagnostic: is the move informative *beyond* the price?

---

## 6b. Layer B′ — Timeform feed and user ratings

* **Blandford / Timeform feed** (`blandford_sync.py`, table `blandford_results`, `model/blandford_features.py`, `train_bfsp.py --blandford-features`): same-day `tf_master_pre` (+ within-race rank, gap to top, vs OR) is the first genuinely new *ability* input since the model was built — the market prices Timeform heavily, so the expected effect is a large gain in resolution *toward* the market rather than beyond it; the lag features (performance rating EWM/best-3/sd, timefigure, `LR_tf_perf_vs_master` improver flag, `LR_race_fsp_pct` race shape, in-play low) are where incremental information may sit. Nightly: `.github/workflows/blandford-sync.yml`. Backfill: `python blandford_sync.py --fetch --from 2022-06-01 --load --match` (≈5 MB per week).
* **HRB Ratings Machine sets** (`hrb_ratings.py`, `model/hrb_features.py`): each set is a same-day rating. Evaluate before adopting: `research_lab.py ratings-eval` reports per set the standalone concordance, race-demeaned correlation, softmax log-loss next to model and market, and the walk-forward *stacked* gain over [ln p_model, ln p_market]; `knockoff_screen` decides which sets survive jointly. Throttle downloads (`--spacing`, `--max-requests`) — the account's download allowance is shared with the nightly scrape.

---

## 7. Evaluation protocol (applies to every new feature block)

1. **Walk-forward** exactly as `train_bfsp.py` does (temporal folds, no peeking); compute `research_lab.py score` on the fold predictions.
2. **Decompose**: report `brier_skill_vs_market`, `log_loss_skill_vs_market`, reliability/resolution (bias-corrected), concordance vs market, per segment (race type, field size, class, month).
3. **Uncertainty**: block-bootstrap by race (`uncertainty.block_bootstrap`) for skill deltas; a feature block is promoted only if the 90 % interval of the skill *improvement* excludes zero.
4. **Selection**: BH on race-demeaned univariate screens, then model-X knockoffs at FDR 0.1 on the survivors; report `conditional_permutation_importance` and ALE for the kept features.
5. **Drift**: monthly JS(model, market) from `divergence_by_period` on the dashboard; a rise precedes P&L damage.
6. **Staking**: conformal intervals on log-BFSP (`SplitConformalRegressor` on a held-out calibration fold) → `conformal_kelly_stake(mode="lower")`; compare drawdown and growth to the current fractional Kelly with `block_bootstrap` on daily P&L (stationary bootstrap for the series).
7. **Profitability paradox** (report caveat): a better score does not guarantee ROI; every promotion also needs a backtested ROI at BSP with commission on the same folds.

---

## 8. Roadmap and commands

| Week | Step | Command | Promote if |
|---|---|---|---|
| 0 | Baseline scoring (done) | `python research_lab.py score` | — (documented in §2) |
| 1 | Perf-figure features | `python train_bfsp.py --perf-features` | log-loss ↓ and Brier skill vs BSP ↑ (block-bootstrap) |
| 1 | Betfair files: backfill + match | `betfair_prices.py --fetch --from 2023-01-01 …` (runner/local), `--load --match --report` | match rate ≥ 90 % (fix `BF_COURSE_MAP` from `--report`) |
| 1 | **Timeform feed backfill + features** | `blandford_sync.py --fetch --from 2022-06-01 --load --match`; `train_bfsp.py --blandford-features` | log-loss ↓, Brier skill vs BSP ↑ (block-bootstrap); expect the biggest single gain of the plan |
| 1 | **HRB rating sets: evaluate** | `hrb_ratings.py --fetch --sets … --from … --spacing 5 --max-requests 60` (throttled), `research_lab.py ratings-eval` | a set is adopted only if its stacked gain in Brier skill vs market is > 0 with a bootstrap CI excluding 0 and it survives knockoffs |
| 2 | Market features | `python research_lab.py market`; `python train_bfsp.py --market-features` | as above; `LR_mkt_ip_low_ratio` survives knockoffs |
| 2 | Causal: first-time headgear, gelding, trainer change | `python research_lab.py causal --treatment …` | E-value > 1.5 with CI excluding null → set coefficient prior / interaction |
| 2–3 | Effects (RAPM) | `python research_lab.py effects --from 2023-01-01`; `walk_forward_effects` columns as features | lag-safe `rapm_*` improve log-loss |
| 3 | ABM calibration | `python research_lab.py abm-calibrate --from 2025-01-01 --races 200` | loss < 1.0; market self-consistency KL < 0.02 |
| 4 | ABM features on 2024+ | `python research_lab.py abm-features --from 2024-01-01 --jobs 8 --out data/abm_features.parquet`; `python train_bfsp.py --abm-features data/abm_features.parquet` | §3.5 (3); else fall back to statistical pace simulation |
| 4 | Draw surfaces | `python research_lab.py draw --track Chester --dist 5` | replace binned draw features with GP-smoothed means where Moran's I is significant |
| 5 | Exploded logit depth 3–4, mixed logit | new | log-loss ↓ on held-out season |
| 6 | Conformal Kelly in the live pipeline | `uncertainty.py` into `model/overlay_detector.py` | lower drawdown at equal growth (bootstrap) |
| later | Price-movement model, optimal stopping, Thompson sampling of strategies | — | needs §6 backfill |

---

## 9. Risks and limitations (kept honest)
* **Identifiability**: per-horse ABM traits (W', surge, style) are weakly identified from a handful of runs; they are deliberately shrunk (±10 %) and the ability term dominates. Do not widen trait spreads without a pattern that demands it.
* **Calibration before belief**: ABM defaults are moderate placeholders; every interaction parameter must come from `abm-calibrate` on real races. The interaction layer currently produces a trouble rate and front-runner effect that will move substantially under calibration.
* **Geometry**: coarse per-course table; Chester/Epsom/Goodwood deserve real geometry before draw conclusions are drawn from the ABM (use `spatial.smooth_draw_bias` on real results in the meantime).
* **Betfair files**: course-abbreviation map is best effort until `--report` has been run on real files; matching falls back to `(date, horse)` which is safe only because a horse runs once a day.
* **Multiple testing**: the toolkit adds many candidate features; BH → knockoffs is not optional.
* **Market as reference**: BSP is the *closing* price; a strategy must beat the price it can actually get (morning/pre-off), which is why §6.4 matters more than any further gain on BSP.

---

## 10. Alignment with the racing² Master Framework (v3)

The master framework (`racing2_master_framework_v3_1.md`) was reviewed against this repo on 15 Sep 2026. Its governing design — **Stage F fundamental (race-grouped softmax, market-free) → Stage C combination with the market → Stage S fractional Kelly**, judged only by **ΔR² over the market out of sample** — is the right spine for Ashcroft, and it exposes the deepest problem in the current pipeline.

### 10.1 The structural finding
`train_bfsp.py` regresses on **log(BFSP)**: the market is the model's *target*, not a Stage-C input. A model trained that way can only reproduce the market (lossily); its "overlays" are its own reconstruction error, so betting them must lose — which is exactly what §2 and the `bets` analysis measured (calibrated, lower resolution than BSP, −4 to −7 % on every overlay tier, actual win rates sitting on the market's number). The stored blend weight (`blend_config.json`: λ = 1.0 = pure market) is the same fact seen from Stage C. The framework's P5 is also violated inside the feature set: `feature_registry.classify` finds **20 market-derived features** in the 419-feature BFSP model (`ORR2`/`LR_ORR2`/`preracehorsecareerORR2`, `PFD3/5/10`, `OFS1/3/5/10`, their ranks, and the expectation-residual family built from market rank).

**Consequence.** ΔR² is not measurable for the current model at all. The fix is architectural, not another feature: build a true Stage F on the winner label with `stage_f_columns()` features, fit Stage C on out-of-fold Stage F output, and report ΔR² with a race-bootstrap CI. `model/stage_f.py` provides the estimator (conditional logit; LightGBM race-softmax objective for the non-linear version — "the change from the existing XGBoost work is the objective and the grouping, not the algorithm"), Stage C, temperature scaling, the conditional calibration tables and the walk-forward harness.

### 10.2 First honest ΔR² on real data (Phase 1 exit test)
Run on the open Blandford/Timeform feed, UK/IRE Flat, 2025-01 → 2026-02 (8,289 races; 5,018 test races from 2025-07 with walk-forward folds; Stage C fitted on out-of-fold fundamentals only): `python research_lab.py phase1`. See the table in §10.5 for the numbers. The market's own R² on modern Betfair (≈ 0.17) is far above Benter's 1980s tote public (0.12): the bar is high, exactly as §I.4 of the framework warns. A first market-free Stage F with 19 features does not clear it; the framework's prescription is to earn ΔR² block by block (Tier 1 metrics, Kalman ratings, pace/draw interactions, LTO-contradiction features), each promoted only with a CI clear of zero.

### 10.3 Adopted (built here)
| Framework item | Module | Notes |
|---|---|---|
| Stage F conditional logit, LightGBM race-softmax objective, Stage C, ΔR² + CI, temperature scaling, conditional calibration | `model/stage_f.py` | II.1–II.2, IV.1–IV.2, VI.1 |
| State-space (Kalman) ratings with ML-fitted q/r, career-stage drift, condition-dependent noise, per-horse uncertainty | `model/state_space.py` | IIA.1 — beat fixed-λ EWM on next-run performance in the real-data test |
| NMFP, FSA-%RB² (par (2N−1)/(6(N−1))), FSS credibility, `plc_fsa`, truncated distance-adjusted beaten lengths, lengths→time, empirical `bl_vs_par` surface, censoring flags, IQM/MIN/slope aggregation, LTO-contradiction and handicapper-gap features, rating-based N_eff, SoS_vs_today, within-race z/rankpct/vs_max | `model/primitives.py` | III Parts 1–3, 7, 8 (Ashcroft's `NFP` and `RB` are both plain %RB; `FSARB` is not the field-size par) |
| Benter / Lo–Bacon-Shone ordering (γ, δ by ML), place probabilities | `model/ordering.py` | IV.3 — fitted γ ≈ 0.75, δ ≈ 0.67 on real data; Harville rejected |
| Feature-stage registry (F/C/S) + market-tautology audit | `model/feature_registry.py` | P5 / Part 9 |
| Murphy decomposition, skill scores, conditional calibration, race bootstrap, conformal Kelly, knockoffs | already in §5 | VI.2 items 4–6; DSR/PBO still to add |

### 10.4 Recorded, not built (equivalent or gated)
* Kalman ≈ Glicko/TrueSkill (specialisations) — one implementation. Heteroskedastic logit / GARCH-on-residuals: test against `kf_sd` first (IIA.3). GARCH on pre-off prices: needs price *paths* (the Betfair files carry only summary stats; the live `betfair_odds` snapshots would); run ARCH-LM before fitting anything (IIA.4). Almgren–Chriss, CVaR, Ledoit–Wolf block covariance for portfolio Kelly: after a Stage F with ΔR² > 0 exists. Copulas, Hawkes, RL staking, Sharpe as headline: skip (IIA.9). Probit for IIA: only if measured substitution effects justify it (II.3).

### 10.5 Results table (real data, UK/IRE Flat, 8,289 races 2025-01 → 2026-02; 5,018 test races from 2025-07)

| Quantity | Value | Reading |
|---|---|---|
| R²_market (Betfair SP, overround removed) | **0.1717** | the bar; Benter's 1980s tote public was 0.1218 |
| R²_fundamental (19 market-free features, conditional logit, walk-forward OOF) | 0.0740 | 43 % of the market's information |
| R²_combined (Stage C on OOF fundamentals) | 0.1717 | γ swamps α |
| ΔR² combined − market, race-bootstrap 90 % CI | **+0.00001 (−0.00014, +0.00017)** | no edge from this feature set — the honest answer |
| Conditional calibration, model > market, band 0.05–0.10 | model 7.3 %, market 3.6 %, actual 3.7 % | Benter Table 3/4 pattern: the market is right |
| Kalman rating vs fixed-λ EWM vs Timeform master, next-run performance MSE | **220** / 233 / 294 | state-space beats both; use `kf_rating` + `kf_sd` |
| Fitted Kalman q, r (performance-rating units) | 170, 124 | run-to-run noise sd ≈ 11 lb; steady-state gain 0.67 |
| Ordering γ, δ (Benter / Lo–Bacon-Shone) | 0.746, 0.665 | vs HK 0.81 / 0.65; Harville nll 10,632 → 10,475 |
| Largest Stage F coefficients (standardised) | tf_master_z +0.38, kf_sd −0.37, nmfp_mean3_z +0.28, first_run +0.21, runs_count −0.17 | uncertainty itself is predictive |
| Data quirks to remember | ratings/timefigures coded 0 when missing; pre-race ratings coded **999** when unrated | handled in `phase1.prepare_blandford_frame` |

### 10.6 What to do next, in the framework's order
1. **Stage F pipeline** (`train_stage_f.py`): winner label, `stage_f_columns(ALL_FEATURE_COLS)` + `primitives`/`state_space` blocks + Blandford ratings, LightGBM with `lgb_race_softmax`, walk-forward; Stage C on OOF; report ΔR² with CI. This replaces the BFSP regression as the model that decides bets; the BFSP regression stays useful only as a price *forecaster* for the pre-off price-movement model (§6.4).
2. Tier-1 metric blocks from `primitives.py` (NMFP/FSA-%RB² aggregates, `bl_vs_par`, `sos_vs_today`, handicapper gaps, LTO contradictions) and the Kalman rating on `perf_lbs` / Timeform performance ratings.
3. Pace × draw interactions (`draw_x_style`, `pace_suit`) — the ABM's `abm_pace_delta` is the simulation route to the same feature.
4. Calibration: temperature scaling on Stage F logits; **the conditional tables must pass before any staking**.
5. Ordering with fitted γ, δ for place/exotic markets; compare with ABM finishing orders.
6. Validation additions: purge/embargo in the walk-forward, deflated Sharpe and PBO; closing-line value as the live edge signal.

---

## 11. The objective, restated: closing-line value, not beating BSP

**Decision (15 Sep 2026).** The goal is not to out-predict the Betfair Starting Price. BSP is treated as the efficient closing price — the evidence in §2, §10 and below all says so — and the edge is **getting on earlier at prices longer than the BSP the horse will close at**. That makes the BFSP regression the right *kind* of model (a price forecaster) and changes what we measure.

### 11.1 Why CLV is the score
Back at morning odds *o*, close at BSP *b*. Green up by laying at *b* and the locked-in profit per unit is **o/b − 1 whether the horse wins or loses**; hold to settlement and the *expected* profit is the same quantity once 1/b is accepted as the best estimate of the true probability. So the strategy's edge is the realised CLV of the runners it selects, and the model's job is to forecast *b* from what is known in the morning. Everything else — win-probability calibration against BSP, ΔR² over the closing market — is secondary.

### 11.2 The model (Benter's blend, applied to prices)
`ln b ≈ β0 + β1 ln π_morning + β2 ln f̂ + β3 ln n (+ morning volume share)`, with f̂ the fundamentals-only BSP forecast from `train_bfsp.py` (walk-forward, out-of-sample) and π_morning the race-normalised morning price. |β2| clearly above zero (walk-forward, per fold) means fundamentals predict the move beyond the morning market. The early-bet rule backs runners whose forecast BSP is shorter than the price on offer by ≥ `min_clv` (net of commission), optionally sized by predicted CLV and capped by morning depth.

Built: `model/clv.py` (`prepare_clv_frame`, `price_move_model`, `early_bet_rule`, `clv_report`, `steam_predictability`) and `research_lab.py clv`, which joins the walk-forward predictions to morning prices from either the Betfair historic files (`betfair_prices`, MORNINGWAP + volume) or the daily pipeline's own `betfair_odds` snapshots (`--source snapshots`, earliest snapshot per runner per day — data we already collect). Report: per-fold coefficients, direction hit rate of the move, mean/median net CLV with a race-bootstrap CI, hit rate (price shortened), hold-to-settlement ROI and BSP-implied EV, CLV by predicted-CLV tier, and the all-runners baseline (is the morning market itself biased?).

### 11.3 What the closing-market tests now mean
* §2 (BFSP model calibrated, lower resolution than BSP, overlays lose): expected for a BSP forecaster scored against BSP; irrelevant to CLV.
* §10 (a market-free Stage F reaches ~57 % of the market's R²; ΔR² ≈ 0; adding previous-run in-play lows or the lagged market to Stage C also gives ΔR² ≤ 0 on 5,017 real races): **BSP is efficient with respect to every history we hold** — the premise that makes BSP the right truth for CLV.
* The remaining question is empirical and needs morning prices joined to the walk-forward forecasts: does f̂ move BSP forecasts beyond the morning price? `research_lab.py clv` answers it the day `betfair_prices.py --load --match` (or the live snapshots) is available; the acceptance test is **mean net CLV of the rule's bets > 0 with the race-bootstrap CI clear of zero, on ≥ 1,000 bets**, and a fill assumption bounded by morning depth.

### 11.4 What still matters from the earlier sections
Sharper BSP forecasts (lower log-error) raise CLV directly, so the feature blocks in §3–§6 and §10 still earn their place — but their promotion criterion becomes *incremental BSP-forecast accuracy given the morning price*, measured by the CLV report, not ΔR² over BSP. The Kalman rating, Timeform feed and connection blocks are the first candidates; the ABM's pace features and the causal intervention effects remain the candidates for information the *morning* market prices late.

---

## 12. Market-blind staking: Kelly and ranks on real out-of-sample output

Full report and tables: **[STAKING_REPORT.md](STAKING_REPORT.md)**. Command:
`python research_lab.py stake --predictions data/oos_predictions.csv`. Module:
`model/staking.py` (Kelly at the settlement price, log-space bankroll paths, rank
staking plans, shrinkage scan, forecast-price diagnostics), tested in
`tests/test_staking.py`.

The question asked was how the model performs on Kelly and on per-race ranks *without
considering the market*. The market can be removed from the selection, the probability
and every filter, but not from the settlement price — with no market price there is no
edge and Kelly stakes nothing. Results on 253,532 runners / 27,223 races
(Jan 2024 – Feb 2026), commission 5%:

| finding | number |
|---|---|
| Full Kelly on model probabilities | bank halves by race 9, under 1% by race 29, ends 10⁻⁷³² |
| 1/20 Kelly | halves by race 417, ends 10⁻⁴·⁴ |
| Share of the loss attributable to variance rather than negative edge (1/20 Kelly) | ≈ 3/4 |
| Kelly-weighted ROI over the same bets vs flat | −1.04% vs −5.69% |
| Model rank 1, flat at BSP | −0.44% (90% CI −2.1 to +1.4); random runner −5.73%, favourite −3.05% |
| Model rank 1, field ≥ 12 | +4.40% (CI −0.7 to +9.4), quarters −3.0 / +3.2 / +7.7 / +9.8 |
| Model rank 1, field ≥ 16 | +14.77% (CI +2.1 to +29.0) |
| Model rank 1, forecast price ≥ 8 | +13.44% (CI −1.8 to +29.0) |
| Flattening probabilities (p ∝ p^λ) | monotonically worse as λ → 0: the ordering is the asset, not the confidence |
| Stage F (market-free), rank 1 | −1.95% (clogit) / −5.11% (LightGBM) on 5,017 races |
| Forecast BFSP bias on the model's top pick | +8.3% (closes shorter than forecast 58% of the time), decaying to 0 by rank 5 |

Consequences for the roadmap: (a) no Kelly sizing on these probabilities at any fraction;
(b) the big-field and long-forecast-price cells join the disagreement cells from §7 as the
only market-blind selections worth live testing; (c) the +8% retransformation bias in the
top-pick price forecast is a correctable defect that directly costs CLV, and should be
fixed before any early-price trigger uses the forecast.

---

## 13. Two leaks found in the pace and draw blocks

Found while auditing what the pace, race-shape and draw work actually does.
Both are confirmed by direct reproduction and both are now fixed and tested
(`tests/test_leakage.py`, 11 tests). **The BFSP model must be retrained before
its numbers mean anything**: the deployed 419-feature model was trained with
both leaks present, so its reported accuracy is inflated and five of its
features are dead at prediction time.

### 13.1 Race-level bias features lagged by row instead of by race

`grp[col].apply(lambda x: x.shift(1).expanding().mean())` is the repo's idiom
for a lag-safe career mean. On a key the horse owns it is correct. On a key
every runner in a race shares -- track+distance, track+distance+going, track,
or a trainer with two in the race -- the previous row is a rival in the *same*
race, so the expanding mean picks up that rival's result.

The structure of the damage matters more than its size: these columns are
otherwise constant across a race, so the leaked same-race outcome was the only
thing that varied within the race, which is exactly the variation a
race-grouped or race-demeaned model keys on. `going_draw_shift`, built on the
smallest cells of all, was the highest-importance pace or draw feature in the
deployed model at rank 40 of 419.

Affected: `td_front_win_share`, `td_holdup_win_share`, `td_avg_winner_pos`,
`track_front_win_share`, `td_low_stall_nfp`, `td_high_stall_nfp`,
`td_draw_bias`, `tdg_low_stall_nfp`, `tdg_high_stall_nfp`, `tdg_draw_bias`,
`going_draw_alignment`, `going_draw_shift`, and everything derived from them
(`draw_bias_alignment`, `track_draw_bias`, `track_style_fit`, `pace_mismatch`,
`draw_advantage_composite`), plus the five `trainer_*` style columns.

Fixed by `model/lagsafe.py`: aggregate to races first, then lag, so a runner
sees every earlier race in its group and no part of its own. The
track x distance x going cell now also requires five earlier races before it
reports a bias at all, instead of handing the model an unshrunk difference of
two means computed from one prior race.

### 13.2 Race pace aggregates built from the race being predicted

`EPF` is parsed from the horse's own in-running comment, so it says how the
horse actually ran *today*. Five race-level aggregates of it were model
features: `RPS`, `pace_pressure`, `prom_runner`, `racepacescore` and
`racepaceindex`. `prom_runner` -- literally "did this horse race prominently
today" -- ranked 35th of 419 by gain.

At prediction time a card has no comments, so the parser returns its default
and all five collapse to constants. The model was trained on information it
never has when it matters. It is worse in backtest: the `--from-db` and
`--last-n-days` paths read completed rows with their comments, so backtests
reproduced the leak while live runs did not, and the two diverged silently.

Fixed by rebuilding all five on `EPF_expected`, the horse's own lag-safe career
EPF. `train_bfsp.py` now refuses to start if any feature column is in
`POST_RACE_ONLY` (`assert_no_post_race_features`), which covers `EPF`,
`placing_numerical`, `NFP`, the raw comment and the parsed run-style columns.
