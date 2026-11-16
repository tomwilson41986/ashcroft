# MAP: repo_features

# MAP repo_features

Gain tags: **FI** = `reports/feature_inventory.md` (501-feature model, 24 Sep); **BP** = `reports/bfsp_model_plan.md` (1,006-feature model, 10 Nov). cm = `model/custom_metrics.py`. "Served" = in `ALL_FEATURE_COLS` (`model/bfsp_features.py:589`); the other ~505 served columns come from `model/blocks/` (not mapped here).

## 1. Feature families

**Served - engine**
- NFP (career, `LRNFP`, `LR3/5/10NFPtotal`) - (N - place)/(N - 1); lagged career mean, last run, rolling means - 1.06% FI - cm:453
- RB / FSARB / FSARB2 - algebraically NFP; x N/median field; squared - duplicate, 0.09% FI - cm:480
- WIV - cumulative wins / cumulative 1/N, for horse, trainer, jockey; plus Wins/Runs/Places - 0.82% FI - cm:507
- WAX / WOA / CWO - won - 1/N as mean and as cumulative sum; WOA is a copy of WAX - 0.87% FI - cm:536
- ORR2 / RWO - N/BSP of past runs: last run, harmonic (1/k) weighted sum and mean - 14.22% FI (`LR_ORR2` 9.0%) - cm:564
- OFS1/3/5/10 - N/BSP again (exact ORR2 copy) - 2.10% FI - cm:801
- PFD3/5/10 - 1/BSP - 1/N, harmonic weighted - 0.16% FI - cm:732
- EPF - regex style score 1-6 from the first positional phrase of past comments; EPF2 = -0.74 + 0.8637 EPF + 0.09375 N; EPF3 = EPF x NFP; lags 1-5; career means; race sum (`RPS`), share above 4 (`pace_pressure`) - 0.57% FI - cm:217, 607
- FSS / FCS - RMS gap between today's field size / race mean OR and the last 10 runs' - 0.03% / 0.12% FI - cm:672, 701
- WPMRF / PMW / RACE_WPMRF - harmonic-weighted race prize; prize/100 x RB²; race prize sum - 2.38% FI - cm:760
- DSLR1-4, diffs, WgtDSLR, FinalDSLR - gaps between consecutive runs and their weighted differences - 0.03% FI - cm:829
- LRP - last-run placing scores 17.41/14.77/12.62 summed per jockey, per ride - 0.42% FI - cm:867
- Pace indices - race sum/mean of expected EPF; horse/trainer/jockey cumulative EPF per run - 0.26% FI - cm:906
- Trainer-jockey pair - WIV, NFP, WAX, WOA, CWO on the pair key - 0.63% FI - cm:937
- Race strength (`RACE_*`, `LR_RACE_*`) - race mean of the field's career RB/WIV/NFP/Wins/WOA, today and at the last run - 2.06% FI - cm:961
- Recency counts, CIL - runs held in window, weight sums, count/(N x w) - ~0 - cm:989
- Within-race ranks (`r*`, `*rank`) - descending rank, missing last, of almost every metric above - 41.39% FI (`rPMW3` 13.9%); 20% BP - cm:2074
- EXP_ decay - 0.85^k weighted NFP/RB/ORR2 over 3/5/10 - cm:1022
- Expectation residuals - NFP less NFP implied by BSP rank (career, decayed 3/5); career mean of won x log BSP - cm:1057
- Unexposure - debut, counts and first-time flags at trip/going/course/C&D, composite, mean NFP per condition - cm:1109
- Class movement - class vs last run and 3-run mean, drop/rise flags - 0.47% FI - cm:1189
- Distance / going preference - best-NFP trip and going, distance from it, change since last run - cm:1228, 1269
- Form trajectory, consistency - OLS slope and variance of last 3/5 NFP (sign inverted, BP fault 11); NFP sd, win/place rates - cm:1349, 1396
- Weight - vs race mean/min, range, change - cm:1431
- Pedigree levels - sire/damsire win, place, NFP, WIV overall and by going/trip, shrunk k=20; debut x sire/trainer - 0.5% BP - cm:1462
- Speed (RSR, SFI) - (standard - winning time)/standard x 100, standard = lagged mean by track x rounded trip x going; career, last, 3/5, best, change; rates the race not the horse - ~1% BP - cm:1674
- Lengths beaten (LB) - parsed cumulative margin: career, last, 3/5 - 3.34% FI - cm:1763
- Equipment, surface, track preference - headgear change flags; horse/trainer/jockey NFP and win rate on surface and at track - cm:1809-1905
- OR trajectory - change over 1/3 runs, vs career best, vs last winning mark - cm:1962
- Hot form - trainer/jockey strike rate 14/30 days, runs, delta - cm:2014
- Context - N, trip, class, going scale, category codes (surface, race type, track, sex, headgear), age, weight, stall, career runs, OR, race max/median OR, claim - OR 5.9% FI - `bfsp_features.py:496`
- Financial - RSI, MACD, Bollinger z, Sharpe, acceleration, skew, drawdown on the NFP series; win density 30/60d; prize/class/OR momentum; 11 ranks - ~1% BP - `financial_features.py:15`
- Pace engine (91) - multi-phase comment parse (early position, mid/late move, keen, trouble); `epf_norm` profile (mean, sd, mode); jockey/trainer style rates; course x trip front-runner win share; predicted scenario, lead-probability softmax (T 0.12), entropy; pace fit; sustainability; position-churn baseline - 1.4% BP - `pace_metrics.py:939-1072`
- Draw engine (60) - stall renumbered after non-runners, handedness, rail move, low-high NFP bias by course x trip (x going, x field size), rating-residual bias per stall with hierarchical shrinkage and 730-day decay, split-half screen, stall x style, horse draw preference - 2.7% BP - `draw_metrics.py:1088-1201`
- Intent (`it_`) - 8 angle flags (new yard, handicap debut, gelded, first headgear, layoff, class drop, code switch, debut); trainer's decayed shrunk mean of (won - BSP probability) per angle; sum/best/count - 1.3% BP - `intent_features.py:62`
- Freshness (`fr_`) - log gap vs the horse's usual, the yard's, the field's; runs since break; workload 30/60/90d; days since win/place/code - 1.7% BP - `freshness_features.py:57`
- Form windows (`fw_`) - 11 per-run measures (win, place, WAX, NFP, pounds beaten, race-centred pounds, OR-scale figure, RSR, market view, A-E, NFP vs market order) x 7 windows - with variants 10% BP - `form_windows.py:56`

**Built, not served**
- Race shape, draw curve - projected style distribution, P(no leader/contested), value of a position in that shape by course x trip x field; draw-fifth edge in NFP and pounds - iteration 25, no gain - `race_shape.py:626`, `draw_curve.py:65`
- Shape form (`sf_`) - past runs net of pace and draw help, windowed; expected leaders drawn inside/outside/near; market A-E by draw cell - iteration 34, nothing - `shape_form.py:71`
- GP draw surface, Moran's I, going drift - `spatial.py:79-249`
- Betfair price-file lags - in-play low/BSP, steam, volume share, drift rate, Parkinson/Garman-Klass volatility, win/place ratio, yard steam rate - `market_features.py:86`
- Same-race markets - ln normalised BSP/SP/place, SP vs BSP, place vs Harville-implied - -0.26 mnats FI - `market_block.py:33`
- Odds metrics - OTRR, gaps, Shin probabilities, entropy field size, OFSR - leaks target - `odds_metrics.py:60`
- A/E by entity - decayed shrunk won - p for trainer, jockey, sire, pair, horse, trainer x track, track x draw - -0.43 mnats FI - `ae_features.py:28`
- In-day - earlier races today: draw/pace slope of beat-the-market residual, connection A/E today and 72h - holdout -18.7% FI - `inday_features.py:57`
- Comment flags (trouble, slow start, keen, wide, finished well, tender, weakened, problem, easy win), lagged - `comment_features.py:27`
- Connections suite - schedule-expected strike rate and residual, context aptitude residuals, class of runner, booking upgrade, A/E, ROI - `connections.py:432`
- Pedigree suite - aptitude residual vs the sire's own mean, dam, siblings, nick, family, median handicap-debut OR - `pedigree.py:512`
- Handicap angles - weight vs mark, penalty edge in pounds - `handicap_features.py:29`
- Performance figures, Kalman rating - `perf_figures.py:179`, `state_space.py:38`
- Primitives - %RB², tanh-truncated margins, finish geometry (gaps, isolation), par surfaces, strength of schedule, PvS, rating-based effective field - `primitives.py:190-1118`

**Retired / unused**: Timeform feed (`blandford_features.py:31`, banned), HRB rating sets (`hrb_features.py:40`), Stage F set (`phase1.py:47`), legacy `features.py:206`, `prerace_builder.py` (FI:163-171).

## 2. Engineering methods in use
- Day-lag as-of priors: aggregate to race, read strictly earlier days - `lagsafe.py:38, 78`
- Horse shift(1) expanding/rolling means, sd, max - cm:36-113; trailing calendar windows excluding today - cm:125
- Decays: harmonic 1/k (cm:185); 0.85^k (cm:1029); linear k..1 (`form_windows.py:155`); half-life in runs (`race_shape.py:404`); half-life in days with effective n (`race_shape.py:273`, `draw_metrics.py:209`); fitted lambda (`primitives.py:635`)
- Empirical-Bayes (s + k m)/(n + k), method-of-moments k (unused in served), shrink chain - `shrinkage.py:40-83`; hierarchical cells pooled coarsest first - `race_shape.py:332`
- Aptitude as residual against the entity's own mean - `connections.py:131`
- Within-race rank, z, rank percent, vs max/median/2nd, share - `primitives.py:834`; race-centred outcomes - `race_shape.py:239`
- Field-size normalisation (NFP, `epf_norm`, sqrt((N-1)/(N+1)) credibility) - `primitives.py:82-97`
- Residual against the market (won - p; NFP vs price rank) and against ratings with a lagged calibration slope - `draw_metrics.py:330, 363`
- Lengths to pounds by trip, cap 20, winner credit cap 5 - `perf_figures.py:26`, `form_windows.py:66`
- Regex comment parsing (first phrase, ordinals, excuses) - `race_shape.py:152`
- Kalman local level with gap-scaled drift - `state_space.py:38`; GP smoothing, walk-forward refit - `spatial.py:46, 173`
- Leave-one-out (siblings, opponents' IQM) - `pedigree.py:347`, `primitives.py:749`; running z vs earlier-day population - `connections.py:162`
- Range volatility, Shin de-margining, entropy field size - `market_features.py:78`, `odds_metrics.py:134, 179`
- Fixed categorical vocabulary, unseen = -1 - `bfsp_features.py:693`; post-race and card-unsafe guards - `bfsp_features.py:635`; card fill - `card_enrich.py:308`; Stage F/C registry - `feature_registry.py`

## 3. What the model leans on most
- Within-race comparisons 57% of gain: z/gap 37%, ranks 20% (BP:45, 125-126).
- Prize-money features 14.6%; past market prices 18.9% (BP:56).
- Top features: `rw_fv_nfp_s8_gap` 6.4%, `rPMW3` 3.3%, `rw_fv_nfp_e3_gap` 3.2%; top 10 = 28%, top 100 = 67% (BP:62-78).
- Lengths/pounds beaten: largest loss when withheld (BP:130); NFP family withheld +0.0048 (`reports/metric_audit.md:43`).
- Largest gain on record: market rating -0.0067 on holdout (`reports/model_review_1111.md:15`).
- No served feature beats BSP on results: 1 of 442 at t > 2 (FI:246); the features predict the morning-to-BSP move (FI:249).
- Tested, no gain: pedigree blocks, pace/draw rebuilds, result Elo, comments, recency/profit weighting, other losses (BP:92-95).

## Blind spots of this map
- No database, S3 or model artefacts reachable: `data/models/bfsp_feature_importance.csv` was not opened, so every gain share is quoted from the two reports, not re-derived. The two reports describe different models (501 features on 24 Sep vs 1,006 on 10 Nov), so FI and BP shares are not comparable with each other; several families have a figure from only one of them, and some families (unexposure, weight, distance/going preference, equipment, EXP_ decay, residuals) have no family-level gain stated in either report - none was invented.
- `model/blocks/` (about half of the served 1,006 columns, including the top features `rw_`, `rr_`, `fv_`, `hca_`, `mr_`) was deliberately not mapped - only the file list was seen. The served-block membership (the "14 served blocks") was not confirmed from code.
- The four largest modules (custom_metrics, pace_metrics, draw_metrics, primitives) were read by docstring, feature lists and function signatures, plus the full body of custom_metrics lines 300-1108 and selected later functions; the bodies of unexposure, class movement, distance/going preference, weight, pedigree, equipment, surface/track, OR trajectory, hot form, and the pace/draw engine internals were not read line by line, so their one-line descriptions rest on docstrings and the audit reports. Line numbers for those come from an AST listing.
- `FEATURE_ENGINEERING_RESEARCH.md` was read by outline only; proposal status (A1-L4) is taken from `reports/feature_inventory.md` section 2. The 13 never-built proposals (J1-J3 seasonal, G2, G4, K1, L2, L4, B2-B4, D3, F1) exist as written ideas and should be treated as known, not new.
- Not opened: `model/abm/`, `model/feature_registry.py` beyond its docstring, `model/feature_cache.py`, `model/effects.py`, `model/causal.py`, `model/trainer.py`, `reports/research_ledger.jsonl`, `RESEARCH_FRAMEWORK.md`, `reports/feature_audit.md` (may not exist yet). The racing² master framework v3.1 (277 items) is not in the repository, so ideas listed there but unbuilt are a blind spot.
- Open defects noted in the reports (e.g. `track_draw_bias` blank, WOA = WAX, speed figure uses the winning time) were not re-verified against current code beyond the lines read.
- No web search was needed or done for this map.
