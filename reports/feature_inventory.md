# Feature inventory, 24 September 2026

This checks every feature named in the project's feature lists against the code at `be2507f`. It also checks them against the model the 06:00 job serves:
- `data/models/bfsp_model.lgb`, feature hash `b31590ee7c5ae461`;
- trained through 22 Sep 2026;
- 501 features, identical in name and order to `ALL_FEATURE_COLS` (`model/bfsp_features.py`).

**Gain** is a feature's share of the model's total LightGBM split gain (`data/models/bfsp_feature_importance.csv`). It shows how much the model leans on a feature to forecast BSP. It does not show whether the feature knows anything BSP does not (§6).

This updates the three read-only audits of 23 September.
- Each of their findings was re-checked against today's code.
- The fixes merged since then are in §5.
- One of their "missing" calls is corrected: A4 (§2).

| Status | Meaning |
|---|---|
| **Served** | Built, and one of the 501 features the live model uses |
| **Served, deviates** | Served, but built differently from the spec |
| **Partial** | Some of what the spec asks for is built |
| **Opt-in** | Built, but only behind a flag or in a research block; never served |
| **Missing** | Not built |

## Summary

| List | Items | Served | Served, deviates | Partial | Opt-in | Missing |
|---|---|---|---|---|---|---|
| The 19 custom metrics (`CLAUDE.md`; spec `claude-code-prompt-custom-metrics (1).md`) | 19 | 12 | 5 | 2 | 0 | 0 |
| Proposals A1–L4 (`FEATURE_ENGINEERING_RESEARCH.md`)¹ | 44 | 19 | 3 | 7 | 2 | 13 |
| Features named in `claude-code-prompt-bfsp-model.md`² | 88 | 84 | – | – | – | 4 |
| Research blocks (`RESEARCH_FRAMEWORK.md` and the research loop) | 14 | 0 | – | – | 14 | 0 |

¹ The document's summary says 42 proposals; it numbers 44.
² 66 are served under the spec's name and 18 under another name (§3).

Where the served model's 501 features come from:

| Source | Features | Share of gain |
|---|---|---|
| The 19 custom metrics | 167 | 67.3% |
| Proposals A–L | 116 | 15.4% |
| Neither list | 218 | 17.3% |

The 218 from neither list are of two kinds:
- **The card's own fields.** Official rating alone carries 5.9% of gain, the median rating in the race 1.3% and days since the last run 1.0%. The rest are career runs, age, sex, weight and trip.
- **Features the framework work added.** These cover running style and pace, form trend (drawdown, MACD, win density), going, distance and course suitability, the rebuilt draw features and expectation residuals.

In short:
1. **The custom metrics are built and served**, apart from 5 of the 56 rank columns and 3 LRP sub-scores.
2. **The model's gain is concentrated.**
   - Within-race ranks carry 41% of it. Prize money won, ranked in the race (`rPMW3/5/10`), accounts for 26% on its own.
   - Past BSP relative to field size (ORR2) carries 14%.
3. **13 of the 44 research proposals were never built.** Most are:
   - the seasonal ideas (J1–J3);
   - the handicap-context ideas (G2, G4, K1, L4);
   - the extra beaten-length measures (B2–B4).
4. **14 research blocks are built, but none is served.** Eleven have been tested against BSP, and none survived (§4).
5. **The served model has open defects** (§5):
   - one feature that is blank on every row;
   - 24 duplicate columns;
   - a bug in FCS.

## 1. The 19 custom metrics

Spec: `claude-code-prompt-custom-metrics (1).md`. Code: `model/custom_metrics.py`. Gain is the total over a metric's served columns. Best rank is the highest any of them reaches, out of 501.

| # | Metric | Status | Served | Gain | Best rank | Notes |
|---|---|---|---|---|---|---|
| 1 | Recency weights | Served, deviates | 6 of 6 | 0.00% | #457 | The spec's weights, but no other metric reads these columns; each rolling metric recomputes its own. Four of the six have zero gain |
| 2 | Confidence intervals | Served | 3 of 3 | 0.07% | #259 | CIL3/5/10 follow the spec's formula, which gives a per-horse data count, not an interval. Nothing uses them |
| 3 | NFP | Served | 5 of 5 | 1.06% | #19 | |
| 4 | RB | Served | 3 of 3 | 0.09% | #185 | The database has no RB column, and the spec's fallback formula is algebraically NFP, so RB duplicates NFP (§5) |
| 5 | xWINRAND / WIV | Served | 12 of 12 | 0.82% | #50 | |
| 6 | WAX / WOA / CWO | Served, deviates | 9 of 9 | 0.87% | #59 | WOA is set equal to WAX (`custom_metrics.py:478`); the spec measures it against the field's average win rate |
| 7 | ORR2 | Served | 8 of 8 | 14.22% | #2 | Past races' BSP only. `LR_ORR2` alone is 9.0% |
| 8 | EPF | Served | 21 of 21 | 0.57% | #75 | Parser rewritten on 23 Sep (§5). Race-level pace uses each horse's expected early position rather than today's comment; the spec's version would have leaked the result |
| 9 | FSS | Served, deviates | 1 of 1 | 0.03% | #283 | `sqrt(sum / count)` where the spec has `sqrt(sum) / count` |
| 10 | FCS | Served, deviates | 1 of 1 | 0.12% | #103 | The same formula change, plus a bug: 0 whenever today's race has no official ratings (§5) |
| 11 | PFD | Served | 3 of 3 | 0.16% | #169 | Past BSP only |
| 12 | WPMRF / PMW | Served | 7 of 7 | 2.38% | #12 | `prize_money` is the race's win prize |
| 13 | OFS | Served | 4 of 4 | 2.10% | #21 | The spec's formula, runners ÷ BSP, is ORR2 again: OFS1 ≡ LR_ORR2, OFSn ≡ LRn_RWO |
| 14 | DSLR | Served | 9 of 9 | 0.03% | #404 | Redundant with `days_since_lr_num` (1.0%) |
| 15 | LRP index | Partial | 4 of 7 | 0.42% | #60 | LRP1–3Score are not served. "Placed" means the first three, whatever the place terms |
| 16 | Race strength | Served | 10 of 10 | 2.06% | #24 | RACE_RB ≡ RACE_NFP. Averaged over the horses that ran in training, but over the declared card live |
| 17 | Pace | Served, deviates | 5 of 5 | 0.26% | #120 | `racepacescore` equals `RPS`. The trainer and jockey pace indices divide a race-lagged sum by a row count |
| 18 | Trainer–jockey | Served | 5 of 5 | 0.63% | #33 | `trainerjockeyWOA` ≡ WAX |
| 19 | Within-race ranks | Partial | 51 of 56 | 41.39% | #1 | See below. `rPMW3` alone is 13.9% |

Five of row 19's ranks were never built: horsexRBMARrank, trainerRBrank, trainerNFPrank, jockeyRBrank and jockeyNFPrank.
- Their source columns, the trainer and jockey career NFP and RB, do not exist.
- The code skips these ranks without a warning (`custom_metrics.py:2110`).

## 2. Proposals A–L

Spec: `FEATURE_ENGINEERING_RESEARCH.md` §2. "Served" counts the proposal's columns in the live model.

| ID | Proposal | Status | Served | Gain | Notes |
|---|---|---|---|---|---|
| A1 | Raw speed rating | Partial | 5 | 0.22% | Built from `comptime_numeric`, which is the race's **winning** time. So it rates how fast the race was run, not the horse. No 10-run window |
| A2 | Speed improvement | Partial | 2 | 0.05% | Inherits A1's race-level time. No trend slope |
| A3 | Best speed rating | Partial | 2 | 0.06% | Inherits A1's race-level time |
| A4 | Going-adjusted speed | Partial | – | – | Folded into A1: the standard time is set per track, trip and going, falling back to track and trip (`custom_metrics.py:1645`). The 23 Sep audit called this missing. Inherits A1's race-level time |
| B1 | Lengths beaten | Served | 5 | 3.34% | `LR_LB` is #7. Parser replaced on 23 Sep (§5) |
| B2 | Field-size-adjusted lengths | Missing | – | – | The unlagged version leaked the result and was removed; there is no lagged version |
| B3 | Lengths per position | Missing | – | – | |
| B4 | Closing-sectional proxy | Missing | – | – | |
| C1 | Sire strike rate | Served | 5 | 0.42% | Shrunk toward the population (k = 20) |
| C2 | Sire going | Served | 3 | 0.36% | The residual-aptitude version is opt-in (pedigree block) |
| C3 | Sire distance | Served | 3 | 0.32% | |
| C4 | Damsire going/distance | Served | 6 | 0.36% | The card fill now supplies the dam's sire live. Debutants' pedigree is still absent from the HTML card |
| C5 | Sire WIV | Served | 2 | 0.25% | |
| D1 | Headgear change | Served | 4 | 0.02% | |
| D2 | Headgear type | Partial | 1 | 0.07% | One category column; no flag per type |
| D3 | Form with/without headgear | Missing | – | – | |
| E1 | Track/distance/going draw bias | Served | 26 | 1.30% | `track_draw_bias` is blank on every row (§5) |
| E2 | Normalised draw | Served | 4 | 0.15% | |
| E3 | Handedness × draw | Served | 5 | 0.08% | |
| F1 | SP-to-BSP spread | Missing | – | – | A same-race version (`mkt_sp_vs_bsp`) is in the opt-in markets block; it added nothing (§4) |
| F2 | Win/place price ratio | Opt-in | – | – | The lagged `LR_mkt_place_ratio` is behind `--market-features`. Betfair price data starts in January 2026 |
| F3 | Favourite flags | Opt-in | – | – | Behind `--odds-features`, and built from the race's own BSP, so it would train on the target (§5) |
| G1 | Race-type flags | Served, deviates | 2 | 1.45% | As category codes (`race_type_cat` is #13), not flags |
| G2 | Horse handicap / non-handicap form | Missing | – | – | Only trainer and jockey race-type splits exist, and they are opt-in |
| G3 | Surface form | Served | 5 | 0.27% | |
| G4 | Age restriction | Missing | – | – | `race_restrictions_age` is never read |
| H1 | Jockey at track | Served | 2 | 0.14% | |
| H2 | Trainer at track | Served | 2 | 0.34% | |
| H3 | Jockey claim | Partial | 1 | 0.02% | Raw claim only; `is_claimer` is opt-in. Live, the fill uses the jockey's last known claim |
| H4 | Trainer 14/30-day form | Served | 5 | 1.17% | |
| H5 | Jockey 14/30-day form | Served | 5 | 0.49% | |
| I1 | OR relative to field | Served | 2 | 1.81% | No percentile. Raw OR is #4 (5.9%) and the median OR #14 |
| I2 | OR trajectory | Served | 4 | 0.49% | No slope |
| I3 | OR vs career best | Served | 4 | 0.31% | |
| J1 | Month/season | Missing | – | – | |
| J2 | Day of week | Missing | – | – | |
| J3 | Horse seasonality | Missing | – | – | |
| J4 | Campaign stage | Partial | – | – | Written in `model/primitives.py`, never called |
| K1 | Major-race experience | Missing | – | – | `major` feeds only the opt-in connections block |
| K2 | Class ladder | Served, deviates | 6 | 0.47% | Measured against a three-run mean, not the career median |
| L1 | Fitness × form | Served, deviates | 2 | 1.04% | A variant: recent form decayed by days off |
| L2 | Class × speed | Missing | – | – | |
| L3 | Debut × sire/trainer | Served | 3 | 0.43% | Debut flag fixed on 23 Sep (§5) |
| L4 | Weight per OR point | Missing | – | – | The handicap block's `hc_wt_vs_mark` is close; it is opt-in and added nothing |

## 3. Features named in `claude-code-prompt-bfsp-model.md`

The spec names 88 features, in `PreRaceBuilder` and in the fundamental model's starting set.
- 66 are served under the spec's name.
- 18 are served under another name (below).
- 4 were never built: trainer career NFP and RB, and jockey career NFP and RB. That is why three of the ranks missing from §1 are skipped.

| Spec name | Served as |
|---|---|
| `last_finish_pos` | `LRNFP` (normalised by field size) |
| `last_bsp` | `LR_ORR2` (runners ÷ last BSP) |
| `last_3_avg_position`, `last_5_avg_position` | `LR3NFPtotal`, `LR5NFPtotal` |
| `days_since_last_run` | `days_since_lr_num` |
| `distance_furlongs`, `race_class_numeric` | `dist_furlongs`, `race_class_num` |
| `weight_lbs`, `draw_position`, `age` | `pounds_num`, `stall_num`, `horse_age_num` |
| `going_preference`, `distance_preference` | `going_from_preferred`, `dist_from_preferred` |
| `weight_change` | `weight_change_lr` |
| `course_win_pct`, `course_runs` | `horse_track_win_rate`, `horse_track_runs` |
| `field_avg_career_nfp`, `_wiv`, `_rb` | `RACE_NFP`, `RACE_WIV`, `RACE_RB` |

The same spec's six components are all in `model/`, and none runs in production:
- `PreRaceBuilder`;
- `FundamentalModel`;
- `BenterBlender`;
- `OverlayDetector`;
- `ModelTrainer`;
- `ModelEvaluator`.

The 06:00 job doesn't use `PreRaceBuilder`: it appends the card to history and recomputes the metrics. The research harness (`scripts/outcome_model.py`) does the job of the first two components: it trains on outcomes, with the market as an offset.

## 4. Research blocks: built, never served

The blocks come from `RESEARCH_FRAMEWORK.md` and the research loop. Each was scored by `scripts/outcome_model.py` against BSP:
- on 40,932 races;
- walk-forward over 2023-01 to 2026-03;
- figures from `reports/research_ledger.jsonl`.

The result is the change in log-likelihood per race over the market alone, in millinats. A value near zero means no information beyond BSP.

| Block | Module | Features | Tested | Result |
|---|---|---|---|---|
| Performance figures (lbs) | `model/perf_figures.py` | 7 | iter8 | Nothing beyond BSP. The six iter8 blocks together (225 features): −0.12 mnats, t −0.64 |
| Kalman rating | `model/state_space.py` | 6 | iter8 | Nothing beyond BSP. On Timeform's performance figures it forecasts the next run better than a weighted mean (error 220 against 233) |
| Pedigree suite | `model/pedigree.py` | in the 225 | iter8 | Nothing beyond BSP |
| Connections | `model/connections.py` | in the 225 | iter8 | Nothing beyond BSP |
| Previous runs' comments | `model/comment_features.py` | in the 225 | iter8 | Nothing beyond BSP. A harness bug, since fixed, left it unattached in iter3 and iter6 |
| Handicap angles | `model/handicap_features.py` | 10 | iter8 | Nothing beyond BSP. It was unattached in iter7 |
| Timeform feed | `model/blandford_features.py` | 18 | iter4, iter9, four audits | See below |
| A/E vs the market, by entity | `model/ae_features.py` | 16 | iter11 | −0.43 mnats (t −1.99): no persistent mispricing |
| The race's other markets | `model/market_block.py` | 7 | iter5 | −0.26 mnats (t −1.0) |
| In-day (earlier races today) | `model/inday_features.py` | 21 | iter12–16, 2022 gate, holdout | See below |
| Betfair price-file features | `model/market_features.py` | 26, plus 4 same-day | The morning-to-off move was tested directly | See below |
| Odds metrics | `model/odds_metrics.py` | 16 | Not scored | Uses the race's own BSP by default, so it leaks |
| ABM race simulator | `model/abm/` | 16 | Not run at scale | No feature file has been built |
| HRB rating sets | `model/hrb_features.py` | 7 sets | Pilot (`88ff402`) | None added anything; a knockoff filter kept none |

Notes on three of the blocks:
- **Timeform feed.** Its same-day fields leak the result in 2026 and are banned. Its lagged figures beat BSP only in data before April 2025, which is the signature of ratings revised after the fact. It is not a candidate.
- **In-day.** The rank-1 rule returned +20.2% in development, +5.1% on the 2022 gate, and −18.7% on the locked holdout. It was rejected.
- **Betfair price-file features.** These were never scored as a block. The move from the morning price to BSP was tested directly, and it carries nothing beyond BSP. The price data only starts in January 2026.

The Stage F market-free model (`model/stage_f.py`, `train_stage_f.py`) is a model rather than a block:
- **On Timeform features**, it added ΔR² +0.00001 over the market (90% interval −0.00014 to +0.00017).
- **As a two-stage model on the production features** (iter6), it was worse than the market alone: −0.40 mnats, t −3.06.

## 5. Defects

### Fixed since the 23 September audits

All of these are in the served model since the 24 September retrain.

| Defect | Fix |
|---|---|
| Trainer, jockey, sire, course and pace priors stepped back one race. A 4.10 runner's figures included the 2.00 result on the same card, and live, that race's runners counted as losers | Priors step back whole days (`model/lagsafe.py`, `LAG_UNIT = "day"`; `a365621`) |
| The early-position parser matched "led to" inside "failed to", so "held up in rear, failed to pick up" scored 6.0, a front-runner's score | It reads the first positional phrase on word boundaries. That comment now scores 1.0 |
| The lengths parser read a blank margin as 0, a winner's margin, and "1½" as 1 | It uses the tested parser from `perf_figures` |
| The debut flag marked the first run at the shortest trip | It is taken from the horse's count of earlier runs |
| Trainer and jockey run counts included their other runners that day | They count earlier days only |
| The live card lacked eleven fields the model reads, including race type, surface, sire, dam's sire, median OR and claim. A missing category was scored as its first value, so every race read as a Beginners Chase on a "Beach" surface | `model/card_enrich.py` fills them (`d086c0a`, `279b122`). The parity check closes about 98% of the gain-weighted gap |
| Serving read history from 2020, but training started in 2021 (QA M6) | Serving starts where the model's training did (`2da8533`) |

These fixes are confirmed in the code and by 50 passing tests in:
- `tests/test_input_repairs.py`;
- `tests/test_card_enrich.py`;
- `tests/test_leakage.py`.

### Still open

| Defect | Where | Effect |
|---|---|---|
| `track_draw_bias` is blank on every row | `draw_metrics.py:643` | Zero gain. A row-by-row lag within each track leaves one side of the low-minus-high difference blank. Filling it would leak a rival's result from the same race |
| 24 duplicate columns: RB ≡ NFP (9), WOA ≡ WAX (7), OFS ≡ ORR2 (7), racepacescore ≡ RPS (1) | `custom_metrics.py:419, 478, 738, 848` | They share 4.35% of gain with their twins. They do no harm to accuracy, but they split importance readings |
| FCS is 0, "stable class", when today's race has no official ratings | `custom_metrics.py:656` | The missing difference is filled with 0 while the count still rises |
| Five spec rank columns were never built and are skipped silently | `custom_metrics.py:2110` | §1, row 19 |
| Zero-gain features: LR3COUNT, LR5COUNT, LR3wsum, LR5wsum, is_debut, track_draw_bias | – | The first five are exact functions of career runs |
| Race aggregates and ranks cover the horses that ran in training, but the declared card live | – | Late non-runners shift every rank live. The forward test reports short-book races separately |
| The speed family measures the race, not the horse | `custom_metrics.py:1628` | §2, A1–A4 |
| `--odds-features` would feed the race's own BSP into training, and the leak guard does not list those columns | `train_bfsp.py:1093` | Opt-in, not in production |
| The card fill cannot know three things at 06:00: debutants' pedigree (absent from the HTML card), geldings since the last run, and the claim for today's ride | `model/card_enrich.py` | Most of the remaining ~2% parity gap |

## 6. Implemented is not the same as useful

The served model forecasts BSP, and gain says how much it leans on a feature to do that. Tested against the result, with BSP as the baseline, no served feature adds information.
- **The screen.** It was fitted before July 2024 and scored on the 22,082 races after. 1 of 442 production features reached t > 2, where about 10 would by chance. None reached t > 3.
- **The blocks.** None of the blocks tested in §4 changed that.

What the features do predict is the move from the morning price to BSP (joint ΔR² × 1000 = +52.7, t +10.4). That is the mechanism behind the early-price trade.
- It passed out of sample: +3.54% net CLV, April to September.
- Its forward test opens with the 06:00 run on 25 September (`reports/preregistration_clv_forward.md`).

Most of the 13 missing proposals are cheap to build. Every fundamental feature tested so far adds nothing beyond BSP, though. So they are more likely to sharpen the BSP forecast behind the early-price trade than to make a rank-1 selection profitable.

Two things cannot be built from this data, because it has no sectional or per-horse times (`RESEARCH_FRAMEWORK.md` §15.8):
- B4, the closing-sectional proxy;
- true per-horse speed figures.

Two other documents were not re-checked:
- **The racing² master framework v3.1.** `RESEARCH_FRAMEWORK.md` §14 counted its 277 items as 75 implemented, 80 partial and 122 missing. §15 then built part of the missing set without a recount. The framework document itself is not in the repository, so its items cannot be re-checked one by one.
- **`live.md`.** It specifies the betting pipeline, not features. Its status is in `reports/qa_review_2026-09-23.md`.
