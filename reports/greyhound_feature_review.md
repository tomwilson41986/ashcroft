# racing² · Greyhound model: data and feature review

17 Nov 2026. The owner: "take a step back. Look at all data - what more can be utilised or add to our model? Feature
engineering - how can we improve our features? Go block by block, feature by feature."

This review was read from the code, the research ledger (`reports/research_ledger.jsonl`) and
`reports/greyhound_model.md`. Nothing new was measured for it. Every value rating (High, Med, Low) is a judgement
until a registered batch measures it. Items marked **verified** were checked in the code for this report. Every other
code claim cites the file and line it comes from.

## 1. Results first

**What the record says.**
- The live recipe is `stack_mlsi`. Its RMSE of log(BSP) is about 0.278 (1 Jun to 10 Nov test), and 0.2777 out of sample
  from 6 Oct (ledger greyhound-world-batch46-1116, amendment).
- No blend of the model with the Betfair SP has beaten the SP alone.
- What has paid:
  - **Model form:** the share objective -0.0241; the stack of trees and networks -0.009 to -0.011; three seeds -0.004.
  - **Market-implied ratings:** the figs rating -0.027.
  - **New information the model lacked:** trials -0.010; Irish form -0.0016, replicated at -0.0029.
  - **Better-built figures:** the fixed engine, expected time, speed and league tables.
  - **Tables of how the market reads a dog:** optics, line, salience, last race.
- **What has failed, repeatedly:** fresh cuts of information the model already holds. These include going variants,
  bends from comments, the linked first sectional, origin, pedigree in any form, the Irish market and weights, and
  most discovery tables.

**The lesson for what to do next:**
- Fix the bugs.
- Correct how the existing measures are built: de-biasing, trap adjustment, forgetting, fitted constants.
- Add genuinely new information: the trainer on the day, the adjustment factor, price paths, place prices.
- Do not add more variants of features the model already has.

## 2. Bugs found (fix before any new test)

| # | Bug | Where | Effect | Status |
|---|---|---|---|---|
| B1 | A pit table counted as "attached" when any column started with its prefix. The engine's `is_handicap`, `is_card` and `is_trial` start with irish_style's `is_` | `greyhound/blocks/__init__.py:37-38` | `pit_irish_style` would never have joined. Batch 51 would have scored the baseline against itself | **verified, fixed** (prefix `irs_`, attach by the table's own columns, collision tests; ledger amendment). No earlier batch affected |
| B2 | Trials never reach ability, life or breeding: `data.clean` drops every trial (no off time) | `greyhound/data.py:95-97`; `pit/life.py:262-276`; `pit/ability.py:55-61`; `pit/breeding.py:151` | `lf_trials` is always 0; `lf_days_since_trial` and `lf_trial_vs_level` are always NaN; the ability filter's trial noise and breeding's "trials included" are dead. Only xtime and the trials table read `ctx["trial_runs"]` | **verified** |
| B3 | `core.trainer_at_the_time` blanks the trainer on the card row: today's date has no GBGB fetch record | `pit/core.py:150-156` | This is why `cx_tr_*`, `cx_new_trainer` and `mk_tr_*` sit on the card-unsafe drop list (`live.py:73-79`). The card does carry a trainer (`card.py:95-100`) | **verified** |
| B4 | The engine's trainer features read the trainer GBGB serves today on every past run | `metrics.py:317-322` (`trainer_win`, `trainer_ae`), `parity.py:147-213` (`gp_trainer30_*`), `hrb.py:160-170`, `intent.py:45` | Possible look-ahead (good dogs move to strong yards). The trainer and owner columns are worth 0.0058 (z +25.1 when dropped). Whether that is real or leaked is not known (ledger greyhound-leak-league-batch37-1114) | from review; the leak itself unmeasured |
| B5 | Every A/E and market rating uses proportionally normalised SP | market, connections, breeding, draw, `figs.market_rating` | The favourite-longshot bias leaks into "A/E": habitual favourites read high, outsiders low | from review |

## 3. Data: what we hold and do not use

Ranked by value for cost. Being "known before the off" is required for a live feature.

| # | Data | Held? | Use | Value / cost |
|---|---|---|---|---|
| D1 | **Trainer on the day** from GreyhoundAPI's GB results (`trainer_id` on each run, `sources/greyhoundapi.py:441`), 2013 on | yes | Rebuild every trainer feature on the trainer at the time (fixes B4). Check it on dogs known to have moved yard (gbgb_dogs snapshots from 14 Nov). Whether the API's trainer is historic is **unverified** | High / low |
| D2 | **Betfair adjustment factor**: Betfair's own pre-market chance for each runner | in the book record (`betfair_recorder.py:57,169`); dropped by `betfair_historic.py rows` | An independent prior for every runner, set when the market is created. Add it to the Historic Data parser and backfill. Whether greyhound markets carry it, and when it is revised, is **unverified**; use the first value only | High if present / low |
| D3 | **Past-run price paths**: morning WAP, pre-play WAP and volumes from Betfair's price files | yes (`prices_<year>.parquet`) | The dog's history of steaming or drifting against its BSP, and market attention (volume). Today these are labels only (`fastloop.py:56-59`); no table reads them from past runs | Med-High / low |
| D4 | **Place-market BSP history** | yes | Place/win BSP ratio of past runs: the market's view of how consistent the dog is | Med / low |
| D5 | **Today's trainer from Betfair's catalogue** | yes (`card.py:66`, used only at :98-100) | Makes a trainer change known on the card. Needs D1 for its history | Med / very low |
| D6 | **GreyhoundAPI runner lists** (scratched and reserve status, owner) | raw held, never built | Reserve and non-runner history; owner as a connection (if historic) | Low-Med / low |
| D7 | **GreyhoundAPI weather** for each race, Sep 2025 on | yes, unused | Fills NOAA's gap after Aug 2025 for past-run figures. Coverage **unverified** | Low-Med / low |
| D8 | **Race title** (finals, heats, sponsored opens) | yes (`gbgb.py:174`) | A class signal beyond the grade | Low-Med / low |
| D9 | **Prize money** of past races | yes | Level of races met | Low / low |
| D10 | **Season, trainer and owner as on the day** (gbgb_dogs daily snapshots) | building since 14 Nov | Correct season data going forward only | Med later / none |
| D11 | **Betfair forecast price** (block built, `blocks/fcast.py`) | from 6 Oct 2026 only | As a blend layer after the model, once months build up, not as a tree feature | Med later / low |

**Not held, worth obtaining:**
- Weigh-in before the off: the best research model used it; source and timing **unverified**.
- Racing manager's seeding: railer, middle or wide (GBGB rule 78).
- Irish trials and whelping data (GRI asks consent for automated collection).
- A hand-kept list of track events (resurfacing, rail moves).

**Already tested and not worth chasing again:**
- origin, Timeform-style going, bends from comments, Irish SP and weights, pedigree abroad, secfig;
- on-the-day weights from GreyhoundAPI (not published before the off for GB or Ireland);
- extra GB sectionals (the API's equal GBGB's to the hundredth).

## 4. Features, block by block

### Engine (`greyhound/metrics.py`) and the base blocks

| Family | Weakness | Improvement | Value / cost |
|---|---|---|---|
| `gsr` (metrics.py:204,222): about 15 consumers | All-history expanding standard; fixed 0.08 s a length at every trip; not linked across tracks | Percent of a rolling, linked standard (the speed table's link offsets, `pit/speed.py:228-262`), with the consumers reading it | High / M |
| `esr`, `fsr` (224-225) | esr has no going, no trap par and no link; fsr takes the whole night's going | esr from secfig's linked, grade-aware sectional; fsr with its own run-home going. Feeds sim, bend, draw and the squeeze columns | High / M |
| `trainer_win/ae`, `gp_trainer30_*`, `hb_` trainer columns | B4 | Trainer at the time (D1), or drop and re-test | Med-High / S |
| Count windows (`*_mean/max{1,3,6}`, 105 `gp_` columns) | No time decay; adjacent windows near-collinear | One day-decayed level per measure; keep last, w5, career | Med (pruning) / S |
| `trap_bias`, `sire_win`, `grade_par`, `trk_gsr`, `cd_gsr` | Unshrunk or superseded (by dr_, pg_, league) | Drop or shrink | Low / S |
| `tn_*`, `lb_kr`, `et_*` | Superseded by fg_, kf_, xt_ | Drop most (but see 6: dropping `kf_`, `et_` and `tn_` once gave back two thirds of the xtime gain) | Med (pruning) / S |
| `sim` block | Hand-set constants (sim.py:24-29); uncorrected inputs; adjacent-trap collisions only; median for unknown dogs | Fit the constants on leader and winner likelihood; feed linked sectionals; add lane; trial and debut priors | Med-High / M |
| `wpred` | Yearly refit (a January row uses a model about 12 months old) | Monthly refit | Low / S |

### Core tables

| Table | Main weakness | Best improvement | Value / cost |
|---|---|---|---|
| **figs** `fg_mr` (the strongest family) | Trap sits inside the observation (a strong trap raises the share with no change in the dog); proportional SP; one noise level for every share; no mean reversion after layoffs | Subtract the track-trip-trap mean log share before the update; share-dependent noise; de-biased SP; a second filter fed by result surprise | **High** / M |
| figs `fg_tot/sec/run` | Not linked across tracks; hand constants (half-life 90, prior 2, troubled-run weight 0.5) | Linked figures in percent; keep form's fitted versions | Med / M |
| **xtime** | Trap and grade effects are cumulative means that never forget (xtime.py:373-382); constants K_TRAP, K_GRADE, K_PREF, STD_CAP, NOISY fixed; preference updated on the same residual as the level (double count, 383-386); share ignores `xt_tot_sd`; a vacant neighbour box gives NaN (bend treats it as room) | Exponential forgetting with a fitted rate; fit the constants; update preference on the post-level residual; P(fastest) with the sd; vacant box as room | **High** / M |
| **market** `mk_*_ae` | +1 shrinkage is weak; favourite-longshot bias (B5); no decay; `mk_drift5` compares an overround SP book with a near-100% BSP book | Gamma-posterior log A/E and z-score on de-biased probability, decay fitted; drift on race-normalised shares | High / low |
| **breeding** `br_` | The dog's own runs and its littermates' sit inside its dam's and sire's means; weighted by runs not dogs; hard cut at 20 | Leave-one-dog-out, average a progeny first, empirical-Bayes shrinkage; or serve `pedigree.py` | High for lightly raced dogs / low |
| **trials** `tl_` | No trial-against-race offset (solo trials run differently); no going for trial sessions; position in a trial unused | Fit the offset from dogs that trial then race within 21 days; a session variant; trial margin | High for debutants and returners / low |
| **ability** `kf_` | Random walk with no reversion over layoffs; no trend; one noise across trips; dead trial noise (B2) | Damped level towards the grade par; noise per trip band; feed trials | Med / M |
| **trouble** `tr_` | A second pricing of the codes in other units from xtime's; confounded by pace and trap | One structured pricing (type, bend, severity) shared with xtime; `tr_excess` = actual less expected trouble given trap and pace | Med / M |
| **connections** `cx_` | Dropped on the card (B3); proportional SP; sparse cells shrunk to 1 | Fix B3; decayed A/E with sd; shrink situations to the trainer's own A/E; trainer at track | High once B3 and D1 are fixed / low |
| **draw** `dr_` | No hierarchy across trips; `dr_lead` mostly measures which dogs are seeded in the trap | Prior from the same trap at the track's other trips; lead less the dogs' own lead chance | Med / M |
| **bend** `bd_` | Trap enters linearly and the same everywhere; raw unshrunk inputs; Monte Carlo expected position | Track-specific trap terms and draw-state input; shrunk pace input; exact closed form `1 + Σσ(u_j - u_i)`; a crowding probability | Med / low |
| **life** `lf_` | Three empty trial columns (B2); grade counts pooled across tracks | Delete the dead columns; key on track and class | Low / trivial |
| **form** `fm_` | Slope extrapolated over the whole gap; age curve from consecutive runs (survivor bias) | Damped trend; within-dog age curve | Med / low |

### Newer live tables

| Table | Weakness | Improvement | Value / cost |
|---|---|---|---|
| **speed** `sf_` (the best figure set) | One winners-only going; sectional not linked; fixed blend 0.5; half-life in runs not days | Clean-runner going by side and half (the going table's method) as its variant; linked sectional; fit the blend; day decay | Med / M |
| **seq** `se_` | One seed; tokens have no going or track; its price head is fitted to the bookmaker SP, not the Betfair SP | 3 seeds; a BSP-share head; speed figures as tokens | Med / M |
| **irish** `ie_` | Irish figures not linked to the GB scale | Link offsets on dogs that move, plus a walk-forward transfer of the last Irish figure to the first GB figure | Med / M |
| **lastrace** `lr_` | Several exact duplicates (lr_ae_last = gp_ae_l1); race speed with no meeting variant | Drop the duplicates; subtract the variant | Med-Low / S |
| **line** `ln_` | Neighbour squeeze ignores whether the neighbour arrives at the same time | Arrival-weighted squeeze (as `nx_` does), traps 1 and 2 away | Med / S |
| **optics, salience** | Overlap on 3 of 5 attributes; equal-weight look | One card table with a fitted card-reader | Med / M |
| **league** `lg_` | Fixed half-life and K; weekly refit misses this week's results | Fit constants; within-week update | Low-Med / M |

## 5. Model structure

- **Race-context features**: spread of ratings, top-two gap, number of debutants and unrated dogs, early-pace pressure.
  Also a per-track stretch that varies with race uncertainty.
- **Pace collision**: for each pair of traps 1 and 2 apart, the closeness of expected arrival, times lane convergence,
  times the neighbour's harm. This replaces the binary squeeze and sim's unfitted crowding.
- **Recency row weights** in the share objective, half-life 6-12 months. The weight argument exists
  (`fastloop.py:253-271`) but is unused by the stack.
- **Track and distance band as categoricals**, or models by band (sprint, standard, stayers, hurdles).
- **The two memory layers that passed but are not live**: residual memory -0.0034 and decay memory -0.0019 (ledger
  L734, L784). Serving one is the largest measured gain still on the shelf.

## 6. Pruning

About 700 live columns fall into roughly nine near-duplicate families: level, early pace, finish, market memory, trap
bias, line, class move, freshness and race-relative triplets. A realistic target is about 300 columns, using a
correlation cluster plus grouped permutation importance on the early-stopping fold.

Pruning alone should be neutral to slightly positive. It buys 2-3x faster fits, which can pay for more seeds.
Caution from the record: dropping `kf_`, `et_` and `tn_` once gave back two thirds of the xtime gain (batch 23). So
merge or fix before dropping, and register each removal like an addition.

## 7. The plan

Each step is one registered batch against the live recipe, with the usual bar: z of -3 or better, a gain of at least
0.0010, log-loss no worse, and the gain in both periods. They run one fast loop at a time.

| Wave | Batch | Content | Why first |
|---|---|---|---|
| 0 | fixes | B1 (done), B2 (feed trials to ability; delete the dead life columns), B3 (card-row trainer) | Correctness; cheap |
| 1 | trainer | D1 trainer at the time across engine and tables, then re-test `cx_tr_`, `mk_tr_` and the engine trainer columns | Settles the 0.0058 leak question and may unlock real signal |
| 1 | debias | Shin or power de-biased SP in every A/E and market rating; gamma-posterior A/E; normalised drift | Touches the strongest family (figs) |
| 1 | mr_trap | Trap-adjusted market-rating observation with share-dependent noise | figs is the top family by importance |
| 2 | xtime2 | Forgetting, fitted constants, preference fix, sd-aware share, vacant box | The second family by importance |
| 2 | trials2 | Trial offset, session variant, trial margin | Debutants are the worst-priced group |
| 2 | breeding2 | Leave-one-dog-out empirical-Bayes breeding | Lightly raced dogs |
| 2 | prices | D3 price paths and D4 place ratios from past runs; D2 adjustment factor if present | New information |
| 3 | engine2 | Linked percent gsr; esr and fsr on linked, grade-aware sectionals; sim refitted | Wide rebuild (a cache rebuild) |
| 3 | memory | Serve the residual or decay memory layer | Largest gain already measured and not live |
| 3 | prune | Cluster pruning to about 300 columns, plus a seed spent from the saving | Speed |
| 3 | context | Race-context features, pace collision, recency weights, track categoricals | Structure |

Already registered and waiting:
- batch 50 (sectional table);
- batch 51 (Irish run style, after the comments backfill);
- the forward tests (from 25 Nov, 2 Dec and 7 Dec);
- the Australian paper test.

## Sources

The four reviews behind this report were read-only and covered:
- every data source and field (`sources/`, `betfair_recorder.py`, `betfair_historic.py`, `greyhound/card.py`);
- the core tables (`greyhound/pit/{core,readers,ability,trouble,connections,market,breeding,life,figs,form,draw,bend,xtime,trials}.py`);
- the newer tables, engine and blocks (`greyhound/metrics.py`, `parity.py`, `hrb.py`, `lib.py`, `times.py`, `intent.py`, `etime.py`, `blocks/*`, the other `pit/*`);
- the evidence (`reports/research_ledger.jsonl`, `reports/greyhound_model.md`, `research/greyhound/experiments/*.json`).

Numbers quoted from earlier batches cite their ledger tags or report sections.
