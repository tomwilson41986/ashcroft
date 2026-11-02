# Greyhounds: QA, review and the roadmap to profit

racing² - 2 November 2026. The owner's ask: a full QA and review; how to work more on feature engineering; other
algorithmic and quantitative approaches; which models to try; ensembles and stacking; a roadmap to bring us to
profit. Every figure names its source; hyphens throughout. Code QA: an independent read-only review of
`greyhound/tradeout.py`, `research/greyhound/*`, the trader's book hook and the workflows (12 tests pass), plus the
checks below.

## 1. Where we stand

| Fact | Figure | Source |
|---|---|---|
| Best greyhound model against the result (log-loss, 18,240 runners, 6 Oct-1 Nov) | log-BSP model 0.44368; full win model 0.44498; 5 Oct base 0.45140; **Betfair SP 0.42612** | report section 15, run 41460911545 |
| The full model's weight beside the SP | -0.172 (adds nothing to the SP over these weeks) | report section 14, run 41450988359 |
| Recorder book age in a race's last 3 minutes | median 77-92 s, 90th pct 162-215 s, every day | report section 13, run 41447610128 |
| Trade-out rules in the 7-day holdout, every model | none with a 90% lower bound above zero | report sections 13-15 |
| Paper trader since 6 Oct (its own rule) | 5,091 bets, -GBP751.11, -3.04% (90% -10.05 to +3.98) | greyhound-live settle run 41440971389 |
| Market depth a GB WIN market | median matched GBP380 at T-3, GBP1,993 at T-1 | reports/open_close/data.md 4.2 |

No greyhound rule has yet made money on any test. The model is still 0.018 of log-loss behind the SP.

## 2. QA findings

Severity: critical = invalidates a reported figure or stops the forward test; major = biases a figure or a choice;
minor = robustness. "Confirmed" = shown by reading the code or running it.

### Critical

| # | Finding | Where | Effect | Fix (roadmap item) |
|---|---|---|---|---|
| C1 | **The trade-out forward test can never start.** It waits for the greyhound trader's own books (`books_greyhound_trader.csv`), but since 2 Nov the trader runs in `paper-models.yml` (the owner's rule: paper models off the live job), which never sets `BETFAIR_RECORD=1` and holds only the delayed key. No trader book is written; `--freeze` reports "0 of 10 days" for ever. Had they been written they would be as stale as the recorder's. `paper-models.yml` has also never run (0 runs; it needs its own self-hosted runner). Confirmed. | `greyhound/tradeout.py:23-27`; `.github/workflows/paper-models.yml`; `greyhound/live.py` `book_recorder` | the test of report section 12 does not exist in practice | 0.1, 0.2, 0.3, 0.4 |
| C2 | **The "closed at the SP" settlement is not an order anyone can place.** The code books `stake x (back/BSP - 1)` whatever wins, a perfect green-up. A BSP lay is sized by liability before the BSP is known, so it pays 0 or `S(B - BSP)/(BSP - 1)` by the result: the same expectation only if the BSP is calibrated, and with the win/lose variance the bootstrap leaves out. At GBP2 the SP lay's liability is under the GBP10 minimum the trader's own config uses (`live_config.json` `min_bsp_liability`) for every price under 6. Confirmed. | `greyhound/tradeout.py` `positions`, `settle` | the gate's 90% bound is too narrow and not a realisable return | 0.5: settle as the liability-sized SP lay pays, by the result; a stake that meets the SP minimum |

### Major

| # | Finding | Where | Effect | Fix |
|---|---|---|---|---|
| J1 | The trader's own books would not carry the marks the test uses: it reads a market only 1.5 to 0.5 minutes before the off (`decide_window_minutes` 1.5, `reconsider` false), so `off_3`, `off_5` never exist and the two-minute `move` is always zero. The tests' synthetic books do not exercise this. | `greyhound/live.py`; `live_config.json`; `tradeout.py` `entry_books` | the microstructure model's move column would be dead | 0.2 (a recorder independent of the trader's decisions) |
| J2 | `per_day` counts only days the rule traded, so it is at least 1 by construction and the "trades at least once a day" filter in `choose_bars` does nothing; the bootstrap runs over active days only. Confirmed. | `tradeout.py` `rule_table`; `tradeout_study.py` `choose_bars` | a high bar trading on 2 of 13 days can be chosen on a two-point bootstrap | 0.5: divide by scored days, bootstrap over every scored day, minimum trades and active days |
| J3 | The holdout was not read once: it is the last 7 days of whatever range runs, recomputed on every push, and the full-score runs read it three times (three price models) across two runs; about 48 rules are reported side by side with no correction for the multiple comparisons. | `tradeout_study.py`; `greyhound-research.yml`; `greyhound-full-score.yml` | the "read once" wording in report sections 13-15 overstates it; the verdicts stand (every rule failed, and selection on the holdout can only flatter) | 0.5: fixed holdout dates written down once; one pre-registered rule; a best-of-K bound |
| J4 | The whole-race filter uses the field that ran, so a race with a dog withdrawn after the entry mark is dropped, and the reduction factor a real back would suffer is never applied. | `tradeout.py` `day_panel` | look-ahead selection on non-runners | 0.5: runners active at the mark, Betfair's `adjustment_factor` applied |
| J5 | `--date` writes a ledger day before the first test day (in-sample) and the summary reads every day file. | `tradeout.py` `main`, `summary` | a re-run of the 5 Oct test's day can contaminate the gate | 0.5: refuse days before `first_test_day` |
| J6 | A day written with part of its BSPs missing is never redone (the ledger file exists), though the docstring says it is. | `tradeout.py` `track_day`, `main` | races with a late BSP drop out for good, not at random | 0.5: BSP coverage per day; redo below a threshold |

### Minor

| # | Finding | Fix |
|---|---|---|
| m1 | The study crashes with 7 or fewer panel days (empty accuracy table). Confirmed by running. | guard the empty case |
| m2 | The shuffled-SP check is a null-model check, not a leak test (it removes any leak too). | rename; add a time-shift test per feature block |
| m3 | One blank `inplay` value makes the column float, `"0.0"` fails the filter, the day is empty. Confirmed by running. | compare numerically |
| m4 | Per-day exceptions are swallowed, so a schema bug reads as "0 of 10 days". | fail the run when most days error |
| m5 | Entry-mark windows overlap (off_1 0-2 min, off_3 2-4, off_5 3.75-6.25) and are measured from the scheduled off. | non-overlapping windows; the actual off |
| m6 | `held` charges commission per winning trade, not on the market's net. | net by market |
| m7 | `feed_lag_s_p50` can be NaN in `frozen.json`. | write null |
| m8 | Test gaps: no `--date` guard test, no mixed-sign netting test. | add |

### Checked and found correct

- **No look-ahead in the models**: the card-safe feature set holds nothing built from the race's own BSP, SP or
  result; `bsp` survives `slim` but never enters the features; every block is day-lagged; the early-stopping slices
  (win and log-BSP) are the last tenth before 6 Oct; the study fits only on earlier days, standardises on training
  rows, chooses bars on development days.
- The settlement formulas are right under their own green-up assumption; commission on each market's positive net;
  the bootstrap is by day; the fill test checks the right side; joins and dtypes match; the freeze timing and the
  workflow conditions are right.
- **The recorder keeps polling after the scheduled off** (every 60 seconds until the delayed feed shows the market
  closed or suspended, or 15 minutes: `betfair_recorder.py` `record`, `_interval`), so its day files probably already
  hold the real last minute or two before the off, stamped late. The study dropped them as "after the off"
  (`tradeout.py` `pre_off_books`: polled before the scheduled off). Roadmap 0.1 recovers them.

## 3. Feature engineering: what to build next

Grouped by what each needs. "Have" = the data is in S3 today; "Need" = something must be recorded or bought first.
Every block is scored the same way: walk-forward, card-safe, its gain over the market measured as Benter's
cross-entropy gain against the SP book (reports/greyhound_literature.md 3.3), not its own log-loss.

### 3.1 Form and race-shape (have: GBGB 2018-2026, 2.6m runs)

| # | Feature block | Why it should help | Source |
|---|---|---|---|
| F1 | **First-bend model**: each dog's predicted position at bend 1 from early sectionals, break history and trap, then the field simulated to bend 1 (who leads, who is crowded, who is checked) | Bend-1 position correlates 0.334 with the finish and the trap barely predicts it; this is where greyhound races are won or lost | Butler, Butler & Gaine 2025; Lyons 2016 (literature 3.1) |
| F2 | **Race simulator (the horse ABM transferred)**: Monte-Carlo of early pace, crossings and crowding to the first bend, then run-in speed; features are the simulated win, lead and crowding probabilities | The horse model already has `model/abm/`; greyhound runs carry a measured sectional for every runner, so calibration is easier than for horses | `model/abm/`, report section 2 |
| F3 | **Track-and-distance-normalised times** against the winners' median, rolling 28/91/365-day windows, a meeting variant from a mixed model (track x day random effect) | The engine's GSR uses a standard time; a mixed-model variant separates the going from the class of the meeting | literature test 8 |
| F4 | **Trap x running-style interactions per track** (railer drawn wide, wide runner drawn inside, neighbours' early speed), shrunk towards the track mean | Trap 4 and 3 effects survive the SP in GB data; crowding of middle traps | Butler et al. 2025; Gaine 2022 |
| F5 | **Trainer intent**: grade drop or rise, distance change, trial before a return, first run for a new kennel, the trainer's strike rate in each case against the SP | The horse model's `intent_features.py` family; the connections' choices carry information the form does not | `model/intent_features.py` |
| F6 | **Ability with uncertainty**: the Kalman rating's variance and a Bayesian hierarchical ability (dog x track) so a dog with three runs is not treated like one with forty | `lb_cv_gsr_ewm_z` and `lb_kr_*` are already the top features (section 14); their uncertainty is not used | report sections 11, 14 |

### 3.2 Late information (have for 12 tracks since 7 Oct; need for the rest)

| # | Feature block | Why | Source |
|---|---|---|---|
| L1 | **The weigh-in weight on the day** from greyhounds.today's cards (12 GB tracks, recorded every 30 minutes since 7 Oct, `greyhound/today_cards.py`): weight change, weight against the dog's mean, against its winning weight | The best research model used these (2026 Betfair bundle log-loss 0.44150, report section 11); they were left out of the live model only because the morning card lacks them, and a T-1 decision does not need the morning card | report sections 8, 11 |
| L2 | **The card's forecast price** (Betfair's `FORECASTPRICE`, already in the catalogue the recorder keeps, `betfair_recorder.py:202`) | A tipster's tissue made before the market forms; a prior where the exchange book is hollow (T-15 and earlier) | literature tests 13 |
| L3 | **Going, prize money, handicap marks** from the same cards | Card-unsafe today (`CARD_UNSAFE`); all published before the off at those tracks | `greyhound/live.py` |

### 3.3 Market microstructure (need: books that are current, dense and carry volume)

| # | Feature | Why | Source |
|---|---|---|---|
| M1 | Order-flow imbalance over 10, 30, 60 seconds, scaled by depth | The best-documented short-horizon predictor; strongest where the tick is large against the spread, as on Betfair | Cont et al. 2014; Gould & Bonart 2015 |
| M2 | Weight of money at three levels; the micro-price | Practitioner evidence on Betfair, rho about 0.26 with the next 30 seconds | Betfair data scientists' tutorial; Stoikov 2018 |
| M3 | Price path: the move from T-5 to T-3 to T-1, its acceleration, the runner's share of new matched money | Late money is informed; odds that fall late beat others at the same final odds | Gramm & McKinney 2009; Hanyu et al. |
| M4 | Traded volume by runner (`runner_total_matched`) | Empty in the recorder's greyhound books today (section 13): the horses' closing model runs without its volume terms | report section 13 |
| M5 | **Cross-market signals**: the WIN book against the PLACE book (recorded since 6 Oct) and the FORECAST market (in the Historic Data bundle) through Harville/Henery place probabilities | A runner priced differently in two markets of the same race is a dislocation that does not need a form model | literature 3.2 |
| M6 | Shin-adjusted market probabilities in place of 1/price | Corrects the favourite-longshot bias in the input; beat raw prices on 1,000 greyhound races | Cain, Law & Peel 2002 |

## 4. Other algorithmic and quantitative approaches

1. **Benter's second stage as the core** (already the horses' closing model): extend it from one market input to
   several (the book at T-5, T-3, T-1, the forecast price, our models), each a log-probability term in a conditional
   logit fitted to the SP book; read every new input by its gain over the market.
2. **Ranking objectives**: the exploded (Plackett-Luce) logit using second and third places (`pl2` exists in
   `greyhound/model.py`, never scored on the trade-out), LambdaRank. More information per race than the winner alone.
3. **Probability calibration**: temperature or isotonic calibration of the softmax per race, and a per-price-band
   calibration of the market; the model overrated the dogs it picked (report 8.2, the base model); not yet checked for the full model.
4. **Bayesian hierarchical ability models** (dog x track x time, partial pooling): honest uncertainty for dogs with
   few runs, and a posterior that can feed fractional Kelly directly (Baker & McHale 2013).
5. **Microstructure models**: a logistic queue-imbalance model and LightGBM on order-flow features for the price in
   the next 30-60 seconds; a Hawkes process for order arrival only if the dense books show bursty flow.
6. **Execution research**: passive entries (post at or inside the best price from T-5, cancel at T-1) against
   taking, scored by CLV conditional on a fill, with a queue-position fill model; makers earn more than takers on
   Betfair (Whelan 2025) but resting orders are adversely selected near the off (Glosten & Milgrom 1985).
7. **Market-only structure**: place-market and forecast-market mispricing against the win market (M5), which needs no
   form model at all.
8. **Regime and segment analysis**: BAGS afternoon cards against evening open races, graded against open, by track;
   the favourite-longshot pattern differs with how informed the market is (Ottaviani & Sorensen 2010).

## 5. Models to try, in order

| # | Model | Target | What it tests | Cost |
|---|---|---|---|---|
| 1 | LightGBM, race softmax, card-safe full features (done: 0.44498) | win | baseline | done |
| 2 | LightGBM L2 on log BSP, 20,000 rounds (done: 0.44368) | log BSP | the horse target | done |
| 3 | The same two with the weigh-in features (L1-L3), 12 tracks | win, log BSP | late information | one fit |
| 4 | Plackett-Luce (`pl2`) objective | finishing order | more signal per race | one fit |
| 5 | CatBoost and XGBoost on the same features | win, log BSP | model diversity for the ensemble | two fits |
| 6 | A set model over the runners of a race (DeepSets or a small transformer, permutation-invariant) | win | interactions between runners the trees miss; 2.6m runs is enough | a GPU-free small net, a day |
| 7 | Bayesian hierarchical ability (PyMC or Stan, subsample) | win | uncertainty for staking | days |
| 8 | Closing-price models: conditional logit (done), microstructure (done), LightGBM (done), plus a per-mark ensemble | SP share | the trade itself | cheap, on the panel |

## 6. Ensembles and stacking

The design that keeps it honest: **nested walk-forward**. Level-0 models produce out-of-sample predictions for every
race by month (each month predicted by models fitted on earlier months only). The level-1 model is fitted on those
predictions for earlier months only and scored on the next month. Nothing at level 1 ever sees a level-0 prediction
made with knowledge of its own race.

- **Level 0** (diverse on purpose, Lessmann, Sung, Johnson & Ma 2012): win softmax, log-BSP regression, `pl2`,
  CatBoost, the set model, the Elo and Kalman ratings alone, the card's forecast price, the market at T-5/T-3/T-1.
- **Level 1**: a conditional logit on the level-0 log-probabilities (Benter's form), target the SP book for the
  trade-out and the result for hold-to-result betting; a LightGBM stacker as the alternative, compared on the
  holdout once.
- **Seed and bagging ensembles** inside each level-0 learner (five seeds) to cut variance before any stacking.
- **Expected gain**: the two models we have differ by 0.0013 of log-loss and both are 0.018 behind the SP
  (section 15); stacking diverse learners typically moves log-loss by thousandths. It is worth doing, but it will not
  close an 0.018 gap by itself. The late information (L1-L3) and the cross-market signals (M5) are the larger levers.

## 7. The roadmap

Each phase ends at a gate fixed before it starts. A phase that fails its gate stops the line it was testing; money is
committed only at Phase 4, and only with the owner's decision.

### Phase 0 - Fix the data (this week)

| Item | Why | Done when |
|---|---|---|
| 0.1 **Re-time the recorder's books** by each snapshot's last-match time, and keep the snapshots polled after the scheduled off (the recorder polls until the delayed feed shows the market closed: `betfair_recorder.py` record loop) | The study dropped the real last 1-2 minutes as "after the off" (section 2, "checked and found correct"); this recovers a true-time panel from the days already held (from 6 Oct), inside Betfair's rules | the re-timed panel's feed lag is about 0 and the study re-runs on it |
| 0.2 **A greyhound book recorder of its own** in `paper-models.yml` (delayed key, every 5 seconds in the last 5 minutes, on past the scheduled off until the market closes, `EX_TRADED` for runner volume), in `greyhound/`, not `betfair_recorder.py` (the owner's rule: nothing in the live horse system changes for another sport) | Dense true-time books with volume (M1-M4) | a day's file with 5-second books to the true off and non-empty runner volume |
| 0.3 The paper-models runner registered | `paper-models.yml` has never run (0 runs); until it does no greyhound paper trading or new recording happens | the first green run |
| 0.4 Point the trade-out forward test at the re-timed books and the full and log-BSP models | Today it waits for trader books that are never written (C1) and prices dogs with the 5 Oct base model | the freeze reports days held |
| 0.5 Fix the QA findings marked critical and major (section 2) | | tests for each |

### Phase 1 - Research on true-time books (weeks 1-3)

- The study on the re-timed panel (0.1), then on the dense books (0.2) as they accumulate.
- Feature blocks F1, L1-L3, M1-M6; models 3-6; the stacked level-1 model.
- Strategy families scored side by side: trade-out at each mark (back and lay), hold to result at BSP, cross-market
  (M5), passive entry (needs 0.2).
- **Gate 1** (fixed now): a rule passes to Phase 2 only if, on a holdout read once, its 90% lower bound of return
  after 2% commission (5% as the sensitivity) is above zero with at least 3 trades a day, and the shuffled-target leak
  test earns nothing. At most three rules go forward, pre-registered with their bars.
- **Kill**: no rule passes Gate 1 by the end of week 3 on true-time books -> the trade-out is closed for greyhounds.
  The hold-to-result and cross-market lines get one more research cycle; if they fail too, greyhounds stop.

### Phase 2 - Paper forward test (30 days, from the freeze)

- The pre-registered rules frozen with their models; paper positions on the dense true-time books; the ledger and the
  daily summary in `greyhound-track.yml`.
- **Gate 2**: after 30 days and 1,000 primary trades, the 90% lower bound of the return after commission above zero
  and the mean CLV against the SP above zero; the fills checked against the size offered.

### Phase 3 - Micro-stake live pilot (owner's decision; 2-4 weeks)

- Real orders, smallest stakes (Betfair's minimum), a daily loss limit, on the live key in a process the owner
  approves. This is the only way to see real-time prices and real fills: Betfair permits no read-only use of live
  data, so paper can never measure execution.
- **Gate 3**: realised CLV within the paper test's 90% range, and slippage against the book at decision below half
  the paper edge.

### Phase 4 - Scale (owner's decision)

- Stakes sized by fractional Kelly on the expected CLV, shrunk for estimation error (Baker & McHale 2013; Benter used
  a half to a third), capped by the market's depth: a greyhound WIN market holds a median GBP380 matched at T-3 and
  GBP1,993 at T-1 (reports/open_close/data.md 4.2), so the capacity of any greyhound rule is small. As arithmetic, not
  a forecast: a rule taking GBP20 a trade, 20 trades a day, at a 3% net edge would make GBP12 a day; the edge and the
  size, not the model, set the ceiling.
- Re-gate monthly; stop on a 90% lower bound below zero over the trailing 30 days.

### What this means for profit

The honest position: on every test so far no greyhound rule has made money (sections 4, 13, 14, 15), and the
greyhound market is thin. The path above is the cheapest way to find out whether there is an edge at all, with a
stop at each gate. The two levers most likely to matter are the late information (the weigh-in, L1) and true-time
books (0.1, 0.2); stacking and new learners are worth thousandths of log-loss, not the 0.018 to the SP.

## 8. Decisions for the owner

1. **Register the paper-models runner** (`paper-models.yml` has never run): until then no greyhound paper trading
   or new greyhound recording happens.
2. **Approve roadmap 0.2**, a greyhound book recorder of its own in `paper-models.yml` (delayed key, read-only,
   nothing in the live horse system touched).
3. **Phase 3, a micro-stake live pilot**, only if a rule passes Gates 1 and 2: the only legitimate way to see
   real-time prices and fills, because Betfair permits no read-only use of live data. Money: the owner's call.
