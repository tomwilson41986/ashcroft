# racing² - edge investigation: QA, cleaning and where the profit can grow

11 Nov 2026. A read-only investigation of the whole system: the live trader, the morning predictions, the features,
the closing model, the data and the research record. Every finding names its file and line or its ledger tag, and
each was then checked against the code and the ledger by a checker told to refute it. Nothing here changed the live
rule. Ids in brackets (for example `trading_code-00`) point to Appendix A and to the evidence file
`reports/edge_investigation_1111_findings.jsonl`.

## 1. In short

| Question | Answer |
|---|---|
| Is the record sound? | Yes. No check refuted the trading record. Two bookkeeping gaps: the 32-day record (+GBP6,314.40, TRADING.md:1771) is GBP67.97 below the sum of its own daily lines (+GBP6,382.37), and 22 Oct carries an unexplained GBP73.04 between Betfair's funds and the trader's settlement (`pnl-03`, `ledger_meta-11`). |
| Errors? | 26 defects were reported and all 26 were upheld: 15 as written, 11 with corrections. Section 3. |
| The four that matter most | (1) Two lost-reply faults in the live trader that can leave a lay with no back, or a back with no lay: fix written and tested, waiting in `reports/pending/`. (2) The rule skips a whole race when one active runner has no model price, and logs nothing. (3) The 04:00 forecast is built on results to two days back; the model was trained on results to the day before. (4) The repository is public: the rule, the weights and the daily balances can be read by anyone. |
| Where is the profit? | In the first look. 66% of the result (GBP+3,292 of GBP+4,986) came from the 06:00 UK hour at +10.5% CLV after cuts, 21 of 22 days up; entries from 10:00 made about nil (10 Oct - 4 Nov, ledger `live-by-hour-1104`). |
| Where does it leak? | Top-ups carry about three quarters of the money at a fraction of the first backs' CLV. The expected CLV does not tell good days from bad. Section 5. |
| How to make it larger? | Capacity binds, not the model: the rule finds about GBP2,986 a day at the bar. Resting the unfilled part of each order, a deeper ladder, the skipped races and the stale history are the first four levers. Section 6. |
| New strategies? | Seven with evidence in our own record; the place pocket by a limit-on-close order and the lay book are the nearest. Section 7. |

## 2. How it was done, and its limits

- 11 lines of enquiry were completed, each by a separate investigator reading the code, the ledger (676 lines) and the
  reports: profit and loss, the live trader's code, the morning predictions' code, the features (engine and blocks),
  the closing model and rule, execution and costs, data quality, the research ledger, horse strategies, other markets.
- 133 findings. Each was checked by a checker who opened the cited lines and the ledger: 31 by two independent
  checkers, 102 by one. Result: 52 confirmed as written, 81 confirmed with corrections, none refuted. Where a checker
  corrected a finding, the corrected claim is the one used here.
- A "confirmed" code fault is a reading of the code, several reproduced on small inputs. It is not a live incident.
  Where how often, or how many pounds, is not known the report says "not measured" and names the query.
- Limits. Three planned lines of enquiry were lost when the run reached the account's usage limit and were not run
  again: modelling upgrades (covered in part by `reports/model_review_1111.md`), operations and risk (covered in
  part by the trader and predictions findings), and outside sources. The planned second sweep for gaps was not run.
  No database was reachable from the investigation: every measurement asked for below is a query still to run.

## 3. Errors

### 3.1 The live trader (`trading/`, `auto_trade.py`)

| # | Fault | Where | Effect | Check | Next |
|---|---|---|---|---|---|
| T1 | A top-up whose reply from Betfair is lost, and which never matched, is booked as a second copy of the horse's earlier fill; the session then lays winnings that do not exist | `trading/exchange.py:408-424`, `trading/session.py:294-312` | A lay at the SP with no back behind it; liability about GBP400 for a copied first fill, more at long prices. No incident in the record | Confirmed by two checkers, reproduced (`trading_code-00`) | Fix and tests in `reports/pending/` |
| T2 | A lost reply on a top-up's lay is called placed whenever the horse's earlier lays cover the new liability | `trading/exchange.py:433-442`, `session.py:383-387` | The top-up (stake up to GBP200) stays open to the result while the ledger says it is closed; never retried | Confirmed by two checkers, reproduced (`trading_code-01`) | Same patch |
| T3 | `close_out` lays every open position in a race, including a horse whose lay is past its five tries or whose state is unknown ("never lay it twice") | `session.py:370-375` | A second lay on the same horse, if another horse in the race is being laid | Confirmed (`trading_code-03`) | Same patch |
| T4 | A refused lay is tried five times in about four minutes, then left for the day, though the session runs to 21:35 to re-send such lays | `session.py:39, 169-172, 392-398`; `live-trade.yml:183-185` | An outright back for the day. Documented design (TRADING.md:296-297) | Confirmed as a design limit (`trading_code-03`) | Owner's call: retry with a back-off until the off |
| T5 | The rule returns nothing for a race if any ACTIVE runner has no model price or no quote, and writes no ledger row | `trading/strategy.py:157-161`, `session.py:221-233` | Races untraded in 05:15-10:00. How often: not measured. Seen so far: 7 runners with a BSP missing from the 06:00 file in 589 races, 24 Oct - 8 Nov | Confirmed (`strategy_horse-00`, `trading_code-09`, `serving_code-08`) | Read-only count on the trader's books 2 Oct - 10 Nov, by hour and country |
| T6 | After a withdrawal the "never under the model's price" guard and the ladder still use the forecast for the full field | `strategy.py:181, 194`; `session.py:326-329` | The guard is stricter than intended by 1/(1 - q) for a removed runner of model chance q | Confirmed with corrections (`trading_code-07`) | Re-normalise the guard's price; replay first |
| T7 | The lay at the SP is sized on the matched price and is not corrected when a reduction factor later cuts the back | `session.py:311-312, 366-368` | The horse is over-laid after a withdrawal: no change in expected profit, more swing | Confirmed (`trading_code-08`) | Measure on live fills with a reduced price |
| T8 | Top-ups read the trader's own footprint: after our take thins the book, the next poll reads a lower market price and can clear the bar again | `session.py:315-332`, `strategy.py:171`, `model/race_book.py:423-430` | Size not measured | Confirmed with corrections (`trading_code-10`) | Split top-up CLV by "price moved by our take" |
| T9 | Void stakes count against the GBP4,000 day limit | `session.py:661`, `trading/risk.py:19, 44` | Small; documented behaviour | Confirmed with corrections (`pnl-08`) | None unless the limit binds again |
| T10 | The 10:00 cut-off includes the whole 10:00 minute | `session.py:159` | Top-ups at 10:00:02, 10:00:20, 10:00:27 on 5, 8 and 10 Nov (GBP2.44, GBP5.42, GBP3.49) | Confirmed (`trading_code-12`) | One-character fix with a test, on the owner's word |

### 3.2 The morning predictions (the 04:00 run and the card)

| # | Fault | Where | Effect | Check | Next |
|---|---|---|---|---|---|
| S1 | The 04:00 run prices on results to two days back. The nightly job asks for the day itself at 21:30 UTC, before it is published, so day D arrives at about 21:30 UTC on D+1 | `scraper.py:43-52, 572-582`; `daily-results.yml`; `predict.yml` | A train-serve gap every day. Cost on a normal day: not measured. In the 15-20 Oct pause 37 of 2,304 runners had an unseen run, error 0.52 against 0.39 (ledger `qa-1021`) | Confirmed (`serving_code-00`, `data_quality-01`) | Share of card runners that ran the day before; then one results request before 04:00 UTC |
| S2 | Headgear text is not normalised: the card can give `CkPcTT`, the table and the model hold `CkPc TT` | `daily_predictions.py:306, 349-350, 592-598` | Compound headgear reads as an unseen category and a false change of headgear. 7% of runners differed, 24 Oct - 8 Nov | Confirmed with corrections (`serving_code-02`, `feature_code_a-00`) | Tabulate the raw strings on the stored cards; normalise at the card |
| S3 | The saddlecloth number is still written into `stall` at the scrape; the 12 Nov fix patches it downstream only where the race type can be told | `daily_predictions.py:327-328, 355-356`; `model/card_enrich.py:297-305` | Residual wrong stalls where the race type is not read | Confirmed with corrections (`serving_code-03`) | Fill `card_no` at the scrape; `stall` from the Stall column only |
| S4 | Race class: Group and Listed are not parsed, Irish races stay blank, compound goings are cut short | `daily_predictions.py:273-283` | Class blank for 27% of runners (Irish 1,331 of 1,361; Class 1 117 of 168) | Confirmed with corrections (`serving_code-05`) | Cross-tab card against table; derive the class |
| S5 | The card header's going and class patterns are tried on every span, so a race name holding a going word ("Hardy", "Goodwood") or "Grade n" is dropped or misread | `daily_predictions.py:256-297` | Size not measured | Confirmed with corrections (`serving_code-01`) | Word boundaries and one pass per field |
| S6 | Where the card prints no claim, the jockey's last recorded claim is filled in | `model/card_enrich.py:390-399` | A rider who has ridden out the claim is served the old allowance | Confirmed (`serving_code-09`) | Count on the stored cards |
| S7 | No input contract: no check of the last results date, no check that the feature code matches the model, unseen categories become -1 silently, the sectionals refresh may fail quietly | `health_check.py:95-124` (never called) | A stale or malformed input prices silently | Confirmed with corrections (`serving_code-04`, `feature_code_b-04`) | Log-only checks first, then fail the run |

### 3.3 The features

None of these leaks a result. A repair changes what the served model reads, and the model was fitted on the faulty
values, so every repair goes with a refit and a screen, not into the served path alone.

| # | Fault | Where | Effect | Check |
|---|---|---|---|---|
| F1 | Under pandas 3 the day number is days/1000, so DSLR1-4 and seven derived columns are 0 unless two runs straddle 4 Oct 2024 or 9 Jan 2022 | `model/custom_metrics.py:837-858` | Twelve served columns, 0.117% of gain. pandas 3 recorded on 7 Oct; versions are not pinned | Confirmed, reproduced on pandas 3.1.1 (`feature_code_a-01`) |
| F2 | `form_slope`: a debutant and a one-run horse read exactly 0, short histories are shrunk four times, the sign is inverted | `custom_metrics.py:1362-1389` | A debutant, a once-raced horse and a steady horse look the same | Confirmed (`feature_code_a-02`) |
| F3 | Career-best figures vanish when the last run has no value | `custom_metrics.py:89-93, 1738-1739, 1987-1995`; `model/financial_features.py:441-445` | `best_RSR`, `RSR_gap`, `career_best_or`, `or_vs_best` missing for such horses | Confirmed (`feature_code_a-03`) |
| F4 | Kalman rating: an unrated horse is based on a race median that counts unrated runners as 0; first ratings fall back to 70 in every code | `model/perf_figures.py:157-162`; `model/blocks/kalman.py:42, 97-101` | Wrong starting level for unrated horses | Confirmed (`feature_code_b-02`) |
| F5 | Flat and jumps ratings are mixed in one series per horse in four blocks | `kalman.py:93`; `handicap_angles.py:54, 83-101`; `time_figure.py:113-125` | A horse that switches code carries a Flat-scale state against a jumps mark | Confirmed (`feature_code_b-03`) |
| F6 | `time_figure`: the going allowance is the meeting's own mean, so each meeting's figures centre on zero and the level between meetings is lost; trips pool by round-half-to-even; surface is not split | `model/blocks/time_figure.py:81-83` | The figure cannot tell a strong card from a weak one | Confirmed (`feature_code_b-01`, `-08`) |
| F7 | `exposure`: the trainer's and sire's gain pools all codes and ages, then shrinks to a code-specific prior | `model/blocks/exposure.py` | A yard's Flat two-year-olds set its expected gain for a chaser | Confirmed (`feature_code_b-05`) |
| F8 | Past-run market features: a race with no BSP scores every runner as beaten by the market | engine | 0.103% of gain | Confirmed with corrections (`feature_code_a-05`) |
| F9 | `head_to_head` treats blank horse names as one horse | `model/blocks/head_to_head.py:77` | False meetings; rows affected not measured | Confirmed (`feature_code_b-10`) |
| F10 | No guard that recent results carry a BSP: a day without prices vanishes from all 14 served blocks | `blocks.attach_as_trained` | Last-run windows, form lines and ratings skip that day silently | Confirmed (`feature_code_b-07`) |
| F11 | `jumps_form` (not served) reads falls from the comment alone | `model/blocks/jumps_form.py:52-53` | Iteration 117's block arm was read with it; noted in the ledger | Confirmed with corrections (`feature_code_b-00`) |
| F12 | No golden-frame test on the engine: a library or card-format change alters served features unseen | tests | F1 and S2 were both invisible to the suite | Confirmed (`feature_code_a-09`) |

Seven of the 15 faults in the 10 Nov audit were re-verified at their lines, with three added details
(`feature_code_a-07`).

### 3.4 The closing model and the scoring code

| # | Fault | Where | Effect | Check |
|---|---|---|---|---|
| C1 | `market_now` steps from the mid to the bare back price when lay/back passes 1.25: a runner in a wide book reads about 6 points more expected CLV than the same runner in a tight one, against a 3% bar | `model/race_book.py:399, 423-430` | Documented design, never fitted. Whether it helps or hurts: not measured | Confirmed with corrections (`closing_model-00`) |
| C2 | The expected CLV is the mean of draws renormalised to the book, which moves it off the fitted closing chance by a term that depends only on one constant and the runner's share | `race_book.py:164-173, 445-448` | A price-dependent tilt with no information in it | Confirmed (`closing_model-01`) |
| C3 | The served file has one sigma for every runner and was fitted on January-March 2026 only (2,504 races) | `data/models/closing_model_novol.json` | Price risk is one number whatever the price, field or hours to the off | Confirmed (`closing_model-11`) |
| C4 | Three different non-runner reduction rules in the research scoring, none checked against Betfair's | `research/arms/core.py:644-680`; `research/arms/quotes.py:257-264`; `research/paper/morning_edges.py:190-199` | Replays slightly mis-state backs, lays and quotes. Live settles at Betfair's own price | Confirmed with corrections (`strategy_horse-06`, `execution-07`) |
| C5 | The desk's trial count for greyhounds and football is 317 ledger lines, not the roughly 31,000 rules tried | `desk/trials.py:87-90` | A latent risk for the desk's deflated-Sharpe gate, not a past wrong pass | Confirmed (`strategy_other-01`) |
| C6 | The with-volume closing model was a look-ahead (+1.6 points in backtest on whole-morning volume, -3.9% CLV live); TRADING.md still shows it as the better column | TRADING.md; ledger lines 240, 259, 260 | A misleading table | Confirmed with corrections (`ledger_meta-06`) |

### 3.5 The data

| # | Fault | Where | Check |
|---|---|---|---|
| D1 | A day saved early or partly published is logged "ok" and never asked for again | `scraper.py:546` | Confirmed (`data_quality-00`, `serving_code-07`) |
| D2 | The scrape does not check the file's own date, and the upsert can neither blank nor de-duplicate an amended result | `scraper.py:307, 386, 408-412, 467-470` | Confirmed (`data_quality-04`); fix before any re-ask of D1 |
| D3 | No nightly completeness check of results against Betfair's race list (run once, by hand, on 11 Nov) | workflows | Confirmed with corrections (`data_quality-03`) |
| D4 | Price-file archive: no content check, a file that fails to parse is not retried after 7 days, replaced rows are not removed | `betfair_prices.py` | Confirmed with corrections (`data_quality-07`) |
| D5 | The database is uploaded whole with no integrity or row-count guard, after steps that can be killed mid-write | workflows | Confirmed with corrections (`data_quality-06`) |
| D6 | The live record is loaded for the last 3 days only; malformed lines are dropped without a count | `betfair_recorder.py` | Confirmed with corrections (`data_quality-10`) |
| D7 | Three BSP sources are not reconciled on a schedule; the nightly "Sync BSPs" step runs where Betfair refuses it | workflows | Confirmed with corrections (`data_quality-05`) |
| D8 | Historic CLV research ignores reduction factors (about 0.7 points of CLV, `reports/preregistration_clv_oos.md:60`); the marks table drops the two fields | research | Confirmed with corrections (`data_quality-02`) |
| D9 | Research pays dead-heaters as full winners; 154 void rows are read as non-finishes | `scripts/clv_betfair.py:67-68` | Confirmed with corrections (`data_quality-09`) |
| D10 | Irish prize money may be euros read as pounds: unverified either way | scraper, model | Confirmed with corrections (`data_quality-08`) |

## 4. Risk controls that are missing

| Control | Evidence | Id |
|---|---|---|
| The repository is public | Checked on 11 Nov: visibility public. TRADING.md, the ledger, `trading/config_live.json` and the closing model are tracked. Already an open item on the desk's lock-down list (`docs/agent_desk.md:132`) | `ledger_meta-07` |
| The server's upgrade timer still cuts the best hour | The service was restarted on three mornings in 19 days in the 06:00 UK hour; each lost first hour costs about GBP130. The approved fix has waited since 5 Nov for three root commands | `pnl-11`, `ledger_meta-08` |
| No reconciliation with Betfair during the session | Positions are read from Betfair's orders once, at the start | `trading_code-02` |
| The kill switch fails open | Any error reading the S3 object counts as "not stopped"; it is the only automatic stop | `trading_code-05` |
| No loss limit can bind while bets go on | The stop-loss reads settled P&L, and nothing settles before 10:00 | `trading_code-06` |
| Stopping by cancelling the run can leave a back without its lay | The documented way to stop | `trading_code-04` |
| No sanity gate on the predictions file | Only the column's presence is checked | `trading_code-11` |
| No monitor of edge decay | Entry margin fell to 3.3% and CLV to about zero on 9-10 Nov with no automatic response | `closing_model-10` |
| The Expert Fee is in no gate or report | On the repository's reading it starts at about week 17; the terms are unverified | `execution-05`, `pnl-13` |
| The morning models are live, unhedged and unread | Their stored CLV has never been read; the research edge sits at the morning price, not at 10:00 | `strategy_horse-02`, `execution-11` |

## 5. Where the profit comes from, and where it leaks

| Reading | Figures | Source |
|---|---|---|
| The first look pays | 06:00 UK hour: GBP+3,292 of GBP+4,986 (66%) at +10.5% CLV after cuts, 21 of 22 days up. 07:00-09:00: +1.1% to +3.4%, about GBP1,539. From 10:00: GBP14,518 staked for GBP+155 | 10 Oct - 4 Nov, ledger `live-by-hour-1104` (`pnl-01`) |
| The record is concentrated | 10-13 Oct GBP+1,665.32 (4 days); 14-21 Oct GBP-0.48 (8 days); 22 Oct - 10 Nov GBP+4,717.53 (20 days, mean GBP235.88, sd GBP152.36 a day) | TRADING.md daily lines (`pnl-12`) |
| Top-ups carry the money, not the edge | 22 Oct - 10 Nov: first backs about GBP740-1,250 a day at +2.6% to +17.6% CLV; top-ups about GBP2,200-3,070 a day, usually 0% to +8% | ledger live-day entries (`closing_model-04`) |
| Horses that reach six fills close lower | Below the rest on 15 of 20 days, but still positive on 15 of 20; confounded by stake size | `pnl-05` |
| Expected CLV does not rank days | The first back's margin over the BSP swings from +2.4% to +17.4% by day; the expected CLV stays at 5.8-7.8% | `pnl-06` |
| No calibration since 20 Oct | The last table says a 3-6% expectation closed level on about 70% of the money (10-20 Oct, before the current rule) | `closing_model-02`, `pnl-00` |
| The replay's scale is stale | Live earns 0.80 of the replay after cuts, measured 10-19 Oct on the old rule; the nightly replay still starts at 06:15 while live starts at 05:15 | `pnl-04`, `execution-08` |
| Capacity binds | The rule stakes about GBP2,986 a day because what is on offer at the bar binds; live horses got 68% of their GBP400 target | `ledger_meta-05`, `execution-01` |
| Costs | Commission GBP327.30 on GBP6,663.33 gross (4.9%), 30 Sep - 8 Nov | `pnl-13` |

## 6. Making the edge larger

Ranked by expected value for the effort. Each first step is read-only; none changes the live rule.

| # | Lever | Why | First step | Effort |
|---|---|---|---|---|
| 1 | Stop losing whole races to one unpriced runner (T5) | The mechanism is confirmed in live and in the replays, so no replay can see its cost | Count market-minutes 05:15-10:00 with an ACTIVE runner and no model price, 2 Oct - 10 Nov | Hours |
| 2 | Close the two-day history gap (S1) | Every card is priced without yesterday's runs; training assumed them | Measure the share of runners that ran the day before, then re-price 20 cards with history cut at D-2 against D-1 | A day |
| 3 | Rest the unfilled part of each approved order | The price the trader takes returns to the top of the book before the off 83-91% of the time; the trader only takes, on a 60-second poll | Register the one-sided resting-back test now; replay real orders on the stream from 12 Nov (`closing_model-08`, `execution-01`, `strategy_horse-05`, `ledger_meta-13`) | Days |
| 4 | A deeper ladder | The trader asks for 3 price levels. Levels 2 and 3 closed better than the blended rule and added 12% CLV x stake on every day replayed | Sum the money at levels 4-10 that clears the bar and the guard, on the stream (`execution-03`) | A day |
| 5 | Top-ups by time and edge, not by count | The measured loss sat in top-ups more than 30 minutes after the first back and at expected CLV under 6%; the gain within 5 minutes | Re-run the top-up breakdown on live fills 22 Oct - 10 Nov, after cuts (`execution-04`, `closing_model-04`) | Hours |
| 6 | Calibrate the expected CLV for the current rule | No table since 20 Oct; the bar is gross of commission; a runner can clear it on the field's overlay alone | Re-run `fill_calibration.py` for 21 Oct - 4 Nov and 5-10 Nov, with a net-of-commission column and the own-overlay share (`closing_model-02`, `-06`, `-09`) | Hours |
| 7 | Bookmakers' early prices as a third input to the closing model | In a wide book the model's "market now" is the very offer it is about to take. 25 bookmakers' prices are captured, but at 07:40 UTC | Control test on the 07:40 snapshot; then a snapshot before 05:15 UK (`strategy_horse-03`) | A day |
| 8 | A forecast-price layer on the 06:00 price | Cut error against the BSP from 0.396 to 0.377 on 7 days (14-20 Oct, 2,624 runners) and was never followed up | Re-run on 1 Oct - 10 Nov with three arms (`ledger_meta-00`) | A day |
| 9 | The sectionals already in the database | 137,183 At The Races runs from 2020 have never been in a fit; the served block reads one year of site files | One registered loop iteration (`ledger_meta-04`) | A day |
| 10 | The combiner | Three-seed averaging gained -0.0026 (on the September main member, not on the blend); the learned gate priced -0.0014 better without trading better. Neither is served | Run on the saved out-of-sample files, no new fits (`ledger_meta-09`) | Hours |
| 11 | The feature repairs with the retrain | Section 3.3, with the market rating already in hand | One repair block, screened, then into the retrain | Days |
| 12 | Read the registered forward CLV test | Registered 23 Sep, opened 25 Sep, never read; its window ends by 20 Nov | Run `scripts/clv_betfair.py` with the registered defaults (`ledger_meta-01`) | Hours |
| 13 | Measure our own effect on the BSP | The hidden limit on scaling; read once, coarsely, on 20 Oct | Regress the BSP's gap to the last mid on our share of the SP lays, 8 Oct - 10 Nov (`execution-06`) | A day |

## 7. New strategies

| # | Strategy | Evidence in our record | First step |
|---|---|---|---|
| 1 | Resting orders in the morning book | Lever 3 above. The only test so far was a two-sided quote with no traded ladder; it lost GBP294.56 in 22 days, both sides pooled | Read the saved fills by side for free; register the stream test (`ledger_meta-13`) |
| 2 | The place pocket by a limit-on-close order | The strongest effect in the ledger outside the win market: 14 rules chosen on 2010-17 were all positive on 2018-26Q1, 12 with the 90% interval above zero; the pocket made +1.15% (t 3.05, 46,481 bets) on 2010-20 and +2.70% (+1.55 to +3.85, 13,829 bets) on 2023-26Q1 at 2% commission. Parked because both SPs are known only at the off | One registered run with the floor set from the win price a minute out, 2018-22 and 2023-26 apart (`strategy_horse-09`, `ledger_meta-14`, `strategy_other-02`) |
| 3 | The lay book | Retired on a figure that ignored the non-runner cut that helps a layer. After cuts: GBP+827 over 21 days, GBP+1,116 over 29 (not a registered reading) | Split the result by whether a cut applied; it is already back as a desk trial (`strategy_horse-04`, `ledger_meta-03`) |
| 4 | The overnight window | The 00:10 forecast equalled the morning's on five nights; the books are two-sided but wide, so only a resting order can trade them | Register the overnight arms before reading; the persistence table first (`pnl-02`, `strategy_horse-07`, `execution-09`) |
| 5 | The steamer signal inside the hedged rule | The steamer score uses inputs the forecast lacks; it is traded at 10:00, where the price has gone | One registered table of its selections at the trader's first read (`strategy_horse-01`) |
| 6 | The place market before the off | Traded place prices sit below the place SP (a median 0.954 of it); read on one day only, 41 days are now held | One query on the recorded place marks, 7 trials fixed first (`strategy_horse-08`) |
| 7 | A stream lab near the off | Takers lose 0.65-2.0% to the BSP on both sides near the off: that is the makers' margin | Register before the 12 Nov stream is opened (`strategy_horse-10`) |

Other markets:

| Market | Evidence | Id |
|---|---|---|
| GB greyhounds, the early lay | The only result outside horses that is positive at replayed prices: +0.71% to +2.22% of matched liability on 6 of 6 stream-replayed days. That is the replay's full-fill reading; with capped fills it is +0.79% at GBP20 a lay and negative at GBP100 and over, and the registered gate uses the optimistic reading. Settle it on real fills before the 24 Nov gate | `strategy_other-00` |
| Football, cup ties across divisions | The one football cell where the early price is measurably slow: +3.5% (95% +0.7 to +6.6) on 480 ties at last traded prices, -0.2% a tick worse each end; one window, 30 Jul - 30 Sep 2026 | `strategy_other-09` |
| Football, the line-up signal | Real (surprise above 0.2: 557 bets, +1.4% at traded prices) but spent within 15 minutes; our feed delivers the elevens about 28 minutes before kick-off against official sheets at about 60-75 | `strategy_other-05` |
| Football, the registered trials | The EV and resting trials count only fills proved by the price moving to the order, which selects against themselves; the frozen model covers divisions with 3.6% of the money | `strategy_other-07` |
| Cricket | The deepest pre-off market on Betfair; nothing is held | `strategy_other-04` |
| Asian handicap | Never scanned or recorded on the exchange | `strategy_other-06` |
| Australia | Its two greyhound leads were scored at 2% commission against about 7.5% measured | `strategy_other-10` |
| Who earns the drift | No code classifies trades by aggressor, so the size of the maker's pool is unmeasured in every market | `strategy_other-03` |

## 8. Do first, and what needs the owner

Decisions for the owner:

| # | Decision | Why |
|---|---|---|
| 1 | Merge the lost-reply fix (`reports/pending/live_trader_lost_reply_1111.patch`) | T1-T3. Five new tests fail on the code as it stands; 347 tests pass with the fix |
| 2 | Make the repository private | Section 4 |
| 3 | Run the three root commands for the upgrade timer | Waiting since 5 Nov; about GBP130 for each lost first hour |
| 4 | Pin the dependency versions for training and the live trader | F1 came from an unpinned library |
| 5 | One definition of a trial for the deflated-Sharpe count | The bar rests on four different counts; at about 320 trials it means a t of about 4.6 (`ledger_meta-02`, `strategy_horse-11`) |
| 6 | Stop the Tote and other-place recorders inside the live job, yes or no | Families the ledger has closed still run in `live-trade.yml` and raise the trial count (`ledger_meta-10`) |
| 7 | The retry rule for a refused lay (T4) | A documented rule, so the owner's to change |

Read-only work I will start, in this order: the skipped-race count (lever 1), the history-gap measure (lever 2), the
expected-CLV calibration for the current rule (lever 6), the top-up breakdown (lever 5), the registered forward CLV
read (lever 12), then the stream tests as its days accumulate (levers 3 and 4).

## Appendix A - every finding

"Value" is the checker's rating, 1 (negligible) to 5 (material). "Checks" is how many independent checkers read it.

| Id | Kind | Result of the check | Value 1-5 | Checks | Finding |
|---|---|---|---|---|---|
| **Profit and loss** (14) | | | | | |
| pnl-00 | execution | confirmed with corrections | 2 | 2 | Every filter was rejected under a binding day limit; the limit no longer binds, so the low-edge money needs re-testing under the 05:15-10:00 window |
| pnl-01 | capacity | confirmed with corrections | 2 | 2 | The edge is a first-look, first-hours, thin-book, long-price edge: 66% of the result came from the 06:00 UK hour |
| pnl-02 | strategy | confirmed with corrections | 2 | 2 | The overnight session (00:15-05:15 UK) is the one untested extension of where the edge lives, and the data now exist |
| pnl-03 | data | confirmed with corrections | 2 | 2 | The headline record (+GBP6,314.40) disagrees with Betfair's funds by GBP67.97 and 22 Oct's GBP73.04 gap was never reconciled |
| pnl-04 | process | confirmed with corrections | 3 | 2 | The replay that decides every rule has an unstable live scale: 0.80 was measured on 10-19 Oct, and on 5-9 Nov live beat the replay 2.5 to 1 |
| pnl-05 | execution | confirmed with corrections | 3 | 2 | Horses at the top-up cap take most of the money and close worse on 11 of 15 reported days; the top-up loss pattern has not been re-measured since the ladder and cap |
| pnl-06 | model | confirmed with corrections | 2 | 2 | The expected CLV does not tell good days from bad: day results track the entry margin, which swings from +3% to +17% while expected edge stays near 6% |
| pnl-07 | execution | confirmed with corrections | 2 | 2 | Taking the offer in a 25-200% spread book pays the whole spread: passive backs on rule horses were never tested and the stream record now allows it |
| pnl-08 | bug | confirmed with corrections | 1 | 2 | Void stakes still count against the GBP4,000 day limit |
| pnl-09 | data | confirmed with corrections | 3 | 2 | Race-type, code and field-size breakdown stops at 20 Oct, before the current rule and before the jumps season |
| pnl-10 | capacity | confirmed with corrections | 2 | 2 | Stake floor at long prices and a larger to-win target were retired before their forward test, on arithmetic that assumed a binding limit |
| pnl-11 | risk control | confirmed with corrections | 2 | 2 | The best hour is still exposed to the server's upgrade timer; each lost first hour costs about GBP130 |
| pnl-12 | process | confirmed with corrections | 3 | 2 | The live rule has no significance statement of its own: profit is concentrated in two stretches and eight owner changes were chosen on overlapping replays |
| pnl-13 | risk control | confirmed with corrections | 3 | 2 | Expert Fee: about 11% of profit from week 17 on terms read from a search snippet; commission already takes 4.9% of gross |
| **Live trader code** (13) | | | | | |
| trading_code-00 | bug | confirmed | 3 | 2 | A top-up whose reply is lost and which never matched is booked as a second copy of the horse's earlier fill, then hedged with a real naked lay at SP |
| trading_code-01 | bug | confirmed | 2 | 2 | A lost reply on a hedge lay is reported as placed whenever the horse already has earlier lays at SP |
| trading_code-02 | risk control | confirmed | 2 | 2 | No in-session reconciliation with Betfair: positions are read from listCurrentOrders only at start-up |
| trading_code-03 | bug | confirmed | 2 | 2 | A refused hedge is abandoned for the day after five one-minute tries, and close_out can lay an UNKNOWN-state horse a second time |
| trading_code-04 | risk control | confirmed with corrections | 2 | 2 | The documented way to stop (cancel the run) can leave a matched back without its lay, and drops every hedge still waiting |
| trading_code-05 | risk control | confirmed with corrections | 2 | 2 | The S3 kill switch fails open and is the only automatic stop |
| trading_code-06 | risk control | confirmed with corrections | 2 | 2 | No loss limit can bind during the trading window: the stop-loss reads settled P&L and the live limits are set to 1,000,000 |
| trading_code-07 | bug | confirmed with corrections | 2 | 2 | After a withdrawal the 'never under the model's price' guard and the ladder still use the full-field predicted BFSP |
| trading_code-08 | risk control | confirmed | 2 | 2 | The lay at SP is sized on the matched price and never corrected when a reduction factor cuts the back |
| trading_code-09 | process | confirmed with corrections | 3 | 2 | A race is skipped silently when any single active runner lacks a model price or any quote, with no ledger row or count |
| trading_code-10 | execution | confirmed with corrections | 2 | 2 | Top-ups read the trader's own footprint as market information |
| trading_code-11 | risk control | confirmed with corrections | 2 | 2 | No pre-trade sanity gate on the predictions file |
| trading_code-12 | clean-up | confirmed with corrections | 1 | 2 | Smaller defects in the window, the restart path and the commission line |
| **Morning predictions code** (10) | | | | | |
| serving_code-00 | bug | confirmed | 2 | 2 | The 04:00 run prices on a history that stops two days back: yesterday's results are not in the database when the card is priced |
| serving_code-01 | bug | confirmed with corrections | 2 | 2 | Card header parser applies the going, class and distance regexes to every span, so a race name containing a going word or 'Grade n' is dropped (or overwrites the going) and the race type... |
| serving_code-02 | bug | confirmed with corrections | 2 | 2 | Headgear is read as 'CkPcTT' on the card but stored as 'CkPc TT' in the results table: compound headgear is an unseen category and a false 'headgear change' at serving |
| serving_code-03 | bug | confirmed with corrections | 2 | 1 | The saddlecloth number is still written into 'stall' at the scrape; the 11 Nov fix patches it downstream and only where race_code can tell the race is over jumps |
| serving_code-04 | risk control | confirmed with corrections | 3 | 1 | No input contract on the serving path: stale database, stale sectionals, changed feature code, unseen categories and short parses all price silently |
| serving_code-05 | bug | confirmed with corrections | 2 | 1 | Race class on the card: 'Group n' and 'Listed' are not parsed, 'Grade 2/3' are read as class 2/3, Irish races stay blank; Irish compound goings are truncated |
| serving_code-06 | model | confirmed with corrections | 2 | 1 | The 48-hour declarations page carries the horse's sex letter and a first-time headgear marker, and the converter throws both away |
| serving_code-07 | data | confirmed with corrections | 2 | 2 | A day scraped while only part-published is marked done and never asked for again |
| serving_code-08 | execution | confirmed with corrections | 3 | 1 | A race with one unpriced runner is never traded: reserves and jockey-less runners dropped at 04:00 are not priced later |
| serving_code-09 | bug | confirmed | 2 | 1 | Jockey claim falls back to the rider's last recorded claim when the card prints none |
| **Features, engine** (10) | | | | | |
| feature_code_a-00 | bug | confirmed with corrections | 2 | 1 | Card headgear text does not match the table's: 'CkPcTT' on the card vs 'CkPc TT' in training - three served headgear features read wrongly live |
| feature_code_a-01 | bug | confirmed | 3 | 1 | DSLR1-4 and seven derived columns are broken under pandas 3: the day number is computed as days/1000 |
| feature_code_a-02 | bug | confirmed | 2 | 1 | form_slope: debutants and one-run horses read exactly 0 ('flat form'), short histories are shrunk 4x, and the sign is inverted |
| feature_code_a-03 | bug | confirmed | 2 | 1 | Career-best figures vanish when the last run has no value: _lag_expand_max leaves NaN where the previous run is NaN |
| feature_code_a-04 | model | confirmed with corrections | 2 | 1 | RSR standard time pools hurdles, chases, bumpers and flat races at the same course and furlong band, weighted by field size |
| feature_code_a-05 | data | confirmed with corrections | 2 | 1 | Past-run market features mishandle runs with no BSP: a whole-race gap scores every runner as beaten by the market, and the run drops out of every price window |
| feature_code_a-06 | model | confirmed with corrections | 2 | 1 | Cumulative-since-2018 counts and sums are calendar clocks: trainer/jockey career Runs, Wins, Places, CWO and ride totals grow without bound |
| feature_code_a-07 | process | confirmed | 2 | 1 | Seven of the audit's fifteen faults verified at the cited lines, with three details the audit did not record |
| feature_code_a-08 | clean-up | confirmed with corrections | 2 | 1 | More exact copies than the audit lists, and RPS is mostly field size |
| feature_code_a-09 | process | confirmed | 4 | 1 | No golden-frame test guards the engine: a library change or a card-format change alters served features silently |
| **Features, blocks** (11) | | | | | |
| feature_code_b-00 | bug | confirmed with corrections | 2 | 1 | jumps_form reads 'refused to settle', 'ran out a winner' and 'left in lead when leader fell' as non-completions - and iteration 117 is being read on it |
| feature_code_b-01 | model | confirmed | 3 | 1 | time_figure: the going allowance is the meeting's own mean, so every meeting's figures are centred on zero and class is erased |
| feature_code_b-02 | bug | confirmed | 2 | 1 | Kalman rating: unrated horses are based on a zero-polluted median_or, and first ratings fall back to 70 in every code |
| feature_code_b-03 | bug | confirmed | 3 | 1 | Flat and jumps ratings are mixed in one per-horse series in four blocks |
| feature_code_b-04 | data | confirmed with corrections | 2 | 1 | Sectionals: the live table is fetched 2.5 hours before the site's morning refresh, and a failed fetch silently serves the committed table (ends 20 Oct) |
| feature_code_b-05 | model | confirmed | 2 | 1 | exposure: the trainer's and sire's 'gain' pools raw gains across codes and ages, then shrinks to a code-specific prior |
| feature_code_b-06 | model | confirmed with corrections | 2 | 1 | form_variants averages uncapped lengths beaten, so one tailed-off run dominates fv_lb, fv_lbpf, fv_lbpp and fv_lbrel windows |
| feature_code_b-07 | risk control | confirmed | 3 | 1 | No guard that recent results carry a BSP: a day without prices silently vanishes from every drop-in block |
| feature_code_b-08 | bug | confirmed with corrections | 2 | 1 | time_figure pools trips by round-half-to-even and never separates surface in the standard |
| feature_code_b-09 | clean-up | confirmed with corrections | 1 | 1 | Dead and duplicate columns inside the served blocks: hca_penalty_edge_lbs is read on 0.4% of runners; WOA/RB copies are z-scored twice |
| feature_code_b-10 | bug | confirmed | 1 | 1 | head_to_head keeps blank horse names as one horse |
| **Closing model and rule** (12) | | | | | |
| closing_model-00 | bug | confirmed with corrections | 3 | 1 | Spread cliff in market_now: a runner crossing lay/back 1.25 gains 5-7 points of expected CLV with no model disagreement, and the threshold sits at the 06:00 median spread |
| closing_model-01 | bug | confirmed | 2 | 1 | Expected CLV is taken from renormalised lognormal draws, which shifts the mean away from the fitted q_hat: about -1.5 points on favourites and +1.4 to +2.4 on outsiders, plus a fixed per-... |
| closing_model-02 | model | confirmed with corrections | 4 | 1 | No calibration of expected CLV exists for the current-rule era; the last live table says 3-6% expected closes level on 70% of the money |
| closing_model-03 | process | confirmed with corrections | 3 | 1 | The replay has rejected every correction to the bar and the closing model, but it is the wrong instrument for removal-only changes now that the day limit does not bind |
| closing_model-04 | execution | confirmed | 3 | 1 | Top-ups carry about three quarters of the money at a fraction of the first backs' CLV; horses that reach the 5 top-up cap close lower than the rest on 15 of 20 days |
| closing_model-05 | model | confirmed with corrections | 2 | 1 | The served weights were fitted on Betfair's morning volume-weighted price with race-day-aware forecasts, then served on a 06:00 best back in a 25% spread with a 04:00 forecast |
| closing_model-06 | model | confirmed with corrections | 2 | 1 | A horse can clear the bar with no overlay of its own: the model disliking the favourite lifts every other runner's expected CLV |
| closing_model-07 | risk control | confirmed with corrections | 2 | 1 | To-win staking with a GBP200 cap and GBP15 floor still puts about 40% of the money under 5.0 where live CLV has been about zero; long prices are depth-bound, not stake-bound |
| closing_model-08 | execution | confirmed | 3 | 1 | Capacity is what binds: convert fill-or-kill takes into resting backs at the already-approved price, tested on the Exchange Stream record |
| closing_model-09 | model | confirmed with corrections | 2 | 1 | The bar is gross: commission asymmetry is not in the expected CLV, and it costs about a tenth of the profit |
| closing_model-10 | risk control | confirmed | 3 | 1 | No edge-decay control on the rule: entry margin fell to 3.3% and CLV to about zero on 9-10 Nov with no automatic response |
| closing_model-11 | clean-up | confirmed | 2 | 1 | Single constant sigma and a 2,504-race fit: the served file's risk bands are degenerate and the fit window is the smallest available |
| **Execution and costs** (12) | | | | | |
| execution-00 | execution | confirmed with corrections | 2 | 1 | Small early fills are refused because their winnings are under the GBP10 SP-lay minimum - in exactly the books that close best |
| execution-01 | strategy | confirmed | 4 | 1 | Resting (maker) backs in the wide early spreads - unblocked by the stream from 12 Nov, never tested one-sided |
| execution-02 | execution | confirmed | 3 | 1 | The order loop does a full ledger rewrite and a fresh-client S3 upload after every order, on a book read once at the start of the pass |
| execution-03 | execution | confirmed | 3 | 1 | The ladder sees only 3 price levels; deeper levels were the better ones when measured |
| execution-04 | execution | confirmed | 3 | 1 | Top-ups are capped by count (5), but the measured discriminators were time since the first back and the size of the edge |
| execution-05 | risk control | confirmed with corrections | 3 | 1 | The Expert Fee turns the cost on profit from 2% to 20% from about week 17 - it is in no gate, report or strategy comparison, and its terms are unverified |
| execution-06 | capacity | confirmed with corrections | 3 | 1 | Our own SP lays' effect on the BSP has never been measured - it is the hidden limit on scaling |
| execution-07 | data | confirmed | 2 | 1 | Two repository statements disagree on whether Betfair cuts the SP lay's liability after a non-runner; the fill model reconciles to only 96% of the ledger |
| execution-08 | execution | confirmed with corrections | 3 | 1 | Live earns 0.80 of the replay after reductions and nobody has decomposed the missing 20% |
| execution-09 | strategy | confirmed with corrections | 2 | 1 | The evening and overnight books show the highest CLV and near-infinite spreads - only a maker can trade them, and the stream does not record them |
| execution-10 | risk control | confirmed | 2 | 1 | The SP trade-out should stay: holding or half-hedging adds swing and commission for no expected profit, and take-profit exits lose |
| execution-11 | execution | confirmed with corrections | 2 | 1 | The morning models take at 10:00 and missed their own fill check; the research edge was measured at the BSP |
| **Data quality** (12) | | | | | |
| data_quality-00 | data | confirmed | 3 | 1 | A day saved early or partial is logged 'ok' and never asked for again - BSPs, evening races and amended results stay missing |
| data_quality-01 | risk control | confirmed | 4 | 1 | The 04:00 forecast and the 05:15 trader have no check that yesterday's results are in the history |
| data_quality-02 | data | confirmed with corrections | 2 | 1 | Historic CLV research and the closing-model fit ignore non-runner reduction factors; the marks table drops the fields needed to fix it |
| data_quality-03 | data | confirmed with corrections | 2 | 1 | No nightly completeness reconciliation of race_results against Betfair's race list |
| data_quality-04 | bug | confirmed | 2 | 1 | Scrape write path: the file's own date is not checked, and the upsert can neither blank nor de-duplicate an amended result |
| data_quality-05 | clean-up | confirmed with corrections | 2 | 1 | Three BSP sources are never reconciled, and the nightly 'Sync BSPs from Betfair' step is a dead step that can overwrite the target by a loose match |
| data_quality-06 | risk control | confirmed with corrections | 3 | 1 | The database is uploaded whole with no integrity, row-count or version guard, after steps that can be killed mid-write in WAL mode |
| data_quality-07 | data | confirmed with corrections | 3 | 1 | Price-file archive and load: no content check on archived files, parse failures never retried, replaced rows never removed |
| data_quality-08 | data | confirmed with corrections | 2 | 1 | Irish prize money in euros is read as pounds |
| data_quality-09 | data | confirmed with corrections | 1 | 1 | Dead heats and void races are settled and labelled as ordinary results in research |
| data_quality-10 | data | confirmed with corrections | 2 | 1 | The live record is loaded only for the last 3 days, with malformed lines skipped silently |
| data_quality-11 | data | confirmed with corrections | 1 | 1 | Race times are sorted as text and converted by a 'before 10 means pm' rule |
| **The research ledger** (15) | | | | | |
| ledger_meta-00 | model | confirmed with corrections | 3 | 1 | Forecast-price layer on the 06:00 price cut BSP error 0.396 -> 0.377 on live days and was never followed up |
| ledger_meta-01 | process | confirmed | 3 | 1 | The pre-registered forward CLV test (clv-forward-live) has never been read |
| ledger_meta-02 | process | confirmed | 3 | 1 | The deflated-Sharpe bar rests on four different trial counts; at about 320 trials it means t of about 4.6 |
| ledger_meta-03 | strategy | confirmed with corrections | 3 | 1 | Lay book was retired as 'value about nil' on a before-cut figure; after cuts it is about GBP38 a day and will be read at the harshest point |
| ledger_meta-04 | model | confirmed | 2 | 1 | 137,183 At The Races sectional runs are in the database but the served sectionals block still reads one year of site files |
| ledger_meta-05 | capacity | confirmed | 3 | 1 | The ledger's own numbers cap the morning taker edge near GBP3,000-5,000 of stake a day; the overnight re-run is unregistered |
| ledger_meta-06 | data | confirmed with corrections | 3 | 1 | The with-volume closing model was a look-ahead: +1.6 points in backtest, -3.9% live; TRADING.md still shows it as the better column |
| ledger_meta-07 | risk control | confirmed with corrections | 4 | 1 | The full live rule, daily balances and withdrawals are in a repository listed as public, and two real-money-capable models never entered the ledger |
| ledger_meta-08 | risk control | confirmed | 2 | 1 | The approved fix for the upgrade timer that kills the live session has waited since 5 Nov for three root commands |
| ledger_meta-09 | model | confirmed with corrections | 3 | 1 | Passed gains that were never served: three seeds, the learned gate, in-running and Timeform-pace blocks, and one untested BSP lead |
| ledger_meta-10 | clean-up | confirmed with corrections | 2 | 1 | Families the ledger has closed, with their dead processes still running inside the live job |
| ledger_meta-11 | data | confirmed | 2 | 1 | The headline record (+GBP6,314.40) breaks the funds rule by GBP67.97 and a GBP73.04 gap on 22 Oct was never reconciled |
| ledger_meta-12 | execution | confirmed with corrections | 2 | 1 | A higher stake floor on long prices won three replays and was retired before its registered forward test; under the DSR bar it is unprovable |
| ledger_meta-13 | execution | confirmed | 3 | 1 | Resting orders were tested only as a two-sided quote under a crude fill rule; the back side was never read alone and the stream has no registered test |
| ledger_meta-14 | strategy | confirmed with corrections | 2 | 1 | Place-market mispricing held out of sample at the SP; a limit-on-close place back was never tested |
| **Horse strategies** (12) | | | | | |
| strategy_horse-00 | bug | confirmed | 4 | 1 | The live rule silently skips a whole race when one Betfair-active runner has no model price (reserves, jockey-less, name misses) |
| strategy_horse-01 | strategy | confirmed with corrections | 3 | 1 | Trade the steamer signal where the price is (05:15-10:00, inside the hedged rule), not at 10:00 |
| strategy_horse-02 | risk control | confirmed with corrections | 3 | 1 | Steamer and Quick Returner models: live, unhedged, at the hour the edge is about nil - read their CLV, hedge at the SP, fix the summary |
| strategy_horse-03 | model | confirmed | 3 | 1 | Bookmakers' early prices as the closing model's missing market leg - captured at 07:40, after the hours that pay |
| strategy_horse-04 | strategy | confirmed | 2 | 1 | The lay book was retired on a biased figure; its reduced hedge value is about GBP35-39 a day at replay scale and it needs its own capital |
| strategy_horse-05 | execution | confirmed | 4 | 1 | Who pays the spread in the morning book? One tape study unblocks resting orders, deeper ladders and an honest fill model |
| strategy_horse-06 | bug | confirmed with corrections | 2 | 1 | Three different non-runner reduction rules in the scoring code, none matching the repo's own reading of Betfair's |
| strategy_horse-07 | capacity | confirmed with corrections | 3 | 1 | Overnight window (00:35-05:15 UK): identical forecasts and formed books; the evening result it would replace was scored with next morning's prices |
| strategy_horse-08 | strategy | confirmed | 2 | 1 | Place market: closed after two days of pre-off books; 41 days can now say whether the place book sits under its SP |
| strategy_horse-09 | strategy | confirmed | 3 | 1 | The place pocket passed its gate but 'cannot be traded as it stands': a limit-on-close back makes it tradable |
| strategy_horse-10 | strategy | confirmed | 3 | 1 | Near the off takers lose 0.65-2.0% to the BSP on both sides: register a horse stream lab before the first full stream day is read |
| strategy_horse-11 | process | confirmed with corrections | 2 | 1 | 76 nightly one-day scans never pooled, and each one raises the bar every new strategy must clear |
| **Other markets** (12) | | | | | |
| strategy_other-00 | strategy | confirmed | 4 | 1 | GB greyhound early drift is the best non-horse evidence - settle it on real fills before the 24 Nov gate is read as a licence |
| strategy_other-01 | bug | confirmed | 3 | 1 | The desk's deflated-Sharpe count for greyhounds and football is 317 ledger lines, not the roughly 31,000 rules tried |
| strategy_other-02 | strategy | confirmed | 3 | 1 | The place SP's mispricing against the win SP held out of sample and was never tried as a LIMIT_ON_CLOSE order |
| strategy_other-03 | capacity | confirmed with corrections | 3 | 1 | Measure the maker pool: aggressor-side CLV on the recorded streams sizes the early-layer edge with no fill model |
| strategy_other-04 | data | confirmed | 3 | 1 | Cricket is the deepest pre-off market on Betfair and nothing is held: a free owner action outstanding since 3 Oct, no recorder, two code gaps |
| strategy_other-05 | execution | confirmed | 2 | 1 | Football's line-up signal is real, but our feed is about 30 minutes late and the books are read every 5 minutes at release time |
| strategy_other-06 | data | confirmed with corrections | 2 | 1 | Asian handicap - football's second-deepest market with its cheapest tick - has never been scanned or recorded |
| strategy_other-07 | process | confirmed with corrections | 2 | 1 | The football EV trial counts only adversely selected fills, trades where 3.6% of the money is, and has no dated kill |
| strategy_other-08 | risk control | confirmed with corrections | 2 | 1 | Score thin non-horse strategies on what the account keeps: in the Expert Fee regime commission is free at the margin and gross profit pays 20% |
| strategy_other-09 | strategy | confirmed | 2 | 1 | Cup ties across divisions: the one football cell where the early price is measurably slow, read on 480 ties |
| strategy_other-10 | data | confirmed with corrections | 2 | 1 | Australia stays parked - and its two surviving greyhound candidates were scored at 2% commission against a measured 8% base rate |
| strategy_other-11 | risk control | confirmed with corrections | 2 | 1 | Three data licences stand between the non-horse models and real stakes |

## Appendix B - the evidence

`reports/edge_investigation_1111_findings.jsonl` holds, for each finding, the investigator's claim with its file and
line or ledger tag, the test proposed, and each checker's verdict, reasons, corrected claim, prior evidence and next
step.
