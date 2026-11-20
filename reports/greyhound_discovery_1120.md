# Greyhound discovery, 20 Nov 2026: the ranked ideas

racing² - greyhounds. The owner, 20 Nov 2026: "find a way to address these. Ideate and run discovery to improve our record on them." The map every agent read is `reports/greyhound_discovery_1120_map.md`; every idea with its full check is in `reports/greyhound_discovery_1120_ideas.jsonl`.

## How it was run

- Workflow `wf_0c719753-e27` (20 Nov): 4 inventors, one lens each (bets at the SP; resting orders and fills; information the market may lack; uncertainty and test design), 12 ideas each = 48. 3 skeptical checkers searched the ledger, the edge-search report and the code for duplicates, confirmed the data is held, checked each idea is safe at card or decision time, and gave two probabilities: a tradeable edge on real fills after 2% commission, and a measurable gain on its weakness.
- The container restarted before the ranking agent finished; its work was lost. The ranking below is a sort of the checkers' own figures (edge probability, then accuracy probability, then effort), not a fourth opinion.
- Rows: 48 invented = 44 ranked + 4 killed.

## What the round found

- **No idea is given more than a 5% chance of a tradeable edge.** The checkers' default is low because this market has resisted every rule tried.
- **The trial count now sets the bar.** With 37,931 greyhound rules counted (ledger `greyhound-trial-count-1120`), the deflated Sharpe needs a t-statistic near 5.8 (the checker of maker-12 computed an expected maximum Sharpe of 4.175 plus 1.645). The best standing candidate, the resting back `rest_back_60_m0`, is at about 3.5 on 1,652 trades: roughly 22 days at the same effect before it could pass. Nothing passes within a month.
- **Where the chance sits:** resting orders (earning the spread) rank above every bet at the SP; the first need is to prove the fill model against live orders.
- **Three leak faults were caught in the ideas as written** and fixed in their definitions: a bet at the SP may not be chosen by the BSP, its rank or its band (known only after the close).

## Wave 1: the six builds

| Build | Ideas | Weakness | What it settles |
|---|---|---|---|
| 0 (done) | S6 | W5 | One trial count from the ledger: 37,931 rules (`research/greyhound/trial_count.py`) |
| 1 | uncert-09, uncert-06 | W5 | Power and futility for every pending read and candidate; by-day cluster intervals |
| 2 | maker-08 | W4 | What every resting order that actually traded earned to the SP, with no queue model |
| 3 | maker-12, uncert-07 | W4 | The resting back under five fill assumptions; the gate is through-only plus 1 s |
| 4 | maker-01 | W4 | Whether any queue model reproduces the live orders of 11-14 Nov |
| 5 | sp-12, sp-10 | W1, W3 | One harness for bets at the SP with 2018-20 as an unseen block; first use: first-time-runner cells |
| 6 | sp-01 | W1 | A residual model on the BSP as an offset: does anything on the card add to the BSP |

Wave 2 after wave 1 reads: the card-time uncertainty model (uncert-01, W2), the two-part order-value model (maker-02), resting earlier (maker-04), queue position (maker-05), then the rest in rank order.

## The ranked list

Edge = the checker's probability of a tradeable edge on real fills after commission. Gain = its probability of a measurable gain on the weakness. Days = effort.

| # | Id | Wk | Kind | Idea | Edge | Gain | Days | Most likely failure |
|---|---|---|---|---|---|---|---|---|
| 1 | maker-08 (wave 1) | W4 | test_design | Maker markout census: what every resting order that actually traded earned to the SP, with no queue model at all | 0.05 | 0.15 | 1.5 | The volume-weighted maker return is the mirror of the takers' loss less commission and comes out near zero after 2% (the 'all' resting orders already read +0.15% / ... |
| 2 | maker-02 | W4 | model | Two-part order-value model: P(fill) x E[log(price/SP) / fill], trained on the replay's own all-runner orders | 0.05 | 0.10 | 2.5 | The model learns the simulator's fills, not live fills (L820), on about 15 days; E[pnl / fill] on a few thousand filled rows is noise; the selected set is the ... |
| 3 | maker-12 (wave 1) | W4 | test_design | The fill bracket as the forward gate for rest_back_60_m0: front of queue, queue as shown, through-only, each with ... | 0.04 | 0.50 | 1.5 | THROUGH keeps only fills where the price went through the order, the adverse ones: it will most likely be negative, as the through fill has been for every rule (map ... |
| 4 | maker-01 (wave 1) | W4 | test_design | Calibrate the stream queue model to the live orders (latency, volume factor, cancel-ahead), then re-score rest_v1 ... | 0.04 | 0.30 | 2 | No cell reproduces live: the aggregate fill rate already matches at the default cell, so the error is in WHICH orders fill (phantom fills on drifters, +0.4388 log ... |
| 5 | maker-04 | W4 | trade_rule | Resting earlier: the 120/180/300 s ladder on the stream and a 5-15 minute arm on the live-key books | 0.04 | 0.10 | 1 | The fill model is unvalidated and resting 2-5 minutes leans on it harder than 60 s does (cancellations ahead are never assumed, a trade beyond the price fills in full ... |
| 6 | maker-05 | W4 | execution | Join, improve or sit one level back: queue position as the decision at placement | 0.04 | 0.10 | 1.5 | One tick is 0.8% to 3.3% of the price between 2 and 20 (Betfair's ladder), larger than m0's +1.259%; the back row fills only on sweeps (the adverse case, L820). 8 to 10 ... |
| 7 | maker-11 | W4 | trade_rule | Rest in the PLACE market, priced off the WIN market: wider spread, takers who do not read the win book | 0.04 | 0.08 | 3 | Three earlier reads say the two books agree; PLACE books are thin, so few orders fill and those that do are the sweeps; capacity is pounds a race; the fill model is ... |
| 8 | maker-03 | W4 | execution | Cancel rules for the 60 s resting back: time-out, forecast flip, touch walks away, micro-imbalance turns | 0.04 | 0.05 | 1.5 | The base rule's +1.259% rests on a queue fill never matched to live (L820) with 82% of orders filling (ES line 475: 3,788 of 4,620). If most fills are early and benign ... |
| 9 | sp-01 (wave 1) | W1 | model | Residual softmax on the BSP offset, monotone in the price, served as limit-on-close limits (S1 sharpened) | 0.03 | 0.15 | 3 | The residual is real but tiny: trap_residual (L885) found A/E at the BSP of 1.0086 for trap 4, an order of magnitude under the commission hurdle; pit features improved ... |
| 10 | sp-10 (wave 1) | W3, W1 | feature | Debutant races at the SP: the newcomer and its rivals against the BSP | 0.03 | 0.08 | 1.5 | Blind portfolios at the SP lose about 1.3% after commission (ES family 5) and debutants are about 1.8% of runners (2,020 of 110,788, map W3), so the cell needs a bias ... |
| 11 | sp-02 | W1, W2 | model | The BSP's favourite-longshot exponent by venue, session and market size (one shrunk power model, no cell selection) | 0.03 | 0.06 | 1.5 | beta0 is about 1 and the segment deviations are noise: family 5's development-chosen segments lost in every holdout year (win backs -1.38%, lays -1.25%). A bias of 1% ... |
| 12 | maker-09 | W4 | feature | Queue turnover gate: money ahead of us divided by the flow that will clear it | 0.03 | 0.05 | 1 | A short queue relative to flow marks thin or just-moved levels, where the fill is the price moving through: the split may show the opposite sign. With 82% of resting ... |
| 13 | uncert-04 | W2, W4 | trade_rule | Resting orders with the margin in sigmas, not percent | 0.03 | 0.05 | 1 | Power and novelty. The quantiles are barely miscalibrated (0.223 and 0.778 against 0.20 and 0.80), so the conformal step changes little; and with the 60 s error near ... |
| 14 | maker-10 | W4 | trade_rule | Sweep catcher: rest away from the touch at the SP forecast's outer quantile and be filled only by sweeps | 0.03 | 0.05 | 1.5 | Too few fills and the wrong ones: the touch is beyond the band on about 50 occasions in 8 days, so a resting order a level further out fills less often still, and the ... |
| 15 | uncert-12 | W2, W4 | model | Forecast the size of the remaining move and rest two-sided quotes only on the calm runners | 0.03 | 0.05 | 1.5 | Selecting on calm leaves one-tick spreads with long queues where the only fills are through fills; selecting on wide spreads brings back the volatile runners. Both ... |
| 16 | maker-06 | W4 | trade_rule | SP-pool pressure gate: rest the back only where SP backers' liability exceeds SP layers', no forecast | 0.03 | 0.04 | 0.5 | The pool is a few tens of pounds and its effect is already in the near price, which the market and the GBM both see; the partial slope given spn_vs_mid will be near ... |
| 17 | sp-06 | W1 | feature | Today's track bias, learned race by race, bet at the SP later on the same card | 0.03 | 0.03 | 1.5 | The evidence is 2 to 8 races: the sum of (won - share) for a trap third over 5 races has a standard deviation near one winner, so the fitted K shrinks the columns to ... |
| 18 | uncert-01 | W2 (feeds ... | model | Card-time sigma: a runner-level forecast of the size of our own error | 0.02 | 0.35 | 2 | It passes a soft baseline and changes nothing: runs before and the forecast band already explain most of the spread (0.44 against 0.28), so the gain over a cell table ... |
| 19 | info-10 | W1 | feature | The kennel tonight: same-meeting results of the yard's earlier runners | 0.02 | 0.05 | 1.5 | Two or three earlier runners a night is a very noisy reading of a kennel's state (the sum of won minus chance over 2 runners is mostly luck), and whatever is real is ... |
| 20 | sp-08 | W1, W2 | model | A race-level temper: which races are more random than their SP book says | 0.02 | 0.05 | 2 | The crowd already prices crowding into each dog (the card-time model holds these columns and adds 0.000 at the decision book); a race-level temper of a few percent on a ... |
| 21 | info-07 | W1 | trade_rule | Room beside a vacant trap, priced at the SP | 0.02 | 0.04 | 1.5 | The market sees the empty box: the vacancy is on the screen for the whole life of the market and the line columns already help forecast the BSP, which says the price ... |
| 22 | info-01 | W3, W1 | trade_rule | Place SP against the dog's experience: the debutant's fat tail | 0.02 | 0.03 | 1 | Blind place bets at the SP sit about 1.3% under water after commission (ES family 5 place portfolios -1.30% backs, -1.25% lays); a cell of about 1.8% of runners needs a ... |
| 23 | info-12 | W1 | trade_rule | The weigh-in shock against the SP, split by sex | 0.02 | 0.03 | 1 | The weight's effect on time is under 0.2% and nil for a gain (model report line 1848), it is public at the track hours before the off, and the forecast gain from it is ... |
| 24 | sp-05 | W1, W4 | trade_rule | The BSP book-sum harvest: dutch the field at the SP when the SP book is forecast under 100% | 0.02 | 0.03 | 1.5 | B's race-to-race spread is reconciliation noise that the near and far prices do not forecast (the projected SP is the worst baseline in every window, edge search ... |
| 25 | info-11 | W1 | feature | The track tonight: a same-meeting trap and rail bias state | 0.02 | 0.02 | 1 | Four to eleven earlier winners cannot identify a rail bias: after shrinkage the index is about zero, and the on-course and exchange crowd see the same results. Blind SP ... |
| 26 | sp-03 | W1 | feature | Vacant-trap geometry against the BSP in four- and five-runner races | 0.02 | 0.02 | 1 | Trap effects at the BSP are about 1% in A/E (L885) while an SP back needs A/E near 1.025 to clear commission and the book (six-runner backs lose 2.15%). Carded ... |
| 27 | info-06 | W1, W4 | execution | Rest in the PLACE book at the win book's implied place price | 0.02 | 0.02 | 2 | The greyhound place book is too thin to fill a resting order except when someone informed crosses to it, so fills are few and adverse; the place SP is noisy for the ... |
| 28 | info-08 | W1 | feature | The non-runner shock: who really gains when a dog comes out | 0.02 | 0.02 | 2 | Too few removals with a meaningful book before them (most withdrawals precede any real book); the market reprices within seconds; four regressors on a few dozen races. |
| 29 | info-09 | W1, W3 | feature | Reserve runners: the dog that was not on the card | 0.02 | 0.02 | 2.5 | The status field does not identify reserves that ran, and the idea ends at the data check. If it does, a reserve is in the Betfair market for the whole betting period ... |
| 30 | uncert-07 (wave 1) | W4, W5 | test_design | Fill-conditional skill: score the forecast on the rows where an order actually fills | 0.01 | 0.40 | 0.5 | It confirms what the P&L already implies: for the 'all' rules pnl_sp is near zero, so on filled rows the mid-to-SP move roughly cancels the half-spread, and the ... |
| 31 | sp-12 (wave 1) | W5, W1 | test_design | The backward holdout and one SP harness: the 2018-22 joined rows read once | 0.01 | 0.40 | 2 | The clean backward block is smaller than claimed: only 2018-20 was unjoined (5-6%); 2021-22 was 55-71% joined and inside the model's fitting years. Price-file-only ... |
| 32 | sp-07 | W4 | execution | An SP reconciliation simulator: what our own SP bet does to the BSP | 0.01 | 0.15 | 3 | The last stream image is not the reconciliation state: Betfair's own near price, which uses the true rule on the full book, misses the SP by about 0.14 in log at 60 s ... |
| 33 | maker-07 | W4 | execution | Two-sided touch quoting scored at market level: paired fills net out, commission on the market's net | 0.01 | 0.05 | 0.5 | It cannot change the sign. A back filled at the best lay price and a lay filled at the best back price on the same runner, each closed at the SP, sum to exactly the ... |
| 34 | sp-09 | W1, W5 | test_design | The edge honesty layer: predicted SP edge mapped to realised SP return out of fold before any bet | 0.01 | 0.05 | 0.5 | It is a filter, not a source of edge: with the models at 0.000 a race beside the market (L994 Benter at the decision book) the mapped return's lower bound is below zero ... |
| 35 | uncert-02 | W2, W1 | test_design | Cross-fit select and value: split the winner's curse into shared and own error | 0.01 | 0.05 | 1 | 187 backs over 8 days give an interval on the share too wide to decide; the answer is most likely 'mostly shared' (the realised -3.5% is about the cost of crossing a ... |
| 36 | uncert-05 | W1, W2 | model | Precision-weighted Benter stage: the model's weight as a function of its own sigma | 0.01 | 0.04 | 1.5 | Low sigma is where the model already agrees with the market, so the model has nothing to add there; b1 comes out near zero, as the depth interaction did. |
| 37 | uncert-10 | W5 | test_design | Anytime-valid e-processes on the forward holdout, with e-BH across the registered rules | 0.01 | 0.20 | 1.5 | It adds a valid way to look early but changes no decision: the registered tests are already read once on fixed dates, the house rule still requires the fixed-date ... |
| 38 | info-05 | W3 | feature | Litter debut transfer: what the brothers and sisters did first time out | 0.01 | 0.04 | 1.5 | Only a fraction of the 1.8% of runners that are debutants have a littermate with an earlier debut and trial; the dam-progeny columns already carry most of it; the ... |
| 39 | info-04 | W3 | feature | Trial honesty: the trial-to-race offset by trainer | 0.01 | 0.02 | 2 | Size: the whole trials2 table gained 0.00102 and this is one more term on it, acting on debutants and dogs with under 3 runs (debutants are 2,020 of 110,788 test ... |
| 40 | uncert-06 (wave 1) | W5 | test_design | Day-cluster intervals for every greyhound gate, with the design effect shown | 0.00 | 0.55 | 0.5 | With 8 to 10 clusters the interval is wide and unstable whatever the method, so every rule fails and the table teaches little beyond 'wait'; a wild cluster bootstrap ... |
| 41 | uncert-09 (wave 1) | W5 | test_design | Power and futility table for every pending read and candidate | 0.00 | 0.55 | 1 | The table's message is one line (at 37,931 counted rules nothing passes within a month) and changes no decision unless the owner revisits how correlated grid rules are ... |
| 42 | uncert-11 | W5 | test_design | Snooping-adjusted model reads: Hansen's SPA test and a model confidence set over every saved arm | 0.00 | 0.25 | 1 | On RMSE it confirms what the paired 3-standard-error bar already says; on win log-loss the confidence set contains nearly every arm, which is W1 restated. Artifacts for ... |
| 43 | uncert-08 | W5 | test_design | One meta reading: empirical null, local false-discovery rate and Tweedie-shrunk returns over every rule tried | 0.00 | 0.20 | 1.5 | With four fifths of the counted rules unloadable the meta reading covers a minority; the one rule of interest (rest_back_60_m0) sits in a class of 12, too few for a ... |
| 44 | info-03 | W3 | model | Residual memory for dogs that have none: yard, litter and sire fallbacks | 0.00 | 0.05 | 1 | Runs-before 0 is 1.8% of test runners (2,020 of 110,788): a 0.0010 overall gain needs their RMSE cut from 0.44 to about 0.40, and two screens on the same miss found ... |

## Killed

| Id | Idea | Why |
|---|---|---|
| sp-04 | Fade the SP's reconciliation jump: limit-on-close orders either side of the last micro-price | The first reading it proposes exists and is run nightly, and it points the other way (weight on the SP's gap 1.20-3.12, model review 2.4); the limit-on-close fade is the only direction the order type can express; the same fade against an earlier reference ... |
| sp-11 | Race books at the SP that net the commission: stake-fixed backs, liability-fixed lays and the ... | Unbuildable on what is held (no passed signal to size) and its nearest version, whole-race books, was closed at 0 of 28 (L815). Keep only the market-net commission scorer as a detail of sp-02/sp-05. |
| info-02 | The yard's debutants against the Betfair SP (td_ table) | Duplicate of cx_tr_debut_ae (already in baseline_v0) and leaky beyond repair on the years it would be fitted and chosen on: the trainer on the day is held only from about Oct 2025. |
| uncert-03 | A bar in standard errors: posterior edge by normal-normal shrinkage at the decision book | A fourth calibration of a taker edge whose three predecessors failed on the same rows and whose signal at the decision book is measured at 0.000 a race; the stated outcome if there is no signal ('takes nothing') is already the ledger's reading, so the ... |

## One decision for the owner

The count of 37,931 treats every cell of a grid as an independent trial. Grid cells are highly correlated, so the number of independent trials is lower, but any lower figure is a judgement. The full count stands unless the owner says otherwise; build 1 shows what each choice implies for the pending reads.

## Build 1 result: power and futility (20 Nov; `research/greyhound/power.py`, inputs `research/greyhound/power_inputs_1120.json`)

A planning table, 0 trials. It cancels no registered read. t = the mean over its standard error from the by-day interval; a daily read needs 30 days before the deflated Sharpe is computed at all (`research/overfit.py MIN_OBS`), so every chance below is 0 at reads shorter than that.

Bar at 37,931 rules counted: t of at least 5.82. At other counts: 1: 1.645, 100: 4.175, 436: 4.656.

| Candidate | Fill | Days | Trades | Mean % | 90% | t | Days for 80% power | At half the effect | Days at its read | Chance of a pass then | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rest_back_60_m0, closed at the SP | queue-aware replay (unproven against live) | 8 | 1652 | 1.259 | 0.593 to 1.771 | 3.52 | 30 | 115 | 10 | 0.0 | reports/greyhound_edge_search.md, 19 Nov entry (close lab 44635535738) |
| lay_30_0.03, closed at the SP | takes the price on offer | 10 | 164 | 2.017 | -1.064 to 4.639 | 1.16 | 328 | 1312 | 10 | 0.0 | reports/greyhound_edge_search.md, 19 Nov entry (close lab 44635535738) |
| lay_30_0.03, run to the result | takes the price on offer | 10 | 164 | 53.5 | 24.611 to 85.4 | 2.9 | 53 | 212 | 10 | 0.0 | reports/greyhound_edge_search.md, 19 Nov entry (close lab 44635535738) |
| lay_10_0.1, closed at the SP | takes the price on offer | 9 | 47 | 7.528 | 0.791 to 13.009 | 2.03 | 98 | 389 | 10 | 0.0 | reports/greyhound_edge_search.md, 19 Nov entry (close lab 44635535738) |
| gap_lay_60_0.1, closed at the SP (stream lab's best) | takes the price on offer | 15 | 193 | 0.304 | -1.875 to 2.288 | 0.24 | 11535 | 46138 | 15 | 0.0 | reports/greyhound_edge_search.md, 19 Nov entry (stream lab 44630527322) |
| Early-lay forward test at the SP, a day | queue-aware replay (unproven against live) | 7 | no data | 1.31 | 1.047 to 1.573 | 8.19 | 30 | 30 | 28 | 0.0 | reports/greyhound_edge_search.md, day table 13-19 Nov; read 13 Dec (greyhound-early-lay-sp-close-1114 is the through-fill version) |
| Early-lay forward test to the result, a day | queue-aware replay (unproven against live) | 7 | no data | -0.336 | -1.351 to 0.68 | -0.54 | no data | no data | 28 | 0.0 | reports/greyhound_edge_search.md, day table 13-19 Nov |
| Live early-lay rule, through fill, live-key books, a day | through (what a live order gets) | 2 | no data | -1.705 | -2.305 to -1.105 | -4.67 | no data | no data | 28 | 0.0 | day replays 44427004281 (18 Nov), 44643820189 (19 Nov); earlier days not re-read here |

What it says:

- **Every taker rule is far from the bar**: the lab's two flagged lays sit at t 1.2 (at the SP) and 2.9 (to the result) against 5.82; at the same effect they need 53 to 328 days.
- **The fill model is the whole question.** The early lay at the SP under the queue-aware replay is positive on all 7 days (mean +1.31% of liability, t 8.2), above the bar even at 37,931 rules. The same rule under the through fill on the live-key books is -1.34% and -2.07% on 18 and 19 Nov. One of the two fill models is wrong, and the live orders of 11-14 Nov say which (build 4); the markout census (build 2) answers it with no fill model at all.
- **The resting back** (t 3.5 after 8 days) needs about 30 days at the same effect, 115 at half of it: late December at the earliest, under a fill model not yet proven.
- **The count matters less than the fill**: the bar is 5.82 at 37,931 rules, 4.66 at the ledger's 436 lines, 4.18 at 100. No candidate's verdict changes between those except the queue-replay early lay, which clears all of them and stands or falls on its fill.
- Gates with no deflated Sharpe written in: the favourite back (`reports/preregistration_greyhound_fav_back.md`, 2 Dec); the model reads (25 Nov, 7 Dec) are judged on paired intervals, not a Sharpe.

## Build 2 result: the maker markout census (20 Nov; run 44721231220; `reports/greyhound_maker_census_1120.md`)

Every trade on the recorded Exchange Stream, 5-19 Nov (15 days, GBP30.2m of signed volume), read against the book just before it. What the RESTING side earned closed at the SP, volume-weighted, net of 2% on a gain; 90% interval by day. No fill model. Descriptive: no cell is a rule.

| Market | Resting side | Net % | 90% |
|---|---|---|---|
| WIN | lay | +0.07 | -0.11 to +0.23 |
| WIN | back | -0.24 | -0.37 to -0.09 |
| PLACE | lay | +0.01 | -0.08 to +0.09 |
| PLACE | back | -0.07 | -0.16 to +0.01 |

- **On average a resting order earns nothing.** The spread a maker collects is given back in the move to the SP.
- **When it rests matters, and the two sides are opposite** (WIN, net):
  - 15 to 5 minutes out: resting lays +1.02% (0.14 to 1.88), resting backs -3.73% (-4.89 to -2.60);
  - 5 to 2 minutes out: resting lays +0.80% (0.56 to 1.04), resting backs -0.63% (-0.93 to -0.36);
  - the last 30 seconds: resting backs +0.30% (30-10 s) and +0.25% (10-0 s), resting lays -0.07% and -0.43%;
  - after the scheduled off: resting lays -0.77% (-0.95 to -0.56).
  Early money backs dogs shorter than their SP (the price then drifts out); late money backs them longer than it.
- **The spread decides whether a maker is paid** (WIN, net): at a 1-tick spread both sides lose (-0.38% lays, -0.43% backs); at 2 ticks both earn (+0.38%, +0.29%); resting lays at 3-4 ticks +1.28% and at 5 or more +2.51% (1.85 to 3.16). In the 5 to 2 minute window a resting lay earns -0.14% at 1 tick, +1.07% at 2, +2.37% at 3-4, +2.76% at 5+.
- **What it says about the two fill models.** The early lay rests minutes before the off, where resting lays that traded did earn about +0.8% to +1.0%: the queue-aware replay's +1.3% is the right size for the average resting lay there. The through fill's -2% is the worst slice of those fills (only when the price trades through the order). A real order earns something between the two, set by its place in the queue; build 4 reads which from the live orders.
- **What it does not say.** These are averages over every maker, most of them ahead of us in the queue; an order that joins the back of a 1-tick queue gets the fills the others did not want. The cells are 15 days read once after the fact; any rule built on a cell (rest lays only at 2 ticks or wider, 15 to 2 minutes out) is a new registration with its cell count, read on forward days. The census read 19 Nov, so such a rule's forward days start on 21 Nov.

## Build 4 result: the stream queue model against the live orders (20 Nov; run 44722114865; ledger `greyhound-queue-calibration-1120`)

The exact live early-lay orders replayed on the recorded stream under 32 settings. Fit on 11-12 Nov (824 orders, 246 matched live), judged once on 13-14 Nov (737 orders, 257 matched live). P&L of both sides closed at the SP, GBP, before commission.

| Replay setting | Days | Live fill rate | Replay fill rate | Same orders fill | Phantom / missed orders | Live P&L at the SP | Replay P&L at the SP | Gap |
|---|---|---|---|---|---|---|---|---|
| queue as shown, capped trade-through, half the printed volume (chosen) | fit 11-12 Nov | 30.0% | 30.6% | 98.7% | 8 / 3 | +2.71 | +1.62 | -1.08 |
| queue as shown, capped trade-through, half the printed volume (chosen) | judge 13-14 Nov | 34.9% | 35.5% | 98.8% | 7 / 2 | -172.40 | -179.74 | -7.34 |
| the same at the full printed volume (the resting test's own) | fit 11-12 Nov | 30.0% | 30.7% | 98.8% | 8 / 2 | +2.71 | +29.73 | +27.02 |
| the same at the full printed volume (the resting test's own) | judge 13-14 Nov | 34.9% | 35.5% | 98.8% | 7 / 2 | -172.40 | -151.40 | +21.00 |
| queue as shown, uncapped, full volume (the 14 Nov reconciliation's) | fit 11-12 Nov | 30.0% | 30.7% | 98.8% | 8 / 2 | +2.71 | +64.35 | +61.65 |
| queue as shown, uncapped, full volume (the 14 Nov reconciliation's) | judge 13-14 Nov | 34.9% | 35.5% | 98.8% | 7 / 2 | -172.40 | -121.13 | +51.27 |
| as the resting test's, order 1 s late | fit 11-12 Nov | 30.0% | 27.4% | 97.4% | 0 / 21 | +2.71 | -34.67 | -37.38 |
| as the resting test's, order 1 s late | judge 13-14 Nov | 34.9% | 33.7% | 98.0% | 3 / 12 | -172.40 | -195.34 | -22.94 |
| through-only | fit 11-12 Nov | 30.0% | 24.8% | 93.0% | 7 / 50 | +2.71 | -66.94 | -69.65 |
| through-only | judge 13-14 Nov | 34.9% | 29.3% | 92.8% | 6 / 47 | -172.40 | -265.18 | -92.78 |
| through-only, order 1 s late | fit 11-12 Nov | 30.0% | 23.5% | 93.5% | 0 / 53 | +2.71 | -114.13 | -116.84 |
| through-only, order 1 s late | judge 13-14 Nov | 34.9% | 28.1% | 92.7% | 2 / 52 | -172.40 | -292.28 | -119.88 |

- **A setting reproduces the live orders: TRUSTED.** Queue as shown, trade-through capped, half the printed volume, no latency. On the judge days it fills 35.6% of orders against live's 34.9%, the same orders fill on 98.8%, and its P&L at the SP is within GBP7 of live's (-179.74 against -172.40). Half the printed volume is what Betfair's two-sided volume print implies.
- **The through-only fill is too harsh, not the truth.** It misses 47 to 53 fills live really got, sits 5.6 to 6.9 points under live's fill rate and GBP93 to GBP129 under live's P&L on the judge days: NOT TRUSTED. The live rule's "-2% a day on the through fill" overstated the loss.
- **The uncapped full-volume replay is too kind**: +GBP51 and +GBP62 over live (the reason the 14 Nov reconciliation refused it).
- **A second of latency costs about GBP23 to GBP60** on these days and loses 12 to 21 real fills.
- **But the live early lay itself did not make money at the SP**: +GBP2.71 on 11-12 Nov and -GBP172.40 on 13-14 Nov on 503 matched lays. The fill model is now proven; the rule it was proven on is flat to losing. The paper forward test's +1.31% a day is the paper rule's own orders (decided on the recorder's books, about 90 s stale), not what the live trader placed: that gap is now the open question, and it is a question about the orders, not the fills.
- **What changes.** Queue-replay results are no longer "fill unproven": they are read under the trusted setting. The fill bracket gains that setting as an arm ("calibrated") with its own registered gate (`greyhound-rest-calibrated-1120`); the first gate (through + 1 s) stands as registered, now known to be the harsh bound.

## Build 3 result: the fill bracket, development days (20 Nov; run 44723196664; ledgers `greyhound-rest-bracket-1120`, `greyhound-rest-calibrated-1120`)

The resting test's 60 s orders on the 8 counted days 11-18 Nov, under six fills. Returns a unit of matched stake, 2% off a gain; t and 90% interval by day. Development only (0 trials): the two registered gates are read on 19 Nov - 18 Dec, once, on or after 19 Dec.

**The resting back where the close forecast agrees (rest_back_60_m0; 1,873 orders):**

| Fill | Matched | At the SP % | t | 90% | Gate settlement % | t |
|---|---|---|---|---|---|---|
| front of the queue | 93% | +1.987 | 4.81 | 1.307 to 2.666 | +1.809 | 4.58 |
| as shown (the resting test's own) | 88% | +1.259 | 2.76 | 0.510 to 2.009 | +1.140 | 2.81 |
| strict | 87% | +1.107 | 2.40 | 0.347 to 1.866 | +0.985 | 2.41 |
| **calibrated (reproduces live)** | 86% | **+0.880** | 1.78 | 0.065 to 1.696 | +0.450 | 0.96 |
| through-only | 80% | -0.317 | -0.63 | -1.148 to 0.514 | +0.692 | 0.53 |
| through-only, 1 s late | 80% | -0.348 | -0.67 | -1.204 to 0.508 | +0.662 | 0.51 |

- **Under the fill that reproduces live orders the rule is +0.88% at the SP, t 1.8** (the bar is 5.8); with late fills left to the result, +0.45%, t 1.0. The lab's headline +1.26% was the fill at the full printed volume, and its t by day is 2.8, not the 3.5 its by-race interval implied.
- **The forecast is what makes it positive.** Every runner, same fill: backs -0.135% (t -0.8), lays -0.713% (t -3.2). Where the forecast agrees: backs +0.880%, lays -0.224%. At a 3% margin: backs +1.611% (129 orders, t 1.5), lays +1.588% (109, t 1.2).
- **Adverse selection, measured.** Under the calibrated fill, filled lays were on runners that then shortened 1.7% to the SP and filled backs on runners that drifted 2.3%; the orders that did not fill were 13% the right way. A maker is filled on the wrong ones.
- **The forecast's skill on the rows that fill** (its squared-error gain over the mid on log SP): filled backs +0.00012 (90% -0.00001 to 0.00025), filled lays -0.00003 (-0.00012 to 0.00007). By the rule fixed before the run a forecast may gate a resting rule only if that interval is above zero: on these 8 days it is not, for either side (the backs miss narrowly).
- **Reading.** Nothing here passes or fails: the gates are forward. What the development days say is that the one standing candidate is a small positive (under +1%) that rests entirely on the close forecast, under a fill model now checked against live orders. At that size it needs months, not weeks, to reach the bar.

## Build 5 result: bets at the SP by a dog's races before (20 Nov; run 44726586742; ledger `greyhound-sp-debut-1120`): FAIL

A back of 1 or a lay risking 1 at the Betfair SP on every runner of a cell, netted by race, 2% off a gain. Return on the money at risk, 90% interval by day. Frame: 5,025,057 GBGB rows in; 3,959,362 race rows (683,263 races); dropped 811,707 rows of races not wholly priced, 2,010 outside four to six runners, 10,363 without one winner; 3,135,282 rows out (539,404 races), 6 Jan 2014 to 18 Nov 2026.

| Rule | Development 2023 - 18 Nov 2026 | 90% | Unseen 2018-20 | 90% | 2021-22 | 90% |
|---|---|---|---|---|---|---|
| Back a first-time runner (18,452 / 14,671 bets) | -4.86% | -7.93 to -1.79 | +3.94% | -0.08 to +7.97 | +4.86% | +0.52 to +9.20 |
| Lay a first-time runner | -0.62% | -1.24 to -0.00 | -0.90% | -1.48 to -0.32 | -2.22% | -2.85 to -1.59 |
| Back a dog with 1-2 races | -0.77% | -2.96 to +1.42 | +2.19% | -0.31 to +4.69 | -3.81% | -6.29 to -1.34 |
| Back a dog with 3-5 races | -1.02% | -2.81 to +0.78 | +1.35% | -0.44 to +3.15 | -3.68% | -5.70 to -1.65 |
| Back a dog with 6 or more (949,322 bets) | -1.10% | -1.50 to -0.70 | -0.99% | -1.45 to -0.53 | -0.90% | -1.35 to -0.45 |
| Lay a dog with 6 or more | +0.00% | -0.03 to +0.03 | +0.08% | +0.05 to +0.11 | +0.03% | -0.02 to +0.07 |
| Back the rival of a first-time runner | -2.16% | -3.51 to -0.81 | -1.20% | -2.48 to +0.08 | -2.38% | -3.88 to -0.88 |
| Lay the rival of a first-time runner | +0.01% | -0.15 to +0.16 | +0.17% | +0.02 to +0.33 | +0.43% | +0.26 to +0.59 |

(The lays of the 1-2 and 3-5 cells are negative in every period.)

- **No rule was chosen on development, so none passes.** On 2023-26 no cell has a return above zero with its interval.
- **The SP was once too long on first-time runners and is not now.** Backing every first-time runner made +3.9% in 2018-20 and +4.9% in 2021-22 (its interval above zero), and lost 4.9% in 2023-26. Their winners against the SP's chances: 1.040 (1.008 to 1.071) in 2018-20, 1.009 (0.985 to 1.032) since 2023. Whatever it was, the market has priced it; the same fade shows in laying their rivals (+0.43%, +0.17%, then +0.01%).
- **The SP is fair for experienced dogs, to the commission.** Backing all 949,322 costs 1.10%; laying them returns 0.00%.
- **For the model.** Its large error on first-time runners (RMSE 0.44) is our forecast's error, not the market's: the SP is calibrated for them now (A/E 1.009). Better first-run features can improve our price forecast; they will not find a bet at the SP.
- The harness (`research/greyhound/sp_harness.py`) stands for build 6 and any later rule at the SP.

## The paper early lay against the live early lay (20 Nov; run 44726189874; ledger `greyhound-paper-vs-live-1120`)

The question from builds 1 and 4: the paper forward test read +1.31% of liability a day, the live lays were flat to losing, and the fill model is now trusted. Fill setting or orders? Both order sets through one recorded stream on the four live days, 11-14 Nov. P&L closed at the SP before commission.

| Lays | Orders | Matched | Matched liability | P&L at the SP | Return on liability |
|---|---|---|---|---|---|
| Paper lays, the paper figure's own fill (full volume, uncapped) | 2,423 | 1,053 | GBP93,770.65 | +GBP1,300.60 | +1.387% |
| Paper lays, the calibrated fill | 2,423 | 1,039 | GBP76,586.00 | -GBP120.38 | -0.157% |
| Live lays, the calibrated fill | 1,561 | 513 | GBP41,021.03 | -GBP147.50 | -0.360% |
| Live lays, as the ledger settled them | 1,561 | 503 | GBP43,019.20 | -GBP169.69 | -0.394% |

Joined on the dog (calibrated fill): the 1,409 common lays -0.304% at the paper's price and time, -0.336% at live's; the 1,014 paper-only lays +0.052%; the 152 live-only lays -0.700%.

- **The paper edge was the fill setting.** By the reading fixed before the run: the calibrated fill removes the paper gain (+1.387% to -0.157%), so the cause was the fill, not the orders. The same holds on each of the four days (own fill +0.98% to +1.67%; calibrated -0.46% to +0.04%).
- **Where the phantom profit came from.** The number of lays filled barely changes (1,053 to 1,039); the liability does (GBP93.8k to GBP76.6k). The uncapped fill gave an order its whole remainder the moment the price traded through it; the capped fill gives it only what crossed. The filled lays' SP was on average 12% above their price, but the big fills are on the ones that went wrong: adverse selection in size, not in count.
- **Paper and live agree once the fill is right**: -0.16% (paper) and -0.36% to -0.39% (live) of liability. There is no paper-versus-live gap left to explain, and no edge: the early lay loses a little at the SP before commission.
- **What changes.** The nightly forward-test line now carries the trusted fill beside the two uncapped ones (`research/greyhound/stream_lab.py forward_check`). The daily "+1.3% to +1.9% at the SP" figures in `reports/greyhound_edge_search.md` (13-19 Nov) were the uncapped fill and should be read as overstated. The early lay stays off; its registered reads (24 Nov gate; SP-close paper test 13 Dec) run to their end as registered.

## Build 6, stage 1: a residual model on the BSP (20 Nov; run 44727761528; ledger `greyhound-sp-offset-1120`)

A race softmax on the winner that starts at the BSP's own log chance and may only learn corrections from the card: 408 card-safe features, no price of the race; 1,431,246 runners of 252,670 wholly priced races (2 Jun 2020 to 10 Nov 2026). The winner's log-loss a race, the BSP alone against the BSP plus the residual; 90% interval by day.

| Test year (fitted on everything before) | Races | Rounds | BSP | BSP + residual | Gain a race | 90% | t |
|---|---|---|---|---|---|---|---|
| 2024 | 51,775 | 60 | 1.52827 | 1.52782 | +0.000446 | 0.000176 to 0.000716 | 2.72 |
| 2025 | 48,829 | 72 | 1.50744 | 1.50665 | +0.000793 | 0.000535 to 0.001052 | 5.05 |
| Jan - May 2026 | 19,464 | 214 | 1.49347 | 1.49266 | +0.000816 | 0.000195 to 0.001437 | 2.16 |
| Pooled | 120,068 | | 1.51416 | 1.51351 | **+0.000647** | 0.000461 to 0.000834 | 5.71 |

- **By the registered reading this is a PASS**: the pooled interval is above zero and every fold is positive. It is the first reading in which anything adds to the Betfair SP out of sample.
- **The registered placebo proves nothing, and that is my fault.** Shuffling the winner within the race, with the BSP as the starting score, teaches the trees to undo the BSP (772 rounds, gain -0.123): its interval is "at or below zero" for a reason that says nothing about a leak. A second placebo is added and registered before its run: each fit race's feature rows dealt at random among its runners, so nothing a dog's own card says can be learnt and the gain must be nothing.
- **It is small.** 0.00065 a race is 0.04% of the loss. The models are small too (60 to 214 rounds of 15 leaves). The features carrying it in the last fold: recent trouble in running (`trouble_mean3`, `trouble_mean1`, `trouble_mean6`), the trainer's record against the market (`trainer_ae`, `gp_trainer30_ae`), the trainer's share at the track, the sire, a class drop's record, days since the last run.
- **A gain in log-loss is not a bet.** Commission takes 2% of a winning back, and a bet at the SP must be chosen without seeing the SP. Stage 2, registered next, tests exactly that with the card-only residual.

## Build 6, stage 2: bets at the SP from the card-only residual (20 Nov; run 44729248379; ledger `greyhound-sp-offset-stage2-1120`)

Same fits as stage 1. The second placebo, then six rules at the SP chosen from the residual centred on its race (no price of the race in it), cuts from each fit's early-stopping days. A back of 1 or a lay risking 1, netted by race, 2% off a gain; return on the money at risk, 90% interval by day.

**The second placebo is clean.** Each fit race's feature rows dealt at random among its runners: the fit stops after 1 round and its gain is +0.000003 (-0.000015 to +0.000021). The stage 1 gain is not an artefact of the method.

| Rule | Bets | Return | 90% | t | 2024 | 2025 | Jan - May 2026 | Mean BSP |
|---|---|---|---|---|---|---|---|---|
| Back the top 10% | 64,452 | **+4.99%** | 2.98 to 6.99 | 4.09 | +4.49% | +4.59% | +7.19% | 11.3 |
| Back the top 5% | 31,444 | +6.67% | 3.66 to 9.69 | 3.64 | +3.38% | +6.51% | +15.10% | 11.5 |
| Back the top 2% | 12,139 | +7.20% | 2.36 to 12.04 | 2.45 | +0.22% | +7.41% | +23.25% | 11.6 |
| Lay the bottom 10% | 64,935 | +0.99% | 0.67 to 1.30 | 5.08 | +0.88% | +0.77% | +1.80% | 11.4 |
| Lay the bottom 5% | 31,987 | +1.30% | 0.86 to 1.74 | 4.82 | +0.78% | +0.94% | +3.51% | 11.9 |
| Lay the bottom 2% | 12,697 | +0.94% | 0.22 to 1.67 | 2.14 | +0.58% | +0.45% | +3.01% | 12.6 |

- **By the registered reading all six rules pass this development stage**: each pooled interval is above zero and each rule is positive in all three test years.
- **It is not accepted yet.** A back that returns 5% at the Betfair SP after commission on 64,452 bets contradicts everything else in this file (the SP's bias by segment: 0 of 17,822 rules; build 5: the SP fair to the commission). The likeliest explanation of a result this good is a fault, and the one the two placebos cannot see is a feature that reads something of its own race's result. The check for exactly that exists (`greyhound.live --check-card`: a past day's features built from the morning card against the same day built from the results, feature by feature) and was last read on 6 Oct, when the model had 316 features, not 408. It is running now for 12 Nov (run 44730736615). **Until it is read, treat the table above as unverified.**
- If the features are clean, the next step is the one registered: the residual served each day from the card and read on forward days from 21 Nov, with the deflated Sharpe at every greyhound rule counted (a t of 5.8). The development t-statistics here are 2.1 to 5.1.

## Build 6, the leak checks (20 Nov)

**1. A day's features do not read that day's results** (`greyhound.live --check-card`, run 44730736615, 12 Nov, 680 runners with a GBGB history): the day built from the morning card (history to the day before) against the same day built from the results. Of 435 features, 400 agree on at least 99% of runners. The features that carry the residual and are NOT among the 35 that differ: `trouble_mean1`, `trouble_mean3`, `trouble_mean6`, `days_since`, `gp_lbsc_l1`, `lb_fb_cross_w`, `gsr_std6`. Every trainer feature (23 of them, `trainer_ae` and `gp_trainer30_ae` among them) agrees on the same 98.09% of runners and differs on the same 13 dogs: the card names the trainer of the dog's last run, the results the trainer on the day (a kennel change) - a different trainer, not today's result. `sire_win`, `age_months` and the birth-month features agree on 98.97%. A feature that included today's results would differ for most runners, not 2%. So: no feature of the residual reads its own day's results. (The 13 trainer changes are a train-to-live skew for the live model, noted for it.)

**2. Does any feature read a LATER day?** Not shown by the check above (both of its builds stop at the day). Registered and running: the fast-loop cache (one build through its last day) against the engine run again on runs and prices up to 30 Jun 2025 only, every card-safe feature of June 2025 compared runner by runner (`research/greyhound/cut_check.py`, ledger `greyhound-cut-check-1120`). Stage 2 stays unverified until it is read.

**3. Does a race see its own result through a twin row?** (`research/greyhound/twin_check.py`, run 44742292946, ledger `greyhound-twin-check-1120`.) Neither check above can see a race held twice in the data: the second copy's last-run features would be the race's own result, in the cache and in any rebuild alike. A twin is a dog with rows in two races on one day; a short gap is a run a day or less after the last.

| Year | Runners | Races | Twin rows | Short-gap rows | Races touched |
|---|---|---|---|---|---|
| 2020 | 10,906 | 1,826 | 0 | 0 | 0 |
| 2021 | 160,792 | 27,122 | 2 | 2 | 4 |
| 2022 | 204,034 | 34,658 | 0 | 0 | 0 |
| 2023 | 278,472 | 48,804 | 0 | 0 | 0 |
| 2024 | 289,564 | 51,775 | 142 | 0 | 24 |
| 2025 | 269,970 | 48,829 | 0 | 0 | 0 |
| 2026 | 217,508 | 39,656 | 6 | 0 | 6 |
| All | 1,431,246 | 252,670 | 150 | 2 | 34 |

34 races of 252,670 are touched, so the study was rerun without them, as registered, and build 6 is read from the rerun:

| | Original (run 44729248379) | Without the 34 races (run 44742292946) |
|---|---|---|
| Stage 1 gain a race, pooled | +0.000647 (0.000461 to 0.000834), t 5.71 | +0.000410 (0.000211 to 0.000609), t 3.40 |
| 2024 / 2025 / Jan-May 2026 | +0.000446 / +0.000793 / +0.000816 | +0.000034 / +0.000551 / +0.001057 |
| Rounds at the early stop | 60 / 72 / 214 | 2 / 264 / 170 |
| Placebo 2 | +0.000003 | +0.000006 (-0.000055 to 0.000068) |
| Back the top 10% | +4.99% (2.98 to 6.99), t 4.09 | +3.53% (1.55 to 5.51), t 2.93 |
| Back the top 5% | +6.67%, t 3.64 | +4.41% (1.60 to 7.22), t 2.58 |
| Back the top 2% | +7.20%, t 2.45 | +2.58% (-1.97 to 7.14), t 0.93 |
| Lay the bottom 10% | +0.99%, t 5.08 | +0.84% (0.53 to 1.15), t 4.44 |
| Lay the bottom 5% | +1.30%, t 4.82 | +1.24% (0.82 to 1.67), t 4.82 |
| Lay the bottom 2% | +0.94%, t 2.14 | +1.24% (0.58 to 1.91), t 3.08 |

- **By the registered reading build 6 still stands**, on the rerun: stage 1's pooled interval is above zero with every fold positive, and five of the six rules keep an interval above zero (the top-2% back does not). No rule reaches the deflated Sharpe bar (a t of 5.82): the best is 4.82.
- **It is not a leak that moved the numbers; the fit is unstable.** 34 races cannot carry a third of the gain, and the 2025 fold changed with no touched race in its test. The fits differ by 24 to 166 rows of 654,204 to 1,213,738, yet the early stop moved from 60 to 2 rounds in the first fold. One fit's stopping point is noise, so the by-day intervals understate the uncertainty, and a single fit is not fit to freeze for a forward test.
- **Next, registered before its run** (`greyhound-sp-offset-seeds-1120`, 14 rules): the same study with the residual as the mean of 10 fits that differ only in the seed, run on two disjoint sets of seeds (1-10 and 11-20), the 34 races left out. Stage 1 and each rule stand only if they hold in both halves. If stage 1 does not, build 6 is closed as not robust and nothing is frozen.
- The 142 twin rows of 2024 (24 races) are a data fault, listed for the data audit.

**2, read. Does any feature read a later day?** (run 44741593052.) 22,291 runners of June 2025 on 408 card-safe features, the cache against the engine run on runs and prices to 30 Jun 2025 only: 382 features agree on every runner, 406 on at least 99%.

| Feature | Agree | Runners that differ | Median gap |
|---|---|---|---|
| `gp_bsp_ae_car` | 91.8% | 1,825 | 0.0054 |
| `gp_bsp_mkt_car` | 91.8% | 1,825 | 0.0131 |
| 24 trainer features | 99.955% to 99.991% | 2 to 10 | - |

- **By the registered reading the single-fit readings of build 6 are withdrawn**: two of the 408 features the residual reads differ on more than 1% of runners (the dog's career mean against the Betfair SP). Neither is in a fold's top 12 by gain, but their gain is not shown to be nothing. The seed-averaged study is the rerun without them (406 features; amended in the ledger before its run).
- **Likely cause, not verified: an older history, not a later day.** The cache was built to 11 Nov, before the BSP join was fixed and backfilled, so today's rebuild knows Betfair SPs for 2018-2022 runs that the cache did not. Only the career window reaches that far back; the last-run, 3, 5 and 10-run windows of the same measure agree on every runner. To verify: the cut check again once the cache is rebuilt on the backfilled prices.

## Build 6, the seed-averaged study (20 Nov; run 44745244638; ledger `greyhound-sp-offset-seeds-1120`)

The residual as the mean of 10 fits that differ only in the seed, run twice on separate seeds. The 34 twin-touched races are out and `gp_bsp_ae_car`, `gp_bsp_mkt_car` are barred (406 features). 1,431,047 runners of 252,636 wholly priced races; 876 test days, 120,038 test races. This replaces the single-fit readings above.

**Stage 1: the gain a race over the BSP.**

| | Seeds 1-10 | Seeds 11-20 |
|---|---|---|
| 2024 | +0.000596 (t 3.99) | +0.000525 (t 4.34) |
| 2025 | +0.001080 (t 5.79) | +0.001164 (t 5.60) |
| Jan - May 2026 | +0.001008 (t 3.24) | +0.000887 (t 3.04) |
| Pooled | **+0.000860** (0.000676 to 0.001043), t 7.70 | **+0.000844** (0.000662 to 0.001025), t 7.65 |
| Placebo 2 | +0.000002 (-0.000027 to 0.000031) | -0.000004 (-0.000040 to 0.000032) |

A single fit's own gain in the 2024 fold ran from 0.000055 to 0.000510 (2 to 199 rounds). The mean of ten is what is stable: the two halves agree to 2%.

**Stage 2: bets at the SP, return on the money at risk after 2% commission.** The deflated Sharpe is at 37,996 greyhound trials; the bar is 0.95.

| Rule | Seeds 1-10 | t | Deflated Sharpe | Seeds 11-20 | t | Deflated Sharpe |
|---|---|---|---|---|---|---|
| Back the top 10% | +6.43% (4.34 to 8.51) | 5.07 | 0.819 | +6.39% (4.41 to 8.38) | 5.30 | 0.846 |
| Back the top 5% | +6.97% (3.99 to 9.95) | 3.85 | 0.228 | +6.26% (3.26 to 9.25) | 3.44 | 0.133 |
| Back the top 2% | +6.66% (1.95 to 11.36) | 2.33 | 0.013 | +6.57% (1.83 to 11.31) | 2.28 | 0.004 |
| **Lay the bottom 10%** | **+1.23%** (0.93 to 1.54) | 6.65 | **0.990** | **+1.42%** (1.11 to 1.73) | 7.56 | **0.9995** |
| Lay the bottom 5% | +1.39% (0.96 to 1.82) | 5.33 | 0.914 | +1.53% (1.09 to 1.96) | 5.75 | 0.931 |
| Lay the bottom 2% | +1.40% (0.73 to 2.07) | 3.44 | 0.355 | +1.64% (0.98 to 2.31) | 4.04 | 0.385 |

Bets: 64,381 and 64,682 for the top-10% back; 64,525 and 64,166 for the bottom-10% lay. Every rule is positive in all three test years in both halves.

- **By the registered reading stage 1 and all six rules stand in both halves.**
- **One rule clears the deflated Sharpe with every greyhound trial counted, in both halves: lay the bottom 10% of the residual at the SP.** It is the first greyhound rule to do so.
- **It is still a development result, not accepted for money.** The test years (2024 to May 2026) have been read by other greyhound work; the feature cache predates the BSP backfill; the settlement used the price files' SP, not a live record; and what the SP can take in a greyhound race is not measured.

## Build 6, the forward paper test (registered 20 Nov; ledger `greyhound-sp-offset-forward-1120`)

- **Frozen once, today**: the mean of 20 fits (seeds 1-20) on the cache to 11 Nov, the same races and features as the study. Never refitted.
- **Served from 21 Nov** at 07:47, 10:47, 13:47 and 16:47 UTC from the morning card; a race is written once, at its first serve 10 minutes or more before its off.
- **Settled daily on paper** at the live recorder's Betfair SP, a different data path from the study's price files. No Betfair key, no order.
- **The gate is one rule, lay the bottom 10%, read once at 120 settled forward days (about 21 Mar 2027)**: the forward days' own 90% interval above zero, and the deflated Sharpe on the study's and the forward days together. One futility look at 45 days.
- Power: at 120 days the expected forward t is about 2.5 to 2.8, so roughly an 80 to 87% chance of clearing zero if the effect is what the study measured.
