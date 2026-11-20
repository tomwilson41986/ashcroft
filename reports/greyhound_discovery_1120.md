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
