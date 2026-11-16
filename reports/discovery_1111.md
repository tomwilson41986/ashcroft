# racing² - discovery: new features, new feature engineering, new model forms

11 Nov 2026. The owner's brief: "Go beyond historic and traditional forms of horse racing analysis, handicapping,
modelling and betting. Identify NEW features that have not been discovered yet that predict the outcomes and winning
probabilities of horse races. Discover new ways to engineer features to make them more predictive. Identify new forms
of modelling that haven't been discovered yet. I only want new, undiscovered approaches."

Nothing here has been tested. Every "prior" is a checker's judgment, not a result. Ids in brackets point to the
evidence file `reports/discovery_1111_ideas.jsonl`.

## 1. In short

| Question | Answer |
|---|---|
| How many ideas? | 95, proposed by six inventors who had first read seven maps of what is already known. 12 were the same idea twice. 83 were checked. |
| How many are new? | 58 survive: not in this repository, buildable from data we hold or can get free, and with only a partial precedent found. 14 were rejected as already published or practised, 10 dropped by the checker, 1 is already here. |
| Is anything wholly new? | No. Not one of the 83 came back as "no precedent found". Every survivor has a known ingredient. What is new is the construction, or its use on the closing price rather than on the result. |
| Will they work? | The checkers' priors are low, and I think rightly. For improving the BSP forecast or closing model: 4 ideas are rated 3 of 5, 52 are rated 2, 2 are rated 1. For predicting the result beyond the BSP: all 58 are rated 1. That agrees with our own record: the BSP is efficient against results, and our largest gain came from the market's own past prices. |
| What kind are they? | 23 features, 17 model forms, 12 feature-engineering methods, 4 data sources, 2 betting forms. 35 aim at the closing price, 23 at both. |
| What do they need? | 27 use data held, 28 data partly held, 3 free public data. Effort: 2 in hours, 10 in a day, 43 in days, 3 in a week or more. |
| What to test first? | Nine that are cheap, use data held, and bear on the move from the morning price to the close. Section 7. |

## 2. How it was done, and its limits

- **The maps.** Seven maps were written first, so that a known idea could not pass as new: the features served or
  built here, every block with its ledger verdict, the data held and not held, the model forms tried; the published
  literature, practitioner and commercial practice, and methods from other fields. They are in
  `reports/discovery_1111_maps/` (11,852 words, each line with its path, ledger tag, citation or URL).
- **The inventors.** Six, each with a family of lenses, 28 lenses in all: the physics and energetics of the horse,
  health and development, weather and ground, National Hunt; the morning book, inverting the market, the information
  that arrives between 06:00 and the off, anomalies, the crowd; trainers' intent, incentives, the handicapper, the race
  programme, riders, unexposed horses; form networks, the hidden race, natural experiments, career dynamics, pedigree,
  text; feature-engineering methods, representation learning, the probability structure of a race, transplants from
  other fields, the model's knowledge of itself; positions and alternative data. Each read the maps, was told that
  nothing on them counts as new, and returned 15 or 16 ideas.
- **The checks.** One checker per batch of four to six ideas did four things for each: searched the repository and
  the ledger for the same quantity; confirmed the inputs are held and known at 06:00; ran at least three web searches
  for a published or practised precedent (272 searches in all); and gave a skeptical prior. When in doubt the checker
  chose the less novel verdict. Where the checker corrected a definition, the corrected one is given.
- **Limits.** This is the lean run. The first run (28 inventors, two checkers an idea) was stopped by the account's
  usage limit before any inventor finished. "Partly known" means a partial precedent was found in about three
  searches; it does not prove that no fuller one exists, and the unpublished practice of syndicates and vendors cannot
  be searched at all. A second generation (ideas bred from the survivors) was not run. The specifications in section 4
  are assembled from the inventor's definition and the checker's corrections, not rewritten by a designer.

## 3. The survivors, ranked

"Forecast" and "result" are the checker's priors, 1 (very unlikely) to 5 (likely and material), that the idea improves the BSP forecast or closing model, and that it predicts the result beyond the BSP. "Mechanism" is how sound the checker found the causal story, 1 to 5.

| # | Idea | Kind | Data | Forecast | Result | Mechanism | Effort | Spec | Id |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Market-implied going gap traded as a meeting basket | betting form | partly held | 3 | 1 | 3 | a week or more | 4.12 | `positions_data-00` |
| 2 | Owner crowd size: racing-club members as late money | feature | partly held | 3 | 1 | 3 | days | 4.11 | `market-11` |
| 3 | Temporal graph memory network trained on closing books | model form | held | 3 | 1 | 3 | a week or more | 4.13 | `method-11` |
| 4 | Sparse functional principal components of market-rating careers | model form | partly held | 3 | 1 | 3 | days | 4.10 | `hidden-10` |
| 5 | Stamina limit as an interval-censored dose-response | model form | held | 2 | 1 | 3 | days | section 5 | `physical-04` |
| 6 | Low-sun obstacle forecast from solar geometry and course bearings | feature | partly held | 2 | 1 | 3 | days | section 5 | `physical-10` |
| 7 | Ghost runs: prices of withdrawn horses as hidden rating readings | feature | partly held | 2 | 1 | 3 | days | 4.1 | `market-03` |
| 8 | Substitution-aware re-formation: cross-elasticities of the book beyond renormalisation | model form | partly held | 2 | 1 | 3 | days | 4.5 | `market-06` |
| 9 | Roy-model inversion: private ability implied by the race chosen | model form | partly held | 2 | 1 | 3 | days | 4.14 | `agents-03` |
| 10 | Campaign-state hidden Markov model of a horse's runs | model form | held | 2 | 1 | 3 | a week or more | section 5 | `agents-09` |
| 11 | Effective resistance of the form graph as a knowability gate | feature | held | 2 | 1 | 3 | days | 4.8 | `hidden-04` |
| 12 | Learned forgiveness: the market's verdict on each past run | feature engineering | held | 2 | 1 | 3 | days | 4.3 | `market-01` |
| 13 | Kin-pooled within-horse response slopes, priced versus realised | feature | held | 2 | 1 | 3 | days | section 5 | `hidden-11` |
| 14 | Owner fingerprint: betting owners and crowd-owned horses | feature | partly held | 2 | 1 | 3 | days | section 5 | `agents-07` |
| 15 | In-running exit price of fallers as a censored performance observation | feature engineering | held | 2 | 1 | 3 | a day | 4.6 | `physical-00` |
| 16 | Survive-then-contest likelihood for jump races | model form | partly held | 2 | 1 | 3 | days | section 5 | `physical-07` |
| 17 | Hoof-traffic wear state of each strip of track | feature | partly held | 2 | 1 | 3 | days | section 5 | `physical-08` |
| 18 | Effort-furlongs - an energy-equivalent trip for every course, going and code | feature engineering | partly held | 2 | 1 | 3 | a day | section 5 | `physical-09` |
| 19 | Reliability-weighted book reconciliation: fresh runners price the stale ones | feature engineering | held | 2 | 1 | 3 | a day | 4.4 | `market-02` |
| 20 | Window-by-window population signatures scaled by expected crowd size | model form | partly held | 2 | 1 | 3 | days | section 5 | `market-10` |
| 21 | Conditions as a decoder: private valuations read from auction and claiming weights | feature engineering | partly held | 2 | 1 | 3 | days | section 5 | `agents-02` |
| 22 | Debut-slot value from the yard's matured earlier cohorts | feature | held | 2 | 1 | 3 | days | section 5 | `agents-06` |
| 23 | Jockey's forgone book: opportunity cost of today's rides | feature | partly held | 2 | 1 | 3 | days | section 5 | `agents-08` |
| 24 | Horses-in-training sale catalogues: shop-window runs and resale valuations | data source | free public | 2 | 1 | 3 | days | section 5 | `agents-12` |
| 25 | Phase-resolved jockey skill matched to the horse's own deficit | feature engineering | held | 2 | 1 | 3 | days | section 5 | `agents-13` |
| 26 | Kinship prior for the market-rating filter (pedigree BLUP on past BSPs) | model form | held | 2 | 1 | 3 | days | 4.18 | `hidden-00` |
| 27 | The handicapper as a sensor: reaction-function residual of the mark | feature | partly held | 2 | 1 | 3 | a day | section 5 | `hidden-02` |
| 28 | Stale-mark news: two-vintage differencing keyed to the horse's last trade | feature engineering | partly held | 2 | 1 | 3 | days | section 5 | `hidden-03` |
| 29 | On-day/off-day mixture and the field-size variance premium | model form | held | 2 | 1 | 3 | days | section 5 | `hidden-09` |
| 30 | Reappearance-hazard residuals: own return surprise and rivals' excess absence | feature | partly held | 2 | 1 | 3 | days | section 5 | `hidden-12` |
| 31 | Reason-split closing combine: TreeSHAP families inside the market blend | model form | held | 2 | 1 | 3 | days | 4.17 | `method-00` |
| 32 | Conditions-aware low-rank market rating from past Betfair SPs | model form | held | 2 | 1 | 3 | days | 4.2 | `method-01` |
| 33 | One market-taught relevance kernel reused for every windowed feature | feature engineering | held | 2 | 1 | 3 | days | section 5 | `method-08` |
| 34 | Stochastic frontier rating: ability as a frontier with one-sided shortfalls | feature | held | 2 | 1 | 3 | days | section 5 | `method-10` |
| 35 | Intraday franking trade on scheduled form-line news | betting form | held | 2 | 1 | 3 | a day | 4.9 | `positions_data-06` |
| 36 | Franking by absence - rivals who never came back | feature | held | 2 | 1 | 2 | days | 4.15 | `physical-01` |
| 37 | Crosswind echelon - wind-vector forecast of the day's draw-side and cover bias | feature | free public | 2 | 1 | 2 | days | 4.16 | `physical-03` |
| 38 | Market-implied going shift read from the meeting's cross-section of moves | model form | partly held | 2 | 1 | 2 | days | section 5 | `market-04` |
| 39 | Odd-lot footprint: stake-size digits separate model money from human money | feature | partly held | 2 | 1 | 2 | a day | section 5 | `market-09` |
| 40 | Rival-avoidance index: field shortfall against eligible supply | feature | partly held | 2 | 1 | 2 | days | section 5 | `agents-00` |
| 41 | Cut-line pressure before oversubscribed handicaps | feature | partly held | 2 | 1 | 2 | days | section 5 | `agents-04` |
| 42 | Market-implied volatility per horse from prices across field compositions | model form | held | 2 | 1 | 2 | days | section 5 | `method-03` |
| 43 | Yard-base training interruption index from reanalysis weather | data source | free public | 2 | 1 | 2 | days | section 5 | `positions_data-03` |
| 44 | Attention share of the day: ex-ante neglect of a race | feature | held | 2 | 1 | 3 | a day | 4.7 | `market-14` |
| 45 | Dead-weight coupling from each jockey's inferred riding floor | feature | held | 2 | 1 | 2 | a day | section 5 | `physical-13` |
| 46 | Habitual late drifter: a paddock trait inferred from last-minutes price history | feature | partly held | 2 | 1 | 2 | a day | section 5 | `market-05` |
| 47 | Desk drift and pool exchange rates for official marks | model form | partly held | 2 | 1 | 2 | days | section 5 | `agents-14` |
| 48 | Conditional surprisal of the in-running comment given the bare result | feature engineering | held | 2 | 1 | 2 | days | section 5 | `hidden-05` |
| 49 | Pool exchange rates: priced versus realised offsets across form circuits | model form | held | 2 | 1 | 2 | days | section 5 | `hidden-06` |
| 50 | Yard informedness from the price-performance lead-lag area | feature engineering | held | 2 | 1 | 2 | days | section 5 | `method-04` |
| 51 | Oestrous phase clock recovered from a filly's own run residuals | feature | held | 1 | 1 | 2 | hours | section 5 | `physical-06` |
| 52 | Coupled hidden yard-health state with a fade-signature emission | model form | partly held | 2 | 1 | 2 | days | section 5 | `physical-15` |
| 53 | Public-eye score: a preference model trained on the industry-SP and Tote wedge | feature engineering | partly held | 2 | 1 | 2 | a day | section 5 | `market-07` |
| 54 | The race programme as a flow graph: prep and target roles of each race | feature engineering | held | 2 | 1 | 2 | days | section 5 | `agents-11` |
| 55 | Error twins: an analog ensemble in the forecast's own leaf space | feature | partly held | 2 | 1 | 2 | days | section 5 | `method-14` |
| 56 | Promotion-driven uninformed lay flow from extra-place offers | data source | partly held | 2 | 1 | 2 | days | section 5 | `positions_data-07` |
| 57 | Late-crowd salience from public attention data | data source | partly held | 2 | 1 | 2 | days | section 5 | `positions_data-10` |
| 58 | Card-order attention with the alphabetical natural experiment | feature | held | 1 | 1 | 2 | hours | section 5 | `market-12` |

## 4. Specifications: 18 ideas, in the order to test

### 4.1 Ghost runs: prices of withdrawn horses as hidden rating readings
*feature - target: closing price - novelty: partly known - data: partly held - effort: days - id `market-03`*

**The idea.** Every time a horse was declared, traded and then withdrawn, the market measured it; feed those prices into the market rating and read the pre-withdrawal drift as a tell.

**Why it could work.** A results database has no row for a horse that did not run, so models and form-driven morning quotes never see that the horse was 5/2 when taken out last week. People who watched do, and they bet later. For lightly raced horses the ghost reading can be the freshest or only one; a drift before withdrawal marks a setback.

**Definition (as corrected by the checker).** Rows: BASIC WIN runners, UK/IE, status REMOVED with a trade before removalDate, Jan 2024 on. p_g = (1/LTP_j) / sum of 1/LTP over runners active and traded at the last minute before removalDate (at least 90% of the field traded). y_g = ln p_g + C_j as market_rating.py; variance lambda x R x (1 + hours to the off/3), lambda fixed on Jan-Jun 2024. Features: mrg_ (mr_ rebuilt with ghosts), gh_n365, gh_days, gh_since_run, gh_drift = ln(LTP at removal / ltp_first). Match on normalised name and day.

**Data.** betfair_historic.py:253-255 keeps runner_status and LTP marks; raw BASIC files (1-minute last traded, market definition) stored from Jan 2024 (betfair_historic.py:14,96) - the share at removal needs a re-parse. Live books from 30 Sep 2026. No volume.

**Known at 06:00?** Yes. The rating's q and R are fixed on races before 2021-01-01 (market_rating.py FIT_UNTIL); lambda can only be fitted on 2024+, inside the scored folds - fix it on Jan-Jun 2024 and score from Jul 2024. Use only trades before removalDate.

**Closest precedent.** A scratch since the last run as a negative signal: scratched horses significantly more likely to need extended rest (https://thoroughbreddailynews.com/in-numbers-scratched-horses-significantly-more-likely-to-need-extended-rest/); Scratched horses' morning-line odds converted to probabilities and used as a model input (for odds error, not ability) (https://amp.horseracingnation.com/news/What_are_the_reasons_for_Morning_Line_Odds_errors_123); Forum threads on withdrawn-horse statistics; no price-as-rating use (https://theracingforum.co.uk/forums/topic/withdrawn-horse-statistics)

**What is new.** Entering a withdrawn horse's pre-removal exchange share as a measurement in a market-rating filter, and the pre-withdrawal drift as a feature; no precedent found for either.

**Already here.** No block or ledger entry reads a removed runner's price as a feature. sources/open_close.py:150 filters REMOVED rows; race_results has runners only (L:career-truncation); market_rating.py reads BSP only. The open lead 'non-runner risk' (reports/model_review_1111.md:83-92) forecasts withdrawals, not ability. L:morning-card-gap-1110: 351 of 5,932 card runners did not run.

**First test.** First a read-only count: removed runners with a trade, and the share of 2024-26 runners with gh_since_run = 1 (stop under 2%). Then block model/blocks/ghost_rating.py: served + mr_ against + mrg_/gh_ against placebo. Pass: next real reading's predictive log-likelihood better on gh_since_run runners (horse-clustered 90%), overall error not worse.

**Most likely to fail because.** Most withdrawals come on the morning from books with about GBP1,400 matched (L:open-close-inventory-1101), so the ghost share is a stale last-traded print, lambda fits large, and too few runners are affected to clear the 0.0007 bar.

### 4.2 Conditions-aware low-rank market rating from past Betfair SPs
*model form - target: closing price - novelty: partly known - data: held - effort: days - id `method-01`*

**The idea.** Turn the scalar market rating into a horse vector times a learned conditions embedding, so past prices say what the market thinks of this horse at this trip, ground and track type.

**Why it could work.** The BSP prices the same horse differently at 5f on firm and 7f on soft; the scalar rating averages them and trees must re-learn aptitude from thin results (cond_form gave -0.0002). The market's own conditions-specific view, pooled across horses through shared factors, is the sharpest prior for a horse on a trip or ground it has rarely met, where thin morning books lean on raw recent form.

**Definition (as corrected by the checker).** r_i(x) = a_i + v_i . g(x), K = 2. x only from columns held on both sides: log dist_furlongs, coarse going group, surface_type, race_type; class and course shape left out. Reading y_i = log p_i + C_i from bfsp as in market_rating.py, rivals' ratings taken at that race's x. g and the stallion and trainer priors fitted on races before 2021-01-01 only; per-horse Kalman on (a_i, v_i); v_i held at its prior under 5 readings spread over at least 2 going or trip bands. Features: mr2_rating_x, mr2_sd_x, mr2_apt = v_i . g(x_today), mr2_share_x.

**Data.** race_results: bfsp, going, distance furlongs, surface, track, class, stallion, trainer, 2010 on (scraper.py:121-193 per repo_data map). Course-shape class is not a held column (only coarse geometry in model/abm/track.py).

**Known at 06:00?** With changes. No look-ahead. Train/serve mismatch: past x uses the result's going, today's the morning going (27% differ per repo_data map, L:card-vs-results-1110), and class is blank on 27% of cards (model_review_1111.md:43 per map). Fix: coarse going groups, drop class and course shape from x.

**Closest precedent.** Covariate-varying dynamic Plackett-Luce ranking model, applied to horse racing and tennis (https://www.arxiv.org/pdf/2406.16507); Elo as a steady-state Kalman filter extended to correlated skills across surfaces (https://econpapers.repec.org/RePEc:bpj:jqsprt:v:17:y:2021:i:3:p:203-219:n:2); Horse rankings built from closing odds and margins, used to forecast closing odds better than the morning line (https://www.inpredictable.com/2017/11/betting-market-rankings-for-horse-racing.html)

**What is new.** Applying a covariate-dependent, low-rank state to odds-implied readings rather than results. Each ingredient is published; the combination was not found.

**Already here.** model/blocks/market_rating.py (scalar Kalman rating from bfsp; L:iter116 -0.0067). model/blocks/cond_form.py already reads the market's view at today's trip band, going group and course against overall: L:iter57 -0.0002 unresolved. sire_apt -0.0006 (L:iter52), pedigree_market -0.0006 (L:iter84). Open lead 5, suitability (reports/model_review_1111.md:87). Differs: class-corrected readings, low-rank horse loading, sire/trainer prior, sd by distance from priced conditions.

**First test.** Drop-in block model/blocks/market_rating_cond.py per model/blocks/__init__.py, not served. 11-fold quick screen as iter114. Arms: base + mr_; + mr2_; + placebo; + cond_form's mkt features as the cheap control. Pass: paired error at least 0.0007 below base + mr_, interval excluding zero, better than the cf_ control, Brier and concordance not worse. Effect size: guess, small.

**Most likely to fail because.** A horse has few readings, mostly at the conditions its trainer chose, so v_i stays at its sire/trainer prior and the block collapses to market-priced sire aptitude, which was already retired, while trees with mr_ and the conditions recover the rest.

### 4.3 Learned forgiveness: the market's verdict on each past run
*feature engineering - target: closing price - novelty: partly known - data: held - effort: days - id `market-01`*

**The idea.** Use how far the next closing price moved after each past run, relative to that run's result surprise, as a per-run reliability weight on all form, and forecast it for the latest run.

**Why it could work.** The closing crowd knows excuses that never reach a database - jockey and vet reports, video, yard illness. Its next price shows which runs it believed. Thin 06:00 books are set largely off raw form, so a bad run of a type the close habitually forgives is over-counted in the morning and corrected by the off.

**Definition (as corrected by the checker).** s_k = race-centred pounds beaten less its expectation given BSP share (fit on days before k). u_k = market-rating reading at run k+1 less mr_rating before run k. g(s) = E[u|s] by code, fitted on matured runs only. Use the additive form a_k = u_k - g(s_k), weighted by reading precision 1/(P+R); no ratio. (a) form-window measures re-weighted by exp(c x a_k), c fixed before 2021. (b) fg_last = walk-forward boosted prediction of a_k from run circumstances (labels: runs whose successor precedes the fold), plus its race z-score.

**Data.** race_results bfsp, total_dst_bt, comment, going, headgear 2010-2026 (scraper.py:121-193; comments from 2012); market rating readings from 2018 (model/blocks/market_rating.py).

**Known at 06:00?** With changes. g(s) and the model of phi need labels (the next run's price) that mature later: each must be fitted only on runs whose NEXT run is before the scored fold's first day. One fit over the whole cached matrix leaks. Today's BSP must never be the 'next reading' of the last run.

**Closest precedent.** Handicapping practice of 'forgiving' a poor run with an excuse (drawing a line through it) (https://www.brisnet.com/racing/news/draw-line-through-poor-past-performances/); A model of closing odds on past finishing margins estimating that the market gives a single race about 25% weight - the market's learned reaction to a run, in aggregate (https://www.inpredictable.com/2018/11/introducing-robocap-automated-horse.html); Analyst adjustments for bad-luck runs entered as model variables (Benter, W. (1994) Computer based horse race handicapping and wagering systems: a report, in Hausch, Lo and Ziemba (eds), Efficiency of Racetrack Betting Markets)

**What is new.** The per-run realised market update set against the update expected for that result surprise, used as a weight on the horse's own runs, and a model forecasting that quantity for the last run before its successor exists.

**Already here.** model/blocks/market_rating.py:37-38 mr_surprise, mr_trend (the market's update after the last run, L:iter116 -0.0067 holdout); fw_ market view and A-E windows (model/form_windows.py); fu_ rivals' later market change (L:iter62: -0.0046 on price, no outcome information); cm_ excuse classes +0.0019 (L:iter45). Differs: the update set against the expected update per own run as a form weight, and a forecast of it for the last run.

**First test.** Block model/blocks/forgiveness.py (fg_; FEATURES, READS, build per model/blocks/__init__.py) on the cached matrix. Arms: served + mr_; + fg_; + placebo; 11 folds, ledger entry first. Pass: paired mean absolute log-BSP error falls by more than 0.0007 (L:iter114) and the placebo does not.

**Most likely to fail because.** The market rating already is the market's posterior after every run and the booster already learns the next price from last-run circumstances, so fg_ re-expresses mr_ and the target, while the per-run ratio of two noisy readings is mostly noise.

### 4.4 Reliability-weighted book reconciliation: fresh runners price the stale ones
*feature engineering - target: closing price - novelty: partly known - data: held - effort: a day - id `market-02`*

**The idea.** Replace proportional normalisation of the thin morning book with a sum-to-one estimate that moves each runner's price in proportion to how wide, shallow and stale its own quote is.

**Why it could work.** At 08:00 the median market has GBP1,423 matched (ledger open-close-inventory-1101): many runners show one stale order. The book must still sum to one, so the well-quoted runners jointly fix the price of the badly quoted ones. Proportional normalisation spreads the error over everyone; later money corrects exactly the runners this estimate moves.

**Definition (as corrected by the checker).** Per active runner at snapshot t: m_i = (1/back1 + 1/lay1)/2; no lay quote: m_i = 0.5/back1 and v_i = the race's largest v. v_i = ((1/back1 - 1/lay1)/2)^2 x (1 + age_i/a0)/(1 + min(back1_size, lay1_size)/d0), age_i = seconds since back1, lay1 or last_traded last changed. x_i = m_i + (1 - sum m) x v_i/sum v, floor 0.001, renormalise. Features: ln(x_i / normalised market_now_i), v_i/sum v. a0, d0 fixed on days before 20 Oct. Drop the cross-implied flag.

**Data.** betfair_recorder.py:54-60: back1-3, lay1-3 with sizes, last_traded, market_total_matched, from 30 Sep 2026; 300 s polls on the delayed key, so quote age is coarse. Stream ladder first due 12 Nov 2026, none held (L:stream-late-signal-1111).

**Known at 06:00?** Yes. None in the feature. a0 and d0 must be fixed on days before the scored days. The delayed key adds minutes of lag, so measured staleness is partly the feed's.

**Closest precedent.** Betfair's cross-matching: virtual best prices on each runner implied by the offers on the others to a 100% book, returned with virtualise=true - the idea's cross-implied price (https://betfair-developer-docs.atlassian.net/wiki/x/QQwp); Margin removal with unequal allocation across outcomes (margin proportional to odds, power, Shin, additive) (https://rdrr.io/cran/implied/f/vignettes/introduction.Rmd); Balancing estimates to an accounting constraint in proportion to their variances (Stone, R., Champernowne, D. and Meade, J. (1942) The precision of national income estimates, Review of Economic Studies 9(2))

**What is new.** Applying variance-proportional balancing to an exchange book, with each quote's variance built from its spread, depth and age; no use of that on a betting book was found. The cross-implied price is not new.

**Already here.** model/race_book.py:423 market_now (geometric mid if lay/back <= 1.25, else the back) and :88 proportional normalising. L:benter-blend-1020 blended by spread and market money (+4.3% to +6.4-7.7% on sampled reads); that spread blend lost in the trader's replay (L:closing-refit-replay-1109, -GBP1,447). L:move-model-1021: spread, path, matched money 2-3% of gain each. Differs: residual-to-one allocated by quote variance with quote age.

**First test.** Read-only query on betfair_live_marks 08:00-10:00 UK, days from 20 Oct: |ln x - ln BSP share| against race_book.market_now normalised, paired by race. Pass: race-bootstrap 90% interval of the drop below zero. Only then a registered replay on replay_1108.py's machinery with x as the closing model's market input.

**Most likely to fail because.** Weighting the market price by spread and liquidity already looked better on sampled reads and lost through the trader's own replay, and the reconciled price mostly moves runners with no tradable quote.

### 4.5 Substitution-aware re-formation: cross-elasticities of the book beyond renormalisation
*model form - target: closing price - novelty: partly known - data: partly held - effort: days - id `market-06`*

**The idea.** Model how a move or withdrawal of one runner redistributes probability to its close substitutes rather than pro rata, and forecast the knock-on moves the morning book has not made.

**Why it could work.** Money leaving a withdrawn or drifting horse goes to what resembles it: the yard's other runner, the other front-runner, the other mudlark. Betfair's factor and quoting bots renormalise pro rata at once; the true pattern appears as later money arrives. The gap is predictable from similarity.

**Definition (as corrected by the checker).** Removals only. For each cut removal j (betfair_live_marks.removal_utc, adjustment_factor) and survivor i: e_i = ln(share_i at BSP / share_i at last mark before removal) - ln(1/(1 - p_j)), shares from back1/lay1 mids. Regress e_i on separate centred kernels: same trainer (betfair_live_markets.trainer), adjacent price rank, stall_draw within 2, same projected run style; one coefficient each, ridge. Drop owner (not held) and the co-movement rows. Feature is event-driven: after a removal seen in 05:15-10:00, nr_i = sum_c theta_c x (K_c,ij - Kbar_c,j) x p_j/(1 - p_j). No 06:00 feature until theta is shown non-zero.

**Data.** betfair_live_marks: adjustment_factor, removal_utc, back/lay 1-3 (betfair_recorder.py:54-62), from 30 Sep 2026 only; trainer in betfair_live_markets. Owner not held at card time (daily_predictions.py:654 strips it). Pace projection built, not served (L:iter25).

**Known at 06:00?** With changes. Co-movement rows are simultaneous: i's and j's moves share one interval, so theta is not causal, and common shocks (a going change moves all mud horses together) give the opposite sign to substitution. Fix: estimate on removals only, or on j's move in the previous interval; one theta per kernel, not a mean K.

**Closest precedent.** IIA and its relaxation by nested/mixed logit (similar alternatives substitute more) - textbook; multinomial logit as the racing standard (https://www.Gwern.net/doc/statistics/decision/1986-bolton.pdf); Betfair reduction factors fixed in advance from forecast prices; traders discuss their mismatch with the live book (https://betangel.freshdesk.com/support/solutions/articles/80001073678-rule-4); Forum discussion of reduction-factor distortions in small fields (https://theracingforum.co.uk/forums/topic/betfair-and-the-reduction-factors/)

**What is new.** Measuring, on a market's own repricing after a withdrawal, whether the freed share goes to similar runners rather than pro rata. The theory (IIA failure, nests) is textbook; no empirical precedent was found in three searches.

**Already here.** Not built. Nearest: L:removal-by-market-1109 (849 cut removals: factor mean 11.20% vs 10.97% of book, survivors +0.19% mean - average only, no split by similarity); stable block st_ +0.0018 worse (L:iter47-quick-stable); mixed/nested logit listed 'next', never built (RESEARCH_FRAMEWORK.md:171). Grep 'substitut|cross.elastic' in model/, research/, ledger: no relevant hit.

**First test.** Read-only query under research/queries/, registered in the ledger first: survivors of every cut removal 2 Oct on; e_i at the next mark and at the BSP on the four kernels, race-clustered. Pass: one kernel's coefficient with the same sign and t > 2.5 in both halves of the days. No block until then.

**Most likely to fail because.** About 849 cut removals in 38 days of thin, wide books (relative spread 18.9% at 08:00, L:open-close-inventory-1101) cannot resolve five kernels, and at 06:00 the term is second-order (guess: under 0.005 log units a runner).

### 4.6 In-running exit price of fallers as a censored performance observation
*feature engineering - target: both - novelty: partly known - data: held - effort: a day - id `physical-00`*

**The idea.** Treat a horse that fell or pulled up as a censored run and impute how well it was going from the lowest in-play price it reached before exit, instead of reading the run as blank or last.

**Why it could work.** About 16% of jumps runs end with no finishing position (the block's own priors 0.06 + 0.10, model/blocks/jumps_form.py:49), so the form book shows F or P and the served features read unknown or last. The in-running market watched the horse travel. The morning price handles 'fell last time' crudely; people who watched the replay correct it by the off.

**Definition (as corrected by the checker).** Jumps rows. For each earlier run whose comment matches FALL_RX / PU_RX (model/blocks/jumps_form.py:52-53) with betfair_prices bsp > 1, ipmin > 1 and ip_vol above a floor fixed in advance: x = max(0, logit(1/ipmin) - logit(1/bsp)). Features, falls and pull-ups apart: x at the last such run, mean of last 3, flag ipmin <= 0.5 x bsp, runs since. Imputed arm: the non-finisher's placing = its rank by ipmin in that race, fed to the nfp0 windows (form_variants). No market-rating update. Earlier days only; no price record = NaN.

**Data.** betfair_prices bsp, ipmin, ip_vol (betfair_prices.py:355-365; from 31 Dec 2017, L:open-close-inventory-1101; win IPMIN sound, L:place-fields-check-1002). race_results comment from 2012 (L:career-truncation). ip_vol is the runner's whole in-play volume, not volume at the low.

**Known at 06:00?** Yes. none found - past runs only. Price files lag a day, so a run yesterday has no record at 06:00: leave NaN in training too.

**Closest precedent.** In-play low/high of a horse's past runs offered as a system-builder field and built here as ir_ (/home/user/ashcroft/model/blocks/inrunning.py:3-9); Press and punters track horses beaten after trading very short in running (https://racingpost.com/news/wetherby-wipeout-in-running-carnage-after-three-horses-beaten-at-1.01-axM1U4R2IRG9); In-play lows by run style (back-to-lay) as a practised data product (https://www.geegeez.co.uk/tag/dobbing/)

**What is new.** Only the censoring step: splitting the in-play low by exit type (fall vs pulled up) and using it to impute the non-finisher's placing in the form windows. No precedent found for that step; the raw quantity is built and practised.

**Already here.** model/blocks/inrunning.py:9-18 already builds ir_lowr_l1/m3 = log(in-play low / BSP) for every past run with a price record, fallers included (main -0.0008 L:iter109; blend -0.0002 unresolved L:iter110). form_variants.py:21 reads non-finishers as last. jumps_form adds -0.0006 on jumps, unresolved (L:iter117-nh-vs-flat). Differs: conditions the low on exit type and imputes a finish. market_rating reads BSPs, not results - nothing to impute there.

**First test.** Ledger entry first. Drop-in block model/blocks/faller_exit.py on the cached matrix, jumps rows, 11 folds. Arms: base+jf, base+jf+ir, base+jf+ir+new, placebo. Pass: new beats the ir arm by more than 0.0007 (L:iter114) and the placebo spread. Read-only query: x at last exit against ln(morningwap/bsp) at the next run, walk-forward t > 3.

**Most likely to fail because.** ir_lowr_l1 already holds the same number on the same rows and gained only -0.0008, and a faller's low often reflects rivals departing or thin trades rather than how it was travelling.

### 4.7 Attention share of the day: ex-ante neglect of a race
*feature - target: closing price - novelty: partly known - data: held - effort: a day - id `market-14`*

**The idea.** Forecast each race's share of the day's exchange turnover before any money arrives and use neglect as a gate on how much to trust the morning price over the model.

**Why it could work.** Traders and bots have finite capital and attention and price the day's main races first. A race expected to take a small share of the day stays near its tissue price longer and moves later, more so on days with many meetings - the distraction effect seen around busy earnings days in equities.

**Definition (as corrected by the checker).** vol_r = sum of pp_vol over a race's runners (betfair_prices), divided by the trailing 90-day mean race volume. E_r = empirical-Bayes mean of ln vol by track x race_type x prize band x weekday, days <= D-2, shrunk to the track mean (k = 20). a_r = exp(E_r) / sum over the day's carded UK/IE races; clash_r = races scheduled within 10 minutes. Interact E_r, -ln a_r and clash_r with ln market-now and ln model in the no-volume closing model. Level (E_r) and share (a_r) are separate arms.

**Data.** betfair_prices pp_vol per runner (betfair_prices.py:355-365), from 31 Dec 2017 (L:open-close-inventory-1101; L:hrb-features-1006 says 11 Apr 2019, 84.8% of runs - unresolved). Day's card at 06:00 has time, track, race type; class blank for 27%, mostly Irish (reports/model_review_1111.md:43).

**Known at 06:00?** With changes. Price files lag a day, so E_r must use days up to D-2. Exchange volume trends over years - scale by a trailing 90-day market total or the feature tracks the calendar. Class is blank on Irish cards at 06:00, so the cell must use card-available fields (course, race type, prize, weekday). The day's race list at 06:00 includes later abandonments - compute a_r on the card as seen.

**Closest precedent.** Hirshleifer, Lim, Teoh 2009: competing announcements slow price reaction (the distraction effect the idea imports) (https://ideas.repec.org/a/bla/jfinan/v64y2009i5p2289-2325.html); Smith, Paton, Vaughan Williams 2006: exchange efficiency compared by race grade, restricted to liquid markets (https://ideas.repec.org/a/bla/econom/v73y2006i292p673-689.html); Brown and Yang 2017: traded volume on UK races predicts fundamentals; speculative trade aids efficiency (https://ueaeprints.uea.ac.uk/58626/)

**What is new.** The within-day share and clash count, known before any trade, used as the weight between morning price and model in the closing forecast. That thin low-grade races are priced later and less well is published and common trader knowledge.

**Already here.** Closing model already interacts both log-probabilities with runner and race traded volume (reports/race_book_model.md:32-43); served without volume since 9 Oct (trading/config.py:51; L:owner-rule-1009: volume model -2.1% CLV vs no-volume +7.9%). Static race-context interactions (hours to off, field size, jumps) changed nothing: 0.3020-0.3025 vs 0.3021 (L:closing-model-v2-0930). Differs: expected, not realised, volume as a share of the day, plus clash count.

**First test.** Module beside model/race_book.py (research/, no live change), ledger entry first. Arms on the closing-model-v2 design (walk-forward by month, mean |log q - log q_close|): served no-volume; + E_r; + a_r; + clash_r. Pass: paired race-bootstrap interval below zero on both scored months and holdout, and the model's weight rising with neglect; CLV at the owner's staking not worse.

**Most likely to fail because.** Expected volume is close to a function of course, race type and weekday, the share adds nothing beyond that level, and every earlier race-context or volume term in the closing model was flat or hurt live.

### 4.8 Effective resistance of the form graph as a knowability gate
*feature - target: closing price - novelty: partly known - data: held - effort: days - id `hidden-04`*

**The idea.** Measure, by electrical resistance through the whole who-met-whom graph, how precisely public form can compare each runner with its rivals, and let that set how far the morning price should be trusted.

**Why it could work.** Where the graph ties a field tightly, anyone can price it and the morning price is near final. Where resistance is high the tissue is guesswork, private knowledge decides, and moves are larger. It says when the model or the morning price deserves weight, which volume bands only proxy.

**Definition (as corrected by the checker).** Two pools (Flat, jumps) as mr_. Graph on day t: horses with a run in 730 days; w_ij = sum over shared races before t of exp(-age_days/365) x q (q = 1 if both within 20 lb of the winner by total_dst_bt, else 0.3). R_ij from a global grounded sparse solve per pool per day (no 3-step truncation - it biases R up; 'exact' holds only for a homoscedastic least-squares rating). Features: er_mean, er_min, er_rank, er_field, er_disc; debutants at the cap with a flag. Used as columns and as a tercile interaction on the model weight in model/race_book.py.

**Data.** race_results raceid, horse, place, total_dst_bt 2010-2026 (scraper.py:121-193; 1,938,313 rows, RESEARCH_FRAMEWORK.md:330); betfair_prices morningwap, bsp for the test.

**Known at 06:00?** Yes. none found if the graph uses days before today only; the 06:00 field includes later non-runners (351 of 5,932, L:morning-card-gap-1110), so er_field moves when the field shrinks.

**Closest precedent.** Effective resistance sets the error of pairwise-comparison rating estimates (Hendrickx et al. 2019; Chen 2023) (https://proceedings.mlr.press/v97/hendrickx19a.html ; https://arxiv.org/abs/2304.06821); 'Connectedness' / prediction error variance of a difference between units - the same statistic, standard practice in animal-breeding evaluations (https://pmc.ncbi.nlm.nih.gov/articles/PMC5439142); Graph-based features for horse-race prediction (university thesis, abstract only seen via search) (https://core.ac.uk/download/217210354.pdf)

**What is new.** Using the pairwise comparison variance of today's field as a gate on how far the morning price is trusted; the statistic itself is published theory and routine in breeding evaluation.

**Already here.** No Laplacian or resistance code (grep model/, research/, scripts/, reports/: 0 hits). Neighbours: cl_rivals, cl_links counts (collateral.py; L:iter66 -0.0015 beside form lines, nothing beside cw L:iter68), mr_sd, mr_n, mr_read_share (market_rating.py), kr_ sd. The gate use repeats L:closing-model-v2-0930 (field size, jumps/flat, gap interactions: 0.3020-0.3025 vs 0.3021, not adopted).

**First test.** Read-only query: one year, |log(morningwap/bsp)| on er_mean with controls (race type, mean career_runs, mr_sd, mr_read_share, log morning_vol); proceed only if er adds out of sample beyond the controls. Then closing logit with er tercile x model weight vs served, walk-forward by month; pass = BSP-book nearness gain with bootstrap interval above zero; ledger first.

**Most likely to fail because.** er tracks race type, career runs, mr_sd and morning volume, so it predicts the size of the move but adds nothing beyond those controls, and a size-only signal does not change a point forecast of the BSP.

### 4.9 Intraday franking trade on scheduled form-line news
*betting form - target: closing price - novelty: partly known - data: held - effort: a day - id `positions_data-06`*

**The idea.** When a horse that recently met a later runner wins or flops earlier the same day, trade the later runner if its price has not yet moved by the amount form-line history implies.

**Why it could work.** Franking is priced day to day here (rivals' later form moved prices: fu_ block -0.0046 on price, L:iter62). Within the day, few people watch a later race's exchange while an earlier race settles. The link list is knowable at 06:00 and the news arrives at a scheduled minute, so the only question is reaction speed, which recorded books can answer.

**Definition (as corrected by the checker).** For runner A, links to B running today at least 20 minutes earlier, where they met in either's last 3 runs (race_results). w_AB = exp(-|total_dst_bt gap| / 3) x 0.85^(runs ago). s_B = (bfsp rank - finishing rank) / (ranked runners - 1), the id_ bmr residual (model/inday_features.py:18). Predicted d ln p_A = -theta x sum w_AB s_B, theta fitted walk-forward. Event study first; trade only if A's move from the last mark before B's off to +5 min is under half the prediction and the rest exceeds spread + 3%; close at SP. Separately test the link count as a sigma input to closing_draws.

**Data.** race_results (time, total_dst_bt, placing_numerical, bfsp) for links and surprises; betfair_live_marks from 30 Sep 2026, 300 s polls, 60 s in the last hour (betfair_recorder.py:634-660); about 40 days. Stream from 12 Nov 2026, none held yet.

**Known at 06:00?** With changes. Not a 06:00 feature: the trade is after B's result, outside the 05:15-10:00 window where the measured edge lies (L:live-by-hour-1104), so it is a separate strategy. s_B needs B's bfsp and placing - known minutes after B's off; require the 20-minute gap and parsed off times as inday_features does. Fit theta on earlier days only. 300 s polls can mis-time the +5 minute read by up to a poll.

**Closest precedent.** Collateral form and form lines as a listed cause of price moves (betting guide) (https://www.beforeyoubet.com.au/horse-racing-market-movers-explained-how-betting-markets-really-work); Interim odds paths and last-minute information in racing pools (JRA, 2004-2023) (https://arxiv.org/abs/2509.14645v1); Review: racing markets absorb most public information beyond odds (https://eprints.soton.ac.uk/51340)

**What is new.** The scheduled-event framing: links listed at 06:00, each earlier rival's result as a dated shock, and an exchange event study of the later runner's reaction. Franking and collateral form are standard handicapping; no measured same-day spillover was found.

**Already here.** model/inday_features.py:1-40 (id_: same-day draw, pace, trainer, jockey residuals; dev +20.2%, holdout -18.7%, rejected, L:confirm-holdout) - no runner-to-runner links, target was the result at BSP. form_uplift (fu_, rivals' later form, -0.0046 on price, no outcome information, L:iter62) is day-level. The weight exp(-lengths/3) is form_lines_close (model/blocks/form_lines_close.py:6,43; L:iter65 -0.0012). Differs: within the day, traded on reaction speed.

**First test.** Read-only query on betfair_live_marks from 30 Sep 2026: A's log price from the last mark before B's off to +5, +15 min and bfsp on the predicted move, day-clustered. Arms: links; placebo links (B from an unrelated race, same time). Pass: bfsp-minus-(+5 min) slope positive, 90% interval above zero, placebo null, paper CLV after 2% above zero. Else close.

**Most likely to fail because.** One rival's single-run surprise moves fair value by less than the spread plus 3% in almost every case, and the afternoon book near the off is the deepest and fastest of the day, so the +5 minute price already holds the move.

### 4.10 Sparse functional principal components of market-rating careers
*model form - target: closing price - novelty: partly known - data: partly held - effort: days - id `hidden-10`*

**The idea.** Learn the few shapes a career's market rating takes on a maturity clock and predict each horse's next reading from its scores on them, replacing the no-change forecast.

**Why it could work.** The local-level filter and the tissue both forecast no change. Careers follow few shapes - early and flat, late-maturing, second-season jump - and the score on them gives the direction of the next reading. Those who see the horse at home know it has strengthened; the close prices it.

**Definition (as corrected by the checker).** Stage 1: mx_gain_cell = decayed mean of (next mr reading less prior mr_rating) by code x age in years x readings so far x days-since-run band, earlier days only, shrunk k = 50; mx_pred = mr_rating + mx_gain_cell. Stage 2 only if stage 1 beats no-change: clock c = days since first reading (both codes; no foaling date at card time); residual readings less the cell curve; covariance by local-linear smoothing of within-horse pairs before 2021-01-01; 3 eigenfunctions; PACE scores with reading variance R from market_rating. Features: fp_pred - mr_rating, fp_slope, fp_score1-3, fp_sd. No yard warp in the first pass.

**Data.** bfsp per run from 2018 -> mr_ readings (model/blocks/market_rating.py READS); age in years on race_results and the card. Foaling date only in blandford_results from 2021 and cut from the card (daily_predictions.py:580,654), so a Flat clock in days is not available at 06:00.

**Known at 06:00?** With changes. Mean curve, covariance surface, eigenfunctions and yard warps must be frozen before FIT_UNTIL (market_rating.py:58, 2021-01-01); pooling within-horse pairs over the whole history leaks later careers. Careers are selected (poor horses stop), so the late part of the mean curve is biased upward - the informative-missingness problem the sports FPCA literature names.

**Closest precedent.** Functional data analysis of athlete aging curves with PACE for sparse careers (https://arxiv.org/pdf/1403.7548); Sparse FPCA by conditional expectation (PACE), the method named (https://www.math.pku.edu.cn/teachers/yaof/papers/2005-jasa.pdf); FPCA individual aging curves for ice-hockey careers, 2025 (https://ideas.repec.org/a/bpj/jqsprt/v21y2025i3p177-189n1002.html)

**What is new.** Fitting the career curve to a price-derived rating and scoring it as a one-step forecast of the market's next reading. Sparse FPCA of athlete careers, individual trajectory models of racehorse performance and the weight-for-age cohort curve are all established.

**Already here.** model/blocks/exposure.py ex_gain_cell (expected next-run gain by runs x age x code, on the performance figure; all3 iter42 -0.0045); market_rating.py:18-20,54 (zero-mean drift, faster over the first 4 readings), mr_trend, mr_surprise; custom_metrics.py:1349 form-trajectory slope; reports/model_review_1111.md:88 lead 4. Differs: a non-zero expected change in the market rating by maturity clock, with per-horse curve scores.

**First test.** Read-only query: one-step squared error of the next mr reading, mx_pred and fp_pred against the filter's prior, races after 2021-01-01, horse-bootstrap interval, split by readings held (1, 2-3, 4+); pass = interval below zero. Then model/blocks/market_curve.py (AFTER market_rating); arms base+mr, +mx, +fp, placebo on 11 folds; pass beyond 0.0007.

**Most likely to fail because.** With 2-6 readings a horse the scores shrink to the cohort mean plus mr_trend, and the trees already learn the cohort drift from age, runs, days off and ex_gain_cell.

### 4.11 Owner crowd size: racing-club members as late money
*feature - target: closing price - novelty: partly known - data: partly held - effort: days - id `market-11`*

**The idea.** Classify each owner as club, syndicate, partnership or individual and estimate the head-count behind the horse, as a measure of small late bets that arrive regardless of its chance.

**Why it could work.** A club with thousands of members produces thousands of small bets on its runner, mostly on the day and late. A sole gambling owner produces early, informed money. Trainer-keyed features blur the two; ownership structure separates crowd money from stable money and says in which window the move comes.

**Definition (as corrected by the checker).** owner_prev = blandford_results.owner at the horse's previous run (lagged join on horse and date). oc_type by regex on owner_prev (club; syndicate/partnership/3+ names; individual or company). oc_size = ln(1 + distinct horses run under owner_prev in the prior 365 days). Per-owner shrunk means (k = 20), earlier days only: early move ln(bfsp/morningwap) from betfair_prices 2021 on; late move ln(bsp/p_t60) from BASIC 2024 on. Plus oc_type x Saturday. No live card change until the test passes; then keep the card's owner.

**Data.** blandford_results.owner from 2021-01, bulk-loaded 23 Sep 2026, revisions unknown (blandford_sync.py:55-71; L:blandford-revisions). Card owner present in the HRB tooltip but cut; Betfair RUNNER_METADATA is requested yet MARKET_FIELDS stores no owner (betfair_recorder.py:63-68). BASIC T-60 prices 2024 on.

**Known at 06:00?** With changes. No owner is stored at card time, so a backtest must use the horse's owner at its previous run (lagged blandford_results), never the race-day row: same-day blandford fields are banned and the 2021-2026 owner values were loaded in one batch. Ownership changes are then seen one run late. The per-owner move means must exclude the race itself and same-day races.

**Closest precedent.** Owner-level tables of runners that shortened versus drifted, UK Flat 2019-2024 (practitioner study) (https://www.geegeez.co.uk/?p=2899113); Owner profiles by market position in the same series (https://www.geegeez.co.uk/tag/marc-chan/); Racing club versus syndicate structure as defined by the owners' body (basis for the type parse) (https://www.roa.co.uk/utilities/guide-to-ownership.html)

**What is new.** Ownership structure and scale as a head-count proxy, and the split of each owner's move into early and late. Per-owner shortener statistics are already practised; no academic study of ownership type against prices was found.

**Already here.** Owner is read by no feature: the only code reference is the feed mapping blandford_sync.py:67; the card's owner is stripped at daily_predictions.py:654; owners absent from the HRB export (reports/hrb_system_builder_review.md:194-198). Nearest: yard steam rate (model/market_features.py), stable block st_ +0.0018 (L:iter47), the steamer model's trainer and jockey steam (reports/strategy_research_2026-11-04.md:124-143).

**First test.** Read-only query, ledger-registered: 2021-2026 betfair_prices joined to lagged owners; mean early and late move by oc_type with trainer fixed effects, horse-clustered. Pass: a type effect whose 90% interval excludes zero in both halves of the years. Then model/blocks/owner_crowd.py through the fast loop against 0.0007 on 11 folds.

**Most likely to fail because.** Most owners have too few runs for a shrunk mean, so the feature collapses to a club/syndicate flag that trainer-keyed steam and bookings features already proxy, and member money goes to bookmakers, who open such horses short before 06:00.

### 4.12 Market-implied going gap traded as a meeting basket
*betting form - target: closing price - novelty: partly known - data: partly held - effort: a week or more - id `positions_data-00`*

**The idea.** Invert the morning price cross-section for the going the market is assuming, compare it with a weather-model going forecast, and back the going-sensitive runners on the side of the gap.

**Why it could work.** The 06:00 book is thin and anchored on the card's going; 27% of runners race on a going different from the morning card (L:card-vs-results-1110). Weather models already hold the rain to come; clerks update later and afternoon money re-prices going-dependent horses together. The price cross-section shows how much the market already holds; only the gap is traded, so a forecast everyone has read shows as D near zero and earns nothing.

**Definition (as corrected by the checker).** b_i = walk-forward shrunk slope (to stallion and dam_stallion, k fitted) of past ln(bfsp share) minus ln(mr_ market-rating share) on race_results.going coded by the custom_metrics going_map, earlier days only. g_m = per-meeting slope, race fixed effects, of ln(first 05:15-06:00 live-mark share) minus ln(mr_ share, which holds no going for today) on b_i. f_m = going step from a weather run issued before 06:00 minus the 06:00 card going. Back runners with b_i x (f_m - g_m) > 0 only when |f_m - g_m| exceeds its walk-forward sd; lay at SP. History proxy: betfair_prices.morningwap, flagged as untimed.

**Data.** Held: race_results.going, bfsp; betfair_prices.morningwap (no timestamp, betfair_prices.py:11,360); live marks from 30 Sep 2026; morning card going only from 14 Jul 2026. Not held: any course weather (repo_data map s.3). Forecast-archive API is free for non-commercial use only - unverified this session.

**Known at 06:00?** With changes. Leak in the first test: the backtest forecast reads the result-row going (reports/preregistration_clv_oos.md:48), so (morning share - forecast share) on b_i correlates with the realised going step mechanically. Use a reference with no today's-going input (mr_ market-rating share, or the forecast with going features dropped). morningwap has no timestamp and can include trades after the clerk's morning update; history holds no 06:00 card going before 14 Jul 2026. Weather must be the run issued before 06:00, not the stitched archive.

**Closest precedent.** Press: layers and punters re-price soft-ground horses on forecast rain before the going changes (https://www.racingpost.com/news/news/harry-on-the-drift-for-sprint-cup-as-layers-factor-rain-into-the-market/299486); Tipster practice: back mudlarks when rain is forecast (https://spectator.com/article/back-two-mudlarks-in-the-big-weekend-handicaps/); Soil-water-balance model of going (Cranfield EngD thesis, 2006): found poor predictions of going (https://dspace.lib.cranfield.ac.uk/server/api/core/bitstreams/7d4dc0a7-d9c3-4f57-8ae6-c1274c698d4a/content)

**What is new.** Only the inversion: a meeting-level slope of morning prices on going sensitivity read as the going the market assumes, and trading forecast minus implied as a basket. Backing going-suited horses ahead of forecast rain, and weather-to-going models, are practised or published.

**Already here.** No weather or implied-going code (grep: open-meteo, soil, implied going = 0 hits in model/, research/, scripts/). Parts exist: preferred_going / going_from_preferred (model/custom_metrics.py:1298-1308), going_change_lr (:1313), sire_going_apt (retired, L:iter52 -0.0006), cond_form going (L:iter57 -0.0002). Open lead 10, unbuilt (reports/model_review_1111.md:92). A forecast that knew result-row going made +14.05% / +10.43% CLV in races after a going change (reports/preregistration_clv_oos.md:62).

**First test.** Read-only query (research/queries/), ledger line first. 2018-2026, no weather: arm 1 g_m against the realised going step; arm 2 (step - g_m) x b_i on ln(morningwap/bsp) beyond the closing model; arm 3 placebo, b_i shuffled within race. Pass: walk-forward t > 3 in arms 1 and 2, placebo null. Only then build the weather feed.

**Most likely to fail because.** Horse-level going sensitivity is too noisy to carry a meeting slope (going-conditioned form moved the price error by -0.0002, L:iter57; sire going aptitude -0.0006, L:iter52), so g_m is mostly noise, and clerks' watering breaks the weather-to-going link on the Flat.

### 4.13 Temporal graph memory network trained on closing books
*model form - target: closing price - novelty: partly known - data: held - effort: a week or more - id `method-11`*

**The idea.** Give every horse, trainer, jockey and sire a learned memory that is updated after each race by its own run and its rivals' memories, trained to predict closing prices.

**Why it could work.** Hand features adjust for opposition once and in one dimension. A memory is multi-dimensional and is revised whenever a past rival runs again, through that rival's memory at their next meeting, and it is trained against the market's own price. It can find aggregations nobody thought to build.

**Definition (as corrected by the checker).** Stage 0: a permutation-invariant set network (as greyhound/nn.py) on the horses' cached matrix, target BSP shares, stacked with the served blend + mr_. Stage 1 only if stage 0 gains: 32-d memories for horse, trainer, jockey, sire, read before and GRU-updated after each race day (never between same-day races) from place, total_dst_bt, bfsp, ipmin/ipmax, comment class and rivals' pooled memories; heads: BSP shares, Plackett-Luce depth 3, log(morningwap/bsp) from 2018. Features prequential only: weights frozen at each fold start and fitted on earlier days, PCA fitted on training rows. Score the memory's increment over stage 0, not over trees alone.

**Data.** race_results 1,938,313 rows 2010-2026 incl. bfsp, place, total_dst_bt, comment (RESEARCH_FRAMEWORK.md:330; scraper.py:121-193); betfair_prices morningwap, ipmin, ipmax from 31 Dec 2017 (betfair_prices.py:360; ledger open-close-inventory-1101). No torch or GPU job in the horse path (torch only in greyhound/nn.py).

**Known at 06:00?** With changes. Stacking leak: pretraining/multi-epoch weights have seen the prices of the boosted members' training rows, so train-row embeddings are in-sample and test-row ones are not. Fix: prequential outputs only - read memory before update, weights frozen per fold and fitted on earlier days, PCA on train rows. No memory update between same-day races. Yesterday's BSP/in-play fields arrive with a day's lag.

**Closest precedent.** Temporal graph networks: per-node memory updated by events with a GRU and neighbour messages - the architecture proposed (https://arxiv.org/abs/2006.10637); JODIE: coupled recurrent updates of two entity embeddings after each interaction (Kumar, Zhang, Leskovec (2019) Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks, KDD 2019); Horse-graph features (static, who-beat-whom) fed to a neural net (https://ir.library.louisville.edu/etd/2953)

**What is new.** Only the application: a published event-memory architecture trained against exchange closing-price shares for horses, trainers, jockeys and sires. No method step is new.

**Already here.** model/blocks/market_rating.py:28-38 is the 1-D case: a Kalman memory per horse, race level from rivals, fitted to BSP shares (iter116 -0.0067). elo retired (iter55 +0.0001). greyhound/nn.py: DeepSets on BSP shares, alone +0.0015, stacked -0.0113 (reports/greyhound_model.md:1526-1527). Horse race-level network parked (reports/bfsp_model_plan.md:261-263). Differs: learned 32-d memories, four entity types, GRU update.

**First test.** A research module (CPU torch, outside model/) on the cached matrix, ledger entry first. Arms on 11 folds: served + mr_; + set network stacked (weights from early-stop rows); + memory components as a block. Pass: the memory arm below the set-network arm by 0.0007 with its interval excluding zero; one read.

**Most likely to fail because.** Sparse horse careers leave the memories holding what mr_ and the windowed features already hold, so any gain is generic neural-member diversity that a plain set network on the cached matrix gives at a fraction of the cost.

### 4.14 Roy-model inversion: private ability implied by the race chosen
*model form - target: both - novelty: partly known - data: partly held - effort: days - id `agents-03`*

**The idea.** Model each trainer's choice of race among the races the horse could have run in, then solve for the ability level that makes the chosen race the rational pick.

**Why it could work.** A trainer knows the horse's home work and picks the race maximising prize times chance less cost. Passing over easier or nearer races for a stronger one reveals a private estimate above the public one; the reverse reveals doubt. The card shows the race run, never the races declined, so the early price cannot use the comparison; stable money later does.

**Definition (as corrected by the checker).** For each horse-run on day d: O = races on days d-6..d-1 (results) and day d (cards) of the same code, age and sex terms met, dist_furlongs within +-20% of its last three trips, official_rating inside that race's runner min-max (handicaps) or matching maiden/novice status, track within the yard's 90th-percentile km. Par level L = trailing three-year mean mr_field of the cell (track x class x race_type), earlier days only. Features: rc_n_alt, rc_par_rank, rc_easier_passed, rc_richer_passed (prize), rc_par_gap (chosen L less O's mean). No forward window. O of size 1: NaN. Structural step only if these pass.

**Data.** Race calendar and conditions in race_results; yard bases in model/blocks/travel.py; mr_ block. Not held: entries, rating-band limits, forward programme history (repo_data map section 3); class blank on 27% of cards (reports/model_review_1111.md:43).

**Known at 06:00?** With changes. The forward window in history leaks: a future race's band 'read from its runner OR range' uses runners unknown on day d. Keep to days d-6..d-1 from results plus today's other cards. Par level must be a trailing mean on earlier days. Abandoned meetings are missing from O.

**Closest precedent.** Players self-select into tournaments on unobserved strength shocks, so entry reveals strength beyond the public rating (chess) (https://researchportal.ip-paris.fr/en/publications/self-selection-in-tournaments-the-case-of-chess-players/); Theory of self-selection into contests, set against the Roy model (https://faculty.haas.berkeley.edu/rjmorgan/Self-selection.pdf); UK racing: trainers' private information inferred from betting-market data (https://ueaeprints.uea.ac.uk/id/eprint/45782/1/i0038_4038_79_2_Brown.pdf)

**What is new.** An explicit set of races the horse could have run in, and features from the options declined (and later an inversion to ability on the market-rating scale). Self-selection on private strength and class-move intent are both known.

**Already here.** No opportunity set of declined races. Related and served or built: class_change, is_class_drop (model/custom_metrics.py:1187-1221), it_class_drop (model/intent_features.py:173), sos_vs_today (model/primitives.py:792), tv_ travel against the yard's habit (iter96 -0.0012), schedule-expected strike rate (model/connections.py:571), mr_ race level. Ledger intent-survey: class drop A/E 1.012 (t +1.16), the price takes it.

**First test.** Drop-in block model/blocks/race_choice.py (rc_, reduced form only) on the cached matrix, 2018-2026, beside the served set and the market rating, with a placebo arm on 11 folds. Pass: paired change below -0.0007 with the interval below zero and rc_ not explained by tv_ or sos_vs_today; ledger entry first. Structural inversion only after a pass.

**Most likely to fail because.** The opportunity set built from results is too noisy (eligibility inferred, ground, owner and jockey constraints unseen), and the chosen race's level against the horse's rating is already carried by mr_ and sos_vs_today.

### 4.15 Franking by absence - rivals who never came back
*feature - target: both - novelty: partly known - data: held - effort: days - id `physical-01`*

**The idea.** Score a past race, and the whole meeting, by how many of its runners have failed to reappear on schedule: an excess of absentees marks a race that damaged horses and a form line biased upward.

**Why it could work.** Form is franked only by rivals who run again - survivors. A race on jarring ground or at a punishing pace leaves victims who simply vanish from the record, so its form line reads better than it was and today's runner out of it carries hidden damage risk. Yards and vets know; the morning price reads the visible form.

**Definition (as corrected by the checker).** From race_results (race_date, horse, trainer, track, race_type, age, class, placing_numerical): on fully observed earlier rows fit a discrete-time hazard of the next run within D days by code x age band x month x class x finishing third, trainer shrunk (k = 30). For today's horse, last race r, D days ago: H_r = sum_j(S_j(D) - R_j) / sqrt(sum_j S_j(1 - S_j)) over rivals j, R_j = ran again by yesterday; H_m the same over the course-day's other races. H = 0 when sum S_j < 3. Features: H_r, H_m, decayed mean of last 3. First version drops the firmness and 1/S_j terms.

**Data.** race_results race_date, horse, trainer, track, race_type, age, placing_numerical; complete from 2010 (L:career-truncation). Not held: runs abroad, point-to-points, entries/non-runner history (repo_data map section 3), so sold, exported and retired horses read as absent.

**Known at 06:00?** Yes. none found if the hazard model is fitted only on rows old enough to be fully observed and R_j stops at yesterday. Bias, not leak: model/freshness_features.py:25 says the history has missing months - false absences.

**Closest precedent.** Hot-race tools list each rival's later runs; those not seen since are shown as blanks, not scored (https://www.geegeez.co.uk/tag/hot-form/); Form-strength guides: wait for rivals to run again before judging a race (https://thehorsesmouth.co.uk/blog/horse-racing-form-strength-part-2); Discrete-time and Cox survival models of racehorse careers and fatality (time in days or races) (https://madbarn.com/research/a-comparison-of-survival-models-for-assessing-risk-of-racehorse-fatality)

**What is new.** Scoring a past race and meeting by the excess of rivals who failed to reappear against a fitted return-time hazard, and using that z-score as a form and price signal. Both halves exist separately (survival models in veterinary work; rivals' later runs in form tools); the join was not found.

**Already here.** No block or query: 'reappear' only in a comment at model/stage_f.py:131. Nearest: form_lines (fl_, served, -0.0038 L:iter59) and future_form (ff_, -0.0003 L:iter90) read rivals who DID run again; hx_lr_nonfin (last race's non-completers) retired +0.0001 (L:iter109). Open lead 3 'franked form' (reports/model_review_1111.md:85) is rivals' rating change, not absences.

**First test.** Ledger entry first. Read-only query before any block: H_r of the last race against next-run win vs BSP chance and ln(morningwap/bsp), with class, age, month and count of rivals run since as controls; pass walk-forward t > 3, same sign in both halves. Only then model/blocks/absence.py against the served base on 11 folds with a placebo (bar 0.0007, L:iter114).

**Most likely to fail because.** Excess absence mostly marks low-grade, end-of-season and older-horse races (retirement, sales, export) rather than damage, and today's runner has by definition come back, so H collapses into class and season already in the model.

### 4.16 Crosswind echelon - wind-vector forecast of the day's draw-side and cover bias
*feature - target: both - novelty: partly known - data: free public - effort: days - id `physical-03`*

**The idea.** On straight courses, project the forecast wind onto the straight: the leeward flank of the field is sheltered by the windward flank, and a headwind raises the value of cover for hold-up horses.

**Why it could work.** Drafting is measured in racing (Spence et al. 2012, https://pmc.ncbi.nlm.nih.gov/articles/PMC3391435). In a crosswind cover exists on one flank only, as in a cycling echelon; in a headwind leaders pay more. The served draw bias is a course average; wind makes the bias of the day. The market learns it after the first straight races; the forecast exists at 06:00.

**Definition (as corrected by the checker).** Straight-course races (model/abm/track.py TRACK_GEOMETRY), fields of 8+. Wind at the off hour from the nearest station (speed, direction); straight bearing measured once per course. h = headwind component, x = crosswind component signed toward high stalls; both 0 under 3 m/s. Step 1, hindsight with observed wind: race-centred pounds beaten on x x (stall/runners - 0.5) and h x run-style score, with course x trip x stall_positioning effects. Step 2 only if step 1 passes: the same terms from the forecast as issued before 06:00. No gust ratio or round-course term in the first version.

**Data.** Held: stall, runners, stall_positioning, run style. Wind: hourly direction and speed from NOAA ISD already coded for greyhound tracks (sources/noaa_isd.py:14-16), horse courses not fetched. Straight bearings not held - one-off measurement. As-issued wind forecasts: no source held.

**Known at 06:00?** With changes. Observed wind at the off hour is not known at 06:00: use it only for the hindsight step; the served feature needs the forecast as issued before 06:00. Whether the card carries stall_positioning at 06:00 is unverified.

**Closest precedent.** Twelve years of chart annotation: no correlation between wind direction and running styles; wind distorts fractions (https://www.drf.com/news/answers-are-blowing-wind); Clerk blames freak wind for a far-side bias on a straight course (https://racingpost.com/news/britain/freak-wind-blamed-for-ebor-festival-track-bias-and-drop-in-attendances-auyZF7s1VhHO); Forum practice: hold-up horses marked up in a strong headwind; drag allowance for front-runners (https://theracingforum.co.uk/forums/topic/should-front-runners-be-marked-up-formwise-to-allow-for-drag/)

**What is new.** Only the signed crosswind x stall term as a forecast of the day's draw-side bias: no study found. Headwind x run style is practised lore and the one long record found (US) reports no effect.

**Already here.** No wind in the horse path (grep crosswind/headwind: only sources/noaa_isd.py, greyhound tracks). Relatives all failed: shape/draw rebuild -0.0002 and -0.0000 (L:iter18, L:iter25), draw_v2 -0.0004/-0.0009 (L:iter35, L:iter54), pace_v2 +0.0009 (L:iter35); draw 'small on average and priced', projected leaders A/E 0.91-1.08 (L:pace-draw-bias). Geometry holds straight lengths only, no bearings (model/abm/track.py:37-40).

**First test.** Ledger entry first. Read-only query, no block: fetch station wind for straight-course meetings with the existing sources/noaa_isd.py code, then step 1 regression. Pass: pooled t > 3 and same sign at 70% of courses with 200+ races. Fail ends the idea. Step 2 on betfair_live_marks for races after the first on the card.

**Most likely to fail because.** Horse fields on a straight split into groups by ground, not into echelons, the drafting effect is small, coarse wind data misses local shelter, and every draw and pace rebuild here has read zero against the market.

### 4.17 Reason-split closing combine: TreeSHAP families inside the market blend
*model form - target: closing price - novelty: partly known - data: held - effort: days - id `method-00`*

**The idea.** Split our forecast's log-odds into a dozen feature-family contributions and let the closing model give each its own weight beside the morning price, learning which of our reasons the close confirms.

**Why it could work.** The morning price already holds some of our reasons (past prices, class) and not others: horses back in 1-4 days are on average 9.4% longer in the morning than at BSP (reports/strategy_research_2026-11-04.md:28-41). One coefficient on ln model averages reasons the market has with reasons it lacks. Splitting by reason lets the blend lean on the unpriced ones and drop the rest.

**Definition (as corrected by the checker).** Main member only. LightGBM pred_contrib on the walk-forward (out-of-fold) booster; each feature mapped by a fixed prefix table to at most 6 families, rr_/rw_ features to the family of their source input; family sums centred within raceid: c_if. Closing logit = the race_book.FEATURES design (lq_market, lq_model, vol and depth terms) + sum_f d_f*c_if, ridge on d_f (d = 0 is the served model), cross-entropy to the bsp book from betfair_prices.morningwap, monthly walk-forward. Report d_f across booster refits and seeds first; a family whose d_f changes sign between refits is dropped.

**Data.** betfair_prices.morningwap, bsp, morning_vol 2018 on (betfair_prices.py:355-365); walk-forward forecasts Jan-Mar 2026 (L:race-book-0930, 2,504 races). Per-fold boosters are not known to be saved, so pred_contrib needs refits.

**Known at 06:00?** With changes. No look-ahead if contributions are out-of-fold. Two timing problems: b_f is fitted on one booster vintage's attributions and applied to another's (attribution among correlated families shifts at each retrain: model/interpret.py:4); rr_/rw_ features are z-scores of inputs from several families, so the family map must follow the source input. Fix: fixed prefix table, refit b_f with every retrain.

**Closest precedent.** One-step conditional logit: every fundamental variable gets its own coefficient beside the odds; compared with two-step and found worse (https://ideas.repec.org/a/buc/jpredm/v1y2007i1p43-59.html); SHAP attributions of first-stage models used as second-stage (meta-learner) inputs (https://arxiv.org/abs/2507.17650); Benter's second stage: one coefficient on log model beside log public (https://www.Gwern.net/doc/statistics/decision/1994-benter.pdf)

**What is new.** Only the middle ground: grouping attributions into families, shrunk to a common coefficient, between the published one-step and two-step combines, and fitted to the closing book rather than the result.

**Already here.** model/race_book.py:52,110-114: one coefficient on lq_model. L:closing-model-v2-0930: five market-side additions, nearness 0.3020-0.3025 vs served 0.3021, not adopted. L:move-model-1021: ridge and LightGBM on raw features beside the price, CLV correlation 0.346/0.359 vs served 0.295, but served best at equal volume. morning/ steamer model fits the move from features directly. Differs: model side split by TreeSHAP family inside the logit. No pred_contrib anywhere in model/.

**First test.** A research/queries module, no change under model/. Ledger entry first. Frame of closing-model-v2-0930 (Jan-Mar 2026). Arms: served closing model; + family terms; + placebo (family sums shuffled within race); + one-step control (top raw features). Pass: nearness below 0.3021 / 0.3131 in both scored months, 90% race-bootstrap interval excluding zero, rule CLV at 3% not lower, placebo flat.

**Most likely to fail because.** Family attributions are unstable across refits and correlated families, and two scored months cannot resolve 6-12 extra coefficients when every earlier closing-model variant moved nearness by 0.0005 or less (L:closing-model-v2-0930).

### 4.18 Kinship prior for the market-rating filter (pedigree BLUP on past BSPs)
*model form - target: closing price - novelty: partly known - data: held - effort: days - id `hidden-00`*

**The idea.** Start and steer each horse's market rating from its relatives' market ratings, weighted by relationship and precision, so debutants and lightly raced horses get a pedigree-informed state that updates when a relative runs.

**Why it could work.** Price error is largest where own form is thin (0.50-0.57 vs 0.34-0.37 in handicaps: model/blocks/dam_line.py docstring). A half-brother's strong run last week is public but diffuse; breeders, agents and the yard act on family level later in the morning. The current filter starts a debutant from its rivals, ignoring family.

**Definition (as corrected by the checker).** Separate block mrk_ (mr_ untouched). For each runner with mr_n <= 3: prior = precision-weighted mean, over sire's progeny (stallion), dam's other foals (dam + dam_stallion) and damsire's grandchildren, of their first-three-run mr readings y = log p + C less the cohort mean (race_code pool, age, sex), days before today only, decayed 365 d; weights 0.25 paternal half-sib, 0.25-0.5 maternal, shrunk by a heritability fixed on races before FIT_UNTIL. Features: mrk_prior, mrk_sd, mrk_post (prior combined with own mr_rating by precision), mrk_z in race, mrk_news30. NaN with no rated relative.

**Data.** race_results stallion, dam, dam_stallion (scraper.py:121-193); mr_ readings from bfsp history 2018 on (model/blocks/market_rating.py:20-22). Dam's own record almost always empty (dam_line.py NOT_SERVED note), so dam nodes rarely rated.

**Known at 06:00?** With changes. Relatives running the same day must be excluded (ratings before day t only). Relatives' filtered ratings are of seasoned runners - biased for a debutant; use their debut/early readings. Do not alter mr_ itself (passed holdout, shadow from 12 Nov): changing INIT changes every rival level C; build a separate block.

**Closest precedent.** Animal-model BLUP with relationship-matrix prior on racing traits (rank, earnings) - standard in breeding evaluation (https://link.springer.com/content/pdf/10.1186/1297-9686-23-2-159.pdf ; https://aab.copernicus.org/articles/47/505/2004/aab-47-505-2004.pdf); Pedigree information content against the UK bookmaker market for two-year-old races (conference talk, results not seen) (https://digitalscholarship.unlv.edu/gaming_institute/2016/June8/12); Unraced two-year-olds priced differently by bookmakers (Peirson and Smith) (https://www.kent.ac.uk/economics/repec/0819.pdf)

**What is new.** Only the phenotype (a cross-class market rating from past BSPs) and its use as a state prior; the estimator is textbook BLUP and the information (how the market prices relatives) was already tested here as pdm_.

**Already here.** model/blocks/pedigree_market.py (pdm_: market's view of sire's debutants, damsire's, dam's other foals, shrunk) - L:iter84 -0.0006, retired. dam_line L:iter49 +0.0008, iter52 -0.0001. Pedigree 'not proposed again' (reports/model_review_1111.md:94). Differs: relatives' cross-class mr_ readings instead of race-normalised price, joint precision weights, used as the filter's first value (market_rating.py:22,53), news term. No BLUP/kinship code (grep: 0 hits).

**First test.** Drop-in block model/blocks/kin_rating.py; arms on the 11-fold quick screen: base+mr_, +pdm_, +mrk_, placebo. Pass: mrk_ beats both base+mr_ and the pdm_ arm by more than 0.0007 overall or on runners with 0-3 prior runs with interval excluding zero, Brier/concordance not worse; ledger entry first.

**Most likely to fail because.** With a two-generation neighbourhood the BLUP collapses to a precision-weighted sire-progeny and dam-progeny mean, which pdm_ and the served debut_market already approximate, so the gain stays inside the 0.0007 bar as four earlier pedigree readings did.

## 5. The other survivors

| Idea | What is new | Most likely to fail because | Id |
|---|---|---|---|
| Stamina limit as an interval-censored dose-response | A per-horse stamina limit estimated as a probit threshold from pass/fail runs with a sire and damsire prior. Stamina indices, distance aptitude and threshold models are each known; fitting the threshold per horse from comments and in-pla... | 'Weakened' in a comment does not separate failing to stay from being outclassed or going too fast, trainers rarely run horses beyond their limit so failures are scarce, and pedigree stamina is the... | `physical-04` |
| Low-sun obstacle forecast from solar geometry and course bearings | Forecasting omissions and glare per straight from ephemeris and course bearings, crossed with each horse's jumping record. The phenomenon, its seasonality and the sun-against-bearing method are known; no racing model or omission dataset... | Too few affected races and no omission label: a press-quoted figure of 28 meetings in a season (search result, page not opened) implies a few percent of jump races, and the per-horse effect of a re... | `physical-10` |
| Campaign-state hidden Markov model of a horse's runs | A fitted HMM whose emissions are the connections' card-known choices, with trainer-specific shrunk transitions, forward-filtered at card time. Form cycles, campaign-stage statistics and latent-form HMMs each exist separately. | The filtered state probabilities are a low-dimensional compression of choices and market surprises the boosters already read with windows, so they add nothing in the fit (as kr_ and elo did), and p... | `agents-09` |
| Kin-pooled within-horse response slopes, priced versus realised | Only the gap between relatives' within-horse responses as priced and as run (the two-currency difference), applied to first tries. Kin-pooled distance response itself is the published random-regression animal model; the idea's own novelt... | Slopes from two to six runs across trip bands are too noisy and still placement-selected, so the kin mean collapses to the cohort slope - and pedigree has screened flat four times here (iter49, ite... | `hidden-11` |
| Owner fingerprint: betting owners and crowd-owned horses | An owner-level morning-to-BSP move rate shrunk inside the trainer's, and owner types derived from the name with a signed test (informed vs loyal money). Owner records to SP/BSP and 'gambling owner' lore are practised. | Most owners have too few runs, so own_move shrinks to the trainer's steam rate (already built, never served), and the syndicate A/E test meets the same BSP efficiency that removed every other entit... | `agents-07` |
| Survive-then-contest likelihood for jump races | Only the joint fit in horse racing: an exit hazard with a shared race frailty feeding a contest among survivors. Fall-risk models and non-finisher handling in ranking likelihoods are both published; no combined jumps model was found. | With exits near-independent of ability the mixture collapses to completion chance times strength, renormalised - which trees given the fall and pull-up rates already approximate, and those rates ju... | `physical-07` |
| Hoof-traffic wear state of each strip of track | A computed, decaying wear state (runners x furlongs x wetness since the ground was last raced) used as a pre-racing predictor of the day's draw and pace bias. Wear, fresh ground and rail position as bias drivers are standard practice; no... | Every draw and pace reading here has been priced or lost out of sample, and without knowing which lane was raced on or today's rail position the wear number reduces to 'busy course, wet spell', whi... | `physical-08` |
| Effort-furlongs - an energy-equivalent trip for every course, going and code | Using an effort-equivalent trip as the axis for stamina features. The physics of incline and the habit of calling a course stiff are known, and course speed-stamina curves have been fitted; no published equivalent-trip table for racing w... | Because the coefficients are fitted to reproduce standard times, E is just the standard time in other units, and trip suitability has added about nothing to this forecast three times. | `physical-09` |
| Window-by-window population signatures scaled by expected crowd size | Only the combination rule: window-specific move models weighted by an ex-ante forecast of each window's share of turnover. Time-of-day bettor populations and session clienteles are published and are standard talk among exchange traders. | The predictable move is made before 10:00 (L:live-by-hour-1104) while the windows with history (T-60 on) are near-efficient (-0.69% and -0.58% CLV), and a boosted pooled model with course, class an... | `market-10` |
| Conditions as a decoder: private valuations read from auction and claiming weights | Keeping the weight-implied price band as a point-in-time attribute of the horse for later races, as a leak-free stand-in for a sales file. Reading weight as price inside the race itself is standard. | Too few runners carry a revealed band (auction races are a thin summer two-year-old programme, claimers are rare), and sex, age and win penalties blur the inversion, so the screen does not resolve. | `agents-02` |
| Debut-slot value from the yard's matured earlier cohorts | A shrunk yard x slot estimate whose target is the earlier newcomers' later market rating (not their debut price or strike rate), including order within the crop, and its gap to what the debut market paid for that slot. | Yard x slot cells are too thin outside the largest yards, so V shrinks to the yard's newcomer mean that dm_ and bk_tr_mkt already read, and the course/race tier of a big yard's debutant is priced b... | `agents-06` |
| Jockey's forgone book: opportunity cost of today's rides | Pricing the choice: the value gap between the ride taken and the best ride forgone from the jockey's supplier yards across all of the day's meetings, on the market-rating scale, plus the passed-over horse's mirrored value. The qualitativ... | Bookings are public from the 48-hour declarations, so the choice is in the price before 06:00 (practitioners call it overbet), and choices are driven by retainers, weight and logistics more than by... | `agents-08` |
| Horses-in-training sale catalogues: shop-window runs and resale valuations | The run-while-catalogued flag as an incentive variable and a resale price residual carried to the first runs for the new yard; sale price as a quality signal is published. | Sale entries and prices are public and reported, so the 06:00 market holds them as it held the pre-racing price (sales-served-screen-1005), and catalogued runs are a small share of runners (guess). | `agents-12` |
| Phase-resolved jockey skill matched to the horse's own deficit | A finishing effect per jockey conditioned on the in-play low and BSP, and each phase effect crossed with the horse's own deficit; per-phase jockey profiles by run style and from tracking are published. | Once horse and trainer are partialled out the phase effects are unstable (split-half under 0.3), because break, early position and leading mostly reflect the horses and instructions a jockey gets,... | `agents-13` |
| The handicapper as a sensor: reaction-function residual of the mark | Residualising the mark change against a fitted reaction function and reading the residual as the handicapper's private view; mark changes themselves, and their relation to next-run results, are published and practised. | The served model already holds or_change, lb beaten, position and class jointly, so the trees form the residual implicitly, and what is left of h is omitted public form (rivals' later runs, time) r... | `hidden-02` |
| Stale-mark news: two-vintage differencing keyed to the horse's last trade | Timing the vintage from the horse's own last BSP and giving the model's revision since then a separate weight beside the morning price; the decomposition is standard in nowcasting and each news source (franked form, yard form, mark chang... | The morning price is compiled afresh from current form, not anchored on the last BSP, so news_lp is collinear with ln model - ln morning price and adds nothing, as every earlier closing-model addit... | `hidden-03` |
| On-day/off-day mixture and the field-size variance premium | A per-horse on/off state identified partly from in-play prices, and the mixture-vs-softmax share ratio as a feature. Unequal-variance order models and variance's field-size effect are published; no two-state racing model was found. | In-play highs and lows mostly record run style and losing rather than an off day, so pi collapses to the horse's loss rate and vp to a function of spread and field size that the trees already see. | `hidden-09` |
| Reappearance-hazard residuals: own return surprise and rivals' excess absence | The return-time residual conditioned on what the horse did last time, used as a price feature, and rivals' excess non-reappearance as evidence about a race. No precedent found for either; hazard models of race gaps and layoff reading are... | The booster already holds days since the run, the usual gap, the last finish and beaten distance and fl_l1_n, so the residual re-expresses served inputs and lands inside the 0.0007 bar, with false... | `hidden-12` |
| One market-taught relevance kernel reused for every windowed feature | Estimating the relevance rates from how past closing prices transfer across days, trip, ground and class, and reusing that single kernel for every form measure. Similarity weighting itself is known (Benter's distance factor, hand-made 'r... | The boosters already combine day-decayed and run-windowed form with trip, going and days features, and the hard-cell version added nothing (L:iter57), so a smooth kernel is likely inside the 0.0007... | `method-08` |
| Stochastic frontier rating: ability as a frontier with one-sided shortfalls | The formal composed-error likelihood with Battese-Coelli determinants applied to a horse's run-by-run figures, giving a separate forecast of today's shortfall. The asymmetry itself and rating on best form with excuses are practised. | Form-figure ratings have repeatedly added nothing beside the served set (kalman, Elo, nfp_z, cond_form, comments), and best-of figures plus freshness and trip features already let the boosters appr... | `method-10` |
| Market-implied going shift read from the meeting's cross-section of moves | Estimating a meeting-level going shock from the cross-section of price moves on going loadings and carrying it, leave-one-race-out, to books that have not moved; no precedent found. | Per-horse going slopes from a handful of price readings are noise, and six weeks of thin morning books give the meeting regression no power, so gamma cannot be told from zero. | `market-04` |
| Odd-lot footprint: stake-size digits separate model money from human money | Applying the size-digit identity marker to a betting exchange's resting book, signed by side, as a predictor of the move to the BSP. The marker itself is published in equities with mixed evidence; no betting-exchange precedent was found. | On aggregated three-level totals the digits mostly record partial fills, currency conversion and this trader's own orders, and the book's microstructure has already shown little power on the move (... | `market-09` |
| Rival-avoidance index: field shortfall against eligible supply | Treating the residual of a field-size model as a runner-level signal of rivals' private views and testing it on the morning-to-BSP move. Expected-field models and the 'frightened off' remark are not new. | Shortfall variance is mostly ground, horse population and programme clashes, and where a horse really did deter rivals it is already short at 06:00 (entries are public), so S adds nothing to field... | `agents-00` |
| Cut-line pressure before oversubscribed handicaps | A signed pounds-to-expected-cut with a trainer aim weight, computed on ordinary prep runs and fed to a price model. The cut statistic, the 'needs pounds to get in' reading and plot-spotting are practised. | Too few runner-rows carry a non-zero value and the aim weight (which horse is aimed at which target) is mostly noise, so neither the cell test nor an 11-fold screen resolves; the big targets' plots... | `agents-04` |
| Market-implied volatility per horse from prices across field compositions | Heteroscedastic normal race models are published; the new step is inferring the variance from past closing prices across field compositions and feeding the share to a price forecast. | A per-horse variance is not identifiable from about ten noisy readings, so what survives is a global link correction by exposure and field size, which the trees already form from mr_lshare, mr_n, m... | `method-03` |
| Yard-base training interruption index from reanalysis weather | A dated, point-located weather anomaly at the yard as a model feature. The effect itself (frozen gallops hurting some yards) is racing folklore discussed in press and forums; no quantified version was found. | Shock events are few and regional, racing is abandoned in the same spells, and most large yards have all-weather gallops, so the within-race spread of the index is near zero and what remains is alr... | `positions_data-03` |
| Dead-weight coupling from each jockey's inferred riding floor | The floor itself is published and the dead-weight argument is folk knowledge; what was not found is any measured test of pounds of lead against pounds of rider on results or prices. The new step is that test, as a per-runner feature. | Within a jockey DW is just weight carried (the floor is constant), and across jockeys it marks rider type (apprentices, lightweights), so the coefficient is weight plus rider quality that bk_ and t... | `physical-13` |
| Habitual late drifter: a paddock trait inferred from last-minutes price history | A horse-level late-window drift trait net of yard and rider, and the split of today's late move into habit and surprise; no horse-level persistence study was found. | Split-half persistence of a horse's T-15-to-BSP share move is near zero once yard and rider are taken out, and the record already shows late moves and horse-level residuals carry nothing beyond the... | `market-05` |
| Desk drift and pool exchange rates for official marks | Exchange rates estimated for every pool pair from crossing horses and a change-point state on a pool's drift; the GB-IE differential and mark drift are public and measured by the regulator. | The crossing rate measures trainers choosing when to cross rather than the unit of the mark, the GB-IE gap has been headline news since 2021 and sits in the market rating and tv_raid, and monthly p... | `agents-14` |
| Conditional surprisal of the in-running comment given the bare result | Scoring a comment by its improbability given the finishing position (and a signed expected-minus-actual finish from learned phrase weights). Text features from comments and eyecatcher reading are known; the conditional-on-result weightin... | Comments are a small formulaic vocabulary the market already reads; fixed classes added nothing on the steadier screen (iter52) and learned weights mostly re-express what ir_ and the form windows c... | `hidden-05` |
| Pool exchange rates: priced versus realised offsets across form circuits | The priced-minus-realised offset per pool pair, estimated from crossers only and fed as a feature. Linking pools through shippers, separate marks for raiders and pool miscalibration in rating systems are all known; no study of the market... | xr_gap is an actual-minus-expected residual at pool-pair grain - the family iter11-ae found not to persist - and the trees already hold tv_raid beside the mr_ and kr_ ratings; the best-known case (... | `hidden-06` |
| Yard informedness from the price-performance lead-lag area | The split terms are lag-0 and lag-1 cross-covariances of increments, a standard lead-lag measure; the signature adds nothing. New: scoring a yard by whether its prices anticipate or follow its horses' form, and using it to weight the mor... | The ratio mostly reflects the yard's mix of unexposed and exposed horses and figure noise, yard effects against prices have not persisted here (L:iter11-ae), and two scored months cannot fix an int... | `method-04` |
| Oestrous phase clock recovered from a filly's own run residuals | Recovering an individual 21-day phase from the filly's own run residuals against the BSP, with male and winter placebos. The belief that oestrus hurts performance is old; no study or product tying cycle phase to race results was found. | One bad run is a near-useless anchor: oestrus lasts several days of each cycle, cycle length varies, many fillies in training are on suppressants (my knowledge, unverified here), so the in-window f... | `physical-06` |
| Coupled hidden yard-health state with a fade-signature emission | No precedent found for a filtered yard-health state used as a price feature, nor for a yard-level 'traded short in running then beaten far' emission, nor for between-yard contagion in a betting model. Each ingredient (yard form gauges, f... | The filtered state is a smoothed re-reading of the same yard results cw_ already holds and the 06:00 market already prices, and four states plus a contagion term are not identifiable from a yard's... | `physical-15` |
| Public-eye score: a preference model trained on the industry-SP and Tote wedge | Only the transfer: the wedge used as a training label for a card-surface preference model in racing, and its prediction used to forecast the exchange's morning-to-close move. The wedge-as-sentiment construct is published (football) and t... | The wedge is mostly overround allocation and odds rounding, not taste, and the taste part keyed by yard and rider already measured +0.0006 (L:iter98). | `market-07` |
| The race programme as a flow graph: prep and target roles of each race | Recurring race slots (not horses) as graph nodes, with a shrunk slot-to-slot lift measured on the market rating; the prep/target idea itself is standard practice. | Node features are constant within a race, stable routes exist only for a small set of big races the market already reads, and edge cells are too sparse after shrinkage - the seasonal and last-year... | `agents-11` |
| Error twins: an analog ensemble in the forecast's own leaf space | Selecting analogs by the sign and size of the model-market gap inside the forecast's leaf space and reading the morning-to-close move; the components (leaf proximity, analog correction, proximity trust score) are all published. | The twins' mean move collapses to the closing model's global weight on the gap, as the parametric gap terms and the LightGBM move model already found no local structure beyond it. | `method-14` |
| Promotion-driven uninformed lay flow from extra-place offers | Only the forecast of the arb-driven lay flow per runner, scaled by the number of books on extra-place terms. The bookmaker-exchange gap itself, and matched bettors laying on those races, are practised and published. | A best bookmaker price above the exchange lay more often marks an exchange steam the books have not yet followed, so the sign is ambiguous, and matched-bet stakes are small against the late WIN poo... | `positions_data-07` |
| Late-crowd salience from public attention data | Racing use, the morning-to-close move as target, and the interaction with a predicted recreational share of the race. Attention data against betting prices is published (tennis, skiing) with weak or null results. | The stable part of name recognition is already in jk_steam, tr_steam and bookings, the time-varying page-view spike is thin and sparse, and the two published tests of attention against betting pric... | `positions_data-10` |
| Card-order attention with the alphabetical natural experiment | Only the identification: alphabetical numbering inside equal-weight groups as an instrument that separates list position from ability, and reading it against the exchange move. The idea's claim that no study of card order in racing odds... | List-pickers bet with bookmakers and the Tote, not on the exchange, so any effect sits in the industry-SP wedge (not tradeable here) and is below noise in ln(BSP share / morning share); equal-weigh... | `market-12` |

## 6. What was rejected, and why

**Already published or practised (14)**

| Idea | Evidence | Id |
|---|---|---|
| Clock-going nowcast and going-surprise exposure per horse | Predictive model of official going from antecedent rain, temperature, wind, sunshine at Irish courses (Sheridan and Sweeney 2001) (https://mural.maynoothuniversity.ie/id/eprint/9963); Soil water-balance model of mean going for UK courses (poor predictions) and an irrigation-effect model (https://dspace.lib.cranfield.ac.uk/handle/1826/1742) | `physical-02` |
| Critical speed and distance reserve fitted per horse from standardised race times | Distance preference modelled per horse through average velocity, as a function of distance (book chapter) (https://www.cambridge.org/core/product/identifier/CBO9780511493614A026/type/BOOK_PART); Power-law speed-stamina 'fingerprints' fitted to times by distance for horses and courses (practitioner blog) (https://vendire-ludorum.blogspot.com/search/label/Stamina) | `physical-11` |
| Latent jumping skill against course stiffness, with lateral jump bias | Timeform 'Jumpability' ratings: horse-level jumping skill built from in-play jumping symbols, and the 'x'/'xx' poor-jumper symbols (https://www.sportinglife.com/racing/news/timeform-assessing-the-best-jumpers-over-fences-using-jumpability/196094); Timeform course table: which courses pose the toughest test of jumping (share of error-strewn rounds by course) (https://www.time... | `physical-12` |
| Thermal hidden going on all-weather surfaces | Timeform feature: how cold weather affects all-weather going, by track (Southwell most affected) (https://www.timeform.com/horse-racing/features/generic-feature/how-does-cold-weather-affect-the-going-on-the-all-weather-1412021); Race times against surface temperature on a synthetic track (Peterson et al. 2010) (https://pubmed.ncbi.nlm.nih.gov/20525055/) | `physical-14` |
| SP auction imbalance decides where the hedge is laid | Betfair's own tutorial: near/far price accuracy against the BSP (near 12.1%, far 57.8%, best back 9.2% mean absolute error) and a random forest on BSP-pool features (8.9%) (https://betfair-datascientists.github.io/tutorials/analysingAndPredictingBSP); Betfair documentation of near and far price as projections from SP money and unmatched exchange money (https://betfair-develo... | `market-13` |
| Shadow handicapper: predicted mark change not yet published | Weekly changes, collateral ones included, published 07:00 Tuesday and effective from Saturday; a winner runs under a penalty meanwhile (https://www.britishhorseracing.com/regulation/guide-to-handicapping/); Collateral re-marking stated by the regulator (https://www.britishhorseracing.com/regulation/adjusting-handicap-rating/) | `agents-01` |
| Variance premium: win share integrated over rating uncertainty | Bayesian state-space rank-ordered logit for multi-competitor events: abilities carried with posterior uncertainty and predictions integrated over it (per search summary; not re-read in full) (https://www.glicko.net/research/multicompetitor.pdf); TrueSkill: per-player mean and sigma, multi-player outcome probability depends on each sigma (https://papers.nips.cc/paper_files/pa... | `agents-05` |
| Martingale-exact percentile of in-running price extremes | Peak-on-loss law P(M >= x / loses) = p0(1-x)/((1-p0)x) for a binary Doob martingale, applied to sports win-probability paths (Jan 2026) (https://arxiv.org/abs/2601.18774); Distribution of max/min price given the outcome in a prediction market (Klein) (https://www.stat.berkeley.edu/~aldous/157/Old_Projects/klein.pdf) | `hidden-01` |
| Win probability added by race phase from sectional gates | Commercial in-running win probabilities: neural nets on GPS position and velocity, seeded with the starting price, sold to 40+ bookmakers (search-result summary) (https://europeangaming.eu/portal/latest-news/2021/06/02/93599/total-performance-data-prepares-for-major-rollout-with-pa-partnership/); Within-race state-space model on frame-level tracking separating horse ability,... | `hidden-08` |
| Stewards' reports, trainers' explanations and vets' findings as a dated feature family | The trade paper's data department logs all the information in the stewards' reports from each meeting (and calls the reports inconsistent) (https://www.racingpost.com/news/opinion/comment/time-for-action-over-infuriatingly-inconsistent-stewards-reports-ahbZi3I0NIQ9); The regulator publishes the reports by fixture, searchable by date (https://www.britishhorseracing.com/racing... | `hidden-14` |
| Race-coupled forecast uncertainty through the softmax Jacobian | Uncertainty in multinomial-logit racing probabilities carried into the bet (shrinkage for estimate error) (https://arxiv.org/pdf/1701.02814); Inverse-error-variance forecast combination (Bates and Granger 1969) (https://metricgate.com/docs/forecast-combinations-bates-granger/) | `method-07` |
| Innovation-form member: model the change since the horse was last priced | First-difference / within estimator removes the entity effect; forecast = lagged level + predicted change (https://panelbox.readthedocs.io/en/latest/user-guide/static-models/first-difference/); Flexible learners on differenced panel data (smooths on first differences) (https://arxiv.org/pdf/2606.12739) | `method-13` |
| Bridge-pinned resting backs priced off the forecast close | Algorithmic Trading with Learning (IJTAF 2016): mid-price as a randomised Brownian bridge with a prior on the terminal price; optimal posting of limit and market orders (https://ideas.repec.org/a/wsi/ijtafx/v19y2016i04ns021902491650028x.html); Optimal Timing to Trade Along a Randomized Brownian Bridge (Leung, Li, Li 2018): limit or market orders on a path pinned to a random... | `positions_data-02` |
| Decision-focused boosting through a smoothed entry rule | Decision-focused learning: train for decision quality, smoothed surrogates for discrete decisions, prediction-loss anchor (https://arxiv.org/pdf/2307.13565); Prediction loss added to the decision loss to smooth it (https://arxiv.org/pdf/2509.08359) | `positions_data-12` |

**Already in this repository (1)**

| Idea | Evidence | Id |
|---|---|---|
| Stale reduction-factor basis: the non-runner cut as a forecastable price | L:removal-by-market-1109 already set each removed runner's factor against its book share at the last read: 849 cut removals, factor mean 11.20% against 10.97%, others gain +0.19% - 'Betfair's factor is fair on average'. L:replay-nr-bar-1021: expected-cut bar lost (-GBP231, -GBP182). L:removal-tilt-replay-1109: -GBP1... | `market-00` |

**Dropped by the checker (10)**

| Idea | Evidence | Id |
|---|---|---|
| Fitness-fatigue impulse response with race-hardness impulses | Not new: a standard published model applied to information the served freshness block already carries; the fatigue arm is the long-practised and disputed 'bounce'; recency and decay re-weightings have already failed here (L:iter72) and days-decayed form is already open lead 6. The true load (training) is not on record. | `physical-05` |
| Horse-specific performance variance from place-versus-win price history | Variant of a retired block: the horse's own lagged win/place price ratio is mh_wp_h (L:iter98, +0.0006), per-horse variance from results is qm_/kr_sd (no gain), place prices add nothing beyond the BSP, and the model form is classical unequal-variance Thurstone. | `market-08` |
| Name topicality: daily public attention to the words in a horse's name | Pointless for this trade: the money it describes is bookmaker and Tote money on a few crowd days a year, the system trades the exchange close, and the test has no power on that sample. Its nearest published relative (page-view buzz in tennis betting) is disputed by a replication, and the build (dated token-to-articl... | `market-15` |
| Runner-specific effective prize from bonus eligibility | Cannot be built as defined (eligibility is paid registration and no dated list with history was found), the proxy mixes registered and unregistered horses, and the repo's own sizing of name-tagged subsets (ledger race-name-conditions, about 0.0001) says it cannot move the forecast. At most an hours-long proxy query. | `agents-10` |
| Twin causal forests for condition changes with a programme-scarcity instrument | Pointless as a forecast input: same inputs as served features, answering a different question (complier effect, not the chosen change the price reflects) with more noise; exclusion fails for the market outcome; a week-plus build on a family (intent angles) that showed nothing beyond BSP. The first-stage check is at... | `hidden-13` |
| Bornhuetter-Ferguson franking: chain-ladder development of form lines | Already here in equivalent form: the served fl_ block applies the same credibility shrink, the market-change franking was tried and rejected (L:iter62), and two deeper weightings added nothing (L:iter65, L:iter90). Franked form is also standard practice. Only the 7-day increment is untested, and that is a one-day re... | `method-05` |
| Going-transition-marginalised forecast and going sensitivity | The oracle version is already measured at nothing (L:card-inputs-swap-1110), the card history (from 14 Jul 2026) cannot support a transition table by course and month or any trained feature, and scenario pricing by going is practised. Revisit only with a weather source (open lead 10) and a year of cards. | `method-09` |
| Amortised Bayesian handicapper trained on simulated careers | Pointless as specified and not buildable with what is held: no career simulator, no GPU path, weeks of work, and the network adds no information over fitting the same prior directly - a family already tested with no gain (iter55, iter96, iter8). The novel part is only the amortisation. | `method-15` |
| The void put: backs whose bad news ends in a withdrawal | The 'void put' as an addition to expected CLV is wrong (double counting; the exchange price and the closing model are both conditional on running). The valid remainder - trainer withdrawal propensity as a feature - is the repository's open lead 9 (reports/model_review_1111.md:91) with one interaction added, and trai... | `positions_data-05` |
| Yard-base respiratory load from pollen and particulates | The core step (21-day pre-race exposure at the training base) was published in 2026. What remains is a between-yard contrast with little variation on a UK card, against the BSP, where every fundamental block here has failed. Yard coordinates and pollen history are not held. A week-plus build for an expected null. | `positions_data-13` |

**The same idea as another inventor's (12)**

| Idea | Evidence | Id |
|---|---|---|
| Rider weight floor: wasting margin and lead carried | same as: Dead-weight coupling from each jockey's inferred riding floor (`physical-13`) | `agents-15` |
| Per-horse critical speed and reserve fitted from race times | same as: Critical speed and distance reserve fitted per horse from standardised race times (`physical-11`) | `hidden-07` |
| Nested substitution at withdrawal against Betfair's proportional reduction | same as: Substitution-aware re-formation: cross-elasticities of the book beyond renormalisation (`market-06`) | `method-02` |
| Race-load fitness-fatigue impulse response per horse and yard | same as: Fitness-fatigue impulse response with race-hardness impulses (`physical-05`) | `method-06` |
| Attrition dividend: a cure-model share for jump races | same as: Survive-then-contest likelihood for jump races (`physical-07`) | `method-12` |
| Runner-level removal carry with unequal inheritance of a withdrawn rival | same as: Stale reduction-factor basis: the non-runner cut as a forecastable price (`market-00`) | `positions_data-01` |
| Low-sun obstacle omission forecast from solar geometry and cloud | same as: Low-sun obstacle forecast from solar geometry and course bearings (`physical-10`) | `positions_data-04` |
| Quote-origin classification of the morning offer being taken | same as: Odd-lot footprint: stake-size digits separate model money from human money (`market-09`) | `positions_data-08` |
| Trainer race-choice inverse model from entries and declarations | same as: Roy-model inversion: private ability implied by the race chosen (`agents-03`) | `positions_data-09` |
| Hoof-traffic wear state since the last rail movement | same as: Hoof-traffic wear state of each strip of track (`physical-08`) | `positions_data-11` |
| Forecast wind vector against run-in bearing for pace and rail bias | same as: Crosswind echelon - wind-vector forecast of the day's draw-side and cover bias (`physical-03`) | `positions_data-14` |
| Trainer excuse validity from official post-race explanations | same as: Stewards' reports, trainers' explanations and vets' findings as a dated feature family (`hidden-14`) | `positions_data-15` |


## 7. What to test first

Each test is written into the ledger before it runs. A block is screened on the 11 folds with a placebo arm (noise bar
about 0.0007, ledger `iter114`); a market study is a read-only query on the recorded books or the price files.

| Order | Idea | Why first | First step | Effort |
|---|---|---|---|---|
| 1 | Ghost runs (4.1) | It extends the market rating, the largest gain on record, with readings the rating throws away today: the price of a horse that was declared, traded and withdrawn | A read-only count of removed runners with a trade; stop if under 2% of runners are touched | Days |
| 2 | Conditions-aware market rating (4.2) | The same rating, made to say what the market thinks of the horse at this trip, ground and track | One block beside the served rating, with the retired conditions block as the cheap control | Days |
| 3 | Learned forgiveness (4.3) | How far the next price moved after each run is the market's own verdict on that run; nothing here uses it as a weight | One block, 11 folds | Days |
| 4 | Reliability-weighted book (4.4) | It acts on the closing model's market leg in the thin morning book, which is where the money is made | One query on the recorded marks from 20 Oct | A day |
| 5 | Substitution after a withdrawal (4.5) | Non-runner cuts are the trade's known weak point, and Betfair's cut is proportional | One query on the roughly 849 cut removals recorded since 2 Oct | Days |
| 6 | The exit price of fallers (4.6) | Jumps season; the in-play low is held and a non-finisher is read as blank today | One block on jumps rows | A day |
| 7 | Attention share of the day (4.7) | A race's expected share of the day's turnover is known before any trade and may say how far to trust its morning price | One module beside the closing model, no live change | A day |
| 8 | Knowability of the field (4.8) | The same question from the form graph: how well can anyone compare these runners | One query on a year of price files, with controls | Days |
| 9 | Intraday franking (4.9) | A rival's result earlier the same day is scheduled news; the links are known at 06:00 | One event study on the recorded marks, with placebo links | A day |

After these, the four the checkers rated highest for the forecast, each needing more building: the career curve of
the market rating (4.10), the size of the crowd behind an owner (4.11), the going the market is assuming at a meeting
(4.12) and a memory network trained on closing books (4.13).

What I would not expect: a feature that predicts the result beyond the BSP. Every checker rated that 1 of 5 for every
idea, and our own tests agree (ledger `bsp-calibration-1007`, `benter-blend-1020`).

## Appendix - the evidence

`reports/discovery_1111_ideas.jsonl` holds all 95 ideas as the inventors wrote them (definition, mechanism, data,
first test) and, for the 83 that were checked, the checker's verdict: where it stands in the repository, the data, the
timing, the closest precedents with their sources, the searches run, what is new, the priors, the likely failure and
the corrected definition. `reports/discovery_1111_maps/` holds the seven maps.
