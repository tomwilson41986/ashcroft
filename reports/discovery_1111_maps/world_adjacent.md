# MAP: world_adjacent

# MAP world_adjacent - methods established elsewhere, checked for racing use (11 Nov 2026)
Key: RU = racing use found (one generic web search per item, plus a repo grep). "repo" = already present in /home/user/ashcroft. [u] = reference DOI from memory, not re-confirmed this session. Format: method - reference - RU - mapping hint.

## Sports analytics
- Action/possession value (xG, VAEP: change in P(score) per action) - https://arxiv.org/abs/1802.07127 - RU: none found (Timeform "expected position" is ratings-based, https://www.sportinglife.com/racing/news/derby-preview-timeform-data-produces-expecting-finishing-positions/209829) - value each in-running move by win-probability change.
- Expected possession value from tracking (multiresolution stochastic process) - https://arxiv.org/abs/1408.0777 - RU: partial, student win-probability-by-frame work on NYRA tracking (https://www.stat.cmu.edu/capstoneresearch/fall2022/490files/poster4.pdf) - continuous in-race win probability per horse.
- Tracking-data generative models - https://arxiv.org/abs/2310.01748 - RU: YES (frame-level Bayesian race model, drafting, lane counterfactuals, Big Data Derby data) - separate ability from trip and tactics.
- Regularised adjusted plus-minus (ridge on all co-participants) - Sill 2010 as cited at model/effects.py:7 - RU: YES (repo model/effects.py:2; hierarchical horse+jockey model https://reference-global.com/article/10.2478/ijcss-2023-0007) - jockey/trainer/sire credit net of horse.
- Elo/Glicko/TrueSkill, multi-competitor and through-time - https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/ , https://arxiv.org/abs/2101.00400 - RU: YES (state-space rank-ordered logit https://www.glicko.net/research/multicompetitor.pdf; repo model/blocks/elo.py:8, model/blocks/kalman.py:1, RESEARCH_FRAMEWORK.md:288) - backward smoothing through a career (TrueSkill Through Time) not seen.
- Ageing curves (delta method, quadratic peak) - https://www.jstage.jst.go.jp/article/jes/21/4/21_4_73/_article/-char/en - RU: YES (peak 4.45 years on speed figures, per that paper) - individual, selection-corrected curves not seen.
- Luck-vs-skill variance decomposition (Bayesian state-space on market probabilities) - https://arxiv.org/abs/1701.05976 - RU: none found - reliability-weighted shrinkage per race type.
- Opponent-adjusted ratings - same references - RU: YES, traditional collateral form (repo model/blocks/collateral.py:1, form_lines.py).

## Endurance physiology and race strategy
- Critical power/speed hyperbola (time-to-fatigue = W'/(P-CP)) - https://madbarn.com/research/hyperbolic-relationship-between-time-to-fatigue-and-workload - RU: YES on treadmill (1999) and in repo ABM (model/abm/agents.py:60); none found fitted per horse from race or sectional data.
- W' balance (depletion/reconstitution integral) - Skiba 2012, doi 10.1249/MSS.0b013e3182517a80 [u] - RU: repo ABM only (model/abm/agents.py:71) - anaerobic reserve left entering the straight.
- Optimal-control pacing (energy ODE, minimise time) - https://arxiv.org/abs/2006.10530 - RU: YES (Chantilly thoroughbreds, PLOS ONE 2020) - deviation from own optimal pace as a feature not seen.
- Motorsport Monte Carlo race-strategy simulation - https://www.mdpi.com/2076-3417/10/12/4229 - RU: analogous repo race ABM (model/abm/); published betting-exchange ABM https://arxiv.org/abs/2105.08310.
- Cycling breakaway/peloton drafting optimisation - https://arxiv.org/abs/2507.02992 - RU: drafting measured (Spence 2012, cited in arXiv 2006.10530); game-theoretic contest for the lead none found.

## Quantitative finance
- Order-flow imbalance (signed best-quote size changes -> next mid move) - https://arxiv.org/abs/1011.6402 - RU: none found for horses in print; repo cites it for greyhounds (reports/greyhound_literature.md:15).
- Hawkes self-exciting order arrivals - https://arxiv.org/abs/1502.04592 - RU: none found (repo mention RESEARCH_FRAMEWORK.md:288) - branching ratio of bets as information-arrival measure.
- Lead-lag networks and clustering - https://arxiv.org/abs/2201.08283 - RU: none found for horses (tennis match-vs-set price discovery https://academic.oup.com/ej/article/124/575/245/5076975) - which runners, markets, courses move first.
- Factor residualisation (signal on residual after removing common factors) - https://doi.org/10.1016/j.jempfin.2011.01.003 - RU: none found - strip market-implied component from every feature.
- Kyle / Glosten-Milgrom informed-trader models - Shin's insider share z: https://www.ucd.ie/economics/t4media/WP2024_19.pdf - RU: YES (Shin 1991-93; repo model/odds_metrics.py) - Kyle lambda per runner on exchange none found.
- Square-root market impact - https://arxiv.org/abs/1105.1694 - RU: none found in print; repo capacity study research/queries/done/trading_capacity.py.
- Alpha decay after publication - McLean-Pontiff 2016, doi 10.1111/jofi.12365 [u] - RU: none found - half-life of each angle since public.
- Cross-sectional momentum/reversal in betting prices - https://econpapers.repec.org/RePEc:bla:jfinan:v:76:y:2021:i:6:p:3153-3209 (US team sports) - RU: practitioner steamer/drifter tables https://www.geegeez.co.uk/the-truth-about-steamers-and-drifters/ ; formal exchange test none found.
- Factor zoo / multiple-testing control, deflated Sharpe - Harvey-Liu-Zhu 2016 [u] - RU: YES, repo research/overfit.py:3-4.
- Price clustering under cognitive load - https://ueaeprints.uea.ac.uk/57660/1/Clustered_Prices_Edited2.pdf - RU: YES (Betfair, Royal Ascot and Cheltenham).
- Return autocorrelation/efficiency of exchange price series - https://arxiv.org/abs/2402.02623 - RU: YES (UK horse racing, Betfair).

## Forecasting science
- Extremising aggregated probabilities by information overlap - https://arxiv.org/abs/1406.2148 - RU: forecast combination yes (https://eprints.soton.ac.uk/349781/); extremising none found (repo grep: 0 files).
- Prediction-market vs poll combination - Atanasov 2017, doi 10.1287/mnsc.2015.2374 [u] - RU: tipster-crowd studies only (https://ueaeprints.uea.ac.uk/67529/4/Accepted_manuscript.pdf) - weight exchange, bookmaker, tipster as forecasters.
- Conformal prediction sets/intervals - https://arxiv.org/abs/2107.07511 - RU: none found in print; repo model/uncertainty.py:8,13 (split conformal, conformal Kelly).
- Score decompositions (CORP reliability, Murphy) - https://arxiv.org/abs/2008.03033 - RU: repo model/diagnostics.py:12.

## Machine learning
- Deep Sets / Set Transformer (permutation-invariant field encoder) - https://arxiv.org/abs/1703.06114 , https://arxiv.org/abs/1810.00825 - RU: none found in print; repo greyhounds only (greyhound/nn.py:5).
- Temporal graph networks - https://arxiv.org/abs/2006.10637 - RU: static horse-graph features only (https://ir.library.louisville.edu/etd/2953) plus a forum sketch (https://discuss.dgl.ai/t/representing-races-geometrically-for-a-gnn/4042) - who-beat-whom graph evolving in time.
- Life-event sequence foundation model - https://arxiv.org/abs/2306.03009 - RU: none found (one unreviewed student transformer report) - career as token sequence, pretrain, embed.
- Tabular foundation models / in-context learning - https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/ - RU: none found (repo grep: 0) - race-as-context prediction.
- Time-series foundation models - https://arxiv.org/abs/2403.07815 , https://arxiv.org/abs/2310.10688 - RU: none found (hobby football projects only) - zero-shot price-path or rating-path forecast.
- Path signatures (iterated integrals of a path) - https://arxiv.org/abs/1603.03788 - RU: none found (soccer possessions https://arxiv.org/abs/2508.12930) - order-aware summary of price path or form sequence.
- Optimal transport / Wasserstein - https://arxiv.org/abs/1306.0895 - RU: none found (team styles https://arxiv.org/abs/2501.10299; repo grep: 0) - distance between fields or run-profile distributions.
- Kernel mean embeddings / distribution regression - https://arxiv.org/abs/1605.09522 - RU: none found - embed the opposition as a distribution.
- Causal forests, double ML - https://arxiv.org/abs/1510.04342 , https://arxiv.org/abs/1608.00060 - RU: none found in print; repo has AIPW (model/causal.py:12), not forests.
- Off-policy evaluation (doubly robust) - https://arxiv.org/abs/1103.4601 - RU: none found - value a new staking rule from logged bets.
- Simulation-based inference - https://arxiv.org/abs/1911.01429 - RU: repo ABC-SMC calibration (model/abm/calibrate.py:7); neural posterior estimation none found.
- Differentiable simulators - https://arxiv.org/abs/1910.00935 , https://arxiv.org/abs/2106.13281 - RU: none found - gradient-fit race simulator to results.
- LLM-derived features/embeddings - https://arxiv.org/abs/2305.03403 - RU: none peer-reviewed (cycling commentary extraction exists); repo model/blocks/comments.py:1 reads comments by pattern classes, not embeddings.
- Symbolic regression / genetic programming - https://arxiv.org/abs/2305.01582 - RU: YES (GP betting rules 1995 https://gpbib.cs.ucl.ac.uk/gp-html/butler_1995_eddie.html; Cartesian GP on pre-race price moves https://bura.brunel.ac.uk/bitstream/2438/16994/5/FullText.pdf, also reports/greyhound_literature.md:63).
- HodgeRank / Rank Centrality - https://link.springer.com/doi/10.1007/s10107-010-0419-x , https://arxiv.org/abs/1209.1688 - RU: none found (repo grep: 0) - cyclic residual = intransitivity of form lines.
- Multinomial, nested, mixed logit - https://eml.berkeley.edu/books/choice2nd/Ch04_p76-96.pdf - RU: multinomial YES (Bolton-Chapman 1986, https://www.Gwern.net/doc/statistics/decision/1986-bolton.pdf); Plackett-Luce estimators on racing data https://arxiv.org/abs/2306.02821 ; repo model/ordering.py:21; nested/mixed none found by search.
- Mixtures of Plackett-Luce - https://arxiv.org/abs/1501.03519 - RU: none found - latent race "types" with different orderings.

## Ecology and epidemiology
- Contagion / latent yard-state models - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6848918/ (virus persisting in a yard, associated with failure to race) - RU: epidemiology yes; latent yard-state inferred from results as a predictor none found (repo model/blocks/stable.py:1 counts stablemates and jockey choice only).
- Survival and frailty models - https://madbarn.com/research/profiling-the-careers-of-thoroughbred-horses-racing-in-hong-kong-between-2000-and-2010 - RU: YES for career length and injury (Cox, Weibull); repo model/effects.py:21 - hazard of non-reappearance as form signal not seen.
- Capture-recapture - https://www.stat.auckland.ac.nz/~fewster/Capture-Recapture-RMFewster-2015.pdf - RU: none found (repo grep: 0) - infer unseen ability from entry/absence patterns.

## Geoscience
- Soil water-balance - https://dspace.lib.cranfield.ac.uk/handle/1826/1742 - RU: YES (2006 thesis, mean-going prediction reported poor; Irish weather-to-going regression https://www.irishtimes.com/news/a-firm-basis-for-heavy-going-on-the-track-1.289851) - per-course moisture deficit versus official going.
- Reanalysis / station weather - repo sources/noaa_isd.py:3-8 (NOAA ISD chosen; ERA5 and Open-Meteo noted) - RU: yes in practice.
- Satellite soil moisture / vegetation indices - https://www.mdpi.com/2076-3417/11/24/11769 (golf turf, Sentinel-2) - RU: none found (courses use ground probes) - rail-to-rail moisture gradient per course.
- Wind fields (head/cross-wind components per straight) - https://eprints.soton.ac.uk/508996/ (race cars) - RU: practitioner speed-figure adjustments yes (https://www.drf.com/news/answers-are-blowing-wind); peer-reviewed none found.

## Behavioural science
- Salience theory - Bordalo-Gennaioli-Shleifer 2012, doi 10.1093/qje/qjs018 [u] - RU: none found - which runner attribute stands out in the field.
- Name fluency / affect - https://papers.ssrn.com/abstract=3469641 - RU: YES (fast-sounding names overbet, over 3 million exchange bets per the abstract) - pronounceability or memorability separately none found.
- Attention (search volume) - https://rady.ucsd.edu/faculty/directory/engelberg/pub/portfolios/GOOGLE.pdf - RU: none found - pre-race search/media volume per horse.
- Narrative economics - Shiller 2017, doi 10.1257/aer.107.4.967 [u] - RU: none found - story strength versus price.

## Blind spots of this map
- "None found" means one generic standard-mode web search per item (US index) plus a filename-level grep of the repo's .py/.md/.jsonl; it is absence of evidence, not proof. Proprietary syndicate, Hong Kong, Japanese-language and commercial (Timeform, Racing Post, tracking vendors) work is not publicly searchable and may already use several of these.
- Not web-searched individually for a racing use (classed "none found" on general knowledge and adjacent searches only): kernel mean embeddings, off-policy evaluation, differentiable simulators, alpha decay, factor residualisation, salience theory, narrative economics, prediction-market/poll combination, score decompositions.
- References marked [u] (Skiba 2012, McLean-Pontiff 2016, Harvey-Liu-Zhu 2016, Atanasov 2017, Bordalo et al. 2012, Shiller 2017) and Sill 2010 are cited from memory or from the repo's own citation; DOIs were not re-fetched. Bolton-Chapman 1986, Spence 2012 and Lauderdale-Hinchcliff 1999 were seen only through search snippets or secondary summaries, not full text. All arXiv IDs were title-checked against the arXiv API except 1810.00825, 2310.01748, 2101.00400, 2107.07511, 1406.2148, 1209.1688, 2306.02821, 2402.02623, 2105.08310, 2507.02992, 2508.12930, 2501.10299, 2006.10530, which were confirmed from search result listings.
- Paper contents were read from search summaries only (no full-text reads), so figures quoted (peak age 4.45 years; over 3 million bets) rest on those summaries of the cited sources.
- Repo flags are from grep hits and module docstrings; I did not check whether each repo item is served, retired or only mentioned (e.g. Hawkes at RESEARCH_FRAMEWORK.md:288 is a mention, not an implementation). The research ledger (reports/research_ledger.jsonl) was not read for this map; the in-repo map should be treated as authoritative on what has been tried.
- Brief items thinly covered or not covered: opponent-adjusted ratings beyond Elo/collateral form, ERA5/Open-Meteo primary references (only the repo's note cited), wind-field products, vegetation indices specifically (NDVI found only for golf turf), mixed/nested logit in racing (one search, could exist in the Southampton group's work), TrueSkill Through Time's own paper (URL not confirmed).
- No database, S3 or live system was touched; nothing in the repository was edited. A scratch draft exists only at /tmp/claude-0/-home-user-ashcroft/20b9529a-9fd3-5b98-8cce-f095c5ce099c/scratchpad/map.md (an earlier draft; this output supersedes it).
