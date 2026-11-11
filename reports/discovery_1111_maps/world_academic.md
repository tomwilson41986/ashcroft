# MAP: world_academic

# MAP world_academic - published work on predicting horse races and racing betting markets, 1986-2026
Status: [V] = record or abstract read in this session (11 Nov 2026); [U] = unverified, from memory or a secondary citation only. No full texts behind paywalls were read.

## 1. Discrete-choice win models; model plus market
- Multinomial logit on horse, jockey and race covariates, rank-ordered "exploded" estimation, 200 races, hold-out betting - Bolton & Chapman 1986, Management Science 32(8) [V] - https://ideas.repec.org/a/inm/ormnsc/v32y1986i8p1040-1060.html
- Same model on 2,000 Hong Kong races, win probabilities under 0.04 excluded - Chapman 1994, repr. 2008 [V] - https://ideas.repec.org/h/wsi/wschap/9789812819192_0018.html
- Two-stage: fundamental logit, then second logit on log model probability and log public probability; pseudo-R2 gain as the edge measure; Kelly staking - Benter 1994 [V] - https://www.gwern.net/doc/statistics/decision/1994-benter.pdf
- One-step v two-step conditional logit, UK data; two-step more profitable - Sung & Johnson 2007, J Prediction Markets 1(1) [V] - https://ideas.repec.org/a/buc/jpredm/v1y2007i1p43-59.html
- Multinomial probit (normal errors) by MCMC stochastic approximation, 1,000 HK races, beats logit - Gu, Huang & Benter c.2003 [seminar abstract only; venue U] - https://www.scribd.com/document/519783748/gu
- Place/show pools priced from win-pool probabilities (Harville), log-utility non-linear programme ("Dr Z") - Hausch, Ziemba & Rubinstein 1981, Management Science 27(12) [V] - https://ideas.repec.org/a/inm/ormnsc/v27y1981i12p1435-1452.html

## 2. Ordering (place, exotic) probabilities
- Renormalised win probabilities for 2nd, 3rd (Luce / Plackett-Luce) - Harville 1973, JASA [U]
- Normal running times - Henery 1981 [U]; gamma running times - Stern 1990 [U]; both fit better than Harville on HK, Meadowlands and Japan data - Lo & Bacon-Shone 1994, The Statistician 43(2); Lo, Bacon-Shone & Busche 1995, Management Science 41(6) [V] - https://ideas.repec.org/a/inm/ormnsc/v41y1995i6p1048-1059.html
- Discount model: p^lambda renormalised per finishing position as a cheap Henery/Stern approximation - Benter 1994; Lo & Bacon-Shone 2008 [venue U]
- Software: softmax regression under Harville/Henery - R package ohenery (Pav) [V] - https://ftp.fau.de/cran/web/packages/ohenery/refman/ohenery.html

## 3. Favourite-longshot bias and efficiency
- Longshots overbet in pari-mutuel pools - Ali 1977; Thaler & Ziemba 1988 [U]
- Misperception v risk-love, tested by pricing exactas, quinellas and trifectas from win-fitted models - Snowberg & Wolfers 2010, JPE 118(4) [V] - https://www.nber.org/papers/w15923
- Insider share z inferred from bookmaker overround - Shin 1993, Economic Journal 103 [V] - https://ideas.repec.org/a/ecj/econjl/v103y1993i420p1141-53.html
- Exchange v bookmaker on matched UK races: smaller bias, weak and strong-form efficiency on the exchange - Smith, Paton & Vaughan Williams 2006, Economica 73 [V] - https://ideas.repec.org/a/bla/econom/v73y2006i292p673-689.html
- Tipster selections against odds: on-track pool discounts them, off-track does not - Figlewski 1979, JPE 87(1) [V] - https://ideas.repec.org/h/wsi/wschap/9789812819192_0029.html
- Post position: rail advantage real but priced - Canfield, Fauman & Ziemba 1987, Management Science 33(11); contested, effect grows with field size - Betton 2008 [V] - https://ideas.repec.org/a/inm/ormnsc/v33y1987i11p1428-1439.html
- Timing of pari-mutuel bets (late bets more informed) - Ottaviani & Sorensen 2007 [listing only] - https://www.kellogg.northwestern.edu/faculty/research/detail/2007/the-timing-of-parimutuel-bets
- Shape of the odds path (modelled price curves) added to conditional logit; closing price does not embed it fully - Johnson, Jones & Tang 2006, Management Science 52(6) [V] - https://ideas.repec.org/a/inm/ormnsc/v52y2006i6p897-908.html
- Odds-trend variables over 6,058 exchange markets, 8.4m price points: shortening prices over-estimate, drifting prices under-estimate probability; correction method - Sung, McDonald, Johnson, Tai & Cheah 2019, EJOR 272(1) [V; sport not confirmed] - https://ideas.repec.org/a/eee/ejores/v272y2019i1p389-405.html

## 4. Exchange microstructure, pre-race and in-play
- 9,562 UK races, about 225m limit-order-book observations: traded volume predicts outcome, mostly in-running - Brown & Yang 2017, Review of Finance 21(2) [V] - https://ueaeprints.uea.ac.uk/id/eprint/58626/
- Lay-side order book tested for corruption, none widespread - Brown & Yang 2017, J Sports Economics 18(7) [V] - https://ideas.repec.org/p/uea/aepppr/2012_66.html
- Betfair fee change as natural experiment, 11,307 races: price discovery not confined to top traders - UEA paper [authors U] - https://ueaeprints.uea.ac.uk/55209/1/Information_in_Speculative_Markets_Revised.pdf
- Pre-off return series: short tails, no long memory - Tondapu 2024 [V] - https://arxiv.org/abs/2402.02623
- Convolutional attention plus recurrent nets on pre-off order-book depth, short-horizon price moves - Goncalves et al. 2025 [V] - https://arxiv.org/abs/2510.16008
- Neural nets and Cartesian genetic programming on pre-race price movement - Brunel thesis [snippet only] - https://bura.brunel.ac.uk/bitstream/2438/16994/5/FullText.pdf
- Agent-based in-play exchange simulator (BBE) - Cliff 2021, arXiv 2105.08310; XGBoost cloning profitable agents - Terawong & Cliff 2024 [V] - https://arxiv.org/abs/2401.06086
- Computer-assisted wagering, rebates, last-minute pool swings - trade press only [V] - https://www.thoroughbredracing.com/articles/blight-late-odds-shifts-we-must-not-allow-it-compromise-racings-new-opportunities

## 5. Machine learning
- SVM adapted to race-stratified outcomes, 200 races in-sample, 12 inputs - Edelman 2007, Annals of OR 151 [V] - https://ideas.repec.org/a/spr/annopr/v151y2007i1p325-33610.1007-s10479-006-0131-7.html
- SVM winner/loser classifier as first stage, conditional logit second - Lessmann, Sung & Johnson 2009, EJOR 196(2) [V] - https://econpapers.repec.org/RePEc:eee:ejores:v:196:y:2009:i:2:p:569-577
- Random forest in the two-stage frame - Lessmann, Sung & Johnson 2010, IJF 26(3) [V] - https://ideas.repec.org/a/eee/intfor/v26yi3p518-536.html
- Model-pool generation, stacking by conditional logit, log-likelihood-ratio forecast selection - Lessmann, Sung, Johnson & Ma 2012, EJOR [abstract only]
- Six classifiers (XGBoost, random forest, CART...), 3,782 Polish races, AUC and return - Borowski & Chlebus 2021, Warsaw WP 2021-13 [V] - https://ideas.repec.org/p/war/wpaper/2021-13.html
- Pointwise v pairwise ranking (RankNet, LambdaMART); relative-ability difference features matter most - Chung, Shin, Hwang & Park 2024, Korean J Applied Statistics 37(2) [blog summary only] - https://junhyoung-chung.github.io/2025/01/07/Horse-race-rank-prediction-using-learning-to-rank-approaches.html
- Perspective on end-to-end deep learning and vision for racing - Colle 2022 [V] - https://arxiv.org/abs/2207.04981
- Sports-betting ML systematic review - arXiv 2410.21484 [V]
- Early feed-forward nets - Williams & Li 2008; Davoodi & Khanteymoori 2010 [U]. Transformer on HK data: student project only (CUHK 2022) [V]

## 6. Bayesian, state-space and rating models
- Hierarchical Bayes linear model of speed, Gibbs v Metropolis - Silverman 2012, J Prediction Markets 6(3) [V] - https://ideas.repec.org/a/buc/jpredm/v6y2012i3p1-13.html
- Hierarchical Bayes horse ability plus jockey skill - IJCSS 2023 [title only] - https://reference-global.com/article/10.2478/ijcss-2023-0007
- State-space rank-ordered logit, Gaussian random-walk ability (applied to skiing) - Glickman & Hennessy 2015, JQAS 11(3) [V] - https://www.glicko.net/research/multicompetitor.pdf
- Frame-level generative model on 4 fps tracking (NYRA 2019): forward and lateral movement, horse speed profiles, counterfactual simulation - JQAS, arXiv 2310.01748 [V] - https://arxiv.org/abs/2310.01748
- Mixed model of average velocity with race, age and jockey effects - Oda & Onogi 2024, J Animal Breeding and Genetics [abstract only]

## 7. Pace, biomechanics, surface, behaviour
- 1 Hz tracking, 44,803 starts, 3,357 UK races: time spent drafting and pacing profile against finish - Spence et al. 2012, Biology Letters 8(4) [V] - https://pmc.ncbi.nlm.nih.gov/articles/PMC3391435
- Optimal control on velocity, propulsive force and anaerobic energy, fitted to 10 Hz Chantilly data; bends - Mercier & Aftalion 2020, PLOS ONE [V] - https://arxiv.org/abs/2006.10530
- Stride length and frequency from about 25,000 starts; speed explains half or less of between-horse stride variation; stride decline across starts precedes injury - Morrice-West et al. 2021; Wong et al. 2022 [V] - https://zendy.io/title/10.1111/evj.13370
- Critical speed and D' across species, horses included - invited review [abstract only] - https://madbarn.com/research/invited-review-the-speed-duration-relationship-across-the-animal-kingdom
- Surface cushioning (Vienna Surface Tester) tracks winning times better than GoingStick - UCLan Ground and Going Study 2024 [V] - https://clok.uclan.ac.uk/id/eprint/53458/1/Ground%20and%20Going%20Study%20Report%20Sept%202024.pdf ; weather-to-going prediction - Maynooth thesis - https://mural.maynoothuniversity.ie/9963
- Paddock behaviour, 867 horses: tail elevation, neck angle, bit resistance weakly predictive, below SP and weight - Hutson & Haskell 1997, Applied Animal Behaviour Science 53(4) [V] - https://www.scilit.net/publications/79eacee7711e8e0abcd8c56334a8f41d ; sweating - Wells, Randle & Williams 2022 [secondary]
- Markerless pose (DeepLabCut) stride and asymmetry at walk and trot - 2023-2025 veterinary papers [V], none linked to race outcome
- Jockey: crouched posture and race times - Pfau et al. 2009, Science [U]; head kinematics by experience - Scientific Reports 2025 [V]

## 8. Genetics
- MSTN g.66493737C>T against best race distance, 148 horses - Hill et al. 2010, PLoS ONE [V] - https://doi.org/10.1371/journal.pone.0008645 ; GWAS confirmation 2010; SINE-insertion mechanism 2018 [V]
- Animal-model heritability of time, rank and earnings - Velie et al. 2015 (164,046 Australian horses); Svobodova et al. 2005 (BLUP); Thurstonian rank model, Brazil [V] - https://zendy.io/title/10.1111/age.12234
- Genomic prediction of racecourse starts (random forest with mixed effects) - McGivney et al. 2019, Animal Genetics [V]

## 9. Syndicate accounts
- Benter-Woods Hong Kong operation - Benter 1994; Bloomberg feature 2018 [mirrors only]; SCMP feature [V] - https://www.scmp.com/postmag/culture/article/3295454/hong-kong-horse-racing-age-gentleman-jockeys-bill-benters-billion-dollar-bet

## THEMES WHERE I FOUND LITTLE OR NOTHING PUBLISHED (one or two generic searches each, not exhaustive)
- Graph neural nets, deep sets, LSTM or transformers on racing form: student projects and one perspective only.
- Elo or TrueSkill applied to horses: no peer-reviewed paper.
- NLP or LLM on race comments, stewards' reports or trainer quotes: journalism and student work only.
- Vision on paddock or race video predicting outcome: none.
- Forecasting the exchange closing price (BSP) from morning information: none; nearest are short-horizon order-book studies.
- Modern exchange-era draw-bias efficiency test: none after Canfield 1987 / Betton 2008.
- Rainfall or going-measurement regressions on times: one industry report.
- Equine critical-speed primary study tied to race results: none.
- Jockey variance decomposition against prices; aggregated tipster "crowd" studies: none.
- Computer-team late money in academic journals: none.

## Blind spots of this map
Evidence base: about 45 generic web searches and one page fetch on 11 Nov 2026; almost everything was read at abstract / bibliographic-record level (RePEc, PMC, arXiv, repository listings), not full text - paywalled journals (IJF, EJOR, Management Science, JPE, Review of Finance) were not opened. Items tagged [U] are from memory or secondary citation and were not confirmed in this session: Harville 1973, Henery 1981, Stern 1990, Ali 1977, Thaler & Ziemba 1988, Pfau et al. 2009, Williams & Li 2008, Davoodi & Khanteymoori 2010, the venue of Gu-Huang-Benter and of Lo & Bacon-Shone 2008, and the authorship of the UEA Betfair fee-change paper. Sung et al. 2019 is confirmed as a betting-exchange study but the abstract does not state the sport. Benter 1994 sample sizes differ between secondary write-ups (2,313 v 3,198 races) and were not checked against the primary text, so I quoted none. The Chung et al. 2024 learning-to-rank paper was seen only through an author blog summary; Lessmann et al. 2012 and Oda & Onogi 2024 through abstracts; the IJCSS 2023 horse-plus-jockey hierarchical model by title only. Not searched or not found (blind spots, absence here is NOT proof of absence): Asch-Malkiel-Quandt and Gramm-McKinney late-money papers; Schnytzer's plunge studies beyond a secondhand insider-share figure; Bruce/Johnson UK draw-bias and each-way papers; weight-carried and handicap-mark studies; trainer-effect and trainer-intent papers; pedigree/dosage and sire-aptitude literature; equine exercise physiology (lactate, heart rate, VO2) as predictors; harness, greyhound and Japanese/Korean-language literature (large and under-indexed in English search); Hong Kong university theses; conference papers on IEEE Xplore / ACM (search returned spam mirrors for neural-net queries); patents and commercial vendor white papers; practitioner books (Beyer, Brohamer, Quirin, Mordin, Timeform methods) which are outside "academic" but contain much known art; Kaggle Big Data Derby winning notebooks beyond a mention. The "little or nothing published" list rests on one or two searches per theme with a US-only search tool, so it should be treated as leads to re-check before any novelty claim, not as a clearance. No repository file, database or secret was read or modified; a scratch draft exists only at /tmp/claude-0/-home-user-ashcroft/20b9529a-9fd3-5b98-8cce-f095c5ce099c/scratchpad/map.md (outside the repository).
