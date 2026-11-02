# Football: a quant-heavy approach to modelling and trading, built for an exchange-only operation (2 Nov 2026)

The owner's ask (2 Nov 2026): research leading practice and cutting-edge football modelling and trading (Jamestown
Analytics, Starlizard, Benham, the Asian handicap syndicates) and define a quant-heavy, world-leading approach to feature
engineering and modelling. Widened the same day: a review of the academic literature (2018-2026 and the classics), and a
section on how our own horse engine transfers to football.

Conventions. Every factual claim carries a source (the list is at the end; how each was read is in
`reports/football_quant_approach_sources.md`). **[R]** marks my own reasoning, not a finding. **[unverified]** marks a
claim I could not confirm from a primary or reputable source. Repository figures are cited by file and section. Nothing
here was run against S3 or Betfair, and no trading code was run.

## The result

1. **The leading firms do not win by having a cleverer goals model than the literature.** The best public account of
   Starlizard (Business Insider, 2016, from former staff) describes a four-team pipeline: about 30 researchers who watch
   matches and record chances, a quant team that turns the data into a score-line distribution and an Asian handicap
   line, about 20 "selectors" who compare that line with the market, and a placement team that bets through Asian
   brokers "as close to kick-off as possible", targeting 1-3% on turnover. Smartodds (Benham) employs "watchers" and a
   data team and was built on the Dixon-Coles model, whose co-author Stuart Coles joined it. The edge is **proprietary
   data collected by people, priced by a sound score model, executed at the right moment, at scale, at 1-3%.**
2. **Jamestown Analytics is real but is not a betting firm on any public evidence.** It is a football-club analytics
   service founded in 2017 as an offshoot of Starlizard (Training Ground Guru, 2026), selling recruitment, player
   valuation, head-coach recruitment and opposition analysis to one club per country (Brighton, Union Saint-Gilloise,
   Como, Hearts, Swansea, Melbourne Victory, Shelbourne and others). Its ownership is reported inconsistently (Bloom by Sky
   Sports; a named individual by one blog: unverified). Nothing public says it bets or what its models are.
3. **No published football model I found beats a sharp close.** The academic "profitable" results (Koopman and Lit 2015,
   Boshnakov et al. 2017, Constantinou and Fenton 2013, Wheatcroft 2020, Holmes and McHale 2024, Egidi et al. 2018) are
   against bookmaker or average odds, not Pinnacle's or Betfair's close. Pinnacle's close returns about 1:1 (Buchdahl);
   Betfair's pre-play price is unbiased (Whelan 2025). Our own model is 0.015 of log-loss behind the open and gets a
   negative pooled weight (`reports/football_model.md` s.3). The close is better than the open by 0.0034; **that move,
   not the result, is the only football target where an edge is plausible for us**, which is the same lesson as the
   horse model (TRADING.md: the model forecasts where the price goes).
4. **The approach, in one line [R]:** a dynamic, player-adjusted goals model that prices every market from one score
   matrix; a Benter-style second stage that pools it with the market and is judged by whether its weight is positive; and,
   the part that trades, a model of Betfair's close from the early book, the sharp book (Pinnacle via Parse) and our price,
   modelled on `model/race_book.py`'s closing model, with entries early and exits at the pre-play close on Betfair.
5. **What we can do in weeks with what we hold:** port the horse engine's market-relative features (form windows of
   performance against the closing price, market-implied ratings, rest and congestion, table state) into the existing
   walk-forward; fit the closing model on football-data's Betfair early and closing prices (2024/25 on); score every rule
   on the Betfair books the recorder has kept with their sizes since 2 Nov. **What needs paying for:** line-ups and
   injuries (API-Football Pro, US$19 a month as the repository records it: to re-check), then event data only if the free
   xG test moves the needle (StatsBomb or Opta, enterprise contracts reported above EUR40,000 a year: to re-check).
6. **The gate is unchanged** (`reports/football_model.md` s.4): CLV against Betfair's own close with a 90% lower bound
   above zero over at least 1,000 bets, at takeable prices with a tick at each end and 2% commission, first walk-forward,
   then pre-registered paper. Expect the honest outcome to be "no edge" for most of the stack; the stages are ordered so
   that each costs little before it is proven.

## 0. The constraints the approach must fit

- **Exchange trades only** (the owner, 1 Nov 2026; `reports/open_close/parse_odds.md` item 5): every leg on Betfair or
  another exchange; bookmaker and Asian prices are information, never a venue. This rules out the syndicates' own
  execution (Asian brokers, accounts in other names) and leaves Betfair, Smarkets and Matchbook. Betdaq stopped trading on
  21 Oct 2026, its customers moving to Matchbook (oddschecker, 14 Oct 2026).
- **The proven edge is the move, traded out.** On UK/IE horses the model carries about half the weight in the best
  forecast of the BSP; backing early and closing at BSP returns +12.7% on the tested rule, held to the result none is
  reliably positive (TRADING.md, the staking backtest). Football has no single-price close: the exit is a lay at the
  pre-play close, costing about a tick, 0.8% at 2.5 and 1.4% at 3.5 (`reports/open_close/markets.md` s.5.2).
- **Money and timing on Betfair.** A day ahead, MATCH_ODDS carries about GBP437,775 a day on 185 markets in the top 200,
  ASIAN_HANDICAP GBP175,058, ALT_TOTAL_GOALS GBP50,013, OVER_UNDER_25 GBP34,173, correct score GBP15,789
  (`reports/open_close/all_markets.md` s.2.3). The last hour adds 55-150%: the Premier League about GBP381k at T-60 and
  GBP592k at T-1 a match (betaminic, Jul-Sep 2025, via `all_markets.md`). Below about GBP23,400 matched on a match, the
  bookmaker price is the better forecast (Flepp, Nuesch and Franck 2017, via `all_markets.md` s.3).
- **Costs.** 2% commission on each market's net profit; the Expert Fee at 20% rising to 40% of weekly gross profit once
  the account's 52-week gross profit passes GBP25,000, net of commission paid (`markets.md` s.5.1). Football profit adds
  to the same account total as the live horse rule, so at the margin a football edge pays the fee too [R].
- **What football has already shown here.** A market-blind pooled Dixon-Coles with a shots-on-target mix: 1X2 log-loss
  1.0166 against the open's 1.0017 and the close's 0.9983 on 92,766 matches; the pooled weight on the model about -0.07 in
  every season; model-only value rules CLV about -2.5% on tens of thousands of bets (`football_model.md` s.3). Open to
  close on bookmaker prices with the margin removed: no drift in match odds or over/under; the Asian handicap arm
  withdrawn for look-ahead by selection; one thin positive cell on football-data's Betfair early snapshot (|forecast| of
  0.10 or more: +7.8% development, +3.4% holdout a tick worse at each end, 5-7 bets a day) that may be stale quotes in
  an empty book (`reports/open_close/model.md` s.5).
- **What we now hold that we lacked.** The Betfair football books with their sizes, MATCH_ODDS and OVER_UNDER_25 in ten
  countries, every 15 minutes and every 2 in the last hour, and the evening before (commit f16dcdc, `paper-models.yml`);
  Pinnacle's lines by polling through Parse, with each line's maximum stake (`sources/parse_odds.py`; median soccer
  moneyline maximum 375 in the first snapshot: `parse_odds.md` s.2).

## 1. The organisations: what is known, and how well

Grades: **sourced** (a named document, filing or on-record interview), **reported** (journalism citing unnamed former
staff or industry figures), **reputation** (commentary, blogs, fan sites), **speculation**.

### 1.1 Starlizard (Tony Bloom)

| Claim | Grade | Source |
|---|---|---|
| Founded 2006; a "betting consultancy" selling statistical football odds to clients and placing bets for them and for Bloom's syndicate; about 160 staff; main office in Camden, quants in a separate office in Exeter | reported | Business Insider (Williams-Grut), republished by TheJournal.ie, 14 Feb 2016 |
| Four teams: data generation, odds construction (quants), bet selection (about 20 "selectors" who decide stakes), bet placement; about 30 football researchers in Camden; knowledge deliberately siloed | reported | BI 2016 |
| Researchers watch matches and record chances and shots on target; contacts (journalists, league experts) and stringers in smaller leagues supply morale, form, likely team sheets, training; "the weather, morale, anyone related to the club" | reported | BI 2016 |
| Missed penalties and woodwork are weighted: a 0-0 fed in as about 0.8 goals for the side that missed a penalty | reported | BI 2016 (a former employee's example) |
| The model gives every score line's probability and an Asian handicap line; the algorithm is "tweaked nearly constantly"; the focus is the Asian handicap market | reported | BI 2016 |
| Bets placed through brokers in Asia by phone or messaging, "as close to kick-off as possible" to limit late-news risk; may add in play | reported | BI 2016 |
| At least GBP1m a Premier League match for Bloom and clients; GBP10,000-20,000 in smaller leagues to avoid moving the market | reported | BI 2016 |
| Target return 1-3% on turnover; profits estimated GBP20m-100m a year (the article thinks GBP20m more likely) | reported | BI 2016 |
| Staff get a free stake in the syndicate after probation; personal betting barred after early front-running | reported | BI 2016 |
| Insider: mathematicians, statisticians, trend analysts and bet placers; only three or four see the full data; "at least six figures on every bet"; minimum investor GBP2m | reported | Hawkins, Bleacher Report, 17 Sep 2014 |
| A former employee (Dudfield, 2008-2017) says he decided "when and where" to place bets; alleges "secret exotic accounts" in third parties' names; Bloom admits one proxy account and denies the sum claimed; profits "believed" about GBP600m a year | allegation in a civil claim, as reported | Casino.org, 4 Dec 2025 (High Court filing); Gambling Insider |
| Up to 2023 Starlizard supplied data to Brighton and Union Saint-Gilloise directly, then focused on "core activities" | reputation (search summary of club coverage; not confirmed in a primary document) | the42.ie and others, see 1.3 |

**What it means for us [R].** The edge described is information (watched-match data, team sheets before the market) plus
a score model plus late execution in the deepest market. We have none of the human data. We can buy line-ups (API-Football)
and record the market. The timing point matters: Starlizard bets late, which says the late window has the size; it does
not say the late price is beatable by a model without their information.

### 1.2 Smartodds and Matthew Benham (Brentford, FC Midtjylland)

| Claim | Grade | Source |
|---|---|---|
| Benham (Oxford physics) worked for Bloom at Premier Bet from 2001, made money in the Asian market with a statistical model, fell out with Bloom and founded Smartodds in 2004 | sourced at second hand | Wikipedia citing The Athletic (Naylor and Harris 2021), Bloomberg (2021), The Guardian (2014) |
| The model was developed by Dixon and Coles at Lancaster; Stuart Coles later joined Smartodds | sourced at second hand | Wikipedia citing The Athletic 2021; Dixon and Coles 1997 itself confirmed (JRSS C 46(2)) |
| Smartodds: "in-depth research and analysis on numerous sporting events"; teams for quantitative research, software, data, bet advisory, analysis and "watcher operations"; "Our research and models are only as strong as our data is accurate" | sourced (company site) | smartodds.co.uk, read 2 Nov 2026 |
| Smartodds recruits quants by sport (NHL, cricket, club football); MSc or PhD in statistics preferred, R, Python, C++; back-testing predictive models and training clients on their strengths and weaknesses | sourced (job adverts) | ssc.ca listing; devjobsscanner aggregator |
| Early profits from Asian handicap and over/under, where score-line precision matters | reputation | theesk.org analysis, 2026 |
| Benham part of Triplebet Ltd, owner of the Matchbook exchange, from 2011 | sourced at second hand | Wikipedia citing Racing Post 2020 |
| Brentford's "entire recruitment philosophy is based on analysing statistics based on the Smartodds model"; a "table of team rankings" Phil Giles calls "the true table" | sourced (club) | brentfordfc.com, 23 Mar 2020 |
| Phil Giles, a former Smartodds quant, co-director of football with Rasmus Ankersen from 2015 | sourced at second hand | Wikipedia |
| Benham spoke at the MIT Sloan Sports Analytics Conference, March 2026 | sourced (the article's free part) | Training Ground Guru, 9 Mar 2026 (rest paywalled, not read) |

**What it means for us [R].** The public trace of Smartodds is the Dixon-Coles lineage, watchers, and a "true table"
ranking every side on underlying performance: an expected-goals-style strength measure across leagues. That is the
same family of model our `football/ratings.py` already fits; their advantage is the watched data and the people.

### 1.3 Jamestown Analytics: verified, and what it is not

- **It exists.** "Jamestown were founded in 2017 as an offshoot of Brighton owner Tony Bloom's Starlizard betting and
  analytics operation"; services "player valuations, recruitment targeting, Head Coach recruitment and opposition
  analysis"; clients include Brighton, Union Saint-Gilloise, Como, Castellon, Hearts, IK Start and Grimsby Town
  (Training Ground Guru, 23 Jun 2026, on Swansea's partnership). Hearts called it a "world leading football data analysis
  firm" for recruitment and opposition analysis (heartsfc.co.uk, 20 Nov 2024); Melbourne Victory and Shelbourne announced
  exclusive partnerships; it works with "only one club per league" (the42.ie, 15 May 2026), with set-piece advice by
  opponent and "an enormous database of players".
- **Ownership is unclear.** Sky Sports calls it owned by Bloom; one jobs-board blog names another individual as owner
  (unverified, single source); Hearts named a managing director. I would not rely on any of these.
- **What it is not, on public evidence:** a betting operation. No source I found says it bets, and nothing describes
  its models beyond the services listed. Anything said about Jamestown's betting methods would be invention.
- **Similarly named or likely intended, labelled as such:** Starlizard itself (1.1); Football Radar, a London firm that
  describes itself as a provider of football betting advice combining "skilled analysts" with modelling (founded 2009 and
  about 200 staff per aggregator profiles: unverified).

### 1.4 The Asian handicap syndicates and their execution

- **The market.** About 90% of money is on the Asian handicap and a single Premier League match can draw up to US$120m in
  the Asian market; Pinnacle, IBCBet and SBOBet named among the biggest (Hawkins 2014: reported, figures unverified).
- **Books that balance and books that limit.** Grant, Oikonomidis, Bruce and Johnson (2018, European Journal of Finance)
  split soccer bookmakers into "position-takers" (odds rarely changed, informed bettors restricted) and "book-balancers"
  (odds moved to manage inventory, few restrictions), finding SBOBet changed its odds on 77% of occasions against 12.4%
  for Ladbrokes, and 545 arbitrage portfolios that the position-takers' practices largely suppress. Levitt (2004,
  Economic Journal) is the classic account of why bookmakers set prices rather than clear the market.
- **Price discovery.** Pinnacle's own resources describe posting early lines at reduced limits, letting sharp money move
  them, and raising limits towards the start (Pinnacle, "Market movement in betting" and "Why Pinnacle offers higher
  betting limits": read as search snippets only, the pages render empty to the fetcher). Kaunitz, Zhong and Kreiner (2017)
  used the consensus of many books to find soft outliers; their accounts were restricted within months. Pinnacle's
  public API closed on 23 Jul 2025 (`reports/open_close/data.md` s.6.3).
- **Timing of sharp money.** Starlizard: "as close to kick-off as possible" (BI 2016). Fidens: two scouts per league send
  a predicted eleven and bench about six hours before kick-off, compared with Asian odds through player ratings (Hawkins
  2014). Line-ups: the Premier League requires team sheets 75 minutes before kick-off and from 2024/25 lets clubs publish
  them then; the EFL holds publication to 60 minutes (Premier League Handbook rules L.23 and K.95, EFL regulation 34.5,
  as reported by club and press sites; to re-check against the current handbook).
- **An exchange market-maker.** Alan Thompson, a Poisson-model pro, lays on Betfair as a market-maker with GBP16m a year
  turnover (Hawkins 2014, reported). Whelan (2025), on Betfair's own transaction data for 200,622 soccer matches: pre-play
  makers +0.6% and takers -2.5% on 902,568 match-odds bets; total goals makers +2.0%, takers -2.2%
  (`reports/open_close/markets_sources.md`).

### 1.5 Other templates worth naming

- **Benter (1994; in Hausch, Lo and Ziemba 2008, ch. 19):** a fundamental multinomial logit, then a second-stage logit
  combining it with the public's implied probabilities. The method is confirmed; the figures quoted for it differ between
  secondary accounts (unverified). It is the template for our stage-2 pooling and for `model/benter_blend.py`.
- **Thorp (2006/2008), Kelly in sports betting**, and Uhrin, Sourek, Hubacek and Zelezny (2021, arXiv 2107.08827): on
  horse racing, basketball and football, risk controls are necessary and an adaptive fractional Kelly suits most settings.
- **Hubacek et al.:** train the model to decorrelate from the market rather than only to be accurate (IJF 2019 on NBA;
  Hubacek and Sir, IJF 2023, "Beating the market with a bad predictive model").

### 1.6 What the leaders share [R]

1. A score-line model as the pricing core (all markets from one distribution). 2. Inputs the market does not have yet:
watched-match chance data, scouts' team sheets, people in the clubs. 3. Execution where the money is, timed against the
line's path (late for size). 4. Thin margins (1-3%) at enormous turnover. 5. Secrecy and siloing. For us, (2) and (4) are
the binding gaps; the honest route is to make the market itself the main input and trade its path.

## 2. The state of the art in football modelling: what the evidence says

### 2.1 Goals models

The line runs from Maher (1982: independent Poisson with attack and defence) through Dixon and Coles (1997: the low-score
correction and down-weighting old matches, and an explicit test against the betting market), Rue and Salvesen (2000:
Bayesian dynamic ratings), Karlis and Ntzoufras (2003: bivariate Poisson with a shared covariance term and diagonal
inflation for draws; 2009: Skellam for the goal difference), Baio and Blangiardo (2010: Bayesian hierarchical, with a
mixture to cure over-shrinkage), Owen (2011: dynamic GLM with the evolution variance estimated), Koopman and Lit (2015:
dynamic bivariate Poisson in state-space form; 2019: score-driven models across six leagues) and Boshnakov, Kharrat and
McHale (2017: Weibull inter-arrival counts with a copula, a better fit than Poisson). Ley, Van de Wiele and Van Eetvelde
(2019) compared ten strength models fitted by time-weighted likelihood: independent and bivariate Poisson came out best.

**Evidence against the market.** Each of Koopman and Lit (2015), Boshnakov et al. (2017) and Koopman and Lit (2019)
reports positive betting returns against bookmaker odds out of sample (abstracts; sizes and significance not read).
None that I found is scored against Pinnacle's or Betfair's close. Our Dixon-Coles, with time decay, division pooling and
a shots-on-target mix, is a faithful member of this family and is 0.015 of 1X2 log-loss behind the open
(`football_model.md` s.3). [R] The family is solved as a pricing engine; its marginal versions differ by 0.001-0.003 of
log-loss, an order smaller than the gap to the market.

### 2.2 Rating systems

Elo with ordered logit (Hvattum and Arntzen 2010): worse than two odds-based benchmarks, better than four others.
Pi-ratings (Constantinou and Fenton 2013): beat Elo and were profitable over five Premier League seasons against
published odds. The 2017 Soccer Prediction Challenge (Berrar, Lopes and Dubitzky 2019; 52 leagues, goals only): the
best entries used rating features (recency and rating-feature learning; Hubacek et al.'s gradient-boosted trees on
pi-ratings and PageRank, which they describe as the winning entry; Constantinou's Dolores, hybrid Bayesian networks,
second). The 2023 challenge (Berrar et al. 2024; 736 matches in late April 2023): simple learners with domain knowledge
did best. Bunker, Yeung and Fujii (2024) review the field: on goals-only data, gradient-boosted trees on soccer ratings
such as pi-ratings perform best. **Market-implied ratings:** Wunderlich and Memmert (2018) built an Elo from betting odds
(about 15,000 matches) and it beat result-based Elo, evidence that the odds carry more information than results.

### 2.3 Expected goals and possession value

xG models from event data (Mead, O'Hare and McMenemy 2023: xG a better predictor of future success than traditional
statistics); possession-value models, each valuing every on-ball action: expected threat (Singh 2018, a Markov grid),
VAEP (Decroos, Bransen, Van Haaren and Davis 2019, KDD best applied paper), EPV on tracking data (Fernandez, Bornn and
Cervone 2021), off-ball scoring opportunity (Spearman 2018, MIT Sloan; secondary descriptions only). How they enter
match prediction: through team ratings built on xG or shots instead of goals. Wheatcroft (2020, IJF) showed ratings fed
by shots and shots on target beat goal-based ratings and were profitable on over/under 2.5 across ten leagues (against
bookmaker odds); our own shots-on-target mix bought 0.0008 of log-loss (`football_model.md` s.5). [R] xG is a variance
reduction on the same strength signal, worth a test, not a new source of information the market lacks: the market
watches xG too.

### 2.4 Player-level and line-up models

Plus-minus and its regularised forms (Hvattum 2019 review; Kharrat, McHale and Lopez Pena 2020: xG and expected-points
plus-minus). Arntzen and Hvattum (2021, Statistical Modelling): team Elo and player plus-minus ratings are about equally
good alone and significantly better together. Holmes and McHale (2024, IJF): a model built on the abilities of the
players in each side, tested against the bookmaker on win-draw-loss with a Kelly-type rule, reports significant positive
returns (abstract; the price benchmark, open or close, not read). [R] This is the strand closest to the syndicates' edge:
the market moves on team news, and a player-level model is what prices team news. Its value to us depends on seeing the
confirmed eleven at the moment the market does and pricing it faster or better.

### 2.5 Machine learning against the market

Gradient boosting on engineered rating features is the strongest generic learner on match data (2.2). Deep sequence
models on event data: Seq2Event (Simpson, Beal, Locke and Norman 2022, KDD), large events models (Mendes-Neves, Meireles
and Mendes-Moreira 2024, Machine Learning). These predict the next event, not the price, and none I found is scored
against a betting market. The useful ML idea for betting is decorrelation: profit depends on how the model's errors relate
to the market's, not on accuracy alone (Hubacek, Sourek and Zelezny 2019; Hubacek and Sir 2023).

### 2.6 Combining with the market

Benter's second stage (1.5); Egidi, Pauli and Torelli (2018): a hierarchical Poisson whose rates blend historical data
with the odds, four leagues, some profitable strategies; Wunderlich and Memmert (2018). Our own stage 2 is this layer, and
it hands the price back to the market (weight about -0.07 on the model in 1X2: `football_model.md` s.3).

### 2.7 In-play

Dixon and Robinson (1998): a birth process in which goal rates change with the score and time. Robberechts, Van Haaren
and Davis (2021, KDD): Bayesian in-game win probability. Croxson and Reade (2014, Economic Journal): Betfair prices absorb
goals "swiftly and fully". Angelini, De Angelis and Singleton (2022, IJF): pre-match and in-play mispricing on Betfair
consistent with a reverse favourite-longshot bias, larger after surprising goals. Whelan (2025): as matches progress,
long-shot bets lose heavily even for makers and accepting offers on favourites pays. [R] In-play is the second edge the
literature leaves open, but it is a latency and microstructure business; we have no fast feed, and it is not where our
proven skill is.

### 2.8 Market efficiency and the close

Forrest, Goddard and Simmons (2005): odds-setters matched or beat a statistical model with many variables on about
10,000 English matches. Franck, Verbeek and Nuesch (2010): Betfair more accurate than eight bookmakers on 5,478 big-five
matches, and backing where a bookmaker beats the exchange price returns above average. Angelini and De Angelis (2019): at
the best odds of 41 bookmakers across 11 leagues and about 11 years, eight markets efficient and three not. Hegarty and
Whelan (2025, IJF): the Asian handicap market shows no favourite-longshot bias, lower margins and better probabilities
than 1X2; Whelan and Hegarty (2024): it still misprices refund structures (no-refund lines worst, full-refund best).
Constantinou (2022): the Asian handicap market shares the 1X2 market's inefficiencies over 13 Premier League seasons.
Pinnacle's close: slope of return on price over close "almost exactly 1.00" on 87,960 pairs (Buchdahl, via
`markets_sources.md`).

**Where inefficiency has been found:** new head coaches, underrated in the first four matches after the change
(Bernardo, Ruberti and Verona 2019, four leagues); promoted sides early in the season, temporarily and mainly away
(Deutscher, Frick and Otting 2018, Bundesliga 2002/03-2015/16); the lost home advantage behind closed doors in 2020
(Winkelmann, Deutscher and Otting 2021). Each is a regime change the market learns within weeks. I found no academic study
of the price response to line-up announcements or referee appointments (the searches returned only betting guides).

**What beats the close, and by how much.** On the evidence read: nothing published, at a sharp close. The leaders'
reported margin is 1-3% of turnover (Starlizard, BI 2016), and that is with proprietary data and late execution. Our
horse rule's +12.7% traded out comes from a thin morning market with no sharp book in it (TRADING.md); football's open
already has Pinnacle in it. [R] A realistic target for a football exchange trade is a CLV of 1-3% before commission on a
few hundred bets a month, which after a tick and 2% commission is near zero unless the forecast move is large: the reason
every rule below is thresholded on the size of the forecast move.

## 3. The literature, paper by paper (2018-2026 first, then the classics)

Each entry: method; data; result against the market where reported; relevance to us. "Abstract" means only the abstract
or a repository record was read.

**Goals, score and rating models**

- Koopman, S.J. and Lit, R. (2019) Forecasting football match results in national league competitions using score-driven
  time series models. IJF 35(2), 797-809. Score-driven dynamic models for goal pairs, goal difference and outcome; six
  European leagues; the pairwise-count model forecasts best and is tested in a betting simulation (abstract). Relevance:
  the cheapest upgrade from our static-window Dixon-Coles to a dynamic rating.
- Ley, C., Van de Wiele, T. and Van Eetvelde, H. (2019) Ranking soccer teams on the basis of their current strength.
  Statistical Modelling 19(1), 55-73; arXiv 1705.09575. Ten models by weighted likelihood; independent and bivariate
  Poisson best by RPS and log-loss. Relevance: confirms our model family; no market test.
- Boshnakov, G., Kharrat, T. and McHale, I.G. (2017) A bivariate Weibull count model for forecasting association football
  scores. IJF 33(2), 458-466, doi 10.1016/j.ijforecast.2016.11.006. Weibull counts plus copula; better fit than Poisson;
  positive betting returns (abstract). Relevance: a dispersion check on our score matrix (draws and 0-0s drive AH lines).
- Koopman, S.J. and Lit, R. (2015) A dynamic bivariate Poisson model for analysing and forecasting match results in the
  English Premier League. JRSS A 178(1), 167-186. State-space, importance sampling; out of sample 2010/11-2011/12 with a
  significant positive return over bookmaker odds (abstract). Relevance: as above.
- Owen, A. (2011) Dynamic Bayesian forecasting models of football match outcomes with estimation of the evolution variance
  parameter. IMA JMM 22(2), 99-113. Dynamic GLM; Scottish Premier League 2003/04-2005/06; dynamic beats static.
- Baio, G. and Blangiardo, M. (2010) Bayesian hierarchical model for the prediction of football results. J. Applied
  Statistics 37(2), 253-264, doi 10.1080/02664760802684177. Serie A; a mixture prior to stop over-shrinkage. Relevance: our
  ridge towards the division mean is the frequentist version.
- Karlis, D. and Ntzoufras, I. (2003) Analysis of sports data by using bivariate Poisson models. JRSS D 52(3), 381-393;
  (2009) Bayesian modelling of football outcomes: using the Skellam's distribution for the goal difference. IMA JMM 20(2),
  133-145 (year cited variously as 2008-2009). Relevance: the AH market is priced on the goal difference; Skellam is its
  direct model.
- Dixon, M.J. and Coles, S.G. (1997) Modelling association football scores and inefficiencies in the football betting
  market. JRSS C 46(2), 265-280. The base of our ratings and of Smartodds' lineage.
- Rue, H. and Salvesen, O. (2000) Prediction and retrospective analysis of soccer matches in a league. JRSS D 49(3),
  399-418. Maher, M.J. (1982) Modelling association football scores. Statistica Neerlandica 36(3), 109-118.
- Constantinou, A.C. and Fenton, N.E. (2013) Determining the level of ability of football teams by dynamic ratings based
  on the relative discrepancies in scores between adversaries. JQAS 9(1), 37-50, doi 10.1515/jqas-2012-0036. Pi-ratings;
  beat Elo; profitable over five Premier League seasons against published odds after margin (abstract).
- Hvattum, L.M. and Arntzen, H. (2010) Using ELO ratings for match result prediction in association football. IJF 26(3),
  460-470, doi 10.1016/j.ijforecast.2009.10.002. Elo inputs to ordered logit; behind the odds-based benchmarks.
- Wunderlich, F. and Memmert, D. (2018) The betting odds rating system: using soccer forecasts to forecast soccer. PLOS
  ONE, doi 10.1371/journal.pone.0198668. Elo updated from the odds, about 15,000 matches 2007/08-2016/17; beats result Elo.
  Relevance: the template for market-implied ratings as features (section 6.2, F5).
- Egidi, L., Pauli, F. and Torelli, N. (2018) Combining historical data and bookmakers' odds in modelling football scores.
  Statistical Modelling 18(5-6), 436-459; arXiv 1802.08848. Odds enter the Poisson rates; nine seasons fit, the tenth
  predicted, four leagues; some profitable strategies. Relevance: a principled stage-2 alternative to our logit pooling.
- Wheatcroft, E. (2020) A profitable model for predicting the over/under market in football. IJF 36(3), 916-932. GAP
  ratings from shots and shots on target; ten leagues; profitable on O/U 2.5 against bookmaker odds. Relevance: our
  shots-on-target mix is this idea; the O/U weight on our model is our one positive weight (`football_model.md` s.3).

**Competitions and machine learning**

- Berrar, D., Lopes, P. and Dubitzky, W. (2019) Incorporating domain knowledge in machine learning for soccer outcome
  prediction. Machine Learning 108, 97-126. The 2017 challenge; recency and rating-feature learning; kNN on rating features
  top (the organisers' entry, out of competition).
- Hubacek, O., Sourek, G. and Zelezny, F. (2019) Learning to predict soccer results from relational data with gradient
  boosted trees. Machine Learning 108(1), 29-47, doi 10.1007/s10994-018-5704-6. Pi-ratings and PageRank into boosted
  trees; describes itself as the 2017 winner. Constantinou, A.C. (2019) Dolores. Machine Learning, doi
  10.1007/s10994-018-5703-7: dynamic ratings plus hybrid Bayesian networks, 52 leagues, second.
- Berrar, D., Lopes, P. and Dubitzky, W. (2024) A data- and knowledge-driven framework for developing machine learning
  models to predict soccer match outcomes. Machine Learning 113, 8165-8204, doi 10.1007/s10994-024-06625-9. The 2023
  challenge (736 matches); kNN and neural networks best; domain knowledge decisive. Yeung et al. (2024, Machine Learning,
  doi 10.1007/s10994-024-06608-w): their boosted trees ranked 16th, RPS 0.2195.
- Bunker, R., Yeung, C. and Fujii, K. (2024) Machine learning for soccer match result prediction. arXiv 2403.07669. Review:
  boosted trees (CatBoost) on pi-ratings best on goals-only data; calls for player-level ratings.
- Hubacek, O., Sourek, G. and Zelezny, F. (2019) Exploiting sports-betting market using machine learning. IJF 35(2),
  783-796, doi 10.1016/j.ijforecast.2019.01.001. NBA 2007-2014: decorrelation from bookmaker odds, CNNs on player stats,
  portfolio staking; positive cumulative profit. Hubacek, O. and Sir, G. (2023) Beating the market with a bad predictive
  model. IJF 39(2), 691-719, doi 10.1016/j.ijforecast.2022.02.001. Relevance: the objective for our stage 1 (section 6.3).
- Uhrin, M., Sourek, G., Hubacek, O. and Zelezny, F. (2021) Optimal sports betting strategies in practice: an experimental
  review. arXiv 2107.08827. Adaptive fractional Kelly; risk controls necessary.

**Expected goals, possession value and players**

- Decroos, T., Bransen, L., Van Haaren, J. and Davis, J. (2019) Actions speak louder than goals: valuing player actions in
  soccer. KDD 2019, 1851-1861. VAEP; open source (socceraction). Fernandez, J., Bornn, L. and Cervone, D. (2021) A
  framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions. Machine Learning
  110, 1389-1427, doi 10.1007/s10994-021-05989-6 (tracking data). Singh, K. (2018) Introducing expected threat (xT),
  karun.in blog. Spearman, W. (2018) Beyond expected goals. MIT Sloan SSAC (not read; secondary descriptions).
- Mead, J., O'Hare, A. and McMenemy, P. (2023) Expected goals in football: improving model performance and demonstrating
  value. PLOS ONE, doi 10.1371/journal.pone.0282295.
- Pappalardo, L. et al. (2019) A public data set of spatio-temporal match events in soccer competitions. Scientific Data,
  doi 10.1038/s41597-019-0247-7. One season of seven competitions (Wyscout), non-commercial licence.
- Arntzen, H. and Hvattum, L.M. (2021) Predicting match outcomes in association football using team ratings and player
  ratings. Statistical Modelling 21(5), 449-470, doi 10.1177/1471082X20929881. Holmes, B. and McHale, I.G. (2024)
  Forecasting football match results using a player rating based model. IJF 40(1), 302-312, doi
  10.1016/j.ijforecast.2023.03.002. Kharrat, T., McHale, I.G. and Lopez Pena, J. (2020) Plus-minus player ratings for
  soccer. EJOR 283(2), 726-736. Hvattum, L.M. (2019) A comprehensive review of plus-minus ratings for evaluating individual
  players in team sports. IJCSS 18(1), doi 10.2478/ijcss-2019-0001 (record only). Relevance: section 6.2, F7.
- Simpson, I., Beal, R.J., Locke, D. and Norman, T.J. (2022) Seq2Event. KDD 2022, doi 10.1145/3534678.3539138.
  Mendes-Neves, T., Meireles, L. and Mendes-Moreira, J. (2024) Towards a foundation large events model for soccer. Machine
  Learning 113, 8687-8709, doi 10.1007/s10994-024-06606-y. Relevance: research only; no market test.

**Market efficiency, exchanges and in-play**

- Whelan, K. (2025) Agreeing to disagree: the economics of betting exchanges. MPRA 126351; CEPR DP20633. Betfair soccer
  maker and taker returns (figures in 1.4). Relevance: the cost model for any resting-order rule.
- Hegarty, T. and Whelan, K. (2025) Forecasting soccer matches with betting odds: a tale of two markets. IJF 41(2),
  803-820. Whelan, K. and Hegarty, T. (2024) Returns on complex bets: evidence from Asian handicap betting on soccer.
  Review of Behavioral Finance, doi 10.1108/RBF-11-2023-0314. Relevance: the AH price is the best market probability;
  quarter and half lines are where mispricing sits.
- Constantinou, A.C. (2022) Investigating the efficiency of the Asian handicap football betting market with ratings and
  Bayesian networks. J. Sports Analytics, doi 10.3233/JSA-200588; arXiv 2003.09384.
- Angelini, G. and De Angelis, L. (2019) Efficiency of online football betting markets. IJF 35(2), 712-721, doi
  10.1016/j.ijforecast.2018.07.008. Angelini, G., De Angelis, L. and Singleton, C. (2022) Informational efficiency and
  behaviour within in-play prediction markets. IJF 38(1), 282-299, doi 10.1016/j.ijforecast.2021.05.012.
- Winkelmann, D., Deutscher, C. and Otting, M. (2021) Bookmakers' mispricing of the disappeared home advantage in the German
  Bundesliga after the COVID-19 break. Applied Economics 53(26), 3054-3064, doi 10.1080/00036846.2021.1873234. Deutscher,
  C., Frick, B. and Otting, M. (2018) Betting market inefficiencies are short-lived in German professional football. Applied
  Economics 50(30), 3240-3246, doi 10.1080/00036846.2017.1418082. Bernardo, G., Ruberti, M. and Verona, R. (2019)
  Semi-strong inefficiency in the fixed odds betting market: underestimating the positive impact of head coach replacement.
  QREF 71, 239-246. Relevance: regime-change features (F8, F10).
- Grant, A., Oikonomidis, A., Bruce, A.C. and Johnson, J.E.V. (2018) New entry, strategic diversity and efficiency in soccer
  betting markets. European J. Finance 24(18), 1799-1816, doi 10.1080/1351847X.2018.1443148.
- Kaunitz, L., Zhong, S. and Kreiner, J. (2017) Beating the bookies with their own numbers. arXiv 1710.02824.
- Flepp, R., Nuesch, S. and Franck, E. (2017) The liquidity advantage of the quote-driven market. QREF 64, 306-317.
  Franck, E., Verbeek, E. and Nuesch, S. (2010) Prediction accuracy of different market structures: bookmakers versus a
  betting exchange. IJF 26(3), 448-459, doi 10.1016/j.ijforecast.2010.01.004.
- Forrest, D., Goddard, J. and Simmons, R. (2005) Odds-setters as forecasters: the case of English football. IJF 21(3),
  551-564, doi 10.1016/j.ijforecast.2005.03.003. Strumbelj, E. (2014) On determining probability forecasts from betting
  odds. IJF 30(4), 934-943 (Shin's method beats basic normalisation; secondary sources). Levitt, S.D. (2004) Why are
  gambling markets organised so differently from financial markets? Economic Journal 114(495), 223-246.
- Croxson, K. and Reade, J.J. (2014) Information and efficiency: goal arrival in soccer betting. Economic Journal
  124(575), 62-91, doi 10.1111/ecoj.12033. Dixon, M.J. and Robinson, M.E. (1998) A birth process model for association
  football matches. JRSS D 47(3), 523-538. Robberechts, P., Van Haaren, J. and Davis, J. (2021) A Bayesian approach to
  in-game win probability in soccer. KDD 2021, 3512-3521; arXiv 1906.05029.
- Benter, W. (1994) Computer based horse race handicapping and wagering systems: a report. In Hausch, Lo and Ziemba (eds)
  Efficiency of Racetrack Betting Markets, World Scientific 2008, 183-198. Thorp, E.O. (2006) The Kelly criterion in
  blackjack, sports betting and the stock market. Handbook of Asset and Liability Management vol. 1, ch. 9.

Not found or not confirmed, so not cited as findings: a peer-reviewed study of odds around line-up announcements; any
paper scoring a football model against Pinnacle's or Betfair's close with a positive result; a Journal of Sports Sciences
paper on Pinnacle's close that a vendor page mentions.

## 4. The trading side

### 4.1 The metric

CLV against the venue's own close is the metric, as the repository already uses it (`football_model.md` s.4; TRADING.md).
For football the close is Betfair's last pre-play book, which `football/live.py` records, and from 2 Nov the recorder's
2-minute books in the last hour. Two refinements [R]: score CLV against the **takeable** close (the best lay at the close
for a back, the best back for a lay), not the mid; and report it with the size that was on offer at entry, because a
2024/25 football-data Betfair early price has no time stamp or volume (`model.md` s.5).

### 4.2 Which markets are sharpest, and where inefficiency persists

- AH and totals are cleaner than 1X2 (Hegarty and Whelan 2025): less room for a bias-based rule there, but the AH market is
  where the syndicates price, so its line is the best single estimate of the match [R]. Quarter and half lines carry
  pricing errors (Whelan and Hegarty 2024).
- Lower leagues: the literature expects softer prices where information is thin; our walk-forward found CLV worse down
  the pyramid (tier 5 -4.17%: `football_model.md` s.3), because the closes there move more on news the model lacks. [R]
  Lower leagues are worth it only with the news (line-ups), and Betfair's money there is small.
- Regime changes the market learns within weeks: new coaches (first four matches), promoted sides early (mainly away),
  structural shifts in home advantage (2.8).
- Early lines: Pinnacle opens at low limits and lets sharp money move it (1.4). The early Betfair book is thin and its
  normalised mid is not a price (`model.md` s.5). [R] Early is where a forecast of the path pays, if the book there
  has size.
- Team news windows: 75 minutes (Premier League) or 60 (EFL) before kick-off (1.4). Fixture congestion, weather and
  referees: no market study found; candidate features only.

### 4.3 Timing and execution on an exchange

- Exchange-only means: take at the best back or lay, or rest inside the spread and accept adverse selection (Whelan 2025:
  pre-play makers +0.6% on match odds, +2.0% on total goals; in play, long-shot makers lose). The exit is a lay or back at
  the pre-play close, a tick (0.8% at 2.5, 1.4% at 3.5) plus 2% on the market's net profit (`markets.md` s.5.2).
- Football is one of the exceptions to Betfair's at-the-off in-play turn (Exchange Rules Part B 1.2 as copied in a forum
  post: unverified, via `all_markets.md` s.3), so a pre-play close is well defined.
- Cross-exchange: Smarkets and Matchbook charge about 2% on net winnings (affiliate pages: to re-check). A back on one
  exchange laid on another is legal under the owner's rule; it needs both books recorded first [R].
- Starlizard bets late for size (BI 2016). [R] For a model of the path, the trade is the opposite: enter where the book has
  just enough size and the forecast move is large, exit at the close where the size is.

### 4.4 Staking

- Level stakes or a fixed fraction of the size shown at the price, capped, as the horse rule does (TRADING.md: scale set by
  liquidity, a tenth of what trades). Kelly on the expected CLV and its spread, not on the result, fractional and adaptive
  (Uhrin et al. 2021) [R].
- Correlation: MATCH_ODDS, ASIAN_HANDICAP and OVER_UNDER on one match are functions of the same two goal rates, so they
  are one position [R]. Size the match as a book over the score matrix: `model/race_book.py`'s `build_book` maximises
  expected log growth over mutually exclusive outcomes; the football version's outcomes are score lines and its positions
  are every market's backs and lays [R].
- Limits: the exchange has none, only liquidity; our cap is a share of the displayed size, so fills are real.

## 5. Our horse engine, transferred to football

The horse engine's lesson in one line: market-blind form loses to the market; form measured **against the market**, the
connections' choices **against the price**, and a closing model of **where the price goes** are what carried value
(TRADING.md; `model/ae_features.py`; `model/intent_features.py`; `reports/race_book_model.md`). The greyhound port is the
warning: the same engine ported cleanly (135 lag-safe features, a lag-safety test that caught a real leak), never added to
the BSP (blend weight 0.03-0.13), and its early-market edge of 2021-2023 had gone by 2024 (`reports/greyhound_model.md`
s.3-4, s.11).

| Horse component | Football equivalent | Look-ahead risk and guard | Likely to add beyond the market? [R] |
|---|---|---|---|
| `model/lagsafe.py`: priors lag by whole days, never a row; nothing of the day enters | The same rule on team, manager, referee and league priors: a match day's results never enter any feature of that day; refits end the day before | Two matches of one side can sit in one day only in friendlies; league priors (home advantage) must aggregate to days first. Carry `race_lagged_expanding_mean` over keyed on day | Not an edge; the precondition for trusting any edge |
| The lag-safety test (greyhound: rewrite the last day's results, every feature must be unchanged) | Rewrite the last match day's scores, stats and closing prices; every feature of that day's matches unchanged. Plus the permuted-feature floor `model.md` already uses | Covers features, not joins (`greyhound_model.md` s.4); add a join test on kick-off times and rescheduled matches | Not an edge; mandatory |
| `model/form_windows.py`: every measure over career, last 1, mean of 3 and 5, recency-weighted 3, 5, 10 | The same seven windows over a side's matches, split home and away, for: points, goal difference, shots-on-target difference, xG difference (where held), the market's view `mkt` (the side's de-vigged closing strength), and `ae` | A window counts matches on earlier days only; the closing price of a past match is safe for later matches | The windows of raw form: no (the ratings already hold it). The `ae` windows: possibly |
| `ae` and `nres` measures; `model/ae_features.py` (entity residuals against the BSP, decayed, shrunk) | `ae` = points won less the points the closing price expected; `ae_g` = goals for less the close-implied goal rate (from the AH and O/U closes via the score matrix); entities: team, manager, league, referee (on cards and goals) | Close-implied rates need the closing AH and O/U of past matches: football-data has them for the main leagues (Pinnacle to 2024/25, Betfair after: `football_model.md` s.1). Shrink by effective matches, half-life about a season | **The best candidate.** If the market under-reacts to a side (a promoted side, a new coach), its lagged `ae` predicts the next one. The literature found exactly such short-lived residuals (2.8) |
| `perf` on the official-rating scale; RSR | Market-implied ratings: invert each past match's closes into attack and defence (an odds Elo after Wunderlich and Memmert 2018) and compare with the results-based ratings | As above | A gap between our rating and the market's implied one is the disagreement the stage-2 weight measures; adds only if its sign predicts the close's move |
| `model/intent_features.py`: trainer x angle (first-time headgear, class drop, return from a break) against the price | Manager x angle: rotation before a European midweek, first match after an international break, a new signing's debut, a goalkeeper change, a formation change, the first matches of a new manager; each manager's decayed, shrunk A-E against the close when he does it | Most angles need the confirmed eleven, known only from T-75 or T-60 (1.4); pre-line-up proxies from the fixture list (European match within 3 days) are safe | Possibly, in the line-up window only; the market sees the same sheet within minutes |
| `model/freshness_features.py`: gap against usual, against the yard, against the field; workload in 30/60/90 days | Days since the last match against the side's usual and against the opponent's; matches in the last 14 and 30 days; extra time last match; travel for European away legs; the next fixture's importance | Fixtures are known; a rescheduled match must use the schedule as it stood | Small. The market prices congestion; a test, not a bet |
| Race shape, draw, pace (`race_shape.py`, `draw_curve.py`, `shape_form.py`) | Style match-ups (pressing against build-up, set-piece strength against set-piece weakness) | Needs event data for style measures | Low without paid data; parked |
| `model/race_book.py` `ClosingModel`: conditional logit of the BSP book on the market's price now and ours, with volume interactions; 0.55-0.57 on the market, 0.37 on ours, 14% nearer the close than the morning market (`race_book_model.md`) | **A closing model of Betfair's pre-play book**: softmax over home, draw, away (and over/under) of the closing probabilities on the Betfair early mid, Pinnacle's current de-vigged price, our stage-1 price, each times the book's matched money and depth; the miss's spread by liquidity band is the position's price risk | Early book and Pinnacle line must be the ones recorded at the decision time; football-data's BFE early price has no time stamp | **The core of the trade** (6.3, L3). The horse version worked because the morning market is thin and our price carries half the weight; football's early book has the sharp book in it, so expect a much smaller weight on ours |
| `build_book`, `dutch_book`: a whole race's backs and lays under a value floor and an underlay limit | A whole match's book over MO, AH lines and totals, priced from one score matrix, sized for growth at the close | None beyond L3's | Not an edge; the right way to size correlated positions |
| `model/benter_blend.py`, stage-2 pooling | Already built (`football/model.py stage2`); keep it as the test that a feature set adds | None | It is the referee of every feature family |
| TRADING.md's finding: the value is forecasting the move, traded out | The move from Betfair's early book to its pre-play close; the close beats the open by 0.0034 log-loss (`football_model.md` s.3) | As L3 | The only target with a plausible edge for us |

**Our other models that benefit from the football research [R]:**

- **Greyhounds:** dynamic state-space ratings (Koopman and Lit; the `lib` block already has a Kalman rating, its second
  strongest feature: `greyhound_model.md` s.11); a decorrelation objective (Hubacek and Sir 2023) for a model that loses to
  the SP on log-loss but may still choose the shorteners; the maker-taker cost model (Whelan 2025) for the lay rule at T-3
  and T-1.
- **UK/IE horses:** Shin's de-vig (Strumbelj 2014) for any bookmaker-price feature from the Oddschecker capture; the
  decorrelation loss as an alternative stage-1 objective; the score-matrix book becomes the race book it already is.
- **Japan (JRA) and Hong Kong** (`data/parse/jra_results.parquet`, `hk_results.parquet`): pari-mutuel pools where
  Benter's second stage is the original method, and market-implied ratings from the final odds (each runner's win odds
  are in both feeds: `international_racing_parse.md` s.1). Neither is known to trade on Betfair, so any model there is
  research, not a trade, until the by-country census shows exchange money.

## 6. The approach

### 6.1 Data: what to collect, ranked by value and cost

| Rank | Data | What it gives | Cost and status | Value [R] |
|---|---|---|---|---|
| 1 | **Betfair football books with sizes** (recorder, tag `football`, `football_eve`) | The venue's early book, its path and its close, at takeable prices, three best levels a side, every 2 minutes in the last hour | Free; running since 2 Nov 2026 in `paper-models.yml`. Add ASIAN_HANDICAP and ALT totals, and the traded ladder (`EX_TRADED`) when the projection weight allows (`data.md` s.6.1) | Highest: every rule is scored here |
| 2 | **Betfair Historic Data, Soccer BASIC** | First trade, T-24h, T-60, T-15, T-1, close and result per market from 2016 for the ten countries (`data.md` s.6.3) | Free; the owner "buys" it signed in (outstanding) | High: years of the path, without sizes |
| 3 | **Pinnacle via Parse** (`sources/parse_odds.py --pinnacle`) | The sharp book's line, its version and its maximum stake, polled | Parse Team plan, 20,000 credits; 3 + 2 credits a sport a snapshot (`parse_odds.md` s.1) | High as a feature: the sharp-versus-exchange gap, and the maximum stake as the book's own confidence |
| 4 | **football-data.co.uk** (held) | Results, shots, corners, cards, referee, Pinnacle open and close to 2024/25, Betfair Exchange early and close (BFE, BFEC) | Free, "intended for private individuals only": written permission needed before staking the business's money (`football_model.md` s.6) | High for history; Pinnacle stale from 23 Jul 2025 |
| 5 | **API-Football Pro** | Line-ups, injuries, fixtures and status, referee | US$19 a month, 7,500 requests a day as recorded (`sources/api_football.py`): to re-check | High for F7, F8 and the line-up window; latency unknown (to measure) |
| 6 | Line-up timing rules | When the eleven is public: 75 minutes (Premier League), 60 (EFL), 75 for UEFA match sheets | Free (handbooks) | Sets the window |
| 7 | **Understat** | Shot-level xG, big five plus Russia from 2014/15 | Free, scrape-only and fragile (`football_model.md` s.1); coverage of 2025/26 per scraper vendors only: to re-check | Medium: one test as a ratings input |
| 8 | FBref | Basic history only: advanced (Opta) data removed in January 2026 after Opta ended the agreement (Sports Reference, 20 Jan 2026, via Awful Announcing) | Free | Low now |
| 9 | StatsBomb open data; Wyscout public set (Pappalardo et al. 2019) | Event samples for building xG and VAEP code | Free, non-commercial licences | Development only |
| 10 | **StatsBomb (Hudl) or Opta event data** | Events (and 360) across 40+ leagues; Opta became FIFA's official betting data distributor in January 2026 (as reported) | Enterprise contracts; third-party estimates above EUR40,000-50,000 a year: to re-check | Only if 7 shows xG adds to the market |
| 11 | Betfair Historic ADVANCED/PRO | Volume and every price change historically | Price not readable here (`data.md` s.6.2): to re-check | Medium; the live recorder is the cheaper substitute |
| 12 | Odds APIs (The Odds API, OpticOdds, Sportmonks) | Other books' historic opens and closes | From USD30 a month to quote-only (`data.md` s.6.3): to re-check | Low: information, never the venue |
| 13 | Weather, travel distance | Wind and rain at kick-off; away travel | Free sources exist [unverified] | Low; a candidate for totals only |

### 6.2 Feature engineering: the families

Every family is built in `football/` on the gold tables, day-lagged through `model.lagsafe`, and must pass the rewrite
test before any figure is read. "Decision time" t is when a rule would act (the evening before, T-120, T-60 after the
eleven, T-15).

| Family | Definition | Look-ahead risk and how to avoid it |
|---|---|---|
| F1 Team strength | Dixon-Coles attack and defence by venue, time-decayed (half-life about 300 days now), pooled across a country's divisions, ridge to the division (held: `football/ratings.py`) | Fit window ends before the match day; promoted sides carry their rating through the pool |
| F2 Dynamic strength | Score-driven or Kalman attack and defence (Koopman and Lit 2019), updated after each match, with the evolution variance estimated (Owen 2011) | State at t uses matches finished before the day |
| F3 Rating set | Pi-ratings, Elo, goal-supremacy ratings (Constantinou and Fenton 2013; Hvattum and Arntzen 2010) | Cheap; same guard |
| F4 Chances | Shots, shots on target and, where held, xG for and against, each shrunk to the division mean by matches played; the GAP construction (Wheatcroft 2020) | xG providers revise data: store an as-of snapshot; football-data stats arrive after the match only |
| F5 Market-implied strength | Each past match's de-vigged closing 1X2, AH and O/U inverted to two goal rates; a market-implied attack and defence per side, windowed; the gap between F1 and F5 | Past matches' closes only, earlier days; Shin's method for bookmaker prices, normalisation for Betfair |
| F6 Performance against the market | `ae` = points less the close's expected points; `ae_g` = goals less the close-implied rate; over the seven horse windows, home and away; entity versions for manager, league, referee | Same as F5; shrink by effective matches (the horse `INTENT_AE_K` idea) |
| F7 Availability-adjusted strength | Player ratings (regularised plus-minus on xG or goals, Kharrat et al. 2020; or minutes-weighted contributions) aggregated over the expected eleven before t and the confirmed eleven after it; the delta between the two; missing starters' share of minutes | Injury and suspension lists must be time-stamped as fetched; the confirmed eleven exists only after the team-sheet time; a lineup fetched after kick-off must never feed a pre-match row |
| F8 Selection and intent | Rotation index (starters changed from the last match, share of the strongest eleven fielded); new manager match count (Bernardo et al.: first four); debutant count; goalkeeper change; manager x angle A-E against the close | As F7 |
| F9 Schedule and rest | Days since the last match against usual and against the opponent; matches in 14 and 30 days; extra time; European away travel; days to the next match and its importance | Schedule as known at t; rescheduled matches flagged |
| F10 State of the table | Points to the title, Europe and safety with matches left; dead rubbers; promoted side and matches into the season (Deutscher et al. 2018); derby flag | Table before the match day; same-day results excluded |
| F11 Market state at t | Betfair best back, lay and size at three levels, mid, spread, depth imbalance (back size less lay size over the sum), matched money, the move since the first book; Pinnacle's de-vigged line, its version count and maximum stake; the Pinnacle-minus-Betfair gap in log odds; cross-market consistency (the goal rates implied by MO, AH and O/U disagreeing); hours to kick-off | Only books recorded at or before t; football-data's BFE early price has no time stamp and is not a decision-time price (`model.md` s.5); Pinnacle's polls are timed by the fetch, not the change |
| F12 Context | Referee (cards and penalties history, A-E on cards and goals), kick-off slot, weather [unverified source], artificial pitch | Referee appointments known days ahead in England [unverified]; time-stamp them |

### 6.3 Modelling: the stack, validation and benchmarks

| Layer | What | Fitted to | Benchmark it must beat | Pass mark [R] |
|---|---|---|---|---|
| L1 Goals model | Dynamic bivariate goal rates (F1-F4, F7, F9, F10) into a Dixon-Coles score matrix; every market priced from it: 1X2, every AH line (quarter lines split), totals, correct score | Results | Our current model (1X2 1.0166, O/U 0.6838) and then the Betfair or Pinnacle open | Close a third of the 0.015 gap to the open before L2 is refitted; else L1 changes stop |
| L1-D Decorrelated variant | The same with the decorrelation term (Hubacek and Sir 2023) against the open | Results and the open | Its own pooled weight | A positive pooled weight, 90% interval above zero, where L1's is negative |
| L2 Market combination | Benter second stage: conditional logit on the de-vigged open (Shin), L1, F5, F6, F8 and F10, with interactions by matched money | Results | The open alone and the close | Weight on the non-market terms positive with the interval above zero; log-loss gain over the open on the holdout |
| L3 Closing model (the trade) | Softmax over each market's outcomes of Betfair's pre-play close, on the Betfair mid at t, Pinnacle at t, L2, F11 and the line-up state, each times depth (the `ClosingModel` form); a spread of the miss by liquidity band | Betfair closes (football-data BFEC from 2024/25; the recorder from 2 Nov 2026; BASIC files once held) | No change; the permuted-feature floor; Pinnacle-only and Betfair-only versions | R2 against no change above the permuted floor in development and holdout; hedged ROI positive a tick worse at each end with the interval above zero |
| L4 Line-up event | At the team sheet: the F7 delta between expected and confirmed elevens, priced through L1, against the Betfair move over the next minutes | The recorder's 2-minute books around T-75 and T-60 | The market's own move in the same minutes | The forecast move has the right sign more than chance after a tick, on at least 300 matches |

**Validation [R], in the repository's existing form:** weekly walk-forward refits; a development window and a holdout
read once; log-loss and calibration by decile for L1 and L2; CLV at takeable prices for L3 and L4; the lag-safety
rewrite test and the permuted-feature floor on every run; thresholds chosen in development only and pre-registered before
paper (as `reports/preregistration_clv_forward.md` does for horses); every threshold reported, none chosen on the holdout.

**The "sharp book leads the exchange" test fits at L3.** The modelling agent's run on football-data's Betfair early and
closing prices with Pinnacle features (`reports/open_close/model_football_exchange.md`, when it lands) is L3 with F11's
Pinnacle terms and no stage-1 model. Its result decides the order of work: if the Pinnacle gap predicts Betfair's close
beyond the permuted floor, L3 is built on it and our model's contribution is measured as the increment over that; if
not, L3's remaining hope is our own price and the line-up state, and the stage-2 evidence says that hope is small.

### 6.4 Trading: markets, timings, rule shapes, the gate

- **Markets:** MATCH_ODDS first (the money and the narrowest tick at 2-3.5), ASIAN_HANDICAP second (two-way, about 40%
  of match-odds money, each line its own market), OVER_UNDER_25 and alternative totals third. Correct score, corners,
  cards and player markets parked (5-10 matches a day, thin: `all_markets.md` s.2.3).
- **Timings:** the evening before and the morning (the recorder's `football_eve` books) to the pre-play close; and the
  team-sheet window to the close. In play excluded until a pre-play rule has passed [R].
- **Rule shapes to pre-register for paper [R]:**
  1. *Early path rule (the horse analogue):* at the first book with at least GBP X on the side within one tick of the best
     price, back (or lay) where L3's forecast move towards our side is at least k ticks, stake the smaller of a fixed GBP
     amount and a tenth of the size shown; green at the pre-play close by the opposite order at the best price.
  2. *Sharp-gap rule:* where Pinnacle's de-vigged price has moved beyond Betfair's mid by more than the round-trip cost and
     Betfair's book has not followed, take Betfair; green at the close. (Live only if the other agent's test passes.)
  3. *Line-up rule:* within m minutes of the team sheet, where L4's price moves by at least k ticks against the book, take
     the book; green at the close.
  4. *Resting rule:* one tick inside the spread on L3's side from T-120, cancelled at T-5, greened at the close; scored with
     Whelan's maker returns as the prior and the adverse selection measured on our own fills.
- **The gate (unchanged):** CLV against Betfair's own pre-play close with a 90% lower bound above zero over at least 1,000
  bets, at takeable prices with a tick at each end and 2% commission, first in the walk-forward and then on paper with the
  rule frozen; fills limited to the size shown; then money at the smallest stake that meets Betfair's GBP2 minimum, with a
  match and day cap, inside the existing STOP switch (`football/live.py`).

### 6.5 The staged plan: what each stage must prove before the next

| Stage | When | Data | Build | Must prove before the next |
|---|---|---|---|---|
| 0 | Weeks 1-3 | Held: gold tables, football-data closes, the recorder from 2 Nov | Port F5, F6, F8 (fixture-list proxies), F9, F10 and the lag-safety rewrite test into `football/model.py`; add F2 and F3 | A positive stage-2 weight with its interval above zero on 1X2 or O/U, in development and on the holdout read once. If not, stop L1/L2 work; go to stage 1 regardless |
| 1 | Weeks 2-8 | football-data BFE/BFEC 2024/25 on; Pinnacle via Parse; the recorder | L3 on football-data (after the other agent's test), then re-scored on the recorder's takeable books | L3 above the permuted floor in development and holdout; rule 1 or 2 hedged-positive a tick worse with the interval above zero; the recorded books confirm the football-data early prices could have been taken (the greyhound lesson) |
| 2 | From month 2 | The Soccer BASIC files (owner's action); API-Football Pro (US$19 a month, to re-check) | Backfill L3 to 2016; F7 and L4; the line-up window on the recorder's 2-minute books; measure API-Football's line-up latency against the book's move | L4's sign right beyond chance after a tick on at least 300 matches, with the feed's latency inside the move |
| 3 | Only if stage 0 or 2 shows the model adds | Understat (free) first; then quotes from StatsBomb and Opta | xG into F4 and player ratings into F7 | A gain in the stage-2 weight or in L3's R2 worth more than the data's cost at the scale the book allows |
| 4 | When a rule passes 1 or 2 | The recorder | Pre-registration; paper trader with the frozen rule | The gate, over at least 1,000 paper bets |
| 5 | After 4 | Live key | Money at minimum stakes, caps, the STOP switch | Live CLV inside the paper interval for a month before any increase |

Scale check [R]: at 5-7 qualifying bets a day (the size of the one positive cell in `model.md` s.5), 1,000 paper bets
take about five to seven months (143-200 days). A rule cannot be proven faster than the market offers bets.

## 7. What I could not establish, and what would change the plan

- The details of Starlizard's and Smartodds' models are not public; the most specific account is a 2016 article from
  former staff. The court papers are reported allegations. Jamestown's methods are not public at all.
- How fast the Betfair price absorbs a team sheet, and whether API-Football's line-up arrives before that: no study found;
  stage 2 measures it.
- Exact current prices for event data and Betfair's ADVANCED/PRO plans: all "to re-check" before any purchase.
- A pass at L3 on the recorded books would make football the second market with an exchange-only path edge; a failure
  at L3 with the Pinnacle features would leave line-ups (stage 2) as the only route, which is the syndicates' own edge
  bought at retail speed [R].

## Sources

Repository: `TRADING.md` (lines 1-120); `reports/football_model.md`; `reports/open_close/model.md` (s.5);
`reports/open_close/all_markets.md`; `reports/open_close/data.md` (s.6); `reports/open_close/markets.md` (s.5);
`reports/open_close/markets_sources.md`; `reports/open_close/parse_odds.md`; `reports/race_book_model.md`;
`reports/greyhound_model.md`; `reports/international_racing_parse.md`; `model/lagsafe.py`, `model/form_windows.py`,
`model/intent_features.py`, `model/freshness_features.py`, `model/ae_features.py`, `model/race_book.py`,
`model/custom_metrics.py`; `football/ratings.py`, `football/model.py`; `sources/football.py`, `sources/api_football.py`,
`sources/parse_odds.py`; commit f16dcdc (the football book record).

Organisations and the market:

- Williams-Grut, O. (2016) The inside story on Britain's greatest sports gambler. Business Insider, republished by
  TheJournal.ie, 14 Feb 2016: https://www.thejournal.ie/tony-bloom-starlizard-2597458-Feb2016/
- Hawkins, E. (2014) Mugs and millionaires: inside the murky world of professional football gambling. Bleacher Report,
  17 Sep 2014: https://bleacherreport.com/articles/2200795-mugs-and-millionaires-inside-the-murky-world-of-professional-football-gambling
- Casino.org (2025) Inside StarLizard: High Court filing: https://www.casino.org/news/inside-starlizard-high-court-filing-lifts-lid-on-tony-blooms-800m-betting-empire/
- Gambling Insider, Bloom denies claim: https://www.gamblinginsider.com/news/105962/tony-bloom-denies-17-5m-betting-syndicate-claim
- Wikipedia, Matthew Benham (secondary; its cited sources listed in the notes file): https://en.wikipedia.org/wiki/Matthew_Benham
- Smartodds: https://www.smartodds.co.uk/ ; job advert: https://ssc.ca/en/node/1224
- Brentford FC (2020) Bees feature in France Football magazine: https://www.brentfordfc.com/en/news/bees-feature-in-france-football-magazine-479997
- Training Ground Guru (2026) Matthew Benham (9 Mar): https://trainingground.guru/matthew-benham-how-data-culture-and-smart-bets-built-brentfords-success/
- Training Ground Guru (2026) Swansea pull off Jamestown Analytics coup (23 Jun): https://trainingground.guru/swansea-pull-off-jamestown-analytics-coup-amid-recruitment-revamp/
- Heart of Midlothian FC (2024) Hearts partner with Jamestown Analytics: https://www.heartsfc.co.uk/blogs/news/hearts-partner-with-jamestown-analytics
- the42.ie (2026), Shelbourne and Jamestown: https://www.the42.ie/article.php?id=7039544
- Melbourne Victory: https://melbournevictory.com.au/news/melbourne-victory-partners-with-globally-renowned-football-analytics-company-jamestown-analytics/
- Sky Sports on Hearts and Jamestown: https://www.skysports.com/football/news/11741/13256698
- jobsinfootball blog (single-source ownership claim, unverified): https://jobsinfootball.com/blog/the-impact-of-jamestown-analytics-on-world-football/
- Football Radar profile (aggregator): https://magnet.me/en/company/football-radar
- Pinnacle, Market movement in betting (snippet only): https://www.pinnacle.com/betting-resources/en/betting-strategy/market-movement-in-betting/4732xnzxpqpzrff5
- Pinnacle, Why Pinnacle offers higher betting limits (snippet only): https://www.pinnacle.com/betting-resources/en/educational/why-pinnacle-offers-higher-betting-limits-than-other-sportsbooks
- oddschecker (2026) Matchbook completes Betdaq takeover, 14 Oct: https://www.oddschecker.com/insight/industry/20261014-matchbook-completes-betdaq-takeover-to-create-unified-exchange-platform
- Premier League line-up rule (as reported): https://sg.news.yahoo.com/premier-league-rule-change-allows-161439114.html ;
  https://liverpoolfc.com/news/explained-premier-league-changes-2024-25-season
- Awful Announcing (2026) Sports Reference pulls advanced data: https://awfulannouncing.com/soccer/sports-reference-pulls-advanced-data-agreement-violation-dispute.html
- Hudl StatsBomb product page: https://hudl.com/products/statsbomb ; third-party price estimates: https://footballanalytics.ai/european-football-technology/best-european-football-data-providers
- Buchdahl, J., The efficiency of the Pinnacle closing line: https://football-data.co.uk/blog/pinnacle_efficiency.php

Papers (DOIs and links in section 3; repository records consulted):

- Koopman and Lit 2015: https://tinbergen.nl/discussion-paper/3773/12-099-iii-a-dynamic-bivariate-poisson-model-for-analysing-and-forecasting-match-results-in-the-english-premier-league
- Koopman and Lit 2019: https://research.vu.nl/en/publications/forecasting-football-match-results-in-national-league-competition-2/
- Boshnakov et al. 2017: https://ideas.repec.org/a/eee/intfor/v33y2017i2p458-466.html
- Ley et al. 2019: https://statmod.org/smij/Vol19/Iss1/Ley/Abstract.html
- Owen 2011: https://pureportal.coventry.ac.uk/en/publications/dynamic-bayesian-forecasting-models-of-football-match-outcomes-wi/
- Baio and Blangiardo 2010: https://discovery.ucl.ac.uk/id/eprint/16040/
- Karlis and Ntzoufras 2003: https://ftp.math.utah.edu/pub/tex/bib/idx/jrss-d-2000/52/3/381-393.html
- Dixon and Coles 1997: https://eprints.lancs.ac.uk/id/eprint/19492
- Rue and Salvesen 2000: https://ftp.math.utah.edu/pub/tex/bib/idx/jrss-d-2000/49/3/399-418.html ; Maher 1982:
  https://ideas.repec.org/a/bla/stanee/v36y1982i3p109-118.html
- Constantinou and Fenton 2013: https://www.degruyterbrill.com/document/doi/10.1515/jqas-2012-0036/html
- Hvattum and Arntzen 2010: https://www.bibsonomy.org/bibtex/a0ef8f0c642f671fd027607d2b69ac6a
- Wunderlich and Memmert 2018: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5988281/
- Egidi et al. 2018: https://arxiv.org/abs/1802.08848
- Wheatcroft 2020: https://researchonline.lse.ac.uk/id/eprint/103712
- Berrar et al. 2019: https://mlanthology.org/mlj/2019/berrar2019mlj-incorporating ; Hubacek et al. 2019 (MLJ):
  https://mlanthology.org/mlj/2019/hubacek2019mlj-learning ; Constantinou 2019: https://mlanthology.org/mlj/2019/constantinou2019mlj-dolores
- Berrar et al. 2024: https://link.springer.com/article/10.1007/s10994-024-06625-9 ; Yeung et al. 2024:
  https://mlanthology.org/mlj/2024/yeung2024mlj-evaluating
- Bunker, Yeung and Fujii 2024: https://arxiv.org/abs/2403.07669
- Hubacek et al. 2019 (IJF): https://ideas.repec.org/a/eee/intfor/v35y2019i2p783-796.html ; Hubacek and Sir 2023:
  https://ideas.repec.org/a/eee/intfor/v39y2023i2p691-719.html
- Uhrin et al. 2021: https://arxiv.org/abs/2107.08827
- Decroos et al. 2019: https://www.ijcai.org/proceedings/2020/648 ; Fernandez et al. 2021: https://arxiv.org/abs/2011.09426 ;
  Singh 2018: https://karun.in/blog/expected-threat.html ; Mead et al. 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10075453/ ;
  Pappalardo et al. 2019: https://pmc.ncbi.nlm.nih.gov/articles/PMC6817871
- Arntzen and Hvattum 2021: https://statmod.org/smij/Vol21/Iss5/Arntzen/Abstract.html ; Holmes and McHale 2024:
  https://livrepository.liverpool.ac.uk/id/eprint/3170651 ; Kharrat et al. 2020: https://livrepository.liverpool.ac.uk/id/eprint/3063562 ;
  Hvattum 2019: https://reference-global.com/article/10.2478/ijcss-2019-0001
- Simpson et al. 2022: https://eprints.soton.ac.uk/458099 ; Mendes-Neves et al. 2024: https://mlanthology.org/mlj/2024/mendesneves2024mlj-foundation
- Whelan 2025: https://mpra.ub.uni-muenchen.de/126351 ; Hegarty and Whelan 2025: https://ideas.repec.org/p/cpr/ceprdp/17949.html ;
  Whelan and Hegarty 2024: https://www.emerald.com/insight/content/doi/10.1108/RBF-11-2023-0314/full/html
- Constantinou 2022: https://arxiv.org/abs/2003.09384
- Angelini and De Angelis 2019: https://ideas.repec.org:443/a/eee/intfor/v35y2019i2p712-721.html ; Angelini et al. 2022:
  https://ideas.repec.org:443/a/eee/intfor/v38y2022i1p282-299.html
- Winkelmann et al. 2021: https://ideas.repec.org/a/taf/applec/v53y2021i26p3054-3064.html ; Deutscher et al. 2018:
  https://ideas.repec.org/a/taf/applec/v50y2018i30p3240-3246.html ; Bernardo et al. 2019:
  https://ideas.repec.org/a/eee/quaeco/v71y2019icp239-246.html
- Grant et al. 2018: https://eprints.nottingham.ac.uk/id/eprint/49935
- Kaunitz et al. 2017: https://arxiv.org/abs/1710.02824
- Flepp et al. 2017: https://ideas.repec.org/p/zrh/wpaper/342.html ; Franck et al. 2010: https://ideas.repec.org:443/a/eee/intfor/v26yi3p448-459.html
- Forrest et al. 2005: https://salford-repository.worktribe.com/output/1472154/odd-setters-as-forecasters-the-case-of-english-football ;
  Strumbelj 2014 (secondary): https://pypi.org/project/shin ; Levitt 2004: https://ideas.repec.org/a/ecj/econjl/v114y2004i495p223-246.html
- Croxson and Reade 2014: https://centaur.reading.ac.uk/34884 ; Dixon and Robinson 1998:
  https://www.ma.imperial.ac.uk/~ejm/M3S4/Problems/football.pdf ; Robberechts et al. 2021: https://arxiv.org/abs/1906.05029
- Benter 1994/2008: https://ideas.repec.org/h/wsi/wschap/9789812819192_0019.html ; Thorp 2006:
  https://www.Gwern.net/doc/statistics/decision/2006-thorp.pdf
