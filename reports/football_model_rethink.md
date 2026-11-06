# racing² football: rethinking the model before it trades

4 Nov 2026. For the owner, before the football model goes live: what to change fundamentally, and why.

**Sources.** Our own evidence: `reports/football_runs/model_diagnostics.json`, `stack_betfair_read.json`, `improve.json`,
`market_scan.json`. Ted Knutson's lessons: his own published pieces, linked below.

## What our evidence says

1. **Results are priced efficiently.**
   - Our Dixon-Coles team model is worse than Betfair's price: its value bets lose more than backing blind.
   - xG adds to forecasting the move, not to forecasting the result.
2. **The price's path is partly forecastable.**
   - The stacked team-feature model explains about 4% of the early-to-close move on Betfair 2025/26.
   - Its strongest calls are right 68-74% of the time.
   - The edge sits only in the strongest 1-2% of signals.
3. **What the model leans on** (gain share, home move):

   | Feature | Share |
   |---|---|
   | The league | 20% |
   | Our price against the early price (`dis_open`) | 12% |
   | Shot-on-target difference, last 5 | 10% |
   | Market-implied team strength | 5% |
   | Points against the market's expectation | 5% each window |

4. **Where the signal works** (signals above 0.03, Betfair 2025/26):

   | Tier | Bets | Right way | Sym | Tick |
   |---|---|---|---|---|
   | Top five leagues | 324 | 56% | -1.3% | -4.5% |
   | Other top flights and second tiers | 802 | 67% | +1.1% | -0.1% |
   | Lower leagues | 185 | 70% | -0.2% | -0.2% |

   - **The top five leagues are where the model is weakest.**
   - Lay-first trades (the price drifts) beat back-first: 66% right way against 61%.
5. **A model trained on Betfair's own two seasons fits better and trades worse.** It learns the early book's noise.
   Football-data holds only two Betfair snapshots a match, and only from 2024/25.

## What Ted Knutson's published lessons add

- **Combine the models with football knowledge.** The edge has moved to people who understand "what the models might
  not know" and play "the holes the robots leave behind". xG alone is now used by "pretty much every professional
  gambler, bookie and team", so it is the baseline, not the edge. ([Fitzdares, "Modelling Citizens"](https://members.fitzdares.com/times/articles/modelling-citizens-ted-knutson/))
- **Home advantage moves, and models built on its history suffer.** Premier League home goal difference fell from
  0.41 (2022-23) to 0.09 (2024-25), while the Championship's stayed strong.
- **Priors are a weak point.** A model that "grossly overrate[d] the promoted teams and never really adjusted" lost
  over a season.
- **Bet selectively.** His Championship bets returned +6.2% (194 bets) and his Premier League bets -1.6% (145). A
  test that bet every match lost. ([The Transfer Flow, 22 Jul 2025](https://www.thetransferflow.com/p/learning-how-to-get-better-at-betting-on-football-2144))
- **Background.** He designed betting products at Pinnacle, then was head of player analytics at Smartodds, working on
  Brentford and Midtjylland. ([The Power Rank podcast](https://thepowerrank.com/2022/12/09/podcast-ted-knutson-on-betting-soccer-and-american-football/), [Unofficial Partner](https://unofficialpartner.substack.com/p/the-ted-knutson-question-doing-a))

**Not found:** any published detail of Smartodds' model design. Nothing here assumes it.

Knutson's lessons are about beating the result's price. Ours is a price-path model on an exchange. What carries over is
his view of where models are blind (home advantage drift, promoted sides, early season, team news) and how to
select (where the market is softer, only when the edge is clear).

## The plan: seven fundamental changes, in order

| # | Change | Why | Data | Effort |
|---|---|---|---|---|
| 1 | **Get Betfair's own price history: Historic Data, BASIC soccer** (free; the last traded price each minute, every soccer market, years back) | Two snapshots for two seasons is too little to learn Betfair's price path. With minute paths we can model the move from any moment, choose the entry time, and train on Betfair, not Pinnacle | Owner: request the BASIC soccer plan on the Betfair account; `betfair_historic.py` is already built | Owner: minutes. Us: a day to load and rebuild the move target |
| 2 | **Model what the market does not know yet: team news** | Line-up features take the move's R2 from about 5% to 9%: team news is the largest single driver of the move. Forecast it before it is public (injuries, suspensions, rotation after midweek games, congestion, a new manager) and trade before the sheets | API-Football injuries and line-ups (held from 2020); fixture congestion from all competitions | 1-2 weeks |
| 3 | **Fix the blind spots Knutson names** | Home advantage drifting by league and season; promoted and relegated sides; early season | Results we hold | Days: blocks in the lab, lag-tested |
| 4 | **Player-based team strength** | Today's ratings are team-level. A side is its available players: strength from players' minutes and xG contribution, recomputed for the expected XI | API-Football players and xG (held from 2023 for top flights, 2025 more) | 1-2 weeks |
| 5 | **Train for the decision we make** | We pick the strongest few signals, so train to rank (a learning-to-rank or top-k objective), with the cost of trading in the label. Fit per tier: the league already carries 20% of the model | The lab matrix | Days |
| 6 | **Concentrate where the market is softer** | Our signal holds in second tiers and other top flights, not the top five. Knutson's profits came from the Championship and Champions League, not the Premier League | Already held | Hours: a tier filter, pre-registered |
| 7 | **Order-book features from the live feed** | Spread, depth, money flow and price velocity: what moves next, and what it costs to trade | The football runner (`football/feed_features.py` is ready) | Weeks of recording first |

**What stays:**
- the discipline: lag tests, walk-forward, every variant in the ledger, a pre-registration before forward data;
- the fast lab;
- the order: no real money until a rule passes its gate on the recorder's books.

**Suggested start:**
- **Now:** 3 and 6 on data we hold.
- **This week:** 2 (team news).
- **Two owner actions** unlock the rest, 1 and 7:
  - request Betfair Historic Data BASIC for soccer;
  - register the football runner.

## Results: items 2-6 (4 Nov 2026)

**Sources.** `research/open_close/football_improve.py --rethink`; figures from `reports/football_runs/rethink.json`.

**What was built (every new feature lag-tested):**
- **Item 3.** `football/blocks/home_adv.py`: the competition's rolling home advantage and its drift, promoted and
  relegated sides, the early season.
- **Items 2 and 4.** `team_news` in `football/player_ratings.py`. From each side's matches before the early price:
  - the expected eleven's rating;
  - rating-weighted regulars missing from the latest match;
  - players sent off;
  - rotation.
  It covers 61% of matches from 2020, wherever API-Football's line-ups link.
- **Item 5.** Cost-aware direction classifiers (P(the move clears the cost) up minus down), and stacks fitted apart for
  the top five leagues and the rest.
- **Item 6.** The candidates outside the top five leagues.

**How it was scored.** Betfair match odds, tuned on 2024/25 and read on 2025/26 to date (read by earlier rules:
development evidence). About 100 variants have now been read on 2025/26.

**The strongest 5 forecasts a week on 2025/26** (285 bets each):

| Features | Model | All leagues: tick / sym | Outside the top five: tick / sym | Right way (outside) |
|---|---|---|---|---|
| Team features (base) | Stack | +1.20% [-0.3] / +3.64% [+2.1] | +1.53% [+0.1] / +3.35% [+1.9] | 75% |
| + home advantage | Stack | +1.11% / +3.51% | +2.10% [+0.6] / +3.88% [+2.4] | 76% |
| + team news | Stack | +1.42% [-0.0] / +3.86% [+2.5] | +1.91% [+0.4] / +3.85% [+2.4] | 77% |
| **+ both** | **Stack** | **+1.97% [+0.5] / +4.24% [+2.8]** | **+2.64% [+1.1] / +4.36% [+2.8]** | **77%** |
| + both | Cost classifiers | +1.23% / +2.02% | +0.93% / +1.57% | 73% |
| + both | Stack per tier | +1.70% [+0.0] / +4.00% | +2.37% [+0.7] / +4.34% | 78% |

(Brackets: the 90% lower bound. "tick" = a tick off fair at each end; "sym" = a book symmetric about fair.)

**The same rule on 2024/25** (both blocks, stack, outside the top five, top 5 a week): +5.33% a tick off fair
[+3.4], +7.26% at a symmetric book [+5.2], 75% right way. At top 10 a week on 2025/26 it is +0.63% / +2.14%.

**Read:**
- **Each block helps a little, and the two together most.** The rule is now above zero a tick off fair in both
  seasons, with the lower bound above zero. The base model was not.
- **The softer leagues help, as Knutson's record suggests.**
- **Training for the decision as built here does not help.** The cost classifiers fall short of the stack, and per-tier
  fits match it without beating it. A learning-to-rank objective is untried.
- **Caveat.** With about 100 variants read on 2025/26, part of the edge over the base is selection. The rule
  must be pre-registered and judged forward on the recorder's books before any money goes on it.

**Candidate for pre-registration:**
- the stack on team features plus `home_adv` plus `team_news`;
- its strongest 5 forecasts a week outside the top five leagues;
- back first where the chance is forecast up, lay first where down;
- greened at the close.

## Results: fatigue, short form, schedule-adjusted xG, territory (4 Nov 2026)

**Sources.** `research/open_close/football_improve.py --fatigue`; figures from
`reports/football_runs/rethink_fatigue.json`. The same harness, seasons and costing as above; each feature group is
added on top of the candidate's features (team features + `home_adv` + `team_news`, "both"). The "both" row reproduces
the figures above exactly.

**What was built (every new feature lag-tested):**
- `football/blocks/fatigue.py` (`fat_`): league matches in the last 7 and 10 days, turnarounds of 3 days or fewer,
  days to the next match. With `rest_days` (days since the last match, matches in 14 and 28 days).
- `team_news` minutes load (`ld_`): the expected eleven's league minutes in the 7 and 14 days before the early price.
  61% of 2024/25 and 2025/26 matches.
- `football/blocks/form_short.py` (`fs_`): the last 2 and 3 matches, and the last 5 and 10 weighted linearly by
  recency, of points, goal difference, shots on target, points and goals against the closing market (form adjusted
  for opponent and venue), the same against our Dixon-Coles prices, and the opponents' strength (schedule).
- xG stage: xG against what each opponent usually concedes and makes (`x_sos`), short xG windows (`x_short`), and
  territory proxies (`x_territory`: possession, pass completion, corner share, share of shots inside the box, shot
  share). 28% (2024/25) and 38% (2025/26) of matches have the schedule-adjusted xG; 52% the territory measures.

**No data found:**
- **Player and squad age:** API-Football's history we hold has no birth dates. A players fetch would add them; it
  costs API quota.
- **xThreat and other location-based metrics:** no event locations. The territory measures are proxies.
- **Cup and European congestion:** our fixture history is the 22 league competitions only.

**Outside the top five leagues, a tick off fair / symmetric book [90% lower bound], right way:**

| Features | 2024/25 top 5 a week | 2024/25 top 10 a week | 2025/26 top 5 a week | 2025/26 top 10 a week |
|---|---|---|---|---|
| Both (candidate) | +5.33% [+3.4] / +7.26% | +2.55% [+1.2] / +4.20% | +2.64% [+1.1] / +4.36%, 77% | +0.63% [-0.4] / +2.14% |
| + fatigue | +5.08% / +6.95% | +2.56% / +4.22% | +2.34% [+0.7] / +4.20%, 78% | +0.90% [-0.3] / +2.32% |
| **+ short form** | **+5.34% [+3.5] / +7.36%** | **+3.69% [+2.5] / +5.31%** | **+3.38% [+1.8] / +5.33%, 80%** | **+1.79% [+0.7] / +3.37%** |
| + xG schedule + territory | +3.96% / +6.33% | +3.03% / +4.97% | +2.26% [+0.6] / +4.36%, 78% | +1.23% [+0.1] / +2.90% |
| + all new | +5.54% [+3.6] / +7.71% | +3.60% [+2.3] / +5.20% | +3.49% [+2.0] / +5.32%, 80% | +1.60% [+0.6] / +3.25% |

(285 bets a season at top 5 a week on 2025/26, 225 on 2024/25; double at top 10.)

**Higher volume, short form, outside the top five:** top 20 a week +2.11% [+1.2] (2024/25) and +0.79% [+0.1]
(2025/26) a tick off fair; the strongest 5 a day +0.69% and -0.06%: still not a daily rule.

**Read:**
- **Short and recency-weighted form is the one that adds.** It lifts every rule in both seasons, and the top 10 a week
  now clears zero a tick off fair in both seasons with the lower bound above zero: about 570 bets a season, twice the
  candidate's volume.
- **Fatigue does not add** to the move forecast here. The market seems to price congestion it can see in the fixture
  list; the minutes load needs line-ups the model does not yet weight well.
- **Schedule-adjusted xG and territory are mixed:** better at top 10, worse at top 5, on a third to a half of the
  matches. Not enough to keep alone.
- **All new together** matches short form alone: the extra 38 columns add nothing reliable.
- **Caveat.** About 120 variants have now been read on 2025/26. The gain from short form is consistent across both
  seasons and every cut-off, which makes chance less likely, but the rule must still be pre-registered and judged
  forward.

**Updated candidate for pre-registration:** the stack on team features + `home_adv` + `team_news` + `form_short` +
`x_short`; its strongest 10 forecasts a week outside the top five leagues (or 5, at higher return); back first where
the chance is forecast up, lay first where down; greened at the close.

## Expected-value rule, trade types and staking (4 Nov 2026)

**Sources.** `reports/football_runs/ev_rule.json`, `staking.json` (`research/open_close/football_staking.py`) and
`bf_historic_test.json` (`research/open_close/football_bf_historic_test.py`). Features: the candidate's plus short
form. 2024/25 and 2025/26 on football-data's two Betfair snapshots, a tick off fair at each end; Aug-Sep 2026 on Betfair's own
traded prices from the owner's Historic Data bundle.

**The rule.** Every selection whose expected value after costs clears a margin, not a fixed count:
- the forecast calibrated into an expected move on the season before;
- the expected CLV at fair, less commission;
- less the round trip's cost at the early book.

| Expected value above | 2024/25 | 2025/26 | Aug-Sep 2026, Betfair prices: a tick worse each end / at the traded price |
|---|---|---|---|
| 0 | +0.68% (46 a window) | -0.19% (68) | -0.40% / +1.83% |
| 1% | +1.25% (22) | +0.51% (34) | +0.06% / +2.36% |
| **2%** | **+2.30% (12)** | **+0.99% (17)** | **+0.74% [-0.3] / +3.19% [+2.1]** (454 bets, 62% right way) |

(Returns at level stakes, all leagues, all selections; windows are collection windows, two a week.)

**Trade types** (expected value above 2%):
- **Greened at the close beats held to the result** in every season. Held, the same selections lost 50-99% of the
  bank in the simulations: holding swaps a small, steady CLV for the result's variance and pays commission on the
  whole win.
- **Home and away carry the edge.** The draw's forecast move (implied by home and away) barely predicts: slope
  0.2-0.3, almost never selected. Over/under 2.5 adds trades at lower return (+0.4% / +0.2% greened in 2025/26).
- **Lay-to-back is most of the trades** (about 80%) and pays in both seasons; back-to-lay too.

**Staking** (expected value above 2%, greened; the bank simulated window by window, capital tied up from the
collection moment to the close: a back's stake, a lay's liability):

| Stakes | Capital at work | 2024/25 bank, worst drawdown | 2025/26 bank, worst drawdown |
|---|---|---|---|
| Level | 25% of the bank | x1.23, 1.8% | x1.18, 4.1% |
| **In proportion to expected value** | **25%** | **x1.27, 1.9%** | **x1.23, 4.0%** |
| Kelly under the capital limit | 25% | x1.42, 4.1% | x1.39, 8.2% |
| In proportion to expected value | 100% | x2.16, 3.7% | x1.87, 5.1% |

- **Stakes in proportion to expected value** give the best return for the risk: the highest window Sharpe and the
  shallowest drawdowns in both seasons.
- Kelly under the capital limit makes more on the capital but doubles the drawdowns: its risk estimates are noisy.
- **The binding limit is capital, not the edge.** Every trade of a window is open at once, and a lay ties up its
  liability, so a lay at a long price costs far more capital for the same expected value.

**Execution decides it.** On Betfair's own prices the rule makes +3.19% at the traded price and +0.74% paying a tick
each end: the spread is worth more than the edge. Resting orders at the traded price, not crossing, is the
difference. That is the resting-order pre-registration's question, answered on the recorder's books.

**Caveats:**
- 2025/26 and Aug-Sep 2026 were read before, on other prices: development evidence.
- The bank figures assume fills at the stakes they imply. The early-price liquidity in lower leagues is unknown
  until the recorder runs.

## The Massey power rating (5 Nov 2026)

**Source:** the owner pointed to markstats.club's Massey rating (75% xG, 25% goals, opponent-adjusted).
- The site shows only today's table, for the top five leagues and the Champions League, so it cannot be tested
  without look-ahead.
- The same construction is fitted on our data instead, point in time, in every league we hold, with goals alone
  where there is no xG (`football/xg_features.py massey_ratings`, family `x_massey`).

**Does it predict results better than raw xG difference?** Yes. Out of sample 2023-2026, on 12,387 matches, its R2 on
the goal difference is 0.177 against 0.154; early in the season it is 0.201 against 0.130
(`reports/football_runs/massey_check.json`). It does not beat the opening price, at 0.203 (early season 0.229).

**Does it add to the price-path model?** Not reliably (`reports/football_runs/massey.json`). Added to the paper
trader's features, outside the top five leagues, a tick off fair:

| Rule | Without | With |
|---|---|---|
| Top 5 a week, 2024/25 | +5.34% | +5.45% |
| Top 5 a week, 2025/26 | +3.38% | +4.10% |
| Top 10 a week, 2024/25 | +3.69% | +3.64% |
| Top 10 a week, 2025/26 | +1.79% | +1.90% |
| Top 20 a week, 2024/25 | +2.11% | +2.26% |
| Top 20 a week, 2025/26 | +0.79% | +0.42% |

Across all leagues it is worse at top 5 a week in 2024/25 (+4.43% against +4.87%).

**Read:** a better result rating, but the market already prices what it knows. It is kept in the lab for later
reads. The frozen paper trader stays as registered.

## Evaluation standard (the owner, 6 Nov 2026)

Every football test is scored against two targets, side by side:

1. **The closing price** (the football counterpart of the horses' BFSP): Betfair's last price before the off, and
   its SP where Betfair offers one. Asks whether the call beats the close (CLV, and the greened return).
2. **Winning the bet**: the same selection held to the result. Asks whether it makes money on the outcome.

The staking study already reports both (greened against held). Cross-league strength, the next piece of work, is
judged the same way.

## Cross-league strength, first read (6 Nov 2026)

**Source:** `football-research.yml` job `cross_league_check`, run 42202323657.

**What was built:** `football/cross_league.py`, a goal-weighted Elo run through 1,140,887 senior club matches (leagues,
cups, continental competitions), point in time. It is scored out of sample from 2018 on the result target (winning
the bet); the price target follows once the live record holds these markets.

| Ties | Cross-league Elo | Competition-only Elo | No information |
|---|---|---|---|
| Cup ties across divisions (25,321) | **0.960 log loss, 55.1% right** | 0.998, 50.6% | 1.057, 40.0% |
| Continental ties across countries (9,641) | 1.025, 49.6% | **1.009, 50.9%** | 1.049, 48.0% |
| League matches (918,434; a sanity check) | 1.012, 50.2% | 1.011, 50.3% | 1.069, 44.2% |

(The competition-only baseline rates each club inside each competition apart. In Europe that is a club's own record
in that competition, an informative baseline.)

**Read:**
- **Across divisions it works.** Cup ties are called right 55% of the time, against 51% without the cross information.
- **Across countries it does not yet.** It falls short of a club's own European record. The league ranking shows why:
  Denmark is second and Slovakia sixth, and Albania and Kosovo sit above Switzerland. Champions of small leagues bank
  rating in early qualifiers against minnows, and their leagues inherit it.

**Next (v2):**
- a club's rating split into its league's level plus its place within the league, the league level learned from
  cross-border results with a larger step;
- the European-competition record kept as its own feature beside it.

## Cross-league strength v2 (6 Nov 2026)

**Source:** `football-research.yml` job `cross_league_check`, run 42203212585 (commit 97d9f9b). Same matches, same
walk-forward from 2018, the result target. v2 = a league level (moved only by matches between clubs of different
leagues) plus a place within the league (centred on zero). Its step sizes were fixed before any result was read.

| Ties (log loss, share right) | v1 cross-league | v2 level + place | v2 + competition record | Competition-only | No information |
|---|---|---|---|---|---|
| Cup ties across divisions (25,321) | 0.960, 55.1% | 0.950, 56.2% | **0.945, 56.5%** | 0.998, 50.6% | 1.057, 40.0% |
| Continental ties across countries (9,641) | 1.025, 49.6% | 0.993, 53.0% | **0.982, 53.2%** | 1.009, 50.9% | 1.049, 48.0% |
| League matches (918,434) | 1.012, 50.2% | 1.018, 49.8% | **1.010, 50.4%** | 1.011, 50.3% | 1.069, 44.2% |

**Read:**
- **v2 fixes Europe.** Across countries it now beats the club's own European record: 0.993 against 1.009 log loss,
  53.0% right against 50.9%. With that record beside it, the best is 0.982 and 53.2%.
- **Across divisions it improves again,** to 56.5% right against 50.6% without the cross information.
- **Within leagues** v2 alone is slightly worse than a plain league Elo (1.018 against 1.011), as expected: a place
  in the league moves only on league results, at the same step. With both, it is level or better (1.010).
- **The league ranking is now sane:** Premier League (2,199), Ligue 1, Bundesliga, La Liga, Serie A, then Belgium,
  Denmark, Czechia, Sweden, Austria. Two oddities remain: Denmark seventh, and the Eredivisie nineteenth below Slovakia.

**What this is and is not:**
- It is a result-target gain, out of sample.
- It is not yet a betting edge. The market prices cup and European ties with far more information than a rating. The
  closing-price target, and the betting return held to the result, need Betfair prices for these matches.
- Those come from two places: the Aug-Sep 2026 Historic bundle (cup and UEFA markets in it) and the live record
  (UEFA and cup markets recorded from 5 Nov). That test comes next. Until it passes, nothing here goes near the
  frozen paper trader.

## Cross-league strength against Betfair's prices (6 Nov 2026)

**Source:** `football-research.yml` job `cross_league_price`, run 42204367947 (commit dde63b4,
`research/open_close/football_cross_league_price.py`). The rules were committed before the run.

**Data:**
- 11,565 Betfair MATCH_ODDS markets from the owner's Historic Data bundle in S3, kicking off 30 Jul to 30 Sep 2026.
- 9,279 of them linked to an API-Football fixture. 2 were dropped where the two sources disagreed on the result.
- Linked by group: 480 cup ties across divisions, 226 continental ties across countries, 7,793 league matches.
- The rating's chances come from v2 plus the competition record, fitted on matches before 1 Aug 2025, so every one
  of these matches is out of sample.

**Rule:** back a home or away side when the rating's expected value at the price is above 2%. The early price is the
last trade 24 h before the off.

| Group | Rating gap vs the move (corr) | Closing price: at the last traded prices | Closing price: a tick worse each end | Held to the result, early price | Held to the result, chosen at the close |
|---|---|---|---|---|---|
| Cup ties across divisions (480 bets) | **0.24** | **+3.5%** (+0.7 to +6.6) | -0.2% (-2.9 to +2.7) | +9.8% (-18.7 to +51.6) | +18.2% (-21.5 to +70.0) |
| Continental ties (231 bets) | 0.09 | +1.5% (-0.9 to +3.9) | -2.4% (-4.7 to -0.1) | -14.4% (-37.4 to +10.3) | -15.6% (-37.3 to +9.3) |
| League matches, the control (7,268 bets) | 0.12 | +1.3% (+0.5 to +2.2) | -1.6% (-2.3 to -0.7) | -2.6% (-6.4 to +1.4) | -4.4% (-8.0 to -0.4) |

(Return on a GBP1 stake after 2% commission; 95% intervals by market in brackets.)

**Log loss of the result** (lower is better):

| Group | The close | The early price | The rating |
|---|---|---|---|
| Cup ties across divisions | 0.813 | 0.840 | 0.876 |
| Continental ties | 0.939 | 0.950 | 0.996 |
| League matches | 0.986 | 0.994 | 1.016 |

**Read:**
- **Cup ties across divisions are where the market is slowest.** There the rating's gap to the early price points the
  way the price moves twice as strongly as in league matches (0.24 against 0.12). On target 1 (the closing price),
  the 2% rule earns +3.5% at the traded prices, with the interval above zero.
- **Nothing survives paying the spread.** A tick worse at each end takes the cup edge to -0.2%. Continental and
  league trades lose (-2.4% and -1.6%). Captured at all, the edge needs resting orders at the early price, as the EV
  paper trader already uses, not crossing the spread.
- **Target 2 (winning the bet) is too noisy at this size.** Cup backs held to the result return +9.8% and +18.2%, but
  the intervals run from about -20% to above +50%. The control is clean: rating-only backs chosen against the close
  lose 4.4% in league matches. The close is the better price.
- **The rating is no price.** It is worse than the early price on log loss in every group. Its use is as a pointer
  to where the early market is wrong, mainly in domestic cups.
- **Draws are not read.** The draw's early last trade is often stale and far from fair, so its returns (+14% in
  leagues, on 44% right-way moves) are artefacts of that, not edge.

**Next:**
- The cup signal with resting orders. Score the cup-tie rule at the recorded books (queue position, fills) once the
  live record holds enough cup rounds; it records TO_QUALIFY and cup match odds from 5 Nov.
- A larger cup sample from more Historic Data months. 480 ties is one window.
- Pre-register a cup-tie paper trial only after both. The frozen EV trader is unchanged.

## Price discovery and a model-free rule scan (6 Nov 2026)

The owner: "keep an open mind, we are looking at price discovery and discovery of profitable trading strategies."

**Source:** `football-research.yml` job `price_discovery`, run 42208953721 (commit 00aa580,
`research/open_close/football_price_discovery.py`). Betfair's own prices only, no model. The Historic Data soccer
tables in S3 hold 263,468 markets (684,807 runners with a result), kicking off 30 Jul to 30 Sep 2026.

**A data bug found first:**
- The first run (42207392243) showed rules returning above +29,000%.
- The cause: in Asian handicap and alternative-totals markets, the tables gave every handicap line of a runner one
  line's prices, against its own result.
- Those markets are now left out of the scan.
- `betfair_historic.py` is fixed to key prices by runner and line. The S3 tables need a rebuild
  (`--build --force`) to pick the fix up: the owner's call, since it overwrites them.

**Where the price is found** (mean absolute move in log price, all market types; how often a mark has a price in
brackets):

| Window | Mean move |
|---|---|
| first trade to 24 h before (9% of runners priced at 24 h) | 0.028 |
| 24 h to 60 min (66% priced at 60 min) | 0.074 |
| 60 to 15 min (84%) | 0.038 |
| 15 to 1 min (96%) | 0.027 |
| last minute to the close | 0.012 |

- The price moves most in the last day before the hour before the off.
- Correct score and half-time markets move most (0.12 to 0.15 in that window).
- Double chance moves least (0.035).
- Moves reverse slightly in the next window (correlation -0.07 to -0.11). With last traded prices that is mostly the
  price flipping between the back and lay sides, not tradable reversal.

**The rule scan:**
- 656 rules tried: momentum and fade on four windows at three thresholds, and back or lay held to the result in eight
  price bands at three marks, across the 12 largest market types and all of them together.
- Each rule was charged a tick worse each end plus 2% commission.
- **One rule passed on the first half:** lay first-half over 0.5 goals at 5.0-10.0 at the close, +20.0% on 539 bets.
- **On the second half it did not hold:** +7.9% on 629 bets, 95% interval -7.7% to +23.8%.
- **No rule is confirmed.**

**Read:**
- At Betfair's traded prices, after paying a tick each way, no simple model-free rule pays in this window. That
  covers buying or selling price bands, following steam and fading drift.
- One month a half is short, so a small edge would not show. But nothing large and simple is there.
- An edge has to come from one of two places:
  - **information** the price does not yet hold: a model, as in the cup-tie signal, where the rating's gap to the
    early price points the way the price moves;
  - **execution**: resting orders that earn the spread instead of paying it.
- Both need the recorded books (depth and queue), not the BASIC last-traded-price files.

**Next leads:**
1. **Spread capture on the recorded books.** Resting orders at the best back and lay through the last 24 h: how
   often they fill, and whether they were run over by the move.
2. **Cross-market consistency.** At the same moment, compare the chances implied by related markets: match odds
   against double chance and draw no bet, over/under against correct score. Do gaps close, and which market moves
   to close them? Which market leads price discovery?
3. The cup-tie signal with resting orders.

## Cross-market consistency: who leads, and a lag to test on the books (6 Nov 2026)

**Source:** `football-research.yml` job `cross_market`, run 42210625430 (commit 08e94bb,
`research/open_close/football_cross_market.py`). Last traded prices from the Historic Data tables: 12,233 events,
30 Jul to 30 Sep 2026.

**Method:** the chances in one market are compared with the same chances worked out from a related market of the
same event, at 60, 15 and 1 minute before the off:
- match odds against double chance (home or draw = home + draw, and so on);
- match odds against draw no bet;
- correct score summed into over/under 2.5, against over/under 2.5.

A market's chances count at a mark only when every runner of it has a price there. The first run (42210118169) did
not check this, and showed gaps of about 30 points: those came from part of a book normalised over itself.

**Agreement:** the related markets disagree by about one point of chance on average.
- Double chance: 0.8-1.3 points.
- Draw no bet: 1.5-1.6 points.
- Over/under 2.5 against correct score: 0.6 points.

**Who leads:**
- **Match odds leads.** When double chance or draw no bet disagrees with it 60 minutes out, the smaller market makes
  up 46-62% of the gap by the close, while match odds barely moves (-8% to -12%). At 15 minutes the follower makes up
  27-41%; in the last minute, 5-14%.
- **Over/under 2.5 leads correct score, weakly.** Correct score makes up 6-13% of the gap; over/under 2.5 moves 4-7%.

**The rule scan:**
- Rule: trade the follower towards the leader when the gap is over 1, 2 or 4 points.
- 126 rules tried; 18 passed on the first half; 9 held on the second half with the 95% interval above zero.
- Every confirmed rule closes at the close (the closing-price target). None of the held-to-result versions held:
  target 2 (winning the bet) shows nothing.
- The strongest:

| Rule (closed at the close, a tick worse each end) | First half | Second half |
|---|---|---|
| Draw no bet, away, 60 min, gap over 2 points | +4.3% on 407 | **+3.3%** on 515 (+2.3 to +4.4) |
| Draw no bet, away, 60 min, gap over 1 point | +2.6% on 840 | +2.1% on 975 (+1.3 to +2.8) |
| Draw no bet, home, 60 min, gap over 2 points | +1.4% on 407 | +1.6% on 515 (+0.6 to +2.7) |
| Double chance, draw or away, 60 min, gap over 1 point | +1.8% on 414 | +1.6% on 629 (+1.0 to +2.2) |
| Double chance, home or draw, 60 min, gap over 1 point | +0.9% on 373 | +1.3% on 575 (+0.7 to +2.0) |
| Double chance, draw or away, 15 min, gap over 2 points | +1.8% on 349 | +1.1% on 454 (+0.7 to +1.6) |
| Double chance, home or draw, 15 min, gap over 1 point | +0.8% on 645 | +0.7% on 799 (+0.4 to +1.1) |
| Draw no bet, home, 15 min, gap over 2 points | +1.4% on 647 | +0.6% on 747 (+0.1 to +1.1) |
| Double chance, draw or away, 15 min, gap over 1 point | +0.9% on 750 | +0.4% on 903 (+0.1 to +0.6) |

**Why this is not yet an edge:**
- Double chance and draw no bet trade far less than match odds, so their last traded price at a mark can be hours
  old.
- A gap to match odds then partly measures how long ago the small market last traded. The book itself may already
  have been in line with match odds, with no stale price left to take.
- The pattern fits that: it pays at the closing price, decays from 60 to 15 minutes, and has no edge held to the
  result. That is what a recording lag would look like, and also what a real lag would look like.
- Last traded prices cannot tell the two apart. The books can.

**The test:** the live feed now records double chance and draw no bet books beside match odds
(`football/exchange_feed.py DEFAULT_TYPES`), read-only, under the same Throttle. The paper book then gives every
selection in those markets executable prices, depth and resting-order fills at 60 and 15 minutes.

The rule, fixed now for that test:
1. At 60 and 15 minutes, take the gap between match odds' mid chances and the follower's mid chances.
2. Over 1 or 2 points, take the follower's best offered price towards match odds.
3. Close at the close, crossing the spread.

It is scored once 30 match days of these books are held.
