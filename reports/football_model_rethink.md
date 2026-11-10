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

## Data: a regular feed with broad coverage of our matches (6 Nov 2026)

The owner wants the most advanced set of football metrics, but only from data that is a regular, updated feed with
broad coverage of the matches we trade.

**API-Football's coverage of our Betfair matches**

**Source:** `football-research.yml` job `feed_coverage`, runs 42238710317 and 42239864493
(`research/open_close/football_feed_coverage.py`). The live record covered 4-6 Nov 2026: 301 match-odds markets,
GBP 28.2m matched.

| | Share of markets | Share of money matched |
|---|---|---|
| Linked to an API-Football fixture | 83.7% | 98.7% |
| The feed has events | 82.1% | 98.7% |
| The feed has line-ups | 72.4% | 98.1% |
| The feed has team statistics | 44.5% | 95.7% |
| The feed has player statistics | 41.9% | 95.6% |
| The feed has injuries | 14.6% | 85.9% |
| Expected goals in its team statistics (one finished fixture read a league, 103 leagues) | 35.9% | 93.5% |
| **History we held: the 22 main divisions** | **7.6%** | **3.6%** |

- On the Historic Data bundle (12,730 markets, 30 Jul - 30 Sep 2026), 87.6% link and 38.2% have team statistics.
- The Europa League and Champions League carried GBP 21.4m of the GBP 28.2m. Neither is among the 22.

**Done:**
- `sources/api_football_leagues.json` lists 361 leagues (339 beyond the 22): every league our markets linked into
  that the feed covers with team statistics or line-ups. They are ordered by money matched.
- `sources/api_football_history.py` fetches the 22 first, then the list in its order.
- `source-data.yml` now fetches this history every night (this season and last), within the plan's quota. It leaves
  1,500 of the day's 7,500 requests for the line-up collector. Before this, history was fetched only by hand.
- The estimate is about 4,400 requests a season, so a two-season backfill takes about two nights.

**Location-based metrics: our own models as a benchmark, not a feed**

**Source:** `reports/football_runs/event_models.json` (`football/event_models.py`).
- The data is the Wyscout public match events (Pappalardo et al. 2019, CC BY 4.0, which allows commercial use with
  attribution): 3,251,294 events from 1,941 matches, 2017/18 and 2018.
- It is one fixed release, not a feed, so it serves only as a yardstick for judging any paid feed with pitch
  positions.
- StatsBomb's open data is not used: its agreement forbids commercial use.

**xG,** each competition predicted by a model fitted on the other six (45,287 non-penalty shots):

| Model | Log loss | AUC |
|---|---|---|
| Base rate | 0.330 | - |
| Distance and angle | 0.287 | 0.756 |
| Our features (logistic regression) | 0.278 | 0.779 |
| Our features (gradient-boosted model) | 0.278 | 0.779 |

- Our features add body part, direct free kick, counter-attack, cross, through ball, set piece and rebound.
- The model is calibrated across deciles: predicted 0.381 against 0.378 scored in the top tenth.
- Penalties are scored at their own rate, 72.5% (of 658).

**At match level:**
- Our xG summed per side has a correlation of 0.51 with goals; shots alone have 0.32.
- xG difference has a correlation of 0.58 with goal difference.
- Expected threat (16 x 12 grid, value iteration) added per side has a correlation of 0.19 with goals.

**Next:**
1. The two-night backfill, then the lab rebuilt with the held history for every covered league.
2. Re-score the cup-tie signal and the EV rule with team and player features now available for cups and UEFA
   matches (the leagues with the money).
3. A paid feed with pitch positions only if its coverage of our matches matches API-Football's, and its metrics beat
   ours on the two targets.

## The history backfill, and cup ties with each club's form from the feed (6 Nov 2026)

**The backfill** (`source-data.yml` run 42241092940, API-Football history for the 22 divisions and the 361-league
list, the 2025 and 2026 seasons):
- 5,928 requests; 101,301 matches fetched.
- It stopped at the quota floor as designed, with 1,481 requests left for the line-up collector.
- 594 of 722 league-seasons done; the nightly fetch takes the rest.
- Held now: 127,107 matches complete (line-ups and events); 141,046 team-statistics rows, 56,206 of them with xG.

**The check** (`football-research.yml` job `cup_form_check`, run 42273275138,
`research/open_close/football_cup_form_check.py`; rules fixed before the run):
- Each club's mean over its last 10 matches (any competition) of shots-on-target difference, shots difference and
  xG difference, before the match.
- Added to the v2 rating and competition record.
- Fitted on 2025/26 (1 Aug 2025 - 31 Jul 2026), tested on the Historic Data bundle's matches from 1 Aug 2026.
- The first run (42271884840) was invalid: the form was joined to the wrong rows, and no test match had any. It is
  fixed and covered by a test.

| Group (test matches, share with form) | Rating-only: log loss / gap-move corr / closed at close (LTP) / a tick worse | Rating plus form: the same |
|---|---|---|
| Cup ties across divisions (480, 20%) | 0.885 / 0.224 / +3.8% (+0.6 to +7.1) / +0.1% | 0.897 / 0.219 / +3.5% (+0.3 to +7.0) / -0.1% |
| Continental ties (226, 85%) | 0.999 / 0.098 / +1.3% / -2.6% | 1.000 / 0.102 / +1.5% / -2.4% |
| League matches (7,790, 42%) | 1.016 / 0.114 / +1.3% (+0.6 to +2.2) / -1.7% | 1.014 / 0.134 / +1.6% (+0.9 to +2.6) / -1.3% |

(Held to the result, every group's interval spans zero for both models.)

**Read:**
- **Form from the feed adds nothing to the cup and continental signal.**
  - Cup ties: only 20% of the test ties (7.5% of the fitting ones) have form for both sides. The lower-division
    clubs that make a cup tie cross divisions are the ones the feed's statistics miss, so the feature is mostly
    empty where it matters.
  - Continental ties: 85% have form, and still no gain. The v2 rating already holds what recent shots and xG say
    about these clubs.
- **In league matches it helps a little.** The rating's gap to the early price points the way the price moves more
  strongly (0.114 to 0.134). At the traded prices the closing-price rule goes from +1.3% to +1.6%. A tick worse each
  end it still loses (-1.3%).
- **Nothing here clears the spread when crossing.** The leads stay as they were:
  - the cup-tie signal, and the double chance / draw no bet lag, both with resting orders on the recorded books;
  - the frozen EV paper trader (first scored run Tue 10 Nov).
- **The data work stands on its own.** The feed's history now covers the leagues that carry our money, and is kept
  current nightly once PR 268 is merged. Every later model on these matches can use it.

## Every strategy so far, and the bar a closing-line model must clear (8 Nov 2026)

The owner: "Evaluate all possible strategies. Any profitable so far? Any edge." Then: "improving our closing line
model ... a profitable, automated, trading solution".

**The real cost, measured at the recorded books** (`football-research.yml` job `paper_lines_scan`, run 42462237480,
`research/open_close/football_paper_lines_scan.py`):
- The paper book, 4-8 Nov 2026: 2,517 markets, 39,138 lines.
- Only tradable lines are counted: entry spread at most 3 ticks, a two-sided close (lay within 5% of back), price at
  most 20. Untradable books at the close (a 1.01 back against a 1,000 lay) otherwise turn a GBP1 lay into thousands
  of per cent: the unfiltered first run, 42462017662, is not usable.

**How much of the record a trader can use:**

| Market | Tradable share | Median spread | Median back depth |
|---|---|---|---|
| Match odds | 38% | 3 ticks | GBP 219 |
| Over/under 2.5 | 31% | 4 ticks | GBP 236 |
| Double chance | 6% | 7 ticks | GBP 80 |
| Draw no bet | 6% | 8 ticks | GBP 71 |

**The unconditional cost of a trade closed at the close (tradable lines, after 2% commission):**

| Entry | Back | Lay |
|---|---|---|
| Crossing the spread (2,939 lines each side) | -2.26% (-2.38 to -2.15) | -2.58% (-2.73 to -2.44) |
| Resting one tick inside (85-89% filled) | -1.54% (-1.89 to -1.20) | -2.32% (-2.73 to -1.95) |

- Every match day sits in the same range (-1.9% to -2.9% crossing).
- Resting saves under a point: its fills are adversely selected. They come mostly when the price is about to move
  through the order.
- **The bar:** a selection is worth trading only when its forecast move to the close beats about 2.3% crossing, or
  1.5-2.3% resting.

**Every strategy tested to date:**

| Strategy | Target 1: closing price | Target 2: held to the result | Verdict |
|---|---|---|---|
| EV rule (model move > 2% after costs), football-data snapshots | +2.3% (2024/25), +1.0% (2025/26), a tick off fair | loses (the bank fell 50-99% in simulation) | best lead; forward test from Tue 10 Nov |
| EV rule on Betfair's Aug-Sep 2026 traded prices | +3.2% at the traded price; +0.7% a tick worse each end (interval below zero) | - | edge about the size of the spread |
| Stacked move rule (strongest 5 a week, outside the top five) | +2.0% / +2.6% a tick off fair (2025/26, about 100 variants read) | - | in forward trial |
| Cup ties, cross-league rating | +3.8% at LTP; +0.1% a tick worse | +11%, interval -21% to +55% | lead, not tradable crossing |
| Continental ties, cross-league rating | +1.3%; -2.6% a tick worse | -13% | no |
| League matches, cross-league rating | +1.3%; -1.7% | -4% | no |
| Club form from the feed added (cups, UEFA, leagues) | no gain in cups or UEFA; +0.3 points in leagues | no gain | no |
| Massey rating | no gain on the price path | - | no |
| 656 model-free price rules (momentum, fade, price bands) | none confirmed on the second half | none | no |
| Double chance / draw no bet lag (match odds leads) | +3.3% on last traded prices; **-2.2% to -2.7% at the real books** | no | dead: stale prices |
| Every paper-book cell (46 tradable cells) | all below zero greened | one cell above zero held (over/under lay at 60 min, 92 lines), noise at 46 cells | no |

**Read:**
- **No strategy is profitable yet.**
- **The EV rule is the only lead whose edge is larger than zero after a realistic cost on two seasons.** Its forward
  test starts with the recorded books (first scored run Tue 10 Nov).
- **The binding constraint is cost, not signal.** The round trip is about 2.3%, and our best forecasts are worth 2-4%
  of price before it.

## Closing-line model: line-up news and the late move (8 Nov 2026)

**Source:** `football-research.yml` job `lineup_late_move`, run 42463318516
(`research/open_close/football_lineup_late_move.py`; rules fixed before the run).
- The first run (42462495893) stopped on fixtures whose line-ups tag both teams with one side; those are left out.

**Data:**
- Betfair match odds from the Historic Data bundle (12,437 markets, 30 Jul - 30 Sep 2026), at 24 h, 60 and 15 minutes
  before the off and at the close.
- 8,473 linked to an API-Football fixture; 3,190 have a line-up surprise and a full price path.
- The surprise: the confirmed eleven's goals plus-minus less the expected eleven's, home less away.
- Ratings were refitted monthly on earlier matches only: 813,148 stints, about 120,000 players rated.

**Does the surprise point the way the price moves?** (the price is the home side's log chance ratio)

| Move | Correlation with the surprise | Slope (95% interval) |
|---|---|---|
| Early: 24 h to 60 min | 0.069 | 0.094 (0.043 to 0.147) |
| **Late: 60 min to the close** | **0.099** | **0.100 (0.065 to 0.134)**; 0.103 given the early move |
| Last: 15 min to the close | 0.026 | - |

**Trades** (back the side the surprise favours, closed at the close):

| At | Surprise above | Bets | At the last traded prices | A tick worse each end |
|---|---|---|---|---|
| 60 min | 0.1 | 1,464 | +0.5% (0.0 to +0.9) | -2.2% |
| **60 min** | **0.2** | **557** | **+1.4% (+0.6 to +2.2); Aug +1.6%, Sep +1.2%** | -1.4% |
| 15 min | 0.1 | 1,464 | -0.4% | -3.0% |
| 15 min | 0.2 | 557 | -0.3% | -3.0% |

**Read:**
- **Line-up news is real signal the market prices late.** The surprise predicts the move from 60 minutes to the
  close, in both months, and beyond the early move. Part of it is already in the price by 60 minutes (the early
  slope), and it is gone by 15 minutes: the market absorbs the line-ups within the hour.
- **At the 60-minute mark it is too small for the spread.** +1.4% at traded prices, -1.4% paying a tick each way.
  The bar measured at the recorded books is about 2.3%.
- **The timing is the opportunity.** The line-ups come out about 60-75 minutes before the off. The 60-minute mark is
  already after part of the reaction. A trader acting within a minute or two of the release catches more of the move.
  The line-up collector records when each eleven is confirmed, and the feed records the books every minute near the
  off, so that move can be measured from now on.

**The forward trials:** the scheduled run of 8 Nov (42461905728) reports "no matches in the window yet" for all
three. The window opened Friday 6 Nov, and the lab only holds those matches once their football-data rows arrive.
The first scored run is Tuesday 10 Nov.

**The closing-line model, next:**
1. **Line-up timing:** measure the move from each confirmed eleven to the close at the recorded books, and the price
   available within 1-2 minutes of the release. Pre-register a line-up trader if the move clears the bar.
2. **Surprise as a feature:** add the expected-eleven and surprise features to the move model's late horizon (60
   minutes and nearer). The early horizon already has the expected eleven (team news).
3. **A spread-aware gate:** trade only where expected value beats each book's own round trip (spread and depth at
   that moment), not a flat tick.
4. **The recorded books as training data:** retrain the move model on recorded checkpoints, with book features
   (spread, depth imbalance, drift so far), as weeks of records build up.

**Line-up release at the recorded books** (`football-research.yml` job `lineup_release`, run 42464673010,
`research/open_close/football_lineup_release.py`):
- The stored line-up tables are only rebuilt at the end of a full football-trade.yml day, and the cancelled days had
  left them behind. They were rebuilt here from the collector's raw replies, in memory: 2,815 polls, 105 matches,
  90 with both elevens, 3-8 Nov.
- **API-Football has the elevens about 28 minutes before kick-off,** every match (10th-90th percentile 27.7-28.7
  minutes).
- That is half an hour after the official team sheets (about 60-75 minutes before kick-off), and after the market has
  priced them. The Historic Data read above shows the line-up move is spent by 15 minutes.
- At the books, 86 matches:
  - the surprise's correlation with the move from release to close is -0.03 (after entry: -0.01);
  - backing the favoured side at the release (median 2.9 minutes after it) and closing at the close: -2.5% (49 bets,
    12% right way) and -3.6% (13 bets);
  - small numbers, but no sign of a move left to take.

**Read:**
- The line-up signal is real (60 minutes to the close, both months), but **our feed brings the elevens too late to
  trade it.** A line-up trader needs the team sheets at release (about an hour before kick-off), from a faster
  source than API-Football.
- Until then, line-up news stays where the move model already has it: the expected eleven, known at the early price.

## Closing-line model on the recorded books, first read (8 Nov 2026)

**Source:** `football-research.yml` job `close_model_books`, run 42481445610
(`research/open_close/football_close_model_books.py`).

**Data and method:**
- 1,574 rows: every tradable match-odds and over/under 2.5 checkpoint (24 h, 6 h, 2 h, 60 min, 15 min) of 515
  markets, 4-8 Nov.
- Features: book only (spread, depth, imbalance, overround, drift so far).
- Target: the move of the mid chance to the close.
- LightGBM, walk-forward by match day: 937 rows scored, on 7-8 Nov.

**Results:**
- The forecast does not yet track the move: correlation 0.009, out-of-sample R2 -0.15. By checkpoint it ranges
  from 0.20 at 60 minutes to -0.25 at 6 hours, on a few hundred rows each.
- Trades closed at the close crossing, after commission:

| Entry rule | Trades | Return |
|---|---|---|
| Spread-aware gate (forecast beats the book's own round trip) | 109 | -2.35% (-3.35 to -1.42) |
| The older flat-tick gate | 472 | -2.70% (-3.18 to -2.23) |
| Every tradable line in those days | 1,874 | -2.59% (-2.70 to -2.48) |

**Read:**
- The spread-aware gate already does what it is for: it trades a quarter as often as the flat-tick gate and loses
  less.
- With five days and book features alone there is no forecast skill: these returns are the cost of the round trip,
  not an edge. That is expected. The book's own path carried nothing in the 656-rule scan either.
- The skill has to come from the fundamentals: our fair price, team news, the cross-league rating.
- **Next for the model:** join each recorded checkpoint to our model's fair price (the EV model's forecast, the lab
  features for the 22 divisions) and train on both.
- The scorer runs with every trials job (Tuesday and Friday, `football-trials.yml`), so the training set grows
  every week.

**Faster line-ups:**
- Sportmonks says its line-ups arrive 60-75 minutes before kick-off, with a confirmed-or-predicted flag; API-Football
  says 20-40 minutes, and we measured 28.
- These are vendor claims, with no independent timing test found.
- Sportmonks' free plan reportedly covers the Danish Superliga and the Scottish Premiership, both of which we
  record. That is enough to measure its real release times against the Betfair price move, at no cost, once the
  owner opens an account and adds its key as a repository secret.

**With our model's forecast joined** (`football-trials.yml` run 42483643138; the first try, 42481887157, stopped on a
selection-id type mismatch, now fixed with a test):
- The join runs, but has nothing to join yet: the lab holds 5 matches in 4-8 Nov, and none link to a recorded
  market.
- The lab takes its matches from football-data.co.uk's rows. The weekend's rows arrive after the matches (usually by
  Monday), so the Tuesday trials run is the first that can join them.
- Until then the model is the book-only one: 1,588 rows, correlation 0.010, the spread-aware gate -2.35% on 109
  trades, every line -2.59%.
- From Tuesday on, each trials run (Tuesday and Friday) adds the week's matches with the model's forecast.

## First forward scores at the recorded books (Tuesday 10 Nov 2026)

**Source:** `football-trials.yml` scheduled run 42769049818, matches of 6-9 Nov (the first window).

| Trial | Bets | Return | Notes |
|---|---|---|---|
| **EV rule** (pre-registered: resting entry, greened at the close) | 11 filled of 22 signals posted | **-2.9%** (95% -5.6% to +0.9%) | CLV -1.7%; paper bank GBP 1,000 to GBP 997.30; 45% right way |
| EV rule, crossing variant | 21 | -1.2% | CLV +0.1%; 52% right way |
| Stacked move rule | 8 | -3.7% | 50% right way |
| Stacked rule, resting | 6 | -7.0% | 33% right way |

- 314 forecasts, 29 signals; skipped: 11 unfilled by the deadline, 3 books too wide.
- **Adverse selection again:** of the EV rule's resting orders, the filled ones went the right way 45% of the time and
  the unfilled ones 60%. A resting order is filled mostly when the market is about to move through it.
- Eleven bets is noise. The gate needs 300 bets over 90 days, and nothing is decided before it.

**The closing-line model on the books, with our forecast joined:**
- 2,155 rows, 4-10 Nov; 321 rows carry the EV model's forecast (162 lab matches in the window).
- What is left of the model's forecast move does not yet track the move to the close: correlation -0.02 (321 rows).
- The walk-forward model: correlation 0.06 overall (0.10 at 60 minutes), out-of-sample R2 -0.10.

| Entry rule | Trades | Return |
|---|---|---|
| Spread-aware gate | 212 | -1.9% (-2.6 to -1.2) |
| Flat-tick gate | 680 | -2.5% |
| Every tradable line | 3,036 | -2.5% |

**Read:**
- No trial makes money in its first window. Nothing is decided at 6-21 bets.
- Two patterns repeat from the research:
  - **resting fills are adversely selected:** the crossing variant beat resting here;
  - **the spread-aware gate loses about a quarter less** than trading every line.
- The model's forecast and the recorded books are joined now. That link needs weeks of matches before it can say
  whether the forecast holds at real prices.

## Model review: feature importance and the first repairs (10 Nov 2026)

The owner's ask (10 Nov): improve the model that predicts the closing price; review every feature, its calculation
and its statistics. The full review and plan are in the PR description of this change. Two runs on this branch, both
read on Betfair's own early-to-close moves of 2025/26. Those seasons were read by earlier rules, so this is
**development evidence, not a holdout**; the holdout is the matches from 10 Nov 2026, which nothing here reads.

### What the features are worth (`football_feature_report`, run 42792047861)

- Expected-value model's arm (115 features), refitted on 82,735 same-book moves 2013/14-2024/25; tested on 9,436
  Betfair matches.
- Skill against "no move": home 5.2%, away 6.9%.

| Family | Gain share | Skill lost, shuffled (points) | Skill change, left out and refitted |
|---|---|---|---|
| Short form `fs_` (32) | 21.3% | 7.2 | -0.49 |
| Team windows `d_` (33) | 26.0% | 2.2 | -0.04 |
| `dis_open` (model vs opening price) | 10.1% | 2.1 (8.2 in the top five) | -1.39 |
| xG `x_` (28) | 12.6% | 1.7 (0.5 in lower leagues) | -0.49 |
| League label | 15.0% | 0.7 | - |
| `r_diff` (market rating) | 3.9% | 0.5 | +0.00 |
| Team news `tn_` (9) | 6.7% | 0.3 | -0.31 |
| Home advantage `ha_` (11) | 4.3% | 0.0 | +0.13 |

- `dis_open` is the most valuable single feature.
- The team windows are redundant once short form is in; the home-advantage columns add nothing.
- The margin's removal (review finding A4) is not a problem. Under the three methods, the mean drift of the chance by
  early-price band is smallest for the proportional split we use (bias score 0.00042; Shin 0.00052; power 0.00063).

### Each change against the current model (`football_review_arms`, run 42793342988)

- Paired by match day; the 95% interval is from 1,000 resamples of days.
- A change keeps its place only if its interval on Betfair excludes zero.

| Change | Features | Betfair skill | vs current | 95% interval | Pinnacle 2016-24 vs current | Keeps its place |
|---|---|---|---|---|---|---|
| Current model | 115 | 6.28% | - | - | - | - |
| All changes together | 137 | 8.29% | +2.01 | +1.57 to +2.45 | +1.54 | yes |
| Move memory (`mm_`) | 120 | 7.77% | +1.49 | +1.14 to +1.85 | +1.24 | yes |
| League effects, label kept (`lf_`) | 120 | 6.61% | +0.32 | +0.18 to +0.48 | +0.32 | yes |
| Kalman form replacing raw points/goal windows (`kf_`) | 105 | 6.44% | +0.16 | +0.05 to +0.25 | +0.03 | yes |
| Kalman form added | 121 | 6.43% | +0.15 | +0.04 to +0.25 | +0.05 | yes |
| League effects, label removed | 120 | 6.43% | +0.15 | -0.04 to +0.36 | +0.23 | no |
| Early price added | 117 | 6.37% | +0.09 | -0.03 to +0.21 | +0.04 | no |
| Shrunk xG and Massey ratings | 119 | 6.35% | +0.07 | -0.04 to +0.17 | -0.06 | no |
| Early price + logit target | 117 | 6.29% | +0.00 | -0.16 to +0.18 | +0.22 | no |
| Logit target | 115 | 6.25% | -0.03 | -0.15 to +0.09 | +0.19 | no |
| League label removed | 115 | 5.89% | -0.39 | -0.54 to -0.25 | -0.21 | no (worse) |

What this says:

- **Move memory is the large gain.** Next check before it is trusted: split the side's own record from the
  price-band record, and cost the trades at real prices. The band part could be reading how the margin sits in the
  normalised chance rather than a move that can be traded.
- **The Kalman form can replace the raw window ladder.** It does so with 10 fewer columns and no loss.
- **Three of the review's findings are not supported on Betfair moves:**
  - A4: margin removal;
  - A5: a logit target or the price level as a feature;
  - A1: shrinking the xG and Massey ratings. The fault is real in the code (a test shows a side with two matches rated
    four goals above average), but the fix adds nothing measurable to this model.
- Nothing here changes the frozen models or the paper trials. Any new model is frozen and pre-registered before it is
  scored on the forward data.
