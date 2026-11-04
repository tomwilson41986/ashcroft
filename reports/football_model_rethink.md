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
