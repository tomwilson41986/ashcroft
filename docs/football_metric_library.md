# racing² - football metric library

Every metric family we know of, what it measures, and whether we hold it. Generated from `football/metric_library.py`; `tests/test_football_metric_library.py` checks every built entry against the code.

**93 entries**: 49 built, 9 built on 24 Nov 2026 and under test (`football-feature-batch-1124`, `football-tactics-batch-1124`), 14 where we hold the data and have not built the metric, 21 where we do not hold the data.


## Market

| Metric | What it measures | Data | Status | Where |
|---|---|---|---|---|
| Fair chances (margin removed) | the price as a probability, the bookmaker's margin split out | prices | Built | `football/team_features.py` |
| Implied goal rates (market xG, xGA) | the two goal rates that reproduce the 1X2 and over/under prices | 1X2 + O/U prices | Built | `football/team_features.py` |
| Implied total, supremacy, draw shape (rho) | goals expected, the margin between the sides, how far the draw price sits from independent Poisson | Betfair historic price files | Built | `football/warehouse/gold_implied.py` |
| Handicap against 1X2 gap | the Asian handicap's supremacy less the match odds' | Betfair historic price files | Built | `football/warehouse/gold_implied.py` |
| Market-implied team ratings | attack and defence solved from many closing prices | closing prices | Built | `football/warehouse/gold_rating.py` |
| Market xPts | points the closing price expects: 3 x win + draw | closing prices | Built | `football/warehouse/club.py` |
| Market xG / xGA / xGD / xPts as rolling form | a side's recent closes, time-decayed: level and trend | closing prices | Built 24 Nov, under test | `football/blocks/mkt_xg.py` |
| Opening-to-closing move (CLV) | how far the price moved after a bet could be struck | open + close | Built | `football/lab.py` |
| Model against the opening price | our ratings' price less the market's (dis_open) | ratings + open | Built | `football/feature_eval.py` |
| Price path: returns, volatility, jumps, run-up, drawdown, slope | the shape of a price's history before a checkpoint | Betfair historic price files | Built | `football/warehouse/gold_path.py` |
| Move memory | how a side's, league's or price band's prices have moved before | open + close history | Built | `football/blocks/move_memory.py` |
| Soft books against the sharp book, dispersion | bookmakers' disagreement at the open | several bookmakers | Built | `football/blocks/book_gaps.py` |
| Market average against Betfair | the average book's price less the exchange's | bookmaker average + Betfair | Built | `football/blocks/avg_anchor.py` |
| Margin, favourite, draw price, early gap | the state of the book | prices | Built | `football/blocks/market_state.py` |
| Spread, depth, queue, traded money, microprice, order flow | the exchange book itself | our recorded Betfair books | Built | `football/feed_features.py` |
| Robust close | the median of the last prints, not the last one | Betfair historic price files | Built | `research/open_close/football_bf_robust_close.py` |
| Volume-weighted price, volume profile in history | traded money at each price back through the years | Betfair ADVANCED plan | Data not held | - |

## Results and ratings

| Metric | What it measures | Data | Status | Where |
|---|---|---|---|---|
| Points, goal difference, shots-on-target difference (windows) | raw form over the last 5, 10, season | football-data.co.uk results and match counts | Built | `football/team_features.py` |
| Short form (last 2, 3, weighted 5 and 10) | the same measures over shorter windows | football-data.co.uk results and match counts | Built | `football/blocks/form_short.py` |
| Performance against the market (points, goals) | result less what the closing price expected | football-data.co.uk results and match counts + closes | Built | `football/team_features.py` |
| Performance against our model | result less what Dixon-Coles expected | football-data.co.uk results and match counts | Built | `football/team_features.py` |
| Dixon-Coles attack and defence | Poisson team ratings with a low-score correction and time decay | football-data.co.uk results and match counts | Built | `football/ratings.py` |
| Massey power rating | least-squares rating on margins | football-data.co.uk results and match counts | Built | `football/xg_features.py` |
| Cross-league Elo | one scale across leagues, from cups and Europe | results incl. cups | Built | `football/cross_league.py` |
| Kalman form with uncertainty | a filtered level of performance, its variance and latest surprise | football-data.co.uk results and match counts + closes | Built | `football/blocks/kalman_form.py` |
| Time-decayed form (30, 120, 365 days) | form weighted by days ago, not matches ago | football-data.co.uk results and match counts + closes | Built 24 Nov, under test | `football/blocks/decay_form.py` |
| Form normalised within the league's round | percentile and z-score among the league's sides that week | football-data.co.uk results and match counts + closes | Built 24 Nov, under test | `football/blocks/norm_rank.py` |
| Pythagorean expectation | points expected from goals for and against | football-data.co.uk results and match counts | Data held, not built | - |
| Glicko / rating deviation | a rating with its own uncertainty | football-data.co.uk results and match counts | Data held, not built | - |
| Bayesian hierarchical team strength | attack and defence with league-level priors, time-varying | football-data.co.uk results and match counts | Data held, not built | - |

## Schedule and venue

| Metric | What it measures | Data | Status | Where |
|---|---|---|---|---|
| Strength of schedule | the mean rating of recent opponents | ratings | Built | `football/team_features.py` |
| Record against better and worse sides | performance split by the opponent's rating against one's own | ratings + results | Built 24 Nov, under test | `football/blocks/sos_split.py` |
| League home advantage and its drift | the league's home goal edge over 190 to 1,900 matches | football-data.co.uk results and match counts | Built | `football/blocks/home_adv.py` |
| Home form at home, away form away | venue-specific form | football-data.co.uk results and match counts | Built | `football/blocks/venue_form.py` |
| Team home and away effect | a side's own home edge beyond the league's, shrunk | football-data.co.uk results and match counts + closes | Built 24 Nov, under test | `football/blocks/venue_fx.py` |
| Rest days and fixture congestion | days since the last match, matches in the last weeks | fixture list | Built | `football/blocks/rest_days.py` |
| Squad load (minutes in 7 and 14 days) | the expected eleven's recent minutes | API-Football line-ups | Built | `football/blocks/fatigue.py` |
| Promoted and relegated sides, early season | tier change flags and their fade | football-data.co.uk results and match counts | Built | `football/blocks/home_adv.py` |
| League effects | shrunk league and tier drift in place of a raw label | football-data.co.uk results and match counts | Built | `football/blocks/league_fx.py` |
| Referee record | home edge and cards under each referee | football-data.co.uk results and match counts (England and a few others) | Built 24 Nov, under test | `football/blocks/referee.py` |
| Travel distance, time zones, altitude | the away side's journey | stadium coordinates | Data not held | - |
| Weather and pitch | wind, rain, temperature, surface | NOAA station data is held for racing; stadium coordinates needed | Data not held | - |
| Attendance and crowd | crowd size against capacity | attendance feed | Data not held | - |
| Motivation: league position stakes, dead rubbers, derbies | what the match is worth to each side | league tables (derivable from results) | Data held, not built | - |

## Shots and xG

| Metric | What it measures | Data | Status | Where |
|---|---|---|---|---|
| xG, xGA, xGD (non-penalty) | chance quality created and conceded | API-Football match statistics (2022/23 on) | Built | `football/xg_features.py` |
| xG difference against the market | xG difference less the closing price's expected one | API-Football + closes | Built | `football/xg_features.py` |
| Schedule-adjusted xG | xG difference less the opponent's usual | API-Football | Built | `football/xg_features.py` |
| Score-state adjusted xG | xG reweighted for the time spent ahead or behind | API-Football events | Built | `football/xg_features.py` |
| Box-shot share, possession | territory proxies | API-Football | Built | `football/xg_features.py` |
| Finishing and goalkeeping over expected, shrunk | goals less xG, goals prevented, with a sample-size prior | API-Football | Built | `football/xg_features.py` |
| xG team ratings | attack and defence fitted on xG | API-Football | Built | `football/xg_features.py` |
| Shot xG model, expected threat grid | our own shot model and a 16 x 12 threat grid | Wyscout public events (2017/18) | Built | `football/event_models.py` |
| xPts from xG | points expected from the two sides' xG (simulated scorelines) | API-Football | Data held, not built | - |
| Proxy xG from shot counts, back to 2012 | an xG stand-in where no provider xG exists | football-data.co.uk results and match counts | Data held, not built | - |
| Post-shot xG (PSxG), shot placement | chance quality after the strike | shot-level event data with coordinates | Data not held | - |
| Set-piece xG, open-play xG, counter-attack xG | xG by phase of play | shot-level event data | Data not held | - |
| Big chances, xG per shot, shot distance | shot quality detail | shot-level event data | Data not held | - |

## Style and tactics

| Metric | What it measures | Data | Status | Where |
|---|---|---|---|---|
| Shot volume, accuracy, corners, cards (for and against) | a side's style from match counts | football-data.co.uk results and match counts | Built 24 Nov, under test | `football/blocks/style.py` |
| Style matchup: output against concession | one side's shots for joined to the other's shots against; tempo and its clash | football-data.co.uk results and match counts | Built 24 Nov, under test | `football/blocks/style.py` |
| Pass volume and accuracy, offsides, fouls, saves, blocked shots | possession style and defensive line | API-Football match statistics | Data held, not built | - |
| Shape matchup (back line and forwards) | the coarse shape each side is expected to start, and how such shapes have met | API-Football line-ups, positions (2020 on) | Built | `football/blocks/formation.py` |
| Named formation (4-2-3-1 against 4-5-1) and its matchup | the shapes named on the team sheets | API-Football line-ups (formation names collected live from Nov 2026; in the raw history files, not yet in the tables) | Data held, not built | - |
| PPDA, pressing intensity, high turnovers | how hard a side presses | event data with locations | Data not held | - |
| Field tilt, final-third entries, deep completions | where the game is played | event data with locations | Data not held | - |
| Directness, build-up speed, sequence length | how a side moves the ball | event data with sequences | Data not held | - |
| Passing networks, centrality | who links play | pass-level event data | Data not held | - |
| Crosses, long balls, set-piece routines | attacking method | event data | Data not held | - |
| Defensive line height, compactness, pitch control | team shape off the ball | tracking data | Data not held | - |

## Players

| Metric | What it measures | Data | Status | Where |
|---|---|---|---|---|
| Plus-minus ratings (market and goals) | each player's effect on the side's results against the market | API-Football line-ups + closes | Built | `football/player_ratings.py` |
| Expected eleven strength, absences, rotation | who is likely to start and what they are worth | API-Football line-ups | Built | `football/player_ratings.py` |
| Line-up surprise | the named eleven against the expected one | API-Football line-ups | Built | `football/player_ratings.py` |
| Injury and suspension adjustment | the rating lost to named absentees, less the likely replacement's | API-Football injuries (collected daily from Nov 2026) | Data held, not built | - |
| Provider player ratings, minutes, age curve | a match rating a player, his minutes and age | API-Football players | Data held, not built | - |
| xG and xA per 90, goals above expected | a player's chance creation and finishing | player-level xG | Data not held | - |
| On-ball value (VAEP, OBV, g+), xT added | every action's worth | event data | Data not held | - |
| Progressive passes and carries, pressures, duels | player action counts | event data | Data not held | - |
| Goalkeeper shot-stopping, sweeping, distribution | keeper value beyond saves | shot-level and event data | Data not held | - |
| Physical output: distance, sprints, high-speed running | work rate | tracking data | Data not held | - |
| Squad value, wages, transfer spend | resources | a paid squad-value feed | Data not held | - |

## Managers

| Metric | What it measures | Data | Status | Where |
|---|---|---|---|---|
| Manager effect on results against the market | what a manager adds beyond his squad | API-Football coach names (from Nov 2026 only) | Data held, not built | - |
| New-manager bounce, tenure | the change after an appointment and how long he has been there | a manager history feed | Data not held | - |
| Manager style and in-game changes | substitution timing, formation changes | API-Football events + line-ups | Data held, not built | - |

## In play

| Metric | What it measures | Data | Status | Where |
|---|---|---|---|---|
| Goals and their minutes from prices | goals inferred from price jumps, reconciled to the settlement | Betfair historic price files | Built | `football/warehouse/gold_goal.py` |
| Minute-by-minute chances | calibrated in-play win, draw, loss chances | Betfair historic price files | Built | `football/warehouse/gold_inplay.py` |
| In-play xPts, performance and luck | points expected through the match against the kick-off price | Betfair historic price files | Built | `football/warehouse/club.py` |
| Minute a match was decided | when the result was priced as settled | Betfair historic price files | Built | `football/warehouse/club.py` |
| Game state: time and goals level, ahead, behind | minutes and goals in each state | API-Football events | Built | `football/tactics.py` |
| State multipliers: scoring when behind, holding a lead | a side's scoring and conceding rate in each state against its overall rate, shrunk | API-Football events | Built | `football/blocks/match_sim.py` |
| Momentum, xG flow, win-probability added | who is on top within the match | live event data | Data not held | - |

## Simulation

| Metric | What it measures | Data | Status | Where |
|---|---|---|---|---|
| Scoreline simulation from venue records | Poisson score matrix from each side's venue scoring, the market as prior | football-data.co.uk results and match counts + opens | Built 24 Nov, under test | `football/blocks/sim.py` |
| Match simulator with game states | the 90 minutes as a chain over the goal difference, each side's rate reacting to the score | API-Football events + opens | Built | `football/blocks/match_sim.py` |
| Simulator with red cards, substitutions and time-varying intensity | the same chain with men on the pitch and the minute | API-Football events | Data held, not built | - |
| Season simulation: title, top four, relegation chances | the league played out many times | ratings + fixture list | Data held, not built | - |

## What the missing data would be

- Betfair ADVANCED plan
- NOAA station data is held for racing; stadium coordinates needed
- a manager history feed
- a paid squad-value feed
- attendance feed
- event data
- event data with locations
- event data with sequences
- live event data
- pass-level event data
- player-level xG
- shot-level and event data
- shot-level event data
- shot-level event data with coordinates
- stadium coordinates
- tracking data
