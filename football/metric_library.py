"""racing² football metric library: every metric family we know of, what it measures, and whether we hold it.

The owner, 24 Nov 2026: "Create a library of all football metrics in the world". One entry a metric or closely tied
group: (family, name, what it measures, the data it needs, status, where). Status:

- ``built``: computed in this repository; ``where`` is (file, a name found in that file) and
  ``tests/test_football_metric_library.py`` checks it.
- ``testing``: built on 24 Nov 2026 and scored by ``football-feature-batch-1124`` or ``football-tactics-batch-1124`` or ``football-sportmonks-xg-1124`` or ``football-fd-lines-1124`` or ``football-sportmonks-pred-1124`` or ``football-eu-form-1124``; not yet in any model.
- ``held``: we hold the data and have not built the metric.
- ``need``: we do not hold the data.

    python -m football.metric_library --out docs/football_metric_library.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TF, XG, PR, WH = "football/team_features.py", "football/xg_features.py", "football/player_ratings.py", "football/warehouse/"
FD = "football-data.co.uk results and match counts"
BF = "Betfair historic price files"
AF = "API-Football"
REC = "our recorded Betfair books"

E = [
    # ---- 1. the market --------------------------------------------------------------------------------------------
    ("Market", "Fair chances (margin removed)", "the price as a probability, the bookmaker's margin split out", "prices", "built", (TF, "def devig")),
    ("Market", "Implied goal rates (market xG, xGA)", "the two goal rates that reproduce the 1X2 and over/under prices", "1X2 + O/U prices", "built", (TF, "def implied_rates")),
    ("Market", "Implied total, supremacy, draw shape (rho)", "goals expected, the margin between the sides, how far the draw price sits from independent Poisson", BF, "built", (WH + "gold_implied.py", "rho_from_draw")),
    ("Market", "Handicap against 1X2 gap", "the Asian handicap's supremacy less the match odds'", BF, "built", (WH + "gold_implied.py", "sup_gap")),
    ("Market", "Market-implied team ratings", "attack and defence solved from many closing prices", "closing prices", "built", (WH + "gold_rating.py", "strength_end")),
    ("Market", "Market xPts", "points the closing price expects: 3 x win + draw", "closing prices", "built", (WH + "club.py", "xpts_pre")),
    ("Market", "Market xG / xGA / xGD / xPts as rolling form", "a side's recent closes, time-decayed: level and trend", "closing prices", "testing", ("football/blocks/mkt_xg.py", "mx_")),
    ("Market", "Opening-to-closing move (CLV)", "how far the price moved after a bet could be struck", "open + close", "built", ("football/lab.py", "y_h")),
    ("Market", "Model against the opening price", "our ratings' price less the market's (dis_open)", "ratings + open", "built", ("football/feature_eval.py", "dis_open")),
    ("Market", "Price path: returns, volatility, jumps, run-up, drawdown, slope", "the shape of a price's history before a checkpoint", BF, "built", (WH + "gold_path.py", "drawdown")),
    ("Market", "Move memory", "how a side's, league's or price band's prices have moved before", "open + close history", "built", ("football/blocks/move_memory.py", "mm_")),
    ("Market", "Soft books against the sharp book, dispersion", "bookmakers' disagreement at the open", "several bookmakers", "built", ("football/blocks/book_gaps.py", "FEATURES")),
    ("Market", "Market average against Betfair", "the average book's price less the exchange's", "bookmaker average + Betfair", "built", ("football/blocks/avg_anchor.py", "FEATURES")),
    ("Market", "Margin, favourite, draw price, early gap", "the state of the book", "prices", "built", ("football/blocks/market_state.py", "FEATURES")),
    ("Market", "Betfair against the average on the handicap and goals; handicap line moves; books pricing the match", "the exchange's two-way prices against the bookmakers', each side's history of line moves, a liquidity count", FD, "built", ("football/blocks/fd_lines.py", "fl_line_move_diff")),
    ("Market", "Spread, depth, queue, traded money, microprice, order flow", "the exchange book itself", REC, "built", ("football/feed_features.py", "def ")),
    ("Market", "Robust close", "the median of the last prints, not the last one", BF, "built", ("research/open_close/football_bf_robust_close.py", "def robust_gap")),
    ("Market", "Bookmakers' 1X2 consensus near the kick-off, incl. European and international matches", "the mean margin-free chance of up to 19 bookmakers, the best price, the number of books", "Sportmonks Odds & Predictions bundle", "built", ("sources/sportmonks.py", "def odds_1x2")),
    ("Market", "A provider's own match prediction against the opening price", "Sportmonks' result, over/under 2.5 and both-teams-to-score chances less the price's", "Sportmonks Odds & Predictions bundle (timestamped snapshots from 24 Nov 2026)", "testing", ("football/blocks/sm_pred.py", "sp_o25")),
    ("Market", "Volume-weighted price, volume profile in history", "traded money at each price back through the years", "Betfair ADVANCED plan", "need", None),
    # ---- 2. results and ratings -----------------------------------------------------------------------------------
    ("Results and ratings", "Points, goal difference, shots-on-target difference (windows)", "raw form over the last 5, 10, season", FD, "built", (TF, "def windows")),
    ("Results and ratings", "Short form (last 2, 3, weighted 5 and 10)", "the same measures over shorter windows", FD, "built", ("football/blocks/form_short.py", "FEATURES")),
    ("Results and ratings", "Performance against the market (points, goals)", "result less what the closing price expected", FD + " + closes", "built", (TF, "gae_mkt")),
    ("Results and ratings", "Performance against our model", "result less what Dixon-Coles expected", FD, "built", (TF, "gae_dc")),
    ("Results and ratings", "Dixon-Coles attack and defence", "Poisson team ratings with a low-score correction and time decay", FD, "built", ("football/ratings.py", "def score_matrix")),
    ("Results and ratings", "Massey power rating", "least-squares rating on margins", FD, "built", (XG, "def fit_massey")),
    ("Results and ratings", "Cross-league Elo", "one scale across leagues, from cups and Europe", "results incl. cups", "built", ("football/cross_league.py", "def elo_hier")),
    ("Results and ratings", "Kalman form with uncertainty", "a filtered level of performance, its variance and latest surprise", FD + " + closes", "built", ("football/blocks/kalman_form.py", "kf_")),
    ("Results and ratings", "Time-decayed form (30, 120, 365 days)", "form weighted by days ago, not matches ago", FD + " + closes", "testing", ("football/blocks/decay_form.py", "dk_")),
    ("Results and ratings", "Form normalised within the league's round", "percentile and z-score among the league's sides that week", FD + " + closes", "testing", ("football/blocks/norm_rank.py", "nz_")),
    ("Results and ratings", "Pythagorean expectation", "points expected from goals for and against", FD, "held", None),
    ("Results and ratings", "Glicko / rating deviation", "a rating with its own uncertainty", FD, "held", None),
    ("Results and ratings", "Bayesian hierarchical team strength", "attack and defence with league-level priors, time-varying", FD, "held", None),
    # ---- 3. schedule, venue, context ------------------------------------------------------------------------------
    ("Schedule and venue", "Strength of schedule", "the mean rating of recent opponents", "ratings", "built", (TF, "opp_r")),
    ("Schedule and venue", "Record against better and worse sides", "performance split by the opponent's rating against one's own", "ratings + results", "testing", ("football/blocks/sos_split.py", "ss_")),
    ("Schedule and venue", "League home advantage and its drift", "the league's home goal edge over 190 to 1,900 matches", FD, "built", ("football/blocks/home_adv.py", "ha_")),
    ("Schedule and venue", "Home form at home, away form away", "venue-specific form", FD, "built", ("football/blocks/venue_form.py", "FEATURES")),
    ("Schedule and venue", "Team home and away effect", "a side's own home edge beyond the league's, shrunk", FD + " + closes", "testing", ("football/blocks/venue_fx.py", "hv_")),
    ("Schedule and venue", "Rest days and fixture congestion", "days since the last match, matches in the last weeks", "fixture list", "built", ("football/blocks/rest_days.py", "FEATURES")),
    ("Schedule and venue", "European matches in a club's form: xG, points above the market, load, rest", "Champions, Europa and Conference League games before the collection day, for the club's league matches", "Sportmonks Euro Club Tournaments (2024/25 on)", "testing", ("football/blocks/eu_form.py", "eu_rest_min")),
    ("Schedule and venue", "Squad load (minutes in 7 and 14 days)", "the expected eleven's recent minutes", AF + " line-ups", "built", ("football/blocks/fatigue.py", "FEATURES")),
    ("Schedule and venue", "Promoted and relegated sides, early season", "tier change flags and their fade", FD, "built", ("football/blocks/home_adv.py", "promo")),
    ("Schedule and venue", "League effects", "shrunk league and tier drift in place of a raw label", FD, "built", ("football/blocks/league_fx.py", "lf_")),
    ("Schedule and venue", "Referee record", "home edge and cards under each referee", FD + " (England and a few others)", "testing", ("football/blocks/referee.py", "rf_")),
    ("Schedule and venue", "Travel distance, time zones, altitude", "the away side's journey", "stadium coordinates", "need", None),
    ("Schedule and venue", "Weather and pitch", "wind, rain, temperature, surface", "NOAA station data is held for racing; stadium coordinates needed", "need", None),
    ("Schedule and venue", "Attendance and crowd", "crowd size against capacity", "attendance feed", "need", None),
    ("Schedule and venue", "Motivation: league position stakes, dead rubbers, derbies", "what the match is worth to each side", "league tables (derivable from results)", "held", None),
    # ---- 4. shots and expected goals ------------------------------------------------------------------------------
    ("Shots and xG", "xG, xGA, xGD (non-penalty)", "chance quality created and conceded", AF + " match statistics (2022/23 on)", "built", (XG, "npxg")),
    ("Shots and xG", "xG difference against the market", "xG difference less the closing price's expected one", AF + " + closes", "built", (XG, "npxgd_vs_mkt")),
    ("Shots and xG", "Schedule-adjusted xG", "xG difference less the opponent's usual", AF, "built", (XG, "npxgd_sos")),
    ("Shots and xG", "Score-state adjusted xG", "xG reweighted for the time spent ahead or behind", AF + " events", "built", (XG, "npxgd_adj")),
    ("Shots and xG", "Box-shot share, possession", "territory proxies", AF, "built", (XG, "box_share")),
    ("Shots and xG", "Finishing and goalkeeping over expected, shrunk", "goals less xG, goals prevented, with a sample-size prior", AF, "built", (XG, "keeper_shrunk")),
    ("Shots and xG", "xG team ratings", "attack and defence fitted on xG", AF, "built", (XG, "def fit_xg_ratings")),
    ("Shots and xG", "Shot xG model, expected threat grid", "our own shot model and a 16 x 12 threat grid", "Wyscout public events (2017/18)", "built", ("football/event_models.py", "xt")),
    ("Shots and xG", "xPts from xG", "points expected from the two sides' xG (simulated scorelines)", AF, "held", None),
    ("Shots and xG", "Sportmonks xG suite as form (xGoT, npxG, open/set-play xG, xPTS, big chances, dangerous attacks)", "a second provider's xG and statistics, time-decayed, against the market", "Sportmonks Pro + xG bundle (2024/25 on, 35 leagues)", "testing", ("football/blocks/sm_xg.py", "sx_xgd_mkt_diff")),
    ("Shots and xG", "Proxy xG from shot counts, back to 2012", "an xG stand-in where no provider xG exists", FD, "held", None),
    ("Shots and xG", "Post-shot xG (PSxG), shot placement", "chance quality after the strike", "shot-level event data with coordinates", "need", None),
    ("Shots and xG", "Set-piece xG, open-play xG, counter-attack xG", "xG by phase of play", "shot-level event data", "need", None),
    ("Shots and xG", "Big chances, xG per shot, shot distance", "shot quality detail", "shot-level event data", "need", None),
    # ---- 5. style and tactics -------------------------------------------------------------------------------------
    ("Style and tactics", "Shot volume, accuracy, corners, cards (for and against)", "a side's style from match counts", FD, "testing", ("football/blocks/style.py", "st_")),
    ("Style and tactics", "Style matchup: output against concession", "one side's shots for joined to the other's shots against; tempo and its clash", FD, "testing", ("football/blocks/style.py", "st_vol_leak")),
    ("Style and tactics", "Pass volume and accuracy, offsides, fouls, saves, blocked shots", "possession style and defensive line", AF + " match statistics", "held", None),
    ("Style and tactics", "Shape matchup (back line and forwards)", "the coarse shape each side is expected to start, and how such shapes have met", AF + " line-ups, positions (2020 on)", "built", ("football/blocks/formation.py", "fm_table")),
    ("Style and tactics", "Named formation (4-2-3-1 against 4-5-1) and its matchup", "the shapes named on the team sheets", AF + " line-ups (formation names collected live from Nov 2026; in the raw history files, not yet in the tables)", "held", None),
    ("Style and tactics", "PPDA, pressing intensity, high turnovers", "how hard a side presses", "event data with locations", "need", None),
    ("Style and tactics", "Field tilt, final-third entries, deep completions", "where the game is played", "event data with locations", "need", None),
    ("Style and tactics", "Directness, build-up speed, sequence length", "how a side moves the ball", "event data with sequences", "need", None),
    ("Style and tactics", "Passing networks, centrality", "who links play", "pass-level event data", "need", None),
    ("Style and tactics", "Crosses, long balls, set-piece routines", "attacking method", "event data", "need", None),
    ("Style and tactics", "Defensive line height, compactness, pitch control", "team shape off the ball", "tracking data", "need", None),
    # ---- 6. players -----------------------------------------------------------------------------------------------
    ("Players", "Plus-minus ratings (market and goals)", "each player's effect on the side's results against the market", AF + " line-ups + closes", "built", (PR, "mpm")),
    ("Players", "Expected eleven strength, absences, rotation", "who is likely to start and what they are worth", AF + " line-ups", "built", (PR, "tn_")),
    ("Players", "Line-up surprise", "the named eleven against the expected one", AF + " line-ups", "built", (PR, "mpm_surprise")),
    ("Players", "Injury and suspension adjustment", "the rating lost to named absentees, less the likely replacement's", AF + " injuries (collected daily from Nov 2026)", "held", None),
    ("Players", "Provider player ratings, minutes, age curve", "a match rating a player, his minutes and age", AF + " players", "held", None),
    ("Players", "xG and xA per 90, goals above expected", "a player's chance creation and finishing", "player-level xG", "need", None),
    ("Players", "On-ball value (VAEP, OBV, g+), xT added", "every action's worth", "event data", "need", None),
    ("Players", "Progressive passes and carries, pressures, duels", "player action counts", "event data", "need", None),
    ("Players", "Goalkeeper shot-stopping, sweeping, distribution", "keeper value beyond saves", "shot-level and event data", "need", None),
    ("Players", "Physical output: distance, sprints, high-speed running", "work rate", "tracking data", "need", None),
    ("Players", "Squad value, wages, transfer spend", "resources", "a paid squad-value feed", "need", None),
    # ---- 7. managers ----------------------------------------------------------------------------------------------
    ("Managers", "Manager effect on results against the market", "what a manager adds beyond his squad", AF + " coach names (from Nov 2026 only)", "held", None),
    ("Managers", "New-manager bounce, tenure", "the change after an appointment and how long he has been there", "a manager history feed", "need", None),
    ("Managers", "Manager style and in-game changes", "substitution timing, formation changes", AF + " events + line-ups", "held", None),
    # ---- 8. in play -----------------------------------------------------------------------------------------------
    ("In play", "Goals and their minutes from prices", "goals inferred from price jumps, reconciled to the settlement", BF, "built", (WH + "gold_goal.py", "agrees_third")),
    ("In play", "Minute-by-minute chances", "calibrated in-play win, draw, loss chances", BF, "built", (WH + "gold_inplay.py", "score_ok")),
    ("In play", "In-play xPts, performance and luck", "points expected through the match against the kick-off price", BF, "built", (WH + "club.py", "PERF_LINE")),
    ("In play", "Minute a match was decided", "when the result was priced as settled", BF, "built", (WH + "club.py", "decided_min")),
    ("In play", "Game state: time and goals level, ahead, behind", "minutes and goals in each state", AF + " events", "built", ("football/tactics.py", "def goal_states")),
    ("In play", "State multipliers: scoring when behind, holding a lead", "a side's scoring and conceding rate in each state against its overall rate, shrunk", AF + " events", "built", ("football/blocks/match_sim.py", "gx_lead_against_diff")),
    ("In play", "Momentum, xG flow, win-probability added", "who is on top within the match", "live event data", "need", None),
    # ---- 9. simulation --------------------------------------------------------------------------------------------
    ("Simulation", "Scoreline simulation from venue records", "Poisson score matrix from each side's venue scoring, the market as prior", FD + " + opens", "testing", ("football/blocks/sim.py", "sm_")),
    ("Simulation", "Match simulator with game states", "the 90 minutes as a chain over the goal difference, each side's rate reacting to the score", AF + " events + opens", "built", ("football/blocks/match_sim.py", "def play")),
    ("Simulation", "Simulator with red cards, substitutions and time-varying intensity", "the same chain with men on the pitch and the minute", AF + " events", "held", None),
    ("Simulation", "Season simulation: title, top four, relegation chances", "the league played out many times", "ratings + fixture list", "held", None),
]

STATUS = {"built": "Built", "testing": "Built 24 Nov, under test", "held": "Data held, not built", "need": "Data not held"}


def render() -> str:
    n = Counter(e[4] for e in E)
    out = ["# racing² - football metric library", "",
           "Every metric family we know of, what it measures, and whether we hold it. Generated from "
           "`football/metric_library.py`; `tests/test_football_metric_library.py` checks every built entry against "
           "the code.", "",
           f"**{len(E)} entries**: {n['built']} built, {n['testing']} built on 24 Nov 2026 and under test "
           f"(`football-feature-batch-1124`, `football-tactics-batch-1124`, `football-sportmonks-xg-1124`, `football-fd-lines-1124`, `football-sportmonks-pred-1124`, `football-eu-form-1124`), {n['held']} where we hold the data and have not built the metric, "
           f"{n['need']} where we do not hold the data.", ""]
    fam = None
    for f, name, what, data, status, where in E:
        if f != fam:
            fam = f
            out += ["", f"## {f}", "", "| Metric | What it measures | Data | Status | Where |", "|---|---|---|---|---|"]
        out.append(f"| {name} | {what} | {data} | {STATUS[status]} | {('`' + where[0] + '`') if where else '-'} |")
    need = sorted({e[3] for e in E if e[4] == "need"})
    out += ["", "## What the missing data would be", ""] + [f"- {d}" for d in need] + [""]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Path(a.out).write_text(render())
    print(Counter(e[4] for e in E))
    return 0


if __name__ == "__main__":
    sys.exit(main())
