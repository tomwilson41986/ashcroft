"""gold/goal and gold/timeline: the goals of a match and its timeline, inferred from the in-play prices and
reconciled with how the match's markets settled.

Betfair's files carry no score. They carry what the score does to prices, and three markets read it differently:

- **match odds** (home, draw, away): a goal moves the scorer's price in and the other side's out. Large when the
  match is open, nothing when it is decided (3-0 to 4-0), and the last minutes' time decay looks the same;
- **over/under 2.5**: every goal moves Under out, whoever scored, until the third goal closes the market. A red
  card or a missed penalty moves match odds and leaves Under where it was: the test that tells them apart;
- **Asian handicap**: every line near evens says where the final margin is expected to be (``margin``: home goals
  less away goals, from the line and its price under a normal margin). A goal moves it by about one, whatever the
  score: the only reading that still works at 3-0.

BASIC files publish about once a minute, so a suspension shorter than the gap between two messages is never seen:
most goals show as a jump between two consecutive price messages with no suspension spell at all (of the timed
goals built from 2026, 30 per cent have one). The unit here is therefore the **row** (one price message of the
match: its match-odds and over/under prices, forward-filled), not the suspension:

1. each runner's in-play prices are median-filtered over three updates (a stray trade is not a jump) and laid on
   the match's rows; a price is the logit of 1 / price, so moves are comparable at 1.05 and at 20;
2. a **window** is two consecutive rows (a goal's move is often split over two messages): the move of each runner
   from the row before the window to its second row, read only on runners that traded inside it;
3. a window is a goal when (a) Under moved out and match odds moved, or (b) the handicap margin stepped and match
   odds moved the same way, or (c) with the over/under silent (closed after three goals, or not trading), match
   odds moved far, by more than the windows around it (time decay moves every window) and in probability as well
   as in logit, or (d) the over/under market stopped for good in the window (the third goal) and match odds moved.
   Overlapping windows are thinned to the strongest (non-maximum suppression), so two goals two minutes apart
   stay two;
4. a later window in which Under comes sharply back in undoes the goal before it (a goal disallowed after the
   market had reopened);
5. a step of the handicap margin with no such window inside it is a goal too (``evidence`` 'ah': the goals that
   match odds cannot see), and one that brackets a single window settles that goal's side; a step and its
   reverse within eight minutes with no goal between them are a goal given and taken back, not two goals;
6. **the settlement**. Steps 1 to 5 are the price reading. The files also say how the markets settled: the 1X2
   winner, Over or Under 2.5, and (off which handicap lines won) the goal difference, exact up to three and
   'four or more' beyond (the lines stop there). ``agrees_result``, ``agrees_over25``, ``agrees_margin`` and
   ``consistent`` record whether the price reading alone passed those checks (71 per cent of usable matches pass
   all three). Where it did not, the list is edited at the least cost that makes it pass (``_solve``): a weakly
   evidenced goal dropped, a sub-threshold window promoted (the candidates: the same rules at looser thresholds,
   ``LOOSE``), a side turned, or a goal added that no price move shows (``evidence`` 'settle', ``timed`` false:
   it is placed after the last price, and its minute is not known). The prices of the edits are log odds of the
   measured shares in ``KEEP`` and ``CANDIDATE``. ``by_settlement`` marks the goals this step put in or turned,
   ``reconciled``, ``n_edits`` and ``n_unplaced`` say what it did to a match.

What the settlement cannot show: once the over/under market has closed (three goals) a goal each side leaves the
margin and every other check as they were, so at 3-1 a late 4-1 and 4-2 with no price move are both lost. That is
the main error left (below), and ``reliability`` falls with the goals in the list for that reason.

``usable`` is the data rule (enough rows, no long blind gap, a whole match, in play from the kick-off); it does
not look at the settlement. ``reliability`` is the measured share of matches like this one whose final score was
exactly right; ``confidence`` the measured share of goals like this one that were real.

Minutes: ``elapsed_min`` is wall-clock time since the market went in play. Half time is not in the files: the
break is found per match as the quietest stretch of 12 to 20 minutes starting 43 to 57 minutes in (prices stop
decaying, so they update less: about 2 messages a minute against 4.5), and the restart is read from that
stretch's end together with the minutes from the turn in play to the close (``_restart``; where no stretch is
found, from those minutes alone). ``match_min`` is the football minute: the elapsed minute in the first half (at
most 45), 45 plus the minutes since the restart in the second (at most 90).

**Measured accuracy** (11 Nov 2026, on the warehouse built from the 2026 download: 52,635 matches in play,
137,410 goals). Every threshold, price and table here (``PARAMS``, ``LOOSE``, ``KEEP``, ``CANDIDATE``,
``UNSEEN_COST``, ``TURN_COST``, the restart's weights, ``CONFIDENCE``, ``RELIABILITY``) was set on matches that
kicked off in odd-numbered months; the figures below are from even-numbered months only, on usable matches:

- against the settlement (20,345 matches): the price reading alone agrees with Over/Under 2.5 on 89.75 per cent,
  the 1X2 result on 89.50, the handicap margin on 72.75, all three on 70.71 (82.7 with 200 or more in-play
  match-odds updates, 49.5 with 50 to 99). After reconciliation 99.99 per cent agree, by construction: that is
  no longer a check, which is why the real scores below are;
- against API-Football goal events (2,681 matches of 22 divisions, through the warehouse's own link): the final
  score is exactly right on 94.3 per cent (97.1 where the price reading agreed, 85.1 after one edit, 73.7 after
  several, 85.7 with an unplaced goal). Of 7,115 real goals 89.6 per cent have a timed inferred goal within four
  minutes (95.1 counting the unplaced goals), and of 6,677 timed inferred goals 95.5 per cent have a real one
  (91.6 and 97.6 per cent when a match's second half may shift as a whole: the rest is the clock). The side is
  right on 99.2 per cent of matched goals. ``match_min`` is within 0.5 of the real minute at the median and 2.1
  at the 90th percentile (first half 0.5 and 1.5, second half 0.9 and 2.5). Of 485 real red cards 23 (4.7 per
  cent) have a false goal within two minutes before to four after, 8 of them 'ah' goals; none of 70 missed
  penalties has;
- against football-data.co.uk scores (4,201 matches of 37 competitions): full time exactly right on 94.1 per
  cent, half time (the timed goals of the first half) on 96.0, both on 92.3. Not usable matches: 69.1 per cent;
- ``confidence`` by band, stated against real (API-Football): 0.6 to 0.8: 0.74 against 0.78; 0.8 to 0.9: 0.86
  against 0.90; 0.9 to 0.96: 0.95 against 0.96; 0.96 to 0.985: 0.98 against 0.97; above: 0.99 against 0.99.
  ``reliability``, stated against real (football-data.co.uk): 0.995 against 0.996, 0.95 against 0.955, 0.92
  against 0.928 and 0.91 against 0.893 for the agreed matches of 0 to 2, 3, 4 and 5 or more goals; the cells of
  edited matches rest on 6 to 184 matches each and three are overstated (one edit and three goals: 0.75 against
  0.65; one edit and five or more: 0.86 against 0.71; an unplaced goal and five or more: 0.71 against 0.56);
- the errors: of 154 wrong scores among the API-Football matches 111 are a goal each side missed with every
  check passing, and 121 are in matches of five or more real goals (36 per cent of those are wrong, against 1.4
  per cent of the rest).

The truth has its own limits, and the figures inherit them: API-Football stamps most stoppage-time goals 45 or 90
with no added minute (such a goal is matched anywhere from that minute to the half's end), its period times are
nominal (so the second half's real goals are put on the wall clock through the restart inferred here), and both
sources cover leagues that trade well: 63 per cent of those matches have 200 or more in-play match-odds updates
against 30 per cent of all usable matches. Lower leagues are checked by the settlement alone.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from football.warehouse.schema import TS, Column, TableSpec

VERSION = 3
SCOPE = "part"
INPUTS = ("gold/event", "silver/market", "silver/runner", "silver/spell", "silver/tick")
COLUMNS = {
    "gold/event": ["event_id", "mo_market_id", "ou_market_id", "ah_market_id", "went_in_play", "kickoff_actual",
                   "ft_pt", "result", "over25", "void", "p_home", "p_draw", "p_away"],
    "silver/market": ["market_id", "closed_pt"],
    "silver/runner": ["market_id", "market_type", "selection_id", "handicap", "role", "won"],
    "silver/spell": ["market_id", "start_pt", "end_pt", "in_play", "status"],
    "silver/tick": ["market_id", "selection_id", "handicap", "pt", "seq", "ltp", "in_play"],
}

ROLES = ("home", "draw", "away", "under", "over")
SIDES = ("away", "", "home")                       # by sign + 1

#: the thresholds, chosen on the odd-numbered months of 2026 and scored on the even ones (the module's report)
PARAMS = {
    "ou": 0.7,            # Under's move out (logit, mean of Under and Over) that makes a window a goal ...
    "mo_min": 0.3,        # ... with a match-odds move (home logit less away logit) of at least this
    "ou_solo": 1.0,       # Under's move when no match-odds runner traded in the window
    "ah": 0.6,            # the handicap margin's step (goals) that makes a window a goal, with mo_min
    "ah_flat": 0.35,      # a handicap reading across the window under this vetoes a match-odds-only goal
    "ou_wide": 0.4,       # the over/under's next price, where it did not trade in the window: Under out by less
                          # than this says no goal to a match-odds-only window
    "mo_only": 1.5,       # match odds alone: the move ...
    "exc_only": 1.2,      # ... its excess over the same-way move of the windows before and after ...
    "p_only": 0.25,       # ... and the summed change of the three chances
    "mo_sure": 2.5,       # match odds against a flat over/under: the move ...
    "p_sure": 0.7,        # ... and the summed change of the three chances
    "mo_third": 0.6,      # match odds where the over/under market stopped for good
    "side": 0.3,          # the match-odds move that names the scorer
    "undo": 0.6,          # Under's move back in that undoes the goal before it ...
    "undo_ms": 360_000,   # ... within this long
    "ah_only": 0.7,       # a handicap step that is a goal with no window inside it
    "ah_unit": 0.9,       # the margin's step for one goal (a step of twice this is two)
    "ah_slack_ms": 90_000,
    "ah_reach_ms": 900_000,   # readings further than this from a shift do not vouch for it
    "ah_undo_ms": 480_000,    # a handicap step reversed within this long, no goal between: given and taken back
    "repair": True,
    "reconcile": True,
}
ROW_GAP_MS = 20_000           # price messages closer than this are one row (a match's markets publish together)
BEFORE_MS = 1_200_000         # a price older than this before a window is no reading of it
OU_WIDE_MS = 600_000          # the over/under's next price may come this long after a window and still be read
AH_BEFORE_MS = 600_000        # the handicap reading before a window may be this old ...
AH_AFTER_MS = 360_000         # ... and the one after must come this soon
AH_P = (0.3, 0.7)             # handicap prices this near evens are read
SIGMA0 = 1.64                 # the standard deviation of a match's margin at the kick-off, goals
PLAY_MIN = 97.0               # minutes of play in a match, added time included
HT_START_MIN = 48.0           # half time where nothing in the match says: the first half's end and the restart
HT_END_MIN = 64.0             # as the real second-half goals of 2026's odd-numbered months place them
THIRD_EARLY_MS = 180_000      # an over/under market closed this long before full time closed on the third goal
BIG = 1 << 42

#: the usable rule
USABLE_ROWS = 60              # price rows in play
USABLE_GAP_MIN = 12.0         # the longest gap between two rows
USABLE_SPAN_MIN = (100.0, 140.0)
USABLE_START = 0.30           # the summed change of the 1X2 chances from the close to the first in-play row

#: the settled margin is read off the handicap lines that won, and the lines stop at four goals
MARGIN_CAP = 4

#: sub-threshold windows: the candidates the reconciliation may add where the settlement asks for a goal
LOOSE = {"ou": 0.35, "mo_min": 0.15, "ou_solo": 0.6, "ah": 0.4, "mo_only": 0.5, "exc_only": 0.3, "p_only": 0.0,
         "mo_sure": 1.5, "p_sure": 0.3, "mo_third": 0.3}
CAND_NEAR_MS = 150_000        # a candidate this near a goal is the same goal
#: the share of each kind that was a real goal on the odd-numbered months of 2026 (API-Football goal events):
#: KEEP for the goals of matches whose price reading disagreed with the settlement, CANDIDATE for sub-threshold
#: windows (match odds alone: by the summed change of the three chances). The reconciliation's prices.
KEEP = {"ou+ah": 0.95, "ou": 0.89, "ou-ah": 0.86, "ah+mo": 0.925, "mo": 0.72, "mo-ou": 0.58, "third": 0.99,
        "ah": 0.53}
CANDIDATE = {"third": 0.60, "ou": 0.50, "ou+ah": 0.45, "mo-ou": 0.30, "ah+mo": 0.26, "ou-ah": 0.25}
CANDIDATE_MO = ((0.11, 0.19), (0.05, 0.08), (0.0, 0.05))      # (prob_jump at least, share real)
UNSEEN_COST = 2.0             # a goal no price move shows, in the same units (log odds)
TURN_COST = 4.0               # a goal given to the other side
MAX_UNSEEN = 3

#: how far to trust a goal, by its evidence and how it came to be in the list: (the price reading agreed with the
#: settlement or there was none to check; kept through a reconciliation; added by one). The share that matched a
#: real goal within four minutes on the odd-numbered months of 2026 (API-Football goal events on usable matches;
#: a few cells rest on under 50 goals: ou+ah, ou-ah, ah+mo, mo and mo-ou added, mo-ou kept). 'settle': the share
#: answered by a real goal of that side that no timed goal matched. The module's docstring gives the even months.
CONFIDENCE = {"ou+ah": (0.99, 0.98, 0.69), "ou": (0.98, 0.96, 0.84), "ou-ah": (0.97, 0.91, 0.75),
              "ah+mo": (0.99, 0.98, 0.67), "mo": (0.96, 0.86, 0.85), "mo-ou": (0.98, 0.90, 0.60),
              "third": (0.99, 0.99, 0.97), "ah": (0.87, 0.80, 0.80), "settle": (0.92, 0.92, 0.92)}
#: how often the final score was exactly the real one, by how the match's list was reached (the price reading
#: agreed; one edit; several; a goal with no price move among them) and by the goals in it (0 to 2, 3, 4, 5 or
#: more: past the third goal the over/under market is closed and a goal each side can go unseen with every check
#: still passing). Odd-numbered months of 2026, 3,898 usable matches with a football-data.co.uk score ('edits'
#: rests on 59 matches: its last three cells are pooled). The timeline's ``reliability``.
RELIABILITY = {"agreed": (0.995, 0.95, 0.92, 0.91), "edit1": (0.95, 0.75, 0.85, 0.86),
               "edits": (0.95, 0.70, 0.70, 0.70), "unplaced": (0.97, 0.97, 0.95, 0.71)}
#: the restart from the quiet stretch's end and the minutes in play (fitted on the odd-numbered months of 2026 to
#: the restart the real second-half goals imply), and from the minutes in play alone where no stretch is found
RESTART_FOUND = (2.7, 0.425, 0.29)
RESTART_END = (23.3, 0.628)
RESTART_SPAN = (6.2, 0.498)
HT_BREAK_MIN = (13.0, 16.0, 19.0)                 # the break's length: least, where not found, most

GOAL = TableSpec(
    "gold/goal",
    "One row an inferred goal: when, which side, the score after it, how sure, and the price moves that say so. "
    "Inferred from Betfair's in-play prices (match odds, over/under 2.5, Asian handicap) and reconciled with how "
    "the markets settled; not a record of the match.",
    (
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("goal_no", pa.int16(), "The goal's order in the match, 1 first: the total after it.", False),
        Column("pt", TS, "The goal's moment as the data places it: the start of the suspension that marks it, "
               "else the first price message that shows the move (the goal is up to a minute before: the gap "
               "between two messages). Where timed is false, a placeholder: after the last price.", False),
        Column("pt_before", TS, "The last price message before the move: the goal is after it."),
        Column("suspended", pa.bool_(), "Whether a suspension of the match's markets marks the goal (about one "
               "in four: a suspension shorter than the minute between messages is not in the files).", False),
        Column("timed", pa.bool_(), "False for a goal no price move shows (evidence 'settle'): the settlement "
               "says it was scored, the data does not say when; pt, elapsed_min and match_min then place it "
               "after the last price.", False),
        Column("elapsed_min", pa.float64(), "Minutes from the market's turn in play to pt, wall clock (half time "
               "included).", False),
        Column("half", pa.int8(), "1 or 2: the half pt falls in, by the match's half-time estimate.", False),
        Column("match_min", pa.float64(), "The football minute: elapsed_min in the first half (at most 45), 45 "
               "plus the minutes since the restart in the second (at most 90). The restart is an estimate: see "
               "the module's measured minute errors.", False),
        Column("side", pa.string(), "home or away: the scorer; blank where the moves do not say.", False),
        Column("confidence", pa.float64(), "How far to trust the goal, 0 to 1: the share of goals with this "
               "evidence, reached this way (the price reading agreed with the settlement, kept through a "
               "reconciliation, added by one), that matched a real goal on 2026's odd-numbered months.", False),
        Column("evidence", pa.string(), "What says it is a goal. ou+ah: Under out, match odds, and the handicap "
               "margin stepped; ou: Under out and match odds, no handicap reading; ou-ah: Under out and match "
               "odds but the handicap margin did not step; ah+mo: the margin stepped and match odds moved, the "
               "over/under silent; mo: match odds alone; mo-ou: a very large match-odds move against a flat "
               "over/under; third: the over/under market stopped for good (the third goal); ah: a handicap step "
               "with no match-odds or over/under move; settle: no price move, the settlement alone.", False),
        Column("by_settlement", pa.bool_(), "The goal is in the list, or on this side, because the settlement "
               "asks for it: a sub-threshold move promoted, a side turned, or a goal with no move at all "
               "(evidence 'settle').", False),
        Column("home_score", pa.int16(), "Home goals after this goal; blank from the first goal with no side."),
        Column("away_score", pa.int16(), "Away goals after this goal; blank from the first goal with no side."),
        Column("mo_jump", pa.float64(), "The match-odds move: the home price's logit move less the away price's "
               "(logit of 1 / price); positive towards the home side."),
        Column("prob_jump", pa.float64(), "The summed absolute change of the three raw 1X2 chances."),
        Column("ou_jump", pa.float64(), "Under's move out: the mean of Under's logit fall and Over's rise; "
               "positive for a goal; blank when neither traded in the window."),
        Column("ah_jump", pa.float64(), "The handicap margin's step across the goal, goals; positive towards "
               "the home side; blank without a reading on both sides."),
        Column("home_before", pa.float64(), "The home price before the move (median-filtered)."),
        Column("home_after", pa.float64(), "The home price after it."),
        Column("draw_before", pa.float64(), "The draw price before the move."),
        Column("draw_after", pa.float64(), "The draw price after it."),
        Column("away_before", pa.float64(), "The away price before the move."),
        Column("away_after", pa.float64(), "The away price after it."),
        Column("under_before", pa.float64(), "The Under 2.5 price before the move."),
        Column("under_after", pa.float64(), "The Under 2.5 price after it."),
        Column("over_before", pa.float64(), "The Over 2.5 price before the move."),
        Column("over_after", pa.float64(), "The Over 2.5 price after it."),
    ),
    key=("event_id", "goal_no"), sort=("event_id", "goal_no"),
)

TIMELINE = TableSpec(
    "gold/timeline",
    "One row a match that went in play: its inferred goals, the time each side led, its stoppages, half time, "
    "whether the in-play record can be used, whether the price reading agreed with how the markets settled, and "
    "how far the final list can be trusted.",
    (
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("usable", pa.bool_(), "The data rule: at least 60 price rows in play, no gap between two rows over "
               "12 minutes, 100 to 140 minutes from the turn in play to the close, a settled result, and first "
               "in-play prices near the closing ones (a market that went in play late has lost its first goals).",
               False),
        Column("consistent", pa.bool_(), "The price reading alone (before any reconciliation) passed every check "
               "the settlement allows: agrees_result, agrees_over25 and agrees_margin, those that are not blank. "
               "Blank when none can be made."),
        Column("reconciled", pa.bool_(), "The price reading disagreed with the settlement and the list was "
               "edited at least cost to agree with it (n_edits says how much). False with consistent false: no "
               "edit within reach agrees, the list is the price reading.", False),
        Column("reliability", pa.float64(), "The share of matches like this one whose final score was exactly "
               "the real one on 2026's odd-numbered months (football-data.co.uk scores): by how the list was "
               "reached (price reading agreed; one edit; several; a goal with no price move) and its goals (0 to "
               "2, 3, 4, 5 or more). Blank where usable is false or no agreement with the settlement was reached: "
               "not measured there."),
        Column("n_goals", pa.int16(), "Inferred goals, after reconciliation.", False),
        Column("home_goals", pa.int16(), "Inferred home goals.", False),
        Column("away_goals", pa.int16(), "Inferred away goals.", False),
        Column("n_side_unknown", pa.int16(), "Inferred goals with no side.", False),
        Column("n_edits", pa.int16(), "Edits the reconciliation made: goals removed, added, turned to the other "
               "side.", False),
        Column("n_unplaced", pa.int16(), "Of the goals, those no price move shows (evidence 'settle', timed "
               "false): their minute is not known.", False),
        Column("min_confidence", pa.float64(), "The least sure goal's confidence; blank without goals."),
        Column("agrees_result", pa.bool_(), "The price reading's goal difference has the sign of the settled 1X2 "
               "result; blank without a result."),
        Column("agrees_over25", pa.bool_(), "The price reading's three or more goals is the settled Over 2.5; "
               "blank without one."),
        Column("final_margin", pa.int16(), "The settled goal difference, home less away, read off the Asian "
               "handicap lines that won (the lines stop at four: 4 and -4 stand for four or more); blank without "
               "a settled handicap market."),
        Column("agrees_margin", pa.bool_(), "The price reading's goal difference is final_margin; blank without "
               "one."),
        Column("n_stoppages", pa.int16(), "In-play suspensions of the match's markets before the final one "
               "(overlapping suspensions of its markets count once).", False),
        Column("n_goal_stoppages", pa.int16(), "Of those, the ones that mark an inferred goal.", False),
        Column("n_other_stoppages", pa.int16(), "The rest: a penalty not scored, a red card, a review, an "
               "injury, a goal disallowed.", False),
        Column("n_disallowed", pa.int16(), "Goals undone by a later move back (Under sharply in again).", False),
        Column("first_goal_min", pa.float64(), "match_min of the first goal."),
        Column("last_goal_min", pa.float64(), "match_min of the last goal."),
        Column("min_home_leading", pa.float64(), "Minutes of play the home side led (half time left out); "
               "blank where a goal has no side. A goal with timed false counts from the last price on."),
        Column("min_level", pa.float64(), "Minutes of play the score was level."),
        Column("min_home_trailing", pa.float64(), "Minutes of play the home side trailed."),
        Column("ht_found", pa.bool_(), "Whether a quiet stretch marks half time in the match's own data (else "
               "the restart is read from the minutes in play alone).", False),
        Column("ht_start_min", pa.float64(), "Elapsed minute half time began (an estimate).", False),
        Column("ht_end_min", pa.float64(), "Elapsed minute the second half began (an estimate).", False),
        Column("inplay_min", pa.float64(), "Minutes from the turn in play to the match-odds market's close."),
        Column("n_rows", pa.int32(), "Price rows in play (messages with a match-odds or over/under price).",
               False),
        Column("max_gap_min", pa.float64(), "The longest gap between two price rows (or the turn in play and "
               "the first, or the last and the close), minutes."),
        Column("start_jump", pa.float64(), "The summed absolute change of the three 1X2 chances from the close "
               "to the first in-play row: large when the market went in play after a goal."),
        Column("ou_closed_min", pa.float64(), "Elapsed minute the over/under market stopped for good, where "
               "that was before full time: the third goal as Betfair settled it."),
    ),
    key=("event_id",), sort=("event_id",),
)
SPECS = (GOAL, TIMELINE)


# ----------------------------------------------------------------------------------------------------------------
# inputs to arrays
# ----------------------------------------------------------------------------------------------------------------

def _ms(col) -> np.ndarray:
    """A timestamp column as float milliseconds (blank is NaN)."""
    a = col.cast(pa.int64()) if not pa.types.is_integer(col.type) else col
    return a.to_pandas().to_numpy(dtype="float64", na_value=np.nan)


def _prepare(inputs: dict) -> dict | None:
    import pandas as pd
    import pyarrow.compute as pc

    ev = inputs["gold/event"].select(COLUMNS["gold/event"])
    e = ev.to_pandas()
    e["ip_ms"], e["ft_ms"] = _ms(ev.column("kickoff_actual")), _ms(ev.column("ft_pt"))
    e = e[e["went_in_play"].fillna(False).astype(bool) & e["mo_market_id"].notna() & e["ip_ms"].notna()]
    e = e.sort_values("event_id").reset_index(drop=True)
    if e.empty:
        return None
    e["e"] = np.arange(len(e))

    mm = pd.concat([e[["e", c]].rename(columns={c: "market_id"}).assign(mkt=k)
                    for k, c in enumerate(("mo_market_id", "ou_market_id", "ah_market_id"))], ignore_index=True)
    mm = mm[mm["market_id"].notna()].drop_duplicates("market_id").reset_index(drop=True)
    ids = pa.array(mm["market_id"], pa.string())

    mk = inputs["silver/market"].select(COLUMNS["silver/market"])
    closed = pd.Series(_ms(mk.column("closed_pt")), index=mk.column("market_id").to_pandas())
    closed = closed[~closed.index.duplicated()]
    mm["closed_ms"] = mm["market_id"].map(closed)

    # in-play suspensions of the match's markets, overlapping ones as one stoppage
    sp = inputs["silver/spell"].select(COLUMNS["silver/spell"])
    sp = sp.filter(pc.and_(pc.fill_null(sp.column("in_play"), False), pc.equal(sp.column("status"), "SUSPENDED")))
    s = pd.DataFrame({"market_id": sp.column("market_id").to_pandas(), "start": _ms(sp.column("start_pt")),
                      "end": _ms(sp.column("end_pt"))}).merge(mm, on="market_id")
    s = s.sort_values(["e", "start", "end"]).reset_index(drop=True)
    run = s.groupby("e")["end"].cummax()
    prev = run.groupby(s["e"]).shift(1)
    s["c"] = (prev.isna() | (s["start"] > prev)).cumsum()
    cl = s.groupby("c").agg(e=("e", "first"), start=("start", "min"), end=("end", "max")).reset_index(drop=True)
    cl = cl.merge(e[["e", "ft_ms"]], on="e")
    cl["final"] = cl["end"] >= cl["ft_ms"] - 1000.0                     # the suspension that runs into the close
    # the over/under market: when it stopped trading for good (the start of the suspension that ran into its close)
    ou = mm[mm["mkt"] == 1].set_index("e")
    last = s[(s["mkt"] == 1) & (s["end"] >= s["closed_ms"] - 1000.0)].groupby("e")["start"].min()
    e["ou_close_ms"] = e["e"].map(ou["closed_ms"])
    e["ou_dead_ms"] = e["e"].map(last).fillna(e["ou_close_ms"])

    # runners: the role of each, and the settled margin off the handicap lines that won
    r = inputs["silver/runner"].select(COLUMNS["silver/runner"]).to_pandas()
    r = r.merge(mm[["market_id", "e", "mkt"]], on="market_id")
    won = r[(r["mkt"] == 2) & (r["won"] == 1) & r["role"].isin(["home", "away"])]
    low = won.groupby(["e", "role"])["handicap"].min().unstack("role").reindex(columns=["home", "away"])
    by_home, by_away = 0.25 - low["home"], low["away"] - 0.25          # a side wins a line when margin + line > 0
    margin = by_home.where(by_home.notna() & ~(by_home.abs() > by_away.abs()), by_away)
    e["final_margin"] = e["e"].map(margin.round().clip(-MARGIN_CAP, MARGIN_CAP))

    t = inputs["silver/tick"].select(COLUMNS["silver/tick"])
    t = t.filter(t.column("in_play"))
    code = pc.index_in(t.column("market_id"), value_set=ids)
    t = t.filter(pc.is_valid(code))
    code = pc.drop_null(code).to_numpy(zero_copy_only=False).astype("int64")
    k = pd.DataFrame({"m": code, "selection_id": t.column("selection_id").to_numpy(),
                      "handicap": t.column("handicap").to_numpy(), "pt": _ms(t.column("pt")).astype("int64"),
                      "seq": t.column("seq").to_numpy(), "ltp": t.column("ltp").to_numpy()})
    k["e"], k["mkt"] = mm["e"].to_numpy()[k["m"].to_numpy()], mm["mkt"].to_numpy()[k["m"].to_numpy()]
    r["m"] = r["market_id"].map(pd.Series(np.arange(len(mm)), index=mm["market_id"]))
    r["r"] = r["role"].map({name: i for i, name in enumerate(ROLES)})
    px = k[k["mkt"] < 2].merge(r.loc[r["r"].notna(), ["m", "selection_id", "r"]], on=["m", "selection_id"])
    px["r"] = px["r"].astype("int64")
    ah = k[k["mkt"] == 2].merge(r.loc[r["role"].isin(["home", "away"]), ["m", "selection_id", "handicap", "role"]],
                                on=["m", "selection_id", "handicap"])
    return {"ev": e, "stops": cl, "px": px, "ah": ah, "all": k[["e", "pt"]]}


# ----------------------------------------------------------------------------------------------------------------
# rows and windows
# ----------------------------------------------------------------------------------------------------------------

def _med3(v: np.ndarray, prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """The median of each value and its neighbours; an end value stands as it is."""
    ends = np.isnan(prev) | np.isnan(nxt)
    mid = np.median(np.vstack([np.where(ends, v, prev), v, np.where(ends, v, nxt)]), axis=0)
    return np.where(ends, v, mid)


def _panel(px) -> dict:
    """The match's rows: one a price message, each runner's filtered price (as a logit) as of it."""
    import pandas as pd

    if not len(px):
        none = np.zeros((0, len(ROLES)))
        return {"e": np.zeros(0, "int64"), "t": np.zeros(0, "int64"), "z": none, "seen": none,
                "obs": np.zeros((0, len(ROLES)), bool), "next": none, "next_seen": none}
    px = px.sort_values(["e", "r", "pt", "seq"], kind="stable").reset_index(drop=True)
    z = -np.log(np.clip(px["ltp"].to_numpy(), 1.01, 1000.0) - 1.0)                 # logit of 1 / price
    g = pd.Series(z).groupby([px["e"], px["r"]], sort=False)
    px["z"] = _med3(z, g.shift(1).to_numpy(), g.shift(-1).to_numpy())
    key = px["e"].to_numpy() * BIG + px["pt"].to_numpy()
    uk = np.unique(key)
    ue, ut = uk // BIG, uk % BIG
    new = np.r_[True, (ue[1:] != ue[:-1]) | (ut[1:] - ut[:-1] > ROW_GAP_MS)]
    row = (np.cumsum(new) - 1)[np.searchsorted(uk, key)]
    e, t = ue[new], ut[new]
    order = np.lexsort((px["seq"].to_numpy(), px["pt"].to_numpy(), row))
    w = np.full((len(e), len(ROLES)), np.nan)
    w[row[order], px["r"].to_numpy()[order]] = px["z"].to_numpy()[order]           # a row's last price stands
    obs = ~np.isnan(w)
    seen = np.where(obs, t[:, None].astype("float64"), np.nan)
    filled = pd.DataFrame(w).groupby(e, sort=False).ffill().to_numpy()
    nxt = pd.DataFrame(w).groupby(e, sort=False).bfill().to_numpy()
    nxt_seen = pd.DataFrame(seen).groupby(e, sort=False).bfill().to_numpy()
    seen = pd.DataFrame(seen).groupby(e, sort=False).ffill().to_numpy()
    return {"e": e, "t": t, "z": filled, "seen": seen, "obs": obs, "next": nxt, "next_seen": nxt_seen}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _windows(pn: dict, ev, stops) -> dict:
    """Each row as the first of a two-row window: the moves from the row before it to the row after it."""
    e, t, z, seen, obs = pn["e"], pn["t"], pn["z"], pn["seen"], pn["obs"]
    n = len(e)
    idx = np.arange(n)
    has_prev = np.r_[False, e[1:] == e[:-1]][:n]
    has_next = np.r_[e[1:] == e[:-1], False][:n]
    ip, ix = np.where(has_prev, idx - 1, idx), np.where(has_next, idx + 1, idx)
    pre = z[ip]
    pre_ok = has_prev[:, None] & ~np.isnan(pre) & ((t[:, None] - seen[ip]) <= BEFORE_MS)
    fresh = obs | (obs[ix] & has_next[:, None])
    use = pre_ok & fresh
    post = z[ix]
    d = np.where(use, post - pre, np.nan)
    dh, _, da, du, do = d.T
    both = ~np.isnan(dh) & ~np.isnan(da)
    mo = np.where(both, dh - da, np.where(~np.isnan(dh), 2.0 * dh, -2.0 * da))
    k = (~np.isnan(du)).astype("int64") + (~np.isnan(do)).astype("int64")
    ou = -(np.nan_to_num(du) - np.nan_to_num(do)) / np.where(k == 0, np.nan, k)
    l1 = np.nansum(np.abs(np.where(use, _sigmoid(post) - _sigmoid(pre), np.nan))[:, :3], axis=1)
    # the over/under's next price after the row before the window, however late (up to OU_WIDE_MS): a market
    # that did not trade inside the window still says, a few minutes on, whether Under went out
    nz, nt = pn["next"][:, 3:], pn["next_seen"][:, 3:]
    wide_ok = pre_ok[:, 3:] & ~np.isnan(nz) & ((nt - t[:, None]) <= OU_WIDE_MS)
    dw = np.where(wide_ok, nz - pre[:, 3:], np.nan)
    kw = wide_ok.sum(axis=1)
    ou_wide = -(np.nan_to_num(dw[:, 0]) - np.nan_to_num(dw[:, 1])) / np.where(kw == 0, np.nan, kw)
    # the move less the larger same-way move of the windows just before and after: time decay moves them all
    m0 = np.nan_to_num(mo)
    sign = np.sign(m0)
    before = np.where(np.r_[False, False, e[2:] == e[:-2]][:n], np.r_[0.0, 0.0, m0[:-2]][:n], 0.0)
    after = np.where(np.r_[e[2:] == e[:-2], False, False][:n], np.r_[m0[2:], 0.0, 0.0][:n], 0.0)
    exc = np.abs(m0) - np.maximum(np.maximum(sign * before, sign * after), 0.0)

    m = ev.set_index("e")
    dead = m["ou_dead_ms"].reindex(e).to_numpy()
    early = m["ou_close_ms"].reindex(e).to_numpy() <= m["ft_ms"].reindex(e).to_numpy() - THIRD_EARLY_MS
    t_prev, t_next = np.where(has_prev, t[ip], t), t[ix]
    third = early & (t_prev - 90_000 <= dead) & (dead <= t_next + 90_000)
    ou_alive = ~(t_next >= dead - 1)                # NaN dead (no over/under market): not alive
    ou_alive &= ~np.isnan(dead)
    # a stoppage between the row before the window and its second row
    ks = stops["e"].to_numpy() * BIG + stops["start"].to_numpy().astype("int64")
    lo = np.searchsorted(ks, e * BIG + t_prev, side="left")
    hi = np.searchsorted(ks, e * BIG + t_next, side="right")
    return {"e": e, "t": t, "t_prev": t_prev, "t_next": t_next, "has_prev": has_prev, "mo": mo, "ou": ou, "l1": l1,
            "exc": exc, "third": third, "ou_alive": ou_alive, "ou_wide": ou_wide, "stop_lo": lo, "stop_hi": hi,
            "before": np.where(pre_ok, pre, np.nan), "after": np.where(fresh, post, np.nan)}


def _thin(score: np.ndarray, e: np.ndarray, radius: int = 1, rounds: int = 6) -> np.ndarray:
    """Non-maximum suppression along each match: the strongest of overlapping windows (those within ``radius``
    places), then the strongest of what is left beside none taken, and so on. Ties go to the earlier."""
    n = len(score)
    alive = score > 0
    picked = np.zeros(n, bool)

    def shifted(a, k, fill):                     # a[i + k] where that is the same match
        out = np.full(n, fill, dtype=a.dtype)
        if k > 0:
            out[:n - k] = np.where(e[k:] == e[:n - k], a[k:], fill)
        else:
            out[-k:] = np.where(e[:n + k] == e[-k:], a[:n + k], fill)
        return out

    for _ in range(rounds):
        s = np.where(alive, score, 0.0)
        pick = alive.copy()
        for k in range(1, radius + 1):
            if k >= n:
                break
            pick &= (s > shifted(s, -k, 0.0)) & (s >= shifted(s, k, 0.0))
        if not pick.any():
            break
        picked |= pick
        kill = pick.copy()
        for k in range(1, radius + 1):
            if k >= n:
                break
            kill |= shifted(pick, -k, False) | shifted(pick, k, False)
        alive &= ~kill
    return picked


# ----------------------------------------------------------------------------------------------------------------
# the Asian handicap margin
# ----------------------------------------------------------------------------------------------------------------

def _margin(ah, ev):
    """One row a moment the handicap market traded near evens: the expected final margin (home less away) its
    lines imply. A side at line h and price q is given chance 1 / q of margin + h > 0 under a normal margin whose
    spread shrinks with the play left; lines nearer evens weigh more; three readings are median-filtered."""
    import pandas as pd
    from scipy.special import ndtri

    p = 1.0 / ah["ltp"].to_numpy()
    a = ah[(p >= AH_P[0]) & (p <= AH_P[1])]
    if a.empty:
        return pd.DataFrame({"e": np.zeros(0, "int64"), "pt": np.zeros(0, "int64"), "mu": np.zeros(0)})
    p = 1.0 / a["ltp"].to_numpy()
    el = (a["pt"].to_numpy() - ev["ip_ms"].to_numpy()[a["e"].to_numpy()]) / 60_000.0
    played = np.where(el <= HT_START_MIN, el, np.where(el <= HT_END_MIN, HT_START_MIN,
                                                        el - (HT_END_MIN - HT_START_MIN)))
    sd = SIGMA0 * np.sqrt(np.clip((PLAY_MIN - played) / PLAY_MIN, 0.03, 1.0))
    x = sd * ndtri(p) - a["handicap"].to_numpy()
    wt = np.maximum(0.02, 0.25 - np.abs(p - 0.5))
    g = pd.DataFrame({"e": a["e"].to_numpy(), "pt": a["pt"].to_numpy(), "wt": wt,
                      "x": np.where(a["role"].to_numpy() == "home", x, -x) * wt})
    g = g.groupby(["e", "pt"], sort=True)[["x", "wt"]].sum().reset_index()
    raw = (g["x"] / g["wt"]).to_numpy()
    by = pd.Series(raw).groupby(g["e"], sort=False)
    g["mu"] = _med3(raw, by.shift(1).to_numpy(), by.shift(-1).to_numpy())
    return g[["e", "pt", "mu"]]


def _across(mu, e, t_prev, t_first, t_next) -> np.ndarray:
    """The margin's change across each window: the last reading at or before the row before it against the
    second reading after its first row (the first may be a trade from before the goal), or the only one."""
    out = np.full(len(e), np.nan)
    if not len(mu):
        return out
    me, mt, mv = mu["e"].to_numpy(), mu["pt"].to_numpy(), mu["mu"].to_numpy()
    mk = me * BIG + mt
    b = np.searchsorted(mk, e * BIG + t_prev, side="right") - 1
    a = np.searchsorted(mk, e * BIG + t_first, side="right")
    bb, aa = np.maximum(b, 0), np.minimum(a, len(mu) - 1)
    ok = (b >= 0) & (me[bb] == e) & (t_prev - mt[bb] <= AH_BEFORE_MS)
    ok &= (a < len(mu)) & (me[aa] == e) & (mt[aa] - t_next <= AH_AFTER_MS)
    a2 = np.minimum(aa + 1, len(mu) - 1)
    second = (me[a2] == e) & (mt[a2] - t_next <= AH_AFTER_MS) & (a2 > aa)
    out[ok] = (np.where(second, mv[a2], mv[aa]) - mv[bb])[ok]
    return out


def _shifts(mu, goals, par: dict):
    """Where the margin moved and the goals so far do not say why: the lasting shifts of the residual (the margin
    less the inferred goal difference at that moment), the median of three readings after against three before."""
    import pandas as pd

    empty = pd.DataFrame({"e": np.zeros(0, "int64"), "lo": np.zeros(0, "int64"), "hi": np.zeros(0, "int64"),
                          "from": np.zeros(0, "int64"), "to": np.zeros(0, "int64"), "d": np.zeros(0)})
    if not len(mu):
        return empty
    e, t, v = mu["e"].to_numpy(), mu["pt"].to_numpy(), mu["mu"].to_numpy()
    n = len(mu)
    gk = goals["e"].to_numpy() * BIG + goals["t"].to_numpy()
    cum = np.r_[0.0, np.cumsum(goals["side"].to_numpy())]
    r = v - (cum[np.searchsorted(gk, e * BIG + t, side="right")] - cum[np.searchsorted(gk, e * BIG, side="left")])
    idx = np.arange(n)

    def at(k):                                   # the residual k readings away, where it is the same match and near
        j = np.clip(idx + k, 0, n - 1)
        ok = (j == idx + k) & (e[j] == e) & (np.abs(t[j] - t) <= par["ah_reach_ms"])
        return np.where(ok, r[j], np.nan), np.where(ok, t[j], t)

    left = np.vstack([at(-3)[0], at(-2)[0], at(-1)[0]])
    right = np.vstack([r, at(1)[0], at(2)[0]])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.nanmedian(right, axis=0) - np.nanmedian(left, axis=0)
    ok = ((~np.isnan(left)).sum(axis=0) >= 2) & ((~np.isnan(right)).sum(axis=0) >= 2)
    own = np.abs(np.nan_to_num(r - at(-1)[0]))
    score = np.where(ok & (np.abs(d) >= par["ah_only"]), np.abs(d) + 0.01 * np.minimum(own, 2.0), 0.0)
    pick = _thin(score, e, radius=2)
    if not pick.any():
        return empty
    return pd.DataFrame({"e": e[pick], "lo": at(-1)[1][pick], "hi": t[pick], "from": at(-2)[1][pick],
                         "to": at(1)[1][pick], "d": d[pick]})


# ----------------------------------------------------------------------------------------------------------------
# goals
# ----------------------------------------------------------------------------------------------------------------

def _detect(w: dict, mu, par: dict):
    """The windows that are goals: one row each, with the evidence."""
    import pandas as pd

    e, t = w["e"], w["t"]
    mo, ou = w["mo"], w["ou"]
    amo = np.abs(mo)
    has_mo, has_ou = ~np.isnan(mo), ~np.isnan(ou)
    dmu = _across(mu, e, w["t_prev"], t, w["t_next"])
    sign = np.sign(np.nan_to_num(mo))
    ah_for = ~np.isnan(dmu) & (dmu * np.where(sign == 0, np.sign(np.nan_to_num(dmu)), sign) >= par["ah"])
    ah_flat = ~np.isnan(dmu) & (np.abs(dmu) < par["ah_flat"])
    ou_flat = has_ou & (ou < 0.25) & w["ou_alive"]
    moved = has_mo & (amo >= par["mo_min"])

    by_ou = has_ou & (ou >= par["ou"]) & moved
    by_ou_solo = has_ou & ~has_mo & (ou >= par["ou_solo"])
    wide = w["ou_wide"]
    ou_says_no = w["ou_alive"] & ~np.isnan(wide) & (wide < par["ou_wide"])
    by_mo = ~has_ou & has_mo & (amo >= par["mo_only"]) & (w["exc"] >= par["exc_only"]) & (w["l1"] >= par["p_only"]) \
        & ~ah_flat & ~ou_says_no
    by_sure = has_mo & (amo >= par["mo_sure"]) & (w["l1"] >= par["p_sure"]) & (w["exc"] >= par["exc_only"]) \
        & ~ah_flat & ~(has_ou & (ou >= par["ou"]))
    by_third = w["third"] & has_mo & (amo >= par["mo_third"]) & (w["exc"] >= 0.5 * par["mo_third"]) & ~ou_flat
    other = by_ou | by_ou_solo | by_mo | by_sure
    if by_third.any():                           # the over/under market stops for good once: one such goal a match
        strength = np.where(by_third & ~other, amo, 0.0)
        top = pd.Series(strength).groupby(e).transform("max").to_numpy()
        by_third &= other | (strength == top)
    goal = (other | by_third) & w["has_prev"]
    score = np.where(goal, 1.5 * np.clip(np.nan_to_num(ou), 0, None) + np.nan_to_num(amo)
                     + np.abs(np.nan_to_num(dmu)), 0.0)
    pick = _thin(score, e)

    # a goal undone: Under sharply back in, match odds the other way, soon after
    anti = np.flatnonzero(has_ou & (ou <= -par["undo"]) & moved & w["has_prev"])
    gi = np.flatnonzero(pick)
    undone = np.zeros(len(e), bool)
    if len(gi) and len(anti):
        j = np.searchsorted(e[gi] * BIG + t[gi], e[anti] * BIG + t[anti], side="left") - 1
        jj = gi[np.maximum(j, 0)]
        gap = t[anti] - t[jj]
        ok = (j >= 0) & (e[jj] == e[anti]) & (gap <= par["undo_ms"]) & (gap >= 100_000) \
            & (sign[anti] == -sign[jj]) & (sign[jj] != 0) & has_ou[jj] & (ou[jj] >= par["ou"])
        undone[jj[ok]] = True
    gi = np.flatnonzero(pick & ~undone)

    kind = np.where(by_ou | by_ou_solo, np.where(ah_for, "ou+ah", np.where(ah_flat, "ou-ah", "ou")),
                    np.where(by_third, "third", np.where(ah_for, "ah+mo", np.where(by_mo, "mo", "mo-ou"))))
    side = np.where(has_mo & (amo >= par["side"]), sign, 0.0)
    side = np.where((side == 0) & ah_for, np.sign(np.nan_to_num(dmu)), side)
    out = pd.DataFrame({"e": e[gi], "t": t[gi], "t_prev": w["t_prev"][gi], "t_next": w["t_next"][gi],
                        "side": side[gi], "kind": kind[gi], "mo": mo[gi], "ou": ou[gi], "l1": w["l1"][gi],
                        "dmu": dmu[gi], "stop_lo": w["stop_lo"][gi], "stop_hi": w["stop_hi"][gi]})
    for i, name in enumerate(ROLES):
        out[f"{name}_before"] = 1.0 + np.exp(-w["before"][gi, i])
        out[f"{name}_after"] = 1.0 + np.exp(-w["after"][gi, i])
    return out, (pick & undone)


WEAK = ("mo", "mo-ou", "third")


def _repair(goals, mu, stops, par: dict):
    """The window goals against the handicap margin. Where the margin shifted and the goals do not say why:
    a shift of two goals around a goal of the other side turns that goal round; a shift of one around a weakly
    evidenced goal of the other side removes it; any other shift is a goal the windows did not see."""
    import pandas as pd

    sh = _shifts(mu, goals, par)
    if not len(sh):
        return goals
    gk = goals["e"].to_numpy() * BIG + goals["t"].to_numpy()
    se, d = sh["e"].to_numpy(), sh["d"].to_numpy()
    sign = np.sign(d)
    units = np.clip(np.round(np.abs(d) / par["ah_unit"]), 1, 3).astype("int64")
    lo = np.searchsorted(gk, se * BIG + sh["from"].to_numpy() - par["ah_slack_ms"], side="left")
    hi = np.searchsorted(gk, se * BIG + sh["to"].to_numpy() + par["ah_slack_ms"], side="right")
    side = goals["side"].to_numpy().copy()
    kind = goals["kind"].to_numpy().astype(object)
    weak = np.isin(kind, WEAK)
    drop = np.zeros(len(goals), bool)
    add = np.ones(len(sh), bool)
    # a step and its reverse soon after with no goal between them: a goal given and taken back, not two goals
    t_hi = sh["hi"].to_numpy()
    for k in np.flatnonzero((se[1:] == se[:-1]) & (sign[1:] == -sign[:-1]) & (np.abs(d[1:] + d[:-1]) <= 0.5)
                            & (t_hi[1:] - t_hi[:-1] <= par["ah_undo_ms"]) & (hi[1:] == lo[:-1])):
        if add[k]:
            add[k] = add[k + 1] = False
    for j in range(3):                           # the goals inside a shift: seldom more than one or two
        g = lo + j
        inside = g < hi
        gg = np.where(inside, g, 0)
        against = inside & (side[gg] == -sign) & ~drop[gg]
        turn = against & (units >= 2) & add
        side[gg[turn]] = sign[turn]
        units = np.where(turn, units - 2, units)
        remove = against & ~turn & (units == 1) & weak[gg] & add
        drop[gg[remove]] = True
        blank = inside & (side[gg] == 0) & add & ~turn & ~remove      # a goal with no side: this is its side
        side[gg[blank]] = sign[blank]
        add &= ~(remove | blank | (turn & (units == 0)) | (against & ~turn & ~remove))
    goals = goals.assign(side=side)[~drop]
    rep = np.repeat(np.flatnonzero(add), units[add])
    if not len(rep):
        return goals
    s = sh.iloc[rep]
    ks = stops["e"].to_numpy() * BIG + stops["start"].to_numpy().astype("int64")
    extra = pd.DataFrame({"e": s["e"].to_numpy(), "t": s["hi"].to_numpy(), "t_prev": s["lo"].to_numpy(),
                          "t_next": s["hi"].to_numpy(), "side": sign[rep], "kind": "ah",
                          "dmu": d[rep] / units[rep],
                          "stop_lo": np.searchsorted(ks, s["e"].to_numpy() * BIG + s["lo"].to_numpy(), side="left"),
                          "stop_hi": np.searchsorted(ks, s["e"].to_numpy() * BIG + s["hi"].to_numpy(),
                                                     side="right")})
    return pd.concat([goals, extra], ignore_index=True)


# ----------------------------------------------------------------------------------------------------------------
# the settlement
# ----------------------------------------------------------------------------------------------------------------

def _logit(p):
    return np.log(p / (1.0 - p))


def _candidates(w: dict, mu, goals, par: dict):
    """The sub-threshold windows that are not already goals (one row each, as ``_detect`` gives them), with
    ``chance``: the share of their kind that was a real goal."""
    import pandas as pd

    cand, _ = _detect(w, mu, dict(par, **LOOSE))
    if len(cand) and len(goals):
        gk = np.sort(goals["e"].to_numpy() * BIG + goals["t"].to_numpy())
        ck = cand["e"].to_numpy() * BIG + cand["t"].to_numpy()
        j = np.searchsorted(gk, ck)
        gap = np.minimum(np.abs(ck - gk[np.maximum(j - 1, 0)]), np.abs(gk[np.minimum(j, len(gk) - 1)] - ck))
        cand = cand[gap > CAND_NEAR_MS]
    cand = cand.sort_values(["e", "t"], kind="stable").reset_index(drop=True)
    chance = pd.Series(cand["kind"].to_numpy()).map(CANDIDATE).to_numpy(dtype="float64", na_value=np.nan)
    l1 = cand["l1"].to_numpy(dtype="float64")
    by_move = np.full(len(cand), CANDIDATE_MO[-1][1])
    for least, share in reversed(CANDIDATE_MO):
        by_move = np.where(np.nan_to_num(l1) >= least, share, by_move)
    cand["chance"] = np.where(np.isnan(chance), by_move, chance)
    return cand


def _allowed(diff: int, total: int, margin, result, over) -> bool:
    """Whether a goal difference and a total (3 stands for three or more) are what the markets settled."""
    if margin is not None:
        if abs(margin) >= MARGIN_CAP:
            if diff * (1 if margin > 0 else -1) < MARGIN_CAP:
                return False
        elif diff != margin:
            return False
    if result is not None and (diff > 0) - (diff < 0) != result:
        return False
    return over is None or (total >= 3) == bool(over)


def _solve(items: list, margin, result, over):
    """The cheapest edit of a match's goals that the settlement allows. ``items``: (is a goal, side, cost) in time
    order: a goal may stay (free), go (its cost) or change side (TURN_COST); a candidate may be added (its cost);
    a goal with no side takes either. Goals no price shows may be added at UNSEEN_COST each, MAX_UNSEEN at most.
    Returns (the side each item ends with, 0 for out; unseen home goals; unseen away goals), or None."""
    states = {(0, 0): (0.0, ())}
    for is_goal, side, cost in items:
        if is_goal:
            options = ((side, 0.0), (0, cost), (-side, TURN_COST)) if side else ((1, 0.0), (-1, 0.0), (0, cost))
        else:
            options = ((0, 0.0), (side, cost)) if side else ((0, 0.0), (1, cost), (-1, cost))
        new: dict = {}
        for (diff, total), (paid, picks) in states.items():
            for s, c in options:
                key = (diff + s, min(total + (s != 0), 3))
                if key not in new or paid + c < new[key][0] - 1e-9:
                    new[key] = (paid + c, picks + (s,))
        states = new
    best = None
    for (diff, total), (paid, picks) in sorted(states.items()):
        for unseen in range(MAX_UNSEEN + 1):
            for home in range(unseen + 1):
                away = unseen - home
                if not _allowed(diff + home - away, min(total + unseen, 3), margin, result, over):
                    continue
                price = paid + UNSEEN_COST * unseen
                if best is None or price < best[0] - 1e-9:
                    best = (price, picks, home, away)
    return None if best is None else best[1:]


def _reconcile(goals, cands, ev, need: np.ndarray, t_end: np.ndarray):
    """The goal lists of the matches in ``need`` (the price reading disagrees with the settlement) edited at least
    cost to agree with it: weak goals out, sub-threshold candidates in, a side turned, goals no price shows added
    after the last price. Returns (the goals, reconciled, edits, unplaced: a value a match)."""
    import pandas as pd

    n = len(ev)
    done, edits, unplaced = np.zeros(n, bool), np.zeros(n, "int64"), np.zeros(n, "int64")
    ge, ce = goals["e"].to_numpy(), cands["e"].to_numpy()
    side = goals["side"].to_numpy().astype("int64")
    cost = _logit(pd.Series(goals["kind"].to_numpy()).map(KEEP).fillna(0.5).to_numpy(dtype="float64"))
    c_side = cands["side"].to_numpy().astype("int64")
    c_cost = -_logit(cands["chance"].to_numpy(dtype="float64"))
    which = np.arange(n)
    g_lo, g_hi = np.searchsorted(ge, which, "left"), np.searchsorted(ge, which, "right")
    c_lo, c_hi = np.searchsorted(ce, which, "left"), np.searchsorted(ce, which, "right")
    margin = ev["final_margin"].to_numpy(dtype="float64", na_value=np.nan)
    result = ev["result"].map({"H": 1, "D": 0, "A": -1}).to_numpy(dtype="float64", na_value=np.nan)
    over = ev["over25"].to_numpy(dtype="float64", na_value=np.nan)
    drop, turned, add = np.zeros(len(goals), bool), np.zeros(len(goals), bool), np.zeros(len(cands), bool)
    extra_e, extra_side = [], []
    for i in np.flatnonzero(need):
        k = g_hi[i] - g_lo[i]
        items = [(True, int(side[j]), float(cost[j])) for j in range(g_lo[i], g_hi[i])]
        items += [(False, int(c_side[j]), float(c_cost[j])) for j in range(c_lo[i], c_hi[i])]
        got = _solve(items, None if np.isnan(margin[i]) else int(margin[i]),
                     None if np.isnan(result[i]) else int(result[i]), None if np.isnan(over[i]) else int(over[i]))
        if got is None:
            continue
        picks, home, away = got
        for j, s in zip(range(g_lo[i], g_hi[i]), picks[:k]):
            drop[j] = s == 0
            turned[j] = s != 0 and side[j] != 0 and s != side[j]
            side[j] = s if s != 0 else side[j]
        for j, s in zip(range(c_lo[i], c_hi[i]), picks[k:]):
            add[j] = s != 0
            c_side[j] = s if s != 0 else c_side[j]
        extra_e += [i] * (home + away)
        extra_side += [1.0] * home + [-1.0] * away
        done[i] = True
        unplaced[i] = home + away
        edits[i] = int(drop[g_lo[i]:g_hi[i]].sum() + turned[g_lo[i]:g_hi[i]].sum() + add[c_lo[i]:c_hi[i]].sum()) \
            + home + away
    kept = goals.assign(side=side.astype("float64"), edit=np.where(turned, "turned", ""))[~drop]
    added = cands.assign(side=c_side.astype("float64"), edit="added")[add].drop(columns=["chance"])
    xe = np.array(extra_e, "int64")
    at = t_end[xe].astype("int64") + (np.arange(len(xe)) - np.searchsorted(xe, xe, "left")) if len(xe) \
        else np.zeros(0, "int64")
    unseen = pd.DataFrame({"e": xe, "t": at, "t_prev": at, "t_next": at, "side": np.array(extra_side, "float64"),
                           "kind": "settle", "stop_lo": 0, "stop_hi": 0, "edit": "unseen"})
    out = pd.concat([f for f in (kept, added, unseen) if len(f)], ignore_index=True) if len(kept) + len(added) \
        + len(unseen) else kept
    return out, done, edits, unplaced


# ----------------------------------------------------------------------------------------------------------------
# half time
# ----------------------------------------------------------------------------------------------------------------

def _half_time(ticks, ev) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(found, start, end) a match, elapsed minutes: the stretch of 12 to 20 minutes starting 43 to 57 minutes in
    where the match's markets updated least against the ten minutes either side (a Poisson likelihood ratio)."""
    n = len(ev)
    lo, hi = 30, 92
    el = (ticks["pt"].to_numpy() - ev["ip_ms"].to_numpy()[ticks["e"].to_numpy()]) / 60_000.0
    keep = (el >= lo) & (el < hi)
    counts = np.bincount(ticks["e"].to_numpy()[keep] * (hi - lo) + (el[keep].astype("int64") - lo),
                         minlength=n * (hi - lo)).reshape(n, hi - lo)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(counts, axis=1)], axis=1)
    best = np.full(n, -np.inf)
    start, end = np.full(n, HT_START_MIN), np.full(n, HT_END_MIN)
    found = np.zeros(n, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for s in range(43, 58):
            for length in range(12, 21):
                a, b = s - lo, s + length - lo
                inside = cs[:, b] - cs[:, a]
                outside = (cs[:, a] - cs[:, a - 10]) + (cs[:, b + 10] - cs[:, b])
                r_in, r_out, r_all = inside / length, outside / 20.0, (inside + outside) / (length + 20.0)
                llr = np.where(inside > 0, inside * np.log(r_in / r_all), 0.0) \
                    + np.where(outside > 0, outside * np.log(r_out / r_all), 0.0)
                ok = (outside >= 20) & (r_in <= 0.6 * r_out) & (llr >= 4.0)
                better = ok & (llr > best)
                best = np.where(better, llr, best)
                start, end = np.where(better, float(s), start), np.where(better, float(s + length), end)
                found |= ok
    return _restart(found, start, end, (ev["ft_ms"].to_numpy() - ev["ip_ms"].to_numpy()) / 60_000.0)


def _restart(found: np.ndarray, start: np.ndarray, end: np.ndarray, span: np.ndarray):
    """(found, start, end) with the restart re-read: the quiet stretch's end is a noisy reading of it (updates
    resume over a minute or two), and the minutes from the turn in play to the close say when it must have been;
    the two are weighed as the real second-half goals of 2026's odd-numbered months place the restart."""
    whole = ~np.isnan(span) & (span >= USABLE_SPAN_MIN[0]) & (span <= USABLE_SPAN_MIN[1])
    sp = np.where(whole, span, 0.0)
    restart = np.where(found & whole, RESTART_FOUND[0] + RESTART_FOUND[1] * end + RESTART_FOUND[2] * sp,
                       np.where(found, RESTART_END[0] + RESTART_END[1] * end,
                                np.where(whole, RESTART_SPAN[0] + RESTART_SPAN[1] * sp, HT_END_MIN)))
    begin = np.where(found, np.clip(start, restart - HT_BREAK_MIN[2], restart - HT_BREAK_MIN[0]),
                     restart - HT_BREAK_MIN[1])
    return found, begin, restart


def _match_min(el: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(half, football minute, minutes of play so far) for elapsed minutes ``el`` in matches whose half time ran
    from ``start`` to ``end``."""
    first = el < (start + end) / 2.0
    half = np.where(first, 1, 2)
    minute = np.where(first, np.clip(el, 0.0, 45.0), np.clip(45.0 + el - end, 45.0, 90.0))
    played = np.where(first, np.minimum(el, start), start + np.maximum(el - end, 0.0))
    return half, minute, np.maximum(played, 0.0)


# ----------------------------------------------------------------------------------------------------------------
# the tables
# ----------------------------------------------------------------------------------------------------------------

def infer(inputs: dict, params: dict | None = None) -> dict | None:
    """The inference as pandas frames ({'goals', 'matches'}), for ``build`` and for the tuning scripts."""
    st = stage(inputs)
    return None if st is None else finish(st, dict(PARAMS, **(params or {})))


def stage(inputs: dict) -> dict | None:
    """Everything that does not depend on the thresholds: the rows, the windows, the handicap margin."""
    d = _prepare(inputs)
    if d is None:
        return None
    pn = _panel(d["px"])
    found, ht_start, ht_end = _half_time(d["all"], d["ev"])
    return {"ev": d["ev"], "stops": d["stops"], "pn": pn, "mu": _margin(d["ah"], d["ev"]),
            "w": _windows(pn, d["ev"], d["stops"]), "ht": (found, ht_start, ht_end)}


def _tally(goals, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(goals, home goals, away goals) a match."""
    e, side = goals["e"].to_numpy(), goals["side"].to_numpy()
    return (np.bincount(e, minlength=n), np.bincount(e[side > 0], minlength=n), np.bincount(e[side < 0], minlength=n))


def finish(st: dict, par: dict) -> dict:
    """The goals and the matches from a stage, under the thresholds ``par``."""
    import pandas as pd

    ev, stops, pn, mu, w = st["ev"], st["stops"], st["pn"], st["mu"], st["w"]
    found, ht_start, ht_end = st["ht"]
    n = len(ev)
    ip, ft = ev["ip_ms"].to_numpy(), ev["ft_ms"].to_numpy()
    goals, undone = _detect(w, mu, par)
    goals = goals.sort_values(["e", "t"], kind="stable").reset_index(drop=True)
    if par["repair"]:
        goals = _repair(goals, mu, stops, par)
    goals = goals.sort_values(["e", "t"], kind="stable").reset_index(drop=True)
    goals["edit"] = ""

    # the price reading against the settlement
    m = ev[["e", "event_id", "result", "over25", "final_margin", "ip_ms", "ft_ms", "ou_dead_ms", "ou_close_ms"]].copy()
    total, home, away = _tally(goals, n)
    res = m["result"].map({"H": 1, "D": 0, "A": -1}).to_numpy(dtype="float64", na_value=np.nan)
    over = m["over25"].to_numpy(dtype="float64", na_value=np.nan)
    fm = m["final_margin"].to_numpy(dtype="float64", na_value=np.nan)
    diff = home - away
    capped = np.where(np.abs(fm) >= MARGIN_CAP, np.clip(diff, -MARGIN_CAP, MARGIN_CAP), diff)
    ok_res, ok_over, ok_fm = np.sign(diff) == res, (total >= 3) == (over == 1), capped == fm
    m["agrees_result"] = _maybe(ok_res, ~np.isnan(res))
    m["agrees_over25"] = _maybe(ok_over, ~np.isnan(over))
    m["agrees_margin"] = _maybe(ok_fm, ~np.isnan(fm))
    checked = ~np.isnan(res) | ~np.isnan(over) | ~np.isnan(fm)
    need = (~np.isnan(res) & ~ok_res) | (~np.isnan(over) & ~ok_over) | (~np.isnan(fm) & ~ok_fm)
    m["consistent"] = _maybe(~need, checked)

    # where it disagrees, the cheapest edit that agrees; a goal no price shows goes after the last price
    done, edits, unplaced = np.zeros(n, bool), np.zeros(n, "int64"), np.zeros(n, "int64")
    if par["reconcile"] and need.any():
        t_end = np.where(np.isnan(ft), ip + PLAY_MIN * 60_000.0, ft)
        fin = stops[stops["final"]].groupby("e")["start"].min()
        t_end[fin.index.to_numpy()] = fin.to_numpy()
        if len(pn["e"]):
            last = np.r_[pn["e"][1:] != pn["e"][:-1], True]
            t_end[pn["e"][last]] = np.maximum(t_end[pn["e"][last]], pn["t"][last] + 1)
        goals, done, edits, unplaced = _reconcile(goals, _candidates(w, mu, goals, par), ev, need, t_end)
        goals = goals.sort_values(["e", "t"], kind="stable").reset_index(drop=True)

    ge = goals["e"].to_numpy()
    marked = goals["stop_hi"].to_numpy() > goals["stop_lo"].to_numpy()
    first_stop = stops["start"].to_numpy()[np.minimum(goals["stop_lo"].to_numpy(), max(len(stops) - 1, 0))] \
        if len(stops) else np.zeros(len(goals))
    pt = np.where(marked, first_stop, goals["t"].to_numpy()).astype("int64")
    el = (pt - ip[ge]) / 60_000.0
    half, minute, played = _match_min(el, ht_start[ge], ht_end[ge])
    kind = goals["kind"].to_numpy().astype(object)
    how = np.where(goals["edit"].to_numpy() == "added", 2, np.where(need[ge], 1, 0))
    conf = np.array([CONFIDENCE[k] for k in kind], dtype="float64").reshape(len(goals), 3)[np.arange(len(goals)), how]
    goals = goals.assign(pt=pt, suspended=marked, elapsed_min=el, half=half, match_min=minute, played=played,
                         confidence=conf, timed=kind != "settle")
    goals["goal_no"] = goals.groupby("e").cumcount() + 1
    known = goals["side"].ne(0).astype("int64").groupby(goals["e"]).cummin().astype(bool)
    goals["home_score"] = goals["side"].gt(0).astype("int64").groupby(goals["e"]).cumsum().where(known)
    goals["away_score"] = goals["side"].lt(0).astype("int64").groupby(goals["e"]).cumsum().where(known)

    # the match
    total, home, away = _tally(goals, n)
    by = goals.groupby("e")
    m["n_goals"], m["home_goals"], m["away_goals"] = total, home, away
    m["n_side_unknown"] = total - home - away
    m["min_confidence"] = by["confidence"].min().reindex(m["e"]).to_numpy()
    m["first_goal_min"] = by["match_min"].min().reindex(m["e"]).to_numpy()
    m["last_goal_min"] = by["match_min"].max().reindex(m["e"]).to_numpy()
    m["reconciled"], m["n_edits"], m["n_unplaced"] = done, edits, unplaced

    # stoppages
    live = stops[~stops["final"]]
    m["n_stoppages"] = live.groupby("e").size().reindex(m["e"]).fillna(0).to_numpy().astype("int64")
    hit = np.zeros(len(stops) + 1, "int64")
    if len(goals):
        lo, hi = goals["stop_lo"].to_numpy(), goals["stop_hi"].to_numpy()
        np.add.at(hit, lo[marked], 1)
        np.add.at(hit, hi[marked], -1)
    covered = (np.cumsum(hit)[:len(stops)] > 0) & ~stops["final"].to_numpy()
    m["n_goal_stoppages"] = pd.Series(covered).groupby(stops["e"].to_numpy()).sum().reindex(m["e"]).fillna(0) \
        .to_numpy().astype("int64")
    m["n_other_stoppages"] = m["n_stoppages"] - m["n_goal_stoppages"]
    m["n_disallowed"] = pd.Series(undone).groupby(w["e"]).sum().reindex(m["e"]).fillna(0).to_numpy().astype("int64")

    # the time each side led, in minutes of play
    span = (ft - ip) / 60_000.0
    _, _, whole = _match_min(np.nan_to_num(span, nan=HT_END_MIN + 50.0), ht_start, ht_end)
    lead, level, trail = np.zeros(n), np.zeros(n), np.zeros(n)
    if len(goals):
        nxt = goals["played"].groupby(goals["e"]).shift(-1).to_numpy()
        seg = np.where(np.isnan(nxt), whole[ge], nxt) - goals["played"].to_numpy()
        seg = np.maximum(seg, 0.0)
        state = np.sign((goals["home_score"] - goals["away_score"]).to_numpy(dtype="float64", na_value=np.nan))
        np.add.at(lead, ge[state > 0], seg[state > 0])
        np.add.at(level, ge[state == 0], seg[state == 0])
        np.add.at(trail, ge[state < 0], seg[state < 0])
        first = goals.groupby("e")["played"].first().reindex(m["e"]).to_numpy()
    else:
        first = np.full(n, np.nan)
    level += np.where(np.isnan(first), whole, np.minimum(first, whole))
    sided = m["n_side_unknown"].to_numpy() == 0
    m["min_home_leading"], m["min_level"] = np.where(sided, lead, np.nan), np.where(sided, level, np.nan)
    m["min_home_trailing"] = np.where(sided, trail, np.nan)
    m["ht_found"], m["ht_start_min"], m["ht_end_min"] = found, ht_start, ht_end
    m["inplay_min"] = span

    # the rows behind it and the usable rule
    e, t = pn["e"], pn["t"]
    m["n_rows"] = np.bincount(e, minlength=n)[:n] if len(e) else np.zeros(n, "int64")
    gap = np.full(n, np.nan)
    start_jump = np.full(n, np.nan)
    if len(e):
        firsts = np.r_[True, e[1:] != e[:-1]]
        lasts = np.r_[e[1:] != e[:-1], True]
        step = np.where(firsts, t - ip[e], t - np.r_[t[:1], t[:-1]]) / 60_000.0
        tail = (ft[e[lasts]] - t[lasts]) / 60_000.0
        mx = pd.Series(step).groupby(e).max()
        gap[mx.index.to_numpy()] = mx.to_numpy()
        gap[e[lasts]] = np.fmax(gap[e[lasts]], tail)
        close = ev[["p_home", "p_draw", "p_away"]].to_numpy(dtype="float64", na_value=np.nan)
        # the first in-play chances: each runner's first price in the first five rows, normalised like the close
        head = _head_mask(e, 5)
        fst = _sigmoid(pd.DataFrame(pn["z"][head][:, :3]).groupby(e[head]).first())
        fst = fst.div(fst.sum(axis=1, skipna=False), axis=0).reindex(np.arange(n)).to_numpy()
        start_jump = np.abs(fst - close).sum(axis=1)
    m["max_gap_min"], m["start_jump"] = gap, start_jump
    early = m["ou_close_ms"].to_numpy() <= ft - THIRD_EARLY_MS
    m["ou_closed_min"] = np.where(early, (m["ou_dead_ms"].to_numpy() - ip) / 60_000.0, np.nan)
    m["usable"] = (m["n_rows"].to_numpy() >= USABLE_ROWS) & (np.nan_to_num(gap, nan=1e9) <= USABLE_GAP_MIN) \
        & (span >= USABLE_SPAN_MIN[0]) & (span <= USABLE_SPAN_MIN[1]) & ~np.isnan(res) \
        & ~(np.nan_to_num(start_jump, nan=0.0) > USABLE_START)
    path = np.where(~checked | (need & ~done), "", np.where(~need, "agreed", np.where(
        unplaced > 0, "unplaced", np.where(edits == 1, "edit1", "edits"))))
    table = np.array([RELIABILITY.get(k, (np.nan,) * 4) for k in path], dtype="float64").reshape(n, 4)
    m["reliability"] = np.where(m["usable"].to_numpy(), table[np.arange(n), np.clip(total, 2, 5) - 2], np.nan)
    return {"goals": goals, "matches": m}


def _head_mask(e: np.ndarray, k: int) -> np.ndarray:
    """True on each match's first ``k`` rows (``e`` sorted)."""
    firsts = np.r_[True, e[1:] != e[:-1]]
    pos = np.arange(len(e)) - np.maximum.accumulate(np.where(firsts, np.arange(len(e)), 0))
    return pos < k


def _maybe(value: np.ndarray, known: np.ndarray):
    """A boolean column that is blank where the check cannot be made."""
    import pandas as pd
    return pd.Series(value, dtype=object).where(pd.Series(known))


def _col(values, type_: pa.DataType) -> pa.Array:
    import pandas as pd
    s = pd.Series(values)
    return pa.array(s.astype(object).where(s.notna(), None).tolist() if s.dtype == object else s, type_,
                    from_pandas=True)


def build(inputs: dict) -> dict:
    return tables(infer(inputs))


def tables(out: dict | None) -> dict:
    """The inference's frames as the two Arrow tables."""
    if out is None:
        return {GOAL.name: GOAL.schema.empty_table(), TIMELINE.name: TIMELINE.schema.empty_table()}
    g, m = out["goals"], out["matches"]
    event_id = m["event_id"].to_numpy()
    gcols = {
        "event_id": pa.array(event_id[g["e"].to_numpy()], pa.int64()),
        "goal_no": _col(g["goal_no"], pa.int16()),
        "pt": pa.array(g["pt"].to_numpy(), pa.int64()).cast(TS),
        "pt_before": _col(g["t_prev"], pa.int64()).cast(TS),
        "suspended": pa.array(g["suspended"].to_numpy(), pa.bool_()),
        "timed": pa.array(g["timed"].to_numpy(), pa.bool_()),
        "elapsed_min": _col(g["elapsed_min"], pa.float64()),
        "half": _col(g["half"], pa.int8()),
        "match_min": _col(g["match_min"], pa.float64()),
        "side": pa.array([SIDES[int(s) + 1] for s in g["side"].to_numpy()], pa.string()),
        "confidence": _col(g["confidence"], pa.float64()),
        "evidence": pa.array(g["kind"].tolist(), pa.string()),
        "by_settlement": pa.array((g["edit"] != "").to_numpy(), pa.bool_()),
        "home_score": _col(g["home_score"], pa.int16()),
        "away_score": _col(g["away_score"], pa.int16()),
        "mo_jump": _col(g["mo"], pa.float64()),
        "prob_jump": _col(g["l1"], pa.float64()),
        "ou_jump": _col(g["ou"], pa.float64()),
        "ah_jump": _col(g["dmu"], pa.float64()),
    }
    for name in ROLES:
        for when in ("before", "after"):
            gcols[f"{name}_{when}"] = _col(g[f"{name}_{when}"], pa.float64())
    mcols = {c.name: _col(m[c.name], c.type) for c in TIMELINE.columns}
    return {GOAL.name: pa.table(gcols), TIMELINE.name: pa.table(mcols)}
