"""gold/team_rating and gold/rating_gap: market-implied team ratings on one global scale, strictly point in time,
and where they disagree with the price at each early checkpoint.

The closing price is the market's own measure of a team. At the close each match gives the two goal expectancies
the market implies (gold/implied at checkpoint 0), and they are modelled as

    log lam_home = level + home_adv + att[home] - def[away]
    log lam_away = level            + att[away] - def[home]

A team is Betfair's team id, the same in every competition, so domestic cups, continental cups, internationals and
friendlies tie the leagues together and the market's own pricing of those matches fixes the scale between them.
Two combinations of a team's states are all the model needs, and each has one observation a match:

    strength = att + def:   log(lam_home / lam_away) = home_adv + strength[home] - strength[away]
    tempo    = att - def:   log(lam_home * lam_away) = 2 level + home_adv + tempo[home] + tempo[away]

so att = (strength + tempo) / 2 and def = (strength - tempo) / 2, and the two problems separate exactly.

The filter. An online update with a diagonal covariance (each team's variance kept, the covariances between teams
dropped) was tried first and is not used: a league's teams move together against the rest of the world, and a
diagonal filter cannot hold that shared uncertainty, so a cup tie between two leagues moves only its two sides and
the leagues' relative level is learned a few per cent a match. On the odd months of 2026 it explained 0.42 to 0.45
of the closing log rate ratio (0.25 to 0.29 between sides of different countries with ten or more earlier matches
each) against 0.56 (0.66) for the filter below. What is used is the same filter in its exact, information form
with fading memory: every match adds its observation to the normal equations, a match's weight halves every
``HALF_LIFE_*`` days (the variance growing with the time since a side last played), a ridge is the prior (every
team at the centre of the scale; wide for strength, tighter for tempo), and the equations are solved for every
team together at the start of each UTC day from the closes known before it. The system is sparse (a team meets a
few dozen others), so a day costs a few hundredths of a second.

Point in time. A match's close is known at the later of its scheduled and its actual kick-off. A rating "as of" a
moment is the solve at the start of that moment's UTC day: it holds only closes known before that midnight, so it
is stamped by time and never by match order, and so are the home advantages and the level. The rating before a
match is as of its kick-off (the earlier of the scheduled and the actual one): it never holds its own close nor
any close of the same day. gold/rating_gap goes further: at each checkpoint the ratings are as of the
checkpoint's own moment, so a row stamped a day before the off does not know a close fixed since. Nothing on a
row of gold/rating_gap with cp_min > 0 comes from the match's own close or result (the closing-price model reads
every numeric column of it as a feature).

Home advantage is not one number: the market prices it about 0.36 (in log rate ratio) in Brazil, Argentina and
Colombia, 0.12 in Japan, 0.20 in England, against 0.24 over all. It is estimated each day beside the ratings (one
alternating step a day, carried from day to day) as one global value plus a deviation for the event's country
code (blank is its own group: internationals and cups, many on neutral ground), shrunk to the global value where
a country has few matches; without the country term the odd months' fit falls from 0.563 to 0.544. The level is
one global value: a league's scoring is carried by its teams' tempo.

One scale, within a pool. Strengths are differences, fixed only between teams that a chain of rated matches joins.
Of the 8,815 teams with a rated match in 2026, 5,570 (89 per cent of the matches) are one pool; women's football,
many youth and reserve leagues and unlinked lower leagues are pools of their own, each centred on its own average
by the prior. ``pool_teams`` (gold/team_rating) and ``linked`` (gold/rating_gap) say so: where the two sides are
not linked the rating columns carry no information about the match (R2 below zero on the 2026 closes), and
strengths must not be compared across pools.

A side with no earlier match has no rating of its own: before its first match it is given its opponent's (an even
match bar the home advantage; the centre of the scale if both are new), flagged by ``n_prior`` = 0. Standard
deviations are conditional (a team's rating given the others'): ``NOISE_*`` over the root of the decayed count of
matches behind the rating plus the prior, with ``NOISE_*`` fitted so that the squared miss of a close is one
match's own noise (sd about 0.19 in log rate ratio, 0.09 in log rate product) plus the two sides' variances. They
do not carry the uncertainty of a pool's level against another.

Betfair's team id is keyed by the runner's name, so two squads Betfair names alike (a first team and a youth or
women's side listed under the same name) share an id and a rating here.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from football.warehouse.gold_implied import draw_independent, home_minus_away, p_over25
from football.warehouse.schema import TS, Column, TableSpec

VERSION = 3          # 2: a close whose books are not one market's prices (no valid rho) rates nobody
SCOPE = "all"
INPUTS = ("gold/event", "gold/implied")
COLUMNS = {
    "gold/event": ["event_id", "home_id", "away_id", "kickoff", "kickoff_actual", "went_in_play", "country_code"],
    "gold/implied": ["event_id", "cp_min", "cp_time", "p_home", "p_draw", "p_away", "p_over", "total", "supremacy",
                     "lam_home", "lam_away", "rho"],
}

DAY_MS = 86_400_000
#: the checkpoints of gold/rating_gap: minutes before the scheduled kick-off; 0 is the close
GAP_CHECKPOINTS = (1440, 360, 120, 60, 15, 0)

# The filter's constants, tuned on the odd-numbered months of 2026 and reported on the even ones
HALF_LIFE_S = 45.0        # days for a match's weight in the strength equations to halve
HALF_LIFE_T = 30.0        # ... in the tempo equations
RIDGE_S = 0.02            # the prior on a strength, in matches of weight 1 (small: wide)
RIDGE_T = 0.5             # ... on a tempo (tighter: sides differ far less in tempo than in strength)
RIDGE_HOME = 5.0          # matches of weight a country's home advantage is shrunk to the global one by
WINDOW_DAYS = 400         # matches older than this are dropped (their weight is under a quarter of a per cent)
NOISE_S = 0.30            # the scale of a strength's standard deviation: NOISE_S / sqrt(RIDGE_S + decayed matches)
NOISE_T = 0.17            # ... of a tempo's
RHO = -0.08               # the market's usual low-score correction (median of the closes): the draw premium

TEAM_RATING = TableSpec(
    "gold/team_rating",
    "One row a team a match: the team's market-implied rating before the match (from closes fixed before the "
    "match's UTC day only) and after it. A team's latest rating is its last row's.",
    (
        Column("team_id", pa.int64(), "Betfair's team id (the side's selection id on match odds).", False),
        Column("event_id", pa.int64(), "Betfair's event id: the match.", False),
        Column("kickoff", TS, "The scheduled kick-off.", False),
        Column("is_home", pa.bool_(), "Whether the team is the side named first.", False),
        Column("opponent_id", pa.int64(), "The other side's team id.", False),
        Column("country_code", pa.string(), "Betfair's country code for the match; blank on many cups and "
               "internationals."),
        Column("as_of", TS, "The moment the rating before the match is as of: the start of the UTC day of the "
               "kick-off. Only matches that went in play before it are in the rating.", False),
        Column("n_prior", pa.int32(), "Matches behind the rating before this one: the team's matches of the 400 "
               "days before as_of whose close was fixed before it. 0: the team has no rating of its own (it is "
               "new, or back after more than 400 days).", False),
        Column("days_since", pa.float64(), "Days from the team's last rated match, however old, to this kick-off; "
               "blank if it has none."),
        Column("att", pa.float64(), "Attack before the match: (strength + tempo) / 2, in log goals.", False),
        Column("def", pa.float64(), "Defence before the match: (strength - tempo) / 2, in log goals (higher "
               "concedes less).", False),
        Column("strength", pa.float64(), "att + def before the match: what decides the ratio of the two sides' "
               "goal expectancies. A difference of 0.1 is about 10 per cent on that ratio. Only differences "
               "between teams mean anything; the scale's centre is the average team Betfair lists. With "
               "n_prior = 0 it is the opponent's (0 if both sides are new).", False),
        Column("tempo", pa.float64(), "att - def before the match: what the team adds to the log of the product "
               "of the two goal expectancies (high: its matches are priced for goals).", False),
        Column("att_sd", pa.float64(), "Standard deviation of att (conditional on the other teams' ratings).",
               False),
        Column("def_sd", pa.float64(), "Standard deviation of def (conditional).", False),
        Column("strength_sd", pa.float64(), "Standard deviation of strength, conditional on the other teams' "
               "ratings: NOISE_S over the root of the decayed count of matches behind it plus the prior.", False),
        Column("tempo_sd", pa.float64(), "Standard deviation of tempo (conditional).", False),
        Column("pool_teams", pa.int32(), "Teams on the same scale as this one before the match: those joined to it "
               "by chains of the rated matches behind the rating (itself included; 1: none). Strengths compare "
               "within a pool only: women's football, many youth leagues and unlinked lower leagues are pools "
               "of their own, each centred on its own average.", False),
        Column("rated", pa.bool_(), "Whether this match's close enters later ratings: it went in play with a "
               "complete closing 1X2 and over/under book that one market could have made (gold/implied gives it "
               "a rho).", False),
        Column("after_as_of", TS, "The moment the rating after the match is as of: the start of the UTC day "
               "after the kick-off.", False),
        Column("n_after", pa.int32(), "Matches behind the rating after the match.", False),
        Column("att_after", pa.float64(), "Attack after the match; blank while the team has no rated match."),
        Column("def_after", pa.float64(), "Defence after the match."),
        Column("strength_after", pa.float64(), "Strength after the match."),
        Column("tempo_after", pa.float64(), "Tempo after the match."),
    ),
    key=("team_id", "event_id"), sort=("team_id", "kickoff", "event_id"),
)

RATING_GAP = TableSpec(
    "gold/rating_gap",
    "One row an event a checkpoint (1440, 360, 120, 60, 15 minutes before the scheduled kick-off, and 0: the "
    "close): what the two sides' ratings as of the checkpoint imply for the match, what the market says there, "
    "and the gap (ratings less market).",
    (
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("cp_min", pa.int32(), "Minutes before the scheduled kick-off; 0 is the close.", False),
        Column("cp_time", TS, "The checkpoint's moment."),
        Column("kickoff", TS, "The scheduled kick-off.", False),
        Column("home_id", pa.int64(), "The home side's team id.", False),
        Column("away_id", pa.int64(), "The away side's team id.", False),
        Column("as_of", TS, "The moment the ratings are as of: the start of the checkpoint's UTC day.", False),
        Column("n_home", pa.int32(), "Matches behind the home side's rating at the checkpoint (0: none of its "
               "own, the opponent's is used).", False),
        Column("n_away", pa.int32(), "Matches behind the away side's rating at the checkpoint.", False),
        Column("linked", pa.bool_(), "Whether the two sides' ratings are on one scale at the checkpoint: a chain "
               "of rated matches joins them. False: each is rated within its own pool (or one has no rating), "
               "and the rating columns say little about the match.", False),
        Column("rt_home_adv", pa.float64(), "The home advantage in force (global plus the event's country), in "
               "log rate ratio.", False),
        Column("rt_log_ratio", pa.float64(), "The ratings' log(lam_home / lam_away): home advantage + home "
               "strength - away strength.", False),
        Column("rt_log_ratio_sd", pa.float64(), "Its standard deviation from the two sides' conditional "
               "standard deviations.", False),
        Column("rt_lam_home", pa.float64(), "The home goal expectancy the ratings imply.", False),
        Column("rt_lam_away", pa.float64(), "The away goal expectancy the ratings imply.", False),
        Column("rt_total", pa.float64(), "The ratings' expected goals in the match.", False),
        Column("rt_supremacy", pa.float64(), "The ratings' expected goal difference (home - away).", False),
        Column("rt_p_home", pa.float64(), "The home win chance at those rates (Dixon-Coles score model with the "
               "market's usual low-score correction RHO).", False),
        Column("rt_p_draw", pa.float64(), "The draw chance at those rates.", False),
        Column("rt_p_away", pa.float64(), "The away win chance at those rates.", False),
        Column("rt_p_over", pa.float64(), "The chance of over 2.5 goals at those rates.", False),
        Column("mk_log_ratio", pa.float64(), "The market's log(lam_home / lam_away) at the checkpoint "
               "(gold/implied); blank without complete 1X2 and over/under books."),
        Column("mk_total", pa.float64(), "The market's expected goals at the checkpoint."),
        Column("mk_supremacy", pa.float64(), "The market's expected goal difference at the checkpoint."),
        Column("mk_p_home", pa.float64(), "The market's home win chance at the checkpoint (complete 1X2 only)."),
        Column("mk_p_draw", pa.float64(), "The market's draw chance at the checkpoint."),
        Column("mk_p_away", pa.float64(), "The market's away win chance at the checkpoint."),
        Column("mk_p_over", pa.float64(), "The market's chance of over 2.5 goals at the checkpoint."),
        Column("gap_log_ratio", pa.float64(), "rt_log_ratio less mk_log_ratio: positive when the ratings make "
               "the home side stronger than the price does."),
        Column("gap_supremacy", pa.float64(), "rt_supremacy less mk_supremacy, in goals."),
        Column("gap_total", pa.float64(), "rt_total less mk_total, in goals."),
        Column("gap_logit_home", pa.float64(), "logit(rt_p_home) less logit(mk_p_home): the gap on the scale of "
               "gold/move.y_logit for the home runner (needs only the 1X2 book)."),
        Column("gap_logit_over", pa.float64(), "logit(rt_p_over) less logit(mk_p_over): the gap on the over 2.5 "
               "runner's scale."),
    ),
    key=("event_id", "cp_min"), sort=("event_id", "cp_min"),
)
SPECS = (TEAM_RATING, RATING_GAP)


# ---------------------------------------------------------------------------------------------------------------
# the filter
# ---------------------------------------------------------------------------------------------------------------

def solve(i, j, y, w, n: int, sign: float, ridge: float):
    """The ``x`` minimising sum w (y - x[i] - sign * x[j])^2 + ridge * sum x^2 over ``n`` teams, and the diagonal
    of its normal equations (ridge plus each team's summed weight). ``sign`` -1: differences (strength); +1: sums
    (tempo)."""
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve
    diag = ridge + np.bincount(i, w, n) + np.bincount(j, w, n)
    k = np.arange(n)
    mat = csc_matrix((np.concatenate([diag, sign * w, sign * w]),
                      (np.concatenate([k, i, j]), np.concatenate([k, j, i]))), shape=(n, n))
    rhs = np.bincount(i, w * y, n) + sign * np.bincount(j, w * y, n)
    return np.asarray(spsolve(mat, rhs, permc_spec="MMD_AT_PLUS_A"), float).reshape(n), diag


def components(i, j, n: int) -> np.ndarray:
    """Each team's pool: a label shared by the teams that chains of these matches join (a team with no match is
    its own pool). Strengths are comparable within a pool only."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    graph = coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def home_advantage(resid, w, country, n_countries: int, ridge: float) -> np.ndarray:
    """The home advantage of each country group from what the strengths leave of the log rate ratio: the weighted
    global mean plus the group's own deviation, shrunk to nothing where the group has little weight."""
    glob = float(np.sum(w * resid) / np.sum(w))
    num = np.bincount(country, w * (resid - glob), n_countries)
    return glob + num / (np.bincount(country, w, n_countries) + ridge)


def rate(home, away, t_obs, observed, ratio, product, country, when, n_teams: int, n_countries: int,
         half_life_s: float = HALF_LIFE_S, half_life_t: float = HALF_LIFE_T, ridge_s: float = RIDGE_S,
         ridge_t: float = RIDGE_T, ridge_home: float = RIDGE_HOME, window_days: int = WINDOW_DAYS) -> dict:
    """The two sides' ratings of every match as of each of its moments ``when`` (ms, shape slots x matches).

    ``home``, ``away``, ``country``: indices; ``t_obs``: when the match's close is known (ms); ``observed``: whether
    it has one; ``ratio`` and ``product``: the close's log(lam_home / lam_away) and log(lam_home * lam_away). A
    moment's ratings are solved from the observed matches with t_obs before the start of its UTC day. Returns
    arrays of shape slots x matches x 2 (home, away): ``strength``, ``tempo``, ``mass_s`` and ``mass_t`` (the
    decayed weight behind each), ``n`` (the matches behind: those of the last ``window_days``, which the rating is
    solved from), ``last`` (the side's last observed match, ms, however old; -1 none), ``pool`` (the
    teams joined to the side by chains of the matches solved from, itself included), and slots x matches:
    ``home_adv``, ``level`` (the constant of the product equation), ``linked`` (the two sides are in one pool:
    their strengths are on one scale).
    """
    obs = np.flatnonzero(observed)
    obs = obs[np.argsort(t_obs[obs], kind="stable")]
    t_sorted = t_obs[obs]
    n_slots, n_ev = when.shape
    day = (when // DAY_MS).ravel()
    order = np.argsort(day, kind="stable")
    days, start = np.unique(day[order], return_index=True)
    stop = np.append(start[1:], len(order))

    shape = (n_slots, n_ev, 2)
    out = {"strength": np.zeros(shape), "tempo": np.zeros(shape), "mass_s": np.zeros(shape),
           "mass_t": np.zeros(shape), "n": np.zeros(shape, np.int64), "last": np.full(shape, -1, np.int64),
           "pool": np.ones(shape, np.int64), "home_adv": np.zeros((n_slots, n_ev)),
           "level": np.zeros((n_slots, n_ev)), "linked": np.zeros((n_slots, n_ev), bool)}
    none, last = np.zeros(n_teams, np.int64), np.full(n_teams, -1, np.int64)
    zeros, alone = np.zeros(n_teams), np.arange(n_teams)
    x_s, x_t, mass_s, mass_t, label, pool = zeros, zeros, zeros, zeros, alone, np.ones(n_teams, np.int64)
    adv, level, done, count = np.zeros(n_countries), 0.0, 0, none
    for d, a, b in zip(days.tolist(), start.tolist(), stop.tolist()):
        hi = int(np.searchsorted(t_sorted, d * DAY_MS, "left"))          # closes fixed before the day began
        if hi > done:
            new = obs[done:hi]
            for side in (home[new], away[new]):
                np.maximum.at(last, side, t_obs[new])
            done = hi
        lo = int(np.searchsorted(t_sorted, (d - window_days) * DAY_MS, "left"))
        idx = obs[lo:hi]
        if len(idx):
            i, j, c = home[idx], away[idx], country[idx]
            # the matches behind a rating are the ones it is solved from: a side back after more than the window
            # has none, and is new again (it is given its opponent's rating below)
            count = np.bincount(i, minlength=n_teams) + np.bincount(j, minlength=n_teams)
            age = (d * DAY_MS - t_obs[idx]) / DAY_MS
            w = 0.5 ** (age / half_life_s)
            adv = home_advantage(ratio[idx] - x_s[i] + x_s[j], w, c, n_countries, ridge_home)
            x_s, diag = solve(i, j, ratio[idx] - adv[c], w, n_teams, -1.0, ridge_s)
            adv = home_advantage(ratio[idx] - x_s[i] + x_s[j], w, c, n_countries, ridge_home)
            mass_s = diag - ridge_s
            w = 0.5 ** (age / half_life_t)
            level = float(np.sum(w * (product[idx] - x_t[i] - x_t[j])) / np.sum(w))
            x_t, diag = solve(i, j, product[idx] - level, w, n_teams, 1.0, ridge_t)
            level = float(np.sum(w * (product[idx] - x_t[i] - x_t[j])) / np.sum(w))
            mass_t = diag - ridge_t
            label = components(i, j, n_teams)
            pool = np.bincount(label, minlength=n_teams)[label]
        else:
            x_s, x_t, mass_s, mass_t, label, pool = zeros, zeros, zeros, zeros, alone, np.ones(n_teams, np.int64)
            count = none
        slot, ev = np.divmod(order[a:b], n_ev)
        for side, team in ((0, home[ev]), (1, away[ev])):
            out["strength"][slot, ev, side] = x_s[team]
            out["tempo"][slot, ev, side] = x_t[team]
            out["mass_s"][slot, ev, side] = mass_s[team]
            out["mass_t"][slot, ev, side] = mass_t[team]
            out["n"][slot, ev, side] = count[team]
            out["last"][slot, ev, side] = last[team]
            out["pool"][slot, ev, side] = pool[team]
        out["linked"][slot, ev] = label[home[ev]] == label[away[ev]]
        out["home_adv"][slot, ev] = adv[country[ev]]
        out["level"][slot, ev] = level

    # a side with nothing behind it is given its opponent's rating (the centre of the scale if both are new)
    new = out["mass_s"] <= 0
    for name in ("strength", "tempo"):
        other = out[name][:, :, ::-1]
        out[name] = np.where(new, np.where(new[:, :, ::-1], 0.0, other), out[name])
    return out


def chances(lam_home, lam_away, rho: float = RHO) -> dict:
    """The 1X2 and over 2.5 chances of a Dixon-Coles score model at these rates (vectorised). The total fixes the
    over 2.5 chance and the goal difference's distribution fixes the draw and home less away (gold_implied's
    functions); the low-score correction ``rho`` then moves weight between the draw and the two wins alike."""
    lh, la = np.asarray(lam_home, float), np.asarray(lam_away, float)
    total, sup = lh + la, lh - la
    draw, edge = draw_independent(total, sup), home_minus_away(total, sup)
    shift = rho * lh * la * np.exp(-total)                      # the weight the correction moves to each win
    return {"p_home": 0.5 * (1.0 - draw + edge) + shift, "p_draw": draw - 2.0 * shift,
            "p_away": 0.5 * (1.0 - draw - edge) + shift, "p_over": p_over25(total)}


# ---------------------------------------------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------------------------------------------

def _ms(series) -> np.ndarray:
    """A timestamp column as integer milliseconds (blank: the smallest integer)."""
    return pa.Array.from_pandas(series).cast(TS).cast(pa.int64()).fill_null(np.iinfo(np.int64).min) \
        .to_numpy(zero_copy_only=False)


def _logit(p):
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((p > 0) & (p < 1), np.log(p / (1.0 - p)), np.nan)


def _f(a) -> pa.Array:
    return pa.array(np.asarray(a, float), pa.float64(), from_pandas=True)


def observations(event, implied) -> dict:
    """The matches as arrays, in event order: the two sides (``home``, ``away``: indices into ``teams``), the
    country group, the scheduled kick-off and the moment the match began (ms), whether its close is an observation
    (it went in play with both closing goal expectancies and a rho a Dixon-Coles market can have: gold/implied
    blanks the rho of books that are not one moment's prices, such as in-running prices where the turn in play
    came late, and a least-squares fit has no defence against those), when that close is known (``t_obs``: the
    later of the scheduled and the actual kick-off) and the close's log rate ratio and log rate product."""
    ev = event[event["home_id"].notna() & event["away_id"].notna() & event["kickoff"].notna()
               & (event["home_id"] != event["away_id"])].drop_duplicates("event_id") \
        .sort_values("event_id", kind="stable").reset_index(drop=True)
    n_ev = len(ev)
    home_id, away_id = ev["home_id"].to_numpy("int64"), ev["away_id"].to_numpy("int64")
    teams, idx = np.unique(np.concatenate([home_id, away_id]), return_inverse=True)
    cc = ev["country_code"].fillna("").to_numpy(str)
    countries, country = np.unique(cc, return_inverse=True)
    kickoff = _ms(ev["kickoff"])
    began = _ms(ev["kickoff_actual"])
    began = np.where(began == np.iinfo(np.int64).min, kickoff, began)
    close = implied[implied["cp_min"] == 0].drop_duplicates("event_id").set_index("event_id")
    lam_h = close["lam_home"].reindex(ev["event_id"]).to_numpy(float)
    lam_a = close["lam_away"].reindex(ev["event_id"]).to_numpy(float)
    sound = np.isfinite(close["rho"].reindex(ev["event_id"]).to_numpy(float)) if "rho" in close.columns else True
    with np.errstate(invalid="ignore"):
        observed = ev["went_in_play"].fillna(False).to_numpy(bool) & (lam_h > 0) & (lam_a > 0) & sound
    lam_h, lam_a = np.where(observed, lam_h, 1.0), np.where(observed, lam_a, 1.0)
    return {"event_id": ev["event_id"].to_numpy("int64"), "home_id": home_id, "away_id": away_id, "teams": teams,
            "home": idx[:n_ev], "away": idx[n_ev:], "cc": cc, "countries": countries, "country": country,
            "kickoff": kickoff, "began": began, "observed": observed, "t_obs": np.maximum(kickoff, began),
            "ratio": np.log(lam_h / lam_a), "product": np.log(lam_h * lam_a)}


def build(inputs: dict) -> dict:
    im = inputs["gold/implied"].select(COLUMNS["gold/implied"])
    im = im.filter(pc.is_in(im["cp_min"], pa.array(GAP_CHECKPOINTS, im["cp_min"].type))).to_pandas()
    o = observations(inputs["gold/event"].select(COLUMNS["gold/event"]).to_pandas(), im)
    n_ev = len(o["event_id"])
    if not n_ev:
        return {s.name: s.schema.empty_table() for s in SPECS}
    home_id, away_id, cc, kickoff, observed = o["home_id"], o["away_id"], o["cc"], o["kickoff"], o["observed"]

    before = np.minimum(kickoff, o["began"])                    # the rating before the match is as of here
    after = (o["t_obs"] // DAY_MS + 1) * DAY_MS                 # ... and the one after it as of the next day
    early = [c for c in GAP_CHECKPOINTS if c]
    when = np.stack([before, after] + [np.minimum(kickoff - c * 60_000, before) for c in early])
    slot = {0: 0, **{c: 2 + k for k, c in enumerate(early)}}
    r = rate(o["home"], o["away"], o["t_obs"], observed, o["ratio"], o["product"], o["country"], when,
             len(o["teams"]), len(o["countries"]))
    sd_s = NOISE_S / np.sqrt(RIDGE_S + r["mass_s"])
    sd_t = NOISE_T / np.sqrt(RIDGE_T + r["mass_t"])
    as_of = (when // DAY_MS) * DAY_MS

    # ---- gold/team_rating: two rows a match -------------------------------------------------------------------
    cols: dict = {k: [] for k in TEAM_RATING.names}
    for side, (team, opp) in enumerate(((home_id, away_id), (away_id, home_id))):
        s0, t0, s1, t1 = (r[k][j, :, side] for j in (0, 1) for k in ("strength", "tempo"))
        n0, n1, last = r["n"][0, :, side], r["n"][1, :, side], r["last"][0, :, side]
        got = r["mass_s"][1, :, side] > 0
        half_sd = 0.5 * np.sqrt(sd_s[0, :, side] ** 2 + sd_t[0, :, side] ** 2)
        part = {
            "team_id": team, "event_id": o["event_id"], "kickoff": kickoff,
            "is_home": np.full(n_ev, side == 0), "opponent_id": opp, "country_code": np.where(cc == "", None, cc),
            "as_of": as_of[0], "n_prior": n0,
            "days_since": np.where(last >= 0, (kickoff - last) / DAY_MS, np.nan),
            "att": 0.5 * (s0 + t0), "def": 0.5 * (s0 - t0), "strength": s0, "tempo": t0,
            "att_sd": half_sd, "def_sd": half_sd, "strength_sd": sd_s[0, :, side], "tempo_sd": sd_t[0, :, side],
            "pool_teams": r["pool"][0, :, side], "rated": observed, "after_as_of": as_of[1], "n_after": n1,
            "att_after": np.where(got, 0.5 * (s1 + t1), np.nan), "def_after": np.where(got, 0.5 * (s1 - t1), np.nan),
            "strength_after": np.where(got, s1, np.nan), "tempo_after": np.where(got, t1, np.nan),
        }
        for k, v in part.items():
            cols[k].append(v)
    cat = {k: np.concatenate(v) for k, v in cols.items()}
    stamps = ("kickoff", "as_of", "after_as_of")
    floats = [c.name for c in TEAM_RATING.columns if c.type == pa.float64()]
    team_rating = pa.table({
        k: (pa.array(v, pa.int64()).cast(TS) if k in stamps else _f(v) if k in floats
            else pa.array(v.tolist(), pa.string()) if k == "country_code" else pa.array(v))
        for k, v in cat.items()})

    # ---- gold/rating_gap: the event's checkpoints that the market has something at -----------------------------
    pos = {e: k for k, e in enumerate(o["event_id"].tolist())}
    g = im[im["cp_min"].isin(GAP_CHECKPOINTS) & im["event_id"].isin(pos)].drop_duplicates(["event_id", "cp_min"])
    e = g["event_id"].map(pos).to_numpy("int64")
    s = g["cp_min"].map(slot).to_numpy("int64")
    strength, tempo = r["strength"][s, e], r["tempo"][s, e]                      # rows x 2
    rt_ratio = r["home_adv"][s, e] + strength[:, 0] - strength[:, 1]
    rt_product = r["level"][s, e] + tempo[:, 0] + tempo[:, 1]
    rt_h, rt_a = np.exp(0.5 * (rt_product + rt_ratio)), np.exp(0.5 * (rt_product - rt_ratio))
    p = chances(rt_h, rt_a, RHO)
    mk_h, mk_a = g["lam_home"].to_numpy(float), g["lam_away"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mk_ratio = np.where((mk_h > 0) & (mk_a > 0), np.log(mk_h / mk_a), np.nan)
    mk_total, mk_sup = g["total"].to_numpy(float), g["supremacy"].to_numpy(float)
    mk_sup = np.where(np.isfinite(mk_ratio), mk_sup, np.nan)
    mk_p_home, mk_p_over = g["p_home"].to_numpy(float), g["p_over"].to_numpy(float)
    rating_gap = pa.table({
        "event_id": pa.array(g["event_id"].to_numpy("int64"), pa.int64()),
        "cp_min": pa.array(g["cp_min"].to_numpy("int64"), pa.int32()),
        "cp_time": pa.Array.from_pandas(g["cp_time"]).cast(TS),
        "kickoff": pa.array(kickoff[e], pa.int64()).cast(TS),
        "home_id": pa.array(home_id[e], pa.int64()),
        "away_id": pa.array(away_id[e], pa.int64()),
        "as_of": pa.array(as_of[s, e], pa.int64()).cast(TS),
        "n_home": pa.array(r["n"][s, e, 0], pa.int32()),
        "n_away": pa.array(r["n"][s, e, 1], pa.int32()),
        "linked": pa.array(r["linked"][s, e]),
        "rt_home_adv": _f(r["home_adv"][s, e]),
        "rt_log_ratio": _f(rt_ratio),
        "rt_log_ratio_sd": _f(np.sqrt(sd_s[s, e, 0] ** 2 + sd_s[s, e, 1] ** 2)),
        "rt_lam_home": _f(rt_h), "rt_lam_away": _f(rt_a),
        "rt_total": _f(rt_h + rt_a), "rt_supremacy": _f(rt_h - rt_a),
        "rt_p_home": _f(p["p_home"]), "rt_p_draw": _f(p["p_draw"]), "rt_p_away": _f(p["p_away"]),
        "rt_p_over": _f(p["p_over"]),
        "mk_log_ratio": _f(mk_ratio), "mk_total": _f(mk_total), "mk_supremacy": _f(mk_sup),
        "mk_p_home": _f(mk_p_home), "mk_p_draw": _f(g["p_draw"].to_numpy(float)),
        "mk_p_away": _f(g["p_away"].to_numpy(float)), "mk_p_over": _f(mk_p_over),
        "gap_log_ratio": _f(rt_ratio - mk_ratio),
        "gap_supremacy": _f(rt_h - rt_a - mk_sup),
        "gap_total": _f(rt_h + rt_a - mk_total),
        "gap_logit_home": _f(_logit(p["p_home"]) - _logit(mk_p_home)),
        "gap_logit_over": _f(_logit(p["p_over"]) - _logit(mk_p_over)),
    })
    return {TEAM_RATING.name: team_rating, RATING_GAP.name: rating_gap}
