"""Club marts: the same price data read as a record of football clubs.

A betting market is a running valuation of every team it prices: how strong it is (the closing prices of its
matches), how its matches went (the in-play path), and whether its results ran ahead of both. These tables turn
the warehouse's match-grain facts round so that the team is the subject. They are what the consultancy work
reads: benchmarking a side against what the market expected of it, finding sides whose results flatter or hide
their level, and putting clubs from different countries on one scale.

**club/team_match**, one row a team a match (two a match): the market's view before the kick-off (win, draw and
loss chances, expected points, goal expectancies), how the market moved on the team in the day before, the result
and points, the inferred goals where the match's timeline holds together, what the result was worth against the
expectation, the in-play record (expected points averaged over the match, minutes leading, level and trailing,
the half-time position), and the team's and the opponent's market-implied rating going in.

Three measures of a result, in order of noise:
- ``pts_vs_xpts`` = points - xpts_pre: the result against the pre-match expectation. Mostly luck in one match;
- ``perf`` = xpts_inplay - xpts_pre: how the match went, every minute counting (gold/inplay). A side that led for
  an hour and conceded twice late did well on this and took one point;
- ``luck`` = points - xpts_inplay: what the final whistle gave beyond the run of the match.
They sum: pts_vs_xpts = perf + luck. Over a season perf is the part that says something about the team.

**club/team_summary**, one row a team: its matches in the warehouse added up (results, expectation, performance,
luck with their standard errors, the goals the market expected, the rating now and how the market has re-rated
the side against its country's other sides, the strength of its opponents, how the market moved on it, recent
form) and its standing among its country's sides. Inferred goals are not added up here: a score can be inferred
only where the match traded through to the end, which one-sided matches often do not, so the matches with a score
are not a fair sample of a team's season (on 2026 they average 1.22 goals a side where the market expected 1.40).

Rankings hold within a pool: ratings are differences, fixed only between teams that a chain of matches joins
(gold/team_rating), so a side is ranked among the sides of its country and category that its recent matches
connect it to. Women's football, many youth leagues and isolated lower leagues are pools of their own.

**club/country_strength**, one row a country code and category: how strong its sides are on the one scale the
ratings share (the best, the mean of the best five, the median), and what its football looks like to the market
(goals expected, home advantage, draws). Betfair gives a country, not a division: the division-level version of
this table needs the link to API-Football's leagues (gold/link).

Everything is the market's valuation, not event data: it says how good and how fortunate a team has been, not
why. A team needs matches for any of it to mean much: ``matches`` and the standard errors are there to be read.
"""

from __future__ import annotations

import pyarrow as pa

from football.warehouse.schema import TS, Column, TableSpec, from_frame

VERSION = 1
SCOPE = "all"
INPUTS = ("gold/event", "gold/implied", "gold/inplay", "gold/timeline", "gold/team_rating", "gold/team")
COLUMNS = {
    "gold/event": ["event_id", "home_id", "away_id", "home_name", "away_name", "country_code", "category", "kickoff",
                   "result", "over25", "went_in_play"],
    "gold/implied": ["event_id", "cp_min", "p_home", "p_draw", "p_away", "lam_home", "lam_away", "total"],
    "gold/inplay": ["event_id", "filled", "xpts_inplay_home", "xpts_inplay_away", "ht_p_home", "ht_p_draw",
                    "ht_p_away", "max_p_home", "min_p_home", "max_p_away", "min_p_away", "travel", "decided_min"],
    "gold/timeline": ["event_id", "usable", "consistent", "reconciled", "reliability", "n_unplaced", "home_goals",
                      "away_goals", "min_home_leading", "min_level", "min_home_trailing"],
    "gold/team_rating": ["team_id", "event_id", "n_prior", "strength", "strength_sd", "pool_teams", "n_after",
                         "strength_after"],
    "gold/team": ["team_id", "team_name", "country_code"],
}

MOVES = (1440, 360, 60)           # the checkpoints the market's move on a team is measured from
MIN_MATCHES = 10                  # matches a team needs to be ranked, and behind a rating for it to have settled
POOL_DAYS = 400                   # matches this recent join teams into a pool (the ratings' own window)
POOL_MIN = 20                     # teams a pool needs for its sides to be ranked
ACTIVE_DAYS = 120                 # a ranked side played within this long of the last match held
DRIFT_MIN = 20                    # sides rated in two consecutive months that a month's scale drift is read from
COUNTRY_MIN = 5                   # ranked sides with a rating change that a country's own change is read from
FORM = (5, 10)

TEAM_MATCH = TableSpec(
    "club/team_match",
    "One row a team a match: the market's expectation, the result against it, the in-play record and the ratings.",
    (
        Column("team_id", pa.int64(), "Betfair's team id.", False),
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("kickoff", TS, "The scheduled kick-off."),
        Column("is_home", pa.bool_(), "Whether the team was the home side.", False),
        Column("opponent_id", pa.int64(), "The other side's team id."),
        Column("team_name", pa.string(), "The team's name in this match."),
        Column("opponent_name", pa.string(), "The opponent's."),
        Column("country_code", pa.string(), "The match's country code (Betfair's; blank on many cups)."),
        Column("category", pa.string(), "senior, women, youth or reserves."),
        Column("p_win", pa.float64(), "The team's chance of winning at the close."),
        Column("p_draw", pa.float64(), "The draw's chance at the close."),
        Column("p_loss", pa.float64(), "The team's chance of losing at the close."),
        Column("xpts_pre", pa.float64(), "Expected points at the close: 3 p_win + p_draw."),
        Column("lam_for", pa.float64(), "The goals the market expected the team to score."),
        Column("lam_against", pa.float64(), "The goals it expected the team to concede."),
        Column("move_24h", pa.float64(), "The move of the team's win chance from 24 hours before the kick-off to "
               "the close, in log odds: positive when the market came for the team."),
        Column("move_6h", pa.float64(), "The same from 6 hours before."),
        Column("move_60m", pa.float64(), "The same from 60 minutes before (team news is out by then)."),
        Column("result", pa.string(), "W, D or L; blank if the match has no result."),
        Column("points", pa.int8(), "3, 1 or 0."),
        Column("goals_for", pa.int16(), "Goals scored as gold/timeline infers them; blank unless the timeline is "
               "usable and its final score agrees with the settlement (as read, or after reconciliation)."),
        Column("goals_against", pa.int16(), "Goals conceded, likewise."),
        Column("score_reliability", pa.float64(), "How often a timeline like this one had exactly the real score "
               "(gold/timeline.reliability): filter on it where exact scores matter. The goal difference is far "
               "safer than the two counts: the usual error is a goal each side missed late in a high-scoring match."),
        Column("pts_vs_xpts", pa.float64(), "points less xpts_pre: the result against the expectation."),
        Column("gd_vs_exp", pa.float64(), "Goal difference less the expected one (lam_for - lam_against)."),
        Column("xpts_inplay", pa.float64(), "Expected points averaged over every minute of the match "
               "(gold/inplay); blank where the match traded too little in play."),
        Column("perf", pa.float64(), "xpts_inplay less xpts_pre: how the match went against the expectation."),
        Column("luck", pa.float64(), "points less xpts_inplay: what the result gave beyond the run of the match."),
        Column("ht_p_win", pa.float64(), "The team's win chance at half time."),
        Column("max_p_win", pa.float64(), "Its highest win chance in play."),
        Column("min_p_win", pa.float64(), "Its lowest."),
        Column("min_leading", pa.float64(), "Minutes the team spent ahead (inferred timeline; blank where a goal "
               "of the match has no minute)."),
        Column("min_level", pa.float64(), "Minutes level."),
        Column("min_trailing", pa.float64(), "Minutes behind."),
        Column("travel", pa.float64(), "How far the match's chances moved in play (gold/inplay): an eventful match."),
        Column("decided_min", pa.int16(), "The minute from which the result was settled in the market's eyes."),
        Column("strength", pa.float64(), "The team's market-implied strength going into the match "
               "(gold/team_rating); blank until it has a match behind it."),
        Column("strength_sd", pa.float64(), "That rating's standard deviation."),
        Column("n_prior", pa.int32(), "Earlier matches behind the rating."),
        Column("pool_teams", pa.int32(), "Teams on the same scale as this one going in: strengths compare only "
               "within a pool (gold/team_rating)."),
        Column("opp_strength", pa.float64(), "The opponent's strength going in."),
        Column("strength_after", pa.float64(), "The team's strength once this match's closing price is known."),
    ),
    key=("team_id", "event_id"), sort=("team_id", "kickoff", "event_id"),
)

TEAM_SUMMARY = TableSpec(
    "club/team_summary",
    "One row a team: its matches in the warehouse added up, its rating now, and its standing in its country.",
    (
        Column("team_id", pa.int64(), "Betfair's team id.", False),
        Column("team_name", pa.string(), "The name it is most often given."),
        Column("country_code", pa.string(), "The country code its home matches most often carry."),
        Column("category", pa.string(), "senior, women, youth or reserves: its matches' most common."),
        Column("matches", pa.int32(), "Matches with a result.", False),
        Column("first_kickoff", TS, "Its first match with a result."),
        Column("last_kickoff", TS, "Its last."),
        Column("wins", pa.int32(), "Matches won.", False),
        Column("draws", pa.int32(), "Matches drawn.", False),
        Column("losses", pa.int32(), "Matches lost.", False),
        Column("ppm", pa.float64(), "Points a match."),
        Column("xpts_pre_pm", pa.float64(), "Expected points a match at the close."),
        Column("pts_vs_xpts_pm", pa.float64(), "Points a match beyond the expectation."),
        Column("pts_vs_xpts_se", pa.float64(), "Its standard error: read the figure against this."),
        Column("n_inplay", pa.int32(), "Matches with an in-play record.", False),
        Column("perf_pm", pa.float64(), "In-play performance a match (xpts_inplay less xpts_pre): the part of the "
               "over-performance that the run of the matches supports."),
        Column("perf_se", pa.float64(), "Its standard error."),
        Column("luck_pm", pa.float64(), "Luck a match (points less xpts_inplay): results beyond the run of play."),
        Column("luck_se", pa.float64(), "Its standard error."),
        Column("xgf_pm", pa.float64(), "Goals the market expected it to score a match (all matches with rates)."),
        Column("xga_pm", pa.float64(), "Goals the market expected it to concede a match."),
        Column("fav_share", pa.float64(), "Share of its matches it started as the likelier winner."),
        Column("move_24h_pm", pa.float64(), "The market's mean move on the team over the last 24 hours, log odds: "
               "persistently positive means the early price undervalues it."),
        Column("opp_strength_pm", pa.float64(), "Its opponents' mean strength: the difficulty of its matches."),
        Column("strength", pa.float64(), "Its market-implied strength after its last match."),
        Column("strength_first", pa.float64(), "Its strength going into the first match that had ten matches "
               "behind the rating (before that a rating is still finding its level)."),
        Column("strength_change", pa.float64(), "How the market has re-rated it since then, against the median "
               "established side of its pool: strength less strength_first, less the drift of the scale between "
               "the two months (the scale's centre is the average listed team and moves as teams join; the "
               "median side rated in both of two consecutive months is held still). Blank where the pool has "
               "too few sides to read the drift."),
        Column("country_change", pa.float64(), "The median strength_change of its country's ranked sides (same "
               "category and pool, five or more of them): how the rating of the country's football as a whole "
               "moved, which in the first seasons of data is mostly the scale between leagues being learned "
               "from cup and continental matches."),
        Column("rerating", pa.float64(), "strength_change less country_change: how the market has re-rated the "
               "side against the other sides of its country. The figure to read for a club; with only a season "
               "of data some of a promoted side's rise is still the level between divisions being learned."),
        Column("ppm_last5", pa.float64(), "Points a match over its last five matches."),
        Column("perf_last5", pa.float64(), "In-play performance a match over its last five with a record."),
        Column("ppm_last10", pa.float64(), "Points a match over its last ten."),
        Column("perf_last10", pa.float64(), "In-play performance a match over its last ten with a record."),
        Column("pool_id", pa.int64(), "The rating pool it is in: the smallest team id among the teams its "
               "matches of the last 400 days join it to. Strengths compare within a pool only."),
        Column("pool_size", pa.int32(), "Teams in that pool."),
        Column("ranked", pa.bool_(), "It is in a ranking: ten or more matches, a rating, a match in the last 120 "
               "days of the data, and a pool of twenty or more teams.", False),
        Column("country_rank", pa.int32(), "Its place by strength among the ranked sides of its country, "
               "category and pool (1 is the strongest)."),
        Column("country_teams", pa.int32(), "Sides in that ranking."),
        Column("pct_overall", pa.float64(), "The share of the ranked sides of its category and pool it is "
               "stronger than."),
    ),
    key=("team_id",), sort=("team_id",),
)

COUNTRY = TableSpec(
    "club/country_strength",
    "One row a country code and category: the strength of its sides on the ratings' one scale, and what its "
    "matches look like to the market.",
    (
        Column("country_code", pa.string(), "Betfair's country code.", False),
        Column("category", pa.string(), "senior, women, youth or reserves.", False),
        Column("teams", pa.int32(), "Ranked sides of the country in the category's largest pool (the one scale "
               "the strength columns are on).", False),
        Column("strength_top", pa.float64(), "The strongest side's strength."),
        Column("strength_top5", pa.float64(), "The mean of the five strongest; blank with fewer than five."),
        Column("strength_median", pa.float64(), "The median side's."),
        Column("strength_mean", pa.float64(), "The mean."),
        Column("matches", pa.int32(), "Matches with a result under the country code.", False),
        Column("total_mean", pa.float64(), "Goals the market expected a match."),
        Column("home_edge_mean", pa.float64(), "The home side's expected goals less the away side's, a match."),
        Column("p_home_mean", pa.float64(), "The mean closing chance of a home win."),
        Column("home_rate", pa.float64(), "The share of matches the home side won."),
        Column("draw_rate", pa.float64(), "The share drawn."),
        Column("over25_rate", pa.float64(), "The share with three or more goals."),
    ),
    key=("country_code", "category"), sort=("country_code", "category"),
)
SPECS = (TEAM_MATCH, TEAM_SUMMARY, COUNTRY)


def _logit(p):
    import numpy as np
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p / (1.0 - p))


def team_match_frame(inputs: dict):
    """club/team_match as a pandas frame."""
    import numpy as np
    import pandas as pd

    ev = inputs["gold/event"].select(COLUMNS["gold/event"]).to_pandas()
    ev = ev[ev["home_id"].notna() & ev["away_id"].notna()].drop_duplicates("event_id").reset_index(drop=True)
    imp = inputs["gold/implied"].select(COLUMNS["gold/implied"]).to_pandas()
    close = imp[imp["cp_min"] == 0].drop(columns="cp_min").drop_duplicates("event_id")
    ev = ev.merge(close, on="event_id", how="left")
    for cp in MOVES:
        e = imp.loc[imp["cp_min"] == cp, ["event_id", "p_home", "p_away"]].drop_duplicates("event_id")
        ev = ev.merge(e.rename(columns={"p_home": f"h{cp}", "p_away": f"a{cp}"}), on="event_id", how="left")
    ev = ev.merge(inputs["gold/inplay"].select(COLUMNS["gold/inplay"]).to_pandas().drop_duplicates("event_id"),
                  on="event_id", how="left")
    ev = ev.merge(inputs["gold/timeline"].select(COLUMNS["gold/timeline"]).to_pandas().drop_duplicates("event_id"),
                  on="event_id", how="left")
    filled = ev["filled"].eq(True).to_numpy()                     # a blank (no in-play record) is not filled
    scored = (ev["usable"].eq(True) & (ev["consistent"].eq(True) | ev["reconciled"].eq(True))).to_numpy()
    timed = scored & ev["n_unplaced"].eq(0).to_numpy()            # every goal has a minute

    def side(home: bool) -> pd.DataFrame:
        me, it = ("home", "away") if home else ("away", "home")
        win = "H" if home else "A"
        res = ev["result"]
        points = np.where(res == win, 3.0, np.where(res == "D", 1.0, np.where(res.notna(), 0.0, np.nan)))
        d = pd.DataFrame({
            "team_id": ev[f"{me}_id"].to_numpy(), "event_id": ev["event_id"].to_numpy(), "kickoff": ev["kickoff"],
            "is_home": home, "opponent_id": ev[f"{it}_id"].to_numpy(), "team_name": ev[f"{me}_name"],
            "opponent_name": ev[f"{it}_name"], "country_code": ev["country_code"], "category": ev["category"],
            "p_win": ev[f"p_{me}"], "p_draw": ev["p_draw"], "p_loss": ev[f"p_{it}"],
            "lam_for": ev[f"lam_{me}"], "lam_against": ev[f"lam_{it}"],
            "result": np.where(res == win, "W", np.where(res == "D", "D", np.where(res.notna(), "L", None))),
            "points": points,
            "goals_for": np.where(scored, ev[f"{me}_goals"].to_numpy(float), np.nan),
            "goals_against": np.where(scored, ev[f"{it}_goals"].to_numpy(float), np.nan),
            "score_reliability": np.where(scored, ev["reliability"].to_numpy(float), np.nan),
            "xpts_inplay": np.where(filled, ev[f"xpts_inplay_{me}"].to_numpy(float), np.nan),
            "ht_p_win": ev[f"ht_p_{me}"], "max_p_win": ev[f"max_p_{me}"], "min_p_win": ev[f"min_p_{me}"],
            "min_leading": np.where(timed, ev["min_home_leading" if home else "min_home_trailing"].to_numpy(float),
                                    np.nan),
            "min_level": np.where(timed, ev["min_level"].to_numpy(float), np.nan),
            "min_trailing": np.where(timed, ev["min_home_trailing" if home else "min_home_leading"].to_numpy(float),
                                     np.nan),
            "travel": ev["travel"], "decided_min": ev["decided_min"],
        })
        own = "h" if home else "a"
        for cp, name in zip(MOVES, ("move_24h", "move_6h", "move_60m")):
            d[name] = _logit(ev[f"p_{me}"].to_numpy(float)) - _logit(ev[f"{own}{cp}"].to_numpy(float))
        return d

    d = pd.concat([side(True), side(False)], ignore_index=True)
    d["xpts_pre"] = 3.0 * d["p_win"] + d["p_draw"]
    d["pts_vs_xpts"] = d["points"] - d["xpts_pre"]
    d["gd_vs_exp"] = (d["goals_for"] - d["goals_against"]) - (d["lam_for"] - d["lam_against"])
    d["perf"] = d["xpts_inplay"] - d["xpts_pre"]
    d["luck"] = d["points"] - d["xpts_inplay"]

    rt = inputs["gold/team_rating"].select(COLUMNS["gold/team_rating"]).to_pandas()
    rt = rt.drop_duplicates(["team_id", "event_id"])
    own = (rt["n_prior"] >= 1).to_numpy()                         # with none, the row carries the opponent's rating
    for c in ("strength", "strength_sd", "pool_teams"):
        rt[c] = np.where(own, rt[c].to_numpy(float), np.nan)
    rt["strength_after"] = np.where((rt["n_after"] >= 1).to_numpy(), rt["strength_after"], np.nan)
    d = d.merge(rt[["team_id", "event_id", "strength", "strength_sd", "n_prior", "pool_teams", "strength_after"]],
                on=["team_id", "event_id"], how="left")
    d = d.merge(rt[["team_id", "event_id", "strength"]].rename(columns={"team_id": "opponent_id",
                                                                        "strength": "opp_strength"}),
                on=["opponent_id", "event_id"], how="left")
    return d.drop_duplicates(["team_id", "event_id"]).reset_index(drop=True)


def _se(v):
    import numpy as np
    n = v.count()
    return v.std(ddof=1) / np.sqrt(n.where(n > 1))


def team_summary_frame(tm, teams):
    """club/team_summary as a pandas frame, from club/team_match and gold/team."""
    import pandas as pd

    d = tm[tm["points"].notna()].sort_values(["team_id", "kickoff", "event_id"], kind="stable")
    if d.empty:
        return pd.DataFrame(columns=TEAM_SUMMARY.names)
    g = d.groupby("team_id", sort=True)
    out = pd.DataFrame({
        "matches": g.size(), "first_kickoff": g["kickoff"].min(), "last_kickoff": g["kickoff"].max(),
        "wins": g["points"].agg(lambda s: int((s == 3).sum())), "draws": g["points"].agg(lambda s: int((s == 1).sum())),
        "losses": g["points"].agg(lambda s: int((s == 0).sum())),
        "ppm": g["points"].mean(), "xpts_pre_pm": g["xpts_pre"].mean(),
        "pts_vs_xpts_pm": g["pts_vs_xpts"].mean(), "pts_vs_xpts_se": _se(g["pts_vs_xpts"]),
        "n_inplay": g["perf"].count(), "perf_pm": g["perf"].mean(), "perf_se": _se(g["perf"]),
        "luck_pm": g["luck"].mean(), "luck_se": _se(g["luck"]),
        "xgf_pm": g["lam_for"].mean(), "xga_pm": g["lam_against"].mean(),
        "fav_share": (d["p_win"] > d["p_loss"]).astype(float).where(d["p_win"].notna()).groupby(d["team_id"]).mean(),
        "move_24h_pm": g["move_24h"].mean(), "opp_strength_pm": g["opp_strength"].mean(),
        "category": g["category"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None),
    })
    # the rating now (after the last match that has one) and once it had settled (ten matches behind it)
    r = d[d["strength_after"].notna()].groupby("team_id")["strength_after"].last()
    f = d[d["strength"].notna() & (d["n_prior"] >= MIN_MATCHES)].groupby("team_id")["strength"].first()
    out["strength"], out["strength_first"] = r, f
    for n in FORM:
        tail = d.groupby("team_id").tail(n)
        out[f"ppm_last{n}"] = tail.groupby("team_id")["points"].mean()
        out[f"perf_last{n}"] = d[d["perf"].notna()].groupby("team_id").tail(n).groupby("team_id")["perf"].mean()
    out = out.reset_index()
    names = teams[["team_id", "team_name", "country_code"]].drop_duplicates("team_id")
    out = out.merge(names, on="team_id", how="left")

    pool = pools(d)
    out = out.merge(pool, on="team_id", how="left")
    # the re-rating, net of the scale's drift between the month the rating settled and the month of the last match
    drift = scale_drift(d, pool)
    month = lambda t: t.dt.year * 12 + t.dt.month - 1                     # noqa: E731
    settled = d[d["strength"].notna() & (d["n_prior"] >= MIN_MATCHES)].groupby("team_id")["kickoff"].first()
    last = d[d["strength_after"].notna()].groupby("team_id")["kickoff"].last()
    when = pd.DataFrame({"m0": month(settled), "m1": month(last)}).reset_index().merge(
        out[["team_id", "pool_id"]], on="team_id", how="right")
    c0 = when.merge(drift.rename(columns={"ym": "m0", "level": "c0"}), on=["pool_id", "m0"], how="left")["c0"]
    c1 = when.merge(drift.rename(columns={"ym": "m1", "level": "c1"}), on=["pool_id", "m1"], how="left")["c1"]
    out["strength_change"] = (out["strength"] - out["strength_first"]) - (c1.to_numpy() - c0.to_numpy())
    end = d["kickoff"].max()
    out["ranked"] = ((out["matches"] >= MIN_MATCHES) & out["strength"].notna() & (out["pool_size"] >= POOL_MIN)
                     & (out["last_kickoff"] >= end - pd.Timedelta(days=ACTIVE_DAYS)))
    pool = out[out["ranked"] & out["country_code"].notna()]
    by = ["country_code", "category", "pool_id"]
    out["country_rank"] = pool.groupby(by)["strength"].rank(ascending=False, method="min").reindex(out.index)
    out["country_teams"] = pool.groupby(by)["strength"].transform("size").reindex(out.index)
    moved = pool[pool["strength_change"].notna()].groupby(by)["strength_change"]
    country = moved.transform("median").where(moved.transform("size") >= COUNTRY_MIN)
    out["country_change"] = country.reindex(out.index)
    out["rerating"] = out["strength_change"] - out["country_change"]
    every = out[out["ranked"]]
    below = every.groupby(["category", "pool_id"])["strength"].rank(method="min") - 1
    size = every.groupby(["category", "pool_id"])["strength"].transform("size")
    out["pct_overall"] = (below / (size - 1).where(size > 1)).reindex(out.index)
    return out


def scale_drift(d, pool):
    """(pool_id, ym, level): the level of each pool's rating scale by calendar month (year x 12 + month - 1),
    zero in the pool's first month. A month's step is the median, over the sides with a settled rating in both
    months, of the change in their mean strength: the median established side is taken not to have changed. A
    step that fewer than ``DRIFT_MIN`` sides can give is not read: the level is blank from there on."""
    import numpy as np
    import pandas as pd

    r = d[d["strength"].notna() & (d["n_prior"] >= MIN_MATCHES)].merge(pool[["team_id", "pool_id"]], on="team_id")
    if r.empty:
        return pd.DataFrame({"pool_id": pd.Series(dtype="int64"), "ym": pd.Series(dtype="int64"),
                             "level": pd.Series(dtype="float64")})
    r = r.assign(ym=r["kickoff"].dt.year * 12 + r["kickoff"].dt.month - 1)
    m = r.groupby(["pool_id", "team_id", "ym"], as_index=False)["strength"].mean()
    nxt = m.assign(ym=m["ym"] - 1).rename(columns={"strength": "next"})
    j = m.merge(nxt, on=["pool_id", "team_id", "ym"])
    step = j.assign(d=j["next"] - j["strength"]).groupby(["pool_id", "ym"])["d"].agg(["median", "size"]).reset_index()
    step = step[step["size"] >= DRIFT_MIN]
    rows = []
    for pid, months in m.groupby("pool_id")["ym"]:
        s = step[step["pool_id"] == pid].set_index("ym")["median"]
        level = 0.0
        for ym in range(int(months.min()), int(months.max()) + 1):
            rows.append((pid, ym, level))
            level = level + s[ym] if ym in s.index else np.nan     # the step from this month to the next
    return pd.DataFrame(rows, columns=["pool_id", "ym", "level"])


def pools(d):
    """(team_id, pool_id, pool_size): the teams the matches of the last ``POOL_DAYS`` join, as connected
    components of the graph whose edges are those matches; a pool is named by its smallest team id."""
    import numpy as np
    import pandas as pd
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    recent = d[d["kickoff"] >= d["kickoff"].max() - pd.Timedelta(days=POOL_DAYS)]
    recent = recent[recent["opponent_id"].notna()]
    ids = np.unique(np.r_[recent["team_id"].to_numpy(np.int64), recent["opponent_id"].to_numpy(np.int64)])
    if not len(ids):
        return pd.DataFrame({"team_id": pd.Series(dtype="int64"), "pool_id": pd.Series(dtype="int64"),
                             "pool_size": pd.Series(dtype="int64")})
    a = np.searchsorted(ids, recent["team_id"].to_numpy(np.int64))
    b = np.searchsorted(ids, recent["opponent_id"].to_numpy(np.int64))
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(ids), len(ids)))
    _, label = connected_components(graph, directed=False)
    first = pd.Series(ids).groupby(label).min().to_numpy()             # ids are sorted: the smallest of each pool
    return pd.DataFrame({"team_id": ids, "pool_id": first[label], "pool_size": np.bincount(label)[label]})


def country_frame(tm, summary, events):
    """club/country_strength as a pandas frame."""
    import numpy as np
    import pandas as pd

    pool = summary[summary["ranked"] & summary["country_code"].notna() & summary["category"].notna()]
    # one scale a category: its pool with the most ranked sides
    main = pool.groupby("category")["pool_id"].agg(lambda s: s.value_counts().index[0])
    pool = pool[pool["pool_id"] == pool["category"].map(main)]
    rows = []
    for (cc, cat), s in pool.groupby(["country_code", "category"]):
        v = np.sort(s["strength"].to_numpy(float))[::-1]
        rows.append({"country_code": cc, "category": cat, "teams": len(v), "strength_top": v[0],
                     "strength_top5": float(v[:5].mean()) if len(v) >= 5 else np.nan,
                     "strength_median": float(np.median(v)), "strength_mean": float(v.mean())})
    strength = pd.DataFrame(rows, columns=["country_code", "category", "teams", "strength_top", "strength_top5",
                                           "strength_median", "strength_mean"])
    h = tm[tm["is_home"] & tm["points"].notna() & tm["country_code"].notna() & tm["category"].notna()]
    h = h.merge(events[["event_id", "over25"]], on="event_id", how="left")
    g = h.groupby(["country_code", "category"])
    played = pd.DataFrame({
        "matches": g.size(), "total_mean": (h["lam_for"] + h["lam_against"]).groupby([h["country_code"],
                                                                                    h["category"]]).mean(),
        "home_edge_mean": (h["lam_for"] - h["lam_against"]).groupby([h["country_code"], h["category"]]).mean(),
        "p_home_mean": g["p_win"].mean(), "home_rate": g["points"].agg(lambda s: float((s == 3).mean())),
        "draw_rate": g["points"].agg(lambda s: float((s == 1).mean())), "over25_rate": g["over25"].mean(),
    }).reset_index()
    out = played.merge(strength, on=["country_code", "category"], how="outer")
    out["teams"] = out["teams"].fillna(0)
    out["matches"] = out["matches"].fillna(0)
    return out


def build(inputs: dict) -> dict:
    tm = team_match_frame(inputs)
    summary = team_summary_frame(tm, inputs["gold/team"].select(COLUMNS["gold/team"]).to_pandas())
    events = inputs["gold/event"].select(["event_id", "over25"]).to_pandas().drop_duplicates("event_id")
    country = country_frame(tm, summary, events)
    return {TEAM_MATCH.name: from_frame(TEAM_MATCH, tm), TEAM_SUMMARY.name: from_frame(TEAM_SUMMARY, summary),
            COUNTRY.name: from_frame(COUNTRY, country)}
