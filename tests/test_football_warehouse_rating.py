"""gold/team_rating and gold/rating_gap: the solve is the least-squares one, a noiseless market is recovered across
two leagues tied by a cup, and nothing a checkpoint did not know is in its row (hand-written tiny inputs)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from football.ratings import markets, score_matrix
from football.warehouse import gold_rating as gr
from football.warehouse.parse import iso_ms
from football.warehouse.schema import TS, conform

DAY, HOUR, MIN = 86_400_000, 3_600_000, 60_000
D0 = iso_ms("2026-03-01T00:00:00.000Z")
HOME_ADV, LEVEL = 0.25, 0.30
#: two leagues of four: the second is much the weaker, and its sides' matches are priced for fewer goals
STRENGTH = {1: 0.5, 2: 0.2, 3: -0.1, 4: -0.4, 11: -0.6, 12: -0.9, 13: -1.2, 14: -1.5}
TEMPO = {1: 0.10, 2: -0.05, 3: 0.00, 4: 0.05, 11: -0.20, 12: -0.10, 13: -0.15, 14: -0.25}


def _country(home: int, away: int) -> str:
    """A league match carries its country; a tie between the leagues has no country code."""
    return "" if (home < 10) != (away < 10) else "GB" if home < 10 else "NO"


def _rates(home: int, away: int) -> tuple[float, float]:
    """The closing goal expectancies of a market that prices exactly by STRENGTH and TEMPO."""
    ratio = HOME_ADV + STRENGTH[home] - STRENGTH[away]
    product = LEVEL + TEMPO[home] + TEMPO[away]
    return float(np.exp(0.5 * (product + ratio))), float(np.exp(0.5 * (product - ratio)))


def _match(event_id, home, away, kickoff, lam=None, delay=MIN, in_play=True, cc=None, early=None, rho=-0.08):
    """A match: ``lam`` its closing rates (the true ones by default), ``early`` the rates at every early
    checkpoint (the close's by default), ``delay`` how long after the scheduled kick-off it went in play, ``rho``
    the closing books' low-score correction (None: books that are not one market's prices)."""
    return {"event_id": event_id, "home": home, "away": away, "kickoff": kickoff, "lam": lam or _rates(home, away),
            "began": kickoff + delay if in_play else None, "in_play": in_play, "early": early, "rho": rho,
            "cc": cc if cc is not None else _country(home, away)}


def _world() -> list[dict]:
    """Each league's double round robin (a round a day from D0, at 15:00), a cup tie between the leagues on day 3
    and another on day 8; then on day 10 at 20:00 side 1 plays at side 3, and on day 11 at 15:00 it hosts side 12."""
    rounds = [((0, 1), (2, 3)), ((0, 2), (3, 1)), ((0, 3), (1, 2))]
    out, eid = [], 100
    for leg in (0, 1):
        for r, pairs in enumerate(rounds):
            for league in ((1, 2, 3, 4), (11, 12, 13, 14)):
                for a, b in pairs:
                    h, w = (league[a], league[b]) if leg == 0 else (league[b], league[a])
                    out.append(_match(eid, h, w, D0 + (leg * 3 + r) * DAY + 15 * HOUR))
                    eid += 1
    out.append(_match(eid, 2, 11, D0 + 3 * DAY + 19 * HOUR))
    out.append(_match(eid + 1, 13, 4, D0 + 8 * DAY + 19 * HOUR))
    out.append(_match(900, 3, 1, D0 + 10 * DAY + 20 * HOUR))
    out.append(_match(901, 1, 12, D0 + 11 * DAY + 15 * HOUR, early=(1.9, 0.9)))
    return out


def _inputs(matches) -> dict:
    ev = {k: [] for k in gr.COLUMNS["gold/event"]}
    im = {k: [] for k in gr.COLUMNS["gold/implied"]}
    for m in matches:
        for k, v in (("event_id", m["event_id"]), ("home_id", m["home"]), ("away_id", m["away"]),
                     ("kickoff", m["kickoff"]), ("kickoff_actual", m["began"]), ("went_in_play", m["in_play"]),
                     ("country_code", m["cc"] or None)):
            ev[k].append(v)
        for cp in gr.GAP_CHECKPOINTS:
            if cp == 0 and not m["in_play"]:
                continue
            lh, la = m["lam"] if cp == 0 or not m["early"] else m["early"]
            p = gr.chances(lh, la)
            row = {"event_id": m["event_id"], "cp_min": cp,
                   "cp_time": m["began"] if cp == 0 else m["kickoff"] - cp * MIN,
                   "p_home": float(p["p_home"]), "p_draw": float(p["p_draw"]), "p_away": float(p["p_away"]),
                   "p_over": float(p["p_over"]), "total": lh + la, "supremacy": lh - la, "lam_home": lh,
                   "lam_away": la, "rho": m["rho"] if cp == 0 else -0.08}
            for k, v in row.items():
                im[k].append(v)
    for t, cols in ((ev, ("kickoff", "kickoff_actual")), (im, ("cp_time",))):
        for c in cols:
            t[c] = pa.array(t[c], pa.int64()).cast(TS)
    im["cp_min"] = pa.array(im["cp_min"], pa.int32())
    im["rho"] = pa.array(im["rho"], pa.float64())
    return {"gold/event": pa.table(ev), "gold/implied": pa.table(im)}


def _build(matches) -> tuple:
    out = gr.build(_inputs(matches))
    team = conform(gr.TEAM_RATING, out["gold/team_rating"]).to_pandas()
    gap = conform(gr.RATING_GAP, out["gold/rating_gap"]).to_pandas()
    return (team.sort_values(["team_id", "kickoff", "event_id"]).reset_index(drop=True),
            gap.sort_values(["event_id", "cp_min"]).reset_index(drop=True))


@pytest.fixture(scope="module")
def world():
    matches = _world()
    return matches, *_build(matches)


def _ms(stamps):
    """A timestamp, or a column of them, as integer milliseconds."""
    if hasattr(stamps, "dt"):
        return stamps.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ms]").astype("int64").to_numpy()
    return int(round(stamps.timestamp() * 1000))


# ---------------------------------------------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_the_solve_is_the_ridge_least_squares_solution(sign):
    i, j = np.array([0, 1, 2, 0, 3]), np.array([1, 2, 0, 3, 1])
    y, w = np.array([0.4, -0.1, 0.3, 1.0, -0.6]), np.array([1.0, 0.5, 0.25, 1.0, 0.8])
    a = np.zeros((5, 4))
    a[np.arange(5), i], a[np.arange(5), j] = 1.0, sign
    dense = np.linalg.solve(a.T @ (w[:, None] * a) + 0.1 * np.eye(4), a.T @ (w * y))
    x, diag = gr.solve(i, j, y, w, 4, sign, 0.1)
    assert np.abs(x - dense).max() < 1e-12
    assert diag.tolist() == pytest.approx([0.1 + 2.25, 0.1 + 2.3, 0.1 + 0.75, 0.1 + 1.8])


def test_a_countrys_home_advantage_is_shrunk_to_the_global_one_where_it_has_few_matches():
    resid, w = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1]), np.ones(6)
    adv = gr.home_advantage(resid, w, np.array([0, 0, 1, 1, 1, 1]), 3, ridge=2.0)
    glob = 1.0 / 6
    assert adv.tolist() == pytest.approx([glob + 2 * (0.3 - glob) / 4, glob + 4 * (0.1 - glob) / 6, glob])


def test_the_chances_are_the_dixon_coles_score_matrixs():
    for lh, la, rho in ((1.6, 1.1, -0.08), (2.7, 0.6, -0.15), (0.8, 1.9, 0.03), (1.2, 1.2, 0.0)):
        m = markets(score_matrix(lh, la, rho, n=40))
        p = gr.chances(np.array([lh]), np.array([la]), rho)
        got = [float(p[k][0]) for k in ("p_home", "p_draw", "p_away", "p_over")]
        assert got == pytest.approx([m["p_h"], m["p_d"], m["p_a"], m["p_o25"]], abs=1e-12)


# ---------------------------------------------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------------------------------------------

def test_two_rows_a_match_and_one_a_checkpoint_under_the_contract(world):
    matches, team, gap = world
    assert len(team) == 2 * len(matches) and not team.duplicated(["team_id", "event_id"]).any()
    assert len(gap) == len(gr.GAP_CHECKPOINTS) * len(matches) and not gap.duplicated(["event_id", "cp_min"]).any()
    assert set(gap["cp_min"]) == set(gr.GAP_CHECKPOINTS)
    row = team[(team["team_id"] == 12) & (team["event_id"] == 901)].iloc[0]
    assert (row["is_home"], row["opponent_id"], row["rated"]) == (False, 1, True)
    assert row["att"] == pytest.approx(0.5 * (row["strength"] + row["tempo"]), abs=1e-12)
    assert row["def"] == pytest.approx(0.5 * (row["strength"] - row["tempo"]), abs=1e-12)


def test_a_side_with_no_match_behind_it_is_given_its_opponents_rating(world):
    _, team, gap = world
    first = team[team["event_id"] == 100]                                 # the first day: nobody is rated
    assert first["n_prior"].tolist() == [0, 0] and first["strength"].tolist() == [0.0, 0.0]
    assert first["days_since"].isna().all() and first["strength_after"].notna().all()
    g = gap[(gap["event_id"] == 100) & (gap["cp_min"] == 0)].iloc[0]
    assert (g["n_home"], g["n_away"], g["rt_log_ratio"]) == (0, 0, 0.0)   # an even match: no home advantage yet
    matches = _world() + [_match(950, 77, 1, D0 + 12 * DAY + 15 * HOUR, lam=(1.0, 1.5), cc="GB")]
    team2, gap2 = _build(matches)
    new, old = (team2[(team2["event_id"] == 950) & (team2["team_id"] == t)].iloc[0] for t in (77, 1))
    assert (new["n_prior"], old["n_prior"]) == (0, 8)
    assert (new["strength"], new["tempo"]) == (old["strength"], old["tempo"])
    g = gap2[(gap2["event_id"] == 950) & (gap2["cp_min"] == 60)].iloc[0]
    assert g["rt_log_ratio"] == pytest.approx(g["rt_home_adv"], abs=1e-12)


def test_a_side_back_after_more_than_the_window_is_new_again():
    """Its old matches are no longer in the solve: it has no rating of its own, and the row must say so (a count
    of every match it ever played would present its opponent's rating as its own)."""
    late = D0 + (gr.WINDOW_DAYS + 100) * DAY + 15 * HOUR
    matches = _world() + [_match(960, 3, 4, late - 50 * DAY, cc="GB"), _match(961, 1, 3, late, cc="GB")]
    team, gap = _build(matches)
    back, opp = (team[(team["event_id"] == 961) & (team["team_id"] == t)].iloc[0] for t in (1, 3))
    assert (back["n_prior"], opp["n_prior"]) == (0, 1)
    assert (back["strength"], back["tempo"], back["pool_teams"]) == (opp["strength"], opp["tempo"], 1)
    assert back["days_since"] == pytest.approx(gr.WINDOW_DAYS + 100 - 11, abs=1.0)    # its last match is a fact still
    g = gap[(gap["event_id"] == 961) & (gap["cp_min"] == 60)].iloc[0]
    assert (g["n_home"], g["n_away"], bool(g["linked"])) == (0, 1, False)
    assert g["rt_log_ratio"] == pytest.approx(g["rt_home_adv"], abs=1e-12)
    assert back["n_after"] == 1 and np.isfinite(back["strength_after"])               # ... and rated again after it


def test_a_noiseless_market_is_recovered_on_one_scale_across_the_two_leagues(world):
    _, team, gap = world
    last = team.groupby("team_id").last()
    got = last["strength_after"] - last["strength_after"].mean()
    want = np.array([STRENGTH[t] for t in last.index]) - np.mean(list(STRENGTH.values()))
    assert np.abs(got.to_numpy() - want).max() < 0.05     # two cup ties hold the leagues 1.1 apart against the prior
    for league in ((1, 2, 3, 4), (11, 12, 13, 14)):       # within a league the sides are apart as priced
        inside = got.loc[list(league)].to_numpy() - np.array([STRENGTH[t] for t in league])
        assert np.ptp(inside) < 0.03
    tempo = last["tempo_after"] - last["tempo_after"].mean()
    want_t = np.array([TEMPO[t] for t in last.index]) - np.mean(list(TEMPO.values()))
    assert np.abs(tempo.to_numpy() - want_t).max() < 0.03                 # tighter prior: a little shrunk
    g = gap[(gap["event_id"] == 901) & (gap["cp_min"] == 0)].iloc[0]      # side 1 at home to side 12: never met
    lh, la = _rates(1, 12)
    assert g["rt_home_adv"] == pytest.approx(HOME_ADV, abs=0.01)
    assert g["rt_log_ratio"] == pytest.approx(np.log(lh / la), abs=0.05)
    assert g["rt_total"] == pytest.approx(lh + la, abs=0.1) and g["rt_supremacy"] == pytest.approx(lh - la, abs=0.1)
    assert g["gap_log_ratio"] == pytest.approx(g["rt_log_ratio"] - g["mk_log_ratio"], abs=1e-12)
    assert abs(g["gap_log_ratio"]) < 0.05 and abs(g["gap_logit_home"]) < 0.1
    assert g["rt_p_home"] + g["rt_p_draw"] + g["rt_p_away"] == pytest.approx(1.0, abs=1e-9)


def test_two_leagues_are_one_pool_only_once_a_match_between_them_has_closed(world):
    _, team, gap = world
    side = team[team["team_id"] == 13].set_index("event_id")              # the weak league's third side
    assert side["pool_teams"].tolist() == [1, 2, 4, 4, 8, 8, 8]           # the first cup tie closed on day 3
    tie = gap[(gap["event_id"] == 124) & (gap["cp_min"] == 0)].iloc[0]    # that tie itself: side 2 v side 11
    assert not tie["linked"] and (tie["n_home"], tie["n_away"]) == (3, 3)
    second = gap[(gap["event_id"] == 125) & (gap["cp_min"] == 0)].iloc[0]  # the second tie, on day 8
    assert second["linked"]
    early, later = (gap[(gap["event_id"] == e) & (gap["cp_min"] == 0)].iloc[0] for e in (104, 108))
    assert not early["linked"] and later["linked"]                        # sides 1 and 3 on day 1; 1 and 4 on day 2
    assert not gap[(gap["event_id"] == 100)]["linked"].any()


def test_the_gap_is_the_ratings_less_the_market_at_that_checkpoint(world):
    _, _, gap = world
    g = gap[(gap["event_id"] == 901) & (gap["cp_min"] == 60)].iloc[0]     # priced 1.9 - 0.9 before the close
    assert g["mk_log_ratio"] == pytest.approx(np.log(1.9 / 0.9), abs=1e-12)
    assert (g["mk_total"], g["mk_supremacy"]) == pytest.approx((2.8, 1.0), abs=1e-12)
    assert g["gap_supremacy"] == pytest.approx(g["rt_lam_home"] - g["rt_lam_away"] - 1.0, abs=1e-12)
    assert g["gap_total"] == pytest.approx(g["rt_total"] - 2.8, abs=1e-12)
    p = gr.chances(np.array([1.9]), np.array([0.9]))
    logit = lambda x: np.log(x / (1 - x))                                 # noqa: E731
    assert g["gap_logit_home"] == pytest.approx(logit(g["rt_p_home"]) - logit(float(p["p_home"][0])), abs=1e-9)
    assert g["gap_logit_over"] == pytest.approx(logit(g["rt_p_over"]) - logit(float(p["p_over"][0])), abs=1e-9)


def test_a_checkpoint_a_day_out_does_not_know_last_nights_close_and_the_morning_one_does(world):
    matches, team, gap = world
    y = gap[gap["event_id"] == 901].set_index("cp_min")                   # side 1 played at 20:00 the evening before
    assert _ms(y.loc[1440, "as_of"]) == D0 + 10 * DAY and y.loc[1440, "n_home"] == 6
    for cp in (360, 120, 60, 15, 0):
        assert _ms(y.loc[cp, "as_of"]) == D0 + 11 * DAY and y.loc[cp, "n_home"] == 7
    assert y.loc[1440, "n_away"] == y.loc[360, "n_away"] == 6
    # the T-1440 row is the one a warehouse without last night's match gives; the later rows are not
    _, gap0 = _build([m for m in matches if m["event_id"] != 900])
    y0 = gap0[gap0["event_id"] == 901].set_index("cp_min")
    assert y.loc[1440].equals(y0.loc[1440])
    assert y.loc[360, "rt_log_ratio"] != y0.loc[360, "rt_log_ratio"]
    # the rating after a match is the one before the next where the next is the day after
    a = team[(team["team_id"] == 1) & (team["event_id"] == 900)].iloc[0]
    b = team[(team["team_id"] == 1) & (team["event_id"] == 901)].iloc[0]
    assert a["after_as_of"] == b["as_of"] and (a["strength_after"], a["tempo_after"]) == (b["strength"], b["tempo"])
    assert (a["n_after"], b["n_prior"], b["days_since"]) == (7, 7, pytest.approx((19 * HOUR - MIN) / DAY))


def test_a_close_is_known_at_the_later_of_the_scheduled_and_the_actual_kick_off():
    # the evening match is scheduled for 23:30 and goes in play 40 minutes late: its close is tomorrow's news
    matches = [m for m in _world() if m["event_id"] != 900]
    late = _match(900, 3, 1, D0 + 10 * DAY + 23 * HOUR + 30 * MIN, delay=40 * MIN)
    _, gap = _build(matches + [late])
    y = gap[gap["event_id"] == 901].set_index("cp_min")
    assert y.loc[360, "n_home"] == 6                                      # known at 00:10 on day 11: after its start
    _, gap = _build(matches + [dict(late, kickoff=D0 + 11 * DAY + 8 * HOUR, began=D0 + 10 * DAY + 20 * HOUR)])
    y = gap[gap["event_id"] == 901].set_index("cp_min")                   # went in play before its scheduled time
    assert y.loc[360, "n_home"] == 6 and y.loc[15, "n_home"] == 6         # as of day 11's start: not yet


def test_changing_a_close_changes_no_row_whose_checkpoint_is_before_it(world):
    """The leak test: every column of a gold/rating_gap row is known at its checkpoint."""
    matches, team, gap = world
    changed_later = 0
    for k in range(0, len(matches), 3):
        m = matches[k]
        if not m["in_play"]:
            continue
        lh, la = m["lam"]
        # the close alone moves: the prices at the early checkpoints stay as they were
        other = [dict(x, lam=(lh * 1.6, la * 0.7), early=x["early"] or x["lam"]) if x is m else x for x in matches]
        team2, gap2 = _build(other)
        closed = max(m["kickoff"], m["began"])
        assert gap2[["event_id", "cp_min"]].equals(gap[["event_id", "cp_min"]])
        before = _ms(gap["cp_time"]) < closed
        assert before.any()
        assert gap2[before].equals(gap[before])                           # all columns, bit for bit
        own = ((gap["event_id"] == m["event_id"]) & (gap["cp_min"] > 0)).to_numpy()
        assert own.sum() == 5 and gap2[own].equals(gap[own])              # its own early rows do not hold its close
        numeric = [c for c in gap.columns if c.startswith(("rt_", "n_"))]
        changed_later += int((~np.isclose(gap2.loc[~before, numeric], gap.loc[~before, numeric], rtol=0,
                                          atol=1e-12, equal_nan=True)).any(axis=1).sum())
        # gold/team_rating: the rating before a match that kicked off before the close is unchanged
        pre = ["n_prior", "days_since", "att", "def", "strength", "tempo", "att_sd", "def_sd", "strength_sd",
               "tempo_sd", "as_of"]
        earlier = _ms(team["kickoff"]) <= closed
        assert team2.loc[earlier, pre].equals(team.loc[earlier, pre])
    assert changed_later > 50                                             # the test has power: later rows do move


def test_a_match_that_never_went_in_play_is_listed_but_rates_nobody(world):
    matches, team, gap = world
    void = _match(960, 2, 3, D0 + 9 * DAY + 12 * HOUR, lam=(5.0, 0.2), in_play=False)
    team2, gap2 = _build(matches + [void])
    rows = team2[team2["event_id"] == 960]
    assert len(rows) == 2 and not rows["rated"].any()
    assert (rows["n_after"] == rows["n_prior"]).all()
    keep = team2["event_id"] != 960
    cols = ["strength", "tempo", "n_prior", "strength_after", "tempo_after", "n_after"]
    assert team2.loc[keep, cols].reset_index(drop=True).equals(team[cols])
    assert len(gap2[gap2["event_id"] == 960]) == 5                        # its early checkpoints; it has no close
    assert gap2[gap2["event_id"] != 960].reset_index(drop=True).equals(gap)


def test_a_close_whose_books_are_not_one_markets_prices_rates_nobody(world):
    # in-running prices taken for a close (the turn in play came late): absurd rates, and no rho gold/implied can give
    matches, team, gap = world
    wild = _match(961, 2, 3, D0 + 9 * DAY + 12 * HOUR, lam=(9.0, 2.2), rho=None)
    team2, _ = _build(matches + [wild])
    rows = team2[team2["event_id"] == 961]
    assert len(rows) == 2 and not rows["rated"].any() and (rows["n_after"] == rows["n_prior"]).all()
    keep = team2["event_id"] != 961
    cols = ["strength", "tempo", "n_prior", "strength_after", "tempo_after", "n_after"]
    assert team2.loc[keep, cols].reset_index(drop=True).equals(team[cols])


def test_the_build_does_not_depend_on_row_order_and_empty_inputs_give_empty_tables(world):
    matches, team, gap = world
    rng = np.random.default_rng(3)
    inp = _inputs(matches)
    shuffled = {k: t.take(pa.array(rng.permutation(t.num_rows))) for k, t in inp.items()}
    out = gr.build(shuffled)
    t2 = conform(gr.TEAM_RATING, out["gold/team_rating"]).to_pandas()
    g2 = conform(gr.RATING_GAP, out["gold/rating_gap"]).to_pandas()
    assert t2.sort_values(["team_id", "kickoff", "event_id"]).reset_index(drop=True).equals(team)
    assert g2.sort_values(["event_id", "cp_min"]).reset_index(drop=True).equals(gap)
    empty = gr.build({k: t.slice(0, 0) for k, t in inp.items()})
    assert [empty[s.name].num_rows for s in gr.SPECS] == [0, 0]
    assert all(empty[s.name].schema.names == s.names for s in gr.SPECS)


def test_no_feature_column_of_the_gap_table_is_named_like_an_id_by_mistake():
    """The closing-price model reads every numeric column whose name does not end in _id: the ids must end in it,
    and nothing else may."""
    ids = [c.name for c in gr.RATING_GAP.columns if c.name.endswith("_id")]
    assert ids == ["event_id", "home_id", "away_id"]
    assert all(c.doc and len(c.doc) > 10 for s in gr.SPECS for c in s.columns)
