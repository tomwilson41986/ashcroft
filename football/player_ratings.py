"""Player ratings from API-Football's line-ups (2020 on), and the line-up surprise they price.

Two ratings, each refitted monthly on the matches before the month only (lag-safe):

- **Market plus-minus (MPM):** what the closing price says a player is worth. A ridge regression of each match's
  closing implied log supremacy (log of the home goal rate over the away, from the de-vigged closing 1X2 and O/U 2.5:
  ``football.team_features.implied_rates``) on the starting elevens (+1 a home starter, -1 an away starter) and a home
  advantage a competition, weighted by exp(-xi days). A rating is a player's share of his side's supremacy in log
  goal-rate units, against the average starter of his pool (supremacy compares two sides of one league, so league
  levels cancel).
- **Goals plus-minus (GPM, "goals added"):** what happened while a player was on the pitch. Each match is cut into
  stints at every substitution and red card; a stint's goal difference is regressed on the players on the pitch (+1 home,
  -1 away) times the stint's share of 90 minutes, with a home advantage, heavily ridge-shrunk. A rating is goal
  difference per 90 minutes against the average player; ``points_per_goal`` (fitted on the season tables) turns goals
  added over a player's minutes into points added.

The line-up surprise (``surprise``): for each side and match, the confirmed eleven's summed rating less the expected
eleven's, the expected being each player's share of the side's starts in its previous five matches (so it sums to about
eleven) times his rating, both under the same month's fit. The match's surprise is home minus away. Ratings of players
the fit has never seen are zero (the average).

The evaluation (``evaluate``) asks two questions of the surprise, walk-forward by season, the holdout read once:

1. **How much of the open-to-close move is team news?** Out-of-sample R2 of the move (closing implied log supremacy less
   the open's) on the surprise, against the permuted floor and against the team features of
   ``football.feature_eval``.
2. **Does the close price the line-up fully?** The stage-2 softmax on the de-vigged close (not the open) with the
   surprise added: a log-loss gain over the close alone, its interval above zero, would mean the market under-reacts to
   team news even at the off.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

XI = 0.0023
WINDOW_DAYS = 1100
RIDGE_MPM = 10.0
RIDGE_GPM = 40.0
EXPECTED_FROM = 5


# ---------------------------------------------------------------------------
# linking and tidying
# ---------------------------------------------------------------------------

def link_matches(hm: pd.DataFrame, gold: pd.DataFrame, aliases=None) -> pd.DataFrame:
    """API-Football's matches with the canonical match_id (football.data.link_other: competition, UK day, both names)."""
    from football.data import link_other
    from football.reference import COMPETITIONS
    comp = {c.apif_league_id: c.competition_id for c in COMPETITIONS if c.apif_league_id}
    o = hm.copy()
    o["competition_id"] = o["apif_league_id"].map(comp)
    o["match_date"] = pd.to_datetime(o["kickoff_utc"], utc=True).dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
    return link_other(gold, o, "api_football", aliases)


def starters(hp: pd.DataFrame) -> pd.DataFrame:
    s = hp[(hp["role"] == "starter") & hp["apif_player_id"].notna()]
    return s.drop_duplicates(["apif_fixture_id", "apif_player_id"])[["apif_fixture_id", "side", "apif_team_id",
                                                                     "apif_player_id"]]


# ---------------------------------------------------------------------------
# stints
# ---------------------------------------------------------------------------

def stints(hp: pd.DataFrame, he: pd.DataFrame, full: int = 90) -> pd.DataFrame:
    """Each match cut at every substitution and red card: (fixture, start, end, home players, away players, home goals,
    away goals). Who went off is read from who was on the pitch, as ``sources.api_football_history`` does."""
    players = hp[hp["apif_player_id"].notna()].drop_duplicates(["apif_fixture_id", "apif_player_id"])
    side_of = {(r.apif_fixture_id, r.apif_player_id): r.side for r in players.itertuples(index=False)}
    team_side = {}
    for r in players.drop_duplicates(["apif_fixture_id", "apif_team_id"]).itertuples(index=False):
        team_side[(r.apif_fixture_id, r.apif_team_id)] = r.side
    xi = {k: set(g["apif_player_id"]) for k, g in players[players["role"] == "starter"].groupby("apif_fixture_id")}
    ev = he.sort_values(["apif_fixture_id", "minute"])
    out = []
    for fid, g in ev.groupby("apif_fixture_id", sort=False):
        if fid not in xi:
            continue
        on = set(xi[fid])
        t0, gh, ga = 0, 0, 0
        cuts = []
        last = full
        for e in g.itertuples(index=False):
            # real minutes, stoppage time included (a cap at 90 lost goals scored after a late substitution)
            m = int(e.minute) if e.minute == e.minute else 0
            last = max(last, m)
            if e.type == "Goal" and "missed" not in str(e.detail).lower():
                # API-Football lists every goal, an own goal too, under the side it counts for (checked on 2 Nov 2026:
                # crediting the listed side reproduces the final score in 562 of 562 matches with an own goal)
                s = team_side.get((fid, e.apif_team_id))
                if s is None:
                    continue
                gh, ga = gh + (s == "home"), ga + (s != "home")
                continue
            change = None
            if e.type == "subst":
                a, b = e.apif_player_id, e.apif_assist_id
                if a in on and b not in on:
                    change = (a, b)
                elif b in on and a not in on:
                    change = (b, a)
            elif e.type == "Card" and any(w in str(e.detail).lower() for w in ("red", "second yellow")) \
                    and e.apif_player_id in on:
                change = (e.apif_player_id, None)
            if change is None or m <= t0:
                if change is not None:                     # a change at the same minute: apply without a stint
                    on.discard(change[0])
                    if change[1] is not None:
                        on.add(change[1])
                continue
            cuts.append((t0, m, frozenset(on), gh, ga))
            gh = ga = 0
            t0 = m
            on.discard(change[0])
            if change[1] is not None:
                on.add(change[1])
        cuts.append((t0, max(last, t0 + 1) if (gh or ga) and last <= t0 else last, frozenset(on), gh, ga))
        for a, b, pl, h, w in cuts:
            if b <= a:
                continue
            home = [p for p in pl if side_of.get((fid, p)) == "home"]
            away = [p for p in pl if side_of.get((fid, p)) == "away"]
            out.append({"apif_fixture_id": fid, "start": a, "end": b, "home": home, "away": away,
                        "goals_home": h, "goals_away": w})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# the fits
# ---------------------------------------------------------------------------

def _ridge(rows, cols, vals, n_rows, n_cols, y, w, ridge):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    sw = np.sqrt(w)
    X = coo_matrix((vals * sw[rows], (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    return lsqr(X, y * sw, damp=np.sqrt(ridge))[0]


def fit_mpm(m: pd.DataFrame, st: pd.DataFrame, as_of: pd.Timestamp, xi: float = XI, window_days: int = WINDOW_DAYS,
            ridge: float = RIDGE_MPM, min_matches: int = 500) -> dict:
    """Market plus-minus from the matches before ``as_of``: {player id: rating}."""
    w = m[(m["date"] < as_of) & (m["date"] >= as_of - pd.Timedelta(days=window_days)) & m["sup_close"].notna()]
    if len(w) < min_matches:
        return {}
    s = st[st["apif_fixture_id"].isin(w["apif_fixture_id"])]
    row_of = pd.Series(np.arange(len(w)), index=w["apif_fixture_id"].to_numpy())
    comps = pd.Index(w["competition_id"].unique())
    players = pd.Index(s["apif_player_id"].unique())
    r_pl = row_of.reindex(s["apif_fixture_id"]).to_numpy()
    c_pl = len(comps) + players.get_indexer(s["apif_player_id"])
    v_pl = np.where(s["side"].to_numpy() == "home", 1.0, -1.0)
    rows = np.r_[np.arange(len(w)), r_pl]
    cols = np.r_[comps.get_indexer(w["competition_id"]), c_pl]
    vals = np.r_[np.ones(len(w)), v_pl]
    wt = np.exp(-xi * (as_of - w["date"]).dt.days.to_numpy(float))
    beta = _ridge(rows, cols, vals, len(w), len(comps) + len(players), w["sup_close"].to_numpy(float), wt, ridge)
    return dict(zip(players, beta[len(comps):]))


def fit_gpm(sti: pd.DataFrame, as_of: pd.Timestamp, xi: float = XI, window_days: int = WINDOW_DAYS,
            ridge: float = RIDGE_GPM) -> dict:
    """Goals plus-minus from the stints of matches before ``as_of``: {player id: goal difference per 90 added}."""
    w = sti[(sti["date"] < as_of) & (sti["date"] >= as_of - pd.Timedelta(days=window_days))]
    if len(w) < 2000:
        return {}
    dur = ((w["end"] - w["start"]) / 90.0).to_numpy(float)
    home_l, away_l = w["home"].to_numpy(), w["away"].to_numpy()
    allp = pd.Index(pd.unique(np.concatenate([np.concatenate(home_l), np.concatenate(away_l)])))
    r, c, v = [np.arange(len(w))], [np.zeros(len(w), int)], [dur]
    for sign, lists in ((1.0, home_l), (-1.0, away_l)):
        lens = np.array([len(x) for x in lists])
        r.append(np.repeat(np.arange(len(w)), lens))
        c.append(1 + allp.get_indexer(np.concatenate(lists)))
        v.append(np.repeat(sign * dur, lens))
    y = (w["goals_home"] - w["goals_away"]).to_numpy(float)
    wt = np.exp(-xi * (as_of - w["date"]).dt.days.to_numpy(float))
    beta = _ridge(np.concatenate(r), np.concatenate(c), np.concatenate(v), len(w), 1 + len(allp), y, wt, ridge)
    return dict(zip(allp, beta[1:]))


def monthly(m: pd.DataFrame, fit_fn, start: str, **kw) -> dict:
    """{month 'YYYY-MM': ratings fitted on the matches before the month's first day}."""
    out = {}
    for mo in pd.period_range(start, m["date"].max(), freq="M"):
        out[str(mo)] = fit_fn(as_of=mo.start_time, **kw)
        log.info("ratings %s: %d players", mo, len(out[str(mo)]))
    return out


# ---------------------------------------------------------------------------
# the surprise
# ---------------------------------------------------------------------------

def surprise(m: pd.DataFrame, st: pd.DataFrame, ratings: dict, name: str, n_prev: int = EXPECTED_FROM) -> pd.DataFrame:
    """Each match's confirmed and expected eleven ratings, home and away, under its month's fit, and the surprise."""
    from football.team_features import collection_day
    s = st.merge(m[["apif_fixture_id", "date"]], on="apif_fixture_id")
    # the fit of the month the match's early price was collected in (a fit made on the 1st holds the 30th's and
    # 31st's closes, after a Friday-30th collection for a Sunday-1st match: QA, 2 Nov 2026)
    s["month"] = collection_day(s["date"]).dt.strftime("%Y-%m").to_numpy()
    s["r"] = [ratings.get(mo, {}).get(p, 0.0) for mo, p in zip(s["month"], s["apif_player_id"])]
    xi_r = s.groupby(["apif_fixture_id", "side"])["r"].sum().unstack()
    # the expected eleven: each player's share of his side's starts in its previous n matches (earlier days only)
    team_matches = s[["apif_fixture_id", "apif_team_id", "date"]].drop_duplicates().sort_values(["apif_team_id", "date"])
    team_matches["k"] = team_matches.groupby("apif_team_id").cumcount()
    s = s.merge(team_matches[["apif_fixture_id", "apif_team_id", "k"]], on=["apif_fixture_id", "apif_team_id"])
    exp_rows = []
    by_team = {t: g for t, g in s.groupby("apif_team_id")}
    for tm in team_matches.itertuples(index=False):
        g = by_team[tm.apif_team_id]
        prev = g[(g["k"] < tm.k) & (g["k"] >= tm.k - n_prev) & (g["date"] < tm.date)]
        if prev["apif_fixture_id"].nunique() < 3:
            continue
        share = prev.groupby("apif_player_id").size() / prev["apif_fixture_id"].nunique()
        mo = collection_day([tm.date]).iloc[0].strftime("%Y-%m")
        rr = ratings.get(mo, {})
        exp_rows.append({"apif_fixture_id": tm.apif_fixture_id, "apif_team_id": tm.apif_team_id,
                         "expected": float(sum(rr.get(p, 0.0) * w for p, w in share.items()))})
    e = pd.DataFrame(exp_rows)
    side = s.drop_duplicates(["apif_fixture_id", "apif_team_id"])[["apif_fixture_id", "apif_team_id", "side"]]
    e = e.merge(side, on=["apif_fixture_id", "apif_team_id"]).pivot(index="apif_fixture_id", columns="side",
                                                                       values="expected")
    out = pd.DataFrame({f"{name}_xi_home": xi_r.get("home"), f"{name}_xi_away": xi_r.get("away"),
                        f"{name}_exp_home": e.get("home"), f"{name}_exp_away": e.get("away")})
    out[f"{name}_surprise"] = (out[f"{name}_xi_home"] - out[f"{name}_exp_home"]) - \
        (out[f"{name}_xi_away"] - out[f"{name}_exp_away"])
    out[f"{name}_xi_diff"] = out[f"{name}_xi_home"] - out[f"{name}_xi_away"]
    return out.reset_index().rename(columns={"index": "apif_fixture_id"})


TEAM_NEWS = ["tn_mpm_exp_diff", "tn_gpm_exp_diff", "tn_absent_mpm_diff", "tn_absent_gpm_diff", "tn_red_mpm_diff",
             "tn_absent_n_diff", "tn_rotation_diff", "tn_mpm_exp_home", "tn_mpm_exp_away"]
REGULAR_SHARE = 0.6


def team_news(m: pd.DataFrame, hp: pd.DataFrame, he: pd.DataFrame, mpm: dict, gpm: dict,
              n_prev: int = EXPECTED_FROM) -> pd.DataFrame:
    """What was known of each side's eleven at the match's early price (Friday 17:00 / Tuesday 13:00 UK), from its
    matches before the collection day only (the owner, 4 Nov 2026: team news and player-based strength; Ted Knutson
    checks injuries, line-ups and suspensions before every bet):

    - ``exp``: the expected eleven's rating (each player's share of the side's starts in its previous ``n_prev``
      matches, times his rating under the collection month's fit), MPM and GPM;
    - ``absent``: the regulars (started at least 60% of those matches) who did not play in the side's latest match
      (injured, suspended or dropped), their share-weighted rating summed, and their count;
    - ``red``: the rating of players sent off in the side's latest match (suspended next);
    - ``rotation``: the share of the latest match's starters who did not start the one before.

    One row a fixture: home less away of each (``tn_*_diff``), and the expected MPM eleven home and away."""
    from football.team_features import collection_day
    st = hp[hp["apif_player_id"].notna()].drop_duplicates(["apif_fixture_id", "apif_player_id"])
    st = st.merge(m[["apif_fixture_id", "date"]], on="apif_fixture_id")
    st["played"] = (st["role"] == "starter") | (st["minutes_events"].fillna(st["minutes_stats"]).fillna(0) > 0)
    reds = he[(he["type"] == "Card") & (he["detail"] == "Red Card") & he["apif_player_id"].notna()]
    red_set = set(zip(reds["apif_fixture_id"], reds["apif_player_id"]))
    fx = st[["apif_fixture_id", "apif_team_id", "side", "date"]].drop_duplicates(["apif_fixture_id", "apif_team_id"])
    fx = fx.sort_values(["apif_team_id", "date"])
    fx["coll"] = collection_day(fx["date"]).to_numpy()
    by_team = {t: g for t, g in st.groupby("apif_team_id")}
    rows = []
    for t, tm in fx.groupby("apif_team_id"):
        g = by_team[t]
        dates_fx = g[["apif_fixture_id", "date"]].drop_duplicates().sort_values("date")
        for r in tm.itertuples(index=False):
            prev_fx = dates_fx[dates_fx["date"] < pd.Timestamp(r.coll)].tail(n_prev)
            if len(prev_fx) < 3:
                continue
            prev = g[g["apif_fixture_id"].isin(prev_fx["apif_fixture_id"])]
            starts = prev[prev["role"] == "starter"]
            share = starts.groupby("apif_player_id").size() / len(prev_fx)
            mo = pd.Timestamp(r.coll).strftime("%Y-%m")
            rm, rg = mpm.get(mo, {}), gpm.get(mo, {})
            last_id = prev_fx["apif_fixture_id"].iloc[-1]
            last = prev[prev["apif_fixture_id"] == last_id]
            played_last = set(last.loc[last["played"], "apif_player_id"])
            regulars = share[share >= REGULAR_SHARE]
            absent = [p for p in regulars.index if p not in played_last]
            sent_off = [p for p in last["apif_player_id"] if (last_id, p) in red_set]
            xi_last = set(last.loc[last["role"] == "starter", "apif_player_id"])
            if len(prev_fx) >= 2:
                before_id = prev_fx["apif_fixture_id"].iloc[-2]
                xi_before = set(prev.loc[(prev["apif_fixture_id"] == before_id) & (prev["role"] == "starter"),
                                         "apif_player_id"])
                rotation = len(xi_last - xi_before) / max(len(xi_last), 1)
            else:
                rotation = np.nan
            rows.append({"apif_fixture_id": r.apif_fixture_id, "side": r.side,
                         "mpm_exp": float(sum(rm.get(p, 0.0) * w for p, w in share.items())),
                         "gpm_exp": float(sum(rg.get(p, 0.0) * w for p, w in share.items())),
                         "absent_mpm": float(sum(rm.get(p, 0.0) * regulars[p] for p in absent)),
                         "absent_gpm": float(sum(rg.get(p, 0.0) * regulars[p] for p in absent)),
                         "absent_n": float(len(absent)),
                         "red_mpm": float(sum(rm.get(p, 0.0) for p in sent_off)),
                         "rotation": rotation})
    e = pd.DataFrame(rows)
    if e.empty:
        return pd.DataFrame(columns=["apif_fixture_id"] + TEAM_NEWS)
    w = e.pivot_table(index="apif_fixture_id", columns="side", aggfunc="first")
    out = pd.DataFrame(index=w.index)
    for k in ("mpm_exp", "gpm_exp", "absent_mpm", "absent_gpm", "red_mpm", "absent_n", "rotation"):
        h = w[(k, "home")] if (k, "home") in w.columns else np.nan
        a_ = w[(k, "away")] if (k, "away") in w.columns else np.nan
        out[f"tn_{k}_diff"] = h - a_
    out["tn_mpm_exp_home"] = w[("mpm_exp", "home")] if ("mpm_exp", "home") in w.columns else np.nan
    out["tn_mpm_exp_away"] = w[("mpm_exp", "away")] if ("mpm_exp", "away") in w.columns else np.nan
    return out.reset_index()[["apif_fixture_id"] + TEAM_NEWS]


def points_per_goal(gold: pd.DataFrame) -> float:
    """The slope of a side's points a match on its goal difference a match, over its season (the 22 divisions)."""
    g = gold[(gold["status"] == "FINISHED") & gold["ft_home"].notna()]
    d = g["ft_home"] - g["ft_away"]
    rows = pd.concat([pd.DataFrame({"t": g["home_team_id"], "s": g["season"], "gd": d,
                                    "pts": np.select([d > 0, d == 0], [3, 1], 0)}),
                      pd.DataFrame({"t": g["away_team_id"], "s": g["season"], "gd": -d,
                                    "pts": np.select([d < 0, d == 0], [3, 1], 0)})])
    a = rows.groupby(["t", "s"]).agg(gd=("gd", "mean"), pts=("pts", "mean"), n=("gd", "size"))
    a = a[a["n"] >= 20]
    return float(np.polyfit(a["gd"], a["pts"], 1)[0])


# ---------------------------------------------------------------------------
# the evaluation
# ---------------------------------------------------------------------------

DEV_FROM, DEV_TO, HOLDOUT_FROM, TRAIN_FROM = 2021, 2024, 2025, 2020
RATINGS_FROM = "2021-07"


def build_all(gold: pd.DataFrame, feats: pd.DataFrame, hm: pd.DataFrame, hp: pd.DataFrame, he: pd.DataFrame,
              aliases=None) -> tuple[pd.DataFrame, dict, dict]:
    """(the team-feature table with the player columns joined, the MPM fits, the GPM fits)."""
    from football.team_features import implied_rates
    linked = link_matches(hm, gold, aliases)
    log.info("linked %d of %d API-Football matches", len(linked), len(hm))
    m = linked[["apif_fixture_id", "match_id", "competition_id"]].merge(
        feats[["match_id", "date", "qc_h", "qc_a", "qc_o"]], on="match_id")
    lh, la = implied_rates(m["qc_h"], m["qc_a"], m["qc_o"])
    m["sup_close"] = np.log(lh / la)
    m["date"] = pd.to_datetime(m["date"])
    st = starters(hp)
    mpm = monthly(m, lambda as_of: fit_mpm(m, st, as_of), RATINGS_FROM)
    sti = stints(hp, he).merge(m[["apif_fixture_id", "date"]], on="apif_fixture_id")
    rec = sti.groupby("apif_fixture_id")[["goals_home", "goals_away"]].sum().join(
        hm.set_index("apif_fixture_id")[["ft_home", "ft_away"]])
    reconciled = float(((rec["goals_home"] == rec["ft_home"]) & (rec["goals_away"] == rec["ft_away"])).mean())
    log.info("stint goals reconcile with the final score in %.2f%% of %d matches", 100 * reconciled, len(rec))
    gpm = monthly(m, lambda as_of: fit_gpm(sti, as_of), RATINGS_FROM)
    sm, sg = surprise(m, st, mpm, "mpm"), surprise(m, st, gpm, "gpm")
    tn = team_news(m, hp, he, mpm, gpm)
    out = feats.merge(m[["match_id", "apif_fixture_id"]], on="match_id").merge(sm, on="apif_fixture_id", how="left") \
        .merge(sg, on="apif_fixture_id", how="left").merge(tn, on="apif_fixture_id", how="left")
    out.attrs["stints"] = int(len(sti))
    out.attrs["stint_goals_reconciled"] = reconciled
    out.attrs["linked"] = int(len(linked))
    return out, mpm, gpm


def evaluate(f: pd.DataFrame, scored: str = "dev") -> dict:
    from football.feature_eval import FAMILIES, move_test, outcome_test, prepare
    d = prepare(f)
    d = d[d["season"] >= TRAIN_FROM]
    kw = {"train_from": TRAIN_FROM, "dev_from": DEV_FROM, "dev_to": DEV_TO, "holdout_from": HOLDOUT_FROM}
    fam = {"mpm_surprise": ["mpm_surprise"], "gpm_surprise": ["gpm_surprise"],
           "mpm_xi": ["mpm_xi_diff"], "gpm_xi": ["gpm_xi_diff"],
           "team_all": FAMILIES["all"], "team_all+surprise": FAMILIES["all"] + ["mpm_surprise", "gpm_surprise"]}
    res = {"rows": int(len(d)), "with_surprise": int(d["mpm_surprise"].notna().sum()),
           "move": move_test(d, fam, scored=scored, **kw),
           "outcome_vs_open": outcome_test(d, {k: v for k, v in fam.items() if not k.startswith("team")},
                                           scored=scored, **kw)}
    c = d.copy()
    for k in ("h", "d", "a"):
        c[f"lq_{k}"] = np.log(np.clip(c[f"qc_{k}"], 1e-6, 1))
    c = c[c[["lq_h", "lq_d", "lq_a"]].notna().all(axis=1)]
    res["outcome_vs_close"] = outcome_test(c, {k: v for k, v in fam.items() if not k.startswith("team")},
                                           scored=scored, **kw)
    return res


def top_players(ratings: dict, hp: pd.DataFrame, month: str, n: int = 15) -> list[dict]:
    r = ratings.get(month, {})
    names = hp.dropna(subset=["apif_player_id"]).drop_duplicates("apif_player_id").set_index("apif_player_id")["player"]
    best = sorted(r.items(), key=lambda kv: -kv[1])[:n]
    return [{"player": names.get(p), "apif_player_id": int(p), "rating": float(v)} for p, v in best]


def main(argv=None) -> int:
    import argparse
    import json
    from pathlib import Path

    from football.data import load_gold
    from football.reference import load_aliases
    from sources.common import Store
    ap = argparse.ArgumentParser(description="Player ratings and the line-up surprise against the market")
    ap.add_argument("--root", default=None, help="local sources folder holding the gold table")
    ap.add_argument("--history-root", default=None, help="local sources folder holding api_football/history_*")
    ap.add_argument("--features", required=True, help="team features parquet (football.feature_eval --features-out)")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--table-out", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    gs, hs = Store(root=a.root), Store(root=a.history_root or a.root)
    gold = load_gold(gs)
    hm, hp, he = (hs.get_parquet(f"api_football/history_{x}.parquet") for x in ("matches", "players", "events"))
    f, mpm, gpm = build_all(gold, pd.read_parquet(a.features), hm, hp, he,
                            load_aliases(extra=gs.get_parquet("football/gold/team_aliases.parquet")))
    if a.table_out:
        f.drop(columns=[c for c in f.columns if f[c].dtype == object and c not in ("match_id",)], errors="ignore") \
            .to_parquet(a.table_out, index=False)
    last = max(k for k, v in mpm.items() if v)
    res = {"history_matches": int(len(hm)), "linked": f.attrs.get("linked"), "stints": f.attrs.get("stints"),
           "stint_goals_reconciled": f.attrs.get("stint_goals_reconciled"),
           "points_per_goal": points_per_goal(gold), "ratings_month_shown": last,
           "players_rated": {"mpm": len(mpm[last]), "gpm": len(gpm.get(last, {}))},
           "top_mpm": top_players(mpm, hp, last), "top_gpm": top_players(gpm, hp, last),
           "dev": evaluate(f, "dev")}
    if a.holdout:
        res["holdout"] = evaluate(f, "holdout")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def default(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x) if np.isfinite(x) else None
        return str(x)
    Path(a.out).write_text(json.dumps(res, indent=1, default=default) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "dev"}, indent=1, default=default)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
