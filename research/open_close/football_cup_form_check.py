"""Cup and European ties with each club's recent form from the feed: does it add to cross-league strength?

racing² football research, 6 Nov 2026. The feed (API-Football, a regular daily feed) now holds team statistics for
every league our Betfair matches link into (``sources/api_football_leagues.json``). For each club, before each
match: the mean over its last ``N_FORM`` matches (any competition) of

- shots on target for less against (``sotd``), shots for less against (``shotd``),
- expected goals for less against (``xgd``) where the feed gives xG (else 0, with ``has_xg`` 0).

Fixed before the run (as ``football_cross_league_price``):

- the groups: cup ties across divisions, continental ties across countries, league matches (the control);
- a multinomial logistic of the result per group, fitted on matches from ``FIT_FROM`` to ``FIT_END`` (the 2025/26
  season), tested on the Historic Data bundle's matches from ``FIT_END`` on (Aug-Sep 2026, out of sample);
- the baseline: v2 rating and competition record (``diff_v2``, ``diff_league_only``); the challenger adds the form;
- both targets for the rule 'back a side whose expected value at the early price is above 2%': closed at the close,
  and held to the result; and the result's log loss.

    python -m research.open_close.football_cup_form_check --out out/cup_form_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_FORM = 10
FIT_FROM = pd.Timestamp("2025-08-01", tz="UTC")
FIT_END = pd.Timestamp("2026-08-01", tz="UTC")
BASE = ["diff_v2", "diff_league_only"]
FORM = ["sotd", "shotd", "xgd", "has_xg"]
GROUPS = ("cup_ties_across_divisions", "continental_ties_across_countries", "league_matches")


def team_form(stats: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Each club's form before each of its matches: means of its last N_FORM matches with statistics (shifted, so
    the match itself never counts). One row a (fixture, team)."""
    st = stats.merge(fx[["apif_fixture_id", "kickoff_utc"]], on="apif_fixture_id", how="inner")
    opp = st[["apif_fixture_id", "apif_team_id", "shots_on", "shots", "xg"]].rename(
        columns={"apif_team_id": "opp_id", "shots_on": "o_on", "shots": "o_sh", "xg": "o_xg"})
    st = st.merge(opp, on="apif_fixture_id")
    st = st[st["apif_team_id"] != st["opp_id"]].copy()
    st["kick"] = pd.to_datetime(st["kickoff_utc"], utc=True, errors="coerce")
    st = st.dropna(subset=["kick"]).sort_values(["apif_team_id", "kick"])
    st["sotd_m"] = st["shots_on"] - st["o_on"]
    st["shotd_m"] = st["shots"] - st["o_sh"]
    st["xgd_m"] = st["xg"] - st["o_xg"]
    g = st.groupby("apif_team_id")
    out = st[["apif_fixture_id", "apif_team_id"]].copy()
    for c in ("sotd", "shotd", "xgd"):
        out[c] = g[f"{c}_m"].transform(lambda s: s.shift(1).rolling(N_FORM, min_periods=3).mean())
    out["n_prior"] = g.cumcount()
    return out


def add_form(d: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    """Home less away form on each rated match; NaN-free (missing form -> 0, has_xg 0/1)."""
    f = form.set_index(["apif_fixture_id", "apif_team_id"])
    out = d.copy()
    for side in ("home", "away"):
        k = pd.MultiIndex.from_arrays([out["apif_fixture_id"], out[f"apif_{side}_id"]])
        got = f.reindex(k)
        for c in ("sotd", "shotd", "xgd"):
            out[f"{c}_{side[0]}"] = got[c].to_numpy()
    out["has_xg"] = (out["xgd_h"].notna() & out["xgd_a"].notna()).astype(float)
    for c in ("sotd", "shotd", "xgd"):
        out[c] = (out[f"{c}_h"] - out[f"{c}_a"]).fillna(0.0)
    out["has_form"] = out["sotd_h"].notna() & out["sotd_a"].notna()
    return out


def probs(d: pd.DataFrame, p: pd.DataFrame, cols: list[str], g: str) -> np.ndarray | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    kick = pd.to_datetime(d["kickoff"], utc=True)
    tr = d[(d["group"] == g) & (kick >= FIT_FROM) & (kick < FIT_END)]
    if len(tr) < 300:
        return None
    y = np.where(tr["gd"] > 0, 0, np.where(tr["gd"] == 0, 1, 2))
    sc = StandardScaler().fit(tr[cols].to_numpy(float))
    m = LogisticRegression(max_iter=1000).fit(sc.transform(tr[cols].to_numpy(float)), y)
    pr = np.zeros((len(p), 3))
    pr[:, m.classes_] = m.predict_proba(sc.transform(p[cols].to_numpy(float)))
    return pr


def run(fx: pd.DataFrame, stats: pd.DataFrame, m: pd.DataFrame) -> dict:
    from research.open_close.football_cross_league_check import rated_matches
    from research.open_close import football_cross_league_price as cp
    d, _ = rated_matches(fx)
    d = add_form(d, team_form(stats, fx))
    w = cp.wide(m)
    if w.empty:
        return {"status": "no Betfair match-odds prices found"}
    lk = cp.link(w, d)
    lk = lk[lk["result_agrees"]]
    extra = d.set_index(d.index)[FORM + ["has_form"]]
    lk = lk.join(extra, on="row")
    lk = lk[lk["kickoff"] >= FIT_END]
    kick = pd.to_datetime(d["kickoff"], utc=True)
    fit = d[(kick >= FIT_FROM) & (kick < FIT_END)]
    out = {"stat_rows": int(len(stats)), "fit_window": [str(FIT_FROM.date()), str(FIT_END.date())],
           "fit_matches_by_group": fit["group"].value_counts().to_dict(),
           "fit_share_with_form": {g: round(float(fit.loc[fit["group"] == g, "has_form"].mean()), 3)
                                   for g in GROUPS if (fit["group"] == g).any()},
           "test_linked_by_group": lk["group"].value_counts().to_dict(), "groups": {}}
    for g in GROUPS:
        p = lk[lk["group"] == g].copy()
        if p.empty:
            out["groups"][g] = {"status": "no data found"}
            continue
        res = {"test_matches": int(len(p)), "test_share_with_form": round(float(p["has_form"].mean()), 3)}
        for name, cols in (("rating_only", BASE), ("rating_plus_form", BASE + FORM)):
            pr = probs(d, p, cols, g)
            if pr is None:
                res[name] = {"status": "too few matches to fit"}
                continue
            for k, o in enumerate("hda"):
                p[f"pm_{o}"] = pr[:, k]
            res[name] = {"diagnostics": cp.diagnostics(p), "home_and_away": cp.targets(p)}
        out["groups"][g] = res
    return out


def main(argv=None) -> int:
    from sources.common import Store
    from research.open_close.football_cross_league_price import load_markets
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    fx = store.get_parquet("api_football/fixtures.parquet")
    stats = store.get_parquet("api_football/history_stats.parquet")
    if fx is None or stats is None:
        res = {"status": "no data found (api_football fixtures or history_stats)"}
    else:
        res = run(fx, stats, load_markets(store))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
