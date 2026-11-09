"""Cross-league strength against Betfair's prices: the closing price and winning the bet (the owner's two targets).

racing² football research, 6 Nov 2026. Rules fixed before any price was read:

- **Prices:** Betfair MATCH_ODDS markets from Historic Data (``betfair_historic/markets_soccer_<year>.parquet``,
  the owner's BASIC bundle, Sep 2025 on), or a local extract (``--match-odds``). Early price = the last trade 24 h
  before the scheduled off (``ltp_t_1440``), else the first trade; the close = the last trade before in play. Chances
  are the three prices' implied chances normalised to one.
- **Link:** each market to the API-Football fixture kicking off within 30 minutes of it whose two sides both match
  the event's names (``football.reference.pair_event``, unique or nothing).
- **The rating's chance:** a multinomial logistic of the result on the v2 rating difference and the competition-only
  difference (``football_cross_league_check``), fitted per group on matches before ``FIT_END`` (1 Aug 2025), so every
  priced match is out of sample. Groups: cup ties across divisions, continental ties across countries, league matches
  (the control).
- **Rule:** back a side (home or away; the draw shown apart) when the rating's expected value at the price,
  p x odds - 1, is above 2%.
- **Target 1, the closing price:** backed at the early price and closed at the close: return odds_early / odds_close
  - 1 a GBP1 stake. Shown at the last traded prices, and a tick worse at each end.
- **Target 2, winning the bet:** the same backs held to the result at the early price; and backs chosen at the
  close, against the close, held to the result.
- 2% commission on each market's net win. 95% bootstrap intervals by market.

    python -m research.open_close.football_cross_league_price --out out/cross_league_price.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIT_END = pd.Timestamp("2025-08-01", tz="UTC")
EV_MIN = 0.02
COMMISSION = 0.02
LINK_MINUTES = 30
GROUPS = ("cup_ties_across_divisions", "continental_ties_across_countries", "league_matches")


def load_markets(store, local: str | None = None) -> pd.DataFrame:
    """MATCH_ODDS runner rows from the Historic Data tables (or a local extract)."""
    if local:
        frames = [pd.read_parquet(local)]
    else:
        keys = [k for k in store.listing("betfair_historic/") if "markets_soccer" in k and k.endswith(".parquet")]
        frames = [store.get_parquet(k) for k in sorted(keys)]
        frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames, ignore_index=True)
    return m[m["market_type"] == "MATCH_ODDS"].copy()


def wide(m: pd.DataFrame) -> pd.DataFrame:
    """One row a market: home/draw/away early and closing prices and the result, from the runner rows."""
    m = m.copy()
    parts = m["event_name"].astype(str).str.split(" v ", n=1, regex=False)
    m["ev_home"], m["ev_away"] = parts.str[0].str.strip(), parts.str[1].str.strip()
    m["leg"] = np.where(m["runner_name"] == "The Draw", "d",
                        np.where(m["runner_name"] == m["ev_home"], "h",
                                 np.where(m["runner_name"] == m["ev_away"], "a", None)))
    m = m[m["leg"].notna()]
    early = m["ltp_t_1440"] if "ltp_t_1440" in m.columns else pd.Series(np.nan, index=m.index)
    m["p_early"] = pd.to_numeric(early, errors="coerce").fillna(pd.to_numeric(m["ltp_first"], errors="coerce"))
    m["p_close"] = pd.to_numeric(m["ltp_close"], errors="coerce")
    w = m.pivot_table(index="market_id", columns="leg", values=["p_early", "p_close", "won"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    meta = m.drop_duplicates("market_id").set_index("market_id")[["event_id", "ev_home", "ev_away", "market_time",
                                                                   "country"]]
    w = w.join(meta).reset_index()
    need = [f"{a}_{o}" for a in ("p_early", "p_close", "won") for o in "hda"]
    w = w.dropna(subset=[c for c in need if c in w.columns])
    if not set(need) <= set(w.columns):
        return pd.DataFrame()
    w = w[(w[[f"won_{o}" for o in "hda"]].sum(axis=1) == 1)]
    w["kickoff"] = pd.to_datetime(w["market_time"], utc=True, errors="coerce")
    for t, pre in (("p_early", "qe"), ("p_close", "qc")):
        inv = 1 / w[[f"{t}_{o}" for o in "hda"]]
        s = inv.sum(axis=1)
        for o in "hda":
            w[f"{pre}_{o}"] = (1 / w[f"{t}_{o}"]) / s
    return w


def link(w: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Each market -> the fixture kicking off within LINK_MINUTES whose two sides both match (unique or nothing)."""
    from football.reference import pair_event
    d = d.copy()
    d["kick"] = pd.to_datetime(d["kickoff"], utc=True)
    d = d[d["kick"] >= w["kickoff"].min() - pd.Timedelta(days=1)].sort_values("kick").reset_index(drop=True)
    ks = d["kick"].dt.tz_convert(None).to_numpy()
    rows = []
    tol = np.timedelta64(LINK_MINUTES, "m")
    for r in w.itertuples(index=False):
        t = np.datetime64(r.kickoff.tz_convert("UTC").tz_localize(None))
        lo, hi = np.searchsorted(ks, t - tol), np.searchsorted(ks, t + tol, side="right")
        if hi <= lo:
            continue
        cand = d.iloc[lo:hi]
        fx = [(i, h, a) for i, h, a in zip(cand.index, cand["home"].astype(str), cand["away"].astype(str))]
        j, sc = pair_event(r.ev_home, r.ev_away, fx, "", "betfair")
        if j is not None:
            rows.append({"market_id": r.market_id, "row": fx[j][0], "link_score": sc})
    if not rows:
        return pd.DataFrame()
    lk = pd.DataFrame(rows).drop_duplicates("row", keep=False)
    keep = ["group", "diff_v2", "diff_league_only", "gd", "competition"] + [
        c for c in ("apif_fixture_id",) if c in d.columns]
    out = w.merge(lk, on="market_id").join(d[keep], on="row")
    # the fixture's result must agree with Betfair's settled result
    res = np.where(out["gd"] > 0, "h", np.where(out["gd"] == 0, "d", "a"))
    bf = np.select([out["won_h"] == 1, out["won_d"] == 1], ["h", "d"], "a")
    out["result_agrees"] = res == bf
    return out


def rating_probs(d: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """The rating's home/draw/away chances for each priced match, a model per group fitted before FIT_END."""
    from sklearn.linear_model import LogisticRegression
    p = p.copy()
    for o in "hda":
        p[f"pm_{o}"] = np.nan
    cols = ["diff_v2", "diff_league_only"]
    kick = pd.to_datetime(d["kickoff"], utc=True)
    for g in GROUPS:
        tr = d[(d["group"] == g) & (kick < FIT_END)]
        te = p["group"] == g
        if len(tr) < 500 or not te.any():
            continue
        y = np.where(tr["gd"] > 0, 0, np.where(tr["gd"] == 0, 1, 2))
        m = LogisticRegression(max_iter=500).fit(tr[cols].to_numpy(float) / 100, y)
        pr = np.zeros((int(te.sum()), 3))
        pr[:, m.classes_] = m.predict_proba(p.loc[te, cols].to_numpy(float) / 100)
        for k, o in enumerate("hda"):
            p.loc[te, f"pm_{o}"] = pr[:, k]
    return p


def _book(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bets, return a GBP1 stake after commission on each market's net win, its 95% interval by market."""
    if bets.empty:
        return {"bets": 0}
    net = bets.groupby("market_id")["profit"].sum()
    n = bets.groupby("market_id").size().to_numpy()
    pnl = np.where(net > 0, net * (1 - COMMISSION), net).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), (n_boot, len(pnl)))
    roi = pnl[idx].sum(1) / n[idx].sum(1)
    out = {"bets": int(len(bets)), "markets": int(len(net)), "roi": round(float(pnl.sum() / n.sum()), 4),
           "roi_lo95": round(float(np.quantile(roi, 0.025)), 4), "roi_hi95": round(float(np.quantile(roi, 0.975)), 4)}
    if "right_way" in bets:
        out["right_way"] = round(float(bets["right_way"].mean()), 4)
    if "won" in bets:
        out["strike"] = round(float(bets["won"].mean()), 4)
    return out


def targets(p: pd.DataFrame, legs=("h", "a")) -> dict:
    """Both targets for the 2% rule on ``legs``."""
    from research.open_close.backtest import tick_over, tick_under
    early, close, held_e, held_c = [], [], [], []
    for o in legs:
        pe, pc, pm, won = (p[f"p_early_{o}"].to_numpy(float), p[f"p_close_{o}"].to_numpy(float),
                           p[f"pm_{o}"].to_numpy(float), p[f"won_{o}"].to_numpy(float))
        sel = pm * pe - 1 > EV_MIN
        base = {"market_id": p["market_id"].to_numpy()[sel], "leg": o}
        ob = np.array([tick_under(x) for x in pe[sel]])
        cl = np.array([tick_over(x) for x in pc[sel]])
        early.append(pd.DataFrame({**base, "profit": pe[sel] / pc[sel] - 1, "right_way": pc[sel] < pe[sel]}))
        close.append(pd.DataFrame({**base, "profit": ob / cl - 1, "right_way": pc[sel] < pe[sel]}))
        held_e.append(pd.DataFrame({**base, "profit": np.where(won[sel] == 1, pe[sel] - 1, -1.0), "won": won[sel]}))
        selc = pm * pc - 1 > EV_MIN
        held_c.append(pd.DataFrame({"market_id": p["market_id"].to_numpy()[selc], "leg": o,
                                    "profit": np.where(won[selc] == 1, pc[selc] - 1, -1.0), "won": won[selc]}))
    cat = lambda xs: pd.concat(xs, ignore_index=True)          # noqa: E731
    return {"closing_price_target": {"backed_early_closed_at_close_ltp": _book(cat(early)),
                                     "same_a_tick_worse_each_end": _book(cat(close))},
            "winning_the_bet_target": {"backed_early_held_to_result": _book(cat(held_e)),
                                       "backed_at_close_held_to_result": _book(cat(held_c))}}


def diagnostics(p: pd.DataFrame) -> dict:
    """Log loss of the result: the close, the early price, the rating; does the rating's gap to the early price point
    the way the price moves (the correlation of p - q_early with q_close - q_early, home and away pooled)?"""
    y = np.select([p["won_h"] == 1, p["won_d"] == 1], [0, 1], 2)
    ll = lambda pre: float(-np.mean(np.log(np.clip(p[[f"{pre}_{o}" for o in "hda"]].to_numpy(float)[  # noqa: E731
        np.arange(len(p)), y], 1e-9, 1))))
    gap = np.r_[p["pm_h"] - p["qe_h"], p["pm_a"] - p["qe_a"]]
    mv = np.r_[p["qc_h"] - p["qe_h"], p["qc_a"] - p["qe_a"]]
    return {"matches": int(len(p)), "log_loss_close": round(ll("qc"), 4), "log_loss_early": round(ll("qe"), 4),
            "log_loss_rating": round(ll("pm"), 4),
            "corr_gap_to_move": round(float(np.corrcoef(gap, mv)[0, 1]), 4) if len(p) > 2 else None}


def run(fx: pd.DataFrame, m: pd.DataFrame) -> dict:
    from research.open_close.football_cross_league_check import rated_matches
    w = wide(m)
    if w.empty:
        return {"status": "no Betfair match-odds prices found"}
    d, _ = rated_matches(fx)
    lk = link(w, d)
    out = {"markets_priced": int(len(w)), "first_market": str(w["kickoff"].min()), "last_market": str(w["kickoff"].max()),
           "fit_end": str(FIT_END.date())}
    if lk.empty:
        out["status"] = "no market linked to a fixture"
        return out
    out["linked"] = int(len(lk))
    out["result_disagrees_dropped"] = int((~lk["result_agrees"]).sum())
    lk = rating_probs(d, lk[lk["result_agrees"]])
    lk = lk[lk["kickoff"] >= FIT_END].dropna(subset=["pm_h"])
    out["linked_by_group"] = lk["group"].value_counts().to_dict()
    out["groups"] = {}
    for g in GROUPS:
        p = lk[lk["group"] == g]
        if p.empty:
            out["groups"][g] = {"status": "no data found"}
            continue
        out["groups"][g] = {"diagnostics": diagnostics(p), "home_and_away": targets(p),
                            "draw_apart": targets(p, legs=("d",))}
    return out


def main(argv=None) -> int:
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--match-odds", default=None, help="a local MATCH_ODDS extract instead of the S3 tables")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    fx = store.get_parquet("api_football/fixtures.parquet")
    if fx is None or not len(fx):
        res = {"status": "no data found (api_football/fixtures.parquet)"}
    else:
        res = run(fx, load_markets(store, a.match_odds))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
