"""The football model, walk-forward, scored against the market's open and close.

Stage 1, market-blind: the pooled Dixon-Coles ratings (``football.ratings``) refitted every ``refit_days`` from the
matches before, each country on its own; every match in the next block is priced from the ratings as they stood
before it (no match ever sees its own result or any later one). Prices: 1X2, over/under 2.5, the Asian handicap at
the open and the closing line.

Stage 2, with the market: the model's chances pooled with the opening price's (logarithmic pooling,
``p ~ p_model^a * p_open^b``, plus a draw term for 1X2), the weights fitted a season at a time on the earlier
seasons only. ``a`` is the honest answer to "does the model know anything the open does not?": a weight near zero
says no.

Scored three ways: the log-loss of each forecast (the model, the open, the pooled, the close); the value rules (back
an outcome where the expected return at the opening price clears an edge), settled at the result and at the
closing line (CLV: the expected return at the price taken if the close is fair, the measure that converges in
hundreds of bets rather than thousands); and by season and division.

    python -m football.model --root <sources folder> --from 2012-07-01 --out out/football_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from football.ratings import ah_fair, ah_return, diff_dist, fit_pool, markets, score_matrix

log = logging.getLogger(__name__)

COM = 0.02                     # Betfair commission on net winnings assumed for exchange prices
EDGES = (0.02, 0.05, 0.10)
DEFAULTS = {"xi": 0.0023, "l2": 3.0, "window_days": 1100, "sot_mix": 0.3, "refit_days": 7}


# --------------------------------------------------------------------------------------------------------------------
# Stage 1: walk-forward ratings
# --------------------------------------------------------------------------------------------------------------------

def _price_rows(r, rows: pd.DataFrame) -> list[dict]:
    out = []
    for x in rows.itertuples(index=False):
        lh, la = r.rates(x.home_team_id, x.away_team_id, x.competition_id)
        m = score_matrix(lh, la, r.rho)
        p = markets(m)
        rec = {"match_id": x.match_id, "lam_h": lh, "lam_a": la, **p,
               "known_h": x.home_team_id in r.teams, "known_a": x.away_team_id in r.teams}
        dd = None
        for phase in ("open", "close"):
            line = getattr(x, f"{phase}_ah_line")
            if pd.notna(line) and abs(line * 4 - round(line * 4)) < 1e-9:
                dd = dd or diff_dist(m)
                rec[f"ah_fair_{phase}_h"] = ah_fair(dd, line, "H")
                rec[f"ah_fair_{phase}_a"] = ah_fair(dd, line, "A")
                for side in ("h", "a"):
                    price = getattr(x, f"{phase}_ah_{side}")
                    if pd.notna(price):
                        rec[f"ah_ev_{phase}_{side}"] = ah_return(dd, line, price, side.upper())
        out.append(rec)
    return out


def walk_pool(args) -> pd.DataFrame:
    """One country's walk-forward (run in its own process)."""
    d, pool, pred_from, pred_to, params = args
    p = {**DEFAULTS, **(params or {})}
    d = d.sort_values("match_date")
    todo = d[(d.match_date >= pred_from) & ((d.match_date < pred_to) if pred_to else True)]
    if todo.empty:
        return pd.DataFrame()
    out = []
    dates = np.array(sorted(str(x) for x in todo.match_date.unique()), dtype=object)
    i = 0
    while i < len(dates):
        start = str(dates[i])
        end = (pd.Timestamp(start) + pd.Timedelta(days=p["refit_days"])).strftime("%Y-%m-%d")
        block = todo[(todo.match_date >= start) & (todo.match_date < end)]
        r, _ = fit_pool(d, pool, start, xi=p["xi"], l2=p["l2"], window_days=p["window_days"],
                        sot_mix=p["sot_mix"])
        if r is not None:
            out += _price_rows(r, block)
        i = int(np.searchsorted(dates, end))
    return pd.DataFrame(out)


def walk_forward(g: pd.DataFrame, pred_from: str, pred_to: str | None = None, params: dict | None = None,
                 workers: int = 4, pools: list[str] | None = None) -> pd.DataFrame:
    """Out-of-sample stage-1 prices for every match from ``pred_from``, joined back to the gold rows."""
    pools = pools or sorted(g.country.unique())
    jobs = [(g[g.country == c], c, pred_from, pred_to, params) for c in pools]
    if workers > 1:
        # one BLAS thread a worker: the fits are many and small, and threads per process oversubscribe the cores
        import multiprocessing as mp
        import os
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = "1"
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(walk_pool, jobs))
    else:
        parts = [walk_pool(j) for j in jobs]
    pr = pd.concat([x for x in parts if len(x)], ignore_index=True)
    return g.merge(pr, on="match_id", how="inner")


# --------------------------------------------------------------------------------------------------------------------
# Market probabilities, outcomes, pooling
# --------------------------------------------------------------------------------------------------------------------

def devig(prices: np.ndarray) -> np.ndarray:
    """Proportional de-vig of rows of decimal prices -> probabilities summing to one (NaN rows stay NaN)."""
    inv = 1.0 / prices
    return inv / inv.sum(axis=1, keepdims=True)


def outcomes(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.DataFrame(index=df.index)
    diff = df.ft_home - df.ft_away
    y["y_h"], y["y_d"], y["y_a"] = (diff > 0).astype(float), (diff == 0).astype(float), (diff < 0).astype(float)
    y["y_o25"] = ((df.ft_home + df.ft_away) >= 3).astype(float)
    return y


def market_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for ph in ("open", "close"):
        p = devig(df[[f"{ph}_h", f"{ph}_d", f"{ph}_a"]].to_numpy(float))
        out[[f"q{ph}_h", f"q{ph}_d", f"q{ph}_a"]] = p
        q = devig(df[[f"{ph}_o25", f"{ph}_u25"]].to_numpy(float))
        out[f"q{ph}_o25"] = q[:, 0]
    return out


def _pool_1x2(theta, pm, pk):
    a, b, c = theta
    z = a * np.log(pm) + b * np.log(pk)
    z[:, 1] += c
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _pool_bin(theta, pm, pk):
    a, b, c = theta
    lg = lambda p: np.log(p / (1 - p))
    return 1 / (1 + np.exp(-(a * lg(pm) + b * lg(pk) + c)))


def fit_pooling(pm, pk, y, kind: str) -> np.ndarray:
    pm, pk = np.clip(pm, 1e-6, 1 - 1e-6), np.clip(pk, 1e-6, 1 - 1e-6)
    if kind == "1x2":
        def nll(t):
            return -np.mean(np.log(np.clip((_pool_1x2(t, pm, pk) * y).sum(axis=1), 1e-12, None)))
    else:
        def nll(t):
            p = np.clip(_pool_bin(t, pm, pk), 1e-12, 1 - 1e-12)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return minimize(nll, np.array([0.3, 0.7, 0.0]), method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000}).x


def stage2(df: pd.DataFrame, min_seasons: int = 2) -> tuple[pd.DataFrame, list[dict]]:
    """Pool the model with the open, season by season, each season's weights fitted on the earlier seasons."""
    df = df.copy()
    y = outcomes(df)
    q = market_probs(df)
    df = pd.concat([df, y, q], axis=1)
    for c in ("pp_h", "pp_d", "pp_a", "pp_o25"):
        df[c] = np.nan
    weights = []
    seasons = sorted(df.season.unique())
    fin = df.status == "FINISHED"
    for s in seasons[min_seasons:]:
        tr = fin & (df.season < s)
        te = df.season == s
        ok1 = df[["qopen_h", "qopen_d", "qopen_a"]].notna().all(axis=1)
        a = tr & ok1
        if a.sum() > 500:
            th = fit_pooling(df.loc[a, ["p_h", "p_d", "p_a"]].to_numpy(), df.loc[a, ["qopen_h", "qopen_d",
                             "qopen_a"]].to_numpy(), df.loc[a, ["y_h", "y_d", "y_a"]].to_numpy(), "1x2")
            b = te & ok1
            df.loc[b, ["pp_h", "pp_d", "pp_a"]] = _pool_1x2(th, np.clip(df.loc[b, ["p_h", "p_d", "p_a"]].to_numpy(),
                                                           1e-6, 1), np.clip(df.loc[b, ["qopen_h", "qopen_d",
                                                                                         "qopen_a"]].to_numpy(), 1e-6, 1))
            weights.append({"season": int(s), "market": "1X2", "a_model": th[0], "b_open": th[1], "draw": th[2],
                            "fit_rows": int(a.sum())})
        ok2 = df.qopen_o25.notna()
        a = tr & ok2
        if a.sum() > 500:
            th = fit_pooling(df.loc[a, "p_o25"].to_numpy(), df.loc[a, "qopen_o25"].to_numpy(),
                             df.loc[a, "y_o25"].to_numpy(), "bin")
            b = te & ok2
            df.loc[b, "pp_o25"] = _pool_bin(th, np.clip(df.loc[b, "p_o25"].to_numpy(), 1e-6, 1 - 1e-6),
                                            np.clip(df.loc[b, "qopen_o25"].to_numpy(), 1e-6, 1 - 1e-6))
            weights.append({"season": int(s), "market": "OU25", "a_model": th[0], "b_open": th[1], "const": th[2],
                            "fit_rows": int(a.sum())})
    return df, weights


# --------------------------------------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------------------------------------

def ll_1x2(p, y):
    return float(-np.mean(np.log(np.clip((p * y).sum(axis=1), 1e-12, None))))


def ll_bin(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def forecast_scores(df: pd.DataFrame) -> list[dict]:
    """Log-loss of each forecast on the same matches (those with every forecast present), by season and overall."""
    out = []
    d = df[df.status == "FINISHED"]
    sets = {"1X2": (["p_h", "p_d", "p_a"], ["qopen_h", "qopen_d", "qopen_a"], ["pp_h", "pp_d", "pp_a"],
                    ["qclose_h", "qclose_d", "qclose_a"], ["y_h", "y_d", "y_a"]),
            "OU25": (["p_o25"], ["qopen_o25"], ["pp_o25"], ["qclose_o25"], ["y_o25"])}
    for mk, (pm, po, pp, pc, yy) in sets.items():
        ok = d[pm + po + pp + pc].notna().all(axis=1)
        e = d[ok]
        for season, grp in [("all", e)] + list(e.groupby("season")):
            if len(grp) < 100:
                continue
            f = ll_1x2 if mk == "1X2" else (lambda p, y: ll_bin(p[:, 0], y[:, 0]))
            Y = grp[yy].to_numpy()
            out.append({"market": mk, "season": season if season == "all" else int(season), "n": int(len(grp)),
                        "model": f(grp[pm].to_numpy(), Y), "open": f(grp[po].to_numpy(), Y),
                        "pooled": f(grp[pp].to_numpy(), Y), "close": f(grp[pc].to_numpy(), Y)})
    return out


SELECTIONS = {
    # market, outcome: (prob column, result column, close-fair-prob column)
    ("1X2", "H"): ("h", "y_h", "qclose_h"), ("1X2", "D"): ("d", "y_d", "qclose_d"),
    ("1X2", "A"): ("a", "y_a", "qclose_a"),
    ("OU25", "O"): ("o25", "y_o25", "qclose_o25"), ("OU25", "U"): ("u25", None, None),
}


def bets(df: pd.DataFrame, prob: str = "pp", venue: str = "open", edge: float = 0.05, cap: float = 10.0,
         com: float | None = None) -> pd.DataFrame:
    """Every bet a value rule makes: back each 1X2 / O/U 2.5 outcome whose expected return at the venue's opening
    price (``open``: the benchmark book, no commission; ``bfe``: Betfair Exchange, after commission) clears
    ``edge``, at prices up to ``cap``. Each settled at the result (pnl a unit staked) and at the close (clv: the
    expected return at the price taken were the close fair)."""
    com = (COM if venue == "bfe" else 0.0) if com is None else com
    d = df[df.status == "FINISHED"]
    rows = []
    for (mk, oc), (k, ycol, ccol) in SELECTIONS.items():
        if oc == "U":
            p = 1 - d[f"{prob}_o25"]
            y = 1 - d["y_o25"]
            qc = 1 - d["qclose_o25"]
        else:
            p, y, qc = d[f"{prob}_{k}"], d[ycol], d[ccol]
        price = d[f"bfe_open_{k}" if venue == "bfe" else f"open_{k}"]
        ev = p * (price - 1) * (1 - com) - (1 - p)
        take = (ev > edge) & (price <= cap) & p.notna() & price.notna()
        if not take.any():
            continue
        t = d[take]
        won = y[take]
        pr = price[take]
        rows.append(pd.DataFrame({
            "match_id": t.match_id, "season": t.season, "competition_id": t.competition_id, "tier": t.tier,
            "match_date": t.match_date, "market": mk, "outcome": oc, "p": p[take], "price": pr, "ev": ev[take],
            "won": won, "pnl": np.where(won == 1, (pr - 1) * (1 - com), -1.0),
            "clv": pr * qc[take] * 1.0 - 1.0}))
    for side in ("h", "a"):                                   # the Asian handicap at the open line
        col = f"ah_ev_open_{side}"
        if col not in d or prob != "p" or venue != "open":
            continue
        price = d[f"open_ah_{side}"]
        take = (d[col] > edge) & (price <= cap)
        if not take.any():
            continue
        t = d[take]
        line = t.open_ah_line
        diff = (t.ft_home - t.ft_away)
        res = _ah_result(diff, line, side)
        same = (t.close_ah_line == t.open_ah_line) & t[f"close_ah_{side}"].notna()
        fair_close = devig(t[["close_ah_h", "close_ah_a"]].to_numpy(float))[:, 0 if side == "h" else 1]
        rows.append(pd.DataFrame({
            "match_id": t.match_id, "season": t.season, "competition_id": t.competition_id, "tier": t.tier,
            "match_date": t.match_date, "market": "AH", "outcome": side.upper(), "p": np.nan,
            "price": price[take], "ev": t[col], "won": (res > 0).astype(float),
            "pnl": res * np.where(res > 0, price[take] - 1, 1.0),
            # the close at the same line only: priced as a two-way market, de-vigged
            "clv": np.where(same, price[take] * fair_close - 1.0, np.nan)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _ah_result(diff, line, side):
    """+1 win, +0.5 half win, 0 push, -0.5 half loss, -1 loss (per unit), for a home handicap ``line``."""
    out = np.zeros(len(diff))
    diff, line = np.asarray(diff, float), np.asarray(line, float)
    quarter = np.abs(np.round(line * 4)) % 2 == 1
    for k, l in enumerate(line):
        halves = [l - 0.25, l + 0.25] if quarter[k] else [l]
        r = 0.0
        for h in halves:
            adj = diff[k] + h if side == "h" else -diff[k] - h
            r += (1.0 if adj > 0 else -1.0 if adj < 0 else 0.0) / len(halves)
        out[k] = r
    return out


def summarise(b: pd.DataFrame) -> dict:
    if b is None or b.empty:
        return {"n": 0}
    pnl, clv = b.pnl.to_numpy(float), b.clv.dropna().to_numpy(float)
    return {"n": int(len(b)), "roi": float(pnl.mean()), "roi_se": float(pnl.std(ddof=1) / np.sqrt(len(pnl)))
            if len(pnl) > 1 else None, "avg_price": float(b.price.mean()),
            "clv": float(clv.mean()) if len(clv) else None,
            "clv_se": float(clv.std(ddof=1) / np.sqrt(len(clv))) if len(clv) > 1 else None,
            "clv_pos_share": float((clv > 0).mean()) if len(clv) else None, "clv_n": int(len(clv))}


def rule_table(df: pd.DataFrame) -> list[dict]:
    out = []
    for prob in ("p", "pp"):
        for venue in ("open", "bfe"):
            for edge in EDGES:
                b = bets(df, prob, venue, edge)
                if b.empty:
                    continue
                for mk, grp in b.groupby("market"):
                    out.append({"prob": "model" if prob == "p" else "pooled", "venue": venue, "edge": edge,
                                "market": mk, **summarise(grp)})
    return out


def by_group(b: pd.DataFrame, key: str) -> list[dict]:
    return [{key: (k if isinstance(k, str) else int(k)), **summarise(g)} for k, g in b.groupby(key)]


def report(df: pd.DataFrame, weights: list[dict], params: dict) -> dict:
    fin = df[df.status == "FINISHED"]
    cover = {"matches": int(len(fin)), "from": fin.match_date.min(), "to": fin.match_date.max(),
             "with_open_1x2": int(fin[["open_h", "open_d", "open_a"]].notna().all(axis=1).sum()),
             "with_close_1x2": int(fin[["close_h", "close_d", "close_a"]].notna().all(axis=1).sum()),
             "with_bfe_open": int(fin[["bfe_open_h", "bfe_open_d", "bfe_open_a"]].notna().all(axis=1).sum()),
             "open_src": fin.open_1x2_src.value_counts().to_dict(),
             "close_src": fin.close_1x2_src.value_counts().to_dict(),
             "known_both_teams": float((fin.known_h & fin.known_a).mean())}
    primary = bets(df, "pp", "bfe", 0.05)
    model_open = bets(df, "p", "open", 0.05)
    return {"params": params, "coverage": cover, "forecasts": forecast_scores(df), "pooling_weights": weights,
            "rules": rule_table(df),
            "pooled_bfe_5pct": {"by_season": by_group(primary, "season"), "by_tier": by_group(primary, "tier"),
                                "by_competition": by_group(primary, "competition_id")},
            "model_open_5pct": {"by_season": by_group(model_open, "season"), "by_tier": by_group(model_open, "tier"),
                                "by_market": by_group(model_open, "market")}}


def main(argv=None) -> int:
    from football.data import load_gold
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="a local sources folder in place of S3")
    ap.add_argument("--from", dest="date_from", default="2012-07-01", help="first match priced")
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--pools", default=None, help="countries, e.g. ENG,GER (default all)")
    ap.add_argument("--xi", type=float, default=DEFAULTS["xi"])
    ap.add_argument("--l2", type=float, default=DEFAULTS["l2"])
    ap.add_argument("--sot-mix", type=float, default=DEFAULTS["sot_mix"])
    ap.add_argument("--refit-days", type=int, default=DEFAULTS["refit_days"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-pred", default=None, help="parquet of every priced match")
    ap.add_argument("--pred", default=None, help="score saved predictions instead of fitting")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    params = {"xi": a.xi, "l2": a.l2, "sot_mix": a.sot_mix, "refit_days": a.refit_days,
              "window_days": DEFAULTS["window_days"]}
    if a.pred:
        df = pd.read_parquet(a.pred)
    else:
        g = load_gold(Store(root=a.root))
        df = walk_forward(g, a.date_from, a.date_to, params, a.workers,
                          a.pools.split(",") if a.pools else None)
        if a.save_pred:
            df.to_parquet(a.save_pred, index=False)
    df, weights = stage2(df)
    rep = report(df, weights, params)
    text = json.dumps(rep, indent=1, default=lambda x: None if x is None or (isinstance(x, float) and np.isnan(x))
                      else (float(x) if isinstance(x, (np.floating, np.integer)) else str(x)))
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(text[:20000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
