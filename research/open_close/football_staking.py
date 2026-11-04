"""Staking for the football price-path trades: how much on each trade the expected-value rule takes.

racing² football research, 4 Nov 2026 (the owner: "all expected value above 2%. What is the optimal trading approach
and staking strategy?"). Each trade is entered at the early price (the collection moment) and greened at the close,
so every trade of a collection window (Friday-Monday, Tuesday-Thursday) is open at once and ties up capital: a back
first trade its stake, a lay first trade its liability (stake x (price - 1)). The bank, not the edge, is the binding
limit: unconstrained Kelly on a greened trade (small variance) would stake several banks.

Per window, with the bank B at its start, each staking rule sets each trade's capital as a share of B:

- ``level``: the same capital on every trade, the window's capital budget shared equally;
- ``ev``: capital in proportion to expected value;
- ``kelly``: mean-variance (Kelly for small edges) under the capital budget: capital share
  f_i = max(0, (mu_i - lambda) / s_i^2), mu_i = EV_i / c_i and s_i^2 = var_i / c_i^2 the expected return and its
  variance per unit of capital (c_i = 1 backing, price - 1 laying), lambda the budget's shadow price (bisection);
  this weighs edge against risk and against the capital a trade ties up;
- each with a cap a trade (``CAP``) and the window's budget ``L`` (the share of the bank at work at once).

Expected value: ``football_improve.by_ev`` (the forecast calibrated on the season before). The variance of a trade's
profit a unit stake: the realised early-to-close move's variance by price decile in the season before (Pinnacle
same-book rows), so nothing from the test season enters the stakes. Profits a tick off fair at each end, 2% commission
on a market's net win, the bank updated window by window.

    python -m research.open_close.football_staking --cache <lab cache> --out reports/football_runs/staking.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COMMISSION = 0.02
BUDGETS = (0.25, 0.5, 1.0)
CAP = 0.10
MARGINS = (0.0, 0.01, 0.02)
SPEC = "team_all+home_adv+team_news+form_short+x_short"


def _implied(q_open: np.ndarray, q_h: np.ndarray, q_a: np.ndarray, ph, pa) -> np.ndarray:
    """The draw's forecast move implied by home and away forecasts (the three chances sum to one)."""
    qd_hat = 1 - q_h * np.exp(ph) - q_a * np.exp(pa)
    return np.log(np.clip(qd_hat, 1e-4, 1) / q_open)


def _under(q_over: np.ndarray, po) -> np.ndarray:
    """The under's forecast move implied by the over's."""
    return np.log(np.clip(1 - q_over * np.exp(po), 1e-4, 1) / (1 - q_over))


def forecasts(d: pd.DataFrame, cols: list[str], season: int, fair: str) -> dict:
    """Every selection's forecast chance move for ``season``'s rows: home and away from the stack, the draw implied by
    them, over 2.5 from the goals stack, under implied by it. ``fair``: 'bf' (Betfair's early book, for trading) or
    'pin' (Pinnacle's early prices, the same book as the realised moves used to calibrate)."""
    from research.open_close import football_improve as fi
    te = d[d["test_season"] == season]
    ph = fi.walk_forward(d, cols, "y_h", "same_book", season)
    pa = fi.walk_forward(d, cols, "y_a", "same_book", season)
    po = fi.walk_forward(d, cols + fi.OU_BLOCK, "y_o", "same_book_ou", season)
    if fair == "bf":
        o = te[["bfe_open_h", "bfe_open_d", "bfe_open_a"]].to_numpy(float)
        q = (1 / o) / (1 / o).sum(axis=1, keepdims=True)
        ou = te[["bfe_open_o25", "bfe_open_u25"]].to_numpy(float)
        qo = (1 / ou[:, 0]) / (1 / ou).sum(axis=1)
        qh, qd, qa = q[:, 0], q[:, 1], q[:, 2]
    else:
        qh, qd, qa, qo = (te[c].to_numpy(float) for c in ("qo_h", "qo_d", "qo_a", "qo_o"))
    return {"H": ph, "A": pa, "D": _implied(qd, qh, qa, ph, pa), "O": po, "U": _under(qo, po)}


def calibrations(d: pd.DataFrame, cols: list[str], season: int) -> dict:
    """Each selection type's expected move on its forecast, fitted on the season before (out of sample, Pinnacle
    same-book rows): home and away pooled, the draw, the over, the under."""
    prev = season - 1
    te = d[d["test_season"] == prev]
    f = forecasts(d, cols, prev, "pin")
    y = {"H": te["y_h"], "A": te["y_a"], "D": np.log(te["qc_d"] / te["qo_d"]), "O": te["y_o"],
         "U": np.log((1 - te["qc_o"]) / (1 - te["qo_o"]))}
    book = {"H": te["same_book"], "A": te["same_book"], "D": te["same_book"], "O": te["same_book_ou"],
            "U": te["same_book_ou"]}
    out = {}
    for name, legs in (("HA", ("H", "A")), ("D", ("D",)), ("O", ("O",)), ("U", ("U",))):
        ys = np.concatenate([y[k].to_numpy(float)[book[k].to_numpy(bool)] for k in legs])
        ps = np.concatenate([np.asarray(f[k])[book[k].to_numpy(bool)] for k in legs])
        ok = np.isfinite(ys) & np.isfinite(ps)
        b, a = np.polyfit(ps[ok], ys[ok], 1)
        out[name] = {"a": float(a), "b": float(b), "rows": int(ok.sum()),
                     "corr": float(np.corrcoef(ps[ok], ys[ok])[0, 1])}
    return out


def variance_model(d: pd.DataFrame, season: int) -> tuple[np.ndarray, np.ndarray]:
    """Deciles of log price and the realised move's variance in each, from the season before ``season`` (same-book
    rows; every selection pooled): a greened trade's profit a unit stake is close to its log price move."""
    prev = d[d["test_season"] == season - 1]
    sb, so = prev["same_book"].to_numpy(bool), prev["same_book_ou"].to_numpy(bool)
    parts = [(prev["open_h"], prev["y_h"], sb), (prev["open_a"], prev["y_a"], sb),
             (prev["open_d"], np.log(prev["qc_d"] / prev["qo_d"]), sb), (prev["open_o25"], prev["y_o"], so),
             (prev["open_u25"], np.log((1 - prev["qc_o"]) / (1 - prev["qo_o"])), so)]
    lp = np.concatenate([np.log(p.to_numpy(float))[m] for p, _, m in parts])
    y = np.concatenate([np.asarray(v, float)[m] for _, v, m in parts])
    ok = np.isfinite(lp) & np.isfinite(y)
    lp, y = lp[ok], y[ok]
    edges = np.quantile(lp, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(edges, lp, side="right") - 1, 0, 9)
    var = np.array([np.var(y[idx == k]) for k in range(10)])
    return edges, var


def trade_var(price: np.ndarray, edges: np.ndarray, var: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(edges, np.log(price), side="right") - 1, 0, 9)
    return var[idx]


def kelly_shares(mu: np.ndarray, s2: np.ndarray, budget: float, cap: float) -> np.ndarray:
    """Capital shares maximising sum(mu f - s2 f^2 / 2) with 0 <= f <= cap and sum(f) <= budget."""
    f = lambda lam: np.clip((mu - lam) / s2, 0, cap)             # noqa: E731
    if f(0.0).sum() <= budget:
        return f(0.0)
    lo, hi = 0.0, float(mu.max())
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid).sum() > budget else (lo, mid)
    return f(hi)


def shares(t: pd.DataFrame, rule: str, budget: float, cap: float) -> np.ndarray:
    n = len(t)
    if n == 0:
        return np.zeros(0)
    if rule == "level":
        return np.full(n, min(cap, budget / n))
    if rule == "ev":
        w = t["ev"].to_numpy(float).clip(min=0)
        f = budget * w / w.sum() if w.sum() > 0 else np.zeros(n)
        return np.minimum(f, cap)
    return kelly_shares(t["mu"].to_numpy(float), t["s2"].to_numpy(float), budget, cap)


def simulate(sel: pd.DataFrame, rule: str, budget: float, cap: float = CAP, how: str = "green") -> dict:
    """The bank (starting at 1) window by window: each trade's profit = capital / capital-per-unit x profit a unit,
    commission off a winning trade. ``how``: 'green' (closed at the close) or 'hold' (left to the result)."""
    pre = "" if how == "green" else "hold_"
    bank, path, turnover, rows = 1.0, [1.0], 0.0, []
    for w, t in sel.groupby("window", sort=True):
        t = t.assign(mu=t[f"{pre}mu"], s2=t[f"{pre}s2"], ev=t[f"{pre}ev"])
        f = shares(t, rule, budget, cap)
        units = bank * f / t[f"{pre}cap_unit"].to_numpy(float)
        prof = units * t[f"{pre}profit"].to_numpy(float)
        prof = np.where(prof > 0, prof * (1 - COMMISSION), prof)
        turnover += float((bank * f).sum())
        rows.append({"window": w, "trades": int((f > 0).sum()), "capital_share": float(f.sum()),
                     "return": float(prof.sum() / bank)})
        bank = max(bank + float(prof.sum()), 1e-9)
        path.append(bank)
    p = np.array(path)
    dd = float(np.max(1 - p / np.maximum.accumulate(p)))
    r = pd.DataFrame(rows)
    return {"bank_end": round(bank, 4), "max_drawdown": round(dd, 4), "windows": int(len(r)),
            "windows_up": round(float((r["return"] > 0).mean()), 3) if len(r) else None,
            "worst_window": round(float(r["return"].min()), 4) if len(r) else None,
            "mean_capital_share": round(float(r["capital_share"].mean()), 3) if len(r) else None,
            "return_on_capital": round((bank - 1) / turnover, 5) if turnover else None,
            "window_sharpe": round(float(r["return"].mean() / r["return"].std()), 3) if len(r) > 2 else None}


def pool(d: pd.DataFrame, cols: list[str], season: int, margin: float) -> tuple[pd.DataFrame, dict]:
    """Every selection (home, away, draw, over, under) whose expected value after costs clears ``margin``, the
    better way round (back first or lay first), with what each costs in capital and risks, greened or held."""
    from football.team_features import collection_day
    from research.open_close import football_improve as fi
    from research.open_close.backtest import tick_over, tick_under
    te = d[d["test_season"] == season]
    f = forecasts(d, cols, season, "bf")
    cal = calibrations(d, cols, season)
    edges, var = variance_model(d, season)
    c = pd.concat([fi.candidates(te, {k: f[k] for k in ("H", "A", "D")}, "MO"),
                   fi.candidates(te, {k: f[k] for k in ("O", "U")}, "OU")], ignore_index=True)
    sel = pd.concat([fi.by_ev(c[c["leg"].isin(legs)], cal[name], "tick", margin)
                     for name, legs in (("HA", ("H", "A")), ("D", ("D",)), ("O", ("O",)), ("U", ("U",)))],
                    ignore_index=True)
    sel["window"] = collection_day(pd.to_datetime(sel["day"])).dt.strftime("%Y-%m-%d").to_numpy()
    # greened at the close
    sel["profit"] = sel["profit_tick"]
    sel["cap_unit"] = np.where(sel["side"] > 0, 1.0, sel["price"] - 1)
    sel["mu"] = sel["ev"] / sel["cap_unit"]
    sel["s2"] = trade_var(sel["price"].to_numpy(float), edges, var) / sel["cap_unit"] ** 2
    # held to the result: backed a tick under fair, laid a tick over; the expected close chance as the chance
    fo = sel["fair_open"].to_numpy(float)
    ob = np.array([tick_under(x) for x in fo])
    ol = np.array([tick_over(x) for x in fo])
    won = sel["won"].to_numpy(float)
    back = sel["side"].to_numpy() > 0
    cal_of = {leg: cal[name] for name, legs in (("HA", "HA"), ("D", "D"), ("O", "O"), ("U", "U")) for leg in legs}
    ey = np.array([cal_of[lg]["a"] + cal_of[lg]["b"] * p for lg, p in zip(sel["leg"], sel["pred"])])
    q = np.clip(np.exp(ey) / fo, 1e-4, 0.999)
    sel["hold_profit"] = np.where(back, np.where(won == 1, ob - 1, -1.0), np.where(won == 1, -(ol - 1), 1.0))
    sel["hold_cap_unit"] = np.where(back, 1.0, ol - 1)
    sel["hold_ev"] = np.where(back, q * (ob - 1) * (1 - COMMISSION) - (1 - q),
                              (1 - q) * (1 - COMMISSION) - q * (ol - 1))
    sel["hold_mu"] = sel["hold_ev"] / sel["hold_cap_unit"]
    sel["hold_s2"] = np.where(back, q * (1 - q) * ob ** 2, q * (1 - q) * ol ** 2) / sel["hold_cap_unit"] ** 2
    sel = sel[np.isfinite(won)]
    return sel, {"calibration": cal, "variance_by_price_decile": dict(zip(np.round(np.exp(edges[1:]), 2).astype(str),
                                                                          np.round(var, 5)))}


def run(cache: str) -> dict:
    from research.open_close import football_improve as fi
    d, cols = fi.load(cache, SPEC)
    out = {"spec": SPEC, "cap": CAP, "budgets": BUDGETS, "commission": COMMISSION}
    for s in fi.TEST_SEASONS:
        res = {}
        for mg in MARGINS:
            sel, info = pool(d, cols, s, mg)
            res.setdefault("models", info)
            net = lambda x: np.where(x > 0, x * (1 - COMMISSION), x)    # noqa: E731
            r = {"trades": int(len(sel)), "trades_per_window": round(len(sel) / max(sel["window"].nunique(), 1), 1),
                 "by_leg": {}, "by_type": {}}
            for lg, g in sel.groupby("leg"):
                r["by_leg"][lg] = {"trades": int(len(g)), "back_first": round(float((g["side"] > 0).mean()), 3),
                                   "green_roi": round(float(net(g["profit"]).mean()), 5),
                                   "hold_roi": round(float(net(g["hold_profit"]).mean()), 5),
                                   "right_way": round(float(g["right_way"].mean()), 3)}
            for nm, g in (("back_to_lay", sel[sel["side"] > 0]), ("lay_to_back", sel[sel["side"] < 0])):
                r["by_type"][nm] = {"trades": int(len(g)), "green_roi": round(float(net(g["profit"]).mean()), 5),
                                    "hold_roi_back_or_lay": round(float(net(g["hold_profit"]).mean()), 5)}
            r["level_green_roi"] = round(float(net(sel["profit"]).mean()), 5)
            r["level_hold_roi"] = round(float(net(sel["hold_profit"]).mean()), 5)
            for how in ("green", "hold"):
                for rule in ("level", "ev", "kelly"):
                    for b in BUDGETS:
                        r[f"{how}_{rule}_L{b}"] = simulate(sel, rule, b, how=how)
            res[f"ev_above_{mg}"] = r
            log.info("season %d margin %.2f: %d trades, green %.4f hold %.4f", s, mg, len(sel),
                     r["level_green_roi"], r["level_hold_roi"])
        out[str(s)] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run(a.cache)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
