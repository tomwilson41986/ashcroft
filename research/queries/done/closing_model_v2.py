"""Closing model v2 (research, dev window only): does anything known at trading time bring the forecast of the
BSP book nearer, and the owner's staking's CLV up?

Walk-forward by month on January-March 2026 (holdout not read): fitted on the months before, February and March
scored. Variants add to the served closing model's seven inputs:
  curv   lq_market squared (the morning price's own favourite-longshot bend)
  time   hours from 11:00 to the off, with each log probability
  field  log field size, with each log probability
  code   jumps (1) or flat (0), with each log probability
  gap    |lq_model - lq_market|, with each log probability
and, for the spread of the miss, bands by volume and by hours to the off.

Result (30 Sep; iteration 105's walk-forward forecasts and the 23 Sep export of the Betfair prices, the inputs of
research query run 36676743958, whose numbers the same local inputs reproduced exactly): nothing moves. Nearness
to the BSP book (mean |log q_hat - log q_close|) 0.3021 / 0.3131 (Feb / Mar) for the served closing model and
0.3020-0.3025 / 0.3130-0.3134 for each single addition; all five together 0.3025 / 0.3327 (March worse). The
owner's staking at expected CLV >= 3%: +7.21% CLV served, +7.06% to +7.28% for the variants, held to the result
+6.43% to +8.10% against +6.91% (noise of the result). The closing model already reads what the prices can say;
its spread by hours to the off changes nothing either. Not adopted.

    python research/queries/done/closing_model_v2.py [--forecasts oos.csv] [--extract extract.csv.gz]
"""

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, ".")
sys.path.insert(0, "research/queries/done")
from model import race_book as rb  # noqa: E402
from race_book_backtest import load  # noqa: E402



def minutes(t) -> float:
    parts = str(t).replace(":", ".").strip(".").split(".")
    try:
        h, m = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return np.nan
    return (h + 12 if h < 10 else h) * 60 + m


def base_inputs(d):
    x = rb.closing_inputs(d)
    x["hours"] = (d["race_time"].map(minutes).to_numpy(float) - 660.0) / 60.0
    x["lfield"] = np.log(d.groupby("race")["horse_name"].transform("size").to_numpy(float))
    x["jumps"] = (~d["race_code"].astype(str).str.lower().str.startswith("flat")).astype(float).to_numpy()
    return x


def design(x, stats, variant):
    v = (x["log_vol"].to_numpy(float) - stats["vm"]) / stats["vs"]
    dep = (x["log_depth"].to_numpy(float) - stats["dm"]) / stats["ds"]
    lm, lf = x["lq_market"].to_numpy(float), x["lq_model"].to_numpy(float)
    cols = [lm, lf, v, lm * v, lf * v, lm * dep, lf * dep]
    if "curv" in variant:
        cols.append(lm ** 2 / 10.0)
    if "time" in variant:
        h = (x["hours"].to_numpy(float) - stats["hm"]) / stats["hs"]
        cols += [lm * h, lf * h]
    if "field" in variant:
        fz = (x["lfield"].to_numpy(float) - stats["fm"]) / stats["fs"]
        cols += [lm * fz, lf * fz]
    if "code" in variant:
        j = x["jumps"].to_numpy(float)
        cols += [lm * j, lf * j]
    if "gap" in variant:
        gp = np.abs(lf - lm)
        cols += [lm * gp, lf * gp]
    return np.column_stack(cols)


def fit(x, bsp, g, variant, l2=1e-4):
    q = rb.normalise_within(1.0 / np.asarray(bsp, float), g)
    sd = lambda a: float(np.std(a)) if np.std(a) > 1e-6 else 1.0  # noqa: E731
    stats = {"vm": x["log_vol"].mean(), "vs": sd(x["log_vol"]), "dm": x["log_depth"].mean(), "ds": sd(x["log_depth"]),
             "hm": x["hours"].mean(), "hs": sd(x["hours"]), "fm": x["lfield"].mean(), "fs": sd(x["lfield"])}
    X = design(x, stats, variant)
    nr = g.max() + 1

    def loss(beta):
        lqh = rb.group_log_softmax(X @ beta, g)
        return (-float(np.sum(q * lqh)) / nr + 0.5 * l2 * float(beta @ beta),
                -((q - np.exp(lqh))[:, None] * X).sum(axis=0) / nr + l2 * beta)

    b0 = np.zeros(X.shape[1])
    b0[0] = b0[1] = 0.5
    beta = minimize(loss, b0, jac=True, method="L-BFGS-B").x
    qh = np.exp(rb.group_log_softmax(X @ beta, g))
    resid = np.log(q) - np.log(qh)
    return {"beta": beta, "stats": stats, "variant": variant, "resid": resid, "x": x,
            "book": float(np.median(np.bincount(g, weights=1.0 / np.asarray(bsp, float))))}


def sigma_model(m, by_time):
    """Robust spread of the miss by volume quintile (and by hours-to-off tercile)."""
    x, r = m["x"], m["resid"]
    lv = x["log_vol"].to_numpy(float)
    ve = np.quantile(lv, [0.2, 0.4, 0.6, 0.8])
    vb = np.searchsorted(ve, lv, side="right")
    te = np.quantile(x["hours"].to_numpy(float), [1 / 3, 2 / 3]) if by_time else np.array([])
    tb = np.searchsorted(te, x["hours"].to_numpy(float), side="right") if by_time else np.zeros(len(lv), int)
    cell = vb * 3 + tb
    table = {}
    for c in np.unique(cell):
        rr = r[cell == c]
        table[c] = 1.4826 * np.median(np.abs(rr - np.median(rr))) if len(rr) > 20 else float(np.std(r))
    return {"ve": ve, "te": te, "table": table, "by_time": by_time, "all": float(np.std(r))}


def sigma(sm, x):
    lv = x["log_vol"].to_numpy(float)
    vb = np.searchsorted(sm["ve"], lv, side="right")
    tb = np.searchsorted(sm["te"], x["hours"].to_numpy(float), side="right") if sm["by_time"] else np.zeros(len(lv), int)
    return np.array([sm["table"].get(c, sm["all"]) for c in vb * 3 + tb])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", help="local copy of the forecasts (default: the iteration 105 artifact)")
    ap.add_argument("--extract", help="local Betfair export (default: the database)")
    a = ap.parse_args()
    d = load(a.forecasts, a.extract)
    d = d.sort_values(["race_date", "race", "predicted_bfsp"]).reset_index(drop=True)
    months = sorted(d["month"].unique())
    variants = [("served", (), False), ("curv", ("curv",), False), ("time", ("time",), False),
                ("field", ("field",), False), ("code", ("code",), False), ("gap", ("gap",), False),
                ("all5", ("curv", "time", "field", "code", "gap"), False),
                ("all5+sigma_t", ("curv", "time", "field", "code", "gap"), True),
                ("served+sigma_t", (), True)]
    res = {name: [] for name, *_ in variants}
    near = {name: [] for name, *_ in variants}
    for tm in months[1:]:
        tr, te = d[d["month"] < tm], d[d["month"] == tm]
        gtr, gte = rb.race_codes(tr["race"]), rb.race_codes(te["race"])
        xtr, xte = base_inputs(tr), base_inputs(te)
        qc = rb.normalise_within(1.0 / te["bsp"].to_numpy(float), gte)
        for name, var, by_time in variants:
            m = fit(xtr, tr["bsp"], gtr, var)
            sm = sigma_model(m, by_time)
            qh = np.exp(rb.group_log_softmax(design(xte, m["stats"], var) @ m["beta"], gte))
            near[name].append((len(te), float(np.mean(np.abs(np.log(qh) - np.log(qc))))))
            sig = sigma(sm, xte)
            rng = np.random.default_rng(int(tm.replace("-", "")))
            ev = np.empty(len(te))
            m_arr = te["morningwap"].to_numpy(float)
            for idx in te.groupby("race", sort=False).indices.values():
                draws = rb.closing_draws(qh[idx], sig[idx], book=m["book"], n_draws=300, rng=rng)
                ev[idx] = (m_arr[idx] * draws).mean(axis=0) - 1.0
            res[name].append(te.assign(ev=ev))
    print(f"{'variant':16s} {'near Feb':>9s} {'near Mar':>9s}   ev>=3%: {'bets':>6s} {'turnover':>9s} {'CLV':>8s} "
          f"{'CLV %':>7s} {'held':>8s} {'held %':>7s}   ev>=5%: {'bets':>6s} {'CLV %':>7s} {'held %':>7s}")
    for name, *_ in variants:
        sc = pd.concat(res[name])
        g3, g5 = rb.to_win_selection(sc, bar=0.03), rb.to_win_selection(sc, bar=0.05)
        nf = " ".join(f"{v:9.4f}" for _, v in near[name])
        print(f"{name:16s} {nf}   {g3['bets'].sum():12,d} {g3['staked'].sum():9,.0f} {g3['clv'].sum():+8,.0f} "
              f"{100 * g3['clv'].sum() / g3['staked'].sum():+6.2f}% {g3['result'].sum():+8,.0f} "
              f"{100 * g3['result'].sum() / g3['staked'].sum():+6.2f}%   {g5['bets'].sum():12,d} "
              f"{100 * g5['clv'].sum() / g5['staked'].sum():+6.2f}% {100 * g5['result'].sum() / g5['staked'].sum():+6.2f}%")


if __name__ == "__main__":
    main()
