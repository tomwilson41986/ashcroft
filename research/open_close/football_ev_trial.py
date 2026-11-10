"""The expected-value rule's paper trader, at the recorded Betfair books (reports/preregistration_football_ev.md).

racing² football research, 4 Nov 2026 (the owner: "set up a paper trader for your recommended approach to maximise
profit"). The rule (reports/football_model_rethink.md, "Expected-value rule, trade types and staking"):

1. **Signals.** Frozen stacks (models/football/ev_v1, hashes checked) forecast each match's home and away chance move
   from the collection moment (Friday 17:00 / Tuesday 13:00 UK) to the close, on the team features plus home
   advantage, team news and short form. The frozen calibration turns the forecast into an expected move.
2. **Selection.** Each side's expected value at Betfair's early book (football-data's collection snapshot): the
   expected CLV at fair less 2% commission on a gain, less the round trip a tick off fair at each end. Every side
   whose expected value is above 2% is traded the better way round: back first (the chance is expected to rise) or
   lay first.
3. **Entry: resting orders**, as the resting trial (football_stack_resting): posted a tick inside the spread at the
   first recorded book within 60 minutes after the collection moment, filled only when a later recorded book proves
   it, cancelled 90 minutes before the off.
4. **Exit.** Greened at the last two-sided recorded book within 30 minutes of the off; 2% commission on each market's
   net.
5. **Stakes.** A paper bank of GBP1,000. Each collection window, 25% of the bank is shared among the window's signals
   in proportion to their expected value, at most 10% of the bank on one; the capital is a back's stake or a lay's
   liability (stake x (price - 1)). An unfilled order's capital comes back. The bank moves window by window.

The headline is the return on the capital staked (and the bank); the crossing variant (taking the best price at the
early book) is reported beside it so the value of resting is measured.

    python -m research.open_close.football_ev_trial freeze --cache <lab cache> --out models/football/ev_v1
    python -m research.open_close.football_ev_trial score --cache <lab cache> --root <sources> --out-json out/ev.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SPEC = "team_all+home_adv+team_news+form_short+x_short"
PREREG = "reports/preregistration_football_ev.md"
MODELS = "models/football/ev_v1"
TRAIN_FROM = 2013
CAL_SEASON = 2025                 # calibration on the 2024/25 out-of-sample forecasts (fitted on 2013-2023)
GATE = {"min_bets": 300, "min_days": 90, "roi_lo95_above": 0, "clv_mean_above": 0}


# ---------------------------------------------------------------------------
# freeze and predict
# ---------------------------------------------------------------------------

def freeze(cache: str, out: str = MODELS) -> dict:
    """Fit home and away stacks on every same-book move from TRAIN_FROM, and the calibration on the latest season's
    out-of-sample forecasts; write both with their SHA-256."""
    import lightgbm as lgb
    from football import lab
    from research.open_close import football_improve as fi
    d, cols = fi.load(cache, SPEC)
    tr = d[d["same_book"] & (d["season"] >= TRAIN_FROM)]
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    X = tr[cols].to_numpy(float)
    keep = np.mean(np.isfinite(X), axis=0) >= 0.5
    kept = [c for c, k in zip(cols, keep) if k]
    mu, sd = np.nanmean(X[:, keep], 0), np.nanstd(X[:, keep], 0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    Z = np.clip(np.nan_to_num((X[:, keep] - mu) / sd), -lab.CLIP, lab.CLIP)
    spec = {"spec": SPEC, "features": cols, "ridge_features": kept, "mu": mu.tolist(), "sd": sd.tolist(),
            "clip": lab.CLIP, "ridge_alpha": lab.RIDGE_ALPHA, "train_from": TRAIN_FROM,
            "train_to_date": str(pd.to_datetime(tr["date"]).max().date()), "rows": int(len(tr)),
            "leagues": sorted(tr["fd_code"].astype(str).unique()), "sides": {}}
    for side, ycol in (("h", "y_h"), ("a", "y_a")):
        t = tr[tr[ycol].notna()]
        Zs = Z[tr[ycol].notna().to_numpy()]
        y = t[ycol].to_numpy(float)
        beta = np.linalg.solve(Zs.T @ Zs + lab.RIDGE_ALPHA * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
        g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        Xg = t[cols + ["league"]].copy()
        Xg["league"] = pd.Categorical(Xg["league"].astype(str), categories=spec["leagues"])
        g.fit(Xg, y)
        g.booster_.save_model(str(o / f"gbm_{side}.txt"))
        spec["sides"][side] = {"ridge_intercept": float(y.mean()), "ridge_beta": beta.tolist(), "gbm": f"gbm_{side}.txt"}
    (o / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    cal = fi.calibration(d, cols, CAL_SEASON)
    (o / "calibration.json").write_text(json.dumps(cal, indent=1) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(o.iterdir())
              if p.is_file() and p.name != "SHA256SUMS"}
    (o / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in hashes.items()))
    return {"out": str(o), "rows": spec["rows"], "features": len(cols), "ridge_features": len(kept),
            "calibration": cal, "sha256": hashes}


def predict(models: str, d: pd.DataFrame) -> pd.DataFrame:
    """Each match's forecast chance moves (home, away): the mean of the frozen ridge and LightGBM."""
    import lightgbm as lgb
    o = Path(models)
    spec = json.loads((o / "spec.json").read_text())
    X = d[spec["ridge_features"]].to_numpy(float)
    Z = np.clip(np.nan_to_num((X - np.array(spec["mu"])) / np.array(spec["sd"])), -spec["clip"], spec["clip"])
    Xg = d[spec["features"] + ["league"]].copy()
    Xg["league"] = pd.Categorical(Xg["league"].astype(str), categories=spec["leagues"])
    out = pd.DataFrame({"match_id": d["match_id"].to_numpy()})
    for side, s in spec["sides"].items():
        ridge = s["ridge_intercept"] + Z @ np.array(s["ridge_beta"])
        gbm = lgb.Booster(model_file=str(o / s["gbm"])).predict(Xg)
        out[f"chance_{side}"] = (ridge + gbm) / 2
    return out


# ---------------------------------------------------------------------------
# signals and stakes
# ---------------------------------------------------------------------------

def expected_value(pred, cal: dict, fair_open, commission: float) -> tuple[np.ndarray, np.ndarray]:
    """(side, expected value) a side: back first gains exp(E[y]) - 1, lay first exp(-E[y]) - 1, commission off a gain,
    less the round trip a tick off fair at each end at the early fair price."""
    from research.open_close.backtest import tick_over, tick_under
    ey = cal["a"] + cal["b"] * np.asarray(pred, float)
    fo = np.asarray(fair_open, float)
    ok = np.isfinite(fo) & (fo > 1)
    under = np.array([tick_under(x) if g else np.nan for x, g in zip(fo, ok)])
    over = np.array([tick_over(x) if g else np.nan for x, g in zip(fo, ok)])
    net = lambda g: np.where(g > 0, g * (1 - commission), g)          # noqa: E731
    evb = net(np.expm1(ey)) - (1 - under / over)
    evl = net(np.expm1(-ey)) - (over / under - 1)
    side = np.where(evb >= evl, 1, -1)
    return side, np.where(ok, np.maximum(evb, evl), np.nan)


def signals(d: pd.DataFrame, pred: pd.DataFrame, cal: dict, margin: float, commission: float) -> pd.DataFrame:
    """The scorer's signal rows: one a side; ``side`` +1 back first, -1 lay first, 0 no trade."""
    from research.open_close import football_paper_trial as pt
    from research.open_close.football_exchange import width_label
    from research.open_close.football_stack_trial import TOP5
    m = d.merge(pred, on="match_id")
    o = m[["bfe_open_h", "bfe_open_d", "bfe_open_a"]].to_numpy(float)
    over = (1 / o).sum(axis=1)
    rows = []
    for leg, k, col in (("H", "h", "bfe_open_h"), ("A", "a", "bfe_open_a")):
        side, ev = expected_value(m[f"chance_{k}"], cal, m[col].to_numpy(float) * over, commission)
        trade = np.isfinite(ev) & (ev > margin)
        rows.append(pd.DataFrame({
            "event_id": m["match_id"].to_numpy(), "market_type": "MATCH_ODDS", "selection_id": leg,
            "yhat": -m[f"chance_{k}"].to_numpy(float), "ev": ev, "side": np.where(trade, side, 0),
            "fd_code": m["fd_code"].astype(str).to_numpy(),
            "tier": np.where(m["fd_code"].isin(TOP5), "top five", "rest"),
            "match_day": m["match_date"].astype(str).to_numpy(), "book_width": width_label(over),
            "fit_month": "frozen"}))
    s = pd.concat(rows, ignore_index=True)
    coll = [pt.collection_of(x) for x in s["match_day"]]
    s["collection"] = [c.isoformat() for c, _ in coll]
    s["moment_utc"] = [mo for _, mo in coll]
    return s


def capital_shares(ev: np.ndarray, budget: float, cap: float) -> np.ndarray:
    """A window's capital shares of the bank: in proportion to expected value, ``budget`` in all, ``cap`` at most each
    (what a capped signal cannot take goes to the others, until none is left or all are capped)."""
    w = np.clip(np.asarray(ev, float), 0, None)
    f = np.zeros(len(w))
    free, left = w > 0, budget
    while free.any() and left > 1e-12:
        add = left * w[free] / w[free].sum()
        f[free] += add
        over = f > cap
        left = float((f[over] - cap).sum())
        f[over] = cap
        free = free & ~over
    return f


def bank_path(sig: pd.DataFrame, trades: pd.DataFrame, rule: dict) -> tuple[pd.DataFrame, dict]:
    """The paper bank window by window: each signal's capital share by expected value (unfilled orders' capital comes
    back), each trade's stake from it (a back's stake, a lay's stake = capital / (price - 1)), the greened profit
    after 2% commission on each market's net."""
    st = rule["staking"]
    bank, rows, trows = float(st["bank_gbp"]), [], []
    s = sig[sig["side"] != 0]
    key = ["event_id", "selection_id"]
    tr = trades.set_index(["match_id", "selection"]) if len(trades) else None
    for w, g in s.groupby("collection", sort=True):
        f = capital_shares(g["ev"].to_numpy(float), st["window_budget"], st["max_per_trade"])
        pnl = 0.0
        for share, r in zip(f, g[key].itertuples(index=False)):
            if tr is None or (r.event_id, r.selection_id) not in tr.index:
                continue
            t = tr.loc[(r.event_id, r.selection_id)]
            capital = bank * share
            stake = capital if t["side"] > 0 else capital / max(t["fill_open"] - 1, 1e-9)
            gross = stake * t["profit"]
            net = gross * (1 - rule["commission"]) if gross > 0 else gross
            pnl += net
            trows.append({"collection": w, "match_id": r.event_id, "selection": r.selection_id,
                          "side": int(t["side"]), "capital_gbp": round(capital, 2), "stake_gbp": round(stake, 2),
                          "fill_open": float(t["fill_open"]), "fill_close": float(t["fill_close"]),
                          "pnl_gbp": round(net, 2), "clv": float(t["clv"])})
        rows.append({"collection": w, "signals": int(len(g)), "bank_start": round(bank, 2), "pnl_gbp": round(pnl, 2)})
        bank += pnl
    path = pd.DataFrame(rows)
    staked = pd.DataFrame(trows)
    b = np.r_[float(st["bank_gbp"]), (path["bank_start"] + path["pnl_gbp"]).to_numpy()] if len(path) else np.array([bank])
    return staked, {"bank_start_gbp": float(st["bank_gbp"]), "bank_now_gbp": round(bank, 2),
                    "pnl_gbp": round(bank - float(st["bank_gbp"]), 2),
                    "capital_staked_gbp": round(float(staked["capital_gbp"].sum()), 2) if len(staked) else 0.0,
                    "return_on_capital": (round(float(staked["pnl_gbp"].sum() / staked["capital_gbp"].sum()), 5)
                                          if len(staked) and staked["capital_gbp"].sum() > 0 else None),
                    "max_drawdown": round(float(np.max(1 - b / np.maximum.accumulate(b))), 4),
                    "windows": int(len(path))}


# ---------------------------------------------------------------------------
# the forward score
# ---------------------------------------------------------------------------

def score(cache: str, models: str, root: str | None, live_root: str | None, start: str | None,
          as_of: str | None, prereg: str = PREREG, n_boot: int = 2000) -> dict:
    from football.data import load_gold
    from football.reference import load_aliases
    from research.open_close import football_improve as fi
    from research.open_close import football_paper_trial as pt
    from research.open_close import football_stack_resting as rs
    from research.open_close import football_stack_trial as stt
    from research.open_close.football_exchange import read_prereg
    from sources.common import Store
    rule, digest = read_prereg(prereg)
    stt.verify_models(models, rule["model_sha256"])
    cal = json.loads((Path(models) / "calibration.json").read_text())
    read_day = pt.recorder_reader(live_root)
    as_of_d = date.fromisoformat(as_of) if as_of else date.today() - timedelta(days=1)
    if start:
        start_d = date.fromisoformat(start)
    else:
        first = pt.first_recorded_day(read_day, max(date.fromisoformat(pt.RECORDER_FROM),
                                                    date.fromisoformat(rule["written_utc"][:10])), as_of_d)
        if first is None:
            return {"status": "no football book record found", "prereg_sha256": digest, "rule": rule}
        start_d = pt.trial_start(first)
    if start_d > as_of_d:                  # the record has begun, the trial's first collection has not
        return {"status": "not started", "start": start_d.isoformat(), "as_of": as_of_d.isoformat(),
                "prereg_sha256": digest, "rule": rule}
    d, _cols = fi.load(cache, SPEC)
    d = d[(d["match_date"] >= start_d.isoformat()) & (d["match_date"] <= as_of_d.isoformat())
          & d["fd_code"].isin(rule["leagues_fd"])]
    if d.empty:
        return {"status": "no matches in the window yet", "start": start_d.isoformat(), "as_of": as_of_d.isoformat(),
                "prereg_sha256": digest, "rule": rule}
    sig = signals(d, predict(models, d), cal, rule["ev_margin"], rule["commission"])
    store = Store(root=root)
    t = SimpleNamespace(gold=load_gold(store), read_day=read_day, tolerance_min=pt.TOLERANCE_MIN,
                        notional=pt.NOTIONAL_GBP,
                        aliases=load_aliases(extra=store.get_parquet("football/gold/team_aliases.parquet")))
    any_sig = bool((sig["side"] != 0).any())
    tr, sk = rs.trade_resting(t, sig, rule["book_width"]) if any_sig else (pd.DataFrame(), pd.DataFrame())
    cr, ck = pt.trade_signals(t, sig, rule["book_width"]) if any_sig else (pd.DataFrame(), pd.DataFrame())
    w = pt.window(tr, start_d, as_of_d, GATE["min_bets"], GATE["min_days"])
    inw = tr[pd.to_datetime(tr["day"]).dt.date <= date.fromisoformat(w["end"])] if len(tr) else tr
    head = pt.scored(inw, rule["commission"], n_boot)
    if len(inw):
        head["roi_lo95"], head["roi_hi95"] = rs.interval95(inw, rule["commission"], n_boot)
    staked, bank = bank_path(sig, inw, rule)
    res = {"prereg": prereg, "prereg_sha256": digest, "rule": rule, "calibration": cal, "start": start_d.isoformat(),
           "as_of": as_of_d.isoformat(), "forecasts": int(len(sig)), "signals": int((sig["side"] != 0).sum()),
           "window": w, "headline": head, "bank": bank, "adverse_selection": rs.adverse_selection(tr, sk),
           "crossing_variant": pt.scored(cr, rule["commission"], 0),
           "skipped": sk["reason"].value_counts().to_dict() if len(sk) else {},
           "staked_trades": staked.to_dict("records")[-200:]}
    res["halves"] = (pt.halves(inw, start_d, date.fromisoformat(w["end"]), rule["commission"]) if len(inw)
                     else {"first": {"bets": 0}, "second": {"bets": 0}})
    for key in ("book_width", "band", "tier", "fd_code", "month"):
        res[f"by_{key}"] = pt.by_key(inw, key, rule["commission"])
    res["overfit"] = pt.overfit(inw, rule["commission"], pt.trials_considered(rule, GATE))
    res["gate"] = rs.gate(res, GATE)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--cache", required=True)
    f.add_argument("--out", default=MODELS)
    c = sub.add_parser("score")
    c.add_argument("--cache", required=True)
    c.add_argument("--models", default=MODELS)
    c.add_argument("--root", default=None)
    c.add_argument("--live-root", default=None)
    c.add_argument("--start", default=None)
    c.add_argument("--as-of", default=None)
    c.add_argument("--prereg", default=PREREG)
    for p in (f, c):
        p.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    res = (freeze(a.cache, a.out) if a.cmd == "freeze"
           else score(a.cache, a.models, a.root, a.live_root, a.start, a.as_of, a.prereg))
    text = json.dumps(res, indent=1, default=str)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(text + "\n")
    print(text[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
