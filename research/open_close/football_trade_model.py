"""Two trading targets on the football warehouse (the owner, 12 Nov 2026), read as registered in the ledger before
the run (football-tradeout-bsp-1112, football-win-bets-1112). Nothing here trades.

**Trade out at the SP.** Back a match-odds runner at a checkpoint (360, 120, 60, 15 minutes before the scheduled
kick-off) where the closing-price model forecasts its chance to rise, and lay it off at the Betfair SP. The hedged
result a unit staked is price / SP - 1, whatever the match does. The SP is a real fill with no spread; the entry is a
last traded price, not a quote, so half the measured round trip is charged on it. Betfair reconciles an SP on few
football markets (in 2026: 5,539 match-odds runners, the top liquidity tier), so the headline is on those; the same
rule closed at the last pre-off traded price, with the full round trip, is shown on every row beside it.

**Win bets.** A model of the result started from the market's own chance (LightGBM on ``won`` with the checkpoint's
log odds as the starting score: model plus market). Its log loss against the market's on the same rows says whether
it knows anything the price does not; bets are backs where model chance x price - 1 clears a threshold, settled on
the result.

Both walk forward by month as the closing-price model does (a month forecast from matches that kicked off at least
a day before it began). Returns are a unit staked with intervals from resampled match days; each headline carries its
deflated Sharpe on daily returns with every football trial in the ledger counted. One year of data.

    python -m research.open_close.football_trade_model --out-json out/football_trade_model.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from research.open_close import football_bf_close_model as cm

log = logging.getLogger(__name__)

ENTRY = cm.ROUND_TRIP / 2          # the entry's share of the measured round trip (the SP has no spread)
MOVE_THRESHOLDS = (0.03, 0.05, 0.08)
EDGE_THRESHOLDS = (0.02, 0.05, 0.10)
HEADLINE_MOVE, HEADLINE_EDGE = 0.05, 0.05
MAX_PRICE = 20.0
WIN_PARAMS = dict(n_estimators=150, learning_rate=0.02, num_leaves=7, min_child_samples=500, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)


def net(cash: np.ndarray) -> np.ndarray:
    """Commission on a win."""
    return np.where(cash > 0, cash * (1 - cm.COMMISSION), cash)


def tradeout_cash(price, exit_price, cost: float) -> np.ndarray:
    """A back of one unit at ``price`` (worse by ``cost`` in log price) laid off at ``exit_price``: the same result
    whatever the match does, before commission."""
    return np.asarray(price, float) * np.exp(-cost) / np.asarray(exit_price, float) - 1.0


def win_cash(price, won, cost: float) -> np.ndarray:
    """A back of one unit at ``price`` (worse by ``cost`` in log price) left to the result, before commission."""
    p = np.asarray(price, float) * np.exp(-cost)
    return np.where(np.asarray(won, float) > 0, p - 1.0, -1.0)


def summary(day, cash, n_trials: int | None = None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bets, the mean return a unit staked with a 90% interval from resampled days, the share that won, and (with
    ``n_trials``) the deflated Sharpe of the daily returns."""
    cash = np.asarray(cash, float)
    if len(cash) < 30:
        return {"bets": int(len(cash))}
    f = pd.DataFrame({"day": np.asarray(day), "c": cash, "n": 1.0}).groupby("day").sum()
    idx = np.random.default_rng(seed).integers(0, len(f), size=(n_boot, len(f)))
    boot = f["c"].to_numpy()[idx].sum(1) / f["n"].to_numpy()[idx].sum(1)
    out = {"bets": int(len(cash)), "days": int(len(f)), "return": float(cash.mean()),
           "lo90": float(np.quantile(boot, 0.05)), "hi90": float(np.quantile(boot, 0.95)),
           "won_share": float((cash > 0).mean())}
    if n_trials:
        from research.overfit import dsr_check
        chk = dsr_check((f["c"] / f["n"]).to_numpy(), n_trials=n_trials)
        out["deflated_sharpe"] = {k: chk.get(k) for k in ("n_obs", "n_trials", "sharpe", "dsr", "passed", "reason")}
    return out


def win_walk_forward(d: pd.DataFrame, cols: list[str], min_train_months: int = 3) -> np.ndarray:
    """The model's chance of each runner winning, out of sample (blank before the fourth month and where the
    result is not known): the market's log odds plus what LightGBM adds to them, one model a role."""
    import lightgbm as lgb
    pred = np.full(len(d), np.nan)
    months = sorted(d["month"].unique())
    known = d["won"].notna() & np.isfinite(d["x_cp"])
    for m in months[min_train_months:]:
        start = pd.Timestamp(m + "-01", tz="UTC")
        for role in cm.ROLES:
            r = (d["role"] == role) & known
            tr = d[r & (d["kickoff"] < start - pd.Timedelta(days=1))]
            te_idx = np.flatnonzero((r & (d["month"] == m)).to_numpy())
            if len(tr) < 2000 or not len(te_idx) or tr["won"].nunique() < 2:
                continue
            te = d.iloc[te_idx]
            g = lgb.LGBMClassifier(n_jobs=cm.THREADS, **WIN_PARAMS)
            g.fit(tr[cols], tr["won"].astype(int), init_score=tr["x_cp"].to_numpy(float))
            raw = g.predict(te[cols], raw_score=True) + te["x_cp"].to_numpy(float)
            pred[te_idx] = 1.0 / (1.0 + np.exp(-raw))
    return pred


def log_loss_read(day, won, p_model, p_market, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean log loss of the model and of the market on the same rows; the difference with a 95% day interval."""
    won, pm, pk = (np.asarray(x, float) for x in (won, p_model, p_market))
    ll = lambda p: -(won * np.log(np.clip(p, 1e-6, 1)) + (1 - won) * np.log(np.clip(1 - p, 1e-6, 1)))   # noqa: E731
    f = pd.DataFrame({"day": np.asarray(day), "d": ll(pm) - ll(pk), "n": 1.0}).groupby("day").sum()
    idx = np.random.default_rng(seed).integers(0, len(f), size=(n_boot, len(f)))
    boot = f["d"].to_numpy()[idx].sum(1) / f["n"].to_numpy()[idx].sum(1)
    return {"rows": int(len(won)), "model": float(ll(pm).mean()), "market": float(ll(pk).mean()),
            "model_less_market": float((ll(pm) - ll(pk)).mean()),
            "lo95": float(np.quantile(boot, 0.025)), "hi95": float(np.quantile(boot, 0.975))}


def football_trials() -> int | None:
    try:
        from desk.trials import LEDGER, count_trials, parse
        return count_trials(parse(LEDGER.read_text())[0], "football")
    except Exception:                                   # the ledger is not there: no deflated Sharpe
        return None


def run(store) -> dict:
    d, fam = cm.load(store)
    if d.empty:
        return {"status": "no data found: gold/move is not built"}
    d = d[d["cp_fresh"] & d["schedule_fixed"]].reset_index(drop=True)
    if d["month"].nunique() < 4:
        return {"status": "no data found: fewer than four months"}
    r = cm.read_rows(store, "silver/runner", ["market_id", "selection_id", "handicap", "bsp"]).to_pandas()
    d = d.merge(r.drop_duplicates(["market_id", "selection_id", "handicap"]), on=["market_id", "selection_id",
                                                                                 "handicap"], how="left")
    cols = [c for v in fam.values() for c in v]
    trials = football_trials()
    res = {"rows": int(len(d)), "rows_with_sp": int(d["bsp"].notna().sum()), "football_trials_counted": trials,
           "cost": {"entry": ENTRY, "round_trip": cm.ROUND_TRIP, "commission": cm.COMMISSION},
           "note": "development evidence unless the registered reading is met: one year of data; entries are last "
                   "traded prices; nothing is traded"}
    d["band"] = pd.cut(d["n_ticks_before"], cm.ACTIVITY[0], labels=cm.ACTIVITY[1], right=False).astype(str)

    # ---- trade out: the closing-price model's forecast, backs only ------------------------------------------------
    move, _ = cm.walk_forward(d, cols)
    t = d.assign(pred=move)[np.isfinite(move)]
    sp = t[t["bsp"].notna() & (t["bsp"] > 1)]
    close = t[t["usable"]]
    at_sp = lambda s: net(tradeout_cash(s["ltp_cp"], s["bsp"], ENTRY))                       # noqa: E731
    at_close = lambda s: net(tradeout_cash(s["ltp_cp"], s["ltp_close"], cm.ROUND_TRIP))      # noqa: E731
    res["trade_out"] = {
        "test_months": sorted(t["month"].unique()),
        "at_the_sp": {
            "rows": int(len(sp)), "matches": int(sp["event_id"].nunique()),
            "every_row_backed": summary(sp["day"], at_sp(sp)),
            "by_threshold": {str(th): summary(s["day"], at_sp(s), trials if th == HEADLINE_MOVE else None)
                             for th in MOVE_THRESHOLDS for s in [sp[sp["pred"] >= th]]},
            "by_checkpoint_at_headline": {str(cp): summary(s["day"], at_sp(s)) for cp in cm.HORIZONS
                                          for s in [sp[(sp["pred"] >= HEADLINE_MOVE) & (sp["cp_min"] == cp)]]},
            "sp_against_the_last_traded_price": {
                "median_log": float(np.median(np.log(sp["bsp"] / sp["ltp_close"]).dropna())),
                "sd_log": float(np.log(sp["bsp"] / sp["ltp_close"]).dropna().std())}},
        "at_the_last_traded_price": {
            "rows": int(len(close)), "every_row_backed": summary(close["day"], at_close(close)),
            "by_threshold": {str(th): summary(s["day"], at_close(s)) for th in MOVE_THRESHOLDS
                             for s in [close[close["pred"] >= th]]},
            "by_price_updates_at_headline": {str(k): summary(s["day"], at_close(s)) for k, s in
                                             close[close["pred"] >= HEADLINE_MOVE].groupby("band")}}}
    log.info("trade out at the SP: %s", res["trade_out"]["at_the_sp"]["by_threshold"])

    # ---- win bets: model plus market -----------------------------------------------------------------------------
    p = win_walk_forward(d, cols)
    w = d.assign(p_model=p)[np.isfinite(p)]
    w = w.assign(edge=w["p_model"] * w["ltp_cp"] * np.exp(-ENTRY) - 1.0)
    ok = w[w["ltp_cp"] <= MAX_PRICE]
    cash = lambda s: net(win_cash(s["ltp_cp"], s["won"], ENTRY))                             # noqa: E731
    res["win_bets"] = {
        "test_months": sorted(w["month"].unique()),
        "log_loss": log_loss_read(w["day"], w["won"], w["p_model"], w["p_cp"]),
        "log_loss_by_role": {k: log_loss_read(s["day"], s["won"], s["p_model"], s["p_cp"])
                             for k, s in w.groupby("role")},
        "every_row_backed": summary(ok["day"], cash(ok)),
        "by_threshold": {str(th): summary(s["day"], cash(s), trials if th == HEADLINE_EDGE else None)
                         for th in EDGE_THRESHOLDS for s in [ok[ok["edge"] >= th]]},
        "by_checkpoint_at_headline": {str(cp): summary(s["day"], cash(s)) for cp in cm.HORIZONS
                                      for s in [ok[(ok["edge"] >= HEADLINE_EDGE) & (ok["cp_min"] == cp)]]},
        "by_price_updates_at_headline": {str(k): summary(s["day"], cash(s)) for k, s in
                                         ok[ok["edge"] >= HEADLINE_EDGE].groupby("band")},
        "by_role_at_headline": {k: summary(s["day"], cash(s)) for k, s in
                                ok[ok["edge"] >= HEADLINE_EDGE].groupby("role")}}
    log.info("win bets: %s", res["win_bets"]["by_threshold"])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="a local warehouse folder in place of the capture bucket")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from sources.common import Store
    res = run(Store(root=a.root) if a.root else Store())
    with open(a.out_json, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
