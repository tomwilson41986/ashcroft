"""The closing-price lab: a self-improving forecast of each dog's Betfair SP from the recorded Exchange Stream
(read-only research; the owner, 6 Nov 2026: "let's review and optimise our model ... now we have the market data?
Let's start recursive self improvement"; earlier: "fundamental is always just to get the most accurate closing price").

The data: every 10 s over the last ``WINDOW`` seconds before each scheduled off, each runner's book as it stood
(rebuilt from the stream by research.greyhound.stream_replay): three levels each side and their money, the money
offered in all, the last traded price, Betfair's projected SPs (near, far), the money traded so far and in the last
30 and 60 s; the race's overround and money; the same dog's place market; our pre-race price
(greyhound/live/predictions/<day>.csv).

Two targets, always (the owner, 6 Nov 2026: "let's always test against two targets; 1. BFSP 2. Win - winning the
bet"): every spec and baseline is scored on both.
- ``bfsp``: the RMSE of log(SP), the SP from the settled definition;
- ``win``: the result. Each forecast is read as win chances (1 / price, normalised within the race at each
  checkpoint) and scored by the winner's log loss (-log of the chance it gave the winner), over races with one winner.
A spec says which target it is fitted to (``"target": "win"`` learns the result, starting from the market's chances;
the default learns log(SP)), and is scored on both all the same.

The loop that keeps it honest:
- every experiment is a line in ``research/greyhound/close_specs.json`` (a name, its features, its model, the date it
  was registered); the night's job (greyhound-close-lab.yml) builds the new day's snapshots, fits every spec on the
  days before each test day only, scores every day, and appends each spec's score on each day to the trial registry
  (s3 close_lab/registry.csv, with the code's commit);
- a spec counts only on days after the day it was registered (``fresh`` days): each target has its own champion,
  the spec with the lowest score on that target over its fresh days, replacing the incumbent only when its paired gain
  over the incumbent on the days both were fresh has a 90% interval (bootstrap over races) above zero;
- the baselines are always scored: the market mid, the last traded price and Betfair's own projected SP.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.greyhound import multitest as mt, stream_replay as sr

log = logging.getLogger(__name__)

WINDOW = 300
STEP = 10
CHECKS = tuple(range(WINDOW, 0, -STEP))
# The review of 10 Nov 2026 (reports/greyhound_model_review.md): the SP is struck at the real off, and the checkpoints
# above stop at the scheduled one. The late checkpoints read the book every 10 s after the scheduled off for as long as
# the market is still open (0 is the scheduled off itself). The headline scores stay on CHECKS, so every night's
# registry rows compare; the late rows are their own window, and only a spec with ``"late": true`` learns from them.
LATE = 180
LATE_CHECKS = tuple(range(0, -LATE - 1, -STEP))
ALL_CHECKS = CHECKS + LATE_CHECKS
FRAMES = "frames_v2"                 # the cached frames with the late checkpoints and the review's columns
SPECS = Path(__file__).with_name("close_specs.json")
PREFIX = "close_lab"
BASELINES = ("mid", "ltp", "spn")
TARGETS = {"bfsp": ("rmse", "e_"), "win": ("win_ll", "w_")}        # target: (its score, its error columns' prefix)


def _mid(b, bs, l, ls):
    if b and l and l / b < 1.2:
        return (b * ls + l * bs) / (bs + ls) if (bs + ls) > 0 else (b + l) / 2
    return b or l


def snapshots(lines) -> pd.DataFrame:
    """Every runner of every WIN and PLACE market at each checkpoint (the book as it stood at off - c)."""
    markets: dict = {}
    nxt: dict = {}
    rows = []

    def take(m, c):
        if m.inplay or str(m.defn.get("status", "OPEN")) != "OPEN":
            return
        for sid, r in m.runners.items():
            bk = sorted(r.atb.items(), reverse=True)[:3]
            ly = sorted(r.atl.items())[:3]
            row = {"market": m.id, "sid": sid, "check": c, "ltp": r.ltp, "spn": r.spn, "spf": r.spf, "tv": r.tv,
                   "atb_total": sum(r.atb.values()), "atl_total": sum(r.atl.values()),
                   "spb_total": sum(r.spb.values()), "spl_total": sum(r.spl.values()),
                   "trd_v": sum(r.trd.values()), "trd_pv": sum(pp * vv for pp, vv in r.trd.items())}
            for i in range(3):
                row[f"b{i+1}"], row[f"bs{i+1}"] = bk[i] if i < len(bk) else (np.nan, 0.0)
                row[f"l{i+1}"], row[f"ls{i+1}"] = ly[i] if i < len(ly) else (np.nan, 0.0)
            row["mid"] = _mid(row["b1"] if np.isfinite(row["b1"]) else None, row["bs1"],
                              row["l1"] if np.isfinite(row["l1"]) else None, row["ls1"])
            rows.append(row)

    import heapq
    due: list = []                     # (checkpoint time, market, checkpoint): every market's marks in time order
    closed: dict = {}                  # market -> when it last left OPEN (suspended at the off, in play or closed)
    for t, mc in sr.events(lines):
        # every mark passed by now, whatever market this message is for: a quiet market keeps its book until it changes
        while due and due[0][0] <= t:
            _, mid0, c = heapq.heappop(due)
            take(markets[mid0], c)
        mid = str(mc.get("id"))
        sr.apply(markets, mc)
        m = markets[mid]
        if m.inplay or str(m.defn.get("status", "OPEN")) != "OPEN":
            closed.setdefault(mid, t)
        else:
            closed.pop(mid, None)
        if mid not in nxt and m.off is not None:
            nxt[mid] = 1
            for c in ALL_CHECKS:
                if t < m.off - timedelta(seconds=c):
                    heapq.heappush(due, (m.off - timedelta(seconds=c), mid, c))
    snap = pd.DataFrame(rows)
    if not len(snap):
        return snap
    meta = []
    for mid, m in markets.items():
        d = m.defn
        for r in d.get("runners", []) or []:
            meta.append({"market": mid, "sid": int(r.get("id", -1)), "bsp": r.get("bsp"), "status": r.get("status"),
                         "mtype": d.get("marketType"), "event": d.get("eventId"), "venue": d.get("venue"),
                         "off": m.off, "n_winners": d.get("numberOfWinners"), "trap": r.get("sortPriority"),
                         "t_closed": closed.get(mid)})
    meta = pd.DataFrame(meta)
    out = snap.merge(meta, on=["market", "sid"], how="left")
    out["bsp"] = pd.to_numeric(out.bsp, errors="coerce")
    return out


def frame(snap: pd.DataFrame, preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """The WIN runners with every feature: the book, its flow, the race, the place market, our pre-race price."""
    if not len(snap):
        return snap
    s = snap.copy()
    s = s[s.status.astype(str) != "REMOVED"]
    w = s[s.mtype.astype(str) == "WIN"].copy()
    p = s[s.mtype.astype(str) == "PLACE"][["event", "off", "sid", "check", "mid", "spn"]]
    # the same dog in the place market: Betfair keeps a runner's selection id across a race's markets
    w = w.merge(p.rename(columns={"mid": "p_mid", "spn": "p_spn"}), on=["event", "off", "sid", "check"], how="left")
    w = w.sort_values(["market", "sid", "check"], ascending=[True, True, False])
    g = w.groupby(["market", "sid"])
    w["tv"] = pd.to_numeric(w.tv, errors="coerce")
    for lag in (3, 6):                                          # 30 and 60 s earlier (checkpoints 10 s apart)
        w[f"tv_d{lag * STEP}"] = w.tv - g.tv.shift(lag)
        w[f"mid_ch{lag * STEP}"] = np.log(w["mid"] / g["mid"].shift(lag))
    lg = lambda c: np.log(pd.to_numeric(w[c], errors="coerce").where(lambda v: v > 1))
    w["l_mid"], w["l_ltp"], w["l_spn"], w["l_spf"] = lg("mid"), lg("ltp"), lg("spn"), lg("spf")
    w["l_b1"], w["l_l1"] = lg("b1"), lg("l1")
    w["spread"] = w.l_l1 - w.l_b1
    w["wom1"] = w.bs1 / (w.bs1 + w.ls1).replace(0, np.nan)
    w["wom3"] = (w.bs1 + w.bs2 + w.bs3) / (w.bs1 + w.bs2 + w.bs3 + w.ls1 + w.ls2 + w.ls3).replace(0, np.nan)
    w["depth_imb"] = np.log1p(w.atb_total) - np.log1p(w.atl_total)
    w["spn_vs_mid"] = w.l_spn - w.l_mid
    w["spf_vs_mid"] = w.l_spf - w.l_mid
    w["ltp_vs_mid"] = w.l_ltp - w.l_mid
    rc = w.groupby(["market", "check"])
    w["overround"] = rc["mid"].transform(lambda x: (1 / x).sum())
    w["race_tv"] = np.log1p(rc.tv.transform("sum"))
    w["log_tv"] = np.log1p(w.tv)
    w["field"] = rc.sid.transform("size")
    w["rank"] = rc["mid"].rank(method="first")
    w["q_mid"] = (1 / w["mid"]) / rc["mid"].transform(lambda x: (1 / x).sum())
    w["p_vs_win"] = np.log((1 / w.p_mid.where(w.p_mid > 1)).clip(1e-6)) - np.log(w.q_mid.clip(1e-6))
    w["secs"] = w.check.astype(float)
    w = review_columns(w)
    if preds is not None and len(preds):
        pr = preds[["market_id", "selection_id", "p_model"]].copy()
        pr = pr.assign(market=pr.market_id.astype(str), sid=pd.to_numeric(pr.selection_id, errors="coerce"))
        w = w.merge(pr[["market", "sid", "p_model"]].drop_duplicates(["market", "sid"]), on=["market", "sid"],
                    how="left")
        w["model_vs_mid"] = -np.log(w.p_model.clip(1e-6)) - w.l_mid
    else:
        w["p_model"], w["model_vs_mid"] = np.nan, np.nan
    w["y"] = np.log(w.bsp.where(w.bsp > 1))
    return w[w.y.notna() & w.l_mid.notna()].reset_index(drop=True)


def review_columns(w: pd.DataFrame) -> pd.DataFrame:
    """The review's columns (10 Nov 2026), each read off the book at or before its checkpoint: the race around the
    runner (``race``), the path of the price and of the trades (``path``), and the money waiting for the SP
    (``sppool``). ``w`` is sorted by market, runner and checkpoint (earliest first)."""
    for c in ("trd_pv", "trd_v", "spb_total", "spl_total", "trap"):
        if c not in w:
            w[c] = np.nan
    nz = lambda v: v.replace(0, np.nan)
    key = [w.market, w.check]
    run = [w.market, w.sid]
    # the race around the runner
    w["tv_share"] = w.tv / nz(w.tv.groupby(key).transform("sum"))
    has = w.mid_ch60.notna()
    race_ch = (w.q_mid * w.mid_ch60).groupby(key).transform("sum") / nz(w.q_mid.where(has).groupby(key).transform("sum"))
    w["mid_ch60_rel"] = w.mid_ch60 - race_ch                    # the runner's move less the race's (chance-weighted)
    d60 = w.tv_d60.clip(lower=0)
    w["tv_d60_share"] = d60 / nz(d60.groupby(key).transform("sum"))
    w["fav_ch60"] = w.mid_ch60.where(w["rank"] == 1).groupby(key).transform("max")
    w["back_book"] = (1 / w.b1).groupby(key).transform("sum")
    w["lay_book"] = (1 / w.l1).groupby(key).transform("sum")
    w["trap"] = pd.to_numeric(w.trap, errors="coerce")
    # the path: two minutes of the price, how much it has been moving, when the money came, where it traded
    g = w.groupby(["market", "sid"])
    w["mid_ch120"] = np.log(w["mid"] / g["mid"].shift(12))
    step = np.log(w["mid"] / g["mid"].shift(1))
    w["rv60"] = step.groupby(run).transform(lambda x: x.rolling(6, min_periods=3).std())
    w["tv_share60"] = w.tv_d60.clip(lower=0) / nz(w.tv)
    w["vwap_vs_mid"] = np.log(nz(w.trd_pv) / nz(w.trd_v)) - w.l_mid
    dv, dpv = w.trd_v - g.trd_v.shift(6), w.trd_pv - g.trd_pv.shift(6)
    w["vwap60_vs_mid"] = np.log((dpv / dv.where(dv > 0)).where(lambda v: v > 1)) - w.l_mid
    # the money waiting for the SP: the SP backers' stakes as the liability they need at the mid, against the SP
    # layers' liability; and the gap between them against the money near the top of the book
    m1 = (np.exp(w.l_mid) - 1).clip(lower=0.01)
    back_liab, lay_liab = w.spb_total.fillna(0) * m1, w.spl_total.fillna(0)
    w["sp_back_liab"], w["sp_lay_liab"] = np.log1p(back_liab), np.log1p(lay_liab)
    w["sp_imb"] = w.sp_back_liab - w.sp_lay_liab
    w["sp_imb_depth"] = (back_liab - lay_liab) / ((w.bs1 + w.bs2 + w.bs3 + w.ls1 + w.ls2 + w.ls3) * m1 + 1.0)
    return w


FEATURES = {
    "book": ["l_mid", "spread", "wom1", "wom3", "depth_imb", "ltp_vs_mid", "log_tv", "secs", "rank", "field",
             "overround", "race_tv"],
    "flow": ["tv_d30", "tv_d60", "mid_ch30", "mid_ch60"],
    "sp": ["spn_vs_mid", "spf_vs_mid"],
    "place": ["p_vs_win"],
    "model": ["model_vs_mid"],
    # the review's families (10 Nov 2026; ``review_columns``)
    "race": ["q_mid", "mid_ch60_rel", "tv_share", "tv_d60_share", "fav_ch60", "back_book", "lay_book", "trap"],
    "path": ["mid_ch120", "rv60", "tv_share60", "vwap_vs_mid", "vwap60_vs_mid"],
    "sppool": ["sp_back_liab", "sp_lay_liab", "sp_imb", "sp_imb_depth"],
}


def load_specs(path: Path = SPECS) -> list[dict]:
    specs = json.loads(path.read_text())["specs"]
    for s in specs:
        s["hash"] = hashlib.sha1(json.dumps({k: s[k] for k in ("features", "model", "target", "late") if k in s},
                                            sort_keys=True).encode()).hexdigest()[:10]
    return specs


def won(d: pd.DataFrame) -> np.ndarray:
    return (d.won if "won" in d else (d.status.astype(str) == "WINNER")).astype(int).to_numpy()


def win_chances(d: pd.DataFrame, f: np.ndarray) -> np.ndarray:
    """A log-price forecast read as win chances: 1 / price, normalised within each race at each checkpoint."""
    raw = pd.Series(np.exp(-np.asarray(f, float)), index=d.index)
    return (raw / raw.groupby([d.market, d.check]).transform("sum")).to_numpy()


BUCKETS = ((-10 ** 6, 0), (1, 60), (61, 180), (181, 10 ** 6))      # a ``by_window`` model: one fit a bucket


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, spec: dict) -> np.ndarray:
    """log(SP) forecast for ``test``, learnt from ``train`` (days before only). A spec without ``"late": true`` learns
    from the first five minutes' checkpoints only, as every spec registered before 10 Nov 2026 did. Model options
    (10 Nov): ``by_window`` (a fit for each of BUCKETS), ``norm`` (the race's forecast book rescaled, ``normalise``)."""
    m = spec["model"]
    if not spec.get("late") and "secs" in train:
        train = train[train.secs > 0]
    if m.get("by_window"):
        f = np.full(len(test), np.nan)
        for lo, hi in BUCKETS:
            te = test.secs.between(lo, hi).to_numpy()
            if te.any():
                tr = train[train.secs.between(lo, hi)]
                f[te] = _fit(tr if len(tr) >= 2000 else train, test[te], spec)
    else:
        f = _fit(train, test, spec)
    return normalise(train, test, f) if m.get("norm") else f


def book_ratio(train: pd.DataFrame) -> pd.Series:
    """By checkpoint: the SP book's sum over the mid book's sum, over the runners priced in both (the median race)."""
    key = [train.market, train.check]
    r = np.exp(-train.y).groupby(key).sum() / np.exp(-train.l_mid).groupby(key).sum()
    return r.groupby(level=1).median()


def normalise(train: pd.DataFrame, test: pd.DataFrame, f: np.ndarray) -> np.ndarray:
    """A race's forecast SPs move together: its forecast book (the sum of 1 / SP) is set to what the SP book has been
    against the mid book at this checkpoint on the training days. Each runner's log forecast shifts by the same amount."""
    ratio = book_ratio(train)
    k = test.check.map(ratio).fillna(float(ratio.median()))
    key = [test.market, test.check]
    target = k * np.exp(-test.l_mid).groupby(key).transform("sum")
    now = pd.Series(np.exp(-np.asarray(f, float)), index=test.index).groupby(key).transform("sum")
    return np.asarray(f, float) + np.log(now / target).to_numpy()


def _clogit(Z: np.ndarray, gi: np.ndarray, y: np.ndarray, w: np.ndarray, lam: np.ndarray, th0: np.ndarray):
    """A conditional logit: chances a softmax of Z theta within each group ``gi``; the weighted log loss of the
    winners plus a ridge ``lam`` (a weight a coefficient). Returns theta and the weighted log likelihood."""
    from scipy.optimize import minimize
    n = int(gi.max()) + 1

    def probs(th):
        u = Z @ th
        mx = np.full(n, -np.inf)
        np.maximum.at(mx, gi, u)
        eu = np.exp(u - mx[gi])
        return eu / np.bincount(gi, weights=eu, minlength=n)[gi]

    def nll(th):
        pr = probs(th)
        return (-(w * y * np.log(np.clip(pr, 1e-12, None))).sum() + (lam * th ** 2).sum(),
                Z.T @ (w * (pr - y)) + 2 * lam * th)

    fit = minimize(nll, th0, jac=True, method="L-BFGS-B")
    pr = probs(fit.x)
    return fit.x, float((w * y * np.log(np.clip(pr, 1e-12, None))).sum()), probs


def _race_groups(d: pd.DataFrame):
    """Group index of each race-checkpoint, the winner flags, and a weight that makes every race count once."""
    gi = pd.factorize(d.market.astype(str) + "|" + d.check.astype(str))[0]
    y = won(d).astype(float)
    one = np.bincount(gi, weights=y)[gi] == 1
    per_race = d.groupby("market").check.transform("nunique").to_numpy(float)
    return gi, y, np.where(one, 1.0 / per_race, 0.0)


def _fit(train: pd.DataFrame, test: pd.DataFrame, spec: dict) -> np.ndarray:
    """One fit: the residual on log(mid); or, for a spec with ``"target": "win"``, the result learnt from the
    market's chances, returned as -log of the chances."""
    feats = [f for fam in spec["features"] for f in FEATURES[fam]]
    m = spec["model"]
    if spec.get("target") == "win" and m["kind"] == "clogit":
        # the result as a conditional logit (10 Nov 2026): a x log(the mid's chance) + the features, a softmax within
        # the race, every race counting once however many checkpoints it has; a handful of coefficients, so it can be
        # fitted on a few hundred races where the gbm cannot
        X = train[feats].astype(float).replace([np.inf, -np.inf], np.nan)
        lo, hi = X.quantile(0.01), X.quantile(0.99)
        mu, sd = X.clip(lo, hi, axis=1).mean(), X.clip(lo, hi, axis=1).std() + 1e-9
        design = lambda d: np.column_stack([
            np.log(win_chances(d, d.l_mid.to_numpy()).clip(1e-6)),
            ((d[feats].astype(float).replace([np.inf, -np.inf], np.nan).clip(lo, hi, axis=1) - mu) / sd).fillna(0)])
        gi, y, wt = _race_groups(train)
        k = len(feats)
        th, _, _ = _clogit(design(train), gi, y, wt, np.r_[0.0, np.full(k, m.get("lam", 5.0))], np.r_[1.0, np.zeros(k)])
        u = pd.Series(design(test) @ th, index=test.index)
        key = [test.market, test.check]
        eu = np.exp(u - u.groupby(key).transform("max"))
        return -np.log((eu / eu.groupby(key).transform("sum")).clip(1e-9).to_numpy())
    if spec.get("target") == "win":
        if m["kind"] != "gbm":
            raise ValueError("a win spec is a gbm or a clogit")
        import lightgbm as lgb

        def base(d):                                         # the market's chances (the mid), as log-odds
            q = win_chances(d, d.l_mid.to_numpy()).clip(1e-4, 1 - 1e-4)
            return np.log(q / (1 - q))
        params = {"objective": "binary", "learning_rate": m.get("lr", 0.03), "num_leaves": m.get("leaves", 15),
                  "min_data_in_leaf": m.get("min_leaf", 300), "feature_fraction": m.get("ff", 0.9),
                  "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": m.get("l2", 5.0), "verbose": -1,
                  "seed": 0, "num_threads": 4}
        ds = lgb.Dataset(train[feats].astype(float), won(train), init_score=base(train))
        model = lgb.train(params, ds, m.get("rounds", 200))
        z = base(test) + model.predict(test[feats].astype(float), raw_score=True)
        p = pd.Series(1 / (1 + np.exp(-z)), index=test.index)
        p = p / p.groupby([test.market, test.check]).transform("sum")
        return -np.log(p.clip(1e-9).to_numpy())
    if m["kind"] == "gbm":
        import lightgbm as lgb
        params = {"objective": "regression", "learning_rate": m.get("lr", 0.05), "num_leaves": m.get("leaves", 31),
                  **({"objective": "huber", "alpha": m.get("huber", 0.3)} if m.get("loss") == "huber" else {}),
                  "min_data_in_leaf": m.get("min_leaf", 100), "feature_fraction": m.get("ff", 0.9),
                  "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": m.get("l2", 1.0), "verbose": -1,
                  "seed": 0, "num_threads": 4}
        model = lgb.train(params, lgb.Dataset(train[feats].astype(float), train.y - train.l_mid), m.get("rounds", 400))
        return test.l_mid.to_numpy() + model.predict(test[feats].astype(float))
    if m["kind"] == "ridge":
        # an infinite feature (a log of a zero) left the standardisation NaN, and every forecast with it (6 Nov)
        X = train[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        Xt = test[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        mu, sd = X.mean(0), X.std(0) + 1e-9
        X, Xt = (X - mu) / sd, (Xt - mu) / sd
        X1 = np.column_stack([np.ones(len(X)), X])
        lam = m.get("lam", 10.0)
        beta = np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ (train.y - train.l_mid).to_numpy())
        return test.l_mid.to_numpy() + np.column_stack([np.ones(len(Xt)), Xt]) @ beta
    raise ValueError(f"model kind {m['kind']}")


def baseline(test: pd.DataFrame, name: str) -> np.ndarray:
    col = {"mid": "l_mid", "ltp": "l_ltp", "spn": "l_spn"}[name]
    return test[col].fillna(test.l_mid).to_numpy()


WINDOWS = (("le60", 1, 60), ("70_180", 61, 180), ("gt180", 181, 10 ** 6), ("late", -10 ** 6, 0))
ORACLE = "oracle_sp"            # the SP itself read as chances: the most a perfect SP forecast could score on Win


def evaluate(days: dict, specs: list[dict], min_train_days: int = 1) -> pd.DataFrame:
    """Each spec and baseline on each day it can be fitted for (trained on the days before only), on both targets:
    RMSE of log(SP) over every runner-checkpoint, and the winner's log loss over every race-checkpoint with one
    winner; each overall and by window (<= 60 s, 70-180 s, > 180 s out)."""
    rows = []
    order = sorted(days)
    for i, d in enumerate(order):
        test = days[d]
        train = pd.concat([days[x] for x in order[:i]], ignore_index=True) if i else pd.DataFrame()
        preds = {b: baseline(test, b) for b in BASELINES}
        if i >= min_train_days and len(train) > 1000:
            for s in specs:
                try:
                    preds[s["name"]] = fit_predict(train, test, s)
                except Exception as exc:                        # a broken spec never stops the others
                    log.warning("%s on %s: %s", s["name"], d, exc)
        preds[ORACLE] = test.y.to_numpy()
        w = won(test)
        one = pd.Series(w, index=test.index).groupby([test.market, test.check]).transform("sum").to_numpy() == 1
        is_w = (w == 1) & one                                  # the winner's row of each race-checkpoint with one
        main = (test.secs > 0).to_numpy()                      # the headline: the first five minutes' checkpoints
        tight = main & (test.spread.to_numpy(float) <= 0.05) if "spread" in test else np.zeros(len(test), bool)
        q = test.q_mid.fillna(0).to_numpy(float) if "q_mid" in test else np.zeros(len(test))
        rms = lambda v, k: float(np.sqrt(np.nanmean(v[k] ** 2))) if k.any() and np.isfinite(v[k]).any() else np.nan
        for name, f in preds.items():
            e = test.y.to_numpy() - f
            ll = np.where(is_w, -np.log(np.clip(win_chances(test, f), 1e-9, None)), np.nan)
            ok = main & np.isfinite(e) & (q > 0)
            row = {"day": d, "spec": name, "runners": int(main.sum()), "races": int(test.market[main].nunique()),
                   "rmse": rms(e, main),
                   "win_ll": float(np.nanmean(ll[main & is_w])) if (main & is_w).any() else np.nan,
                   "win_races": int((is_w & main).sum()),
                   # where the book is real (the spread within 5%), and weighted by the runner's chance
                   "rmse_tight": rms(e, tight),
                   "rmse_p": float(np.sqrt((q[ok] * e[ok] ** 2).sum() / q[ok].sum())) if ok.any() else np.nan}
            for lab, lo, hi in WINDOWS:
                k = test.secs.between(lo, hi).to_numpy()
                row[f"rmse_{lab}"] = rms(e, k)
                row[f"win_ll_{lab}"] = float(np.nanmean(ll[k & is_w])) if (k & is_w).any() else np.nan
            if name == ORACLE:
                row.update({c: np.nan for c in row if c.startswith("rmse")})
            rows.append(row)
            test = test.assign(**{f"e_{name}": e, f"w_{name}": ll})
        days[d] = test
    return pd.DataFrame(rows)


def paired_gain(days: dict, a: str, b: str, use_days: list, n: int = 2000, seed: int = 0, target: str = "bfsp"):
    """The loss of ``b`` less that of ``a`` (positive: ``a`` better) over the days given, with a 90% interval by a
    bootstrap over races: squared error of log(SP) (``bfsp``) or the winner's log loss (``win``)."""
    pre = TARGETS[target][1]
    parts = [days[d].loc[days[d].secs > 0, ["market", f"{pre}{a}", f"{pre}{b}"]].dropna() for d in use_days
             if f"{pre}{a}" in days[d] and f"{pre}{b}" in days[d]]
    if not parts:
        return None
    x = pd.concat(parts)
    sq = 2 if target == "bfsp" else 1
    by_race = (x[f"{pre}{b}"] ** sq - x[f"{pre}{a}"] ** sq).groupby(x.market).mean().to_numpy()
    if len(by_race) < 20:
        return None
    rng = np.random.default_rng(seed)
    bs = by_race[rng.integers(0, len(by_race), size=(n, len(by_race)))].mean(1)
    se = float(bs.std())
    gain = float(by_race.mean())
    return {"gain": gain, "lo90": float(np.percentile(bs, 5)), "hi90": float(np.percentile(bs, 95)),
            "races": int(len(by_race)), "se": se, "z": gain / se if se > 0 else 0.0,
            # the races a gain of this size needs to reach the bar, at this spread of the race-level differences
            "races_for_bar": int(np.ceil((BAR_Z * float(by_race.std()) / gain) ** 2)) if gain > 0 else None}


# The bar, version 2 (10 Nov 2026, reports/greyhound_model_review.md). The first bar (a 90% lower bound above zero) was
# read every night against the best of several challengers: on 9 Nov the Win crown changed on a lower bound of 0.00004
# at the third nightly look among seven challengers, which is what chance does. From 10 Nov a crown needs the paired
# gain to be BAR_Z bootstrap standard errors above zero (one-sided p of about 0.001 a look, so nightly looks at a
# dozen specs stay near 5% in all), over at least BAR_DAYS fresh days and BAR_RACES races; and a spec holding a crown
# must still clear the same bar against the best baseline, or the crown returns to that baseline.
BAR_Z, BAR_DAYS, BAR_RACES = 3.0, 3, 300


def clears(g: dict | None, n_days: int, bar: tuple = (BAR_Z, BAR_DAYS, BAR_RACES)) -> bool:
    return bool(g and g["z"] >= bar[0] and n_days >= bar[1] and g["races"] >= bar[2])


def champion(results: pd.DataFrame, specs: list[dict], days: dict, incumbent: str | None,
             target: str = "bfsp", bar: tuple = (BAR_Z, BAR_DAYS, BAR_RACES)) -> dict:
    """The best spec on ``target`` over its fresh days (after its registration); it takes over only when its paired
    gain over the incumbent on the days both were fresh clears the bar (``clears``)."""
    col = TARGETS[target][0]
    results = results.assign(rmse=results[col]) if col != "rmse" else results
    reg = {s["name"]: s["registered"] for s in specs}
    # the bar a spec must clear: the incumbent spec, else the best baseline over the days scored (the baselines are
    # read off the book, never fitted, so choosing among the three needs no fresh days)
    base = results[results.spec.isin(BASELINES)].groupby("spec").rmse.mean()
    best_base = base.idxmin() if len(base) else "mid"
    fresh = results[results.spec.map(lambda n: n in reg) & (results.day > results.spec.map(reg))]
    fresh_days = lambda n: sorted(set(fresh[fresh.spec == n].day))
    inc, note = best_base, ""
    if incumbent and incumbent in reg:
        held = paired_gain(days, incumbent, best_base, fresh_days(incumbent), target=target)
        if clears(held, len(fresh_days(incumbent)), bar):
            inc = incumbent
        else:
            note = (f"{incumbent} no longer clears the bar against {best_base} "
                    f"(z {held['z']:.2f} over {held['races']} races)" if held else
                    f"{incumbent} has too few fresh races to hold the crown") + f": the crown returns to {best_base}"
    if not len(fresh):
        return {"champion": inc, "incumbent": inc, "note": "no spec has a fresh day yet"}
    score = fresh.groupby("spec").agg(days=("day", "nunique"), rmse=("rmse", "mean")).sort_values("rmse")
    # the challenger: the spec with the strongest paired evidence against the incumbent on the days both were fresh
    # (specs registered on different days have different fresh days, and days differ, so their raw means do not compare)
    cands = {}
    for name in score.index:
        if name == inc:
            continue
        both = sorted(set(fresh_days(name)) & ({d for d in days} if inc in BASELINES else set(fresh_days(inc))))
        g = paired_gain(days, name, inc, both, target=target)
        if g:
            cands[name] = (g, len(both))
    out = {"incumbent": inc, "metric": col, "note": note, "fresh_scores": score.reset_index().to_dict("records")}
    if not cands:
        return {**out, "champion": inc, "challenger": None, "paired": None}
    best = max(cands, key=lambda n: cands[n][0]["z"])
    g, n_days = cands[best]
    return {**out, "champion": best if clears(g, n_days, bar) else inc, "challenger": best, "paired": g,
            "paired_days": n_days}


# --------------------------------------------------------------------------------------------------------------------
# The trade test on the forecast (registered 8 Nov 2026, before its first day): does the price on offer lag the
# forecast? The first fresh day (7 Nov) made gbm_all_small the BFSP champion (paired gain 0.041, 90% 0.032 to 0.051,
# 121 races; run 42428569854). Fixed now: that spec only; at 60, 30 and 10 s out, back at the best back price when it
# is more than m above the forecast SP, lay at the best lay price when it is more than m below (GBP2 or more shown);
# m in 3, 6, 10%; each scored closed at the SP and left to run to the result (2% commission on a gain). Counted on days
# after TRADE_REGISTERED only; a rule is a candidate on a target after TRADE_MIN_DAYS days with its 90% lower bound
# (bootstrap over races) above zero: 18 rules, 36 tests, about one in twenty passing by chance.
# --------------------------------------------------------------------------------------------------------------------

TRADE_SPEC = "gbm_all_small"
TRADE_REGISTERED = "2026-11-08"
TRADE_MIN_DAYS = 10
TRADE_CHECKS = (60, 30, 10)
TRADE_MARGINS = (0.03, 0.06, 0.10)


def trade_rules(days: dict, spec: str = TRADE_SPEC, comm: float | None = None) -> pd.DataFrame:
    """Every rule's unit trades on every day the spec was forecast (``e_<spec>`` set by ``evaluate``)."""
    from greyhound.costs import commission
    c = commission() if comm is None else comm
    rows = []
    for d, f in days.items():
        if f"e_{spec}" not in f:
            continue
        x = f[f.secs.isin(TRADE_CHECKS)].copy()
        x["fc"] = np.exp(x.y - x[f"e_{spec}"])
        x["sp"] = np.exp(x.y)
        x["won"] = won(x)
        for m in TRADE_MARGINS:
            b = x[(x.b1 > 1) & (x.bs1 >= 2) & (x.b1 / x.fc - 1 > m)]
            raw_sp = b.b1 / b.sp - 1
            raw_w = np.where(b.won == 1, b.b1 - 1, -1.0)
            rows.append(pd.DataFrame({"day": d, "market": b.market, "rule": [f"back_{int(s)}_{m}" for s in b.secs],
                                      "pnl_sp": np.where(raw_sp > 0, raw_sp * (1 - c), raw_sp),
                                      "pnl_win": np.where(raw_w > 0, raw_w * (1 - c), raw_w)}))
            l = x[(x.l1 > 1) & (x.ls1 >= 2) & (1 - x.l1 / x.fc > m)]
            raw_sp = 1 - l.l1 / l.sp
            raw_w = np.where(l.won == 1, -(l.l1 - 1), 1.0)
            rows.append(pd.DataFrame({"day": d, "market": l.market, "rule": [f"lay_{int(s)}_{m}" for s in l.secs],
                                      "pnl_sp": np.where(raw_sp > 0, raw_sp * (1 - c), raw_sp),
                                      "pnl_win": np.where(raw_w > 0, raw_w * (1 - c), raw_w)}))
    return pd.concat([r for r in rows if len(r)], ignore_index=True) if rows else pd.DataFrame()


def trade_scores(tr: pd.DataFrame, registered: str = TRADE_REGISTERED, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Each rule on its fresh days: trades, races, days, mean P&L a unit on each target, its 90% interval by race."""
    if not len(tr):
        return pd.DataFrame()
    tr = tr[tr.day > registered]
    if not len(tr):
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    out = []
    for rule, g in tr.groupby("rule"):
        row = {"rule": rule, "trades": len(g), "days": int(g.day.nunique())}
        for col in ("pnl_sp", "pnl_win"):
            by_race = g.groupby(["day", "market"])[col].mean().to_numpy()
            row[f"{col}%"] = round(100 * float(g[col].mean()), 3)
            if len(by_race) >= 10:
                bs = by_race[rng.integers(0, len(by_race), size=(n, len(by_race)))].mean(1)
                lo, hi = np.percentile(bs, [5, 95])
                row[f"{col}_ci90%"] = [round(100 * float(lo), 3), round(100 * float(hi), 3)]
                row[f"{col}_candidate"] = bool(lo > 0 and row["days"] >= TRADE_MIN_DAYS)
            else:
                row[f"{col}_ci90%"], row[f"{col}_candidate"] = None, False
            row[f"{col}_z"] = mt.z_by_race(by_race)
        out.append(row)
    res = pd.DataFrame(out).sort_values("pnl_sp%", ascending=False).reset_index(drop=True)
    # the family read as one (research/greyhound/multitest.py): every rule on both targets
    z = np.r_[res.pnl_sp_z.to_numpy(float), res.pnl_win_z.to_numpy(float)]
    q = mt.bh(mt.p_one_sided(z))
    res["pnl_sp_q"], res["pnl_win_q"] = q[:len(res)], q[len(res):]
    return res


def second_stage(x: pd.DataFrame, fc: np.ndarray) -> dict | None:
    """Benter's second stage, one checkpoint a race: the result as a conditional logit on the mid's chances alone,
    then with the forecast's chances beside them (the log of forecast over mid). ``b`` is the weight the result puts
    on the forecast's disagreement with the mid (0: it adds nothing; 1: believe the forecast); the likelihood-ratio
    test says whether that weight is above chance."""
    import math
    lq = np.log(np.clip(win_chances(x, x.l_mid.to_numpy()), 1e-9, None))
    lf = np.log(np.clip(win_chances(x, np.asarray(fc, float)), 1e-9, None))
    gi, y, wt = _race_groups(x)
    wt = (wt > 0).astype(float)
    races = int((wt * y).sum())
    if races < 50 or not np.isfinite(lf).all():
        return None
    _, ll0, _ = _clogit(lq[:, None], gi, y, wt, np.zeros(1), np.ones(1))
    th, ll1, _ = _clogit(np.column_stack([lq, lf - lq]), gi, y, wt, np.zeros(2), np.r_[1.0, 0.0])
    lr = max(2 * (ll1 - ll0), 0.0)
    return {"races": races, "a_mid": round(float(th[0]), 3), "b_forecast": round(float(th[1]), 3),
            "ll_mid": round(-ll0 / races, 5), "ll_both": round(-ll1 / races, 5),
            "gain": round((ll1 - ll0) / races, 5), "lr_p": float(math.erfc(math.sqrt(lr / 2)))}


def diagnostics(days: dict, specs: list[dict], champs: dict) -> list[str]:
    """The review's standing readings (10 Nov 2026), written under the night's report: when the races really go off,
    whether the SP pool is in the stream, what a perfect SP forecast would be worth on Win, and whether the champion's
    forecast tells the result anything the mid does not."""
    md = ["", "### The review's readings (reports/greyhound_model_review.md)", ""]
    if not days:
        return md + ["no data found"]
    # 1. the real off against the scheduled one
    md += ["When the market really closed, against its scheduled off (seconds; WIN markets with an SP):", "",
           "| Day | Markets | Median | 10% | 90% | Still open at the scheduled off | 30 s after | 60 s after |",
           "|---|---|---|---|---|---|---|---|"]
    for d in sorted(days):
        f = days[d]
        if "t_closed" not in f or not f.t_closed.notna().any():
            md.append(f"| {d} | no data found | | | | | | |")
            continue
        one = f.drop_duplicates("market")
        late = (pd.to_datetime(one.t_closed, utc=True) - pd.to_datetime(one.off, utc=True)).dt.total_seconds().dropna()
        n = f.market.nunique()
        still = lambda c: f"{f[f.check == c].market.nunique() / n:.0%}"
        md.append(f"| {d} | {n} | {late.median():.0f} | {late.quantile(0.1):.0f} | {late.quantile(0.9):.0f} | "
                  f"{still(0)} | {still(-30)} | {still(-60)} |")
    # 2. the SP pool in the stream
    last = days[sorted(days)[-1]]
    main = last[last.secs > 0]
    if "spb_total" in main and len(main):
        b, l = main.spb_total.fillna(0), main.spl_total.fillna(0)
        md += ["", f"The SP pool on {sorted(days)[-1]} (the first five minutes' runner-checkpoints): SP backs present "
               f"on {(b > 0).mean():.0%} (median GBP{b[b > 0].median() if (b > 0).any() else 0:.0f} staked), SP lays on "
               f"{(l > 0).mean():.0%} (median GBP{l[l > 0].median() if (l > 0).any() else 0:.0f} liability)."]
    # 3. the SP book
    x = pd.concat([f[f.check == 60] for f in days.values()])
    full = x.groupby("market").filter(lambda g: len(g) == g.field.iloc[0]) if len(x) else x
    if len(full):
        bk = np.exp(-full.y).groupby(full.market).sum()
        md += ["", f"The SP book (the sum of 1 / SP over a race, {len(bk)} races with every runner priced): median "
               f"{bk.median():.4f}, 10% {bk.quantile(0.1):.4f}, 90% {bk.quantile(0.9):.4f}."]
    # 4. the second stage: what the forecast tells the result beyond the mid, and the ceiling (the SP itself)
    reg = {s["name"]: s["registered"] for s in specs}
    md += ["", "Benter's second stage (the result on the mid's chances, then with a forecast beside them; one "
           "checkpoint a race, days after each spec's registration):", "",
           "| Forecast | Seconds out | Races | Weight on the mid | Weight on the forecast's gap | Log loss, mid alone | "
           "With the forecast | Gain | p (likelihood ratio) |", "|---|---|---|---|---|---|---|---|---|"]
    names = [ORACLE] + sorted({c.get("champion") for c in champs.values() if c.get("champion") in reg}
                              | {c.get("challenger") for c in champs.values() if c.get("challenger") in reg})
    for name in names:
        for c in (60, 10, -30):
            parts = [f[f.check == c] for d, f in days.items() if f"e_{name}" in f and d > reg.get(name, "")]
            x = pd.concat(parts) if parts else pd.DataFrame()
            x = x[x[f"e_{name}"].notna()] if len(x) else x
            r = second_stage(x, (x.y - x[f"e_{name}"]).to_numpy()) if len(x) else None
            if r:
                md.append(f"| {'the SP itself (the ceiling)' if name == ORACLE else name} | {c} | {r['races']} | "
                          f"{r['a_mid']} | {r['b_forecast']} | {r['ll_mid']} | {r['ll_both']} | {r['gain']} | "
                          f"{r['lr_p']:.4f} |")
    return md


def main(argv=None) -> int:
    """Build the snapshots of any recorded day not yet built, score every spec on every day, append the night's
    scores to the trial registry, pick the champion and write the report."""
    import argparse
    import os
    import subprocess
    from betfair_recorder import bucket, s3_client
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/close_lab")
    ap.add_argument("--rebuild", action="store_true", help="rebuild every day's snapshots")
    ap.add_argument("--dry-run", action="store_true", help="score and report only: no registry or champion written")
    a = ap.parse_args(argv)
    s3, bk, store = s3_client(), bucket(), Store()

    def listing(prefix):
        out, tok = [], None
        while True:
            r = s3.list_objects_v2(Bucket=bk, Prefix=prefix, **({"ContinuationToken": tok} if tok else {}))
            out += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return out
            tok = r["NextContinuationToken"]

    def get(k):
        try:
            return s3.get_object(Bucket=bk, Key=k)["Body"].read()
        except Exception:
            return None

    # complete days only (before today, UTC): a day still being recorded would be cached half-built and never rebuilt
    stream_days = sorted(d for d in {k.split("/")[2] for k in listing(f"{sr.PREFIX}/") if k.count("/") >= 3}
                         if d < datetime.now(timezone.utc).date().isoformat())
    days = {}
    for d in stream_days:
        key = f"{PREFIX}/{FRAMES}/{d}.parquet"
        raw = None if a.rebuild else get(key)
        if raw:
            days[d] = pd.read_parquet(io.BytesIO(raw))
            continue
        lines = sr.load_lines(get, listing, date.fromisoformat(d))
        snap = snapshots(lines)
        pr = store.get(f"greyhound/live/predictions/{d}.csv")
        preds = pd.read_csv(io.BytesIO(pr), dtype={"market_id": str}) if pr else None
        f = frame(snap, preds)
        log.info("%s: %d stream lines, %d runner-checkpoints with an SP", d, len(lines), len(f))
        if len(f):
            buf = io.BytesIO()
            f.to_parquet(buf, index=False)
            s3.put_object(Bucket=bk, Key=key, Body=buf.getvalue())
            days[d] = f
    specs = load_specs()
    res = evaluate(days, specs) if days else pd.DataFrame()
    sha = os.getenv("GITHUB_SHA") or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                     text=True).stdout.strip()
    night = date.today().isoformat()
    inc_raw = get(f"{PREFIX}/champion.json")
    prev = json.loads(inc_raw) if inc_raw else {}
    champs = {}
    for t in TARGETS:                                   # each target its own champion (a file before 6 Nov held one)
        inc = (prev.get(t) or {}).get("champion") if t in prev else (prev.get("champion") if t == "bfsp" else None)
        champs[t] = champion(res, specs, days, inc, target=t) if len(res) else {"note": "no days"}
    if len(res):
        res = res.assign(night=night, commit=sha, spec_hash=res.spec.map({s["name"]: s["hash"] for s in specs}))
    if len(res) and not a.dry_run:
        old = get(f"{PREFIX}/registry.csv")
        reg = pd.concat([pd.read_csv(io.BytesIO(old)), res], ignore_index=True) if old else res
        s3.put_object(Bucket=bk, Key=f"{PREFIX}/registry.csv", Body=reg.to_csv(index=False).encode())
        s3.put_object(Bucket=bk, Key=f"{PREFIX}/champion.json",
                      Body=json.dumps({**champs, "night": night, "commit": sha}, default=str).encode())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"## Greyhound closing-price lab, night of {night}", "",
          f"Days recorded: {', '.join(sorted(days)) or 'none'}. Specs: {len(specs)}, each scored on two targets: "
          "1. BFSP (RMSE of log(SP)) and 2. Win (the winner's log loss, the forecast read as chances normalised within "
          "the race), over the first five minutes' checkpoints. A spec counts only on days after it was registered; a "
          f"crown changes only on a paired gain {BAR_Z:g} standard errors above zero over at least {BAR_DAYS} fresh days "
          f"and {BAR_RACES} races (the bar from 10 Nov 2026)." + (" DRY RUN: nothing written." if a.dry_run else ""), ""]
    for t, lab in (("bfsp", "1. BFSP"), ("win", "2. Win")):
        c = champs[t]
        g = c.get("paired")
        paired = (f"gain {g['gain']:.5f}, 90% {g['lo90']:.5f} to {g['hi90']:.5f}, z {g['z']:.2f}, {g['races']} races over "
                  f"{c.get('paired_days')} days" + (f"; a gain this size needs {g['races_for_bar']} races to clear the bar"
                                                    if g.get("races_for_bar") else "")) if g else "none"
        md.append(f"Champion, {lab}: **{c.get('champion')}** (incumbent {c.get('incumbent')}, strongest fresh "
                  f"challenger {c.get('challenger')}; paired: {paired}). {c.get('note', '')}")
    md += ["", "The last three days (every day is in the registry). 'Tight': the spread within 5%. 'By chance': each "
           "runner weighted by its chance. 'Late': after the scheduled off, the market still open.", "",
           "| Day | Spec | Runner-checkpoints | Races | BFSP: RMSE log(SP) | <= 60 s | 70-180 s | > 180 s | Tight | "
           "By chance | Late | Win: log loss | <= 60 s | 70-180 s | > 180 s | Late | Race-checkpoints |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f4 = lambda v: "" if pd.isna(v) else f"{v:.4f}"
    shown = res[res.day.isin(sorted(res.day.unique())[-3:])] if len(res) else res
    for r in shown.sort_values(["day", "rmse"]).to_dict("records") if len(res) else []:
        md.append(f"| {r['day']} | {r['spec']} | {r['runners']} | {r['races']} | {f4(r['rmse'])} | "
                  f"{f4(r['rmse_le60'])} | {f4(r['rmse_70_180'])} | {f4(r['rmse_gt180'])} | {f4(r['rmse_tight'])} | "
                  f"{f4(r['rmse_p'])} | {f4(r['rmse_late'])} | {f4(r['win_ll'])} | {f4(r['win_ll_le60'])} | "
                  f"{f4(r['win_ll_70_180'])} | {f4(r['win_ll_gt180'])} | {f4(r['win_ll_late'])} | {r['win_races']} |")
    tr = trade_rules(days) if len(res) else pd.DataFrame()
    ts = trade_scores(tr)
    md += ["", f"### The trade test on {TRADE_SPEC}'s forecast (registered {TRADE_REGISTERED}; counted on later days, a "
           f"candidate after {TRADE_MIN_DAYS} days; 36 tests)", ""]
    if len(ts):
        q2 = lambda v: "" if pd.isna(v) else f"{v:.2f}"
        md += ["q: the share of false discoveries expected among every test of the family at least as strong "
               "(research/greyhound/multitest.py); under 0.10 is support from the family as a whole.", "",
               "| Rule | Trades | Days | At the SP: % a unit | 90% | q | To the result: % a unit | 90% | q | "
               "Candidate (SP / Win) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in ts.to_dict("records"):
            md.append(f"| {r['rule']} | {r['trades']} | {r['days']} | {r['pnl_sp%']} | {r['pnl_sp_ci90%']} | "
                      f"{q2(r['pnl_sp_q'])} | {r['pnl_win%']} | {r['pnl_win_ci90%']} | {q2(r['pnl_win_q'])} | "
                      f"{r['pnl_sp_candidate']} / {r['pnl_win_candidate']} |")
    else:
        md.append("No fresh day yet.")
    md += diagnostics(days, specs, champs)
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
