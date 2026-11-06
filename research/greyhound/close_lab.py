"""The closing-price lab: a self-improving forecast of each dog's Betfair SP from the recorded Exchange Stream
(read-only research; the owner, 6 Nov 2026: "let's review and optimise our model ... now we have the market data?
Let's start recursive self improvement"; earlier: "fundamental is always just to get the most accurate closing price").

The data: every 10 s over the last ``WINDOW`` seconds before each scheduled off, each runner's book as it stood
(rebuilt from the stream by research.greyhound.stream_replay): three levels each side and their money, the money
offered in all, the last traded price, Betfair's projected SPs (near, far), the money traded so far and in the last
30 and 60 s; the race's overround and money; the same dog's place market; our pre-race price
(greyhound/live/predictions/<day>.csv). The target: log of the SP in the settled definition.

The loop that keeps it honest:
- every experiment is a line in ``research/greyhound/close_specs.json`` (a name, its features, its model, the date it
  was registered); the night's job (greyhound-close-lab.yml) builds the new day's snapshots, fits every spec on the
  days before each test day only, scores every day, and appends each spec's score on each day to the trial registry
  (s3 close_lab/registry.csv, with the code's commit);
- a spec counts only on days after the day it was registered (``fresh`` days): the champion is the spec with the
  lowest RMSE of log(SP) over its fresh days, and replaces the incumbent only when its paired gain over the incumbent
  on the days both were fresh has a 90% interval (bootstrap over races) above zero;
- the baselines are always scored: the market mid, the last traded price and Betfair's own projected SP.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from research.greyhound import stream_replay as sr

log = logging.getLogger(__name__)

WINDOW = 300
STEP = 10
CHECKS = tuple(range(WINDOW, 0, -STEP))
SPECS = Path(__file__).with_name("close_specs.json")
PREFIX = "close_lab"
BASELINES = ("mid", "ltp", "spn")


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
                   "atb_total": sum(r.atb.values()), "atl_total": sum(r.atl.values())}
            for i in range(3):
                row[f"b{i+1}"], row[f"bs{i+1}"] = bk[i] if i < len(bk) else (np.nan, 0.0)
                row[f"l{i+1}"], row[f"ls{i+1}"] = ly[i] if i < len(ly) else (np.nan, 0.0)
            row["mid"] = _mid(row["b1"] if np.isfinite(row["b1"]) else None, row["bs1"],
                              row["l1"] if np.isfinite(row["l1"]) else None, row["ls1"])
            rows.append(row)

    import heapq
    due: list = []                     # (checkpoint time, market, checkpoint): every market's marks in time order
    for t, mc in sr.events(lines):
        # every mark passed by now, whatever market this message is for: a quiet market keeps its book until it changes
        while due and due[0][0] <= t:
            _, mid0, c = heapq.heappop(due)
            take(markets[mid0], c)
        mid = str(mc.get("id"))
        sr.apply(markets, mc)
        m = markets[mid]
        if mid not in nxt and m.off is not None:
            nxt[mid] = 1
            for c in CHECKS:
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
                         "off": m.off, "n_winners": d.get("numberOfWinners")})
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


FEATURES = {
    "book": ["l_mid", "spread", "wom1", "wom3", "depth_imb", "ltp_vs_mid", "log_tv", "secs", "rank", "field",
             "overround", "race_tv"],
    "flow": ["tv_d30", "tv_d60", "mid_ch30", "mid_ch60"],
    "sp": ["spn_vs_mid", "spf_vs_mid"],
    "place": ["p_vs_win"],
    "model": ["model_vs_mid"],
}


def load_specs(path: Path = SPECS) -> list[dict]:
    specs = json.loads(path.read_text())["specs"]
    for s in specs:
        s["hash"] = hashlib.sha1(json.dumps({k: s[k] for k in ("features", "model")}, sort_keys=True)
                                 .encode()).hexdigest()[:10]
    return specs


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, spec: dict) -> np.ndarray:
    """log(SP) forecast for ``test``: the residual on log(mid) learnt from ``train`` (days before only)."""
    feats = [f for fam in spec["features"] for f in FEATURES[fam]]
    m = spec["model"]
    if m["kind"] == "gbm":
        import lightgbm as lgb
        params = {"objective": "regression", "learning_rate": m.get("lr", 0.05), "num_leaves": m.get("leaves", 31),
                  "min_data_in_leaf": m.get("min_leaf", 100), "feature_fraction": m.get("ff", 0.9),
                  "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": m.get("l2", 1.0), "verbose": -1,
                  "seed": 0, "num_threads": 4}
        model = lgb.train(params, lgb.Dataset(train[feats].astype(float), train.y - train.l_mid), m.get("rounds", 400))
        return test.l_mid.to_numpy() + model.predict(test[feats].astype(float))
    if m["kind"] == "ridge":
        X = train[feats].astype(float).fillna(0).to_numpy()
        Xt = test[feats].astype(float).fillna(0).to_numpy()
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


def evaluate(days: dict, specs: list[dict], min_train_days: int = 1) -> pd.DataFrame:
    """Each spec and baseline on each day it can be fitted for (trained on the days before only): RMSE of log(SP)
    over every runner-checkpoint, and by window (<= 60 s, 70-180 s, > 180 s out)."""
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
        for name, f in preds.items():
            e = test.y.to_numpy() - f
            row = {"day": d, "spec": name, "runners": len(test), "races": int(test.market.nunique()),
                   "rmse": float(np.sqrt(np.nanmean(e ** 2)))}
            for lab, lo, hi in (("rmse_le60", 0, 60), ("rmse_70_180", 61, 180), ("rmse_gt180", 181, 10 ** 6)):
                k = test.secs.between(lo, hi).to_numpy()
                row[lab] = float(np.sqrt(np.nanmean(e[k] ** 2))) if k.any() else np.nan
            rows.append(row)
            test = test.assign(**{f"e_{name}": e})
        days[d] = test
    return pd.DataFrame(rows)


def paired_gain(days: dict, a: str, b: str, use_days: list, n: int = 2000, seed: int = 0):
    """Mean squared error of ``b`` less ``a`` (positive: ``a`` better) over the days given, with a 90% interval by a
    bootstrap over races."""
    parts = [days[d][["market", f"e_{a}", f"e_{b}"]].dropna() for d in use_days
             if f"e_{a}" in days[d] and f"e_{b}" in days[d]]
    if not parts:
        return None
    x = pd.concat(parts)
    by_race = (x[f"e_{b}"] ** 2 - x[f"e_{a}"] ** 2).groupby(x.market).mean().to_numpy()
    if len(by_race) < 20:
        return None
    rng = np.random.default_rng(seed)
    bs = by_race[rng.integers(0, len(by_race), size=(n, len(by_race)))].mean(1)
    return {"gain": float(by_race.mean()), "lo90": float(np.percentile(bs, 5)), "hi90": float(np.percentile(bs, 95)),
            "races": int(len(by_race))}


def champion(results: pd.DataFrame, specs: list[dict], days: dict, incumbent: str | None) -> dict:
    """The best spec over its fresh days (after its registration); it takes over only when its paired gain over the
    incumbent on the days both were fresh clears zero at the 90% lower bound."""
    reg = {s["name"]: s["registered"] for s in specs}
    fresh = results[results.spec.map(lambda n: n in reg) & (results.day > results.spec.map(reg))]
    if not len(fresh):
        return {"champion": incumbent or "spn", "note": "no spec has a fresh day yet"}
    score = fresh.groupby("spec").agg(days=("day", "nunique"), rmse=("rmse", "mean")).sort_values("rmse")
    best = score.index[0]
    inc = incumbent or "spn"
    both = sorted(set(fresh[fresh.spec == best].day) & ({d for d in days} if inc in BASELINES
                                                          else set(fresh[fresh.spec == inc].day)))
    g = paired_gain(days, best, inc, both) if best != inc else None
    take = bool(g and g["lo90"] > 0)
    return {"champion": best if take else inc, "challenger": best, "incumbent": inc, "paired": g,
            "fresh_scores": score.reset_index().to_dict("records")}


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

    stream_days = sorted({k.split("/")[2] for k in listing(f"{sr.PREFIX}/") if k.count("/") >= 3})
    days = {}
    for d in stream_days:
        key = f"{PREFIX}/frames/{d}.parquet"
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
    incumbent = json.loads(inc_raw).get("champion") if inc_raw else None
    champ = champion(res, specs, days, incumbent) if len(res) else {"note": "no days"}
    if len(res):
        res = res.assign(night=night, commit=sha, spec_hash=res.spec.map({s["name"]: s["hash"] for s in specs}))
        old = get(f"{PREFIX}/registry.csv")
        reg = pd.concat([pd.read_csv(io.BytesIO(old)), res], ignore_index=True) if old else res
        s3.put_object(Bucket=bk, Key=f"{PREFIX}/registry.csv", Body=reg.to_csv(index=False).encode())
        s3.put_object(Bucket=bk, Key=f"{PREFIX}/champion.json",
                      Body=json.dumps({**champ, "night": night, "commit": sha}, default=str).encode())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"## Greyhound closing-price lab, night of {night}", "",
          f"Days recorded: {', '.join(sorted(days)) or 'none'}. Specs: {len(specs)}. A spec counts only on days after "
          "it was registered; the champion changes only on a paired gain with its 90% lower bound above zero.", "",
          f"Champion: **{champ.get('champion')}** (incumbent {champ.get('incumbent')}, best fresh challenger "
          f"{champ.get('challenger')}; paired: {json.dumps(champ.get('paired'))}). {champ.get('note', '')}", "",
          "| Day | Spec | Runner-checkpoints | Races | RMSE log(SP) | <= 60 s | 70-180 s | > 180 s |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res.sort_values(["day", "rmse"]).itertuples(index=False) if len(res) else []:
        md.append(f"| {r.day} | {r.spec} | {r.runners} | {r.races} | {r.rmse:.4f} | {r.rmse_le60:.4f} | "
                  f"{r.rmse_70_180:.4f} | {r.rmse_gt180:.4f} |")
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
