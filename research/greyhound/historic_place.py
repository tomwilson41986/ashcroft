"""The place market against the win market from Betfair's historic data (read-only research). The owner's download
of the BASIC greyhound data (sources/betfair_imports.json: Jan-Sep 2026, GB/AU/NZ, every market type; built by
betfair_historic.py into ``betfair_historic/markets_*.parquet``): each runner's last traded price 60, 15 and 1 minute
before the scheduled off, the close, the SP and the result, in the win and the place market.

1. The rule, with only what is known before the off: at ``t_1`` (and ``t_15``), Harville's chance of placing from the
   win market's last traded prices (normalised in the race), against the place market's last traded price; back at the
   place SP where p x (price - 1) x (1 - c) - (1 - p) clears a threshold, lay where (1 - p) x (1 - c) - p x (price - 1)
   does; settled at the place SP and the place result. Chosen on ``DEV_MONTHS`` (Jan-Apr), read on May on, by country
   (GB, AU), 90% intervals by a bootstrap over days. Bets at the SP: matched at the SP.
2. The closing price (the owner, 4 Nov: "fundamental is always just to get the most accurate closing price"): log of the
   win SP from the win market's last traded price at 1 minute out, against a LightGBM forecast (fitted on the
   development months) reading both markets' prices at 60, 15 and 1 minute out and the race's field: RMSE of log(SP),
   by country, on the later months.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from research.greyhound import place_value as pv

log = logging.getLogger(__name__)

DEV_MONTHS = (1, 2, 3, 4)
THRESHOLDS = (0.0, 0.05, 0.10, 0.20)
from greyhound.costs import COMMISSION as COMM   # the account's rate (greyhound/costs.py)
COLS = ["market_id", "event_id", "market_type", "country", "venue", "market_time", "number_of_winners",
        "selection_id", "runner_name", "runner_status", "won", "bsp", "ltp_t_60", "ltp_t_15", "ltp_t_1", "ltp_close"]


def load(store) -> pd.DataFrame:
    keys = sorted(k for k in store.listing("betfair_historic/")
                  if re.search(r"markets_greyhound.*\.parquet$", k))
    log.info("tables: %s", keys)
    parts = []
    for k in keys:
        t = store.get_parquet(k)
        if t is None or not len(t):
            continue
        t = t[t.market_type.isin(["WIN", "PLACE"])]
        parts.append(t[[c for c in COLS if c in t.columns]])
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["market_id", "selection_id"], keep="last")
    for c in ("bsp", "ltp_t_60", "ltp_t_15", "ltp_t_1", "ltp_close", "won", "number_of_winners"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["name"] = d.runner_name.astype(str).str.replace(r"^\d+\.\s*", "", regex=True).str.strip().str.lower()
    d["race"] = d.event_id.astype(str) + "|" + d.market_time.astype(str)
    return d


def pair(d: pd.DataFrame) -> pd.DataFrame:
    """A row a runner with its win (``_w``) and place (``_p``) market, races with every runner whole in both."""
    keep = ["race", "name", "country", "venue", "market_time", "number_of_winners", "runner_status", "won", "bsp",
            "ltp_t_60", "ltp_t_15", "ltp_t_1", "ltp_close"]
    w = d[d.market_type == "WIN"][keep]
    p = d[d.market_type == "PLACE"][keep]
    j = w.merge(p.drop(columns=["country", "venue", "market_time"]), on=["race", "name"], suffixes=("_w", "_p"))
    j = j[(j.runner_status_w.astype(str) != "REMOVED") & (j.runner_status_p.astype(str) != "REMOVED")]
    g = j.groupby("race")
    j["n"] = g.name.transform("size")
    j["k"] = j.number_of_winners_p
    j["placed_n"] = g.won_p.transform("sum")
    j["winners"] = g.won_w.transform("sum")
    j["day"] = pd.to_datetime(j.market_time, utc=True).dt.strftime("%Y-%m-%d")
    j["month"] = pd.to_datetime(j.market_time, utc=True).dt.month
    whole = (j.winners == 1) & j.k.isin([2, 3]) & (j.placed_n == j.k) & j.n.between(4, pv.MAXN) & (j.k < j.n)
    return j[whole].sort_values(["race", "name"]).reset_index(drop=True)


def harville_of(j: pd.DataFrame, col: str) -> np.ndarray:
    inv = 1.0 / j[col].where(j[col] > 1)
    ok = inv.groupby(j.race).transform(lambda s: s.notna().all())
    q = (inv / inv.groupby(j.race).transform("sum")).where(ok)
    codes, idx = np.unique(j.race.to_numpy(), return_inverse=True)
    pos = j.groupby("race").cumcount().to_numpy()
    Q = np.zeros((len(codes), pv.MAXN))
    Q[idx, pos] = q.fillna(0).to_numpy()
    K = j.groupby("race").k.first().reindex(codes).to_numpy()
    h = pv.harville(Q, K)[idx, pos]
    return np.where(ok.to_numpy(), h, np.nan)


def rules(j: pd.DataFrame, mark: str) -> list[dict]:
    p = harville_of(j, f"ltp_{mark}_w")
    price = j[f"ltp_{mark}_p"].to_numpy(float)
    sp = j.bsp_p.to_numpy(float)
    hit = j.won_p.to_numpy(float)
    dev = j.month.isin(DEV_MONTHS).to_numpy()
    out = []
    for country in ("GB", "AU"):
        cm = (j.country.astype(str) == country).to_numpy() & np.isfinite(p) & (price > 1) & (sp > 1)
        for side in ("back", "lay"):
            ev = (p * (price - 1) * (1 - COMM) - (1 - p)) if side == "back" else ((1 - p) * (1 - COMM) - p * (price - 1))
            pnl = np.where(hit == 1, (sp - 1) * (1 - COMM), -1.0) if side == "back" else np.where(hit == 1, -(sp - 1), 1 - COMM)
            for th in THRESHOLDS:
                sel = cm & (ev > th)
                row = {"mark": mark, "country": country, "side": side, "edge>": th}
                for part, m in (("dev", dev), ("holdout", ~dev)):
                    s = sel & m
                    if s.sum() < 50:
                        row[part] = {"bets": int(s.sum())}
                        continue
                    t = pd.DataFrame({"day": j.day.to_numpy()[s], "pnl": pnl[s]})
                    byd = t.groupby("day").agg(pnl=("pnl", "sum"), n=("pnl", "size"))
                    lo, hi = pv._boot(byd.pnl, byd.n)
                    row[part] = {"bets": int(s.sum()), "days": len(byd), "mean_sp": round(float(sp[s].mean()), 2),
                                 "roi%": round(100 * t.pnl.mean(), 2), "ci90%": [round(100 * lo, 2), round(100 * hi, 2)]}
                h, dv = row.get("holdout", {}), row.get("dev", {})
                row["passes"] = bool(h.get("ci90%") and h["ci90%"][0] > 0 and dv.get("roi%", -1) > 0)
                out.append(row)
    return out


def closing(j: pd.DataFrame) -> list[dict]:
    """log(win SP) from the win market's last price at 1 minute out, against a forecast reading both markets."""
    import lightgbm as lgb
    x = j[(j.bsp_w > 1) & (j.ltp_t_1_w > 1)].copy()
    x["y"] = np.log(x.bsp_w)
    feats = []
    for mk in ("w", "p"):
        for m in ("t_60", "t_15", "t_1"):
            c = f"ltp_{m}_{mk}"
            x[f"l_{m}_{mk}"] = np.log(x[c].where(x[c] > 1))
            feats.append(f"l_{m}_{mk}")
    inv = 1.0 / x.ltp_t_1_w
    x["q1"] = inv / inv.groupby(x.race).transform("sum")
    x["book1"] = inv.groupby(x.race).transform("sum")
    invp = 1.0 / x.ltp_t_1_p.where(x.ltp_t_1_p > 1)
    x["qp1"] = invp / invp.groupby(x.race).transform("sum") * x.k
    x["move_15_1"] = x.l_t_1_w - x.l_t_15_w
    x["place_vs_win"] = np.log(x.qp1.clip(1e-6)) - np.log(x.q1.clip(1e-6))
    x["is_au"] = (x.country.astype(str) == "AU").astype(int)
    feats += ["q1", "book1", "qp1", "move_15_1", "place_vs_win", "n", "k", "is_au"]
    dev = x.month.isin(DEV_MONTHS)
    params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 0}
    resid = x.y - x.l_t_1_w
    m = lgb.train(params, lgb.Dataset(x.loc[dev, feats], resid[dev]), 600)
    x["f"] = x.l_t_1_w + m.predict(x[feats])
    out = []
    for country in ("GB", "AU"):
        h = x[~dev & (x.country.astype(str) == country)]
        if not len(h):
            continue
        e0, e1 = h.y - h.l_t_1_w, h.y - h.f
        daily = pd.DataFrame({"day": h.day, "d": e1 ** 2 - e0 ** 2}).groupby("day").d.mean()
        rng = np.random.default_rng(0)
        bs = [daily.sample(len(daily), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 500)]
        out.append({"country": country, "runners": len(h), "rmse_ltp_t1": round(float(np.sqrt((e0 ** 2).mean())), 4),
                    "rmse_forecast": round(float(np.sqrt((e1 ** 2).mean())), 4),
                    "mse_diff_ci90": [round(float(np.percentile(bs, 5)), 5), round(float(np.percentile(bs, 95)), 5)]})
    imp = dict(zip(feats, m.feature_importance("gain")))
    tot = sum(imp.values()) or 1
    out.append({"importance": {k: round(v / tot, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}})
    return out


def run(store) -> dict:
    d = load(store)
    j = pair(d)
    rep = {"rows": len(d), "paired_runners": len(j), "races": int(j.race.nunique()),
           "by_country": j.drop_duplicates("race").country.value_counts().to_dict(),
           "months": sorted(int(m) for m in j.month.unique()),
           "ltp_t_1_share": {c: round(float(j[c].gt(1).mean()), 3) for c in ("ltp_t_1_w", "ltp_t_1_p", "bsp_p")}}
    rep["rules"] = rules(j, "t_1") + rules(j, "t_15")
    rep["passing"] = [r for r in rep["rules"] if r["passes"]]
    rep["rules_tried"] = len(rep["rules"])
    try:
        rep["closing"] = closing(j)
    except Exception as exc:
        rep["closing"] = {"error": str(exc)[:300]}
    return rep


if __name__ == "__main__":
    from sources.common import Store
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(Store()), indent=1, default=str))
