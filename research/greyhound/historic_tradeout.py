"""Trading to the SP on a forecast of the close (read-only research; the owner, 4 Nov 2026: "fundamental is always
just to get the most accurate closing price, optimise towards this"). Betfair's historic BASIC data (Jan-Sep 2026,
GB and AU; research/greyhound/historic_place.py loads it): at 1 minute out, the win SP forecast from both markets'
last traded prices (research.greyhound.historic_place.closing's model, fitted on the development months); then

- back now where the price we would get beats the forecast close by a margin, and close at the SP: back 1 at B, lay
  B/S at the SP; the result locked whatever wins: B/S - 1, less commission on a gain;
- lay now where the price we would lay at is under the forecast close by a margin, and close at the SP: 1 - L/S.

The price we would get: the data has only each runner's last traded price (no book), so a back is taken one tick
below it and a lay one tick above (``TICKS_THROUGH``), a price worse than the last trade, which is the side the
spread sits on near the off. Liquidity is not modelled: a unit stake each. Chosen on Jan-Apr, read on May-Sep, by
country, 90% intervals by a bootstrap over days.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from research.greyhound import historic_place as hp, limit_study as ls, place_value as pv

log = logging.getLogger(__name__)

MARGINS = (0.0, 0.02, 0.05, 0.10)
TICKS_THROUGH = (1, 2)
#: the last traded price of a long shot can be minutes old near the off and its book many ticks wide, so a fill a tick
#: through it is not credible there (run 41895217384: the 'passes' came from entries averaging 89-390): the rules are
#: read with the entry price capped
MAX_ENTRY = (5.0, 10.0, 20.0)
from greyhound.costs import COMMISSION as COMM   # the account's rate (greyhound/costs.py)


def forecast(j: pd.DataFrame) -> pd.DataFrame:
    """The win SP forecast at 1 minute out (fitted on the development months only)."""
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
    dev = x.month.isin(hp.DEV_MONTHS)
    params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 0}
    m = lgb.train(params, lgb.Dataset(x.loc[dev, feats], (x.y - x.l_t_1_w)[dev]), 600)
    x["f_bsp"] = np.exp(x.l_t_1_w + m.predict(x[feats]))
    x["dev"] = dev
    x.attrs["model"], x.attrs["feats"] = m, feats
    return x


def features_from(x: pd.DataFrame) -> pd.DataFrame:
    """The forecast's inputs from a frame carrying ``ltp_<mark>_<w|p>``, ``race``, ``n``, ``k``, ``country``
    (the same transforms as ``forecast``)."""
    x = x.copy()
    for mk in ("w", "p"):
        for m in ("t_60", "t_15", "t_1"):
            c = f"ltp_{m}_{mk}"
            x[f"l_{m}_{mk}"] = np.log(pd.to_numeric(x.get(c), errors="coerce").where(lambda v: v > 1))
    inv = 1.0 / x.ltp_t_1_w
    x["q1"] = inv / inv.groupby(x.race).transform("sum")
    x["book1"] = inv.groupby(x.race).transform("sum")
    invp = 1.0 / pd.to_numeric(x.get("ltp_t_1_p"), errors="coerce").where(lambda v: v > 1)
    x["qp1"] = invp / invp.groupby(x.race).transform("sum") * x.k
    x["move_15_1"] = x.l_t_1_w - x.l_t_15_w
    x["place_vs_win"] = np.log(x.qp1.clip(1e-6)) - np.log(x.q1.clip(1e-6))
    x["is_au"] = (x.country.astype(str) == "AU").astype(int)
    return x


def rules(x: pd.DataFrame) -> list[dict]:
    out = []
    for ticks in TICKS_THROUGH:
        for cap in MAX_ENTRY:
            out += _rules(x, ticks, cap)
    return out


def _rules(x: pd.DataFrame, ticks: int, cap: float) -> list[dict]:
    ltp = x.ltp_t_1_w.to_numpy(float)
    back_at = ls.ticks_from(ltp, ticks, up=False)               # a back taken ticks worse than the last trade
    lay_at = ls.ticks_from(ltp, ticks, up=True)
    sp, f = x.bsp_w.to_numpy(float), x.f_bsp.to_numpy(float)
    dev = x.dev.to_numpy()
    out = []
    for country in ("GB", "AU"):
        cm = (x.country.astype(str) == country).to_numpy() & np.isfinite(f) & (sp > 1)
        for side in ("back", "lay"):
            if side == "back":
                edge = back_at / f - 1
                raw = back_at / sp - 1
            else:
                edge = 1 - lay_at / f
                raw = 1 - lay_at / sp
            pnl = np.where(raw > 0, raw * (1 - COMM), raw)
            for m in MARGINS:
                entry = back_at if side == "back" else lay_at
                sel = cm & np.isfinite(edge) & (edge > m) & (entry <= cap)
                row = {"country": country, "side": side, "margin": m, "ticks": ticks, "max_entry": cap}
                for part, mask in (("dev", dev), ("holdout", ~dev)):
                    s = sel & mask
                    if s.sum() < 50:
                        row[part] = {"trades": int(s.sum())}
                        continue
                    t = pd.DataFrame({"day": x.day.to_numpy()[s], "pnl": pnl[s]})
                    byd = t.groupby("day").agg(pnl=("pnl", "sum"), n=("pnl", "size"))
                    lo, hi = pv._boot(byd.pnl, byd.n)
                    row[part] = {"trades": int(s.sum()), "days": len(byd), "mean_entry": round(float(
                        (back_at if side == "back" else lay_at)[s].mean()), 2), "mean_pnl%": round(100 * t.pnl.mean(), 3),
                                 "ci90%": [round(100 * lo, 3), round(100 * hi, 3)],
                                 "win_share": round(float((pnl[s] > 0).mean()), 3)}
                h, dv = row.get("holdout", {}), row.get("dev", {})
                row["passes"] = bool(h.get("ci90%") and h["ci90%"][0] > 0 and dv.get("mean_pnl%", -1) > 0)
                out.append(row)
    return out


def run(store) -> dict:
    j = hp.pair(hp.load(store))
    x = forecast(j)
    rep = {"runners": len(x), "rules": rules(x)}
    rep["passing"] = [r for r in rep["rules"] if r["passes"]]
    rep["rules_tried"] = len(rep["rules"])
    for c in ("GB", "AU"):
        h = x[~x.dev & (x.country.astype(str) == c)]
        if len(h):
            rep[f"rmse_{c}"] = {"ltp_t_1": round(float(np.sqrt(np.mean((np.log(h.bsp_w) - np.log(h.ltp_t_1_w)) ** 2))), 4),
                                "forecast": round(float(np.sqrt(np.mean((np.log(h.bsp_w) - np.log(h.f_bsp)) ** 2))), 4)}
    return rep


if __name__ == "__main__":
    from sources.common import Store
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(Store()), indent=1, default=str))
