"""The Australian early lay, closed at the BSP: the pre-registered test (reports/preregistration_au_early_lay.md).

Reads the stored WIN tables (au/historic.py) and the Betfair-only model's out-of-sample prices (au/model.py), applies
the rule fixed in the pre-registration, and scores it once against its gate. The variants beside it are reported, not
gated. No money, no Betfair key.

    python -m au.lay_test --out-key au/lay_test_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from au.historic import BANDS, runners_ran

COMMISSION = 0.02
PRIMARY = {"mark": "ltp_t_60", "model_filter": True, "lo": 1.5, "hi": 21.0, "settle": "green"}
GATE_MIN_BETS = 5000


def joined(store) -> pd.DataFrame:
    tables = sorted(k for k in store.listing("betfair_historic/") if re.fullmatch(r"betfair_historic/horse_racing_win_\d{4}\.parquet", k))
    h = runners_ran(pd.concat([store.get_parquet(k) for k in tables], ignore_index=True))
    h = h[h["country"] == "AU"].copy()
    h["market_id"] = h["market_id"].astype(str).str.replace(r"^1\.", "", regex=True).astype("int64")
    h["selection_id"] = h["selection_id"].astype("int64")
    o = store.get_parquet("au/model_report_oos.parquet")[["market_id", "selection_id", "p_model"]].copy()
    o["market_id"] = o["market_id"].astype("int64")
    o["selection_id"] = o["selection_id"].astype("int64")
    return h.merge(o, on=["market_id", "selection_id"], how="inner")


def bets(j: pd.DataFrame, mark: str, model_filter: bool, lo: float, hi: float) -> pd.DataFrame:
    x = j[(j[mark] > 1) & (j["bsp"] > 1)].copy()
    lay = x[mark]
    keep = (lay >= lo) & (lay <= hi)
    if model_filter:
        keep &= lay * x["p_model"] < 1
    s = x[keep].copy()
    s["L"] = s[mark]
    s["green"] = 1.0 - s["L"] / s["bsp"]                       # lay 1 at L, back L/BSP at the BSP: level
    s["result"] = np.where(s["won"] == 1, -(s["L"] - 1.0), 1.0)  # the lay left to the result
    return s


def score(s: pd.DataFrame, col: str, boot: int = 1000, seed: int = 5) -> dict:
    """Profit per unit laid after 2% commission on each market's net profit, with a 90% interval over markets."""
    if s.empty:
        return {"bets": 0}
    m = s.groupby("market_id").agg(net=(col, "sum"), n=(col, "size"), year=("year", "first"))
    m["after"] = m["net"] - COMMISSION * m["net"].clip(lower=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m), size=(boot, len(m)))
    a, n = m["after"].to_numpy(), m["n"].to_numpy()
    sims = a[idx].sum(1) / n[idx].sum(1)
    by_year = (m.groupby("year")["after"].sum() / m.groupby("year")["n"].sum()).round(5)
    return {"bets": int(len(s)), "markets": int(len(m)), "mean_lay_price": round(float(s["L"].mean()), 2),
            "per_unit_pct": round(100 * float(a.sum() / n.sum()), 3),
            "ci90_pct": [round(100 * float(np.percentile(sims, 5)), 3), round(100 * float(np.percentile(sims, 95)), 3)],
            "per_unit_before_commission_pct": round(100 * float(s[col].mean()), 3),
            "by_year_pct": {int(y): round(100 * float(v), 3) for y, v in by_year.items()}}


def by_band(s: pd.DataFrame, col: str) -> dict:
    out = {}
    for b, g in s.groupby(pd.cut(s["L"], BANDS), observed=True):
        r = score(g, col, boot=300)
        out[str(b)] = {k: r[k] for k in ("bets", "per_unit_pct", "ci90_pct")}
    return out


def run(j: pd.DataFrame) -> dict:
    p = PRIMARY
    prim = bets(j, p["mark"], p["model_filter"], p["lo"], p["hi"])
    sc = score(prim, p["settle"])
    years = sc.get("by_year_pct", {})
    gate = {"profit_above_zero_with_interval": bool(sc.get("ci90_pct", [0])[0] > 0),
            "every_year_above_zero": bool(years) and all(years.get(y, -1) > 0 for y in (2024, 2025, 2026)),
            "enough_bets": sc.get("bets", 0) >= GATE_MIN_BETS}
    gate["passed"] = all(gate.values())
    rep = {"rule": p, "universe": {"runners_joined": int(len(j)), "markets": int(j["market_id"].nunique()),
                                   "dates": [str(j["market_time"].min()), str(j["market_time"].max())]},
           "primary": sc, "gate": gate, "primary_by_band": by_band(prim, "green")}
    v = {}
    v["every_runner_t60"] = score(bets(j, "ltp_t_60", False, 1.5, 21.0), "green")
    v["model_filter_t15"] = score(bets(j, "ltp_t_15", True, 1.5, 21.0), "green")
    v["every_runner_t15"] = score(bets(j, "ltp_t_15", False, 1.5, 21.0), "green")
    v["model_filter_t60_over_21"] = score(bets(j, "ltp_t_60", True, 21.0, 1000.0), "green")
    v["model_filter_t60_settled_by_result"] = score(prim, "result")
    rep["variants_not_gated"] = v
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-key", default="au/lay_test_report.json")
    a = ap.parse_args(argv)
    from sources.common import Store
    store = Store()
    rep = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "preregistration": "reports/preregistration_au_early_lay.md (commit 0fdac24)", **run(joined(store))}
    store.put(a.out_key, json.dumps(rep, indent=1, default=str).encode())
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
