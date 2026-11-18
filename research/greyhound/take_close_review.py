"""The take and close's backs against the SP that followed (read-only): each back's price, the forecast SP, the mid
at the first traded book, the Betfair SP and the result, from its ledger and the day's settled live-key books
(greyhound/fav_back.py ``closing``). Did the backed dogs shorten (SP under the price taken), as the rule expects?

    python -m research.greyhound.take_close_review --date 2026-11-18
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import date

import numpy as np
import pandas as pd


def review(led: pd.DataFrame, close: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    b = led[led.action == "back_matched"].copy()
    if not len(b):
        return b, {"backs": 0}
    b["market_id"] = b.market_id.astype(str)
    b["selection_id"] = pd.to_numeric(b.selection_id, errors="coerce")
    for c in ("price", "matched", "f_bsp", "mid", "back1", "edge"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.merge(close, on=["market_id", "selection_id"], how="left")
    ok = b.bsp.notna() & (b.bsp > 1)
    b["clv%"] = np.where(ok, 100 * (b.price / b.bsp - 1), np.nan)            # + : the dog shortened after the back
    b["drifted"] = np.where(ok, b.bsp > b.price, np.nan)
    b["fc_err%"] = np.where(ok, 100 * (b.bsp / b.f_bsp - 1), np.nan)          # the SP against our forecast
    b["pnl"] = np.where(ok, np.where(b.won == 1, 0.0, -b.matched + b.matched * (b.price - 1) / (b.bsp - 1)), np.nan)
    s = b[ok]
    w = s.matched
    out = {"backs": int(len(b)), "with_sp": int(ok.sum()), "staked": round(float(b.matched.sum()), 2),
           "drifted": int((s.bsp > s.price).sum()), "shortened": int((s.bsp < s.price).sum()),
           "clv_mean%": round(float(s["clv%"].mean()), 2) if len(s) else None,
           "clv_stake_weighted%": round(float((s["clv%"] * w).sum() / w.sum()), 2) if len(s) else None,
           "sp_vs_forecast_mean%": round(float(s["fc_err%"].mean()), 2) if len(s) else None,
           "sp_vs_mid_mean%": round(float((100 * (s.bsp / s.mid - 1)).mean()), 2) if len(s) else None,
           "price_vs_mid_mean%": round(float((100 * (s.price / s.mid - 1)).mean()), 2) if len(s) else None,
           "winners": int((s.won == 1).sum()), "pnl_before_commission": round(float(s.pnl.sum()), 2)}
    return b, out


def main(argv=None) -> int:
    from greyhound.fav_back import closing
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--mode", default="live")
    a = ap.parse_args(argv)
    day = date.fromisoformat(a.date)
    raw = Store().get(f"greyhound/take_close/{a.mode}/{day:%Y-%m-%d}/ledger.csv")
    if not raw:
        print("no data found: no ledger")
        return 0
    b, out = review(pd.read_csv(io.BytesIO(raw), dtype={"market_id": str}), closing(day))
    cols = ["time_utc", "track", "race_time", "trap", "dog", "mid", "f_bsp", "price", "matched", "bsp", "won",
            "clv%", "fc_err%", "pnl"]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(b[[c for c in cols if c in b]].round(3).to_string(index=False))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
