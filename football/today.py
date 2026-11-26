"""The football traders' day so far, read from S3 (read-only): back-to-lay trades and the value bets, paper and live.

racing² football, 26 Nov 2026 (the owner: "do we have any bets / trades today?"). The trader's run log is only
readable when its 23-hour run ends; its records are in S3 within 10 minutes of each decision:
``football/b2l/trades/<UK day>.jsonl`` (back-to-lay) and ``football/live/<paper|live>/<UK day>/ledger.csv`` (value
bets). Nothing is placed or changed.

    python -m football.today --day 2026-11-26 --out out/football_today.json --md out/football_today.md
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd


def b2l_day(store, day: str) -> dict:
    from football import b2l
    from football.b2l_trader import TRADES_PREFIX
    model = b2l.load(store)
    out = {"model": {k: model.get(k) for k in ("fitted_utc", "sha256", "trained_on")} if model else "no data found"}
    raw = store.get(f"{TRADES_PREFIX}/{day}.jsonl")
    if not raw:
        out["trades"] = "no data found (no b2l records for the day)"
        return out
    recs = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    d = pd.DataFrame(recs)
    out["records"] = int(len(d))
    dec = d[d["kind"] == "decision"] if "kind" in d else d.iloc[:0]
    fc = dec[dec["forecast"].notna()] if "forecast" in dec else dec.iloc[:0]
    out["checkpoints_read"] = int(dec[["market_id", "cp_min"]].drop_duplicates().shape[0]) if len(dec) else 0
    out["forecasts"] = int(len(fc))
    out["cells_cleared"] = int(fc["cell"].notna().sum()) if "cell" in fc else 0
    out["no_forecast_why"] = dec.loc[dec.get("outcome", pd.Series(index=dec.index, dtype=object)).notna(),
                                     "outcome"].value_counts().to_dict() if "outcome" in dec else {}
    if len(fc):
        out["forecast_quantiles"] = {q: round(float(fc["forecast"].quantile(q)), 4) for q in (0.5, 0.9, 0.99)}
        out["by_checkpoint"] = fc.groupby("cp_min").size().to_dict()
    tr = d[d["kind"].isin(["back", "lay"])] if "kind" in d else d.iloc[:0]
    rows = []
    for r in tr.to_dict("records"):
        rows.append({k: r.get(k) for k in ("at", "mode", "kind", "market_id", "selection_id", "cell", "price", "size",
                                           "status", "matched", "avg_price", "outcome", "locked_profit_before_commission")})
    out["orders"] = rows
    for mode in ("live", "paper"):
        b = tr[(tr["kind"] == "back") & (tr["mode"] == mode)] if len(tr) else tr
        ok = b[b.get("status", pd.Series(dtype=str)) == "SUCCESS"] if len(b) else b
        out[f"{mode}_backs_matched"] = int(len(ok))
        out[f"{mode}_staked_gbp"] = float(pd.to_numeric(ok.get("matched", pd.Series(dtype=float)), errors="coerce").sum()) if len(ok) else 0.0
    return out


def value_day(store, day: str) -> dict:
    out = {}
    for mode in ("live", "paper"):
        raw = store.get(f"football/live/{mode}/{day}/ledger.csv")
        if not raw:
            out[mode] = "no data found"
            continue
        d = pd.read_csv(io.BytesIO(raw))
        st = pd.to_numeric(d.get("stake"), errors="coerce").fillna(0)
        bets = d[st > 0]
        out[mode] = {"rows": int(len(d)), "bets": int(len(bets)),
                     "matched_gbp": round(float(pd.to_numeric(bets.get("matched"), errors="coerce").fillna(0).sum()), 2),
                     "by_action": bets["action"].value_counts().to_dict() if len(bets) else {},
                     "bets_list": bets[[c for c in ("time_utc", "event", "outcome", "action", "back", "lay", "stake",
                                                    "matched", "avg_price", "status") if c in bets.columns]]
                     .tail(30).to_dict("records")}
    return out


def markdown(res: dict) -> str:
    b, v = res["b2l"], res["value"]
    lines = [f"# racing² football - the traders' day, {res['day']}", "", "Source: S3 `football/b2l/trades/` and "
             "`football/live/<mode>/<day>/ledger.csv` (`football/today.py`).", "", "## Back-to-lay", ""]
    if isinstance(b.get("trades"), str):
        lines.append(b["trades"])
    else:
        lines.append(f"Model fitted {b['model'].get('fitted_utc') if isinstance(b['model'], dict) else b['model']}; "
                     f"checkpoints read {b['checkpoints_read']}, runner forecasts {b['forecasts']} "
                     f"(quantiles {b.get('forecast_quantiles')}), cells cleared {b['cells_cleared']}, no forecast: "
                     f"{b['no_forecast_why']}; live backs matched "
                     f"{b['live_backs_matched']} (GBP {b['live_staked_gbp']:.2f}), paper {b['paper_backs_matched']}.")
        if b["orders"]:
            lines += ["", "| At (UTC) | Mode | Kind | Market | Selection | Cell | Price | Status | Matched | Locked |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for o in b["orders"]:
                lines.append(f"| {o['at']} | {o['mode']} | {o['kind']} | {o['market_id']} | {o['selection_id']} | "
                             f"{o.get('cell') or ''} | {o.get('price')} | {o.get('status') or o.get('outcome')} | "
                             f"{o.get('matched')} | {o.get('locked_profit_before_commission') or ''} |")
    lines += ["", "## Value bets", ""]
    for mode in ("live", "paper"):
        x = v[mode]
        lines.append(f"- {mode}: " + (x if isinstance(x, str) else f"{x['bets']} bets, GBP {x['matched_gbp']} matched, "
                                       f"{x['by_action']}"))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", default=date.today().isoformat())
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    from sources.common import Store
    s = Store()
    res = {"day": a.day, "b2l": b2l_day(s, a.day), "value": value_day(s, a.day)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    if a.md:
        Path(a.md).write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
