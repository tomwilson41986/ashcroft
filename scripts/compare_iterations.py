#!/usr/bin/env python3
"""Put research-loop iterations side by side.

    python scripts/compare_iterations.py path/to/result_iter1.json path/to/result_iter2.json ...

Reads the result_<tag>.json files the rank-1 scorer writes and prints one row
per iteration: the model's log-likelihood gain over the market, rank-1 ROI,
the EV-filtered rules, and the quarter-Kelly bank. Development figures only --
the holdout is scored separately, once, and says so in its own header.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RULES = ("rank1", "rank1_ev>0", "rank1_ev>0.02", "rank1_ev>0.05", "rank1_ev>0.1")


def row(res: dict) -> dict:
    s, ll, c = res["strategies"], res["loglik"], res["config"]
    out = {"tag": c["tag"], "mode": c.get("mode", "offset"), "features": c["features"],
           "dll_mnats": ll["dll_mnats"], "dll_t": ll["t"], "fav_roi": s["fav"]["roi"]}
    for r in RULES:
        v = s.get(r, {})
        out[f"{r}_bets"] = v.get("bets", 0)
        out[f"{r}_roi"] = v.get("roi")
        out[f"{r}_lo90"] = v.get("lo90")
        h = s.get(f"{r}_halves", [None, None])
        out[f"{r}_halves"] = h
    k = s.get("kelly", {}).get("rank1_ev>0", {})
    out["kelly_ev0_bank"] = k.get("final_bank")
    return out


def main(paths):
    rows = [row(json.loads(Path(p).read_text())) for p in paths]
    pct = lambda x: "" if x is None else f"{x:+.2%}"
    print("| iteration | mode | feats | ΔLL mnats (t) | fav ROI | rank-1 ROI (lo90) | EV>0 bets | EV>0 ROI (lo90) | "
          "EV>0 halves | EV>0.05 bets | EV>0.05 ROI (lo90) | ¼-Kelly EV>0 bank |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        h = r["rank1_ev>0_halves"]
        bank = "" if r["kelly_ev0_bank"] is None else f"{r['kelly_ev0_bank']:.3f}"
        print(f"| {r['tag']} | {r['mode']} | {r['features']} | {r['dll_mnats']:+.2f} ({r['dll_t']:+.1f}) | {pct(r['fav_roi'])} | "
              f"{pct(r['rank1_roi'])} ({pct(r['rank1_lo90'])}) | {r['rank1_ev>0_bets']:,} | "
              f"{pct(r['rank1_ev>0_roi'])} ({pct(r['rank1_ev>0_lo90'])}) | {pct(h[0])} / {pct(h[1])} | "
              f"{r['rank1_ev>0.05_bets']:,} | {pct(r['rank1_ev>0.05_roi'])} ({pct(r['rank1_ev>0.05_lo90'])}) | {bank} |")


if __name__ == "__main__":
    main(sys.argv[1:])
