#!/usr/bin/env python3
"""The Blandford site's sectionals as one table, a row a run (the owner, 22 Oct: "bring them through and test them").

The site (Abiqb93/horses-website-deployed, public/data/rtv/) holds two runs of files:

- history/meetings/: the racing2 export, TPD and Timeform sectionals, UK/IE (and French) Flat and all-weather,
  30 Jun 2024 - 29 Jun 2025: per runner the finishing speed (fsp), stride length (sl), the last four splits and
  three phases (early, middle, late, seconds);
- meetings/: Racing TV and Timeform's PDFs, UK/IE from 18 Apr 2026, refreshed twice a day: per runner the finishing
  speed, per-furlong splits and positions, stride length.

Both are read to the readings they share, each against the field it met in that race (nothing from any other race):

    fsp          finishing speed, % of the horse's own average speed for the race (both sources)
    fsp_rel      fsp less the field's mean
    fsp_vs_pos   its fsp rank in the field (1 the fastest finisher, as a share) less its finishing rank (1 the
                 winner, as a share): it finished faster than it placed when positive
    early_pct    its place early, as a share of the field (0 in front): the early phase's time (history) or the
                 time over the first quarter of the furlongs (from 2026), ranked in the field
    late_rel     its last two furlongs' time less the field's mean, seconds (below zero: quicker late)
    sl_rel       its stride length less the field's mean, metres

French meetings are left out (our history has none). The block (model/blocks/sectionals.py) reads each run only on
the days after it.

Usage
    python scripts/build_sectionals.py --site /home/user/horses-website-deployed/public/data/rtv \\
        --out data/external/sectionals.csv.gz
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.blocks.sectionals import norm_name  # noqa: E402

FRANCE = {"CHANTILLY", "COMPIEGNE", "PARIS-LONGCHAMP", "SAINT CLOUD", "LYON PARILLY", "LA TESTE DE BUCH", "DIEPPE",
          "ARGENTAN", "BORDEAUX LE BOUSCAT", "MARSEILLE BORELY", "MARSEILLE VIVAUX", "CRAON", "NANCY", "NANTES",
          "SALON DE PROVENCE", "TOULOUSE", "LE LION DANGERS", "STRASBOURG", "TARBES", "DEAUVILLE", "MAISONS-LAFFITTE",
          "FONTAINEBLEAU", "VICHY", "CLAIREFONTAINE", "CAGNES-SUR-MER", "PAU", "AUTEUIL", "ENGHIEN", "VINCENNES"}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if math.isfinite(f) else np.nan


def _history_runs(path: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(path, "history", "meetings", "*.json"))):
        with open(p) as fh:
            d = json.load(fh)
        if str(d.get("track", "")).upper() in FRANCE:
            continue
        for r in d.get("races", []):
            for x in r.get("runners", []):
                sp = x.get("splits") or []
                ph = x.get("phases") or []
                rows.append({"day": d["date"], "track": d["track"], "time": r.get("time"), "src": "history",
                             "horse": x.get("horse"), "pos": _num(x.get("pos")), "fsp": _num(x.get("fsp")),
                             "sl": _num(x.get("sl")), "early": _num(ph[0]) if len(ph) == 3 else np.nan,
                             "late2": _num(sp[-2]) + _num(sp[-1]) if len(sp) >= 2 else np.nan})
    return rows


def _season_runs(path: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(path, "meetings", "*.json"))):
        if "_fr-" in os.path.basename(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        if str(d.get("track", "")).upper() in FRANCE:
            continue
        for r in d.get("races", []):
            for x in r.get("runners", []):
                sp = [_num(v) for v in (x.get("splits") or [])]
                q = max(1, math.ceil(len(sp) / 4))
                early = sum(sp[:q]) if len(sp) >= 4 and all(math.isfinite(v) for v in sp[:q]) else np.nan
                late = sp[-2] + sp[-1] if len(sp) >= 2 else np.nan
                rows.append({"day": d["date"], "track": d["track"], "time": r.get("time"), "src": "season",
                             "horse": x.get("horse"), "pos": _num(x.get("pos")), "fsp": _num(x.get("fsp")),
                             "sl": _num((x.get("strideLength") or {}).get("value")), "early": early, "late2": late})
    return rows


def build(site: str) -> pd.DataFrame:
    h = pd.DataFrame(_history_runs(site) + _season_runs(site))
    h["name_norm"] = h["horse"].map(norm_name)
    h = h[h["name_norm"].str.len() > 0].copy()
    h["race"] = h["day"].astype(str) + "|" + h["track"].astype(str) + "|" + h["time"].astype(str)
    g = h.groupby("race")
    h["n_field"] = g["horse"].transform("size")
    h["fsp_rel"] = h["fsp"] - g["fsp"].transform("mean")
    fsp_rank = g["fsp"].rank(ascending=False, pct=True)              # small = the field's fastest finisher
    pos_rank = g["pos"].rank(ascending=True, pct=True)               # small = the winner
    h["fsp_vs_pos"] = pos_rank - fsp_rank                            # > 0: finished faster than it placed
    er = g["early"].rank(ascending=True, method="average")
    n_e = g["early"].transform("count")
    h["early_pct"] = (er - 1) / (n_e - 1).where(n_e > 1)             # 0 = in front early
    h["late_rel"] = h["late2"] - g["late2"].transform("mean")
    h["sl_rel"] = h["sl"] - g["sl"].transform("mean")
    h.loc[h["n_field"] < 3, ["fsp_rel", "fsp_vs_pos", "early_pct", "late_rel", "sl_rel"]] = np.nan
    cols = ["day", "name_norm", "track", "src", "n_field", "pos", "fsp", "fsp_rel", "fsp_vs_pos", "early_pct",
            "late_rel", "sl_rel"]
    out = h[cols].sort_values(["day", "track", "name_norm"]).reset_index(drop=True)
    for c in ("fsp", "fsp_rel", "fsp_vs_pos", "early_pct", "late_rel", "sl_rel"):
        out[c] = out[c].round(4)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", required=True, help="the site's public/data/rtv folder")
    ap.add_argument("--out", default="data/external/sectionals.csv.gz")
    a = ap.parse_args(argv)
    t = build(a.site)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t.to_csv(a.out, index=False, compression="gzip")
    by = t.groupby("src").agg(runs=("day", "size"), first=("day", "min"), last=("day", "max"))
    print(by.to_string())
    print("readings present:", {c: f"{t[c].notna().mean():.1%}" for c in ("fsp_rel", "fsp_vs_pos", "early_pct",
                                                                           "late_rel", "sl_rel")})
    print(f"wrote {a.out}: {len(t):,} runs, {os.path.getsize(a.out) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
