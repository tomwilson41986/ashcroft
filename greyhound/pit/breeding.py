"""Breeding as a prior: what the sire's and the dam's runners have shown, for the dogs with little form of their own.

For the sire and for the dam, from their progeny's runs on earlier days (trials included for the figures, not for
the A/E): runs, the mean speed rating (gsr), early pace (esr) and finishing speed (fsr), A/E against the SP (shrunk to
1), and the progeny's speed in today's distance band (sprint under 400 m, standard 400-549 m, staying 550 m and over)
against their speed overall: a stamina reading. A mean over fewer than ``MIN_N`` runs is NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "br_"
MIN_N = 20
K = 5.0
SOURCE = "GBGB results (sire, dam, distance, times, SP: sources/gbgb.py) through the engine's figures"


def band(dist: pd.Series) -> pd.Series:
    return pd.cut(dist, [0, 399, 549, 10_000], labels=["sprint", "standard", "stay"]).astype(str)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    df = df.assign(_band=band(df.distance_m))
    racing = ~df.is_trial.fillna(False) & df.sp_p_norm.notna()
    df["_w"] = df.won.where(racing).fillna(0.0)
    df["_e"] = df.sp_p_norm.where(racing).fillna(0.0)
    out = pd.DataFrame({"race_id": df.race_id, "dog_id": df.dog_id})
    for who in ("sire", "dam"):
        key = df[who].fillna("?").astype(str).str.strip().str.lower()
        d = df.assign(_k=key)
        for f in ("gsr", "esr", "fsr"):
            d[f"_{f}"] = d[f].fillna(0.0)
            d[f"_{f}n"] = d[f].notna().astype(float)
        cols = ["_w", "_e", "_gsr", "_gsrn", "_esr", "_esrn", "_fsr", "_fsrn"]
        w = core.prior_window(d, ["_k"], cols, None)
        known = key != "?"
        out[f"{PREFIX}{who}_runs"] = w.n.where(known)
        for f in ("gsr", "esr", "fsr"):
            out[f"{PREFIX}{who}_{f}"] = (w[f"_{f}"] / w[f"_{f}n"]).where(known & (w[f"_{f}n"] >= MIN_N))
        out[f"{PREFIX}{who}_ae"] = ((w._w + K) / (w._e + K)).where(known)
        b = core.prior_window(d, ["_k", "_band"], ["_gsr", "_gsrn"], None)
        in_band = (b._gsr / b._gsrn).where(known & (b._gsrn >= MIN_N))
        out[f"{PREFIX}{who}_band_vs_all"] = in_band - out[f"{PREFIX}{who}_gsr"]
    return out.reset_index(drop=True), {"min_n": MIN_N}


COLUMNS = {
    **{f"{PREFIX}{w}_runs": f"the {w}'s progeny's earlier runs" for w in ("sire", "dam")},
    **{f"{PREFIX}{w}_{f}": f"the {w}'s progeny's mean {f} (lengths)" for w in ("sire", "dam") for f in ("gsr", "esr", "fsr")},
    **{f"{PREFIX}{w}_ae": f"the {w}'s progeny's winners over the SP's expected (shrunk)" for w in ("sire", "dam")},
    **{f"{PREFIX}{w}_band_vs_all": f"the {w}'s progeny's speed in today's distance band less overall (stamina)"
       for w in ("sire", "dam")},
}
