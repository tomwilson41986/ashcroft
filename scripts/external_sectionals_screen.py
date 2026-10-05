#!/usr/bin/env python3
"""Do a horse's past sectionals know something the market does not?

A first screen of the sectional history the Blandford site holds (the racing2
export: TPD and Timeform sectionals, UK/IE Flat and all-weather, 30 Jun 2024 to
29 Jun 2025; horses-website-deployed/public/data/rtv/history). Every feature is
read from the horse's sectional runs strictly before the race, so nothing from
the day itself reaches it.

The test is residual_screen.py's (Benter's): a conditional logit on each race's
winner,

    P(i wins) ∝ exp(a1·ln π_i + a2·(ln π_i)² + b·f_i)

against the market alone (b = 0), on races the fit never saw, in millinats per
race. π is the race-normalised Betfair SP. The same is run with our model's
probability beside the market (an older walk-forward model's out-of-sample
predictions, data/oos_predictions.csv), and the forecast check asks whether the
features cut the within-race error of that model's log-BSP forecast.

Windows: the features are fresh only while the export runs, so the screens are
cross-fitted by calendar-month parity on Aug 2024 - Jun 2025 (fit on even
months, score odd, and the other way round), and checked forward (fit Aug 2024 -
Jan 2025, score Feb - Jun 2025). Nothing on or after 1 Apr 2026 (the locked
holdout) is read: the OOS file stops on 21 Mar 2026 and the export on 29 Jun 2025.

Usage
    python scripts/external_sectionals_screen.py \\
        --history /home/user/horses-website-deployed/public/data/rtv/history/meetings \\
        --oos data/oos_predictions.csv --out reports/external_sectionals_screen.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from betfair_prices import normalise_horse  # noqa: E402
from residual_screen import (OutcomeScreen, MoveScreen, Part, feature_design, gain_summary,  # noqa: E402
                             joint_design, market_columns)

FRANCE = {"CHANTILLY", "COMPIEGNE", "PARIS-LONGCHAMP", "SAINT CLOUD", "LYON PARILLY", "LA TESTE DE BUCH", "DIEPPE",
          "ARGENTAN", "BORDEAUX LE BOUSCAT", "MARSEILLE BORELY", "MARSEILLE VIVAUX", "CRAON", "NANCY", "NANTES",
          "SALON DE PROVENCE", "TOULOUSE", "LE LION DANGERS", "STRASBOURG", "TARBES"}

SECTIONAL = ["fsp_last", "fsp_rel_last", "fsp_rel_mean3", "fsp_vs_pos_last", "fsp_vs_pos_mean3", "early_rel_last",
             "early_rel_mean3", "late_rel_last", "late_rel_mean3", "s0p_last", "secPct_last", "timePct_last",
             "sl_last", "slPct_last", "cad_last", "cadPct_last", "best_fsp_rel3"]
CONTEXT = ["sec_n", "sec_days", "pos_pct_last"]
RATINGS = ["r_tf_last", "r_tfig_last", "r_rpr_last"]


def load_history(path: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(path, "*.json"))):
        with open(p) as fh:
            d = json.load(fh)
        if d["track"] in FRANCE:
            continue
        for r in d["races"]:
            for x in r["runners"]:
                rt = x.get("ratings") or {}
                sp = x.get("splits") or [None] * 4
                spp = x.get("splitPcts") or [None] * 4
                ph = x.get("phases") or [None] * 3
                rows.append(dict(date=d["date"], track=d["track"], time=r["time"], field=len(r["runners"]),
                                 pos=x.get("pos"), horse=x.get("horse"), fsp=x.get("fsp"), sl=x.get("sl"),
                                 slPct=x.get("slPct"), cad=x.get("cad"), cadPct=x.get("cadPct"),
                                 timePct=x.get("timePct"), secPct=x.get("secPct"), s0=sp[3], s0p=spp[3],
                                 ph0=ph[0], ph2=ph[2], r_or=rt.get("or"), r_rpr=rt.get("rpr"), r_tf=rt.get("tf"),
                                 r_tfig=rt.get("tfig")))
    h = pd.DataFrame(rows)
    for c in h.columns.difference(["date", "track", "time", "horse"]):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h["horse_norm"] = h["horse"].map(normalise_horse)
    h["date"] = pd.to_datetime(h["date"])
    race = h["date"].astype(str) + "|" + h["track"] + "|" + h["time"]
    h["race"] = race

    # Within-race readings: each run against the field it met, nothing from any other race.
    g = h.groupby("race")
    h["pos_pct"] = (h["pos"] - 1) / (h["field"] - 1).clip(lower=1)
    h["fsp_rel"] = h["fsp"] - g["fsp"].transform("mean")
    h["early_rel"] = h["ph0"] - g["ph0"].transform("mean")          # seconds; below zero = quicker early
    h["late_rel"] = h["ph2"] - g["ph2"].transform("mean")           # seconds; below zero = quicker late
    fsp_rank = g["fsp"].rank(pct=True)                              # 1 = the field's fastest finisher
    pos_rank = g["pos"].rank(pct=True, ascending=False)             # 1 = the winner
    h["fsp_vs_pos"] = fsp_rank - pos_rank                           # finished faster than it placed
    return h.sort_values(["horse_norm", "date"]).reset_index(drop=True)


def lag_features(sample: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    """For each runner, its sectional runs strictly before the race date (merge_asof on date - 1 day)."""
    cols = ["fsp", "fsp_rel", "fsp_vs_pos", "early_rel", "late_rel", "s0p", "secPct", "timePct", "sl", "slPct",
            "cad", "cadPct", "pos_pct", "r_tf", "r_tfig", "r_rpr"]
    hh = h[["horse_norm", "date"] + cols].copy()
    hh = hh.drop_duplicates(["horse_norm", "date"], keep="last")   # one run a day per name
    gg = hh.groupby("horse_norm")
    roll = {}
    for c in ("fsp_rel", "fsp_vs_pos", "early_rel", "late_rel"):
        roll[f"{c}_mean3"] = gg[c].transform(lambda s: s.rolling(3, min_periods=1).mean())
    hh = hh.assign(**roll)
    hh["best_fsp_rel3"] = gg["fsp_rel"].transform(lambda s: s.rolling(3, min_periods=1).max())
    hh["sec_n"] = gg.cumcount() + 1
    hh["sec_date"] = hh["date"]
    hh = hh.rename(columns={c: f"{c}_last" for c in cols})
    hh["key_date"] = hh["date"] + pd.Timedelta(days=1)              # usable from the next day on
    s = sample[["row", "horse_norm", "date"]].sort_values("date")
    m = pd.merge_asof(s, hh.drop(columns=["date"]).sort_values("key_date"), left_on="date", right_on="key_date",
                      by="horse_norm", direction="backward")
    m["sec_days"] = (m["date"] - m["sec_date"]).dt.days
    return m.set_index("row").drop(columns=["horse_norm", "date", "key_date", "sec_date"])


def build_sample(oos_path: str) -> pd.DataFrame:
    o = pd.read_csv(oos_path, low_memory=False)
    o = o[o["race_code"].isin(["Flat", "All Weather"])].copy()
    o["date"] = pd.to_datetime(o["race_date"])
    o["race"] = o["race_date"].astype(str) + "|" + o["track"].astype(str) + "|" + o["race_time"].astype(str)
    o["bfsp"] = pd.to_numeric(o["bfsp"], errors="coerce")
    ok = o.groupby("race").agg(n=("bfsp", "size"), good=("bfsp", lambda s: bool((s > 1).all())),
                               wins=("won", "sum"))
    keep = ok.index[(ok["good"]) & (ok["wins"] == 1) & (ok["n"] >= 2)]
    o = o[o["race"].isin(keep)].copy()
    inv = 1.0 / o["bfsp"]
    o["ln_pi"] = np.log(inv / inv.groupby(o["race"]).transform("sum"))
    p = o["predicted_win_prob_norm"].clip(lower=1e-6)
    o["ln_model"] = np.log(p / p.groupby(o["race"]).transform("sum"))
    o["ln_bsp"] = np.log(o["bfsp"])
    o["ln_pred_bsp"] = np.log(pd.to_numeric(o["predicted_bfsp"], errors="coerce").clip(lower=1.01))
    o["y"] = o["won"].astype(float)
    o["horse_norm"] = o["horse_name"].map(normalise_horse)
    o = o.sort_values(["date", "race"], kind="stable").reset_index(drop=True)
    o["row"] = np.arange(len(o))
    return o


def parts_for(sample: pd.DataFrame, mask_tr: np.ndarray, mask_te: np.ndarray, mcol: str = "ln_pi"):
    codes = pd.factorize(sample["race"])[0]
    pos = np.arange(len(sample))
    y = sample["y"].to_numpy(float)
    M = market_columns(sample[mcol].to_numpy(float))
    return (Part(pos[mask_tr], codes[mask_tr], y[mask_tr], M[mask_tr]),
            Part(pos[mask_te], codes[mask_te], y[mask_te], M[mask_te]))


def design_present(values, train_mask, quadratic=True):
    """z (and z²) after winsorising on the training rows, missing -> 0 and no missing flag: on the
    covered races the flag only marks where the export happened to start, which is not racing."""
    x = np.array(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    tr = np.asarray(train_mask, bool) & np.isfinite(x)
    if tr.sum() < 200:
        return np.zeros((len(x), 0))
    lo, hi = np.percentile(x[tr], [0.5, 99.5])
    xc = np.clip(x, lo, hi)
    mu, sd = float(xc[tr].mean()), float(xc[tr].std())
    z = np.where(np.isfinite(xc), (xc - mu) / sd, 0.0)
    if not quadratic:
        return z[:, None]
    z2 = z ** 2
    return np.column_stack([z, z2 - z2[tr].mean()])


def screen(sample: pd.DataFrame, vals: dict, names: list[str], splits, train_mask, base_extra=None, label="",
           present_only=False):
    scr = OutcomeScreen(splits)
    out = {"market_R2": scr.market_r2(), "per_feature": {}, "joint": {}}
    for nm in names:
        F = design_present(vals[nm], train_mask) if present_only else feature_design(vals[nm], train_mask)[0]
        if F.shape[1] == 0:
            continue
        g = scr.gains(F, base_extra=base_extra)
        out["per_feature"][nm] = gain_summary(g, scr.null)
    for set_name, set_cols in (("sectional", SECTIONAL), ("sectional+context", SECTIONAL + CONTEXT),
                               ("context", CONTEXT), ("ratings", RATINGS), ("all", SECTIONAL + CONTEXT + RATINGS)):
        if present_only:
            F = np.column_stack([design_present(vals[c], train_mask, quadratic=False) for c in set_cols if c in vals])
        else:
            F, labels = joint_design(vals, [c for c in set_cols if c in vals], train_mask)
        best = None
        for ridge in (1.0, 10.0, 100.0):
            g = scr.gains(F, ridge=ridge, base_extra=base_extra)
            s = gain_summary(g, scr.null)
            s["ridge"] = ridge
            if best is None or s["dll_mnats"] > best["dll_mnats"]:
                best = s
        out["joint"][set_name] = best      # the best of three penalties: generous to the features
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--history", required=True)
    ap.add_argument("--oos", default="data/oos_predictions.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    h = load_history(args.history)
    sample = build_sample(args.oos)
    feats = lag_features(sample, h)
    sample = sample.join(feats, on="row")
    fresh = (sample["date"] >= "2024-08-01") & (sample["date"] <= "2025-06-30")
    s = sample[fresh].reset_index(drop=True)
    vals = {c: s[c].to_numpy() for c in SECTIONAL + CONTEXT + RATINGS}
    has = s["sec_n"].notna()
    rep = {"rows": int(len(s)), "races": int(s["race"].nunique()), "share_with_sectional_history": float(has.mean()),
           "races_half_covered": int((has.groupby(s["race"]).mean() >= 0.5).sum())}

    month = s["date"].dt.year.to_numpy() * 12 + s["date"].dt.month.to_numpy()
    even, odd = month % 2 == 0, month % 2 == 1
    fwd_tr = (s["date"] < "2025-02-01").to_numpy()
    fwd_te = ~fwd_tr
    for base_name, base_extra in (("market", None), ("market+model", s["ln_model"].to_numpy(float))):
        parity = [parts_for(s, even, odd), parts_for(s, odd, even)]
        rep[f"parity|{base_name}"] = screen(s, vals, SECTIONAL + CONTEXT + RATINGS, parity, even | odd, base_extra)
        fwd = [parts_for(s, fwd_tr, fwd_te)]
        rep[f"forward|{base_name}"] = screen(s, vals, SECTIONAL + CONTEXT + RATINGS, fwd, fwd_tr, base_extra)

    # The clean reading: races where at least 80% of the field has a sectional run behind it, and no
    # missing flag (it marks where the export started, whose meaning changes with the month).
    cov = (has.groupby(s["race"]).transform("mean") >= 0.8).to_numpy()
    c = s[cov].reset_index(drop=True)
    cvals = {k: c[k].to_numpy() for k in vals}
    cm = c["date"].dt.year.to_numpy() * 12 + c["date"].dt.month.to_numpy()
    ce, co = cm % 2 == 0, cm % 2 == 1
    ctr = (c["date"] < "2025-02-01").to_numpy()
    rep["covered"] = {"rows": int(len(c)), "races": int(c["race"].nunique())}
    for base_name, base_extra in (("market", None), ("market+model", c["ln_model"].to_numpy(float))):
        rep[f"covered|parity|{base_name}"] = screen(c, cvals, SECTIONAL + CONTEXT + RATINGS,
                                                    [parts_for(c, ce, co), parts_for(c, co, ce)], ce | co, base_extra,
                                                    present_only=True)
        rep[f"covered|forward|{base_name}"] = screen(c, cvals, SECTIONAL + CONTEXT + RATINGS,
                                                     [parts_for(c, ctr, ~ctr)], ctr, base_extra, present_only=True)

    # The forecast check: within-race OLS of ln BSP on our forecast, and on our forecast plus the features.
    for set_name, set_cols in (("sectional", SECTIONAL), ("sectional+context", SECTIONAL + CONTEXT),
                               ("all", SECTIONAL + CONTEXT + RATINGS)):
        F, _ = joint_design(vals, set_cols, even | odd)
        res = {}
        for split_name, sp in (("parity", [parts_for(s, even, odd, "ln_pred_bsp"), parts_for(s, odd, even, "ln_pred_bsp")]),
                               ("forward", [parts_for(s, fwd_tr, fwd_te, "ln_pred_bsp")])):
            ms = MoveScreen(sp, s["ln_bsp"].to_numpy(float))
            g = ms.gains(F, ridge=10.0)
            base_sse = sum(float(b.sum()) for b, _ in ms.base)
            res[split_name] = {"sse_cut_pct": 100 * float(g.sum()) / base_sse,
                               "mse_base_within_race": base_sse / sum(len(te.y) for _, te in sp),
                               "t": float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))}
        rep[f"forecast|{set_name}"] = res

    txt = json.dumps(rep, indent=1, default=float)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
