"""More features from the Betfair price files for the Betfair-native closing-price model.

racing² football research, 24 Nov 2026 (the owner: "Betfair Price Files - engineer more features out of this").
Registered as ``football-bf-features2-1124`` in reports/research_ledger.jsonl before its run. The model
(``football_bf_close_model``) reads the match-odds runner's own price path, the implied goals at the checkpoint, the
market's ratings and move memory. Built in the warehouse but not read by it, each known at the checkpoint:

- ``ou_path`` (``oup_``): the over 2.5 runner's own price path at the same checkpoint (gold/path's over/under rows):
  its returns over 15 minutes to 6 hours, volatility, activity, trend, tick imbalance, last tick and catch-up;
- ``imp_delta`` (``idl_``): how the whole event moved into the checkpoint, from the grid point before it
  (720 -> 360, 180 -> 120, 90 -> 60, 30 -> 15 minutes): the change of each 1X2 chance, of the over 2.5 chance, of
  both books' overround, of the handicap-against-1X2 gap, of the draw premium, of the implied total and supremacy;
  and the row's own outcome's change (``idl_own_dp``) beside the other two;
- ``pace`` (``pace_``): the runner's trading pace now against the last hour (ticks in 15 minutes over the hour's), the
  15-minute return against the hour's average quarter (acceleration), the 1-hour return against the 6-hour quarter.

**Arms**, on the same rows and walk-forward as the model (by calendar month, a model a role, ridge and LightGBM
averaged, fitted on reliable rows a day or more before the month): ``full`` (every family the model reads) and
``full`` plus each new family, and plus all three. The change in skill against "no move" paired by match day,
1,000 resamples, overall and by checkpoint.

    python -m research.open_close.football_bf_features2 --out-json out/bf_features2.json [--root <warehouse folder>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.open_close import football_bf_close_model as bf

log = logging.getLogger(__name__)

TAG = "football-bf-features2-1124"
PREV = {360: 720, 120: 180, 60: 90, 15: 30}
OU_COLS = ["ret_15m", "ret_30m", "ret_1h", "ret_2h", "ret_6h", "rv_1h", "rv_6h", "n_ticks_15m", "n_ticks_1h",
           "slope_6h", "tick_imb_1h", "last_dx", "book_catchup", "x"]
IMP_COLS = ["p_home", "p_draw", "p_away", "p_over", "mo_book_sum", "ou_book_sum", "sup_gap", "rho", "total",
            "supremacy"]


def ou_path(store) -> pd.DataFrame | None:
    """event_id, cp_min and the over 2.5 runner's path columns (``oup_``)."""
    t = bf.read_rows(store, "gold/path", None)
    if t is None:
        return None
    p = t.to_pandas()
    p = p[(p["market_type"] == "OVER_UNDER_25") & (p["role"] == "over")]
    cols = [c for c in OU_COLS if c in p.columns]
    out = p[["event_id", "cp_min"] + cols].drop_duplicates(["event_id", "cp_min"]).copy()
    out[cols] = out[cols].astype("float32")
    return out.rename(columns={c: f"oup_{c}" for c in cols})


def implied_deltas(imp: pd.DataFrame) -> pd.DataFrame:
    """event_id, cp_min and each implied column's change from the grid point before the checkpoint (``idl_d_``)."""
    cols = [c for c in IMP_COLS if c in imp.columns]
    parts = []
    for cp, prev in PREV.items():
        a = imp[imp["cp_min"] == cp][["event_id"] + cols]
        b = imp[imp["cp_min"] == prev][["event_id"] + cols]
        j = a.merge(b, on="event_id", how="left", suffixes=("", "_prev"))
        out = pd.DataFrame({"event_id": j["event_id"], "cp_min": cp})
        for c in cols:
            out[f"idl_d_{c}"] = (j[c] - j[f"{c}_prev"]).astype("float32")
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def own_change(d: pd.DataFrame) -> np.ndarray:
    """The row's own outcome's chance change (home, draw or away) from ``idl_d_p_*``."""
    out = np.full(len(d), np.nan, dtype="float32")
    for role, c in (("home", "idl_d_p_home"), ("draw", "idl_d_p_draw"), ("away", "idl_d_p_away")):
        r = (d["role"] == role).to_numpy()
        out[r] = d.loc[r, c].to_numpy("float32")
    return out


def pace(d: pd.DataFrame) -> pd.DataFrame:
    """Trading pace and acceleration from the runner's own path columns (``pace_``)."""
    g = lambda c: d[c].astype(float) if c in d.columns else pd.Series(np.nan, index=d.index)   # noqa: E731
    return pd.DataFrame({
        "pace_ticks_15_over_1h": (g("path_n_ticks_15m") / (g("path_n_ticks_1h") + 1.0)).astype("float32"),
        "pace_acc_15m": (g("path_ret_15m") - g("path_ret_1h") / 4.0).astype("float32"),
        "pace_acc_1h": (g("path_ret_1h") - g("path_ret_6h") / 6.0).astype("float32")}, index=d.index)


def add_families(store, d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    new: dict[str, list[str]] = {}
    ou = ou_path(store)
    if ou is not None:
        d = d.merge(ou, on=["event_id", "cp_min"], how="left")
        new["ou_path"] = [c for c in ou.columns if c.startswith("oup_")]
    imp = bf.read_rows(store, "gold/implied", ["event_id", "cp_min"] + IMP_COLS, horizons=False)
    if imp is not None:
        dl = implied_deltas(imp.to_pandas())
        d = d.merge(dl, on=["event_id", "cp_min"], how="left")
        d["idl_own_dp"] = own_change(d)
        new["imp_delta"] = [c for c in dl.columns if c.startswith("idl_")] + ["idl_own_dp"]
    pc = pace(d)
    d = pd.concat([d, pc], axis=1)
    new["pace"] = list(pc.columns)
    return d, new


def run(store) -> dict:
    d, fam = bf.load(store)
    if d.empty:
        return {"tag": TAG, "status": "no data found: gold/move is not built"}
    d = d[d["cp_fresh"]].reset_index(drop=True)
    d, new = add_families(store, d)
    full = [c for cols in fam.values() for c in cols]
    res = {"tag": TAG, "rows": int(len(d)), "rows_usable": int(d["usable"].sum()),
           "families_in_full": {k: len(v) for k, v in fam.items()}, "new_families": {k: v for k, v in new.items()},
           "coverage_usable_rows": {k: round(float(d.loc[d["usable"], v].notna().mean().mean()), 4)
                                    for k, v in new.items()},
           "note": "development evidence: last traded prices; nothing is traded", "arms": {}}
    if d["month"].nunique() < 4:
        res["status"] = "no data found: fewer than four months"
        return res
    p0, _ = bf.walk_forward(d, full)
    arms = {**{k: full + v for k, v in new.items()}, "all_new": full + [c for v in new.values() for c in v]}
    preds = {}
    for name, cols in arms.items():
        preds[name], _ = bf.walk_forward(d, cols, with_gain=False)
        log.info("arm %s fitted", name)
    ok = np.isfinite(p0) & d["schedule_fixed"].to_numpy() & d["usable"].to_numpy()
    for p in preds.values():
        ok &= np.isfinite(p)
    sc = d[ok]
    y, day = sc["y_logit"].to_numpy(float), sc["day"].to_numpy()
    base = p0[ok]
    res["scored_rows"], res["test_months"] = int(len(sc)), sorted(sc["month"].unique())
    res["full"] = bf.skill_ci(day, y, base)
    for name, p in preds.items():
        q = p[ok]
        r = {**bf.skill_ci(day, y, q, base=base), "by_horizon": {}}
        for cp in bf.HORIZONS:
            m = (sc["cp_min"] == cp).to_numpy()
            if m.sum() >= 500:
                r["by_horizon"][str(cp)] = bf.skill_ci(day[m], y[m], q[m], base=base[m])
        r["keeps_its_place"] = bool(name != "all_new" and r["vs_base_lo95"] > 0)
        res["arms"][name] = r
    every = d[np.isfinite(p0) & d["schedule_fixed"].to_numpy()]
    res["trades_at_last_traded_prices_0.05"] = {
        "full": bf.trades(every.assign(pred=p0[every.index]), 0.05),
        "all_new": bf.trades(every.assign(pred=preds["all_new"][every.index]), 0.05)}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="a local warehouse folder in place of the capture bucket")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store
    res = run(Store(root=a.root) if a.root else Store())
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k in ("tag", "status", "scored_rows", "full")}, indent=1,
                     default=float))
    for k, v in res.get("arms", {}).items():
        print(k, {x: v.get(x) for x in ("vs_base", "vs_base_lo95", "vs_base_hi95", "keeps_its_place")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
