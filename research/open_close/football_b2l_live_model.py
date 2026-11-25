"""The back-to-lay strategy re-scored with a move model that reads only what the live feed can see.

racing² football research, 25 Nov 2026 (the owner: back-to-lay live, GBP 2 a bet, GBP 50 a day). Registered as
``football-b2l-live-model-1125`` in reports/research_ledger.jsonl before its run. The scan's four passing cells
(``football-strategy-100-1124``) used a model fed the whole tick history. Live, the football feed polls a market every
30, 10, 5, 2 and 1 minutes towards the off, from 24 hours out; it does not record Asian handicaps; and the warehouse
holds a match's price files only some days after it. So the live model:

- **price path**: the history thinned to the feed's polling times (``football.live_features.thin_ticks``), the path
  features rebuilt from those "ticks" by ``gold_path.build``, keeping only the columns a poll series supports (the
  price at given times, its time-weighted path, its 24-hour range): ``POLL_PATH``. The columns that need every tick
  (realised volatility, tick counts, efficiency, jumps, last change, runs, imbalance, autocorrelation, Roll's spread,
  the stale-runner catch-up) are left out;
- **level**: the checkpoint's chance, the book's sum, the checkpoint, the kick-off hour and weekday (not the ages
  and tick counts, which come from the full tick history);
- **implied**: total, supremacy, draw premium and the over chance and its age; not the handicap's columns;
- **implied changes** (``imp_delta``, kept on 24 Nov) without the handicap gap's change;
- **memory**: each side's and country's earlier moves, counting only matches that kicked off 7 days or more before
  the checkpoint (``MEMORY_LAG_MIN``);
- no market ratings (they read every recent close).

The same rows, walk-forward and costs as the scan: match-odds runners at 360, 120, 60 and 15 minutes, fresh
checkpoint book, kick-off fixed; a model a role a month, ridge and LightGBM averaged; back-to-lay at last traded
prices a tick worse each side, 2% commission; odd months find, even months confirm; the deflated Sharpe with every
football rule counted. The full model's numbers on the same rows are reported beside it.

    python -m research.open_close.football_b2l_live_model --out-json out/b2l_live_model.json
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

TAG = "football-b2l-live-model-1125"
POLL_PATH = ["x", "ret_15m", "ret_30m", "ret_1h", "ret_2h", "ret_6h", "ret_24h", "ret_first", "hours_since_first",
             "slope_6h", "slope_24h", "slope_r2_6h", "slope_r2_24h", "range_pos_24h", "drawdown_24h", "runup_24h",
             "twap_dev_1h", "twap_dev_6h"]
LEVEL = ["x_cp", "cp_book_sum", "cp_min", "ko_hour", "ko_dow"]
IMPLIED = ["imp_total", "imp_supremacy", "imp_rho", "imp_p_over", "imp_ou_age_s"]
MEMORY = ["mm_own", "mm_own_n", "mm_opp", "mm_opp_n", "mm_country", "mm_country_n"]
MEMORY_LAG_DAYS = 7
CELLS = {"360 over 0.05": (360, 0.05), "360 over 0.03": (360, 0.03), "15 over 0.05": (15, 0.05),
         "15 over 0.08": (15, 0.08)}


def poll_path(store) -> pd.DataFrame:
    """gold/path's columns rebuilt from the history thinned to the feed's polling times, every partition."""
    from football import live_features as lf
    from football.warehouse import gold_path as gp
    from football.warehouse.store_io import list_parts, read_part
    out = []
    for y, m in list_parts(store, "silver/tick"):
        tick = read_part(store, "silver/tick", y, m, gp.COLUMNS["silver/tick"])
        cp = read_part(store, "gold/checkpoint", y, m, gp.COLUMNS["gold/checkpoint"])
        if tick is None or cp is None:
            continue
        c = cp.to_pandas()
        c = c[c["market_type"].isin(gp.MARKET_TYPES)]
        cpt = pd.to_datetime(c["cp_time"], utc=True)
        ko = ((cpt - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)) + c["cp_min"].astype("int64") * 60_000
        kickoff = dict(zip(c["market_id"], ko))
        t = tick.to_pandas()
        t["pt"] = (pd.to_datetime(t["pt"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
        t = t[t["market_type"].isin(gp.MARKET_TYPES)]
        thin = lf.thin_ticks(t, kickoff)
        if thin.empty:
            continue
        import pyarrow as pa
        from football.warehouse.schema import TS
        tt = pa.Table.from_pandas(thin.assign(pt=thin["pt"].astype("int64")), preserve_index=False)
        tt = tt.set_column(tt.schema.get_field_index("pt"), "pt", tt.column("pt").cast(pa.timestamp("ms", "UTC"))
                           .cast(TS))                        # the same unit as gold/checkpoint's cp_time
        built = gp.build({"silver/tick": tt, "gold/checkpoint": cp})
        p = built[gp.SPEC.name].to_pandas()
        out.append(p[p["market_type"] == "MATCH_ODDS"])
        log.info("poll path %d-%02d: %d ticks in, %d thinned, %d rows", y, m, len(t), len(thin), len(out[-1]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def frame(store) -> tuple[pd.DataFrame, list[str], list[str]]:
    """The rows with both column sets: (frame, the full model's columns, the live model's columns)."""
    from research.open_close import football_bf_features2 as f2
    old = bf.MEMORY_LAG_MIN
    d, fam = bf.load(store)                                           # the full model's memory, as in the scan
    bf.MEMORY_LAG_MIN = MEMORY_LAG_DAYS * 1440
    try:
        d_lag, _ = bf.load(store)
    finally:
        bf.MEMORY_LAG_MIN = old
    d = d[d["cp_fresh"] & d["schedule_fixed"]].reset_index(drop=True)
    d_lag = d_lag[d_lag["cp_fresh"] & d_lag["schedule_fixed"]].reset_index(drop=True)
    d, new = f2.add_families(store, d)
    full_cols = [c for v in fam.values() for c in v] + new.get("imp_delta", [])
    lag = d_lag[bf.KEY + MEMORY].rename(columns={c: f"lv_{c}" for c in MEMORY})
    d = d.merge(lag, on=bf.KEY, how="left")
    pp = poll_path(store)
    keep = [c for c in POLL_PATH if c in pp.columns]
    d = d.merge(pp[bf.KEY + keep].rename(columns={c: f"pp_{c}" for c in keep}), on=bf.KEY, how="left")
    live_cols = (LEVEL + IMPLIED + [f"pp_{c}" for c in keep] + [f"lv_{c}" for c in MEMORY]
                 + [c for c in new.get("imp_delta", []) if "sup_gap" not in c])
    live_cols = [c for c in live_cols if c in d.columns]
    return d, full_cols, live_cols


def score_cells(d: pd.DataFrame, pred: np.ndarray, n_rules) -> dict:
    from research.open_close.football_strategy_100 import back_to_lay, score, tick_longer
    from research.open_close.football_strategy_scan import tick_worse
    out = {}
    p1, p2 = d["ltp_cp"].to_numpy(float), d["ltp_close"].to_numpy(float)
    ok_close = d["usable"].to_numpy(bool)
    for name, (cp, t) in CELLS.items():
        m = (d["cp_min"] == cp).to_numpy() & np.isfinite(pred) & (pred > t) & ok_close
        r = back_to_lay(p1[m], p2[m])
        rw = back_to_lay(tick_worse(p1[m]), tick_longer(p2[m]))
        ok = np.isfinite(r) & np.isfinite(rw)
        out[name] = score(name, d["day"].to_numpy()[m][ok], r[ok], rw[ok], n_rules)
    return out


def run(store) -> dict:
    from research.open_close.football_strategy_100 import rules_counted
    d, full_cols, live_cols = frame(store)
    res = {"tag": TAG, "rows": int(len(d)), "full_features": len(full_cols), "live_features": live_cols,
           "poll_path_coverage": round(float(d[[c for c in live_cols if c.startswith("pp_")]].notna().mean().mean()), 4),
           "note": "development evidence: last traded prices; nothing is traded"}
    p_full, _ = bf.walk_forward(d, full_cols)
    p_live, _ = bf.walk_forward(d, live_cols)
    sc = d[np.isfinite(p_full) & np.isfinite(p_live) & d["usable"].to_numpy()]
    y, day = sc["y_logit"].to_numpy(float), sc["day"].to_numpy()
    res["move_skill"] = {"full": bf.skill_ci(day, y, p_full[sc.index]),
                         "live": bf.skill_ci(day, y, p_live[sc.index], base=p_full[sc.index])}
    n_rules = rules_counted()
    res["cells_live_model"] = score_cells(d, p_live, n_rules)
    res["cells_full_model"] = score_cells(d, p_full, n_rules)
    res["passes_live_model"] = sorted(k for k, v in res["cells_live_model"].items() if v["pass"])
    res["leads_live_model"] = sorted(k for k, v in res["cells_live_model"].items() if v["lead"])
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
    print(json.dumps({k: res.get(k) for k in ("tag", "rows", "move_skill", "passes_live_model", "leads_live_model")},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
