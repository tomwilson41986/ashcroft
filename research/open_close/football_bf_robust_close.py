"""Is the closing-price model forecasting the price, or the noise of the last print? (development read)

Registered as ``football-robust-close-1112`` before its run. The closing-price model
(research/open_close/football_bf_close_model.py) targets the move from a checkpoint to the last traded price before
the off. A last print bounces between the back and the lay side, so part of that "move" is noise a model can learn
from stale prices and nobody can trade. The robust close here is the median of a runner's last prints (up to 5,
OPEN, in the 10 minutes before the off, at least 3): the target becomes y_logit + (median - last) in raw log odds.

Three scores, all out of sample (the model's own monthly walk-forward), on rows that have a robust close:
the model as it is against the last print; the same forecasts against the robust close; a model trained on the
robust close against the robust close. Each for every row and for busy rows (150 or more price updates).

    python -m research.open_close.football_bf_robust_close --out-json out/bf_robust_close.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np

from research.open_close import football_bf_close_model as cm

LAST_PRINTS, WINDOW_S, MIN_PRINTS, BUSY = 5, 600, 3, 150


def robust_gap(store):
    """(market_id, selection_id) -> median less last, in raw log odds, of the last prints before the off."""
    from football.warehouse import catalog
    con = catalog.connect(store)
    g = con.execute(f"""
        WITH t AS (SELECT market_id, selection_id, ln((1 / ltp) / (1 - 1 / ltp)) AS x,
                          row_number() OVER (PARTITION BY market_id, selection_id ORDER BY pt DESC, seq DESC) AS k
                   FROM tick WHERE market_type = 'MATCH_ODDS' AND pre_off AND status = 'OPEN' AND ltp > 1.01
                     AND t_off_s BETWEEN 0 AND {WINDOW_S})
        SELECT market_id, selection_id, median(x) - max(CASE WHEN k = 1 THEN x END) AS gap, count(*) AS n_prints
        FROM t WHERE k <= {LAST_PRINTS} GROUP BY 1, 2 HAVING count(*) >= {MIN_PRINTS}""").df()
    con.close()
    return g


def score(s, target: str, pred: str) -> dict:
    return cm.skill_ci(s["day"].to_numpy(), s[target].to_numpy(float), s[pred].to_numpy(float))


def run(store) -> dict:
    d, fam = cm.load(store)
    if d.empty:
        return {"status": "no data found: gold/move is not built"}
    d = d[d["cp_fresh"]].reset_index(drop=True)
    cols = [c for v in fam.values() for c in v]
    d = d.merge(robust_gap(store), on=["market_id", "selection_id"], how="left")
    d["y_last"] = d["y_logit"]
    d["y_robust"] = d["y_logit"] + d["gap"]
    d["pred_last"], _ = cm.walk_forward(d, cols)
    r = d.copy()
    r["usable"] = r["usable"] & r["y_robust"].notna()
    r["y_logit"] = r["y_robust"]
    d["pred_robust"], _ = cm.walk_forward(r, cols)
    s = d[d["usable"] & d["schedule_fixed"] & d["y_robust"].notna() & np.isfinite(d["pred_last"])
          & np.isfinite(d["pred_robust"])]
    res = {"rows_scored": int(len(s)), "rows_without_a_robust_close": int((d["usable"] & d["gap"].isna()).sum()),
           "gap_abs_median": float(s["gap"].abs().median()), "gap_sd": float(s["gap"].std()),
           "note": "development evidence: one year, last traded prices; nothing is traded"}
    for name, part in (("every_row", s), ("busy_rows_150_updates", s[s["n_ticks_before"] >= BUSY])):
        res[name] = {"rows": int(len(part)),
                     "model_as_it_is_against_the_last_print": score(part, "y_last", "pred_last"),
                     "the_same_forecasts_against_the_robust_close": score(part, "y_robust", "pred_last"),
                     "trained_on_the_robust_close_against_it": score(part, "y_robust", "pred_robust")}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="a local warehouse folder in place of the capture bucket")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from sources.common import Store
    res = run(Store(root=a.root) if a.root else Store())
    with open(a.out_json, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
