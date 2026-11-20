"""Does a race see its own result through a twin row? (the owner, 20 Nov 2026: the discovery round, build 6; ledger
greyhound-twin-check-1120). Descriptive, 0 trials; read-only.

Leak check 3 for the residual on the BSP. The card check proves a day's features do not read that day's results; the
cut check proves they read no later day. Neither sees a race held twice in the data (two race ids for one race, a
track's name spelt two ways): the second copy's "last run" features would then be the race's own result, in the cache
and in any rebuild alike. A dog cannot run twice in a day, and hardly ever on consecutive days, so the tell is a dog
with two rows on one day (a twin) or a run a day or less after its last (a short gap).

Counted by year over the wholly priced races the residual reads. If any race is touched, the residual study is run
again with every touched race removed from the fit and the test, and read beside the original.

    python -m research.greyhound.twin_check --out out/study
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "greyhound-twin-check-1120"
SHORT_GAP_DAYS = 1.0


def flags(d: pd.DataFrame) -> pd.DataFrame:
    """Per row: ``twin`` (the dog has rows in more than one race that day), ``short`` (days since its last run at
    most ``SHORT_GAP_DAYS``), ``touched`` (its race holds a runner with either)."""
    twin = d.groupby(["dog_id", "day"]).race_id.transform("nunique").to_numpy() > 1
    gap = pd.to_numeric(d["days_since"], errors="coerce").to_numpy(float) if "days_since" in d else np.full(len(d), np.nan)
    short = np.isfinite(gap) & (gap <= SHORT_GAP_DAYS)
    bad = twin | short
    touched = pd.Series(bad).groupby(d.race_id.to_numpy()).transform("any").to_numpy()
    return pd.DataFrame({"twin": twin, "short": short, "touched": touched}, index=d.index)


def audit(d: pd.DataFrame, f: pd.DataFrame) -> dict:
    """The counts by year and in all: rows, races, twins, short gaps, races touched."""
    year = d.day.str[:4].to_numpy()
    rows = []
    for y in list(np.unique(year)) + ["all"]:
        m = np.ones(len(d), bool) if y == "all" else year == y
        races = d.race_id.to_numpy()[m]
        rows.append({"year": y, "rows": int(m.sum()), "races": int(len(np.unique(races))),
                     "twin_rows": int(f.twin.to_numpy()[m].sum()), "short_gap_rows": int(f.short.to_numpy()[m].sum()),
                     "races_touched": int(len(np.unique(races[f.touched.to_numpy()[m]])))})
    return {"by_year": rows, "races_touched": rows[-1]["races_touched"], "races": rows[-1]["races"]}


def markdown(res: dict) -> str:
    a = res["audit"]
    md = [f"## A race that sees its own result: twin rows and short gaps ({TAG})", "",
          f"{a['by_year'][-1]['rows']:,} runners of {a['races']:,} wholly priced races. A twin: the dog has rows in two "
          f"races on one day. A short gap: a run {SHORT_GAP_DAYS:g} day or less after its last.", "",
          "| Year | Runners | Races | Twin rows | Short-gap rows | Races touched |", "|---|---|---|---|---|---|"]
    md += [f"| {r['year']} | {r['rows']:,} | {r['races']:,} | {r['twin_rows']:,} | {r['short_gap_rows']:,} | "
           f"{r['races_touched']:,} |" for r in a["by_year"]]
    if not a["races_touched"]:
        md += ["", "No race is touched: no rerun is needed; the residual study stands on this check."]
    elif res.get("rerun_markdown"):
        md += ["", f"### The residual study with the {a['races_touched']:,} touched races removed", "", res["rerun_markdown"]]
    return "\n".join(md)


def main(argv=None) -> int:
    from greyhound import fastloop as fl
    from research.greyhound import sp_offset as so
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/twin_check")
    a = ap.parse_args(argv)
    df, meta = fl.load_cache(Store())
    feats = [f for f in fl.feature_set(meta, "card_safe") if f in df.columns and f not in fl.LABELS]
    df = df.assign(day=pd.to_datetime(df.t).dt.strftime("%Y-%m-%d"))
    # the flags are read on every row of the cache (a twin may be an unpriced copy), then cut to the priced races
    f_all = flags(df)
    keep = df.bsp_p_norm.notna().to_numpy()
    d, f = df[keep], f_all[keep]
    del df
    res = {"tag": TAG, "cache": {"hash": meta.get("hash"), "last_day": meta.get("last_day")}, "audit": audit(d, f)}
    log.info("audit: %s", res["audit"]["by_year"][-1])
    if res["audit"]["races_touched"]:
        from greyhound.costs import commission
        from research.greyhound.trial_count import greyhound_trials
        clean = d[~f.touched.to_numpy()]
        rerun = so.run(clean, feats, stage2_args=(commission(), greyhound_trials()))
        res["rerun"] = rerun
        res["rerun_markdown"] = so.markdown(rerun)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({k: v for k, v in res.items() if k != "rerun_markdown"}, indent=1, default=str))
    (out / "report.md").write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
