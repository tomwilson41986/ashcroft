#!/usr/bin/env python3
"""The morning's forecasts against the evening's results, day by day.

    python scripts/recent_results.py --live-dir live --db horse_racing.db --out recent

The 06:00 job writes one CSV a day to s3://<bucket>/predictions/<date>.csv:
every runner's predicted BFSP and win probability, with the racecard and
Betfair prices as they stood when it ran. The 21:30 job puts the result and
the BSP into horse_racing.db. This joins the two with
research_lab._live_clv_frame -- the join the CLV report already uses, and
already debugged (a race time read as a number turns "2.30" into 2.3 and
silently matches nothing) -- and reports what the model said against what
happened.

Fetching the files is the workflow's job (.github/workflows/recent-results.yml),
because that is where the credentials live.

Two comparisons, because they answer different questions:

* model's top pick against the market favourite, at BSP after commission --
  whether the selection is any good;
* the model's forecast of BSP against the 06:00 Betfair price as a forecast of
  the same BSP -- whether the model knows anything about where the price will
  close that the morning market does not. That is the objective the model
  exists for: getting on early at a better price.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from compare_oos_runs import concordance  # noqa: E402

#: Betfair's standard rate on net winnings, as everywhere else in the repo.
COMMISSION = 0.05


def score(d: pd.DataFrame) -> pd.DataFrame:
    """Ranks, errors and returns on rows already joined to a result.

    Ranks are within the runners that actually ran: a non-runner is in the
    morning file but not the results, and drops out of the join. The price
    shown is the morning's as issued -- normalised over the declared field,
    so a race that lost runners has a book a little under 1. That is what was
    published, and it is what is reported.
    """
    d = d.copy()
    d["race_date"] = d["race_date"].astype(str).str.slice(0, 10)
    for c in ("predicted_bfsp", "bfsp", "bf_best_back", "racecard_odds"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d["predicted_bfsp"] > 1) & (d["bfsp"] > 1)].copy()
    # method="first" breaks ties, so every rank is a whole number: store it as one.
    d["model_rank"] = d.groupby("raceid")["predicted_bfsp"].rank(method="first").astype(int)
    d["market_rank"] = d.groupby("raceid")["bfsp"].rank(method="first").astype(int)
    d["log_err"] = np.abs(np.log(d["predicted_bfsp"] / d["bfsp"]))
    d["ret"] = np.where(d["won"], (d["bfsp"] - 1.0) * (1.0 - COMMISSION), -1.0)
    return d


def _block(g: pd.DataFrame, label: str) -> dict:
    top = g[g["model_rank"] == 1]
    fav = g[g["market_rank"] == 1]
    return {
        "date": label,
        "races": g["raceid"].nunique(),
        "runners": len(g),
        "top_pick": f"{int(top['won'].sum())}/{len(top)}",
        "top_pick_pnl": round(float(top["ret"].sum()), 2),
        "top_pick_roi_%": round(100 * float(top["ret"].mean()), 1) if len(top) else np.nan,
        "favourite": f"{int(fav['won'].sum())}/{len(fav)}",
        "fav_pnl": round(float(fav["ret"].sum()), 2),
        "fav_roi_%": round(100 * float(fav["ret"].mean()), 1) if len(fav) else np.nan,
        "median_abs_log_err": round(float(g["log_err"].median()), 3),
        "median_pred/actual": round(float((g["predicted_bfsp"] / g["bfsp"]).median()), 3),
        "concordance_model": round(concordance(g, "predicted_bfsp"), 3),
        "concordance_market": round(concordance(g, "bfsp"), 3),
    }


def day_summary(d: pd.DataFrame) -> pd.DataFrame:
    """One row a day, then one for the whole span.

    Profit is per 1-unit stake at BSP after commission, so it can be read as
    units won or lost. A day is a few dozen races: the rows are what happened,
    not an estimate of anything.
    """
    rows = [_block(g, day) for day, g in d.groupby("race_date")]
    rows.append(_block(d, "all"))
    return pd.DataFrame(rows)


def top_picks(d: pd.DataFrame) -> pd.DataFrame:
    """The model's first choice in every race, beside what the market made it."""
    t = d[d["model_rank"] == 1].copy()
    cols = ["race_date", "race_time", "track", "horse_name", "predicted_bfsp",
            "bfsp", "market_rank", "placing_numerical", "won", "ret"]
    for c in ("bf_best_back", "racecard_odds"):
        if c in t.columns:
            cols.insert(5, c)
    t = t[[c for c in cols if c in t.columns]]
    t = t.sort_values(["race_date", "race_time", "track"])
    return t.rename(columns={"bfsp": "actual_bsp", "market_rank": "mkt_rank",
                             "placing_numerical": "placed", "ret": "pnl"})


def forecast_vs_morning(d: pd.DataFrame, morning_col: str = "bf_best_back") -> dict | None:
    """Is the model a better forecaster of BSP than the 06:00 Betfair price?

    Scored only on runners that have both, so the two are measured on the same
    rows. A thin 06:00 book is a noisy price, which is exactly why the question
    is worth asking: if the model beats it, knowing the model's number early is
    worth something; if it does not, the morning price already says it.
    """
    if morning_col not in d.columns:
        return None
    m = d[pd.to_numeric(d[morning_col], errors="coerce") > 1].copy()
    if m.empty:
        return None
    m["morning_err"] = np.abs(np.log(m[morning_col] / m["bfsp"]))
    beats = m["log_err"] < m["morning_err"]
    top = m[m["model_rank"] == 1]
    return {
        "runners_with_morning_price": len(m),
        "coverage_%": round(100 * len(m) / len(d), 1),
        "model_median_abs_log_err": round(float(m["log_err"].median()), 3),
        "morning_median_abs_log_err": round(float(m["morning_err"].median()), 3),
        "model_closer_to_bsp_%": round(100 * float(beats.mean()), 1),
        "top_picks_with_morning_price": len(top),
        # > 1 means the morning price was longer than the BSP: backing early
        # would have got more than waiting for the off.
        "top_pick_median_morning/bsp": (round(float((top[morning_col] / top["bfsp"]).median()), 3)
                                        if len(top) else np.nan),
        "top_pick_morning_beat_bsp_%": (round(100 * float((top[morning_col] > top["bfsp"]).mean()), 1)
                                        if len(top) else np.nan),
    }


def _md(df: pd.DataFrame) -> str:
    """A markdown table without `tabulate`, which the CI image does not carry."""
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def unmatched_diagnosis(live_dir: str, db_path: str, day: str, n: int = 4) -> list[str]:
    """Why a morning file joined to nothing, in terms that settle it.

    "No results yet" and "the keys do not agree" print the same empty table,
    and the first time this report ran it said the former about five days that
    had results. So for any day that matched nothing: how many results the
    database holds for that date, and a few join keys from each side, built
    exactly as `_live_clv_frame` builds them.
    """
    import sqlite3

    from betfair_prices import normalise_horse, normalise_track, race_time_to_24h

    text = {c: str for c in ("date", "race_date", "venue", "track", "race_time",
                             "runner_name", "horse_name")}
    out = []
    conn = sqlite3.connect(db_path)
    # read_sql_query, unlike read_csv, refuses a dtype for a column it did not
    # select, so the mapping names exactly the four it does.
    res = pd.read_sql_query(
        "SELECT race_date, track, race_time, horse_name FROM race_results "
        "WHERE substr(race_date, 1, 10) = ?", conn, params=(day,),
        dtype={c: str for c in ("race_date", "track", "race_time", "horse_name")})
    conn.close()
    out.append(f"  database rows dated {day}: {len(res)}")
    paths = glob.glob(os.path.join(live_dir, "**", f"{day}.csv"), recursive=True)
    if not paths:
        return out
    pred = pd.read_csv(paths[0], dtype=text)
    for col, alt in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
        if col not in pred.columns and alt in pred.columns:
            pred[col] = pred[alt]

    def keys(d):
        return (d["race_date"].astype(str).str.slice(0, 10) + "|"
                + d["track"].map(normalise_track) + "|"
                + d["race_time"].map(race_time_to_24h).astype(str) + "|"
                + d["horse_name"].map(normalise_horse))

    out.append("  morning raw  : " + "; ".join(
        f"{r.race_date}|{r.track}|{r.race_time}|{r.horse_name}" for r in pred.head(n).itertuples()))
    out.append("  morning keys : " + "; ".join(keys(pred.head(n))))
    if len(res):
        out.append("  database raw : " + "; ".join(
            f"{r.race_date}|{r.track}|{r.race_time}|{r.horse_name}" for r in res.head(n).itertuples()))
        out.append("  database keys: " + "; ".join(keys(res.head(n))))
    return out


def prediction_counts(live_dir: str) -> dict[str, int]:
    """Rows per morning file, so the match rate can be stated rather than assumed."""
    out = {}
    for p in sorted(glob.glob(os.path.join(live_dir, "**", "*.csv"), recursive=True)):
        try:
            out[os.path.splitext(os.path.basename(p))[0]] = len(pd.read_csv(p, usecols=[0]))
        except Exception:                                   # noqa: BLE001
            continue
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-dir", required=True, help="directory of the morning CSVs")
    ap.add_argument("--db", default="horse_racing.db")
    ap.add_argument("--out", default="recent")
    a = ap.parse_args(argv)

    from research_lab import _live_clv_frame

    counts = prediction_counts(a.live_dir)
    joined = _live_clv_frame(a.live_dir, a.db)
    os.makedirs(a.out, exist_ok=True)
    L = ["# Morning predictions against results", ""]

    if joined.empty:
        L.append(f"No settled runners: {len(counts)} morning files "
                 f"({', '.join(counts) or 'none'}), none joined to a result.")
        L += ["", "```"]
        for day in counts:
            L.append(day)
            L += unmatched_diagnosis(a.live_dir, a.db, day)
        L.append("```")
        text = "\n".join(L)
        print(text)
        open(os.path.join(a.out, "summary.md"), "w").write(text + "\n")
        return 1

    d = score(joined)
    matched = d.groupby("race_date").size().to_dict()
    L.append("Per 1-unit stake at BSP, after 5% commission. Morning files and how "
             "many of their runners joined to a settled result:")
    L.append("")
    unmatched = []
    for day, n in counts.items():
        got = matched.get(day, 0)
        pct = f"{100 * got / n:.0f}%" if n else "-"
        L.append(f"- {day}: {n} predicted, {got} settled ({pct})"
                 + ("" if got else " — nothing joined; see the diagnosis below"))
        if not got:
            unmatched.append(day)
    if unmatched:
        L += ["", "## Days that joined nothing", "",
              "Either the database holds no results for the date, or it does and the "
              "keys disagree. These say which:", "", "```"]
        for day in unmatched:
            L.append(day)
            L += unmatched_diagnosis(a.live_dir, a.db, day)
        L.append("```")
    L += ["", "## By day", "", _md(day_summary(d)), ""]

    fm = forecast_vs_morning(d)
    if fm:
        # Built row by row: transposing a dict of ints and floats into one
        # column upcasts every count to a float, and "24.0 runners" reads wrong.
        L += ["## The model's BSP forecast against the 06:00 Betfair price", "",
              _md(pd.DataFrame({"measure": list(fm), "value": [str(v) for v in fm.values()]})), ""]
    else:
        L += ["No 06:00 Betfair prices in these files, so the forecast cannot be "
              "set against the morning market.", ""]

    tp = top_picks(d)
    L += ["## The model's top pick in every race", "", _md(tp.round(2)), ""]

    d.to_csv(os.path.join(a.out, "runners.csv"), index=False)
    tp.to_csv(os.path.join(a.out, "top_picks.csv"), index=False)
    text = "\n".join(L)
    open(os.path.join(a.out, "summary.md"), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
