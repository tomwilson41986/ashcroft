"""Gate A for the Australian model: Betfair Historic Data's early prices against the BSP (au/README.md).

The owner's bundle from historicdata.betfair.com (BASIC, Horse Racing, every country and market type, from Jan 2024;
data.tar shared 4 Nov 2026) is streamed once: every WIN market, all countries, to one table a year with each runner's
first traded price, the last traded price at 24 h, 60, 15 and 1 minute before the scheduled off, the close (last
before in play), the BSP and the result (betfair_historic.parse_market, imported unchanged). The tar itself is kept in
S3 as one object so it can be read again without the Drive link.

``--gate`` then asks the two questions gate A needs:

  1. the market alone: for Australia, and for GB/IE beside it, how far a price at each mark sits from the BSP
     (median log(BSP / price), the share that shortened, mean CLV = price / BSP - 1: a back at the mark closed by a
     lay at the SP), by the mark's price band and year;
  2. the model: does the Betfair-only model's out-of-sample price (au/model.py) pick the runners whose price at a
     mark beats the BSP by more than runners at the same price level do? Excess CLV = CLV - the mean CLV of every
     runner in the same mark-price band, with a 90% race-bootstrap interval; the UK rule's form (back where the price
     at the mark is over the model's expected BSP, by 0/5/10%).

BASIC holds last traded prices only: no book, no volume. A last traded price is evidence of a price, not proof that a
GBP15-200 back would have filled there (gate B, the recorded books, answers that).

    python -m au.historic --tar data.tar --name horse_racing_basic_2024 --keep-bundle
    python -m au.historic --gate
"""

from __future__ import annotations

import argparse
import bz2
import json
import logging
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
TABLE = "betfair_historic/horse_racing_win_{year}.parquet"
REPORT = "au/gate_a_report.json"
MARKS = ["ltp_first", "ltp_t_1440", "ltp_t_60", "ltp_t_15", "ltp_t_1", "ltp_close"]
KEEP = ["market_id", "event_id", "country", "venue", "market_time", "market_name", "selection_id", "runner_name",
        "runner_status", "won", "bsp"] + MARKS
BANDS = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 51.0, 1001.0]
COMMISSION = 0.02
WIN_RE = __import__("re").compile(rb'"marketType"\s*:\s*"WIN"')


def _win_rows(item: tuple[str, bytes]) -> tuple[str, list[dict] | None]:
    """One stream file -> its WIN markets' runner rows (others skipped before the full parse)."""
    from betfair_historic import parse_market
    name, raw = item
    try:
        text = bz2.decompress(raw) if raw[:3] == b"BZh" else raw
        if not WIN_RE.search(text):
            return name, []
        rows = [r for r in parse_market(text) if r.get("market_type") == "WIN"]
        return name, [{k: r.get(k) for k in KEEP} for r in rows]
    except Exception:
        return name, None


def stream_tar(path: str, chunk: int = 4000) -> tuple[dict[int, pd.DataFrame], dict]:
    rep = {"files": 0, "unreadable": 0, "win_rows": 0, "by_year": {}}
    by_year: dict[int, list[pd.DataFrame]] = {}

    def members():
        with tarfile.open(path, mode="r|*") as t:
            for m in t:
                if m.isfile() and m.name.endswith(".bz2") and "PaxHeaders" not in m.name:
                    yield m.name, t.extractfile(m).read()

    batch = []
    with ProcessPoolExecutor() as pool:
        def flush():
            for _, rows in pool.map(_win_rows, batch, chunksize=64):
                if rows is None:
                    rep["unreadable"] += 1
                    continue
                if not rows:
                    continue
                df = pd.DataFrame(rows)
                y = pd.to_datetime(df["market_time"], errors="coerce", utc=True).dt.year
                for yy, g in df.groupby(y):
                    by_year.setdefault(int(yy), []).append(g)
                rep["win_rows"] += len(df)
        for item in members():
            batch.append(item)
            rep["files"] += 1
            if len(batch) >= chunk:
                flush()
                batch = []
                if rep["files"] % 100000 < chunk:
                    log.info("%d files read, %d WIN runner rows", rep["files"], rep["win_rows"])
        if batch:
            flush()
    out = {}
    for y, frames in sorted(by_year.items()):
        df = pd.concat(frames, ignore_index=True)
        for c in ["won", "bsp"] + MARKS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.drop_duplicates(["market_id", "selection_id"], keep="last")
        out[y] = df
        rep["by_year"][y] = {"markets": int(df["market_id"].nunique()), "runners": int(len(df)),
                             "by_country": df.groupby("country")["market_id"].nunique().sort_values(
                                 ascending=False).head(10).to_dict(),
                             "first": str(df["market_time"].min()), "last": str(df["market_time"].max())}
    return out, rep


# ------------------------------------------------------------------------------------------------------------ gate

def _boot(df: pd.DataFrame, col: str, n: int = 1000, seed: int = 11) -> list[float]:
    per = df.groupby("market_id")[col].agg(["sum", "size"])
    if len(per) < 30:
        return [float("nan"), float("nan")]
    idx = np.random.default_rng(seed).integers(0, len(per), size=(n, len(per)))
    m = per["sum"].to_numpy()[idx].sum(1) / per["size"].to_numpy()[idx].sum(1)
    return [round(100 * float(np.percentile(m, 5)), 2), round(100 * float(np.percentile(m, 95)), 2)]


def runners_ran(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["runner_status"].isin(["WINNER", "LOSER", "PLACED"]) & (df["bsp"] > 1)].copy()
    d["year"] = pd.to_datetime(d["market_time"], utc=True).dt.year
    return d


def market_only(d: pd.DataFrame) -> dict:
    """Each mark's price against the BSP, by the mark's price band (and pooled by year)."""
    out = {}
    for mk in MARKS:
        x = d[d[mk] > 1].copy()
        if x.empty:
            continue
        x["band"] = pd.cut(x[mk], BANDS)
        x["lr"] = np.log(x["bsp"] / x[mk])
        x["clv"] = x[mk] / x["bsp"] - 1
        x["short"] = (x["bsp"] < x[mk]).astype(float)
        t = x.groupby("band", observed=True).agg(n=("clv", "size"), median_log_bsp_over_price=("lr", "median"),
                                                 mean_clv_pct=("clv", "mean"), shortened=("short", "mean"))
        t["mean_clv_pct"] *= 100
        out[mk] = {"runners": int(len(x)), "coverage": round(len(x) / len(d), 4),
                   "mean_clv_pct_all": round(100 * float(x["clv"].mean()), 2),
                   "by_band": {str(k): {c: (round(float(v), 4) if c != "n" else int(v)) for c, v in r.items()}
                               for k, r in t.iterrows()},
                   "by_year_mean_clv_pct": {int(k): round(100 * float(v), 2)
                                            for k, v in x.groupby("year")["clv"].mean().items()}}
    return out


def model_clv(d: pd.DataFrame, oos: pd.DataFrame) -> dict:
    """The model's picks at each mark against the runners at the same price level."""
    o = oos[["market_id", "selection_id", "p_model", "p_norm"]].copy()
    o["market_id"] = o["market_id"].astype("int64")
    o["selection_id"] = o["selection_id"].astype("int64")
    h = d.copy()
    h["market_id"] = h["market_id"].astype(str).str.replace(r"^1\.", "", regex=True).astype("int64")
    h["selection_id"] = h["selection_id"].astype("int64")
    j = h.merge(o, on=["market_id", "selection_id"], how="inner")
    out = {"joined_runners": int(len(j)), "historic_runners": int(len(h)),
           "join_rate": round(len(j) / max(len(h), 1), 4), "years": sorted(int(y) for y in j["year"].unique())}
    if j.empty:
        return out
    j["pred_bsp"] = 1.0 / j["p_model"]
    for mk in MARKS:
        x = j[j[mk] > 1].copy()
        if len(x) < 1000:
            continue
        x["band"] = pd.cut(x[mk], BANDS)
        x["clv"] = x[mk] / x["bsp"] - 1
        x["excess"] = x["clv"] - x.groupby("band", observed=True)["clv"].transform("mean")
        x["ret_sp_lay"] = x["clv"]                                  # back at the mark, laid at the SP: price/BSP - 1
        x["ret_result"] = np.where(x["won"] == 1, (x[mk] - 1) * (1 - COMMISSION), -1.0)
        res = {"runners": int(len(x)), "all_mean_clv_pct": round(100 * float(x["clv"].mean()), 2)}
        for t in (0.0, 0.05, 0.10):
            s = x[x[mk] > x["pred_bsp"] * (1 + t)]
            if len(s) < 200:
                continue
            res[f"price_over_model_by_{int(t * 100)}pct"] = {
                "bets": int(len(s)), "share": round(len(s) / len(x), 4),
                "mean_clv_pct": round(100 * float(s["clv"].mean()), 2), "ci90": _boot(s, "clv"),
                "excess_clv_pct": round(100 * float(s["excess"].mean()), 2), "excess_ci90": _boot(s, "excess"),
                "roi_to_result_pct": round(100 * float(s["ret_result"].mean()), 2),
                "by_year_excess_pct": {int(k): round(100 * float(v), 2)
                                       for k, v in s.groupby("year")["excess"].mean().items()}}
        # the model's ordering within each band: top quintile of model edge against the bottom
        x["medge"] = x["p_model"] * x[mk]
        x["q"] = x.groupby("band", observed=True)["medge"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if len(s) >= 5 else np.nan)
        qs = x.groupby("q")["excess"].mean()
        res["excess_clv_pct_by_model_quintile"] = {int(k): round(100 * float(v), 2) for k, v in qs.items()}
        out[mk] = res
    return out


def gate(store) -> dict:
    tables = sorted(k for k in store.listing("betfair_historic/") if k.startswith("betfair_historic/horse_racing_win_"))
    if not tables:
        raise SystemExit("no horse_racing_win tables: run --tar first")
    df = pd.concat([store.get_parquet(k) for k in tables], ignore_index=True)
    d = runners_ran(df)
    rep = {"tables": tables, "runners_ran": int(len(d)),
           "markets_by_country": d.groupby("country")["market_id"].nunique().sort_values(ascending=False).to_dict()}
    rep["market_only_AU"] = market_only(d[d["country"] == "AU"])
    rep["market_only_GB_IE"] = market_only(d[d["country"].isin(["GB", "IE"])])
    oos = store.get_parquet("au/model_report_oos.parquet")
    rep["model_AU"] = model_clv(d[d["country"] == "AU"], oos) if oos is not None else "no out-of-sample prices"
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tar", help="the Historic Data bundle (data.tar) to stream")
    ap.add_argument("--name", default="horse_racing_basic", help="the bundle's name in the store")
    ap.add_argument("--keep-bundle", action="store_true", help="also keep the tar in S3 as one object")
    ap.add_argument("--gate", action="store_true", help="run gate A on the stored tables")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store, PREFIX
    store = Store()
    rep = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if a.tar:
        if a.keep_bundle:
            key = f"{PREFIX}/betfair_historic/bundles/{a.name}.tar"
            store.s3.upload_file(a.tar, store.bucket, key)
            rep["bundle"] = f"s3://{store.bucket}/{key}"
        tables, rep["stream"] = stream_tar(a.tar)
        for y, df in tables.items():
            store.put_parquet(TABLE.format(year=y), df)
        store.put(f"betfair_historic/imports/{a.name}.json", json.dumps(rep, default=str).encode())
    if a.gate:
        rep["gate"] = gate(store)
        store.put(REPORT, json.dumps(rep["gate"], indent=1, default=str).encode())
    if len(rep) == 1:
        ap.error("pass --tar and/or --gate")
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
