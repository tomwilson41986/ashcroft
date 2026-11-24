"""Joseph Buchdahl's ideas (football-data.co.uk) tested on our data: margin removal, move retention, the new xG columns.

racing² football research, 24 Nov 2026 (the owner: "Test Buchdahl's ideas on the same rows"). Registered as
``football-buchdahl-1124`` in reports/research_ledger.jsonl before its run. Three readings, each descriptive (no rule
is selected and nothing is traded):

1. **Margin removal** (``margins``). Bookmakers' 1X2 closing prices made into chances five ways: proportional (the
   repository's), additive, power, Shin and odds ratio. Each is scored against the results by log loss, with the
   paired change against proportional (1,000 resamples of match days), and by its bias at the favourite and the
   outsider end (mean chance less the win rate, by price band). Books: Pinnacle's close (PSC), the market average's
   close (AvgC), Bet365's close (B365C), from football-data's files (silver ``football/matches.parquet``).
2. **Move retention** (``retention``). On Betfair's own price paths (the warehouse): how much of the move into a
   checkpoint survives to the close. For each match-odds runner at 360, 120, 60 and 15 minutes, the move to the close
   regressed on the move over the hour before (``path_ret_1h``) and the six hours before (``path_ret_6h``);
   retention = 1 + slope (1: the move holds; above 1: it continues; below 1: part of it reverts). By checkpoint and by
   activity before the checkpoint (price updates), with 95% intervals from resampled days.
3. **football-data's xG columns** (``fd_xg``). Since 2026/27 football-data publishes HxG and AxG for some divisions.
   Coverage by division and month, and their agreement with Sportmonks' xG on the same matches (the lab's
   ``smx_xg_xg_*``): correlation and mean difference.

    python -m research.open_close.football_buchdahl --src <lab input folder> [--cache <lab cache>] \
        --out-json out/buchdahl.json [--root <warehouse folder>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "football-buchdahl-1124"
BOOKS = {"pinnacle_close": ("PSCH", "PSCD", "PSCA"), "average_close": ("AvgCH", "AvgCD", "AvgCA"),
         "bet365_close": ("B365CH", "B365CD", "B365CA")}
METHODS = ("proportional", "additive", "power", "shin", "odds_ratio")
BANDS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1000.0]


# ---------------------------------------------------------------------------------------------------------------
# 1. margin removal
# ---------------------------------------------------------------------------------------------------------------

def devig(odds: np.ndarray, method: str) -> np.ndarray:
    """Chances summing to one from decimal odds (rows of three), by ``method``."""
    q = 1.0 / odds
    book = q.sum(1, keepdims=True)
    if method == "proportional":
        return q / book
    if method == "additive":
        p = q - (book - 1.0) / q.shape[1]
        p = np.clip(p, 1e-6, None)
        return p / p.sum(1, keepdims=True)
    if method == "power":                       # p_i = q_i ** k, sum to one: k found by bisection
        lo, hi = np.ones(len(q)), np.full(len(q), 5.0)
        for _ in range(60):
            k = (lo + hi) / 2
            s = (q ** k[:, None]).sum(1)
            lo, hi = np.where(s > 1, k, lo), np.where(s > 1, hi, k)
        p = q ** ((lo + hi) / 2)[:, None]
        return p / p.sum(1, keepdims=True)
    if method == "shin":                        # Shin (1993): z, the share of insider money, by bisection
        lo, hi = np.zeros(len(q)), np.full(len(q), 0.5)
        for _ in range(60):
            z = (lo + hi) / 2
            p = (np.sqrt(z[:, None] ** 2 + 4 * (1 - z[:, None]) * q ** 2 / book) - z[:, None]) / (2 * (1 - z[:, None]))
            s = p.sum(1)
            lo, hi = np.where(s > 1, z, lo), np.where(s > 1, hi, z)
        z = (lo + hi) / 2
        p = (np.sqrt(z[:, None] ** 2 + 4 * (1 - z[:, None]) * q ** 2 / book) - z[:, None]) / (2 * (1 - z[:, None]))
        return p / p.sum(1, keepdims=True)
    if method == "odds_ratio":                  # p_i / (1 - p_i) = (q_i / (1 - q_i)) / c, sum to one: c by bisection
        lo, hi = np.ones(len(q)), np.full(len(q), 50.0)
        qq = np.clip(q, 1e-9, 1 - 1e-9)
        for _ in range(60):
            c = (lo + hi) / 2
            o = qq / (1 - qq) / c[:, None]
            s = (o / (1 + o)).sum(1)
            lo, hi = np.where(s > 1, c, lo), np.where(s > 1, hi, c)
        o = qq / (1 - qq) / ((lo + hi) / 2)[:, None]
        p = o / (1 + o)
        return p / p.sum(1, keepdims=True)
    raise ValueError(method)


def match_days(silver: pd.DataFrame) -> pd.Series:
    """Each row's match day: the silver table's ``match_date``, else football-data's raw ``Date`` (day first)."""
    if "match_date" in silver.columns:
        return pd.to_datetime(silver["match_date"], errors="coerce")
    if "Date" in silver.columns:
        return pd.to_datetime(silver["Date"], dayfirst=True, errors="coerce")
    return pd.Series(pd.NaT, index=silver.index)


def margins(silver: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    res = {}
    res_col = silver["FTR"].astype(str).str.upper() if "FTR" in silver.columns else None
    if res_col is None:
        return {"status": "no data found: no FTR column"}
    outcome = res_col.map({"H": 0, "D": 1, "A": 2})
    day = match_days(silver).dt.strftime("%Y-%m-%d")
    rng = np.random.default_rng(seed)
    for name, cols in BOOKS.items():
        if not all(c in silver.columns for c in cols):
            res[name] = {"status": "no data found: columns missing"}
            continue
        o = silver[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ok = np.isfinite(o).all(1) & (o > 1.0).all(1) & outcome.notna().to_numpy()
        book = (1.0 / o[ok]).sum(1)
        ok_idx = np.flatnonzero(ok)
        keep = (book > 1.0) & (book < 1.25)
        o, y, dd = o[ok][keep], outcome.to_numpy()[ok][keep].astype(int), day.to_numpy()[ok_idx][keep]
        if len(o) < 1000:
            res[name] = {"status": f"no data found: {len(o)} matches"}
            continue
        r = {"matches": int(len(o)), "mean_overround": round(float((1 / o).sum(1).mean() - 1), 4), "methods": {}}
        ll = {}
        for mth in METHODS:
            p = devig(o, mth)
            ll[mth] = -np.log(np.clip(p[np.arange(len(p)), y], 1e-12, None))
            # bias by price band: mean chance less the win rate, every runner
            flat_o, flat_p = o.ravel(), p.ravel()
            won = (np.arange(3)[None, :] == y[:, None]).ravel()
            band = pd.cut(flat_o, BANDS)
            bias = pd.DataFrame({"b": band, "p": flat_p, "w": won}).groupby("b", observed=True).agg(
                n=("p", "size"), chance=("p", "mean"), rate=("w", "mean"))
            r["methods"][mth] = {"log_loss": round(float(ll[mth].mean()), 6),
                                 "bias_by_price": {str(k): {"n": int(v.n), "chance_less_rate": round(v.chance - v.rate, 4)}
                                                   for k, v in bias.iterrows()}}
        f = pd.DataFrame({"day": dd, **{m: ll[m] for m in METHODS}}).groupby("day").sum()
        idx = rng.integers(0, len(f), size=(n_boot, len(f)))
        n = len(o)
        for mth in METHODS:
            if mth == "proportional":
                continue
            diff = (f["proportional"].to_numpy() - f[mth].to_numpy())          # log loss saved a day
            boot = diff[idx].sum(1) / n
            r["methods"][mth]["log_loss_saved_vs_proportional"] = {
                "mean": round(float(diff.sum() / n), 7), "lo95": round(float(np.quantile(boot, 0.025)), 7),
                "hi95": round(float(np.quantile(boot, 0.975)), 7)}
        r["best_by_log_loss"] = min(METHODS, key=lambda m: r["methods"][m]["log_loss"])
        res[name] = r
    return res


# ---------------------------------------------------------------------------------------------------------------
# 2. move retention on Betfair's paths
# ---------------------------------------------------------------------------------------------------------------

def retention_fit(x: np.ndarray, y: np.ndarray, day: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Slope of y on x through the origin by least squares, retention 1 + slope, a 95% interval from resampled days."""
    f = pd.DataFrame({"day": day, "xy": x * y, "xx": x * x}).groupby("day").sum()
    xy, xx = f["xy"].to_numpy(), f["xx"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(f), size=(n_boot, len(f)))
    b = xy.sum() / xx.sum()
    boot = xy[idx].sum(1) / xx[idx].sum(1)
    return {"rows": int(len(x)), "days": int(len(f)), "retention": round(float(1 + b), 4),
            "lo95": round(float(1 + np.quantile(boot, 0.025)), 4), "hi95": round(float(1 + np.quantile(boot, 0.975)), 4)}


def retention(store) -> dict:
    from research.open_close import football_bf_close_model as bf
    d, fam = bf.load(store)
    if d.empty or "path" not in fam:
        return {"status": "no data found: gold/move or gold/path is not built"}
    d = d[d["cp_fresh"] & d["usable"] & d["schedule_fixed"]]
    out = {}
    bands = pd.cut(d["n_ticks_before"], [0, 20, 60, 150, 10**9], labels=["under 20", "20-59", "60-149", "150+"],
                   right=False)
    for w in ("path_ret_1h", "path_ret_6h"):
        if w not in d.columns:
            continue
        s = d[np.isfinite(d[w]) & (d[w] != 0)]
        r = {"all": retention_fit(s[w].to_numpy(float), s["y_logit"].to_numpy(float), s["day"].to_numpy()),
             "by_checkpoint": {}, "by_activity": {}, "by_checkpoint_and_activity": {}}
        for cp, g in s.groupby("cp_min"):
            r["by_checkpoint"][str(cp)] = retention_fit(g[w].to_numpy(float), g["y_logit"].to_numpy(float),
                                                        g["day"].to_numpy())
        for b, g in s.groupby(bands.loc[s.index], observed=True):
            if len(g) >= 500:
                r["by_activity"][str(b)] = retention_fit(g[w].to_numpy(float), g["y_logit"].to_numpy(float),
                                                         g["day"].to_numpy())
        for (cp, b), g in s.groupby([s["cp_min"], bands.loc[s.index]], observed=True):
            if len(g) >= 500:
                r["by_checkpoint_and_activity"][f"{cp} | {b}"] = retention_fit(
                    g[w].to_numpy(float), g["y_logit"].to_numpy(float), g["day"].to_numpy())
        out[w] = r
    return out


# ---------------------------------------------------------------------------------------------------------------
# 3. football-data's xG columns
# ---------------------------------------------------------------------------------------------------------------

def fd_xg(silver: pd.DataFrame, lab: pd.DataFrame | None, gold: pd.DataFrame | None = None) -> dict:
    cols = [c for c in ("HxG", "AxG") if c in silver.columns]
    if len(cols) < 2:
        return {"status": "no data found: football-data's files as loaded carry no HxG/AxG columns"}
    s = silver.copy()
    s["hxg"], s["axg"] = pd.to_numeric(s["HxG"], errors="coerce"), pd.to_numeric(s["AxG"], errors="coerce")
    s["month"] = match_days(s).dt.strftime("%Y-%m")
    have = s[s["hxg"].notna() & s["axg"].notna()]
    out = {"matches_with_xg": int(len(have)),
           "by_division": have.groupby("Div").size().sort_values(ascending=False).to_dict() if "Div" in have else {},
           "by_month": have.groupby("month").size().to_dict(),
           "first_month": have["month"].min() if len(have) else None}
    if lab is not None and gold is not None and len(have):
        from football.data import _with_ids
        ids = _with_ids(have, gold)                    # its own rows: our divisions, with the canonical match_id
        j = ids[["match_id", "hxg", "axg"]].merge(
            lab[["match_id", "smx_xg_xg_h", "smx_xg_xg_a"]], on="match_id", how="inner").dropna()
        if len(j) >= 30:
            a = np.r_[j["hxg"], j["axg"]]
            b = np.r_[j["smx_xg_xg_h"], j["smx_xg_xg_a"]]
            out["against_sportmonks"] = {"sides": int(len(a)), "corr": round(float(np.corrcoef(a, b)[0, 1]), 4),
                                         "mean_fd_less_sportmonks": round(float((a - b).mean()), 4)}
        else:
            out["against_sportmonks"] = {"status": f"no data found: {len(j)} matches linked to Sportmonks' xG"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="the lab's input folder (football/matches.parquet and the gold tables)")
    ap.add_argument("--cache", default=None, help="the lab cache (for Sportmonks' xG)")
    ap.add_argument("--root", default=None, help="a local warehouse folder in place of the capture bucket")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store
    from football.data import load_gold
    gs = Store(root=a.src)
    silver = gs.get_parquet("football/matches.parquet")
    if silver is None:
        raise SystemExit("no data found: football/matches.parquet")
    res = {"tag": TAG, "note": "descriptive readings; nothing selected or traded"}
    res["margins"] = margins(silver)
    log.info("margins done")
    lab = gold = None
    if a.cache:
        from football import lab as labmod
        lab = labmod.latest(a.cache)
    gold = load_gold(gs)
    res["fd_xg"] = fd_xg(silver, lab, gold)
    res["retention"] = retention(Store(root=a.root) if a.root else Store())
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: (v if k != "retention" else {w: r.get("all") for w, r in v.items() if isinstance(r, dict)})
                      for k, v in res.items()}, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
