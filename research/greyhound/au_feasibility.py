"""Australian greyhounds on Betfair: can a model built on GreyhoundAPI's Australian results add to the Betfair SP?
(the owner, 16 Nov 2026: "we can bet on australian races on betfair too", then "Build the Australian model and start
the feasibility study"; ledger greyhound-au-feasibility-1116). Research only: nothing reaches Betfair.

**Data.** GreyhoundAPI's Australian results (``sources/greyhoundapi/runs_AU_<year>.parquet``: box, position, run
time, first sectional, weight, grade, track, trip, the last Betfair back and lay price before the off) for the form
and the market; Betfair's own price files (``sources/greyhound_prices/prices_<year>.parquet``, country AU: BSP) where
they hold the race, joined on name, track and UTC day. Amendment (16 Nov 2026, before any reading): the archived price
files hold 167 Australian runners, so the market price is the BSP where held, else the last Betfair back price before
the off as the API recorded it; the value rule backs at that price.

**Features** (a dog's Australian runs on days strictly before the race's local day; greyhound/pit/speed.py
``earlier``): runs, win and place rates (shrunk), the time figure (100 x (the track and trip's standard - ln time),
the standard the mean ln winning time there over the 365 days before) last, mean and best of three and weighted by
recency, the first-sectional figure the same, days since its last run, its last position and grade, runs at the track
and the trip, its box's record, its last weight; the race's field size, trip, grade, box; each figure against the
field and its rank (the early-speed rank above all, the Australian form student's first read); the market's view of
it in its past runs (its share at the last price before the off) and how it ran against it.

**Model.** LightGBM, walk-forward by year: for each test year Y the win model is fitted on the ``TRAIN_YEARS`` before
Y and the race's chances normalised to sum to 1. Scored on test-year races where every runner has a BSP:

1. log-loss of the model, of the BSP's chance (overround out) and of the two blended (a conditional logit on
   logit(model) and logit(BSP), fitted on the year before Y's out-of-sample rows), paired by day;
2. Benter's test: the model's coefficient beside the BSP's in a conditional logit on the test year (z);
3. a value rule at the BSP: back when the blended chance x (BSP - 1) x (1 - commission) - (1 - chance) exceeds an
   edge (0, 0.05, 0.10, 0.20), BSP at most ``MAX_BSP``; the return on stakes with a 90% interval by day.

The reading is registered in reports/research_ledger.jsonl before the first run. Writes out/au_feasibility/
{report.json, report.md, pred_<year>.parquet}.

    python -m research.greyhound.au_feasibility --first-year 2016 --test-years 2024,2025,2026
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound.pit import core
from greyhound.pit.speed import earlier

log = logging.getLogger(__name__)

COMMISSION = 0.02
TRAIN_YEARS = 5
STD_DAYS = 365
MIN_STD = 20
K = 6.0
MAX_BSP = 30.0
EDGES = (0.0, 0.05, 0.10, 0.20)
DRAWS = 2000
API_COLS = ["race_id", "dog_id", "dog_name", "track", "start_utc", "start_local", "trap", "position", "run_time_s",
            "adjusted_time_s", "sectional_s", "distance_m", "class_number", "grade", "field_size", "weight_kg",
            "bf_back", "bf_lay", "status", "result_status"]


def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


# ---- data ------------------------------------------------------------------------------------------------------------

def load_api(store, first_year: int) -> pd.DataFrame:
    parts = []
    for k in sorted(store.listing("greyhoundapi/")):
        m = re.search(r"greyhoundapi/runs_AU_(\d{4})\.parquet$", k)
        if m and int(m.group(1)) >= first_year:
            t = store.get_parquet(k)
            if t is not None and len(t):
                parts.append(t[[c for c in API_COLS if c in t.columns]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=API_COLS)


def load_bsp(store, first_year: int) -> pd.DataFrame:
    parts = []
    for k in sorted(store.listing("greyhound_prices/")):
        m = re.search(r"prices_(\d{4})\.parquet$", k)
        if m and int(m.group(1)) >= first_year:
            t = store.get_parquet(k)
            if t is not None and len(t):
                t = t[(t.market == "win") & (t.country == "AU")]
                parts.append(t[["race_date", "race_time", "track", "trap", "dog", "bsp", "win_lose", "event_id"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def join_bsp(api: pd.DataFrame, bf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The BSP on each API runner: name, track and UTC day (the price files' times are UK times)."""
    a = api.copy()
    a["utc_day"] = pd.to_datetime(a.start_utc, utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    a["dk"], a["tk"] = norm(a.dog_name), norm(a.track)
    if bf is None or not len(bf):
        a["bsp"], a["bf_won"] = np.nan, np.nan
        return a, {"bsp_files_runners": 0}
    b = bf.copy()
    uk = pd.to_datetime(b.race_date.astype(str) + " " + b.race_time.astype(str), errors="coerce")
    b["utc_day"] = uk.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC") \
        .dt.strftime("%Y-%m-%d")
    b["dk"], b["tk"] = norm(b.dog), norm(b.track)
    b = b.dropna(subset=["utc_day"]).drop_duplicates(["dk", "tk", "utc_day"], keep="last")
    m = a.merge(b[["dk", "tk", "utc_day", "bsp", "win_lose"]].rename(columns={"win_lose": "bf_won"}),
                on=["dk", "tk", "utc_day"], how="left")
    notes = {"api_runners": int(len(a)), "bsp_file_runners": int(len(b)), "matched": int(m.bsp.notna().sum()),
             "match_share_of_bsp_files": round(float(m.bsp.notna().sum() / max(len(b), 1)), 4)}
    return m, notes


# ---- features --------------------------------------------------------------------------------------------------------

def figures(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["race_date"] = d.start_local.astype(str).str[:10]
    d["pos"] = pd.to_numeric(d.position, errors="coerce")
    t = pd.to_numeric(d.adjusted_time_s, errors="coerce").fillna(pd.to_numeric(d.run_time_s, errors="coerce"))
    d["_ltime"] = np.log(t.where(t > 5))
    sec = pd.to_numeric(d.sectional_s, errors="coerce")
    d["_lsec"] = np.log(sec.where((sec > 1) & (sec < 15)))
    d["_key"] = d.track.astype(str) + "|" + pd.to_numeric(d.distance_m, errors="coerce").round().astype("Int64").astype(str)
    win = (d.pos == 1) & d._ltime.notna()
    d["_lw"], d["_w"] = np.where(win, d._ltime, 0.0), win.astype(float)
    d["_ls"], d["_sn"] = np.nan_to_num(d._lsec), d._lsec.notna().astype(float)
    w = core.prior_window(d[["race_date", "_key", "_lw", "_w", "_ls", "_sn"]], ["_key"], ["_lw", "_w", "_ls", "_sn"],
                          STD_DAYS)
    std = np.where(w._w >= MIN_STD, w._lw / w._w.replace(0, np.nan), np.nan)
    spar = np.where(w._sn >= 30, w._ls / w._sn.replace(0, np.nan), np.nan)
    d["fig"] = 100.0 * (std - d._ltime)
    d["sec"] = 100.0 * (spar - d._lsec)
    d["won"] = (d.pos == 1).astype(float).where(d.pos.notna())
    d["placed"] = (d.pos <= 2).astype(float).where(d.pos.notna())
    back = pd.to_numeric(d.bf_back, errors="coerce").where(lambda x: x > 1)
    inv = 1.0 / back
    d["mkt_share"] = inv / inv.groupby(d.race_id).transform("sum")
    d["mkt_resid"] = d.won - d.mkt_share
    return d


def _vs(race, x):
    s = pd.Series(x)
    tot, cnt = s.groupby(race).transform("sum").to_numpy(), s.notna().groupby(race).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(x) & (cnt > 1), x - (tot - np.nan_to_num(x)) / np.maximum(cnt - 1, 1), np.nan)


def features(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dog = pd.to_numeric(d.dog_id, errors="coerce").to_numpy(float)
    day = pd.to_datetime(d.race_date)
    dn = ((day - pd.Timestamp("2000-01-01")).dt.days).to_numpy(float)
    race = d.race_id.to_numpy()
    f = pd.DataFrame(index=d.index)
    f["n_runs"] = earlier(dog, dn, d.won.to_numpy(float), "count")
    wins = earlier(dog, dn, d.won.to_numpy(float), "mean", n=10_000)
    f["win_rate"] = (wins * f.n_runs + K / 8.0) / (f.n_runs + K)
    pl = earlier(dog, dn, d.placed.to_numpy(float), "mean", n=10_000)
    f["place_rate"] = (pl * f.n_runs + 2 * K / 8.0) / (f.n_runs + K)
    for c in ("fig", "sec"):
        v = d[c].to_numpy(float)
        f[f"{c}_last"] = earlier(dog, dn, v, "last")
        f[f"{c}_mean3"] = earlier(dog, dn, v, "mean", n=3)
        f[f"{c}_best3"] = earlier(dog, dn, v, "max", n=3)
        f[f"{c}_ewm"] = earlier(dog, dn, v, "ewm")
    f["pos_last"] = earlier(dog, dn, d.pos.to_numpy(float), "last")
    f["pos_mean3"] = earlier(dog, dn, d.pos.to_numpy(float), "mean", n=3)
    f["cls_last"] = earlier(dog, dn, pd.to_numeric(d.class_number, errors="coerce").to_numpy(float), "last")
    f["weight_last"] = earlier(dog, dn, pd.to_numeric(d.weight_kg, errors="coerce").to_numpy(float), "last")
    f["mkt_share_mean3"] = earlier(dog, dn, d.mkt_share.to_numpy(float), "mean", n=3)
    f["mkt_resid_mean5"] = earlier(dog, dn, d.mkt_resid.to_numpy(float), "mean", n=5)
    last_dn = earlier(dog, dn, dn, "last")
    f["days_since"] = dn - last_dn
    # runs at the track and the trip, and the box's record, earlier days only
    for name, key in (("track", d.track.astype(str)), ("trip", pd.to_numeric(d.distance_m, errors="coerce").astype(str)),
                      ("box", pd.to_numeric(d.trap, errors="coerce").astype(str))):
        k = pd.Series(dog).astype(str) + "|" + key.to_numpy()
        codes = pd.factorize(k)[0].astype(float)
        f[f"n_{name}"] = earlier(codes, dn, d.won.to_numpy(float), "count")
        if name == "box":
            bw = earlier(codes, dn, d.won.to_numpy(float), "mean", n=10_000)
            f["box_win"] = (bw * f.n_box + K / 8.0) / (f.n_box + K)
    f["box"] = pd.to_numeric(d.trap, errors="coerce").to_numpy(float)
    f["field"] = pd.Series(race).map(pd.Series(race).value_counts()).to_numpy(float)
    f["distance"] = pd.to_numeric(d.distance_m, errors="coerce").to_numpy(float)
    f["cls"] = pd.to_numeric(d.class_number, errors="coerce").to_numpy(float)
    f["track_code"] = pd.factorize(d.track.astype(str))[0].astype(float)
    for c in ("fig_mean3", "fig_ewm", "sec_mean3", "sec_ewm", "win_rate", "mkt_share_mean3"):
        f[f"{c}_vs"] = _vs(race, f[c].to_numpy(float))
        f[f"{c}_rank"] = pd.Series(f[c].to_numpy(float)).groupby(race).rank(ascending=False).to_numpy()
    return f.astype(np.float32), list(f.columns)


# ---- model and scores ------------------------------------------------------------------------------------------------

def race_norm(p, race) -> np.ndarray:
    s = pd.Series(p).groupby(race).transform("sum").to_numpy()
    return p / s


def ll(p, y) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clogit(X: np.ndarray, y: np.ndarray, race: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Conditional logit (one winner a race) by Newton's method: coefficients and their standard errors."""
    b = np.zeros(X.shape[1])
    codes = pd.factorize(race)[0]
    for _ in range(iters):
        u = X @ b
        u = u - pd.Series(u).groupby(codes).transform("max").to_numpy()
        e = np.exp(u)
        p = e / np.bincount(codes, e)[codes]
        g = X.T @ (y - p)
        Xm = np.vstack([np.bincount(codes, p * X[:, j])[codes] for j in range(X.shape[1])]).T
        Xc = X - Xm
        H = (Xc * p[:, None]).T @ Xc
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H + 1e-9 * np.eye(len(b)))))
    return b, se


def apply_clogit(X, b, race) -> np.ndarray:
    u = X @ b
    codes = pd.factorize(race)[0]
    u = u - pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u)
    return e / np.bincount(codes, e)[codes]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def paired(day, a, b, draws=DRAWS, seed=0) -> dict:
    g = pd.DataFrame({"d": day, "a": a, "b": b}).groupby("d")[["a", "b"]].agg(["sum", "size"])
    sa, sb, n = g[("a", "sum")].to_numpy(), g[("b", "sum")].to_numpy(), g[("a", "size")].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (draws, len(g)))
    diff = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    d0 = (sa.sum() - sb.sum()) / n.sum()
    return {"a": round(float(sa.sum() / n.sum()), 5), "b": round(float(sb.sum() / n.sum()), 5), "diff": round(float(d0), 5),
            "z": round(float(d0 / diff.std()), 2) if diff.std() > 0 else None, "days": int(len(g))}


def value_rule(p, bsp, won, day, edge) -> dict:
    ev = p * (bsp - 1.0) * (1 - COMMISSION) - (1 - p)
    sel = (ev > edge) & (bsp <= MAX_BSP)
    if not sel.any():
        return {"edge": edge, "bets": 0}
    ret = np.where(won[sel] == 1, (bsp[sel] - 1.0) * (1 - COMMISSION), -1.0)
    g = pd.DataFrame({"d": day[sel], "r": ret}).groupby("d").r.agg(["sum", "size"])
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(g), (DRAWS, len(g)))
    roi = g["sum"].to_numpy()[idx].sum(1) / g["size"].to_numpy()[idx].sum(1)
    return {"edge": edge, "bets": int(sel.sum()), "roi%": round(100 * float(ret.mean()), 2),
            "roi_90": [round(100 * float(np.quantile(roi, 0.05)), 2), round(100 * float(np.quantile(roi, 0.95)), 2)],
            "mean_bsp": round(float(bsp[sel].mean()), 2), "days": int(len(g))}


def run(store, first_year: int, test_years: list[int], out_dir: Path) -> dict:
    import lightgbm as lgb
    api = load_api(store, first_year)
    if not len(api):
        return {"note": "no data found: no GreyhoundAPI Australian results in the store"}
    api = api[api.position.notna()] if "position" in api else api
    d, jnotes = join_bsp(api, load_bsp(store, first_year))
    # the market price: the BSP where Betfair's price files hold it, else the last Betfair back price before the off
    # as GreyhoundAPI recorded it (amendment of 16 Nov 2026: the archived price files hold 167 Australian runners)
    back = pd.to_numeric(d.bf_back, errors="coerce").where(lambda x: x > 1)
    jnotes["bsp_rows"] = int(d.bsp.notna().sum())
    d["bsp"] = d.bsp.where(d.bsp > 1).fillna(back)
    jnotes["market_price_rows"] = int(d.bsp.notna().sum())
    jnotes["market_price"] = "BSP where held, else the last Betfair back price before the off (GreyhoundAPI)"
    d = figures(d)
    d = d[d.won.notna()].reset_index(drop=True)
    X, cols = features(d)
    d["year"] = d.race_date.str[:4].astype(int)
    out = {"join": jnotes, "rows": int(len(d)), "races": int(d.race_id.nunique()), "features": cols,
           "from": str(d.race_date.min()), "to": str(d.race_date.max()), "years": {}}
    prev = None
    for y in test_years:
        tr = (d.year < y) & (d.year >= y - TRAIN_YEARS)
        te = d.year == y
        if tr.sum() < 10_000 or te.sum() < 1_000:
            out["years"][str(y)] = {"note": "too few rows"}
            continue
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        m.fit(X[tr], d.won[tr])
        raw = m.predict_proba(X[te])[:, 1]
        t = d[te].copy()
        t["p_model"] = race_norm(raw, t.race_id.to_numpy())
        # races where every runner has a BSP and one winner
        g = t.groupby("race_id")
        full = g.bsp.transform(lambda s: s.notna().all() and (s > 1).all()) & (g.won.transform("sum") == 1)
        s = t[full].copy()
        if len(s) < 1000:
            out["years"][str(y)] = {"note": "too few races with a BSP for every runner", "rows": int(len(s))}
            continue
        rid = s.race_id.to_numpy()
        s["p_bsp"] = race_norm(1.0 / s.bsp.to_numpy(float), rid)
        Xb = np.column_stack([_logit(s.p_model.to_numpy()), _logit(s.p_bsp.to_numpy())])
        b, se = clogit(Xb, s.won.to_numpy(float), rid)
        yr = {"rows": int(len(s)), "races": int(s.race_id.nunique()),
              "benter": {"coef_model": round(float(b[0]), 4), "z_model": round(float(b[0] / se[0]), 2),
                         "coef_bsp": round(float(b[1]), 4), "z_bsp": round(float(b[1] / se[1]), 2)}}
        if prev is not None:
            p_bl = apply_clogit(Xb, prev, rid)
            s["p_blend"] = p_bl
            yr["logloss_blend_vs_bsp"] = paired(s.race_date.to_numpy(), ll(p_bl, s.won.to_numpy()),
                                                ll(s.p_bsp.to_numpy(), s.won.to_numpy()))
            yr["value_rule"] = [value_rule(p_bl, s.bsp.to_numpy(float), s.won.to_numpy(), s.race_date.to_numpy(), e)
                                for e in EDGES]
        yr["logloss_model_vs_bsp"] = paired(s.race_date.to_numpy(), ll(s.p_model.to_numpy(), s.won.to_numpy()),
                                            ll(s.p_bsp.to_numpy(), s.won.to_numpy()))
        imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
        yr["top_features"] = {k: int(v) for k, v in imp.head(12).items()}
        out["years"][str(y)] = yr
        prev = b
        out_dir.mkdir(parents=True, exist_ok=True)
        s[["race_id", "dog_id", "race_date", "track", "trap", "won", "bsp", "p_model", "p_bsp"]
          + (["p_blend"] if "p_blend" in s else [])].to_parquet(out_dir / f"pred_{y}.parquet", index=False)
        log.info("AU %d: %s", y, {k: v for k, v in yr.items() if k != "top_features"})
    return out


def report_md(r: dict) -> str:
    if "note" in r:
        return "# Australian greyhound feasibility\n\n" + r["note"] + "\n"
    L = ["# Australian greyhound feasibility", "",
         f"{r['rows']:,} runners in {r['races']:,} races ({r['from']} to {r['to']}); the BSP join: {r['join']}.", ""]
    for y, yr in r["years"].items():
        L.append(f"## {y}")
        if "note" in yr:
            L += [yr["note"], ""]
            continue
        bt = yr["benter"]
        L.append(f"{yr['races']:,} races with a BSP for every runner. Benter: model {bt['coef_model']} (z {bt['z_model']}), "
                 f"BSP {bt['coef_bsp']} (z {bt['z_bsp']}). Log-loss model {yr['logloss_model_vs_bsp']['a']} against the "
                 f"BSP's {yr['logloss_model_vs_bsp']['b']} (z {yr['logloss_model_vs_bsp']['z']}).")
        if "logloss_blend_vs_bsp" in yr:
            lb = yr["logloss_blend_vs_bsp"]
            L.append(f"Blend (fitted on the year before): {lb['a']} against {lb['b']}, {lb['diff']:+} (z {lb['z']}).")
            L += ["", "| Edge | Bets | Return % | 90% | Mean BSP |", "|---|---|---|---|---|"]
            for v in yr["value_rule"]:
                if v["bets"]:
                    L.append(f"| {v['edge']} | {v['bets']:,} | {v['roi%']} | {v['roi_90']} | {v['mean_bsp']} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first-year", type=int, default=2016)
    ap.add_argument("--test-years", default="2024,2025,2026")
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="out/au_feasibility")
    a = ap.parse_args(argv)
    o = Path(a.out)
    r = run(Store(root=a.store), a.first_year, [int(x) for x in a.test_years.split(",")], o)
    o.mkdir(parents=True, exist_ok=True)
    (o / "report.json").write_text(json.dumps(r, indent=1, default=str))
    md = report_md(r)
    (o / "report.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
