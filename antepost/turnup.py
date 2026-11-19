"""The ante-post turn-up model (docs/antepost_plan.md s.4; ledger antepost-turnup-1119).

For each horse listed in a GB/IE ante-post market at each mark before the off (180 days to 3 days), the chance of
each of three outcomes, which is what an ante-post back settles on (Betfair's rules, reports/antepost_data.md s.0):

  ran   ran in the race: the back is closed by a lay at the SP in the race-day market
  lost  did not run in it (all bets stand run or not): the stake is lost
  void  balloted out or removed by Betfair: the bet is void

Features are what was known at the mark only (antepost/phase0.py runner rows): the last traded price standing then,
its age, the trades so far, the price's move since the previous marks, its share of the market's prices, the number
listed, the market's suspensions so far, days listed, and the horse's own recent runs from betfair_prices (by name,
the day before the mark at the latest). Nothing from after the mark: not the final status, the last price, the total
trades or the removal time.

The test (registered before its run): fitted on races to 30 Jun 2025 (early stopping on 1 Jul to 31 Dec 2025), refit
on both, scored on 2026 races against two baselines fitted on 2024-25: the outcome shares by mark and price band, and
a multinomial logistic on the price and the days out. Multiclass log loss, paired by row, with a race-bootstrap
interval.

    python -m antepost.turnup --runners historic_runners.parquet --markets historic_markets.parquet \\
        --db horse_racing.db --out out/antepost_turnup
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
MARKS = {"d180": 180, "d120": 120, "d90": 90, "d60": 60, "d45": 45, "d30": 30, "d21": 21, "d14": 14, "d7": 7, "d3": 3}
ORDER = list(MARKS)                                    # far to near
CLASSES = ["lost", "ran", "void"]
BANDS = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 1001.0]
TRAIN_END, VALID_END = "2025-06-30", "2025-12-31"     # races to 30 Jun 2025 fit; to 31 Dec 2025 stop; 2026 test
GAP_MEETINGS = {("Haydock 22nd Nov", "2025-11-22"), ("Leopardstown 31st Jan", "2026-01-31")}  # betfair_prices lacks
FEATURES = ["days_out", "log_ltp", "has_price", "share", "price_rank", "n_listed", "n_priced", "age", "log_ntr",
            "susp", "chg_prev", "chg_prev2", "days_listed", "handicap", "jumps", "festival", "ireland", "month",
            "days_since_run", "runs_365", "ran_14"]
LGB_PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.03, "num_leaves": 15,
              "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 1.0, "verbose": -1, "seed": 7}
HANDICAP = re.compile(r"handicap|hcap|\bcup\b|national|cesarewitch|cambridgeshire|\bplate\b|wokingham|stewards|"
                      r"hunt cup|lincoln|chase$", re.I)
JUMPS = re.compile(r"chase|hurdle|national|bumper|gold cup", re.I)


# ----------------------------------------------------------------------------------------------------------------
# The panel: one row a horse a mark
# ----------------------------------------------------------------------------------------------------------------

def outcome(final_status, label) -> str:
    """What an ante-post back on the horse settles as: void when Betfair removed it (balloted out: void by the
    rules), ran when it ran in the race, else lost (run or not)."""
    if final_status == "REMOVED":
        return "void"
    return "ran" if label == "ran" else "lost"


def usable_markets(markets: pd.DataFrame, runners: pd.DataFrame) -> tuple[pd.Index, dict]:
    """GB/IE markets whose race was found, off the two meetings betfair_prices lacks, and not wholly removed (a market
    whose every runner was removed was replaced, not settled: Grand National 2025, 43 of 43)."""
    mk = markets.set_index("market_id")
    rep = {"markets_in": int(len(mk))}
    gbie = mk[mk["country"].isin(["GB", "IE"])].index
    rn = runners[runners["market_id"].isin(gbie)]
    found = rn.groupby("market_id")["label"].apply(lambda s: (s != "race_not_found").any())
    ok = found[found].index
    gap = rn.groupby("market_id")["race_date"].first()
    gap_ids = [m for m in ok if (mk.loc[m, "event_name"], gap.get(m)) in GAP_MEETINGS]
    removed_share = rn.groupby("market_id")["final_status"].apply(lambda s: (s == "REMOVED").mean())
    replaced = removed_share[removed_share > 0.9].index
    keep = pd.Index([m for m in ok if m not in set(gap_ids) and m not in set(replaced)])
    rep.update({"gbie": int(len(gbie)), "race_found": int(len(ok)), "dropped_gap_meetings": len(gap_ids),
                "dropped_wholly_removed": int(len(set(replaced) & set(ok))), "markets_used": int(len(keep))})
    return keep, rep


def build_panel(runners: pd.DataFrame, markets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keep, rep = usable_markets(markets, runners)
    mk = markets.set_index("market_id")
    rn = runners[runners["market_id"].isin(keep)].copy()
    rn["y"] = [outcome(f, l) for f, l in zip(rn["final_status"], rn["label"])]
    rows = []
    for i, k in enumerate(ORDER):
        if f"listed_{k}" not in rn:
            continue
        at = rn[rn[f"listed_{k}"] == True].copy()                      # noqa: E712 (None is not listed)
        if at.empty:
            continue
        p = pd.to_numeric(at[f"ltp_{k}"], errors="coerce")
        prev = ORDER[i - 1] if i >= 1 else None
        prev2 = ORDER[i - 2] if i >= 2 else None
        off = pd.to_datetime(at["market_id"].map(mk["market_time"]), utc=True)
        t = off - pd.Timedelta(days=MARKS[k])
        f = pd.DataFrame({
            "market_id": at["market_id"].values, "selection_id": at["selection_id"].values,
            "name_norm": at["name_norm"].values if "name_norm" in at else "", "race_date": at["race_date"].values,
            "mark": k, "days_out": MARKS[k], "t": t.values, "y": at["y"].values,
            "log_ltp": np.log(p).values, "has_price": p.notna().astype(int).values,
            "age": pd.to_numeric(at[f"age_{k}"], errors="coerce").values if f"age_{k}" in at else np.nan,
            "log_ntr": np.log1p(pd.to_numeric(at[f"ntr_{k}"], errors="coerce")).values if f"ntr_{k}" in at else np.nan,
            "susp": pd.to_numeric(at[f"susp_{k}"], errors="coerce").values if f"susp_{k}" in at else np.nan,
            "chg_prev": (np.log(p) - np.log(pd.to_numeric(at[f"ltp_{prev}"], errors="coerce"))).values
            if prev else np.nan,
            "chg_prev2": (np.log(p) - np.log(pd.to_numeric(at[f"ltp_{prev2}"], errors="coerce"))).values
            if prev2 else np.nan,
            "days_listed": ((t - pd.to_datetime(at["first_seen"], utc=True)).dt.total_seconds() / 86400).values,
        })
        rows.append(f)
    if not rows:
        return pd.DataFrame(), dict(rep, rows=0)
    df = pd.concat(rows, ignore_index=True)
    g = df.groupby(["market_id", "mark"])
    df["n_listed"] = g["y"].transform("size")
    df["n_priced"] = g["has_price"].transform("sum")
    inv = np.exp(-df["log_ltp"])
    df["share"] = inv / inv.groupby([df["market_id"], df["mark"]]).transform("sum")
    df["price_rank"] = df.groupby(["market_id", "mark"])["log_ltp"].rank(pct=True)
    name = df["market_id"].map(mk["market_name"]).fillna("")
    ev = df["market_id"].map(mk["event_name"]).fillna("")
    df["handicap"] = name.str.contains(HANDICAP).astype(int)
    df["jumps"] = name.str.contains(JUMPS).astype(int)
    df["festival"] = (~ev.str.contains(r"\d{1,2}(?:st|nd|rd|th)\s", regex=True)).astype(int)
    df["ireland"] = (df["market_id"].map(mk["country"]) == "IE").astype(int)
    df["month"] = pd.to_datetime(df["race_date"]).dt.month
    rep.update({"rows": int(len(df)), "horses": int(df[["market_id", "selection_id"]].drop_duplicates().shape[0]),
                "duplicates": int(df.duplicated(["market_id", "selection_id", "mark"]).sum()),
                "outcome_rows": df["y"].value_counts().to_dict(),
                "race_dates": [str(df["race_date"].min()), str(df["race_date"].max())]})
    return df, rep


def add_recent_runs(df: pd.DataFrame, db: str | None) -> tuple[pd.DataFrame, dict]:
    """The horse's own runs before the mark's day (betfair_prices WIN, by normalised name): days since the last, runs
    in the 365 days before, and whether it ran in the 14 days before. A day before the mark at the latest."""
    for c in ("days_since_run", "runs_365", "ran_14"):
        df[c] = np.nan
    if df.empty or not db or not Path(db).exists():
        return df, {"skipped": f"no database at {db}"}
    names = sorted(set(df["name_norm"].dropna()) - {""})
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    lo = (pd.to_datetime(df["t"]).min() - pd.Timedelta(days=400)).date().isoformat()
    frames = []
    for i in range(0, len(names), 500):
        part = names[i:i + 500]
        q = (f"SELECT DISTINCT horse_norm AS name_norm, race_date FROM betfair_prices WHERE lower(market_type)='win' "
             f"AND race_date >= ? AND horse_norm IN ({','.join('?' * len(part))})")
        frames.append(pd.read_sql_query(q, con, params=[lo] + part))
    con.close()
    runs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["name_norm", "race_date"])
    runs["day"] = pd.to_datetime(runs["race_date"])
    by = {n: np.sort(g["day"].values) for n, g in runs.groupby("name_norm")}
    mday = pd.to_datetime(df["t"]).dt.tz_convert(None).dt.normalize().values     # runs strictly before this day
    dsr, r365, r14 = [], [], []
    for n, d in zip(df["name_norm"], mday):
        a = by.get(n)
        if a is None or not len(a):
            dsr.append(np.nan), r365.append(0), r14.append(0)
            continue
        j = np.searchsorted(a, d, side="left")                      # runs on days < d
        before = a[:j]
        if not len(before):
            dsr.append(np.nan), r365.append(0), r14.append(0)
            continue
        gap = (d - before[-1]) / np.timedelta64(1, "D")
        dsr.append(gap)
        r365.append(int((before >= d - np.timedelta64(365, "D")).sum()))
        r14.append(int(gap <= 14))
    df["days_since_run"], df["runs_365"], df["ran_14"] = dsr, r365, r14
    return df, {"run_rows": int(len(runs)), "horses_with_a_run": int(df["days_since_run"].notna().sum())}


# ----------------------------------------------------------------------------------------------------------------
# Baselines, model, scoring
# ----------------------------------------------------------------------------------------------------------------

def y_index(y: pd.Series) -> np.ndarray:
    return y.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def band_of(df: pd.DataFrame) -> pd.Series:
    b = pd.cut(np.exp(df["log_ltp"]), BANDS).astype(str)
    return df["mark"] + "|" + b.where(df["has_price"] == 1, "no_price")


def fit_table(train: pd.DataFrame, alpha: float = 1.0):
    """B1: the outcome shares by mark and price band, smoothed toward the mark's own shares."""
    yi = y_index(train["y"])
    key = band_of(train)
    mark_p = {m: (np.bincount(yi[(train["mark"] == m).to_numpy()], minlength=3) + alpha) for m in train["mark"].unique()}
    mark_p = {m: v / v.sum() for m, v in mark_p.items()}
    allp = (np.bincount(yi, minlength=3) + alpha) / (len(yi) + 3 * alpha)
    tab = {}
    for k in key.unique():
        sel = (key == k).to_numpy()
        prior = mark_p.get(k.split("|")[0], allp) * 10
        c = np.bincount(yi[sel], minlength=3) + prior
        tab[k] = c / c.sum()

    def predict(df):
        keys = band_of(df)
        return np.vstack([tab.get(k, mark_p.get(k.split("|")[0], allp)) for k in keys])
    return predict


def _logit_x(df: pd.DataFrame) -> np.ndarray:
    lp = df["log_ltp"].fillna(np.log(1000.0)).to_numpy()
    d = np.log(df["days_out"].to_numpy())
    return np.column_stack([lp, lp ** 2, df["has_price"].to_numpy(), d, lp * d])


def fit_logit(train: pd.DataFrame):
    """B2: a multinomial logistic on the price (and its square, and whether there is one) and the days out."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(_logit_x(train))
    m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(_logit_x(train)), y_index(train["y"]))
    order = np.argsort(m.classes_)
    return lambda df: m.predict_proba(sc.transform(_logit_x(df)))[:, order]


def fit_gbm(train: pd.DataFrame, valid: pd.DataFrame | None, rounds: int | None = None):
    import lightgbm as lgb
    dtr = lgb.Dataset(train[FEATURES], y_index(train["y"]))
    if rounds is None:
        dva = lgb.Dataset(valid[FEATURES], y_index(valid["y"]), reference=dtr)
        b = lgb.train(LGB_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return b, int(b.best_iteration or b.current_iteration())
    return lgb.train(LGB_PARAMS, dtr, num_boost_round=rounds), rounds


def row_logloss(p: np.ndarray, yi: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(yi)), yi], 1e-12, 1.0))


def race_bootstrap(diff: np.ndarray, groups: np.ndarray, draws: int = 2000, seed: int = 11) -> list[float]:
    """90% interval of the mean paired difference, resampling whole races (markets) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv)
    stats = []
    for _ in range(draws):
        pick = rng.integers(0, len(uniq), len(uniq))
        stats.append(sums[pick].sum() / counts[pick].sum())
    return [round(float(np.quantile(stats, 0.05)), 5), round(float(np.quantile(stats, 0.95)), 5)]


def ece(p_ran: np.ndarray, ran: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p_ran, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p_ran, side="right") - 1, 0, bins - 1)
    return float(sum(abs(p_ran[idx == b].mean() - ran[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def evaluate(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rd = pd.to_datetime(df["race_date"])
    tr, va, te = df[rd <= TRAIN_END], df[(rd > TRAIN_END) & (rd <= VALID_END)], df[rd > VALID_END]
    trva = pd.concat([tr, va])
    rep = {"split": {name: {"rows": int(len(x)), "races": int(x["market_id"].nunique()),
                            "dates": [str(x["race_date"].min()), str(x["race_date"].max())],
                            "outcomes": x["y"].value_counts().to_dict()}
                     for name, x in (("train", tr), ("valid", va), ("test", te))}}
    if te.empty or tr.empty or va.empty:
        return dict(rep, skipped="a split is empty"), pd.DataFrame()
    _, rounds = fit_gbm(tr, va)
    gbm, _ = fit_gbm(trva, None, rounds)
    preds = {"model": gbm.predict(te[FEATURES]), "b1_table": fit_table(trva)(te), "b2_logit": fit_logit(trva)(te)}
    yi = y_index(te["y"])
    ll = {k: row_logloss(p, yi) for k, p in preds.items()}
    ran = (te["y"] == "ran").to_numpy().astype(float)
    groups = te["market_id"].to_numpy()
    rep["rounds"] = rounds
    rep["test"] = {k: {"logloss": round(float(v.mean()), 5),
                       "brier_ran": round(float(((preds[k][:, 1] - ran) ** 2).mean()), 5),
                       "ece_ran": round(ece(preds[k][:, 1], ran), 5)} for k, v in ll.items()}
    from sklearn.metrics import roc_auc_score
    for k in preds:
        rep["test"][k]["auc_ran"] = round(float(roc_auc_score(ran, preds[k][:, 1])), 4)
    rep["paired"] = {}
    for b in ("b1_table", "b2_logit"):
        d = ll["model"] - ll[b]
        rep["paired"][f"model_less_{b}"] = {"mean": round(float(d.mean()), 5), "interval_90": race_bootstrap(d, groups)}
    by_mark = {}
    for m in ORDER:
        sel = (te["mark"] == m).to_numpy()
        if sel.sum():
            by_mark[m] = {"rows": int(sel.sum()), **{k: round(float(v[sel].mean()), 5) for k, v in ll.items()}}
    rep["by_mark"] = by_mark
    imp = gbm.feature_importance("gain")
    rep["importance_gain"] = {f: round(float(g), 1) for f, g in sorted(zip(FEATURES, imp), key=lambda x: -x[1])}
    # calibration of P(ran) by decile, the model
    pr = preds["model"][:, 1]
    q = pd.qcut(pr, 10, labels=False, duplicates="drop")
    rep["calibration_ran_model"] = [{"decile": int(b), "pred": round(float(pr[q == b].mean()), 4),
                                     "obs": round(float(ran[q == b].mean()), 4), "n": int((q == b).sum())}
                                    for b in sorted(set(q))]
    out = te[["market_id", "selection_id", "race_date", "mark", "y", "log_ltp"]].copy()
    for k, p in preds.items():
        for i, c in enumerate(CLASSES):
            out[f"{k}_{c}"] = p[:, i]
    return rep, out


def reading(rep: dict) -> dict:
    """The registered reading (ledger antepost-turnup-1119), applied as written."""
    if "paired" not in rep:
        return {"verdict": "NOT RUN", "why": rep.get("skipped")}
    p = rep["paired"]
    beats = all(v["interval_90"][1] < 0 for v in p.values())
    marks = [m for m in ("d60", "d45", "d30", "d21", "d14", "d7", "d3") if m in rep["by_mark"]]
    better = sum(rep["by_mark"][m]["model"] <= rep["by_mark"][m]["b2_logit"] for m in marks)
    calib = rep["test"]["model"]["ece_ran"] <= rep["test"]["b2_logit"]["ece_ran"] + 0.01
    ok = beats and better * 2 > len(marks) and calib
    return {"verdict": "PASS" if ok else "FAIL", "beats_both_baselines_interval_below_0": beats,
            "marks_no_worse_than_b2": f"{better} of {len(marks)}", "calibration_ok": calib}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runners", required=True)
    ap.add_argument("--markets", required=True)
    ap.add_argument("--db", default="horse_racing.db")
    ap.add_argument("--out", default="out/antepost_turnup")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from betfair_prices import normalise_horse
    rn = pd.read_parquet(a.runners)
    rn["name_norm"] = rn["runner_name"].map(normalise_horse)
    mk = pd.read_parquet(a.markets)
    rep = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    df, rep["panel"] = build_panel(rn, mk)
    df, rep["recent_runs"] = add_recent_runs(df, a.db)
    rep["eval"], preds = evaluate(df)
    rep["reading"] = reading(rep["eval"])
    if not preds.empty:
        preds.to_parquet(out / "turnup_test_predictions.parquet", index=False)
    df.drop(columns=["t"]).to_parquet(out / "turnup_panel.parquet", index=False)
    (out / "antepost_turnup.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
