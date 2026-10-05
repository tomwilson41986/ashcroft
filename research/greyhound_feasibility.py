"""Can a greyhound model beat the market? A first feasibility study (the owner's question, 5 Oct 2026).

Data: the owner's Betfair Historic Data bundle (BASIC, GB greyhound WIN markets, Jan-Sep 2026: BSP, the last traded
price a minute before the off, the result) joined to GBGB's results (every GB run from Jul 2025: trap, sectional,
calculated time, position, comment, grade, weight).

Design, fixed before the first run:
- each runner's features come only from runs before the race (GBGB, by dog id);
- the model is a LightGBM classifier of "won" that starts from the market's own view at T-1 minute (the log-odds of
  the implied chance from the last traded price a minute before the off, normalised in the race) and learns the form
  features' correction to it; its output normalised within each race;
- fit: Jan-Apr 2026; the betting threshold is chosen on May-Jun; Jul-Sep is scored once, with the chosen rule;
- the rule decides at T-1 on the model's chance and the T-1 price, backs at the Betfair SP (a MARKET_ON_CLOSE back),
  and is settled at the BSP with 2% commission on winnings; CLV is the T-1 price against the BSP.

    python research/greyhound_feasibility.py --store <store root>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

COM = 0.02
#: Betfair's venue names -> GBGB's track names, where they differ (none in 2025-26: both say "Dunstall Park", "Monmore")
VENUE_TO_TRACK: dict[str, str] = {}


def norm_track(s: str) -> str:
    s = re.sub(r"[^a-z ]", "", str(s).lower()).strip()
    return VENUE_TO_TRACK.get(s, s)


def norm_name(s: str) -> str:
    s = re.sub(r"^\d+\.\s*", "", str(s))
    return re.sub(r"[^a-z]", "", s.lower())


# --------------------------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------------------------

def betfair_win(store: Path) -> pd.DataFrame:
    df = pd.read_parquet(store / "betfair_historic" / "markets_greyhound_racing_2026.parquet")
    w = df[(df.market_type == "WIN") & (df.country == "GB") & df.runner_status.isin(["WINNER", "LOSER"])].copy()
    w["t"] = pd.to_datetime(w.market_time, utc=True)
    uk = w.t.dt.tz_convert("Europe/London")
    w["race_date"] = uk.dt.strftime("%Y-%m-%d")
    w["hhmm"] = uk.dt.strftime("%H:%M")
    w["track"] = w.venue.map(norm_track)
    w["trap"] = pd.to_numeric(w.runner_name.str.extract(r"^(\d+)\.")[0], errors="coerce")
    w["name"] = w.runner_name.map(norm_name)
    return w


def gbgb_runs(store: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted((store / "gbgb").glob("runs_*.parquet"))]
    g = pd.concat(parts, ignore_index=True)
    g = g.dropna(subset=["dog_id", "race_date"])
    g["track"] = g.track.map(norm_track)
    g["hhmm"] = g.race_time.astype(str).str[:5]
    g["name"] = g.dog_name.map(norm_name)
    g["t"] = pd.to_datetime(g.race_date + " " + g.race_time.astype(str), errors="coerce")
    return g.sort_values(["t", "race_id", "trap"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------
# Features: each from the dog's runs before this one
# --------------------------------------------------------------------------------------------------------------------

def class_rank(c) -> float:
    """GBGB grades: A1 (best) .. A11, D1..D4 sprint, OR open race; lower is better."""
    m = re.match(r"([A-Z]+)(\d+)?", str(c or ""))
    if not m:
        return np.nan
    letter, num = m.group(1), float(m.group(2) or 0)
    return {"OR": 0.0, "A": num, "D": num, "S": num, "H": num, "B": num + 2, "P": num}.get(letter, np.nan)


def features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["won"] = (g.position == 1).astype(float)
    g["cls"] = g.race_class.map(class_rank)
    key = ["track", "distance_m"]
    # the race's standard: each run's calculated time against the track and distance's median of earlier runs
    g["std_time"] = g.groupby(key)["adjusted_time"].transform(lambda s: s.expanding().median().shift(1))
    g["time_vs_std"] = g.adjusted_time - g.std_time
    g["std_sec"] = g.groupby(key)["sectional"].transform(lambda s: s.expanding().median().shift(1))
    g["sec_vs_std"] = g.sectional - g.std_sec
    c = g.comment.fillna("")
    g["rails"] = c.str.contains(r"Rls|Rails|RlsRn", case=False).astype(float)
    g["wide"] = c.str.contains(r"(?:^|,)W(?:,|$)|Wide|VW", case=False).astype(float)
    g["quick"] = c.str.contains(r"QAw|QuickAway|EP|EarlyPace|Led1|LedRunUp", case=False).astype(float)
    g["slow"] = c.str.contains(r"SAw|SlowAway|Slow", case=False).astype(float)
    g["crowd"] = c.str.contains(r"Crd|Bmp|Blk|Chk", case=False).astype(float)
    by = g.groupby("dog_id", sort=False)

    def past(col, n, how="mean"):
        s = by[col].transform(lambda x: getattr(x.shift(1).rolling(n, min_periods=1), how)())
        return s

    out = pd.DataFrame(index=g.index)
    out["runs_before"] = by.cumcount()
    out["days_since"] = (g.t - by.t.shift(1)).dt.days
    for n in (1, 3, 6):
        out[f"pos_{n}"] = past("position", n)
        out[f"time_{n}"] = past("time_vs_std", n)
        out[f"sec_{n}"] = past("sec_vs_std", n)
    out["win_6"] = past("won", 6)
    out["best_time_6"] = past("time_vs_std", 6, "min")
    out["best_sec_6"] = past("sec_vs_std", 6, "min")
    for col in ("rails", "wide", "quick", "slow", "crowd"):
        out[f"{col}_6"] = past(col, 6)
    out["cls_change"] = g.cls - by.cls.shift(1)
    out["weight_change"] = g.weight_kg - by.weight_kg.shift(1)
    out["same_track_last"] = (g.track == by.track.shift(1)).astype(float)
    out["dist_change"] = g.distance_m - by.distance_m.shift(1)
    out["trap"] = g.trap
    out["runners"] = g.runners
    out["distance_m"] = g.distance_m
    out["cls"] = g.cls
    # the trap against the dog's style: a railer drawn wide, a wide runner drawn on the inside
    out["rail_x_trap"] = out.rails_6 * (g.trap - 3.5)
    out["wide_x_trap"] = out.wide_6 * (3.5 - g.trap)
    # the track and distance's trap bias from earlier races (win rate by trap)
    tb = g.groupby(key + ["trap"])["won"].transform(lambda s: s.expanding().mean().shift(1))
    out["trap_bias"] = tb
    feats = list(out.columns)
    # race-relative: each runner against the field
    out["race_id"] = g.race_id
    for col in ("time_3", "sec_3", "best_time_6", "best_sec_6", "pos_3", "quick_6"):
        grp = out.groupby("race_id")[col]
        out[f"{col}_rel"] = out[col] - grp.transform("mean")
        out[f"{col}_rank"] = grp.rank(method="average")
        feats += [f"{col}_rel", f"{col}_rank"]
    keep = ["race_id", "race_date", "hhmm", "track", "trap", "name", "dog_id", "position"]
    res = pd.concat([g[keep].reset_index(drop=True), out.drop(columns=["race_id", "trap"]).reset_index(drop=True)],
                    axis=1)
    res.attrs["features"] = [f for f in feats if f != "race_id"]
    return res


# --------------------------------------------------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------------------------------------------------

def join(w: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    m = w.merge(f, on=["race_date", "track", "hhmm", "trap"], how="inner", suffixes=("", "_g"))
    m = m[(m.name == m.name_g) | m.name_g.isna()]                  # the trap's dog is the market's runner
    sizes = w.groupby("market_id").size()
    got = m.groupby("market_id").size()
    whole = got[got == sizes.reindex(got.index)].index               # every runner of the market matched
    return m[m.market_id.isin(whole)].copy()


def race_norm(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] / df.groupby("market_id")[col].transform("sum")


def settle(sel: pd.DataFrame, price: str = "bsp") -> dict:
    p = sel[price]
    pnl = np.where(sel.won == 1, (p - 1) * (1 - COM), -1.0)
    n = len(sel)
    if not n:
        return {"bets": 0}
    se = pnl.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"bets": n, "roi%": round(100 * pnl.mean(), 2), "90%": [round(100 * (pnl.mean() - 1.645 * se), 2),
                                                                   round(100 * (pnl.mean() + 1.645 * se), 2)],
            "clv_t1_vs_bsp%": round(100 * (sel.bsp / sel.ltp_t_1 - 1).median(), 2),
            "per_day": round(n / max(1, sel.race_date.nunique()), 1)}


def main(argv=None) -> int:
    import lightgbm as lgb
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    store = Path(a.store)
    w = betfair_win(store)
    f = features(gbgb_runs(store))
    feats = f.attrs["features"]
    m = join(w, f)
    m = m.dropna(subset=["ltp_t_1", "bsp"])
    m = m[m.ltp_t_1 > 1]
    m["mkt_p"] = 1 / m.ltp_t_1
    m["mkt_p"] = race_norm(m, "mkt_p")
    m["mkt_logit"] = np.log(m.mkt_p / (1 - m.mkt_p))
    m["mkt_rank"] = m.groupby("market_id").mkt_p.rank(ascending=False)
    month = m.t.dt.month
    fit, val, test = m[month <= 4], m[month.isin([5, 6])], m[month >= 7]
    report = {"joined_markets": int(m.market_id.nunique()), "betfair_markets": int(w.market_id.nunique()),
              "split_markets": [int(x.market_id.nunique()) for x in (fit, val, test)]}
    params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)
    # the model learns the form's correction to the market: the market's T-1 view is its starting point (init_score),
    # the form features the residual; rounds by early stopping on the fit window's last fifth (by time)
    cut = fit.t.quantile(0.8)
    tr, es = fit[fit.t < cut], fit[fit.t >= cut]
    model = lgb.train({**params, "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 400},
                      lgb.Dataset(tr[feats], tr.won, init_score=tr.mkt_logit),
                      2000, valid_sets=[lgb.Dataset(es[feats], es.won, init_score=es.mkt_logit)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    report["rounds"] = model.best_iteration
    for name, d in (("val", val), ("test", test)):
        d["p_market"] = d.mkt_p
        d["p_model"] = 1 / (1 + np.exp(-(d.mkt_logit + model.predict(d[feats], num_iteration=model.best_iteration,
                                                                      raw_score=True))))
        d["p_model"] = race_norm(d, "p_model")
        ll = {t: float(-np.mean(np.log(np.where(d.won == 1, d[f"p_{t}"], 1 - d[f"p_{t}"]).clip(1e-9))))
              for t in ("market", "model")}
        bsp_p = race_norm(d.assign(bp=1 / d.bsp), "bp")
        ll["bsp"] = float(-np.mean(np.log(np.where(d.won == 1, bsp_p, 1 - bsp_p).clip(1e-9))))
        report[f"logloss_{name}"] = {k: round(v, 5) for k, v in ll.items()}
    # the rule: back at the SP where the model's expected return at the T-1 price clears the edge
    for d in (val, test):
        d["edge"] = d.p_model * (1 + (d.ltp_t_1 - 1) * (1 - COM)) - 1
    grid = {}
    for e in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        for cap in (10.0, 20.0, 1000.0):
            sel = val[(val.edge > e) & (val.ltp_t_1 <= cap)]
            grid[(e, cap)] = settle(sel)
    report["val_grid"] = {f"edge>{e:.2f}, T-1 price<={c:g}": v for (e, c), v in grid.items()}
    ok = {k: v for k, v in grid.items() if v.get("bets", 0) >= 300}
    best = max(ok, key=lambda k: ok[k]["90%"][0]) if ok else (0.10, 1000.0)
    report["chosen_on_val"] = {"edge": best[0], "price_cap": best[1]}
    sel = test[(test.edge > best[0]) & (test.ltp_t_1 <= best[1])]
    report["test"] = settle(sel)
    report["test_at_t1_price"] = settle(sel, "ltp_t_1")
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    report["top_features"] = {k: round(float(v / imp.sum()), 3) for k, v in imp.head(12).items()}
    txt = json.dumps(report, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
