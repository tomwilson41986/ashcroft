"""The owner's race book (30 Sep): the closing-price model and the book, backtested on Betfair's morning prices.

Development window only: January-March 2026, nothing on or after 1 April read. The served blend's walk-forward
forecasts (iteration 105's oos_gate_step.csv, run 36407030755) joined to Betfair's win prices: the morning
volume-weighted price (the entry), the morning volume, and the BSP (the close). A race is used only when every
runner the forecasts carry has both, so each book is a whole race.

Walk-forward by month: the closing model is fitted on the months before each test month (January for February,
January-February for March) and scored on it; nothing is fitted on the month it is scored on.

1. The closing price: how near each forecast is to the BSP book (mean |log q - log q_close| per runner, and the
   KL divergence per race): the morning market alone, the model alone, the two averaged, the closing model.
2. Positions and books, closed at the BSP (closing-line value, commission on each race's net winnings) and held
   to the result at the morning price. Positions only where the runner had at least GBP100 matched in the
   morning (a real price). Lays at the morning price, and 2% worse as a check.
   rule22       back where the model is at least 22% shorter than the morning price (the pre-registered rule)
   single_back  back where the closing model's expected CLV is at least 5%
   single_lay   lay where the closing model's expected CLV on the lay is at least 5%
   dutch        the owner's subset book: runners by expected edge, each at least -3%, the subset's book at
                least +3%, stakes dutched, one unit a race
   close_back   Kelly on the value at the close, backs only (exposure one unit, half of it at most on a runner)
   close_both   the same with lays
   hold_back    Kelly on the result with the closing model's probabilities (a quarter-unit cap), backs only
   hold_both    the same with lays
Money at risk is stakes plus lay liabilities; every return is per unit of it.
"""

from __future__ import annotations

import io
import os
import sqlite3
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model import race_book as rb  # noqa: E402

RUN, ARTIFACT, MEMBER = 36407030755, "research-iter105-served-gated-race-xent-126", "oos_gate_step.csv"
KEY = ["race_date", "track", "race_time", "horse_name"]
START, UNTIL = "2026-01-01", "2026-04-01"
MIN_VOL = 100.0
DELTA, THETA = 0.03, 0.03
OUT = Path("out/race_book")


def fetch_forecasts() -> pd.DataFrame:
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts", headers=auth,
                     params={"name": ARTIFACT}, timeout=60)
    arts = r.json().get("artifacts", []) if r.ok else []
    if not arts:
        raise SystemExit(f"artifact {ARTIFACT} of run {RUN} not found ({r.status_code})")
    z = requests.get(arts[0]["archive_download_url"], headers=auth, timeout=600)
    z.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(z.content))
    inner = next(n for n in zf.namelist() if n.split("/")[-1] == MEMBER)
    return pd.read_csv(io.BytesIO(zf.read(inner)), dtype={"race_time": str})


def load(forecasts: str | None = None, extract: str | None = None) -> pd.DataFrame:
    """The forecasts joined to Betfair's win prices; by default the artifact and the database, or local copies
    (``forecasts`` a csv, ``extract`` the export betfair_prices.py writes). Nothing on or after UNTIL is kept."""
    o = pd.read_csv(forecasts, dtype={"race_time": str}) if forecasts else fetch_forecasts()
    o = o[(o["race_date"].astype(str) >= START) & (o["race_date"].astype(str) < UNTIL)].copy()
    cols = KEY + ["morningwap", "morning_vol", "bsp"]
    if extract:
        ex = pd.read_csv(extract, dtype={"race_time": str})
        ex = ex[ex["race_results_id"].notna() & (ex["market_type"] == "win")]
        ex = ex[(ex["race_date"].astype(str) >= START) & (ex["race_date"].astype(str) < UNTIL)][cols]
    else:
        conn = sqlite3.connect("horse_racing.db")
        ex = pd.read_sql_query(
            """SELECT r.race_date, r.track, r.race_time, r.horse_name, b.morningwap, b.morning_vol, b.bsp
               FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
               WHERE b.market_type = 'win' AND r.race_date >= ? AND r.race_date < ?""", conn, params=(START, UNTIL))
        conn.close()
    ex = ex.drop_duplicates(KEY)
    o["race"] = o["race_date"].astype(str) + "|" + o["track"].astype(str) + "|" + o["race_time"].astype(str)
    field = o.groupby("race").size().rename("field")
    d = o.merge(ex, on=KEY, how="inner")
    d = d[(d["morningwap"] > 1) & (d["bsp"] > 1) & (d["predicted_bfsp"] > 1)]
    got = d.groupby("race").size().rename("got")
    whole = pd.concat([field, got], axis=1).dropna()
    whole = whole[(whole["got"] == whole["field"]) & (whole["field"] >= 3)].index
    d = d[d["race"].isin(whole)].copy()
    d["month"] = d["race_date"].astype(str).str[:7]
    d["won"] = d["won"].astype(bool)
    print(f"forecasts {len(o):,} runners in {o['race'].nunique():,} races, {START} to {UNTIL} (exclusive); "
          f"whole races with Betfair's morning price and BSP for every runner: {d['race'].nunique():,} "
          f"({len(d):,} runners)")
    return d.sort_values(["race_date", "race", "predicted_bfsp"]).reset_index(drop=True)


def boot(num: np.ndarray, den: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Ratio of sums with a 90% race-bootstrap interval."""
    if den.sum() <= 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(num.sum() / den.sum()), float(np.percentile(r, 5)), float(np.percentile(r, 95))


def closing_accuracy(test: pd.DataFrame, qhat: np.ndarray, g: np.ndarray) -> dict:
    qc = rb.normalise_within(1 / test["bsp"].to_numpy(float), g)
    srcs = {"morning market": rb.normalise_within(1 / test["morningwap"].to_numpy(float), g),
            "model": rb.normalise_within(1 / test["predicted_bfsp"].to_numpy(float), g)}
    srcs["the two averaged"] = rb.normalise_within(np.sqrt(srcs["morning market"] * srcs["model"]), g)
    srcs["closing model"] = qhat
    out = {}
    n_races = g.max() + 1
    for k, q in srcs.items():
        mae = float(np.mean(np.abs(np.log(q) - np.log(qc))))
        kl = float(np.sum(qc * (np.log(qc) - np.log(q))) / n_races)
        out[k] = (mae, kl)
    return out


def race_positions(r: pd.DataFrame, model: rb.ClosingModel, rng, lay_worse: float = 0.0) -> dict:
    """Every strategy's stakes for one race: {name: (back stakes, lay stakes, lay prices)} plus the edges."""
    g = np.zeros(len(r), int)
    inputs = rb.closing_inputs(r)
    qhat = model.predict(inputs, g)
    sig = model.sigma(inputs)
    m = r["morningwap"].to_numpy(float)
    lay = m * (1 + lay_worse)
    allow = r["morning_vol"].fillna(0).to_numpy(float) >= MIN_VOL
    draws = rb.closing_draws(qhat, sig, book=model.book, n_draws=300, rng=rng)
    A = rb.close_payoffs(draws, m, lay)
    e = A.mean(0)
    n = len(r)
    z = np.zeros(n)
    pos = {}
    f = r["predicted_bfsp"].to_numpy(float)
    pos["rule22"] = ((np.log(m / f) >= 0.2) & allow).astype(float), z, lay
    pos["single_back"] = ((e[:n] >= 0.05) & allow).astype(float), z, lay
    pos["single_lay"] = z, ((e[n:] >= 0.05) & allow).astype(float), lay
    pos["dutch"] = rb.dutch_book(m, e[:n], draws.mean(0), 1.0, DELTA, THETA, allow=allow), z, lay
    pos["dutch_6"] = rb.dutch_book(m, e[:n], draws.mean(0), 1.0, DELTA, 0.06, allow=allow), z, lay
    pos["dutch_10"] = rb.dutch_book(m, e[:n], draws.mean(0), 1.0, DELTA, 0.10, allow=allow), z, lay
    no = np.zeros(n, bool)
    for name, payoff, w, cap, lays in (("close_back", A, np.ones(len(A)), 1.0, no),
                                       ("close_both", A, np.ones(len(A)), 1.0, allow),
                                       ("hold_back", rb.result_payoffs(m, lay), qhat, 0.25, no),
                                       ("hold_both", rb.result_payoffs(m, lay), qhat, 0.25, allow)):
        b = rb.build_book(payoff, w, lay, delta=DELTA, theta=THETA, exposure_cap=cap,
                          stake_cap=0.5 * cap, allow_back=allow, allow_lay=lays)
        pos[name] = b.back, b.lay, lay
    return {"pos": pos, "edges": e, "qhat": qhat}


def settle(r: pd.DataFrame, name: str, back_s, lay_s, lay, edges) -> dict:
    m, bsp = r["morningwap"].to_numpy(float), r["bsp"].to_numpy(float)
    won = np.flatnonzero(r["won"].to_numpy())
    winner = int(won[0]) if len(won) else None
    n = len(r)
    exposure = float(back_s.sum() + lay_s @ (lay - 1))
    if exposure <= 0:
        return {}
    e_book = float(edges[:n] @ back_s + edges[n:] @ lay_s)
    level = (back_s > 0) & (edges[:n] <= 0)
    return {"strategy": name, "race": r["race"].iat[0], "month": r["month"].iat[0], "exposure": exposure,
            "clv": rb.settle_close(back_s, lay_s, m, lay, bsp), "result": rb.settle_result(back_s, lay_s, m, lay, winner),
            "expected": e_book, "n_back": int((back_s > 0).sum()), "n_lay": int((lay_s > 0).sum()),
            "n_level_or_under": int(level.sum()),
            "clv_level_or_under": float(back_s[level] @ (m[level] / bsp[level] - 1)) if level.any() else 0.0}


def summarise(rows: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'strategy':12s} {'races':>6s} {'pos/race':>8s} {'at risk':>8s} {'CLV per unit (90%)':>28s} "
          f"{'race-level CLV':>16s} {'races +':>7s} {'expected':>8s} {'result per unit (90%)':>28s} "
          f"{'Sharpe CLV':>10s} {'Sharpe result':>13s}")
    for name, s in rows.groupby("strategy", sort=False):
        c = boot(s["clv"].to_numpy(), s["exposure"].to_numpy())
        eq = boot((s["clv"] / s["exposure"]).to_numpy(), np.ones(len(s)))
        res = boot(s["result"].to_numpy(), s["exposure"].to_numpy(), seed=1)
        npos = (s["n_back"] + s["n_lay"]).mean()
        # per race at each race's own stakes scaled to one unit at risk: mean over spread, a scale-free score
        v, rr = (s["clv"] / s["exposure"]).to_numpy(), (s["result"] / s["exposure"]).to_numpy()
        print(f"  {name:12s} {len(s):6,d} {npos:8.2f} {s['exposure'].sum():8.1f} "
              f"{100 * c[0]:+7.2f}% ({100 * c[1]:+6.2f} to {100 * c[2]:+6.2f}) {100 * eq[0]:+14.2f}% "
              f"{(s['clv'] > 0).mean():7.0%} {100 * s['expected'].sum() / s['exposure'].sum():+7.2f}% "
              f"{100 * res[0]:+7.2f}% ({100 * res[1]:+6.2f} to {100 * res[2]:+6.2f}) "
              f"{v.mean() / v.std():10.3f} {rr.mean() / rr.std():13.3f}")


def paired(rows: pd.DataFrame, a: str, b: str, n: int = 2000) -> str:
    """CLV per unit at risk of strategy a less strategy b, races resampled together (every race either bet)."""
    x = rows[rows["strategy"] == a].set_index("race")[["clv", "exposure"]]
    y = rows[rows["strategy"] == b].set_index("race")[["clv", "exposure"]]
    races = x.index.union(y.index)
    x, y = x.reindex(races).fillna(0), y.reindex(races).fillna(0)
    rng = np.random.default_rng(7)
    w = np.stack([np.bincount(rng.integers(0, len(races), len(races)), minlength=len(races)) for _ in range(n)])
    d = (w @ x["clv"].to_numpy()) / (w @ x["exposure"].to_numpy()) - (w @ y["clv"].to_numpy()) / (w @ y["exposure"].to_numpy())
    point = x["clv"].sum() / x["exposure"].sum() - y["clv"].sum() / y["exposure"].sum()
    return f"  {a:12s} less {b:12s} {100 * point:+6.2f} points ({100 * np.percentile(d, 5):+.2f} to {100 * np.percentile(d, 95):+.2f})"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", help="local copy of the forecasts (default: the iteration 105 artifact)")
    ap.add_argument("--extract", help="local Betfair export (default: the database)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load(args.forecasts, args.extract)
    months = sorted(d["month"].unique())
    rows, rows_worse, books = [], [], []
    for test_month in months[1:]:
        train, test = d[d["month"] < test_month], d[d["month"] == test_month]
        gtr, gte = rb.race_codes(train["race"]), rb.race_codes(test["race"])
        model = rb.fit_closing_model(rb.closing_inputs(train), train["bsp"], gtr)
        qhat = model.predict(rb.closing_inputs(test), gte)
        print(f"\n===== test month {test_month}: fitted on {train['race'].nunique():,} races "
              f"({', '.join(sorted(train['month'].unique()))}), scored on {test['race'].nunique():,} =====")
        print("  closing model weights: " + ", ".join(f"{k} {v:+.3f}" for k, v in zip(rb.FEATURES, model.beta)))
        print(f"  BSP book (median race sum of 1/BSP) {model.book:.3f}; spread of the miss by volume band: "
              + ", ".join(f"{s:.3f}" for s in model.band_sigma))
        print("  nearness to the BSP book:   mean |log q - log q_close|   KL per race")
        for k, (mae, kl) in closing_accuracy(test, qhat, gte).items():
            print(f"    {k:18s} {mae:10.4f} {kl:22.4f}")
        rng = np.random.default_rng(int(test_month.replace("-", "")))
        for race, r in test.groupby("race", sort=False):
            r = r.reset_index(drop=True)
            out = race_positions(r, model, rng)
            for name, (bs, ls, lay) in out["pos"].items():
                row = settle(r, name, bs, ls, lay, out["edges"])
                if row:
                    rows.append(row)
                    if name in ("dutch", "dutch_10", "close_back", "close_both", "hold_back", "hold_both"):
                        books.append({"strategy": name, "race": race, "n_back": row["n_back"], "n_lay": row["n_lay"]})
            worse = race_positions(r, model, rng, lay_worse=0.02)
            for name in ("single_lay", "close_both", "hold_both"):
                bs, ls, lay = worse["pos"][name]
                row = settle(r, name, bs, ls, lay, worse["edges"])
                if row:
                    rows_worse.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "race_book_rows.csv", index=False)
    summarise(res, "===== February-March, every strategy (per unit of money at risk; race bootstrap) =====")
    for mth, s in res.groupby("month"):
        summarise(s, f"----- {mth} -----")
    summarise(pd.DataFrame(rows_worse), "===== the lay strategies with the lay price 2% worse =====")
    print("\n===== paired against the pre-registered rule (CLV per unit at risk) =====")
    for a in ("single_back", "close_back", "close_both", "hold_back", "dutch"):
        print(paired(res, a, "rule22"))
    b = pd.DataFrame(books)
    print("\n===== how many runners each book holds =====")
    for name, s in b.groupby("strategy", sort=False):
        k = s["n_back"] + s["n_lay"]
        bands = pd.cut(k, [0, 1, 2, 4, 7, 99], labels=["1", "2", "3-4", "5-7", "8+"]).value_counts().sort_index()
        print(f"  {name:12s} " + "  ".join(f"{lab}: {v}" for lab, v in bands.items())
              + f"   (backs per race {s['n_back'].mean():.2f}, lays {s['n_lay'].mean():.2f})")
    print("\n===== level or small-underlay horses the books took (expected edge between -3% and 0) =====")
    for name, s in res[res["strategy"].isin(["dutch", "close_back", "close_both", "hold_back", "hold_both"])].groupby("strategy", sort=False):
        k = s["n_level_or_under"].sum()
        print(f"  {name:12s} {k:5d} such positions in {(s['n_level_or_under'] > 0).sum():5d} races; "
              f"their CLV {s['clv_level_or_under'].sum():+.2f} units against the books' {s['clv'].sum():+.2f}")


if __name__ == "__main__":
    main()
