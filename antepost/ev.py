"""The ante-post EV and the entry rule (docs/antepost_plan.md s.6; ledger antepost-ev-1119).

At each mark from 60 to 3 days out, the expected value of a GBP1 back on each listed horse, from the turn-up model
(antepost/turnup.py, the chances of ran / lost / void) and the price model (antepost/price.py, the race-day SP of a
horse that runs):

    EV = P(ran) * net(A_fill * E[1 / BSP | ran]) - P(lost)        (void: the stake back, 0)

where A_fill is the price a back would have been matched at (the last traded price less half the median spread of
its price band in the recorded books: reports/antepost_data.md s.5), E[1 / BSP] is taken lognormal from the price
model's forecast with its mean error and spread by mark on Jul-Dec 2025, and net() takes 2% commission off a winning
green (Betfair nets the ante-post and race-day markets for commission: s.0).

The rule: back GBP1 at the first mark (far to near) where a horse's EV reaches the threshold, one back per horse per
race, held to race day; a runner is laid at the SP for its winnings, a non-runner loses the stake, a horse balloted
out is void. The threshold is chosen on Jul-Dec 2025 with models fitted to 30 Jun 2025, then the rule is scored once
on 2026 with the models refit to 31 Dec 2025.

    python -m antepost.ev --panel turnup_panel.parquet --books recorded_books.parquet --out out/antepost_ev
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from antepost import price as pr
from antepost import turnup as tu

log = logging.getLogger(__name__)
RULE_MARKS = ["d60", "d45", "d30", "d21", "d14", "d7", "d3"]     # where both models passed their tests
THRESHOLDS = [0.0, 0.05, 0.10, 0.20, 0.30]
COMMISSION = 0.02
MIN_VALID_BETS = 30
PRIOR_TRIALS = 368      # python -m desk.trials --count, horse_racing, 19 Nov 2026 before this registration


def spread_by_band(books: pd.DataFrame) -> dict[str, float]:
    """The median relative spread (best lay / best back - 1) of active runners in open markets with a real lay, by
    price band of the best back (the recorded books from 7 Oct 2026)."""
    b = books.copy()
    for c in ("back1", "lay1"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b[(b["status"] == "OPEN") & (b["runner_status"] == "ACTIVE") & b["back1"].notna() & b["lay1"].notna()
          & (b["lay1"] < 1000)]
    rel = b["lay1"] / b["back1"] - 1
    idx = np.clip(np.searchsorted(tu.BANDS, b["back1"].to_numpy(), side="left") - 1, 0, len(tu.BANDS) - 2)
    return {f"b{i}": round(float(rel[idx == i].median()), 4) for i in range(len(tu.BANDS) - 1) if (idx == i).any()}


def fill_price(log_ltp: np.ndarray, spreads: dict[str, float]) -> np.ndarray:
    """The last traded price less half its band's median spread: what a back would have been matched at."""
    p = np.exp(log_ltp)
    idx = np.clip(np.searchsorted(tu.BANDS, p, side="left") - 1, 0, len(tu.BANDS) - 2)
    half = np.array([spreads.get(f"b{i}", np.nan) for i in idx]) / 2
    return p / (1 + half)


def net(x):
    """Return on GBP1 from a green of x per GBP1 staked: commission off a win only."""
    x = np.asarray(x, float)
    return np.where(x > 0, x * (1 - COMMISSION), x)


def models(fit: pd.DataFrame, stop: pd.DataFrame, refit: bool):
    """The turn-up and price models: fitted on ``fit`` (early-stopped on ``stop``), or refit on both with those
    rounds; with the price model's mean error and spread by mark on ``stop`` from the model fitted on ``fit``."""
    turn, t_rounds = tu.fit_gbm(fit, stop)
    pfit, pstop = pr.price_rows(fit), pr.price_rows(stop)
    pm, p_rounds = pr.fit_price_gbm(pfit, pstop)
    pm_fit, _ = pr.fit_price_gbm(pfit, None, p_rounds)
    res = pstop["z"].to_numpy() - pm_fit.predict(pstop[tu.FEATURES])
    err = {m: (float(res[(pstop["mark"] == m).to_numpy()].mean()), float(res[(pstop["mark"] == m).to_numpy()].std()))
           for m in tu.ORDER if (pstop["mark"] == m).any()}
    if refit:
        both = pd.concat([fit, stop])
        turn, _ = tu.fit_gbm(both, None, t_rounds)
        pm, _ = pr.fit_price_gbm(pr.price_rows(both), None, p_rounds)
    else:
        pm = pm_fit
    return turn, pm, err


def score(df: pd.DataFrame, turn, pm, err: dict, spreads: dict) -> pd.DataFrame:
    """Every listed, priced horse at the rule's marks with its EV and what a back would have returned."""
    d = df[df["mark"].isin(RULE_MARKS) & (df["has_price"] == 1)].copy()
    p = turn.predict(d[tu.FEATURES])
    d["p_lost"], d["p_ran"], d["p_void"] = p[:, 0], p[:, 1], p[:, 2]
    mu = pm.predict(d[tu.FEATURES]) + d["mark"].map({m: e[0] for m, e in err.items()}).fillna(0).to_numpy()
    sd = d["mark"].map({m: e[1] for m, e in err.items()}).fillna(0.7).to_numpy()
    d["a_fill"] = fill_price(d["log_ltp"].to_numpy(), spreads)
    d["e_ratio"] = d["a_fill"] * np.exp(-mu + sd ** 2 / 2)              # E[A_fill / BSP | ran], lognormal
    d["ev"] = d["p_ran"] * net(d["e_ratio"] - 1) - d["p_lost"]
    bsp = pd.to_numeric(d["bsp"], errors="coerce")
    d["ret"] = np.select([d["y"] == "ran", d["y"] == "void"], [net(d["a_fill"] / bsp - 1), 0.0], default=-1.0)
    return d


def bets(d: pd.DataFrame, theta: float) -> pd.DataFrame:
    """The first mark, far to near, at which each horse's EV reaches theta: one back per horse per race."""
    hit = d[d["ev"] >= theta].copy()
    hit["order"] = hit["mark"].map({m: i for i, m in enumerate(tu.ORDER)})
    return hit.sort_values("order").drop_duplicates(["market_id", "selection_id"])


def summary(b: pd.DataFrame, n_trials: int | None = None, trial_sharpes=None) -> dict:
    out = {"bets": int(len(b)), "races": int(b["market_id"].nunique()) if len(b) else 0}
    if b.empty:
        return out
    out.update({"roi": round(float(b["ret"].mean()), 4), "pred_ev": round(float(b["ev"].mean()), 4),
                "outcomes": b["y"].value_counts().to_dict(),
                "interval_90": tu.race_bootstrap(b["ret"].to_numpy(), b["market_id"].to_numpy()),
                "by_mark": {m: {"bets": int((g := b[b["mark"] == m]).shape[0]), "roi": round(float(g["ret"].mean()), 4)}
                            for m in RULE_MARKS if (b["mark"] == m).any()},
                "by_band": {k: {"bets": int(len(g)), "roi": round(float(g["ret"].mean()), 4)}
                            for k, g in b.groupby(tu.band_of(b).str.split("|").str[1])}})
    if n_trials:
        from research.overfit import dsr_check
        per_race = b.groupby("market_id")["ret"].sum().to_numpy()
        out["dsr"] = dsr_check(per_race, n_trials, trial_sharpes)
    return out


def sharpe(b: pd.DataFrame) -> float:
    r = b.groupby("market_id")["ret"].sum().to_numpy() if len(b) else np.array([])
    return float(r.mean() / r.std(ddof=1)) if len(r) > 2 and r.std(ddof=1) > 0 else float("nan")


def run(df: pd.DataFrame, books: pd.DataFrame) -> dict:
    tr, va, te = pr.splits(df)
    spreads = spread_by_band(books)
    rep = {"spread_by_band": spreads, "thresholds": THRESHOLDS, "rule_marks": RULE_MARKS,
           "n_trials": PRIOR_TRIALS + len(THRESHOLDS)}
    if tr.empty or va.empty or te.empty:
        return dict(rep, skipped="a split is empty")
    # choose the threshold on Jul-Dec 2025, models fitted to 30 Jun 2025
    turn, pm, err = models(tr, va, refit=False)
    dv = score(va, turn, pm, err, spreads)
    rep["valid"] = {str(t): summary(bets(dv, t)) for t in THRESHOLDS}
    ok = [t for t in THRESHOLDS if rep["valid"][str(t)]["bets"] >= MIN_VALID_BETS]
    theta = max(ok, key=lambda t: rep["valid"][str(t)]["roi"]) if ok else min(THRESHOLDS)
    rep["theta"] = theta
    # score once on 2026, models refit to 31 Dec 2025
    turn, pm, err = models(tr, va, refit=True)
    rep["price_error_by_mark_valid"] = {m: [round(a, 4), round(s, 4)] for m, (a, s) in err.items()}
    dt = score(te, turn, pm, err, spreads)
    tb = {t: bets(dt, t) for t in THRESHOLDS}
    trial_sharpes = [sharpe(b) for b in tb.values()]
    rep["test"] = summary(tb[theta], rep["n_trials"], trial_sharpes)
    rep["test_all_thresholds_for_the_record"] = {str(t): {k: v for k, v in summary(b).items()
                                                          if k in ("bets", "races", "roi", "pred_ev", "interval_90")}
                                                 for t, b in tb.items()}
    rep["test_all_horses"] = {"rows": int(len(dt)), "mean_ev": round(float(dt["ev"].mean()), 4),
                              "mean_ret_first_mark": round(float(bets(dt, -9.0)["ret"].mean()), 4)}
    return rep


def reading(rep: dict) -> dict:
    """The registered reading (ledger antepost-ev-1119), applied as written."""
    t = rep.get("test") or {}
    if not t.get("bets"):
        return {"verdict": "FAIL", "why": "no bets on the test" if "test" in rep else rep.get("skipped")}
    lo = t["interval_90"][0]
    dsr = (t.get("dsr") or {}).get("passed", False)
    ok = lo > 0 and dsr and t["bets"] >= 100
    return {"verdict": "PASS" if ok else "FAIL", "theta": rep["theta"], "bets": t["bets"], "roi": t["roi"],
            "interval_90": t["interval_90"], "dsr": (t.get("dsr") or {}).get("dsr"), "dsr_passed": dsr,
            "at_least_100_bets": t["bets"] >= 100}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--books", required=True)
    ap.add_argument("--out", default="out/antepost_ev")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    rep.update(run(pd.read_parquet(a.panel), pd.read_parquet(a.books)))
    rep["reading"] = reading(rep)
    (out / "antepost_ev.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
