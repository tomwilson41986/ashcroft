"""The team-feature forecast of the early-to-close move, traded on Betfair's own prices (reports/football_team_features.md).

The forecaster: least squares of each side's move from the open to the close (the log of the close's de-vigged chance
over the open's, home and away separately) on every team-feature family (``football.feature_eval.FAMILIES["all"]``),
fitted on Pinnacle-to-Pinnacle matches only (football-data's PS open and PSC close) of the seasons before the scored
one: 2024/25 by a fit on 2013/14-2023/24; the holdout (from ``HOLDOUT_FROM``) by a fit on 2013/14-2024/25. Pinnacle's
early prices are stale from 23 Jul 2025 (football-data data.php), so no Pinnacle row after 2024/25 trains anything.

The trade, on football-data's Betfair Exchange prices (BFE early, BFEC close, from 2024/25): GBP1 a selection on the
home and the away side (never the draw). The forecast in price terms is minus the forecast in chance terms. Shorter
by more than the threshold: back at the early price, lay at the close; longer: lay then back. Greened. Headline: a tick
worse at each end on the fair prices (each Betfair book scaled to one at both ends), 2% commission on each market's
net (``research.open_close.backtest``), with the as-quoted back side beside it (back at the recorded early back price,
lay a tick over the recorded closing back). Every figure by the width of the Betfair early book (tight under 1.02,
medium to 1.05, wide), because the exchange study found the apparent edge only in wide books, where the quote is stale
(``reports/open_close/model_football_exchange.md``).

    python -m research.open_close.football_team_move --features <team_features.parquet> --develop --out <dev.json>
    python -m research.open_close.football_team_move --features <...> --holdout --prereg <file.md> --out <holdout.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from football.feature_eval import FAMILIES, TRAIN_FROM, prepare
from .backtest import by_group, hedged_trades, summarise
from .features import price_band
from .football_exchange import WIDTH_LABELS, read_prereg, width_label

DEV_SEASON = 2024
HOLDOUT_FROM = 2025
THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)
SIDES = {"H": ("h", "qo_h", "qc_h"), "A": ("a", "qo_a", "qc_a")}


def forecaster(train: pd.DataFrame, feats: list[str]) -> dict:
    """Least squares of each side's chance move on the standardised features, Pinnacle-to-Pinnacle rows only."""
    t = train[train["same_book"] & (train["open_1x2_src"] == "PS")]
    X = t[feats].to_numpy(float)
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    Z = np.c_[np.ones(len(t)), np.nan_to_num((X - mu) / sd)]
    out = {"mu": mu, "sd": sd, "rows": int(len(t)), "seasons": [int(t["season"].min()), int(t["season"].max())]}
    for s, (_k, qo, qc) in SIDES.items():
        out[s] = np.linalg.lstsq(Z, np.log(t[qc] / t[qo]).to_numpy(float), rcond=None)[0]
    return out


def predict(fit: dict, d: pd.DataFrame, feats: list[str], side: str) -> np.ndarray:
    Z = np.c_[np.ones(len(d)), np.nan_to_num((d[feats].to_numpy(float) - fit["mu"]) / fit["sd"])]
    return Z @ fit[side]


def betfair_rows(d: pd.DataFrame, fits: dict, feats: list[str]) -> pd.DataFrame:
    """One row a (match, home or away) with Betfair's early and closing prices, raw and fair, the book's width, and
    the forecast of the price's move (log close / open, fair) from the fit for the match's season."""
    b = d[d[["bfe_open_h", "bfe_open_d", "bfe_open_a", "bfe_close_h", "bfe_close_d", "bfe_close_a"]].gt(1).all(axis=1)]
    over_o = 1 / b["bfe_open_h"] + 1 / b["bfe_open_d"] + 1 / b["bfe_open_a"]
    over_c = 1 / b["bfe_close_h"] + 1 / b["bfe_close_d"] + 1 / b["bfe_close_a"]
    parts = []
    for s, (k, _qo, _qc) in SIDES.items():
        p = pd.DataFrame({"match_id": b["match_id"].to_numpy(), "market_id": (b["match_id"] + ":MO").to_numpy(),
                          "selection_id": s, "season": b["season"].to_numpy(), "fd_code": b["fd_code"].to_numpy(),
                          "day": b["match_date"].astype(str).to_numpy(), "month": b["match_date"].astype(str).str[:7].to_numpy(),
                          "sport": "football", "country": b["country"].to_numpy(), "market_type": "MATCH_ODDS",
                          "open_raw": b[f"bfe_open_{k}"].to_numpy(float), "close_raw": b[f"bfe_close_{k}"].to_numpy(float),
                          "bfe_over_open": over_o.to_numpy(float), "bfe_over_close": over_c.to_numpy(float)})
        p["open_price"] = p["open_raw"] * p["bfe_over_open"]
        p["close_price"] = p["close_raw"] * p["bfe_over_close"]
        p["y"] = np.log(p["close_price"] / p["open_price"])
        pred = np.full(len(b), np.nan)
        for season, fit in fits.items():
            m = (b["season"] == season).to_numpy()
            if m.any():
                pred[m] = -predict(fit, b[m], feats, s)          # chance up = price down
        p["pred"] = pred
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    out = out[out["pred"].notna()].reset_index(drop=True)
    out["book_width"] = width_label(out["bfe_over_open"])
    out["band"] = price_band(out["open_price"].to_numpy()).astype(str)
    return out


def quoted_back(r: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """The back side as quoted: back GBP1 at the recorded early back price, lay a tick over the recorded closing back."""
    from .backtest import tick_over
    t = hedged_trades(r, "pred", threshold)
    t = t[t["side"] > 0].copy()
    if t.empty:
        return t
    rows = t["row"].to_numpy()
    o = r["open_raw"].to_numpy(float)[rows]
    c = np.array([tick_over(x) for x in r["close_raw"].to_numpy(float)[rows]])
    t["fill_open"], t["fill_close"], t["profit"] = o, c, o / c - 1.0
    return t


def sweep(r: pd.DataFrame, widths=None, n_boot: int = 2000) -> list[dict]:
    rows = r if widths is None else r[r["book_width"].isin(widths)]
    out = []
    for th in THRESHOLDS:
        sp, mid = hedged_trades(rows, "pred", th, spread=True), hedged_trades(rows, "pred", th)
        q = quoted_back(rows, th)
        out.append({"threshold": th, "spread": summarise(sp, n_boot=n_boot), "mid": summarise(mid, n_boot=0),
                    "quoted_back": summarise(q, n_boot=0) if len(q) else {"bets": 0}})
    return out


def report(r: pd.DataFrame, threshold: float, widths: list[str], n_boot: int = 2000) -> dict:
    rows = r[r["book_width"].isin(widths)]
    sp = hedged_trades(rows, "pred", threshold, spread=True)
    clv = (np.where(sp["side"] > 0, sp["fill_open"] / sp["close_price"] - 1, 1 - sp["fill_open"] / sp["close_price"])
           if len(sp) else np.array([]))
    q = quoted_back(rows, threshold)
    return {"rows": int(len(rows)), "spread": summarise(sp, n_boot=n_boot), "mid": summarise(
        hedged_trades(rows, "pred", threshold), n_boot=n_boot), "clv_mean_spread": float(clv.mean()) if len(clv) else None,
        "quoted_back": summarise(q, n_boot=n_boot) if len(q) else {"bets": 0},
        "by_band": by_group(rows, "pred", threshold, "band", spread=True, n_boot=0).to_dict("index"),
        "by_month": by_group(rows, "pred", threshold, "month", spread=True, n_boot=0).to_dict("index"),
        "by_width": by_group(r, "pred", threshold, "book_width", spread=True, n_boot=500).to_dict("index")}


def fits_for(d: pd.DataFrame, feats: list[str], seasons: list[int], last_train: int | None = None) -> dict:
    """A fit for each scored season, on the Pinnacle rows of the seasons before it (or to ``last_train``)."""
    out = {}
    for s in seasons:
        hi = min(s - 1, last_train) if last_train is not None else s - 1
        out[s] = forecaster(d[(d["season"] >= TRAIN_FROM) & (d["season"] <= hi)], feats)
    return out


def develop(f: pd.DataFrame, n_boot: int = 2000) -> dict:
    d = prepare(f)
    feats = FAMILIES["all"]
    fits = fits_for(d, feats, [DEV_SEASON])
    r = betfair_rows(d[d["season"] == DEV_SEASON], fits, feats)
    res = {"season": DEV_SEASON, "fit": {s: {"rows": v["rows"], "seasons": v["seasons"]} for s, v in fits.items()},
           "rows": int(len(r)), "matches": int(r["match_id"].nunique()),
           "width_rows": r["book_width"].value_counts().to_dict(),
           "forecast_vs_move": {"corr": float(np.corrcoef(r["pred"], r["y"])[0, 1]),
                                "sign_right_top_decile": float(np.mean(np.sign(
                                    r.loc[r["pred"].abs() >= r["pred"].abs().quantile(0.9), "pred"]) == np.sign(
                                    r.loc[r["pred"].abs() >= r["pred"].abs().quantile(0.9), "y"])))},
           "sweep_all": sweep(r, n_boot=n_boot), "sweep_by_width": {w: sweep(r, [w], n_boot=500) for w in WIDTH_LABELS},
           "sweep_takeable": sweep(r, list(WIDTH_LABELS[:2]), n_boot=n_boot)}
    return res


def holdout(f: pd.DataFrame, prereg: str, n_boot: int = 2000) -> dict:
    rule, digest = read_prereg(prereg)
    d = prepare(f)
    feats = FAMILIES["all"]
    seasons = sorted(int(s) for s in d.loc[d["season"] >= HOLDOUT_FROM, "season"].unique())
    fits = fits_for(d, feats, seasons, last_train=DEV_SEASON)
    r = betfair_rows(d[d["season"] >= HOLDOUT_FROM], fits, feats)
    out = report(r, rule["threshold"], rule["book_width"], n_boot=n_boot)
    s, gt = out["spread"], rule["gate"]
    checks = {"bets": s.get("bets", 0) >= gt["min_bets"],
              "roi_lo90_above_0": (s.get("roi_lo90") if s.get("roi_lo90") == s.get("roi_lo90") else -1) > 0,
              "clv_above_0": (out.get("clv_mean_spread") or -1) > 0}
    return {"prereg": prereg, "prereg_sha256": digest, "rule": rule, "seasons": seasons,
            "fit": {s: {"rows": v["rows"], "seasons": v["seasons"]} for s, v in fits.items()},
            "rows": int(len(r)), "matches": int(r["match_id"].nunique()), "result": out,
            "gate": {"checks": checks, "passed": bool(all(checks.values()))},
            "sweep_for_the_record": sweep(r, rule["book_width"], n_boot=0)}


def _json(o) -> str:
    def default(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x) if np.isfinite(x) else None
        return str(x)
    return json.dumps(o, indent=1, default=default)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", required=True, help="the team features parquet (football.feature_eval --features-out)")
    ap.add_argument("--develop", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--prereg", default="reports/preregistration_football_team_move.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    f = pd.read_parquet(a.features)
    res = develop(f, a.n_boot) if a.develop else holdout(f, a.prereg, a.n_boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(_json(res) + "\n")
    print(_json(res)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
