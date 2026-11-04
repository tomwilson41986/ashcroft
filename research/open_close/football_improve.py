"""Improving the football move model on Betfair: rank selection, a cost filter (meta-label), an over/under model.

racing² football research, 4 Nov 2026 (the owner: "work on the improvements"; Betfair only). The base is the
stacked team-feature move model (football_stack_trial, arm team_all, ridge + LightGBM), fitted walk-forward on the
moves of the seasons before each test season and traded on Betfair's own prices (takeable books, back early and lay at
the close or the reverse, greened, 2% commission on each market's net). Three ideas, each scored the same way:

1. **Rank selection.** Take the strongest K forecasts a collection week, or a match day (the owner, 4 Nov: "anything
   daily? top 5, 10, 20, 50 per day"), not every forecast past a fixed threshold, so the number of bets does not swing
   with the model's scale. Also by day: the K biggest gaps between the model's price and Betfair's early price,
   traded toward the model. K is chosen on 2024/25 and read on 2025/26.
2. **A cost filter (meta-label).** A second model, fitted on 2024/25's candidate trades, predicts whether a trade
   clears its cost from what is known at entry (the forecast's size and direction, the price, the book's margin, the
   league, home or away). Read on 2025/26.
3. **An over/under 2.5 move model.** The same stack on the over chance's move, with a goals block (the Dixon-Coles
   goals model against the early price; market and model goal rates), traded on Betfair's over/under prices.

Every trade is costed two ways: a tick off the fair price at each end, and at a book symmetric about fair. 2025/26
was read by earlier pre-registered rules: development evidence. Every variant tried is in the output, to be read
against the count of forks.

    python -m research.open_close.football_improve --cache <lab cache> --out-json reports/football_runs/improve.json
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

COMMISSION = 0.02
TEST_SEASONS = (2024, 2025)          # 2025 = 2025/26 to date (2026 rows join it)
THRESHOLDS = (0.03, 0.04, 0.05)
WEEKLY_K = (5, 10, 20, 40)
DAILY_K = (5, 10, 20, 50)
TAKEABLE = ("tight <1.02", "medium 1.02-1.05")


# --------------------------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------------------------

def load(cache: str) -> tuple[pd.DataFrame, list[str]]:
    from research.open_close import football_stack_trial as st
    d, cols = st._matrix(cache)
    d = d.copy()
    d["test_season"] = np.where(d["season"] >= 2025, 2025, d["season"])
    d["week"] = pd.to_datetime(d["date"]).dt.to_period("W-THU").astype(str)      # a Friday-to-Thursday week
    # the goals block: the model's over chance against the early price's, and the goal rates behind them
    qo = d["qo_o"].clip(1e-4, 1 - 1e-4)
    po = d["p_o25"].clip(1e-4, 1 - 1e-4)
    d["ou_dis_open"] = np.log(po / (1 - po)) - np.log(qo / (1 - qo))
    d["ou_market_goals"] = d["molam_h"] + d["molam_a"]
    d["ou_model_goals"] = d["lam_h"] + d["lam_a"]
    d["ou_goals_gap"] = d["ou_model_goals"] - d["ou_market_goals"]
    d["y_o"] = np.log(d["qc_o"] / d["qo_o"])
    d["same_book_ou"] = (d["open_ou_src"].astype(str) == "P") & (d["close_ou_src"].astype(str) == "PC")
    return d, cols


OU_BLOCK = ["ou_dis_open", "ou_market_goals", "ou_model_goals", "ou_goals_gap"]


def walk_forward(d: pd.DataFrame, cols: list[str], target: str, same: str, season: int) -> np.ndarray:
    """The stack fitted on every same-book move of the seasons before ``season``, forecast for the test season's
    rows (2025 covers 2025/26 and later)."""
    from football import lab
    from research.open_close.football_stack_trial import TRAIN_FROM
    tr = d[d[same] & (d["season"] >= TRAIN_FROM) & (d["season"] < season) & d[target].notna()]
    te = d[d["test_season"] == season]
    return lab.MODELS["stack"](tr, te, cols, tr[target].to_numpy(float))


# --------------------------------------------------------------------------------------------------------------------
# candidate trades
# --------------------------------------------------------------------------------------------------------------------

def _sym(raw: np.ndarray, fair: np.ndarray) -> np.ndarray:
    from trading.exchange import tick_up
    return np.array([tick_up(v - 1e-9) if np.isfinite(v) else np.nan for v in fair ** 2 / raw])


def candidates(te: pd.DataFrame, preds: dict[str, np.ndarray], market: str) -> pd.DataFrame:
    """One row a selection with a forecast: the trade its forecast points to (back first when the chance is
    forecast up), its profit both ways, and what is known at entry."""
    from research.open_close.backtest import _with_spread
    from research.open_close.football_exchange import width_label
    if market == "MO":
        legs = {"H": ("bfe_open_h", "bfe_close_h"), "A": ("bfe_open_a", "bfe_close_a")}
        model_p = {"H": te["p_h"].to_numpy(float), "A": te["p_a"].to_numpy(float)}
        o_cols = ["bfe_open_h", "bfe_open_d", "bfe_open_a"]
        c_cols = ["bfe_close_h", "bfe_close_d", "bfe_close_a"]
    else:
        legs = {"O": ("bfe_open_o25", "bfe_close_o25"), "U": ("bfe_open_u25", "bfe_close_u25")}
        model_p = {"O": te["p_o25"].to_numpy(float), "U": 1 - te["p_o25"].to_numpy(float)}
        o_cols, c_cols = ["bfe_open_o25", "bfe_open_u25"], ["bfe_close_o25", "bfe_close_u25"]
    ok = te[o_cols + c_cols].gt(1).all(axis=1).to_numpy()
    over_o = (1 / te[o_cols]).sum(axis=1).to_numpy(float)
    over_c = (1 / te[c_cols]).sum(axis=1).to_numpy(float)
    rows = []
    for leg, (oc, cc) in legs.items():
        pred = preds[leg]
        ob, cb = te[oc].to_numpy(float), te[cc].to_numpy(float)
        of, cf = ob * over_o, cb * over_c                         # fair prices
        side = np.where(pred > 0, 1, -1)                          # chance up: price down: back first
        # each direction's profit, so any rule can pick its side: back first (b) or lay first (l)
        sym_b = ob / _sym(cb, cf) - 1
        sym_l = 1 - _sym(ob, of) / cb
        ob_t, cb_t = _with_spread(of, cf, np.ones(len(of)))
        ol_t, cl_t = _with_spread(of, cf, -np.ones(len(of)))
        tick_b, tick_l = ob_t / cb_t - 1, 1 - ol_t / cl_t
        p_sym = np.where(side > 0, sym_b, sym_l)
        p_tick = np.where(side > 0, tick_b, tick_l)
        gap = np.log(np.clip(model_p[leg], 1e-6, 1)) - np.log(1 / of)      # the model's chance against Betfair's fair
        rows.append(pd.DataFrame({
            "market_id": (te["match_id"] + ":" + market).to_numpy(), "match_id": te["match_id"].to_numpy(),
            "day": te["match_date"].astype(str).to_numpy(), "week": te["week"].to_numpy(),
            "season": te["test_season"].to_numpy(), "league": te["fd_code"].astype(str).to_numpy(),
            "leg": leg, "side": side, "pred": pred, "apred": np.abs(pred), "price": ob, "over": over_o,
            "width": width_label(over_o), "profit_sym": p_sym, "profit_tick": p_tick,
            "sym_b": sym_b, "sym_l": sym_l, "tick_b": tick_b, "tick_l": tick_l, "gap": gap,
            "move_dir": np.sign(np.log(cf / of)), "right_way": np.sign(np.log(cf / of)) == -side, "ok": ok}))
    c = pd.concat(rows, ignore_index=True)
    return c[c["ok"] & np.isin(c["width"], TAKEABLE) & np.isfinite(c["profit_sym"])].drop(columns="ok")


def scored(c: pd.DataFrame, n_boot: int = 1000) -> dict:
    from research.open_close.backtest import summarise
    if c.empty:
        return {"bets": 0}
    out = {}
    for cost in ("tick", "sym"):
        t = c.assign(profit=c[f"profit_{cost}"], stake=1.0)
        s = summarise(t, commission=COMMISSION, n_boot=n_boot)
        out[cost] = {k: (round(float(s[k]), 5) if s.get(k) is not None and np.isfinite(s[k]) else None)
                     for k in ("roi", "roi_lo90", "roi_hi90")}
    out["bets"] = int(len(c))
    out["right_way"] = round(float(c["right_way"].mean()), 4)
    return out


# --------------------------------------------------------------------------------------------------------------------
# selection rules
# --------------------------------------------------------------------------------------------------------------------

def by_threshold(c: pd.DataFrame, th: float) -> pd.DataFrame:
    return c[c["apred"] > th]


def by_rank(c: pd.DataFrame, k: int) -> pd.DataFrame:
    """The k strongest forecasts of each collection week."""
    return c.sort_values("apred", ascending=False).groupby("week", sort=False).head(k)


def by_day(c: pd.DataFrame, k: int) -> pd.DataFrame:
    """The k strongest forecast moves of each match day."""
    return c.sort_values("apred", ascending=False).groupby("day", sort=False).head(k)


def by_day_gap(c: pd.DataFrame, k: int) -> pd.DataFrame:
    """The k biggest gaps of each match day between the model's price and Betfair's early price, traded toward
    the model: back first where the model's chance is higher (the price should shorten), lay first where lower."""
    g = c.assign(agap=c["gap"].abs()).sort_values("agap", ascending=False).groupby("day", sort=False).head(k).copy()
    side = np.where(g["gap"] > 0, 1, -1)
    g["side"] = side
    g["profit_sym"] = np.where(side > 0, g["sym_b"], g["sym_l"])
    g["profit_tick"] = np.where(side > 0, g["tick_b"], g["tick_l"])
    g["right_way"] = g["move_dir"] == -side
    return g


META_FEATURES = ["apred", "side", "log_price", "over", "is_home_or_over", "league_code"]


def meta_frame(c: pd.DataFrame, leagues: list[str]) -> pd.DataFrame:
    x = pd.DataFrame({"apred": c["apred"], "side": c["side"], "log_price": np.log(c["price"]), "over": c["over"],
                      "is_home_or_over": c["leg"].isin(["H", "O"]).astype(int)})
    x["league_code"] = pd.Categorical(c["league"], categories=leagues).codes
    return x


def meta_filter(train: pd.DataFrame, test: pd.DataFrame, min_apred: float = 0.02) -> tuple[np.ndarray, dict]:
    """A small LightGBM classifier: does the trade clear its cost (profit at the symmetric book after commission
    above zero)? Fitted on the training season's candidates past ``min_apred``; returns the test candidates' scores."""
    import lightgbm as lgb
    tr = train[train["apred"] > min_apred]
    leagues = sorted(set(train["league"]) | set(test["league"]))
    y = (tr["profit_sym"] * np.where(tr["profit_sym"] > 0, 1 - COMMISSION, 1) > 0).astype(int)
    g = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    g.fit(meta_frame(tr, leagues), y, categorical_feature=["league_code"])
    imp = dict(zip(META_FEATURES, map(int, g.feature_importances_)))
    return g.predict_proba(meta_frame(test, leagues))[:, 1], {"train_rows": int(len(tr)), "base_rate": round(float(y.mean()), 4),
                                                               "importance": imp}


# --------------------------------------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------------------------------------

def run(cache: str, n_boot: int = 1000) -> dict:
    d, cols = load(cache)
    out = {"arm": "team_all", "note": "development evidence; 2025 = 2025/26 to date, read by earlier rules"}
    cands = {}
    for s in TEST_SEASONS:
        te = d[d["test_season"] == s]
        ph = walk_forward(d, cols, "y_h", "same_book", s)
        pa = walk_forward(d, cols, "y_a", "same_book", s)
        cands[("MO", s)] = candidates(te, {"H": ph, "A": pa}, "MO")
        po = walk_forward(d, cols + OU_BLOCK, "y_o", "same_book_ou", s)
        cands[("OU", s)] = candidates(te, {"O": po, "U": -po}, "OU")
        po0 = walk_forward(d, cols, "y_o", "same_book_ou", s)           # the over/under model without the block
        cands[("OU_noblock", s)] = candidates(te, {"O": po0, "U": -po0}, "OU")
        log.info("season %d: %s", s, {k[0]: len(v) for k, v in cands.items() if k[1] == s})
    for mk in ("MO", "OU", "OU_noblock"):
        res = {}
        for s in TEST_SEASONS:
            c = cands[(mk, s)]
            res[str(s)] = {"candidates": int(len(c)),
                           "threshold": {str(t): scored(by_threshold(c, t), n_boot) for t in THRESHOLDS},
                           "weekly_top_k": {str(k): scored(by_rank(c, k), n_boot) for k in WEEKLY_K},
                           "daily_top_k_forecast": {str(k): scored(by_day(c, k), n_boot) for k in DAILY_K},
                           "daily_top_k_model_gap": {str(k): scored(by_day_gap(c, k), n_boot) for k in DAILY_K},
                           "match_days": int(c["day"].nunique())}
        out[mk] = res
    # the cost filter: fitted on 2024/25's candidates, read on 2025/26 at the bet counts of the threshold rule
    for mk in ("MO", "OU"):
        tr, te = cands[(mk, 2024)], cands[(mk, 2025)].copy()
        score, info = meta_filter(tr, te)
        te["meta"] = score
        res = {"fit": info}
        for t in THRESHOLDS:
            n = int((te["apred"] > t).sum())
            pool = te[te["apred"] > 0.02].sort_values("meta", ascending=False)
            res[f"same_count_as_threshold_{t}"] = {"meta_top_n": scored(pool.head(n), n_boot),
                                                    "threshold": scored(by_threshold(te, t), n_boot)}
            both = te[(te["apred"] > t) & (te["meta"] > 0.5)]
            res[f"threshold_{t}_and_meta_above_half"] = scored(both, n_boot)
        out[f"meta_{mk}"] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run(a.cache, a.boot)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    log.info("written %s", a.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
