"""Football on Betfair: forecast the move of Betfair's early price to its close from the books of the same moment,
and score it as the hedged exchange trade (reports/open_close/model_football_exchange.md).

The panel (``exchange_panel``): every MATCH_ODDS outcome and OVER_UNDER_25 side of a finished match with Betfair
Exchange's early and closing prices in football-data.co.uk's files (BFE* and BFEC*, from 2024/25), the open Betfair's
early price and the close Betfair's close (panels.football_panel(gold, "bfe")), with features from the early
collection only (football-data collects the first set "after market opening at times specified on my fixtures
page": weekend fixtures on Friday afternoons by 17:00 UK, midweek on Tuesdays by 13:00 UK; data.php, matches.php):

- Pinnacle's early price (PS* for 1X2, P>2.5 / P<2.5 for the goals; never PSC* or PC*, the closes) and its book;
- the market average and maximum at the early collection (Avg*, Max*) and Bet365's (B365*), with their books;
- the Betfair early book's overround (its width), before the margin is taken out;
- the log gap between Betfair's early fair price and each bookmaker's early fair price, both books normalised to
  one (``gap_ps`` = log(Betfair fair / Pinnacle fair): positive where Betfair is the longer, so the thesis says
  Betfair shortens and the trade backs);
- the Dixon-Coles probability (football.model.walk_forward: each match priced from the ratings fitted on the matches
  before its week only) and its gap to Betfair's fair price;
- league, top-five league, month of the year, day of the week, kick-off hour, the days from the collection to the
  kick-off implied by the collection rule, and the season's stage (the share of the season's matches played before).

Every price is scaled to a book of one at both ends before the target is taken (features.fair_prices), as section 5
of reports/open_close/model.md does. The raw early and closing prices are kept (``open_raw``, ``close_raw``) for the
as-quoted check (``quoted_back_trades``).

Models (``exchange_models``): no change; the training mean; the pre-registered gap rule (the forecast close is the
bookmaker's fair price: y_hat = -gap, nothing fitted); the gap-only least-squares line; LightGBM on Betfair's own
book only (section 5's arm); LightGBM on every feature except Pinnacle's; LightGBM on every feature.

    python -m research.open_close.football_exchange --root <sources> --build-panel --panel <panel.parquet>
    python -m research.open_close.football_exchange --panel <panel.parquet> --develop --out <dev.json>
    python -m research.open_close.football_exchange --panel <panel.parquet> --holdout \\
        --prereg reports/preregistration_football_exchange.md --out <holdout.json>

The development step never scores a month from ``HOLDOUT_FROM`` on; the holdout step scores only the rule the
pre-registration file names (its JSON block), and the other models beside it for the record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import THRESHOLDS, accuracy_table, by_group, hedged_trades, summarise, sweep
from .drift import GradientBoosting, MeanShift, NoChange, shuffle_test, walk_forward
from .features import prepare
from .panels import football_panel
from .run import choose_threshold

HOLDOUT_FROM = "2026-08"
PANEL_FROM = "2024-07-01"
MIN_TRAIN_MONTHS = 6
PIN_STALE_FROM = "2025-07-23"          # football-data.co.uk data.php: Pinnacle's odds systematically out of date since
TOP5 = ("E0", "D1", "I1", "SP1", "F1")
WIDTH_EDGES = (1.02, 1.05)             # the stale-quote split of the Betfair early book: tight, medium, wide
WIDTH_LABELS = ("tight <1.02", "medium 1.02-1.05", "wide >1.05")
#: football-data's column suffix of each leg, and each bookmaker's prefix (1X2, goals): notes.txt
LEG_SUFFIX = {("MATCH_ODDS", "H"): "H", ("MATCH_ODDS", "D"): "D", ("MATCH_ODDS", "A"): "A",
              ("OVER_UNDER_25", "Over 2.5"): ">2.5", ("OVER_UNDER_25", "Under 2.5"): "<2.5"}
BOOK_PREFIX = {"ps": ("PS", "P"), "avg": ("Avg", "Avg"), "max": ("Max", "Max"), "b365": ("B365", "B365")}

BASE_FEATURES = ["log_open", "p_share", "rank_open", "gap_fav", "bfe_over_open", "dow", "off_hour", "days_to_ko_est",
                 "month_of_year", "season_stage", "top5", "league_code"]
MARKET_FEATURES = ["gap_avg", "gap_max", "gap_b365", "avg_over", "max_over", "b365_over"]
DC_FEATURES = ["dc_gap", "dc_p"]
PIN_FEATURES = ["gap_ps", "ps_over"]


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------

def days_to_kickoff(match_date: pd.Series) -> pd.Series:
    """Days from football-data's early collection to the match day, from the collection rule (matches.php): Friday
    for Friday to Monday matches (0-3), Tuesday for Tuesday to Thursday (0-2). An estimate: no time stamp is kept."""
    dow = pd.to_datetime(match_date).dt.dayofweek
    return pd.Series(np.where(dow.isin([1, 2, 3]), dow - 1, (dow - 4) % 7), index=match_date.index).astype(float)


def season_stage(gold: pd.DataFrame) -> pd.Series:
    """The share of the competition-season's matches played before each match's day (by match_id)."""
    g = gold[["match_id", "competition_id", "season", "match_date"]].copy()
    g["t"] = pd.to_datetime(g["match_date"])
    total = g.groupby(["competition_id", "season"])["match_id"].transform("size")
    # matches on earlier days: the rank of the first match of the day, minus one
    before = g.groupby(["competition_id", "season"])["t"].rank(method="min") - 1
    return pd.Series((before / total).to_numpy(), index=g["match_id"].to_numpy())


def book_frame(silver_ids: pd.DataFrame) -> pd.DataFrame:
    """One row a (match_id, market_type, selection_id) with each bookmaker's early price (columns ps, avg, max,
    b365), from football-data's silver rows carrying the canonical match_id."""
    parts = []
    for (mt, sel), suf in LEG_SUFFIX.items():
        p = pd.DataFrame({"event_id": silver_ids["match_id"].to_numpy(), "market_type": mt, "selection_id": sel})
        for name, (pre_1x2, pre_ou) in BOOK_PREFIX.items():
            col = (pre_1x2 if mt == "MATCH_ODDS" else pre_ou) + suf
            v = pd.to_numeric(silver_ids[col], errors="coerce") if col in silver_ids.columns else np.nan
            p[name] = np.where(np.asarray(v, float) > 1.0, v, np.nan)
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["event_id", "market_type", "selection_id"], keep="first")


def _market_fair(d: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """A bookmaker's book (the sum of 1/price over the market, only where every leg is priced) and its fair prices."""
    inv = 1.0 / d[col]
    over = inv.groupby(d["market_id"]).transform(lambda s: s.sum() if s.notna().all() else np.nan)
    return over, d[col] * over


def add_exchange_features(panel: pd.DataFrame, books: pd.DataFrame, gold: pd.DataFrame,
                          dc: pd.DataFrame | None = None) -> pd.DataFrame:
    """The football panel with the same-moment features (raw prices; ``prepare`` normalises the books after)."""
    d = panel.merge(books, on=["event_id", "market_type", "selection_id"], how="left")
    meta = gold.drop_duplicates("match_id").set_index("match_id")
    d["fd_code"] = d["event_id"].map(meta["fd_code"]).astype(str)
    d["match_date"] = d["event_id"].map(meta["match_date"]).astype(str)
    d["open_raw"], d["close_raw"] = d["open_price"].astype(float), d["close_price"].astype(float)
    over, bfe_fair = _market_fair(d.assign(bfe=d["open_raw"]), "bfe")
    d["bfe_over_open"] = over
    d["bfe_over_close"] = (1.0 / d["close_raw"]).groupby(d["market_id"]).transform("sum")
    for name in BOOK_PREFIX:
        o, fair = _market_fair(d, name)
        d[f"{name}_over"] = o
        if name != "max":
            d[f"gap_{name}"] = np.log(bfe_fair / fair)
    d["gap_max"] = np.log(d["open_raw"] / d["max"])            # Betfair's early back against the best bookmaker price
    if dc is not None and len(dc):
        x = dc.drop_duplicates("match_id").set_index("match_id")
        p = pd.Series(np.nan, index=d.index)
        for sel, col, flip in (("H", "p_h", False), ("D", "p_d", False), ("A", "p_a", False),
                               ("Over 2.5", "p_o25", False), ("Under 2.5", "p_o25", True)):
            idx = d["selection_id"] == sel
            v = d.loc[idx, "event_id"].map(x[col]).astype(float)
            p.loc[idx] = (1.0 - v) if flip else v
        d["dc_p"] = p
        d["dc_gap"] = np.log(bfe_fair * d["dc_p"])             # log(Betfair fair / the model's fair price)
    else:
        d["dc_p"], d["dc_gap"] = np.nan, np.nan
    d["pin_period"] = np.where(d["ps"].isna() | d["ps_over"].isna(), "absent",
                               np.where(d["match_date"] < PIN_STALE_FROM, "fresh", "stale"))
    d["top5"] = d["fd_code"].isin(TOP5).astype(float)
    d["tier"] = np.where(d["top5"] > 0, "top five", "rest")
    d["league_code"] = pd.Categorical(d["fd_code"])
    d["days_to_ko_est"] = days_to_kickoff(d["match_date"])
    d["month_of_year"] = pd.to_datetime(d["match_date"]).dt.month.astype(float)
    d["season_stage"] = d["event_id"].map(season_stage(gold)).astype(float)
    d["book_width"] = width_label(d["bfe_over_open"])
    return d


def width_label(over) -> np.ndarray:
    o = np.asarray(over, float)
    return np.where(o < WIDTH_EDGES[0], WIDTH_LABELS[0], np.where(o <= WIDTH_EDGES[1], WIDTH_LABELS[1],
                                                                   WIDTH_LABELS[2]))


def exchange_panel(gold: pd.DataFrame, silver_ids: pd.DataFrame, dc: pd.DataFrame | None = None,
                   start: str = PANEL_FROM) -> tuple[pd.DataFrame, dict]:
    """The exchange panel and its provenance counts (rows built, dropped and why, duplicates)."""
    g = gold[gold["match_date"] >= start]
    base = football_panel(g, venue="bfe")
    prov = {"gold_matches_from_start": int(len(g)), "finished": int(g["status"].eq("FINISHED").sum()),
            "panel_rows": int(len(base)), "panel_markets": int(base["market_id"].nunique())}
    dup = base.duplicated(["market_id", "selection_id"]).sum()
    prov["duplicate_selection_rows_dropped"] = int(dup)
    base = base.drop_duplicates(["market_id", "selection_id"])
    d = add_exchange_features(base, book_frame(silver_ids), gold, dc)
    return d, prov


def silver_with_ids(silver: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    from football.data import _with_ids
    return _with_ids(silver, gold)


# ---------------------------------------------------------------------------
# the models
# ---------------------------------------------------------------------------

@dataclass
class GapRule:
    """The pre-registered simple rule: the close goes to the bookmaker's fair price, y_hat = -gap (nothing fitted;
    no forecast, so no trade, where the bookmaker has no book)."""

    col: str = "gap_ps"
    name: str = "gap_rule_ps"

    def fit(self, train: pd.DataFrame) -> "GapRule":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(-test[self.col].to_numpy(float), nan=0.0)


@dataclass
class GapOLS:
    """y = a + b gap by least squares on the training rows that carry the gap; no change where the gap is absent."""

    col: str = "gap_ps"
    name: str = "gap_ols_ps"
    coef: tuple = (0.0, 0.0)

    def fit(self, train: pd.DataFrame) -> "GapOLS":
        x, y = train[self.col].to_numpy(float), train["y"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 50 or np.std(x[ok]) < 1e-12:
            self.coef = (0.0, 0.0)
            return self
        b, a = np.polyfit(x[ok], y[ok], 1)
        self.coef = (float(a), float(b))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        x = test[self.col].to_numpy(float)
        return np.where(np.isfinite(x), self.coef[0] + self.coef[1] * np.nan_to_num(x), 0.0)


def exchange_models(n_jobs: int = 4) -> list:
    return [NoChange(), MeanShift(),
            GapRule("gap_ps", "gap_rule_ps"), GapOLS("gap_ps", "gap_ols_ps"),
            GapRule("gap_avg", "gap_rule_avg"), GapOLS("gap_avg", "gap_ols_avg"),
            GradientBoosting(name="gbm_base", features=list(BASE_FEATURES), n_jobs=n_jobs),
            GradientBoosting(name="gbm_nopin", features=BASE_FEATURES + MARKET_FEATURES + DC_FEATURES, n_jobs=n_jobs),
            GradientBoosting(name="gbm_all", features=BASE_FEATURES + MARKET_FEATURES + DC_FEATURES + PIN_FEATURES,
                             n_jobs=n_jobs)]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def r2_above_floor(acc: pd.DataFrame) -> dict:
    """Each model's R2 against no change, less the training mean's (the floor a permuted target still earns)."""
    floor = float(acc.loc["mean_shift", "r2_vs_no_change"]) if "mean_shift" in acc.index else 0.0
    return {k: float(v) - floor for k, v in acc["r2_vs_no_change"].items()}


def clv(trades: pd.DataFrame) -> pd.Series:
    """Each trade's value at the price taken if the close is fair, before commission and the closing spread: a back
    o / c - 1, a lay 1 - o / c, o the price filled at the open and c the fair close."""
    o, c = trades["fill_open"].to_numpy(float), trades["close_price"].to_numpy(float)
    return pd.Series(np.where(trades["side"] > 0, o / c - 1.0, 1.0 - o / c), index=trades.index)


def quoted_back_trades(scored: pd.DataFrame, pred_col: str, threshold: float) -> pd.DataFrame:
    """The back side as quoted: back GBP1 at Betfair's early price as football-data records it (``open_raw``; its
    books sum over one, as best backs do), lay at the close a tick over the recorded closing back (the best a lay can
    be). The lay side has no recorded lay price, so it is not scored this way."""
    from trading.exchange import tick_up
    t = hedged_trades(scored, pred_col, threshold)
    t = t[t["side"] > 0].copy()
    if t.empty:
        return t
    rows = t["row"].to_numpy()
    o = scored["open_raw"].to_numpy(float)[rows]
    c = np.array([tick_up(tick_up(p) + 1e-6) for p in scored["close_raw"].to_numpy(float)[rows]])
    t["fill_open"], t["fill_close"] = o, c
    t["profit"] = o / c - 1.0
    return t


GROUPS = ("book_width", "band", "tier", "pin_period", "fd_code", "month")


def score(scored: pd.DataFrame, models: list[str], thresholds: dict | None = None, n_boot: int = 2000) -> dict:
    """Accuracy, the threshold sweep (mid and a tick worse each end), each model's chosen threshold (the best lower
    90% bound a tick worse, among thresholds trading at least once a day) unless ``thresholds`` fixes them, and the
    breakdowns at that threshold, a tick worse each end."""
    pred_cols = [f"pred_{m}" for m in models]
    acc = accuracy_table(scored, pred_cols)
    out = {"rows": int(len(scored)), "markets": int(scored["market_id"].nunique()),
           "days": int(scored["day"].nunique()), "months": [str(scored["month"].min()), str(scored["month"].max())],
           "accuracy": acc.to_dict(orient="index"), "r2_above_mean_shift": r2_above_floor(acc), "models": {}}
    for m in models:
        if m in ("no_change", "mean_shift"):
            continue
        c = f"pred_{m}"
        sw = sweep(scored, c, n_boot=n_boot)
        t = thresholds[m] if thresholds and m in thresholds else choose_threshold(sw)
        e = {"sweep": sw.to_dict(orient="records"), "threshold": t, "by": {}}
        tr = hedged_trades(scored, c, t, spread=True)
        if len(tr):
            e["clv_mean_spread"] = float(clv(tr).mean())
        for key in GROUPS:
            e["by"][key] = by_group(scored, c, t, key, spread=True, n_boot=0 if key == "month" else 500
                                    ).to_dict(orient="index")
            e["by"][f"{key}_mid"] = by_group(scored, c, t, key, spread=False, n_boot=0).to_dict(orient="index")
        q = quoted_back_trades(scored, c, t)
        e["quoted_back"] = summarise(q, n_boot=n_boot) if len(q) else {"bets": 0}
        # the stale-quote test in full: the whole sweep inside each book width, and the back side as quoted there
        e["width_sweeps"] = {}
        for w in WIDTH_LABELS:
            sub = scored[scored["book_width"] == w]
            if len(sub) < 50:
                continue
            ws = sweep(sub, c, n_boot=500).to_dict(orient="records")
            for r in ws:
                if r.get("bets"):
                    qb = quoted_back_trades(sub, c, r["threshold"])
                    r["quoted_back"] = summarise(qb, n_boot=0) if len(qb) else {"bets": 0}
            e["width_sweeps"][w] = ws
        out["models"][m] = e
    return out


def rule_rows(d: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """The scored rows a pre-registered rule may trade: its market type, leagues and book widths."""
    keep = d["market_type"].isin(rule["market_types"])
    if rule.get("leagues", "all") == "top5":
        keep &= d["top5"] > 0
    elif rule.get("leagues", "all") == "rest":
        keep &= d["top5"] == 0
    if rule.get("book_width"):
        keep &= d["book_width"].isin(rule["book_width"])
    return d[keep]


def score_rule(scored: pd.DataFrame, rule: dict, n_boot: int = 2000) -> dict:
    """The pre-registered rule on the rows it may trade: GBP1 a selection, 2% on each market's net, a tick worse at
    each end (the headline), at the mid beside it, its mean CLV at the taken price, and the as-quoted back side."""
    rows = rule_rows(scored, rule)
    c = f"pred_{rule['model']}"
    sp = hedged_trades(rows, c, rule["threshold"], spread=True)
    mid = hedged_trades(rows, c, rule["threshold"])
    out = {"rule": rule, "rows": int(len(rows)), "spread": summarise(sp, n_boot=n_boot),
           "mid": summarise(mid, n_boot=n_boot), "clv_mean_spread": float(clv(sp).mean()) if len(sp) else None}
    q = quoted_back_trades(rows, c, rule["threshold"])
    out["quoted_back"] = summarise(q, n_boot=n_boot) if len(q) else {"bets": 0}
    for key in ("book_width", "band", "tier", "month"):
        out[f"by_{key}"] = by_group(rows, c, rule["threshold"], key, spread=True,
                                    n_boot=0 if key == "month" else 500).to_dict(orient="index")
    return out


def gate(result: dict, rule: dict) -> dict:
    """The pre-registered gate: at least N bets and N days, the 90% lower bound of the return a tick worse above
    zero, and the mean CLV at the taken price above zero."""
    s, gt = result["spread"], rule["gate"]
    checks = {"bets": s.get("bets", 0) >= gt["min_bets"], "days": s.get("days", 0) >= gt["min_days"],
              "roi_lo90_above_0": (s.get("roi_lo90") or -1) > 0, "clv_above_0": (result.get("clv_mean_spread") or -1) > 0}
    return {"checks": checks, "passed": bool(all(checks.values()))}


# ---------------------------------------------------------------------------
# the runs
# ---------------------------------------------------------------------------

def prepared(panel: pd.DataFrame) -> pd.DataFrame:
    d = prepare(panel, normalise_book=True)
    d["league_code"] = pd.Categorical(d["fd_code"].astype(str))
    return d


def develop(panel: pd.DataFrame, n_jobs: int = 4, n_boot: int = 2000, log=print,
            save_scored: str | None = None) -> dict:
    """Walk-forward by month over the development months (the panel's start to the month before HOLDOUT_FROM), each
    market type on its own; the leak test on permuted closes; the gap rule over every month (it fits nothing)."""
    t0 = time.time()
    d_all = prepared(panel)
    d = d_all[d_all["month"] < HOLDOUT_FROM]
    names = [m.name for m in exchange_models()]
    res = {"holdout_from": HOLDOUT_FROM, "min_train_months": MIN_TRAIN_MONTHS, "markets": {}}
    for mt, part in d.groupby("market_type"):
        log(f"[{mt}] {len(part):,} rows, {part['month'].min()} to {part['month'].max()}")
        sc = walk_forward(part, exchange_models(n_jobs), min_train_months=MIN_TRAIN_MONTHS, refit_every=1, log=log)
        if save_scored:
            sc.drop(columns=["league_code"]).to_parquet(f"{save_scored}.{mt}.parquet", index=False)
        r = {"development": score(sc, names, n_boot=n_boot)}
        for period in ("fresh", "stale", "absent"):
            sub = sc[sc["pin_period"] == period]
            if len(sub) > 100:
                acc = accuracy_table(sub, [f"pred_{m}" for m in names])
                r[f"development_pin_{period}"] = {
                    "rows": int(len(sub)), "months": [str(sub["month"].min()), str(sub["month"].max())],
                    "accuracy": acc.to_dict(orient="index"), "r2_above_mean_shift": r2_above_floor(acc),
                    "trade": {m: {str(t): summarise(hedged_trades(sub, f"pred_{m}", t, spread=True), n_boot=500)
                                  for t in (0.0, 0.05, 0.10)} for m in names if m not in ("no_change", "mean_shift")}}
        # the gap rule fits nothing: score it on every development month, the fit-only months too
        full = part.copy()
        for m in (GapRule("gap_ps", "gap_rule_ps"), GapRule("gap_avg", "gap_rule_avg")):
            full[f"pred_{m.name}"] = m.predict(full)
        r["gap_rule_all_months"] = {}
        for m in ("gap_rule_ps", "gap_rule_avg"):
            for period in ("fresh", "stale", "absent", "all"):
                sub = full if period == "all" else full[full["pin_period"] == period]
                if len(sub) < 100:
                    continue
                r["gap_rule_all_months"][f"{m}:{period}"] = {
                    "rows": int(len(sub)), "months": [str(sub["month"].min()), str(sub["month"].max())],
                    "accuracy": accuracy_table(sub, [f"pred_{m}"]).to_dict(orient="index"),
                    "sweep": sweep(sub, f"pred_{m}", n_boot=n_boot).to_dict(orient="records"),
                    "by_book_width": {str(t): by_group(sub, f"pred_{m}", t, "book_width", spread=True, n_boot=500)
                                      .to_dict(orient="index") for t in (0.0, 0.05, 0.10)},
                    "by_book_width_mid": {str(t): by_group(sub, f"pred_{m}", t, "book_width", n_boot=0)
                                          .to_dict(orient="index") for t in (0.0, 0.05, 0.10)}}
        sh = shuffle_test(part, exchange_models(n_jobs), min_train_months=MIN_TRAIN_MONTHS, refit_every=6)
        acc = accuracy_table(sh, [f"pred_{m}" for m in names])
        r["shuffle_test"] = {"refit_every": 6, "accuracy": acc.to_dict(orient="index"),
                             "r2_above_mean_shift": r2_above_floor(acc)}
        fit = [m for m in exchange_models(n_jobs) if m.name == "gbm_all"][0].fit(part)
        r["feature_gain_gbm_all"] = dict(sorted(fit.importance().items(), key=lambda kv: -kv[1]))
        res["markets"][mt] = r
    res["seconds"] = round(time.time() - t0, 1)
    return res


def read_prereg(path: str) -> tuple[dict, str]:
    """The rule in the pre-registration's JSON block, and the file's SHA-256 (the record of what was read)."""
    text = Path(path).read_text()
    m = re.search(r"```json\n(.*?)\n```", text, re.S)
    if not m:
        raise SystemExit(f"{path}: no ```json block holding the rule")
    return json.loads(m.group(1)), hashlib.sha256(text.encode()).hexdigest()


def holdout(panel: pd.DataFrame, prereg: str, n_jobs: int = 4, n_boot: int = 2000, log=print) -> dict:
    """The holdout, read once: every month from HOLDOUT_FROM scored walk-forward (each fitted on every month before
    it), the registered rule scored and gated, the other models beside it for the record."""
    rule, digest = read_prereg(prereg)
    d = prepared(panel)
    names = [m.name for m in exchange_models()]
    res = {"prereg": prereg, "prereg_sha256": digest, "rule": rule, "markets": {}, "secondary": []}
    scored = {}
    for mt, part in d.groupby("market_type"):
        sc = walk_forward(part, exchange_models(n_jobs), min_train_months=MIN_TRAIN_MONTHS, refit_every=1,
                          score_from=HOLDOUT_FROM, log=log)
        if sc.empty:
            continue
        scored[mt] = sc
        res["markets"][mt] = {"holdout": score(sc, names, thresholds=rule.get("thresholds_by_model", {}).get(mt),
                                               n_boot=n_boot)}
    for i, r in enumerate([rule] + list(rule.get("secondary", []))):
        sc = pd.concat([scored[mt] for mt in r["market_types"] if mt in scored], ignore_index=True)
        out = score_rule(sc, r, n_boot=n_boot)
        out["gate"] = gate(out, {**rule, **r})
        if i == 0:
            res["rule_result"], res["gate"] = out, out["gate"]
        else:
            res["secondary"].append(out)
    return res


# ---------------------------------------------------------------------------
# the tables (markdown from the JSON figures)
# ---------------------------------------------------------------------------

def _p(x) -> str:
    try:
        return f"{100 * float(x):+.2f}%" if np.isfinite(float(x)) else "-"
    except (TypeError, ValueError):
        return "-"


def _ci(r: dict, k: str = "roi") -> str:
    return f"{_p(r.get(k))} ({_p(r.get(k + '_lo90'))} to {_p(r.get(k + '_hi90'))})"


def accuracy_rows(acc: dict, above: dict) -> list[str]:
    out = ["| model | n | R2 vs no change | R2 above the training mean | direction hit |", "|---|---|---|---|---|"]
    for k, a in acc.items():
        r2 = a.get("r2_vs_no_change")
        out.append(f"| {k} | {int(a.get('n', 0)):,} | {float(r2 if r2 is not None else np.nan):.4f} | "
                   f"{float(above.get(k) if above.get(k) is not None else np.nan):+.4f} | {_p(a.get('direction_hit'))} |")
    return out


def sweep_rows(sw: list[dict]) -> list[str]:
    out = ["| threshold | bets | bets a day | at the mid (90%) | a tick worse each end (90%) | right way | worst drawdown, "
           "a tick worse (GBP) |", "|---|---|---|---|---|---|---|"]
    for r in sw:
        if not r.get("bets"):
            out.append(f"| {r['threshold']:.2f} | 0 | - | - | - | - | - |")
            continue
        sp = {"roi": r.get("roi_spread"), "roi_lo90": r.get("roi_spread_lo90"), "roi_hi90": r.get("roi_spread_hi90")}
        out.append(f"| {r['threshold']:.2f} | {r['bets']:,} | {r['bets_per_day']:.1f} | {_ci(r)} | {_ci(sp)} | "
                   f"{_p(r['right_way'])} | {float(r.get('worst_drawdown_spread') or 0):.1f} |")
    return out


def group_rows(by: dict, mid: dict | None = None) -> list[str]:
    out = ["| group | bets | bets a day | at the mid | a tick worse each end (90%) | right way |",
           "|---|---|---|---|---|---|"]
    for k, r in by.items():
        m = (mid or {}).get(k, {})
        out.append(f"| {k} | {int(r.get('bets', 0)):,} | {float(r.get('bets_per_day') or 0):.1f} | {_p(m.get('roi'))} | "
                   f"{_ci(r)} | {_p(r.get('right_way'))} |")
    return out


def tables(res: dict, models=("gap_rule_ps", "gap_rule_avg", "gap_ols_ps", "gap_ols_avg", "gbm_base", "gbm_nopin",
                              "gbm_all")) -> str:
    """Every table of a development or holdout JSON, for the report."""
    out = []
    for mt, r in res.get("markets", {}).items():
        w = r.get("development") or r.get("holdout")
        out += [f"## {mt}: {'development' if 'development' in r else 'holdout'}", "",
                f"{w['rows']:,} selections, {w['markets']:,} markets, {w['days']} days, {w['months'][0]} to "
                f"{w['months'][1]}.", ""]
        out += accuracy_rows(w["accuracy"], w["r2_above_mean_shift"]) + [""]
        for m in models:
            e = w["models"].get(m)
            if not e:
                continue
            out += [f"### {m} (chosen threshold {e['threshold']:.2f}; mean CLV at the taken price "
                    f"{_p(e.get('clv_mean_spread'))})", ""] + sweep_rows(e["sweep"]) + [""]
            for key in ("book_width", "band", "tier", "pin_period"):
                out += [f"{m} at {e['threshold']:.2f} by {key}:", ""] + group_rows(e["by"][key], e["by"][f"{key}_mid"])
                out += [""]
            for wl, ws in e.get("width_sweeps", {}).items():
                out += [f"{m}, {wl} books only:", "",
                        "| threshold | bets | bets a day | at the mid (90%) | a tick worse each end (90%) | right way | "
                        "back side as quoted |", "|---|---|---|---|---|---|---|"]
                for x in ws:
                    if not x.get("bets"):
                        continue
                    sp = {"roi": x.get("roi_spread"), "roi_lo90": x.get("roi_spread_lo90"),
                          "roi_hi90": x.get("roi_spread_hi90")}
                    qb = x.get("quoted_back", {})
                    out.append(f"| {x['threshold']:.2f} | {x['bets']:,} | {x['bets_per_day']:.1f} | {_ci(x)} | {_ci(sp)} | "
                               f"{_p(x['right_way'])} | {int(qb.get('bets', 0)):,} bets, {_p(qb.get('roi'))} |")
                out += [""]
            q = e.get("quoted_back", {})
            if q.get("bets"):
                out += [f"{m} at {e['threshold']:.2f}, the back side as quoted (back at Betfair's recorded early back, "
                        f"lay a tick over its recorded closing back): {q['bets']:,} bets, {q['bets_per_day']:.1f} a day, "
                        f"{_ci(q)}, right way {_p(q.get('right_way'))}.", ""]
        for period in ("fresh", "stale", "absent"):
            x = r.get(f"development_pin_{period}")
            if not x:
                continue
            out += [f"### Development rows with Pinnacle {period} ({x['rows']:,}, {x['months'][0]} to "
                    f"{x['months'][1]})", ""] + accuracy_rows(x["accuracy"], x["r2_above_mean_shift"]) + [""]
            out += ["| model | threshold | bets | a tick worse each end (90%) | right way |", "|---|---|---|---|---|"]
            for m, by_t in x["trade"].items():
                for t, s in by_t.items():
                    if s.get("bets"):
                        out.append(f"| {m} | {float(t):.2f} | {s['bets']:,} | {_ci(s)} | {_p(s.get('right_way'))} |")
            out += [""]
        for k, x in r.get("gap_rule_all_months", {}).items():
            out += [f"### {k}, every development month (the rule fits nothing): {x['rows']:,} rows, {x['months'][0]} "
                    f"to {x['months'][1]}", ""] + sweep_rows(x["sweep"]) + [""]
            for t, by in x["by_book_width"].items():
                out += [f"{k} at {float(t):.2f} by book width:", ""]
                out += group_rows(by, x.get("by_book_width_mid", {}).get(t)) + [""]
        if "shuffle_test" in r:
            sh = r["shuffle_test"]
            out += ["### Leak test: the training closes permuted", ""]
            out += accuracy_rows(sh["accuracy"], sh["r2_above_mean_shift"]) + [""]
        if "feature_gain_gbm_all" in r:
            out += ["### gbm_all feature gain (fitted on every development month)", "", "| feature | gain |", "|---|---|"]
            out += [f"| {k} | {v:,.0f} |" for k, v in list(r["feature_gain_gbm_all"].items())[:14]] + [""]
    registered = [("The pre-registered rule", res["rule_result"])] if "rule_result" in res else []
    registered += [(f"Secondary rule {i + 1}", x) for i, x in enumerate(res.get("secondary", []))]
    for title, x in registered:
        s = x["spread"]
        shown = {k: v for k, v in x["rule"].items() if k not in ("secondary", "thresholds_by_model")}
        out += [f"## {title}: {json.dumps(shown)}", "",
                f"Rows it may trade: {x['rows']:,}. Gate: {json.dumps(x.get('gate'))}", "",
                f"- a tick worse at each end: {s.get('bets', 0):,} bets, {_ci(s)}, "
                f"{float(s.get('bets_per_day') or 0):.1f} a day, right way {_p(s.get('right_way'))}, "
                f"worst drawdown {float(s.get('worst_drawdown') or 0):.1f}",
                f"- at the mid: {_ci(x['mid'])}", f"- mean CLV at the taken price: {_p(x.get('clv_mean_spread'))}",
                f"- the back side as quoted: {x['quoted_back'].get('bets', 0):,} bets, {_ci(x['quoted_back'])}", ""]
        for key in ("book_width", "band", "tier", "month"):
            out += [f"by {key}:", ""] + group_rows(x[f"by_{key}"]) + [""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _json(obj) -> str:
    def conv(x):
        if isinstance(x, (np.floating,)):
            return None if np.isnan(x) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return str(x)
    return json.dumps(obj, indent=1, default=conv)


def build(root: str, panel_path: str, dc_path: str | None, workers: int = 4) -> dict:
    from football.data import load_gold
    from football.model import walk_forward as dc_walk_forward
    from sources.common import Store
    store = Store(root=root)
    gold = load_gold(store)
    silver = store.get_parquet("football/matches.parquet")
    if dc_path and Path(dc_path).exists():
        dc = pd.read_parquet(dc_path)
    else:
        dc = dc_walk_forward(gold, PANEL_FROM, workers=workers)[["match_id", "p_h", "p_d", "p_a", "p_o25"]]
        if dc_path:
            dc.to_parquet(dc_path, index=False)
    d, prov = exchange_panel(gold, silver_with_ids(silver, gold), dc)
    prov["dc_matches"] = int(len(dc))
    p = prepared(d)
    prov["rows_after_clean"] = int(len(p))
    prov["rows_dropped_by_clean"] = int(len(d) - len(p))
    prov["by_market_type"] = p.groupby("market_type").agg(rows=("y", "size"), markets=("market_id", "nunique"),
                                                          first=("day", "min"), last=("day", "max")).to_dict("index")
    cols = ["ps", "avg", "max", "b365", "gap_ps", "gap_avg", "gap_max", "gap_b365", "dc_p", "season_stage",
            "off_hour", "bfe_over_open"]
    prov["null_share"] = {c: float(p[c].isna().mean()) for c in cols}
    prov["pin_period_rows"] = p["pin_period"].value_counts().to_dict()
    prov["book_width_rows"] = p.groupby(["market_type", "book_width"]).size().rename("n").reset_index().to_dict("records")
    d.to_parquet(panel_path, index=False)
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="the local sources folder (football-data's silver and gold)")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--dc-cache", default=None, help="parquet of the walk-forward Dixon-Coles prices")
    ap.add_argument("--build-panel", action="store_true")
    ap.add_argument("--develop", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--prereg", default="reports/preregistration_football_exchange.md")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tables", default=None, help="write the markdown tables of --out's JSON here")
    ap.add_argument("--save-scored", default=None, help="development: keep the scored rows, <prefix>.<market>.parquet")
    a = ap.parse_args(argv)
    if a.build_panel:
        print(_json(build(a.root, a.panel, a.dc_cache, a.n_jobs)))
    if a.develop or a.holdout:
        panel = pd.read_parquet(a.panel)
        res = (develop(panel, a.n_jobs, a.n_boot, save_scored=a.save_scored) if a.develop
               else holdout(panel, a.prereg, a.n_jobs, a.n_boot))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(_json(res) + "\n")
    if a.tables and a.out:
        Path(a.tables).write_text(tables(json.loads(Path(a.out).read_text())) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
