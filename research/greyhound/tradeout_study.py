"""The greyhound trade-out study (the owner, 2 Nov 2026): can a price taken before the off be closed at the SP for a
profit, which forecast of the SP does it best, and is the data enough?

Read-only. The panel is ``greyhound.tradeout``'s: every GB race the forward test of 5 Oct priced (its frozen win
model, out of sample on every day from 6 Oct) against the recorder's greyhound books (60 seconds apart in a race's
last hour, the delayed key) at four entry marks (the dog's first traded book, 5, 3 and 1 minutes before the off),
closed at the Betfair SP.

1. **The data** (a row a day): races, the polls a market gets in its last five minutes, the feed's lag (the poll's
   time less the market's last match, in the last three minutes, when money trades every few seconds), the books'
   depth and spread at each mark, the SP's coverage, and the trader's 5-second books where they are kept.
2. **The forecasts of the SP**, walk-forward by day (each day forecast by models fitted on the days before it only,
   from the 8th day), at each mark: the market alone (a conditional logit on the mid); the market with its
   microstructure (the last traded price, the move over two minutes, the queue imbalance, the spread); the horses'
   closing model unchanged (``race_book``: the market, our price, the money); the full microstructure model with our
   price (``greyhound.tradeout.MICRO``); LightGBM on the move from the mid to the SP's share (fixed settings, chosen
   before the data); and the microstructure model fitted to SPs shuffled within each race (the leak test: it must
   earn nothing).
3. **The trade**, GBP2 a position, back at the best back or lay at the best lay with the stake offered, a price up to
   20, closed at the SP, 2% commission on each market's net; bars on the expected CLV of 0, 0.02, 0.03, 0.05, 0.08.
   The last seven scored days are the holdout, read once, each rule at the bar its development days chose (the best
   lower 90% bound among bars trading at least once a day).
4. **The learning curve:** the same forecasts fitted on only the last 3, 7 or 14 days, on the same scored days: if
   the longer windows still gain, more days of books will too.
5. **What carries the forecast:** LightGBM's gain and the microstructure model's standardised coefficients, fitted on
   the development days.

    python -m research.greyhound.tradeout_study --out out/greyhound_tradeout_study
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound import tradeout as to
from model import race_book

log = logging.getLogger(__name__)

STUDY_BARS = (0.0, 0.02, 0.03, 0.05, 0.08)
MIN_TRAIN_DAYS = 7
HOLDOUT_DAYS = 7
WINDOWS = (3, 7, 14, None)
CL_SETS = {
    "market": ["lq_market"],
    "market_micro": ["lq_market", "last_vs_mid", "move", "imbalance", "log_spread", "lq_market_x_vol"],
    "micro": list(to.MICRO),
}
GBM_FEATURES = ["lq_market", "lq_model", "last_vs_mid", "move", "imbalance", "log_spread", "log_vol", "log_depth",
                "overround", "rank", "minutes_to_off", "trap", "field"]
GBM_PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 100,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
              "seed": 0, "num_threads": 2}
GBM_ROUNDS = 300
MODEL_NAMES = ["market", "market_micro", "racebook", "micro", "gbm", "micro_shuffled"]


def _log(msg: str) -> None:
    log.info(msg)
    print(msg, flush=True)


# --------------------------------------------------------------------------------------------------------------------
# 1. The data, a day at a time
# --------------------------------------------------------------------------------------------------------------------

def _q(x, qs=(0.5, 0.9)) -> dict:
    x = pd.Series(x).dropna()
    return {f"p{int(100 * q)}": round(float(x.quantile(q)), 2) for q in qs} if len(x) else None


def audit_day(day: date, pred, books, markets, panel: pd.DataFrame, trader) -> dict:
    out = {"day": f"{day}", "races_priced": int(pred.race_id.nunique()) if pred is not None else 0}
    if books is None or markets is None or not len(books):
        out["note"] = "no recorded books"
        return out
    w = markets[(markets.market_type == "WIN") & (markets.country.fillna("GB") == "GB")]
    out["gb_win_markets"] = int(w.market_id.nunique())
    pre = to.pre_off_books(books, markets)
    last5 = pre[pre.minutes_to_off <= 5].drop_duplicates(["market_id", "polled"])
    polls = last5.groupby("market_id").polled.nunique()
    out["polls_last5_per_market"] = _q(polls)
    gaps = last5.sort_values("polled").groupby("market_id").polled.diff().dt.total_seconds()
    out["poll_gap_last5_s"] = _q(gaps)
    if "last_match_utc" in pre:
        l3 = pre[pre.minutes_to_off <= 3].drop_duplicates(["market_id", "polled"])
        lm = pd.to_datetime(l3.last_match_utc, utc=True, errors="coerce")
        out["feed_lag_last3_s"] = _q((l3.polled - lm).dt.total_seconds())
    if "is_delayed" in pre:
        out["is_delayed_share"] = round(float(pre.is_delayed.astype(str).str.lower().isin(["true", "1"]).mean()), 3)
    if len(panel):
        g = panel.groupby("mark")
        out["races_in_panel"] = {m: int(x.race_id.nunique()) for m, x in g}
        out["at_mark"] = {m: {"minutes_to_off_p50": round(float(x.minutes_to_off.median()), 2),
                              "spread_p50": round(float(np.exp(x.log_spread).median()), 3),
                              "tight_share": round(float((np.exp(x.log_spread) <= 1.1).mean()), 3),
                              "back1_size_p50": round(float(x.back1_size.median()), 1),
                              "lay1_size_p50": round(float(x.lay1_size.median()), 1),
                              "market_matched_p50": round(float(x.drop_duplicates("market_id")
                                                                 .market_total_matched.median()), 0)}
                          for m, x in g}
    if trader is not None and len(trader):
        tp = to.pre_off_books(trader, markets)
        t3 = tp[tp.minutes_to_off <= 3].drop_duplicates(["market_id", "polled"])
        out["trader_books"] = {"rows": int(len(trader)), "markets": int(tp.market_id.nunique()),
                               "polls_last3_per_market": _q(t3.groupby("market_id").polled.nunique()),
                               "feed_lag_last3_s": _q(to.feed_lag(tp))}
    return out


def load(store, first: date, last: date, source: str = "recorder") -> tuple[pd.DataFrame, list[dict]]:
    """Every day's panel (entry books from ``source``: the recorder's 60-second delayed books, or the trader's live
    5-second books; the SP from the recorder's settled books) and its audit row."""
    from betfair_recorder import _read_day_csv
    frames, audit, d = [], [], first
    while d <= last:
        try:
            pred = store.get_parquet(f"{to.TRACK_PRED}/{d:%Y-%m-%d}.parquet")
            books = _read_day_csv(d, "books_greyhound.csv")
            markets = _read_day_csv(d, "markets_greyhound.csv")
            trader = _read_day_csv(d, "books_greyhound_trader.csv")
            entry = books if source == "recorder" else trader
            panel = (to.day_panel(pred, entry, markets, d, closing_books=books)
                     if pred is not None and entry is not None and books is not None else pd.DataFrame())
            audit.append(audit_day(d, pred, books, markets, panel, trader))
            if len(panel):
                frames.append(panel)
            _log(f"{d}: {len(panel)} panel rows")
        except Exception as exc:                                    # one bad day: noted, the rest still read
            audit.append({"day": f"{d}", "error": str(exc)[:300]})
            _log(f"{d}: failed ({exc})")
        d += timedelta(days=1)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), audit


# --------------------------------------------------------------------------------------------------------------------
# 2. The forecasts of the SP
# --------------------------------------------------------------------------------------------------------------------

def _shares(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    g = to._codes(d)
    return g, race_book.normalise_within(np.exp(d.lq_close.to_numpy(float)), g)


def fit_cl(d: pd.DataFrame, cols: list[str], lq_close=None) -> dict:
    """A conditional logit on the SP book's shares (or on ``lq_close`` given instead: the leak test)."""
    if lq_close is None:
        g, q = _shares(d)
    else:
        g = to._codes(d)
        q = race_book.normalise_within(np.exp(np.asarray(lq_close, float)), g)
    X = to.micro_raw(d)[:, [to.MICRO.index(c) for c in cols]]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    start = np.zeros(len(cols))
    start[0] = sd[0]
    beta = to.fit_clogit((X - mu) / sd, q, g, start=start)
    inv = 1.0 / d.bsp.to_numpy(float)
    return {"kind": "cl", "cols": cols, "beta": beta, "mu": mu, "sd": sd,
            "book": float(np.median(np.bincount(g, weights=inv)))}


def fit_gbm(d: pd.DataFrame) -> dict:
    import lightgbm as lgb
    y = d.lq_close - d.lq_market
    m = lgb.train(GBM_PARAMS, lgb.Dataset(d[GBM_FEATURES].astype(float), y), GBM_ROUNDS)
    g = to._codes(d)
    return {"kind": "gbm", "booster": m, "book": float(np.median(np.bincount(g, weights=1.0 / d.bsp.to_numpy(float))))}


def fit_all(d: pd.DataFrame, names=MODEL_NAMES, seed: int = 0) -> dict:
    out = {}
    for n in names:
        if n in CL_SETS:
            out[n] = fit_cl(d, CL_SETS[n])
        elif n == "racebook":
            out[n] = {"kind": "racebook", "fitted": to.fit_models(d)}
        elif n == "gbm":
            out[n] = fit_gbm(d)
        elif n == "micro_shuffled":
            rng = np.random.default_rng(seed)                    # the SPs shuffled within each race
            lq = pd.Series(d.lq_close.to_numpy(float)).groupby(to._codes(d)).transform(
                lambda s: rng.permutation(s.to_numpy()))
            out[n] = fit_cl(d, CL_SETS["micro"], lq_close=lq.to_numpy())
    return out


def predict(d: pd.DataFrame, f: dict) -> tuple[np.ndarray, float]:
    g = to._codes(d)
    if f["kind"] == "racebook":
        return to.predict_share(d, f["fitted"], "racebook")
    if f["kind"] == "cl":
        X = to.micro_raw(d)[:, [to.MICRO.index(c) for c in f["cols"]]]
        return np.exp(race_book.group_log_softmax(((X - f["mu"]) / f["sd"]) @ f["beta"], g)), f["book"]
    u = d.lq_market.to_numpy(float) + f["booster"].predict(d[GBM_FEATURES].astype(float))
    return np.exp(race_book.group_log_softmax(u, g)), f["book"]


def accuracy(d: pd.DataFrame, q_hat: np.ndarray, book: float) -> dict:
    """Cross-entropy of the SP book's shares against the forecast and against the market now (per race), and the R2
    of log(SP / mid) against no change."""
    g, q = _shares(d)
    n = g.max() + 1
    qm = race_book.normalise_within(1.0 / d.mid.to_numpy(float), g)
    y = np.log(d.bsp.to_numpy(float) / d.mid.to_numpy(float))
    yhat = -np.log(book * q_hat) - np.log(d.mid.to_numpy(float))
    won = d.won.to_numpy(float) == 1
    return {"xent": -float(np.sum(q * np.log(np.clip(q_hat, 1e-12, 1)))) / n,
            "xent_market": -float(np.sum(q * np.log(qm))) / n,
            # Benter's test against the result too: the winners' log chance, the forecast against the market now
            "ll_win": float(np.sum(np.log(np.clip(q_hat[won], 1e-12, 1)))), "ll_win_market": float(np.sum(np.log(qm[won]))),
            "sse": float(np.sum((y - yhat) ** 2)), "sse_nochange": float(np.sum(y ** 2)),
            "right_way": float(np.mean(np.sign(yhat) == np.sign(y))), "rows": int(len(d)), "races": int(n)}


def walk_forward(panel: pd.DataFrame, names=MODEL_NAMES, window=None, score_days=None) -> tuple[pd.DataFrame, list]:
    """Each scored day forecast by models fitted on the days before it only (the last ``window`` of them when
    given). Returns every rule's positions and each day's accuracy."""
    days = sorted(panel.race_date.unique())
    pos, acc = [], []
    for i, day in enumerate(days):
        if i < MIN_TRAIN_DAYS or (score_days is not None and day not in score_days):
            continue
        tr_days = days[:i] if window is None else days[max(0, i - window):i]
        assert max(tr_days) < day                                   # never a row the model saw
        for mark, g_all in panel.groupby("mark"):
            tr = g_all[g_all.race_date.isin(tr_days)]
            te = g_all[g_all.race_date == day]
            if not len(tr) or not len(te):
                continue
            fitted = fit_all(tr, names)
            for n, f in fitted.items():
                q, book = predict(te, f)
                acc.append({"day": day, "mark": mark, "model": n, "window": window or 0, **accuracy(te, q, book)})
                p = to.positions(te, to.expected_clv(te, q, book), n, bars=STUDY_BARS)
                if len(p):
                    pos.append(p.assign(window=window or 0))
            if window is None:                                      # the old selection, for comparison
                p = to.positions(te, None, "win_edge")
                if len(p):
                    pos.append(p.assign(window=0))
    return (pd.concat(pos, ignore_index=True) if pos else pd.DataFrame()), acc


def accuracy_table(acc: list, days=None) -> list[dict]:
    a = pd.DataFrame(acc)
    if days is not None:
        a = a[a.day.isin(days)]
    out = []
    for (mark, model, w), g in a.groupby(["mark", "model", "window"]):
        races = g.races.sum()
        out.append({"mark": mark, "model": model, "window": int(w), "days": int(len(g)), "races": int(races),
                    "xent_gain_per_race": round(float((g.xent_market * g.races).sum() - (g.xent * g.races).sum())
                                                / races, 5),
                    "r2_vs_no_change": round(1 - float(g.sse.sum() / g.sse_nochange.sum()), 4),
                    "win_ll_gain_per_race": round(float(g.ll_win.sum() - g.ll_win_market.sum()) / races, 5),
                    "right_way%": round(100 * float((g.right_way * g.rows).sum() / g.rows.sum()), 1)})
    return out


def baselines(panel: pd.DataFrame) -> list[dict]:
    """How far each price at each mark sits from the SP (Betfair's own BSP study's measure, the absolute percentage
    error): the best back, the best lay, the geometric mid of the two, the closing model's mid (race_book.market_now),
    the last traded price."""
    out = []
    for mark, g in panel.groupby("mark"):
        bsp = g.bsp.to_numpy(float)
        cand = {"back1": g.back1, "lay1": g.lay1.where(g.lay1 > 1), "geo_mid": np.sqrt(g.back1 * g.lay1.where(g.lay1 > 1)),
                "market_now": g["mid"], "last_traded": g.last_traded.where(g.last_traded > 1)}
        for name, x in cand.items():
            e = (np.abs(np.asarray(x, float) / bsp - 1))
            e = e[np.isfinite(e)]
            out.append({"mark": mark, "price": name, "rows": int(len(e)),
                        "mape%": round(100 * float(e.mean()), 2) if len(e) else None,
                        "median_ape%": round(100 * float(np.median(e)), 2) if len(e) else None,
                        "mean_log_bsp_over_price": round(float(np.nanmean(np.log(bsp / np.asarray(x, float)))), 4)})
    return out


def choose_bars(dev_table: pd.DataFrame) -> dict:
    """Each (mark, model, side)'s bar on the development days: the best lower 90% bound among bars trading at least
    once a day."""
    out = {}
    t = dev_table[(dev_table.per_day >= 1) & dev_table["roi_90%"].notna()].copy()
    if not len(t):
        return out
    t["lo"] = t["roi_90%"].map(lambda x: x[0])
    for key, g in t.groupby(["mark", "model", "side"]):
        out[key] = float(g.sort_values("lo", ascending=False).bar.iloc[0])
    return out


def by_band(p: pd.DataFrame, keys=("mark", "model", "side")) -> list[dict]:
    if not len(p):
        return []
    p = p.assign(band=pd.cut(p.price, [1, 2, 3, 5, 8, 13, 21], labels=["1-2", "2-3", "3-5", "5-8", "8-13", "13-20"]))
    t = to.rule_table(p.assign(bar=p.band.astype(str)), keys=tuple(keys) + ("bar",), n_boot=500)
    return t.rename(columns={"bar": "band"}).to_dict("records")


def importance(panel: pd.DataFrame, dev_days: list) -> dict:
    out = {}
    for mark in ("off_1", "off_3", "first_traded"):
        d = panel[(panel["mark"] == mark) & panel.race_date.isin(dev_days)]
        if not len(d):
            continue
        f = fit_all(d, ["micro", "gbm"])
        gain = f["gbm"]["booster"].feature_importance("gain")
        out[mark] = {"micro_beta_standardised": dict(zip(to.MICRO, np.round(f["micro"]["beta"], 4).tolist())),
                     "gbm_gain_share": dict(sorted(zip(GBM_FEATURES, np.round(gain / gain.sum(), 4).tolist()),
                                                   key=lambda kv: -kv[1]))}
    return out


# --------------------------------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------------------------------

def _tbl(rows: list[dict], cols: list[str]) -> list[str]:
    if not rows:
        return ["(none)", ""]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    return out + [""]


def report_md(res: dict) -> str:
    L = [f"# Greyhound trade-out study ({res['first_day']} to {res['last_day']})", "",
         f"Entry books: {res.get('source')}. Panel: {res['panel_rows']} runner-marks, {res['panel_races']} races, {res['panel_days']} days; scored days "
         f"{res['dev_days'][0] if res['dev_days'] else None} to {res['holdout_days'][-1] if res['holdout_days'] else None}"
         f" (development {len(res['dev_days'])}, holdout {len(res['holdout_days'])}).", "",
         "## 1. Data audit (a day a row)", ""]
    L += _tbl([{"day": a["day"], "races": a.get("races_priced"), "panel races (off_1)":
                (a.get("races_in_panel") or {}).get("off_1"), "polls last 5 min (p50)":
                (a.get("polls_last5_per_market") or {}).get("p50"), "poll gap s (p50)":
                (a.get("poll_gap_last5_s") or {}).get("p50"), "feed lag s (p50/p90)": a.get("feed_lag_last3_s"),
                "trader books (lag p50)": ((a.get("trader_books") or {}).get("rows"),
                                          ((a.get("trader_books") or {}).get("feed_lag_last3_s") or {}).get("p50")), "note": a.get("note") or a.get("error")}
               for a in res["audit"]], ["day", "races", "panel races (off_1)", "polls last 5 min (p50)",
                                        "poll gap s (p50)", "feed lag s (p50/p90)", "trader books (lag p50)", "note"])
    L += ["### Each price at each mark against the SP (all panel days)", ""]
    L += _tbl(res.get("baselines", []), ["mark", "price", "rows", "mape%", "median_ape%", "mean_log_bsp_over_price"])
    L += ["## 2. Accuracy of the SP forecast, development days", "",
          "xent_gain: the SP book's cross-entropy, the market now less the forecast, a race (Benter's gain over the "
          "market, the SP as the target); win_ll_gain: the same against the result.", ""]
    L += _tbl(res["accuracy_dev"], ["mark", "model", "days", "races", "xent_gain_per_race", "win_ll_gain_per_race",
                                    "r2_vs_no_change", "right_way%"])
    L += ["### Holdout", ""]
    L += _tbl(res["accuracy_holdout"], ["mark", "model", "days", "races", "xent_gain_per_race",
                                        "win_ll_gain_per_race", "r2_vs_no_change", "right_way%"])
    cols = ["mark", "model", "side", "bar", "trades", "per_day", "roi%", "roi_90%", "roi_liability%", "held_roi%",
            "clv_mean%", "clv_positive%", "price_median"]
    L += ["## 3. The trade closed at the SP, development days (every bar)", ""] + _tbl(res["dev_rules"], cols)
    L += ["## 4. Holdout, read once (each rule at its development bar)", ""] + _tbl(res["holdout_rules"], cols)
    L += ["## 5. Learning curve (development days scored by every window)", ""]
    L += _tbl(res["learning_accuracy"], ["mark", "model", "window", "days", "xent_gain_per_race", "r2_vs_no_change"])
    L += _tbl(res["learning_rules"], ["mark", "model", "window", "side", "bar", "trades", "roi%", "roi_90%"])
    L += ["## 6. By price band (development, bar 0.03)", ""]
    L += _tbl(res["dev_by_band"], ["mark", "model", "side", "band", "trades", "roi%", "roi_90%", "clv_mean%"])
    L += ["## 7. What carries the forecast (development days)", "", "```", json.dumps(res["importance"], indent=1),
          "```", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--from", dest="first", default=str(to.FIRST_BOOK_DAY))
    ap.add_argument("--to", dest="last", default=None, help="default: yesterday (UK)")
    ap.add_argument("--out", default="out/greyhound_tradeout_study")
    ap.add_argument("--source", choices=["recorder", "trader"], default="recorder",
                    help="whose books at the entry marks: the recorder's (delayed, 60 s) or the trader's (live, 5 s)")
    a = ap.parse_args(argv)
    t0 = time.time()
    store = Store(root=a.store)
    last = date.fromisoformat(a.last) if a.last else uk_today() - timedelta(days=1)
    panel, audit = load(store, date.fromisoformat(a.first), last, source=a.source)
    res = {"first_day": a.first, "last_day": f"{last}", "source": a.source, "audit": audit,
           "panel_rows": int(len(panel))}
    if not len(panel):
        res["note"] = "no panel"
    else:
        days = sorted(panel.race_date.unique())
        scored = days[MIN_TRAIN_DAYS:]
        hold = scored[-HOLDOUT_DAYS:] if len(scored) > HOLDOUT_DAYS + 3 else []
        dev = [d for d in scored if d not in hold]
        res.update({"panel_races": int(panel.race_id.nunique()), "panel_days": len(days), "dev_days": dev,
                    "holdout_days": hold})
        res["baselines"] = baselines(panel)
        _log(f"walk-forward over {len(scored)} scored days")
        pos, acc = walk_forward(panel)
        res["accuracy_dev"] = accuracy_table(acc, dev)
        res["accuracy_holdout"] = accuracy_table(acc, hold)
        dev_t = to.rule_table(pos[pos.race_date.isin(dev)])
        res["dev_rules"] = dev_t.to_dict("records")
        bars = choose_bars(dev_t)
        res["chosen_bars"] = {"|".join(k): v for k, v in bars.items()}
        hp = pos[pos.race_date.isin(hold)]
        if len(hp):
            chosen = pd.Series([bars.get(k) for k in zip(hp["mark"], hp.model, hp.side)], index=hp.index)
            keep = (hp.model == "win_edge") | (chosen == hp.bar)
            res["holdout_rules"] = to.rule_table(hp[keep]).to_dict("records")
        else:
            res["holdout_rules"] = []
        res["dev_by_band"] = by_band(pos[pos.race_date.isin(dev) & (pos.bar == 0.03)
                                         & pos.model.isin(["racebook", "micro", "gbm"])])
        _log("learning curve")
        lc_days = dev[-max(1, len(dev) // 2):]                       # the later development days: every window fits
        sub = panel[panel["mark"].isin(["off_1", "off_3"])]
        lc_acc, lc_pos = [], []
        for w in WINDOWS:
            p, ac = walk_forward(sub, ["market_micro", "micro", "gbm"], window=w, score_days=lc_days)
            lc_acc += ac
            if len(p):
                lc_pos.append(p[p.bar == 0.03])
        res["learning_accuracy"] = accuracy_table(lc_acc)
        lp = pd.concat(lc_pos, ignore_index=True) if lc_pos else pd.DataFrame()
        res["learning_rules"] = to.rule_table(lp, keys=("mark", "model", "window", "side", "bar"),
                                              n_boot=500).to_dict("records") if len(lp) else []
        res["importance"] = importance(panel, dev)
    res["seconds"] = round(time.time() - t0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    md = report_md(res) if len(panel) else json.dumps(res, indent=1, default=str)
    out.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
