"""The Phase 0 market conditions, pre-match (build spec 8): does the model add to Betfair's price, and does a
strategy on it survive commission?

    python -m cricket.model.benter --root data/cricket/store

Inputs: the walk-forward pre-match probabilities (cricket/model/prematch.py, ``cricket/eval/prematch.parquet``) and
Betfair's Historic Data for cricket on the free BASIC plan (betfair_historic.py: ``betfair_historic/markets_cricket_
<year>.parquet``, one row a runner of each MATCH_ODDS market: the price last traded before it went in play, and at
60, 15 and 1 minute before the scheduled off). BASIC has the last traded price each minute and no ladder: the
"mid" of the spec is not in it, so the market's probability is the two closing prices' implied probabilities,
normalised for the overround.

1. **Matching.** A market is a Cricsheet match when its scheduled day is the match day or one either side and its
   two runners are the two sides (names compared without case and punctuation, or one inside the other, or sharing
   half their words). Every market and every name left unmatched is counted in the report.
2. **Benter.** A logistic regression of the result on logit(model p) and logit(market p). The model's weight, with
   its standard error from the information matrix, says whether the model carries anything the price does not.
   Walk-forward: each year scored by a fit on the years before, against the market alone (refitted the same way);
   the difference with a 95% interval from resampling matches.
3. **Strategy (indicative).** Walk-forward combined p; back the side whose closing price is worth more than
   ``threshold`` a unit after 5% commission; GBP 1 level stakes; settled on the result. Filled at the last traded
   price, which BASIC cannot show was available: an upper bound, not a paper result.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cricket.model.calibration import logit
from cricket.model.metrics import log_loss, paired_bootstrap
from sources.common import Store

SPORT_DIR = "cricket"
COMMISSION = 0.05


# ---------------------------------------------------------------------------------------------------------------
# Betfair markets to Cricsheet matches
# ---------------------------------------------------------------------------------------------------------------

def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(name).lower()).strip()


# words that make a different side of the same name: women's, legends, A and age-group teams
MARKERS = {"w", "women", "womens", "legends", "a", "u19", "u23", "xi", "emerging", "masters", "champions"}


def same_side(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if (set(a.split()) & MARKERS) != (set(b.split()) & MARKERS):
        return False
    if a == b or a in b or b in a:
        return True
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / len(ta | tb) >= 0.5


def market_table(betfair: pd.DataFrame) -> pd.DataFrame:
    """One row a two-runner MATCH_ODDS market with both closing prices and a winner."""
    b = betfair[betfair.market_type == "MATCH_ODDS"].copy()
    b = b[b.runner_status.isin(["WINNER", "LOSER"])]
    g = b.groupby("market_id")
    ok = g.selection_id.transform("size").eq(2) & g.won.transform("sum").eq(1)
    b = b[ok].sort_values(["market_id", "sort_priority", "selection_id"])
    rows = []
    for mid, m in b.groupby("market_id", sort=False):
        r1, r2 = m.iloc[0], m.iloc[1]
        rows.append({"market_id": mid, "event_name": r1.event_name, "market_time": r1.market_time,
                     "went_in_play_utc": r1.went_in_play_utc, "runner_1": r1.runner_name, "runner_2": r2.runner_name,
                     "won_1": int(r1.won), **{f"{k}_{i}": r[k] for i, r in ((1, r1), (2, r2))
                                              for k in ("ltp_close", "ltp_t_60", "ltp_t_1")}})
    return pd.DataFrame(rows)


def match_markets(markets: pd.DataFrame, pre: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Each market to at most one pre-match row, by day (+-1) and both sides; returns the joined rows and counts."""
    pre = pre.assign(day=pd.to_datetime(pre.start_date))
    by_day = {d: g for d, g in pre.groupby("day")}
    out, unmatched = [], []
    for m in markets.itertuples():
        t = pd.Timestamp(str(m.market_time)[:10])
        cands = pd.concat([by_day[d] for d in (t - pd.Timedelta(days=1), t, t + pd.Timedelta(days=1))
                           if d in by_day]) if any(d in by_day for d in (t - pd.Timedelta(days=1), t,
                                                                         t + pd.Timedelta(days=1))) else None
        hit = []
        if cands is not None:
            for c in cands.itertuples():
                if same_side(m.runner_1, c.bat_first) and same_side(m.runner_2, c.bowl_first):
                    hit.append((c, 1))
                elif same_side(m.runner_1, c.bowl_first) and same_side(m.runner_2, c.bat_first):
                    hit.append((c, 2))
        if len(hit) != 1:
            unmatched.append({"market_id": m.market_id, "event_name": m.event_name, "market_time": m.market_time,
                              "candidates": len(hit)})
            continue
        c, bat_runner = hit[0]
        o_bat = getattr(m, f"ltp_close_{bat_runner}")
        o_bowl = getattr(m, f"ltp_close_{3 - bat_runner}")
        won_bat = m.won_1 if bat_runner == 1 else 1 - m.won_1
        if not (o_bat and o_bowl) or pd.isna(o_bat) or pd.isna(o_bowl) or int(won_bat) != int(c.bat_won):
            unmatched.append({"market_id": m.market_id, "event_name": m.event_name, "market_time": m.market_time,
                              "candidates": 1, "why": "no close" if pd.isna(o_bat) or pd.isna(o_bowl)
                              else "Betfair's result differs from Cricsheet's"})
            continue
        q_bat, q_bowl = 1.0 / o_bat, 1.0 / o_bowl
        out.append({"match_id": c.match_id, "market_id": m.market_id, "year": c.year, "event_name": c.event_name,
                    "bat_won": int(c.bat_won), "p_model": c.p_bat_first, "o_bat": o_bat, "o_bowl": o_bowl,
                    "overround": q_bat + q_bowl, "p_market": q_bat / (q_bat + q_bowl)})
    j = pd.DataFrame(out).drop_duplicates("match_id", keep=False) if out else pd.DataFrame(out)
    why = pd.Series([_why(u) if u.get("why") else ("no Cricsheet match in scope that day" if u["candidates"] == 0
                     else "more than one candidate (a series)") for u in unmatched]).value_counts().to_dict()
    counts = {"markets": int(len(markets)), "matched": int(len(j)), "unmatched": len(unmatched),
              "unmatched_why": why, "unmatched_sample": unmatched[:20]}
    return j, counts


# ---------------------------------------------------------------------------------------------------------------
# the regression
# ---------------------------------------------------------------------------------------------------------------

def logistic(X: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Unpenalised logistic regression by Newton-Raphson, with an intercept first; (coefficients, standard errors)."""
    X = np.c_[np.ones(len(X)), X]
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(-X @ b))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    return b, np.sqrt(np.diag(cov))


def _predict(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(b[0] + X @ b[1:])))


def benter(j: pd.DataFrame, threshold: float = 0.02) -> dict:
    y = j.bat_won.to_numpy(dtype=float)
    Xc = np.c_[logit(j.p_model), logit(j.p_market)]
    b, se = logistic(Xc, y)
    head = {"model": log_loss(y, j.p_model), "market": log_loss(y, j.p_market),
            "model_minus_market": paired_bootstrap(y, j.p_model, j.p_market, j.match_id),
            "corr_logit": float(np.corrcoef(Xc[:, 0], Xc[:, 1])[0, 1]),
            "by_event": j.event_name.value_counts().head(12).to_dict()}
    res = {"rows": int(len(j)), "years": sorted(int(x) for x in j.year.unique()), "head_to_head": head,
           "in_sample": {"intercept": b[0], "w_model": b[1], "se_model": se[1], "z_model": b[1] / se[1],
                         "w_market": b[2], "se_market": se[2], "z_market": b[2] / se[2]}}
    # walk-forward: each year from a fit on the years before
    rows = []
    for yr in res["years"][1:]:
        tr, te = j[j.year < yr], j[j.year == yr]
        if len(tr) < 50 or not len(te):
            continue
        bm, _ = logistic(logit(tr.p_market)[:, None], tr.bat_won.to_numpy(dtype=float))
        bc, _ = logistic(np.c_[logit(tr.p_model), logit(tr.p_market)], tr.bat_won.to_numpy(dtype=float))
        rows.append(te.assign(p_mkt_fit=_predict(bm, logit(te.p_market)[:, None]),
                              p_comb=_predict(bc, np.c_[logit(te.p_model), logit(te.p_market)])))
    if rows:
        w = pd.concat(rows, ignore_index=True)
        res["walk_forward"] = {"rows": int(len(w)), "market_raw": log_loss(w.bat_won, w.p_market),
                               "market_refit": log_loss(w.bat_won, w.p_mkt_fit),
                               "combined": log_loss(w.bat_won, w.p_comb), "model_alone": log_loss(w.bat_won, w.p_model),
                               "combined_minus_market": paired_bootstrap(w.bat_won, w.p_comb, w.p_mkt_fit, w.match_id)}
        res["strategy"] = strategy(w, threshold)
    return res


def strategy(w: pd.DataFrame, threshold: float, c: float = COMMISSION) -> dict:
    """GBP 1 backs at the closing price on the side worth more than ``threshold`` a unit after commission."""
    v_bat = w.p_comb * (w.o_bat - 1) * (1 - c) - (1 - w.p_comb)
    v_bowl = (1 - w.p_comb) * (w.o_bowl - 1) * (1 - c) - w.p_comb
    side = np.where(v_bat >= v_bowl, "bat", "bowl")
    val = np.maximum(v_bat, v_bowl)
    bet = val > threshold
    won = np.where(side == "bat", w.bat_won == 1, w.bat_won == 0)
    odds = np.where(side == "bat", w.o_bat, w.o_bowl)
    pnl = np.where(won, (odds - 1) * (1 - c), -1.0)[bet]
    return {"threshold": threshold, "bets": int(bet.sum()), "pnl_after_commission": float(pnl.sum()),
            "roi": float(pnl.mean()) if bet.any() else None,
            "by_year": {int(y): float(np.where(won, (odds - 1) * (1 - c), -1.0)[bet & (w.year == y).to_numpy()].sum())
                        for y in sorted(w.year.unique())}}


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------

def load_betfair(store: Store) -> tuple[pd.DataFrame | None, list[str]]:
    keys = sorted(k for k in store.listing("betfair_historic/") if re.fullmatch(
        rf"betfair_historic/markets_{SPORT_DIR}_\d{{4}}\.parquet", k))
    frames = [store.get_parquet(k) for k in keys]
    return (pd.concat(frames, ignore_index=True) if frames else None), keys


def _why(u: dict) -> str:
    return u.get("why") or f"{u['candidates']} candidate matches"


MIN_MATCHED = 100


def report_md(res: dict | None, counts: dict | None, keys: list[str], store: Store) -> str:
    L = ["# Cricket - Phase 0 market conditions, pre-match (Benter against Betfair's close)", ""]
    if res is None and counts is not None:
        L += [f"Betfair tables {', '.join(f'`{k}`' for k in keys)}: {counts['markets']:,} markets, "
              f"{counts['matched']:,} matched to Cricsheet, fewer than the {MIN_MATCHED} the test needs. "
              "Not tested.", "", "| market | event | time | why |", "|---|---|---|---|"]
        L += [f"| {u['market_id']} | {u['event_name']} | {u['market_time']} | {_why(u)} |"
              for u in counts["unmatched_sample"]]
        return "\n".join(L) + "\n"
    if res is None:
        L += [f"No data found: no Betfair cricket Historic Data table at `{store.where('betfair_historic/')}` "
              f"(`markets_{SPORT_DIR}_<year>.parquet`). The free BASIC plan for Cricket has to be added once on "
              "historicdata.betfair.com with the account's login; other-data.yml then fetches it and "
              "source-data.yml builds the table.", ""]
        return "\n".join(L)
    i = res["in_sample"]
    L += [f"Betfair: {', '.join(f'`{k}`' for k in keys)} (BASIC: last traded prices, no ladder). Model: "
          "walk-forward pre-match probabilities (`cricket/eval/prematch.parquet`).", "",
          f"Markets {counts['markets']:,}; matched to Cricsheet {counts['matched']:,}; "
          f"unmatched {counts['unmatched']:,} ("
          + "; ".join(f"{k}: {v:,}" for k, v in counts.get("unmatched_why", {}).items()) + "). In scope are men's "
          "20-over T20s decided without D/L that Cricsheet holds: most unmatched markets are women's, legends and "
          "domestic leagues it does not cover.",
          "", "## Benter regression (all matched matches)", "",
          "| term | weight | s.e. | z |", "|---|---:|---:|---:|",
          f"| logit(model p) | {i['w_model']:.3f} | {i['se_model']:.3f} | {i['z_model']:.2f} |",
          f"| logit(market p) | {i['w_market']:.3f} | {i['se_market']:.3f} | {i['z_market']:.2f} |", "",
          f"Model weight significantly positive (z above 1.96): **{'yes' if i['z_model'] > 1.96 else 'no'}**", ""]
    h = res["head_to_head"]
    b = h["model_minus_market"]
    L += ["## Model against the close, same matches", "",
          "| probability | log loss |", "|---|---:|", f"| market close (overround removed) | {h['market']:.4f} |",
          f"| model (walk-forward, pre-match) | {h['model']:.4f} |", "",
          f"Model minus market: {b['diff']:+.4f} (95% interval {b['lo']:+.4f} to {b['hi']:+.4f}, {b['groups']:,} "
          f"matches). Correlation of the two logits: {h['corr_logit']:.2f}.", "",
          "Matched matches by competition: " + "; ".join(f"{k} {v}" for k, v in h["by_event"].items()) + ".", ""]
    wf = res.get("walk_forward")
    if not wf:
        L += [f"Walk-forward and strategy: not run. They fit on earlier years, and the matched markets cover "
              f"{', '.join(map(str, res['years']))} only.", ""]
    if wf:
        b = wf["combined_minus_market"]
        L += ["## Walk-forward (each year from a fit on the years before)", "",
              "| probability | log loss |", "|---|---:|",
              f"| market, raw close | {wf['market_raw']:.4f} |", f"| market, refitted | {wf['market_refit']:.4f} |",
              f"| model alone | {wf['model_alone']:.4f} |", f"| combined | {wf['combined']:.4f} |", "",
              f"Combined minus market: {b['diff']:+.4f} (95% interval {b['lo']:+.4f} to {b['hi']:+.4f}, "
              f"{b['groups']:,} matches). The model adds to the price: "
              f"**{'yes' if b['hi'] < 0 else 'not shown'}**.", ""]
        s = res["strategy"]
        L += ["## Strategy, indicative", "",
              f"GBP 1 backs where the combined p is worth more than {s['threshold']:.2f} a unit after "
              f"{int(COMMISSION * 100)}% commission, at the close: {s['bets']:,} bets, P&L GBP "
              f"{s['pnl_after_commission']:.2f}" + (f", ROI {s['roi']:+.3f}" if s["roi"] is not None else "") + ".",
              "Filled at the last traded price, which BASIC cannot show was there to take: an upper bound.", "",
              "| year | P&L GBP |", "|---|---:|", *[f"| {y} | {v:.2f} |" for y, v in s["by_year"].items()], ""]
    if counts["unmatched_sample"]:
        L += ["## Unmatched markets (first 20)", "", "| market | event | time | why |", "|---|---|---|---|"]
        L += [f"| {u['market_id']} | {u['event_name']} | {u['market_time']} | {_why(u)} |"
              for u in counts["unmatched_sample"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--out", default="reports/cricket")
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    betfair, keys = load_betfair(store)
    res = counts = None
    if betfair is not None:
        pre = store.get_parquet("cricket/eval/prematch.parquet")
        if pre is None:
            raise SystemExit(f"no data found: {store.where('cricket/eval/prematch.parquet')} (cricket.model.prematch)")
        j, counts = match_markets(market_table(betfair), pre)
        res = benter(j, a.threshold) if len(j) >= MIN_MATCHED else None
    md = report_md(res, counts, keys, store)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "benter_prematch.md").write_text(md)
    store.put("cricket/reports/benter_prematch.md", md.encode())
    print(md)
    print(json.dumps({"betfair_tables": keys, "counts": counts and {k: v for k, v in counts.items()
                                                                    if k != "unmatched_sample"}}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
