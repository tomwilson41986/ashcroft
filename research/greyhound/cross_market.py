"""The WIN book against the PLACE book (reports/greyhound_roadmap.md M5; the owner, 4 Nov 2026: recommendation 4).

A dog priced differently in two markets of the same race is a dislocation that needs no form model. From the WIN
book at a mark (first traded, 5, 3 and 1 minutes out) each dog's chance of a place under Harville (the next
finisher drawn in proportion to the win chances of those left; research/arms/place.py) is compared with the PLACE
book's own chance at the same moment: ``disloc = log(q_place_market / q_place_harville)``. A PLACE book longer than
the WIN book implies (disloc < 0) says the place money rates the dog below its win price, and the WIN price should
drift; and the other way.

Scored walk-forward by day on the recorder's GB greyhound books (WIN and PLACE recorded since 6 Oct 2026):

- **The move to the close:** log(BSP / mid) regressed on the dislocation (and on the dislocation beside the move
  already in the book, the last trade against the mid), fitted on earlier days; the RMSE of log(BSP) against the
  mid's and the correlation of the dislocation with the move.
- **The result (Benter's test):** a conditional logit on the race's log market chance, with and without the
  dislocation, fitted on earlier days; the gain in log-loss over the market alone.

    python -m research.greyhound.cross_market --from 2026-10-06 --out out/greyhound_cross_market
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

log = logging.getLogger(__name__)

MARKS = ("first_traded", "off_5", "off_3", "off_1")
MIN_TRAIN_DAYS = 7


def _log(msg: str) -> None:
    log.info(msg)
    print(msg, flush=True)


def harville_place(p, k: int) -> np.ndarray:
    from research.arms.place import harville_place as hp
    return hp(p, k)


def pair_markets(markets: pd.DataFrame) -> pd.DataFrame:
    """Each WIN market's PLACE market of the same race (the same event and off), one row a WIN market."""
    m = markets.drop_duplicates("market_id").copy()
    m["market_id"] = m.market_id.astype(str)
    win = m[m.market_type.astype(str) == "WIN"]
    plc = m[m.market_type.astype(str) == "PLACE"]
    key = ["event_id", "market_start_utc"]
    j = win[key + ["market_id"]].merge(plc[key + ["market_id"]].rename(columns={"market_id": "place_market_id"}),
                                        on=key, how="inner")
    return j.drop_duplicates("market_id")[["market_id", "place_market_id"]]


def book_at_marks(books: pd.DataFrame, markets: pd.DataFrame, day: date) -> pd.DataFrame:
    """Each runner's WIN book at each mark with the PLACE book at the same moment (the PLACE runner's latest poll at
    or before the WIN mark's poll, within 90 s), the WIN BSP and the result."""
    from greyhound import tradeout as to
    pairs = pair_markets(markets)
    if not len(pairs):
        return pd.DataFrame()
    pre = to.pre_off_books(books, markets, retimed=True)
    if not len(pre):
        return pd.DataFrame()
    pre["market_id"] = pre.market_id.astype(str)
    win = pre[pre.market_id.isin(set(pairs.market_id))]
    plc = pre[pre.market_id.isin(set(pairs.place_market_id))].copy()
    e = to.entry_books(win)
    if not len(e):
        return pd.DataFrame()
    e = e.merge(pairs, on="market_id", how="inner")
    p = plc[["market_id", "selection_id", "polled", "back1", "lay1", "number_of_winners"]].rename(
        columns={"market_id": "place_market_id", "polled": "place_polled", "back1": "place_back1", "lay1": "place_lay1"})
    p["place_lay1"] = pd.to_numeric(p.place_lay1, errors="coerce")
    p["number_of_winners"] = pd.to_numeric(p.number_of_winners, errors="coerce")
    e = e.sort_values("polled")
    p = p.sort_values("place_polled")
    out = pd.merge_asof(e, p, left_on="polled", right_on="place_polled", by=["place_market_id", "selection_id"],
                        direction="backward", tolerance=pd.Timedelta(seconds=90))
    bsp = to.closing_prices(books, markets, day)
    out = out.merge(bsp, on=["market_id", "selection_id"], how="left")
    final = books[(books.source.astype(str) == "final") & (books.status.astype(str) == "CLOSED")]
    won = final.assign(market_id=final.market_id.astype(str), selection_id=to._num(final.selection_id)) \
        .groupby(["market_id", "selection_id"]).runner_status.last().rename("runner_status").reset_index()
    out = out.merge(won, on=["market_id", "selection_id"], how="left")
    out["won"] = (out.runner_status.astype(str) == "WINNER").astype(int)
    out["race_date"] = str(day)
    return out


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Per runner at a mark: the WIN mid's chance normalised in the race, the Harville place chance from it, the
    PLACE book's chance normalised to the places paid, and the dislocation between them."""
    from model.race_book import market_now
    x = d.copy()
    x["mid"] = market_now(x.back1.to_numpy(float), x.lay1.where(x.lay1 > 1).to_numpy(float))
    x["place_mid"] = market_now(x.place_back1.to_numpy(float), x.place_lay1.where(x.place_lay1 > 1).to_numpy(float))
    x = x[(x["mid"] > 1) & (x.place_mid > 1)]
    g = x.race_id.astype(str) + "|" + x["mark"].astype(str) if "race_id" in x else x.market_id.astype(str) + "|" + x["mark"].astype(str)
    x["grp"] = g
    whole = x.groupby("grp").selection_id.transform("size")
    x["q_win"] = (1.0 / x["mid"]) / (1.0 / x["mid"]).groupby(x.grp).transform("sum")
    k = x.number_of_winners.fillna(2).clip(lower=1)
    x["k"] = k
    qp = (1.0 / x.place_mid)
    x["q_place_mkt"] = qp / qp.groupby(x.grp).transform("sum") * k
    hv = np.empty(len(x))
    for key, idx in x.groupby("grp").indices.items():
        kk = int(k.iloc[idx[0]])
        hv[idx] = harville_place(x.q_win.to_numpy()[idx], kk) if len(idx) > kk else 1.0
    x["q_place_hv"] = hv
    x["disloc"] = np.log(x.q_place_mkt.clip(1e-6, 1.0) / x.q_place_hv.clip(1e-6, 1.0))
    last = pd.to_numeric(x.last_traded, errors="coerce")
    x["log_last_vs_mid"] = np.log(last.where(last > 1, x["mid"]) / x["mid"])
    x["y"] = np.log(x.bsp.astype(float) / x["mid"])
    x["log_q_win"] = np.log(x.q_win)
    return x[x.bsp > 1].dropna(subset=["disloc", "y"])


def _fit_lin(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _pred_lin(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return beta[0] + X @ beta[1:]


def _cl_fit(X: np.ndarray, won: np.ndarray, grp: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """A conditional logit on the winner over the race's runners (the horses' race_book fit)."""
    from greyhound.tradeout import fit_clogit
    from model.race_book import race_codes
    g = race_codes(pd.Series(grp))
    return fit_clogit(X, won.astype(float), g, l2=l2)


def _cl_logloss(beta: np.ndarray, X: np.ndarray, won: np.ndarray, grp: np.ndarray) -> float:
    from model.race_book import group_log_softmax, race_codes
    g = race_codes(pd.Series(grp))
    lq = group_log_softmax(X @ beta, g)
    return float(-np.mean(lq[won == 1]))


def score(panel: pd.DataFrame, n_boot: int = 1000) -> list[dict]:
    """Walk-forward by day, by mark: the move to the close and the result, with and without the dislocation."""
    rng = np.random.default_rng(0)
    rows = []
    for mark, x in panel.groupby("mark"):
        days = sorted(x.race_date.unique())
        if len(days) <= MIN_TRAIN_DAYS:
            continue
        got = []
        for day in days[MIN_TRAIN_DAYS:]:
            tr, te = x[x.race_date < day], x[x.race_date == day]
            if not len(tr) or not len(te):
                continue
            b1 = _fit_lin(tr[["disloc"]].to_numpy(float), tr.y.to_numpy(float))
            b2 = _fit_lin(tr[["disloc", "log_last_vs_mid"]].to_numpy(float), tr.y.to_numpy(float))
            b0 = _fit_lin(tr[["log_last_vs_mid"]].to_numpy(float), tr.y.to_numpy(float))
            te = te.assign(p_mid=0.0, p_d=_pred_lin(b1, te[["disloc"]].to_numpy(float)),
                           p_last=_pred_lin(b0, te[["log_last_vs_mid"]].to_numpy(float)),
                           p_both=_pred_lin(b2, te[["disloc", "log_last_vs_mid"]].to_numpy(float)))
            # the result: the market's log chance alone, and with the dislocation
            try:
                X0, X1 = tr[["log_q_win"]].to_numpy(float), tr[["log_q_win", "disloc"]].to_numpy(float)
                c0 = _cl_fit(X0, tr.won.to_numpy(), tr.grp.to_numpy())
                c1 = _cl_fit(X1, tr.won.to_numpy(), tr.grp.to_numpy())
                te = te.assign(ll0=_cl_logloss(c0, te[["log_q_win"]].to_numpy(float), te.won.to_numpy(), te.grp.to_numpy()),
                               ll1=_cl_logloss(c1, te[["log_q_win", "disloc"]].to_numpy(float), te.won.to_numpy(),
                                               te.grp.to_numpy()), beta_d=float(c1[1]))
            except Exception as exc:                                    # a day too thin for the logit
                log.warning("%s %s: result fit failed (%s)", mark, day, exc)
                te = te.assign(ll0=np.nan, ll1=np.nan, beta_d=np.nan)
            got.append(te)
        if not got:
            continue
        f = pd.concat(got, ignore_index=True)
        y = f.y.to_numpy(float)
        def rmse(col):
            return float(np.sqrt(np.mean((f[col].to_numpy(float) - y) ** 2)))
        # paired by day: RMSE with the dislocation less the RMSE of the last-trade model, 90% interval
        days_ = f.race_date.to_numpy()
        ud = np.array(sorted(set(days_)))
        a = (f.p_both.to_numpy(float) - y) ** 2
        b = (f.p_last.to_numpy(float) - y) ** 2
        sa = pd.Series(a).groupby(days_).sum().reindex(ud).to_numpy()
        sb = pd.Series(b).groupby(days_).sum().reindex(ud).to_numpy()
        n = pd.Series(a).groupby(days_).size().reindex(ud).to_numpy()
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(ud), len(ud))
            boots.append(np.sqrt(sa[i].sum() / n[i].sum()) - np.sqrt(sb[i].sum() / n[i].sum()))
        res_days = f.dropna(subset=["ll0"]).groupby("race_date")[["ll0", "ll1"]].first()
        rows.append({"mark": mark, "runners": int(len(f)), "races": int(f.grp.nunique()), "days": int(len(ud)),
                     "corr_disloc_move": round(float(np.corrcoef(f.disloc, y)[0, 1]), 4),
                     "rmse_mid": round(rmse("p_mid"), 4), "rmse_disloc": round(rmse("p_d"), 4),
                     "rmse_last_trade": round(rmse("p_last"), 4), "rmse_both": round(rmse("p_both"), 4),
                     "both_vs_last_90%": [round(float(np.quantile(boots, 0.05)), 4),
                                          round(float(np.quantile(boots, 0.95)), 4)],
                     "result_logloss_market": round(float(res_days.ll0.mean()), 5) if len(res_days) else None,
                     "result_logloss_with_disloc": round(float(res_days.ll1.mean()), 5) if len(res_days) else None,
                     "beta_disloc_last_fit": round(float(f.beta_d.dropna().iloc[-1]), 4) if f.beta_d.notna().any()
                     else None,
                     "mean_disloc": round(float(f.disloc.mean()), 4), "sd_disloc": round(float(f.disloc.std()), 4)})
    return rows


def report_md(res: dict) -> str:
    L = [f"# The WIN book against the PLACE book ({res['first_day']} to {res['last_day']})", "",
         f"Recorder's GB greyhound books, re-timed; {res.get('rows')} runner-marks over {res.get('days')} days, "
         f"walk-forward by day (each day scored from the days before it only). The dislocation is log(PLACE book's "
         f"chance / Harville place chance from the WIN book): negative where the place money rates the dog below its "
         f"win price. The move is log(BSP / mid). Lower RMSE is better; 'both vs last' is the paired difference by "
         f"day with its 90% interval (negative: the dislocation helps beyond the last trade).", "",
         "| Mark | Runners | Races | Days | corr(disloc, move) | RMSE mid | RMSE disloc | RMSE last trade | RMSE both "
         "| both vs last (90%) | Result log-loss: market | with disloc | beta on disloc | disloc mean / sd |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {m: i for i, m in enumerate(MARKS)}
    for r in sorted(res.get("scores", []), key=lambda r: order.get(r["mark"], 9)):
        L.append(f"| {r['mark']} | {r['runners']} | {r['races']} | {r['days']} | {r['corr_disloc_move']} | "
                 f"{r['rmse_mid']} | {r['rmse_disloc']} | {r['rmse_last_trade']} | {r['rmse_both']} | "
                 f"{r['both_vs_last_90%']} | {r['result_logloss_market']} | {r['result_logloss_with_disloc']} | "
                 f"{r['beta_disloc_last_fit']} | {r['mean_disloc']} / {r['sd_disloc']} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from betfair_recorder import _read_day_csv, uk_today
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="first", default="2026-10-06")
    ap.add_argument("--to", dest="last", default=None, help="default: yesterday (UK)")
    ap.add_argument("--out", default="out/greyhound_cross_market")
    a = ap.parse_args(argv)
    t0 = time.time()
    first = date.fromisoformat(a.first)
    last = date.fromisoformat(a.last) if a.last else uk_today() - timedelta(days=1)
    frames, d = [], first
    while d <= last:
        try:
            books = _read_day_csv(d, "books_greyhound.csv")
            markets = _read_day_csv(d, "markets_greyhound.csv")
            if books is None or markets is None:
                _log(f"{d}: no data")
            else:
                x = book_at_marks(books, markets, d)
                if len(x):
                    frames.append(features(x))
                _log(f"{d}: {len(x)} runner-marks")
        except Exception as exc:
            _log(f"{d}: failed ({exc})")
        d += timedelta(days=1)
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    res = {"first_day": str(first), "last_day": str(last), "rows": int(len(panel)),
           "days": int(panel.race_date.nunique()) if len(panel) else 0}
    res["scores"] = score(panel[panel["mark"].isin(MARKS)]) if len(panel) else []
    res["seconds"] = round(time.time() - t0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    md = report_md(res)
    out.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
