"""The trade-to-SP rule checked against real books (read-only research). research/greyhound/historic_tradeout.py found
lays at 1 minute out, where the forecast close (from both markets' last traded prices, fitted on Betfair's historic
data Jan-Apr 2026) is 5-10% above the price, profitable May-Sep when filled a tick or two through the last trade
(run 41896084511). The worry: the model may be reading last traded prices that are stale, which no one would fill.

Here the same forecast (fitted on the historic data's development months only, so every recorded day is out of
sample) is applied to the recorded GB books at ``SECS`` before the scheduled off: each runner's last traded price and
the place market's at that moment, at 15 minutes out where the books reach back that far; then a lay is entered at
the book's real best price to lay (``lay1``, with at least ``MIN_SIZE`` on offer) and a back at ``back1``, closed at
the SP: lay 1 - L/S, back B/S - 1, commission on a gain. Beside it, on the same rows, the fill the historic study
assumed (a tick through the last trade), to measure the gap. Books: the live key's (every 5 s, from 2 Nov) where a
day has them, else the recorder's (re-timed, about 90 s stale).
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from research.greyhound import historic_place as hp, historic_tradeout as ht, limit_study as ls, place_live as pl
from research.greyhound import place_value as pv

log = logging.getLogger(__name__)

SECS = 60
MIN_SIZE = 2.0
MARGINS = (0.02, 0.05, 0.10)
CAPS = (5.0, 10.0, 20.0)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def at(b: pd.DataFrame, offs: pd.Series, secs: float, cols: list[str]) -> pd.DataFrame:
    """Each runner's last open pre-play book at least ``secs`` before the scheduled off, with its age."""
    x = b.copy()
    x["polled"] = pd.to_datetime(x.polled_utc, utc=True, errors="coerce")
    x["off"] = pd.to_datetime(x.market_id.map(offs), utc=True, errors="coerce")
    ok = ((x.source.astype(str) != "final") & (x.status.astype(str) == "OPEN")
          & ~x.inplay.astype(str).str.lower().isin(["true", "1"])
          & (x.runner_status.fillna("ACTIVE").astype(str) == "ACTIVE")
          & (x.polled <= x.off - pd.Timedelta(seconds=secs)))
    x = x[ok].sort_values("polled").drop_duplicates(["market_id", "selection_id"], keep="last")
    x["age_s"] = (x.off - pd.Timedelta(seconds=secs) - x.polled).dt.total_seconds()
    for c in cols:
        x[c] = _num(x[c])
    return x[["market_id", "selection_id", "age_s"] + cols]


def day_rows(day: date, root=None) -> pd.DataFrame:
    from betfair_recorder import _read_day_csv
    out = None
    for bname, mname, label in pl.FEEDS:
        b, m = _read_day_csv(day, bname, root=root), _read_day_csv(day, mname, root=root)
        if b is None or m is None or not len(b):
            continue
        out = (b, m, label)
        break
    if out is None:
        return pd.DataFrame()
    b, m, feed = out
    b = b.assign(market_id=b.market_id.astype(str), selection_id=_num(b.selection_id))
    m = m.assign(market_id=m.market_id.astype(str), selection_id=_num(m.selection_id))
    mk = m.drop_duplicates(["market_id", "selection_id"])
    mk = mk[mk.country.astype(str).isin(["GB", "IE"])] if "country" in mk else mk
    offs = mk.drop_duplicates("market_id").set_index("market_id").market_start_utc
    mk = mk.assign(race=mk.event_id.astype(str) + "|" + mk.market_start_utc.astype(str),
                   name=mk.runner_name.astype(str).str.replace(r"^\d+\.\s*", "", regex=True).str.strip().str.lower())
    win = mk[mk.market_type.astype(str) == "WIN"][["race", "name", "market_id", "selection_id", "country"]]
    plc = mk[mk.market_type.astype(str) == "PLACE"][["race", "name", "market_id", "selection_id"]]
    st = pl.settle(b)
    now = at(b, offs, SECS, ["last_traded", "back1", "back1_size", "lay1", "lay1_size"])
    early = at(b, offs, 15 * 60, ["last_traded"]).rename(columns={"last_traded": "lt15", "age_s": "age15"})
    x = win.merge(now, on=["market_id", "selection_id"])
    x = x.merge(early[early.age15 < 120][["market_id", "selection_id", "lt15"]], on=["market_id", "selection_id"],
                how="left")
    x = x.merge(st[["market_id", "selection_id", "sp"]], on=["market_id", "selection_id"], how="left")
    if len(plc):
        pn = at(b, offs, SECS, ["last_traded"]).rename(columns={"last_traded": "ltp_t_1_p", "age_s": "age_p"})
        p = plc.merge(pn, on=["market_id", "selection_id"]).merge(
            st[["market_id", "selection_id", "places_paid"]], on=["market_id", "selection_id"], how="left")
        x = x.merge(p[["race", "name", "ltp_t_1_p", "places_paid"]], on=["race", "name"], how="left")
    else:
        x["ltp_t_1_p"], x["places_paid"] = np.nan, np.nan
    g = x.groupby("race")
    x["n"] = g.name.transform("size")
    x["k"] = pd.to_numeric(x.places_paid, errors="coerce").fillna(2)      # GB greyhound place markets pay 2
    x = x.rename(columns={"last_traded": "ltp_t_1_w", "lt15": "ltp_t_15_w"})
    x["ltp_t_60_w"], x["ltp_t_60_p"], x["ltp_t_15_p"] = np.nan, np.nan, np.nan
    x["country"] = x.country.fillna("GB").replace({"IE": "GB"})
    return x.assign(day=f"{day}", feed=feed)


def score(x: pd.DataFrame) -> dict:
    x = x[(x.sp > 1) & (x.ltp_t_1_w > 1)].copy()
    sp, f = x.sp.to_numpy(float), x.f_bsp.to_numpy(float)
    lay1, back1 = x.lay1.to_numpy(float), x.back1.to_numpy(float)
    lay_tick = ls.ticks_from(x.ltp_t_1_w.to_numpy(float), 1, up=True)
    back_tick = ls.ticks_from(x.ltp_t_1_w.to_numpy(float), 1, up=False)
    out = {"runners": len(x), "races": int(x.race.nunique()), "days": sorted(x.day.unique()),
           "feeds": x.drop_duplicates("race").feed.value_counts().to_dict(),
           "median_book_age_s": float(x.age_s.median()),
           "rmse_log_sp": {"ltp": round(float(np.sqrt(np.mean(np.log(sp / x.ltp_t_1_w) ** 2))), 4),
                           "forecast": round(float(np.sqrt(np.mean(np.log(sp / f) ** 2))), 4)},
           "rules": []}
    for side in ("lay", "back"):
        for fill, entry, size in (("book", lay1 if side == "lay" else back1,
                                   x.lay1_size.to_numpy(float) if side == "lay" else x.back1_size.to_numpy(float)),
                                  ("tick_through_ltp", lay_tick if side == "lay" else back_tick, None)):
            edge = (1 - entry / f) if side == "lay" else (entry / f - 1)
            raw = (1 - entry / sp) if side == "lay" else (entry / sp - 1)
            pnl = np.where(raw > 0, raw * (1 - ht.COMM), raw)
            for cap in CAPS:
                for m in MARGINS:
                    sel = np.isfinite(edge) & (edge > m) & (entry > 1) & (entry <= cap)
                    if size is not None:
                        sel &= size >= MIN_SIZE
                    row = {"side": side, "fill": fill, "max_entry": cap, "margin": m, "trades": int(sel.sum())}
                    if sel.sum() >= 20:
                        t = pd.DataFrame({"day": x.day.to_numpy()[sel], "pnl": pnl[sel]})
                        byd = t.groupby("day").agg(pnl=("pnl", "sum"), n=("pnl", "size"))
                        lo, hi = pv._boot(byd.pnl, byd.n)
                        row.update({"mean_entry": round(float(entry[sel].mean()), 2),
                                    "mean_pnl%": round(100 * float(t.pnl.mean()), 3),
                                    "ci90%": [round(100 * lo, 3), round(100 * hi, 3)]})
                    out["rules"].append(row)
    return out


def run(store, first: date, last: date, root=None) -> dict:
    j = hp.pair(hp.load(store))
    fx = ht.forecast(j)
    model, feats = fx.attrs["model"], fx.attrs["feats"]
    parts, d = [], first
    while d <= last:
        try:
            r = day_rows(d, root)
        except Exception as exc:
            log.warning("%s: %s", d, exc)
            r = pd.DataFrame()
        if len(r):
            parts.append(r)
        d += timedelta(days=1)
    if not parts:
        return {"note": "no data found: no recorded books"}
    x = ht.features_from(pd.concat(parts, ignore_index=True))
    for c in feats:
        if c not in x:
            x[c] = np.nan
    x["f_bsp"] = np.exp(x.l_t_1_w + model.predict(x[feats].astype(float)))
    return score(x)


if __name__ == "__main__":
    from sources.common import Store
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(Store(), date(2026, 10, 6), date.today() - timedelta(days=1)), indent=1, default=str))
