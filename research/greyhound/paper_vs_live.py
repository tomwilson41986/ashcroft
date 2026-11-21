"""Why the paper early lay and the live early lay disagree (the owner, 20 Nov 2026: "yes" to the open question of the
discovery round; reports/greyhound_discovery_1120.md, builds 1 and 4; ledger greyhound-paper-vs-live-1120).
Descriptive, 0 trials; read-only.

The paper forward test's lays read +1.31% of liability a day at the SP under its own stream replay; the live early
lay's lays, closed at the SP, were flat to losing on 11-14 Nov. The fill model is now checked against the live orders
(greyhound-queue-calibration-1120: the queue as shown, trade-through capped, half the printed volume). So the gap is
either the fill setting the paper figure used (full volume, uncapped) or the ORDERS: which dogs, at what price, when.

Per live day, three order sets through the same recorded stream:

- the paper rule's lays (research/greyhound/stream_lab.py ``forward_test_orders``) under the paper figure's own
  setting (full volume, uncapped): reproduces the reported figure;
- the same paper lays under the calibrated setting;
- the live lays (the ledger) under the calibrated setting, with what live really gave them beside it.

Then the paper and live lays are joined on (market, dog): common, paper only, live only; for the common ones the
difference in limit price and in the time the order went in. Every figure is P&L closed at the SP before commission
and its return on the matched liability.

    python -m research.greyhound.paper_vs_live --days 2026-11-11,2026-11-12,2026-11-13,2026-11-14 --out out/study
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "greyhound-paper-vs-live-1120"
CALIBRATED = {"vol_factor": 0.5, "through_cap": True}
PAPER_OWN = {"vol_factor": 1.0, "through_cap": False}


def replay_set(lines, orders: list, setting: dict, close_secs: float) -> pd.DataFrame:
    """Orders through the stream under a fill setting: one row an order with its matched stake, average price, SP,
    liability and P&L closed at the SP (before commission)."""
    from research.greyhound import stream_replay as sr
    if not orders or not lines:
        return pd.DataFrame(columns=["market", "sid", "price", "size", "t", "placed", "matched", "avg", "sp",
                                     "liability", "pnl_sp"])
    sims = [sr.SimOrder(market=o.market, sid=o.sid, side="LAY", price=o.price, size=o.size, t=o.t, tag=str(i))
            for i, o in enumerate(orders)]
    df = sr.replay(lines, sims, close_secs=close_secs, comm=0.0, **setting)
    ix = df.tag.astype(int)
    m, a, sp = pd.to_numeric(df.matched, errors="coerce").fillna(0.0), pd.to_numeric(df.avg, errors="coerce"), \
        pd.to_numeric(df.sp, errors="coerce")
    ok = (m > 0) & (sp > 1)
    return pd.DataFrame({"market": df.market.astype(str), "sid": df.sid.astype(int), "price": df.price.astype(float),
                         "size": df["size"].astype(float), "t": ix.map(lambda k: orders[k].t),
                         "placed": df.placed.astype(bool), "matched": m, "avg": a, "sp": sp,
                         "liability": (m * (a - 1)).where(ok, 0.0), "pnl_sp": (m * (1 - a / sp)).where(ok, 0.0)})


def tot(d: pd.DataFrame) -> dict:
    """Orders, fills, liability, P&L at the SP and the return on the matched liability."""
    if not len(d):
        return {"orders": 0, "matched_orders": 0, "matched_liability": 0.0, "pnl_at_sp": 0.0, "roi_on_liability_%": None}
    liab, pnl = float(d.liability.sum()), float(d.pnl_sp.sum())
    return {"orders": int(len(d)), "matched_orders": int((d.matched > 0).sum()),
            "fill_rate": round(float((d.matched > 0).mean()), 4), "matched_liability": round(liab, 2),
            "pnl_at_sp": round(pnl, 2), "roi_on_liability_%": round(100 * pnl / liab, 3) if liab > 0 else None,
            "mean_log_sp_over_price_filled": round(float(np.log(d.sp / d.avg)[d.matched > 0].mean()), 4)
            if (d.matched > 0).any() else None}


def live_actual(o: pd.DataFrame) -> pd.DataFrame:
    """The live lays as the ledger settled them, in the same columns (the SP from the books or the stream)."""
    m, a, sp = o.live_matched.astype(float), pd.to_numeric(o.live_avg, errors="coerce"), pd.to_numeric(o.sp, errors="coerce")
    ok = (m > 0) & (sp > 1)
    return pd.DataFrame({"market": o.market_id.astype(str), "sid": o.selection_id.astype(int), "price": o.price,
                         "size": o["size"], "t": o.placed_utc, "placed": True, "matched": m, "avg": a, "sp": sp,
                         "liability": (m * (a - 1)).where(ok, 0.0), "pnl_sp": (m * (1 - a / sp)).where(ok, 0.0)})


def compare(paper: pd.DataFrame, live: pd.DataFrame) -> dict:
    """The paper and live lays joined on (market, dog), both under the calibrated fill: common, paper only, live
    only; for the common ones the gap in limit price and in the order's time."""
    k = ["market", "sid"]
    j = paper.merge(live, on=k, how="outer", suffixes=("_p", "_l"), indicator=True)
    both, po, lo = j[j._merge == "both"], j[j._merge == "left_only"], j[j._merge == "right_only"]
    side = lambda d, s: d[[c for c in d if c.endswith(s)]].rename(columns=lambda c: c[:-2])
    out = {"common_as_paper": tot(side(both, "_p")), "common_as_live": tot(side(both, "_l")),
           "paper_only": tot(side(po, "_p")), "live_only": tot(side(lo, "_l"))}
    if len(both):
        dt = (pd.to_datetime(both.t_l, utc=True) - pd.to_datetime(both.t_p, utc=True)).dt.total_seconds()
        ratio = both.price_l / both.price_p
        out["common_orders"] = {
            "n": int(len(both)), "live_price_over_paper_price_median": round(float(ratio.median()), 4),
            "live_price_higher_share": round(float((ratio > 1.0001).mean()), 4),
            "live_price_lower_share": round(float((ratio < 0.9999).mean()), 4),
            "live_seconds_after_paper_median": round(float(dt.median()), 1),
            "live_seconds_after_paper_10%_90%": [round(float(dt.quantile(0.1)), 1), round(float(dt.quantile(0.9)), 1)]}
    return out


def run_day(store, day: date, loader=None) -> dict:
    from greyhound import early_lay as el
    from research.greyhound import fill_recon as fr, stream_lab as sl
    gs = el.CONFIG["green_seconds"]
    data = (loader or (lambda d: fr.load_day(store, d)))(day)
    led = data.get("ledger")
    out = {"day": str(day)}
    if led is None or not len(led):
        return {**out, "note": "no data found: no live ledger"}
    o = fr.live_orders(led, data.get("final"), data.get("offs"), gs)
    paper = data["paper_orders"] if "paper_orders" in data else sl.forward_test_orders(store, day)
    if not paper:
        return {**out, "note": "no data found: no paper lays for the day"}
    mids = set(o.market_id.astype(str)) | {p.market for p in paper}
    lines = data["stream_all"] if "stream_all" in data else fr.stream_lines(day, mids)
    if not lines:
        return {**out, "note": "no data found: no stream lines"}
    from research.greyhound import stream_replay as sr
    live_sims = [sr.SimOrder(market=r.market_id, sid=int(r.selection_id), side="LAY", price=float(r.price),
                             size=float(r.size), t=r.placed_utc.to_pydatetime()) for r in o.itertuples(index=False)]
    p_own, p_cal = replay_set(lines, paper, PAPER_OWN, gs), replay_set(lines, paper, CALIBRATED, gs)
    l_cal, l_act = replay_set(lines, live_sims, CALIBRATED, gs), live_actual(o)
    stream_sp = l_cal.drop_duplicates(["market", "sid"]).set_index(["market", "sid"]).sp.reindex(
        pd.MultiIndex.from_frame(l_act[["market", "sid"]])).to_numpy()
    l_act["sp"] = np.where(l_act.sp.notna(), l_act.sp, stream_sp)       # the books' SP, else the stream's
    ok = (l_act.matched > 0) & (l_act.sp > 1)
    l_act["liability"], l_act["pnl_sp"] = (l_act.matched * (l_act.avg - 1)).where(ok, 0.0), \
        (l_act.matched * (1 - l_act.avg / l_act.sp)).where(ok, 0.0)
    out.update({"paper_own_setting": tot(p_own), "paper_calibrated": tot(p_cal), "live_calibrated": tot(l_cal),
                "live_actual": tot(l_act), "sets": compare(p_cal, l_cal),
                "markets": {"paper": len({p.market for p in paper}), "live": int(o.market_id.nunique()),
                            "both": len({p.market for p in paper} & set(o.market_id.astype(str)))}})
    out["_frames"] = {"paper_own": p_own, "paper_cal": p_cal, "live_cal": l_cal, "live_act": l_act}
    return out


def pooled(days: list[dict]) -> dict:
    fr_ = [d["_frames"] for d in days if "_frames" in d]
    if not fr_:
        return {"note": "no data found: no day with paper lays, live lays and a stream record"}
    cat = lambda k: pd.concat([f[k] for f in fr_], ignore_index=True)
    return {"days": len(fr_), "paper_own_setting": tot(cat("paper_own")), "paper_calibrated": tot(cat("paper_cal")),
            "live_calibrated": tot(cat("live_cal")), "live_actual": tot(cat("live_act")),
            "sets": compare(cat("paper_cal"), cat("live_cal"))}


def markdown(res: dict) -> str:
    md = [f"## The paper early lay against the live early lay ({TAG})", "",
          "P&L closed at the SP before commission (GBP); return on the matched liability.", ""]
    p = res.get("pooled", {})
    if p.get("note"):
        return "\n".join(md + [p["note"]])
    row = lambda name, t: (f"| {name} | {t['orders']} | {t['matched_orders']} | {t.get('fill_rate')} | "
                           f"{t['matched_liability']:,.2f} | {t['pnl_at_sp']:+,.2f} | {t['roi_on_liability_%']} | "
                           f"{t.get('mean_log_sp_over_price_filled')} |")
    head = ["| Lays | Orders | Matched | Fill rate | Matched liability | P&L at the SP | Return on liability % | "
            "Mean log(SP / price), filled |", "|---|---|---|---|---|---|---|---|"]
    md += [f"### Pooled, {p['days']} days", ""] + head + [
        row("Paper lays, the paper figure's own fill (full volume, uncapped)", p["paper_own_setting"]),
        row("Paper lays, the calibrated fill", p["paper_calibrated"]),
        row("Live lays, the calibrated fill", p["live_calibrated"]),
        row("Live lays, as the ledger settled them", p["live_actual"]), "",
        "Paper and live lays joined on (market, dog), calibrated fill:", ""] + head + [
        row("Common, at the paper's price and time", p["sets"]["common_as_paper"]),
        row("Common, at live's price and time", p["sets"]["common_as_live"]),
        row("Paper only", p["sets"]["paper_only"]), row("Live only", p["sets"]["live_only"]), ""]
    if p["sets"].get("common_orders"):
        md += ["Common orders: " + json.dumps(p["sets"]["common_orders"]), ""]
    for d in res["days"]:
        if d.get("note"):
            md += [f"- {d['day']}: {d['note']}"]
            continue
        md += [f"### {d['day']} (markets: {d['markets']})", ""] + head + [
            row("Paper, own fill", d["paper_own_setting"]), row("Paper, calibrated", d["paper_calibrated"]),
            row("Live, calibrated", d["live_calibrated"]), row("Live, ledger", d["live_actual"]),
            row("Common as paper", d["sets"]["common_as_paper"]), row("Common as live", d["sets"]["common_as_live"]),
            row("Paper only", d["sets"]["paper_only"]), row("Live only", d["sets"]["live_only"]), ""]
    return "\n".join(md)


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", default="2026-11-11,2026-11-12,2026-11-13,2026-11-14")
    ap.add_argument("--out", default="out/paper_vs_live")
    a = ap.parse_args(argv)
    store = Store()
    days = []
    for s in a.days.split(","):
        try:
            d = run_day(store, date.fromisoformat(s))
        except Exception as exc:
            log.exception("%s failed", s)
            d = {"day": s, "note": f"failed: {str(exc)[:200]}"}
        log.info("%s: %s", s, {k: v for k, v in d.items() if k != "_frames"})
        days.append(d)
    res = {"tag": TAG, "pooled": pooled(days), "days": [{k: v for k, v in d.items() if k != "_frames"} for d in days]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "report.md").write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
