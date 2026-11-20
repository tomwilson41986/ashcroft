"""The maker markout census: what every resting order that actually traded earned to the SP, with no queue model
(the owner, 20 Nov 2026: the discovery round; idea maker-08, reports/greyhound_discovery_1120.md). Descriptive, 0
trials: any cell chosen from it is registered afterwards with its cell count and read on forward days.

Every fill model the greyhound tests use is an assumption about WHICH resting orders trade. This reads the answer off
the recorded Exchange Stream instead: each increase in a runner's traded volume at a price is a trade, and the book
just before it says who was resting. A trade at or under the best price on offer to backers took a resting LAY (the
maker laid at that price); at or over the best price on offer to layers it took a resting BACK. The maker's return
closed at the Betfair SP, a unit of stake: a resting lay at L earns 1 - L / SP, a resting back at L earns L / SP - 1.
Volume-weighted over every trade, by seconds to the scheduled off, price band, the spread before the trade and
market type (WIN, PLACE), with a by-day interval; gross, and net of 2% on each gain (greyhound/costs.py).

Kept honest by a reconciliation: the volume first seen plus every increment must equal the runner's final traded
volume. Increments in a full-image message (a resubscription: the book before it is not known) and those inside the
spread are counted unsigned and shown, never dropped silently.

    python -m research.greyhound.maker_census --from 2026-11-05 --out out/study
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "greyhound-maker-census-1120"
FIRST_DAY = date(2026, 11, 5)
T_BUCKETS = ((900, "900+"), (300, "900-300"), (120, "300-120"), (60, "120-60"), (30, "60-30"), (10, "30-10"),
             (0, "10-0"))                                            # seconds to the scheduled off, lower edge
BANDS = ((2.0, "under 2"), (3.0, "2-3"), (5.0, "3-5"), (8.0, "5-8"), (15.0, "8-15"))
KEYS = ["mtype", "side", "t_bucket", "band", "spread"]


def t_bucket(to_off: float) -> str:
    if to_off < 0:
        return "after the off"
    for lo, name in T_BUCKETS:
        if to_off >= lo:
            return name
    return "10-0"


def band(price: float) -> str:
    for hi, name in BANDS:
        if price < hi:
            return name
    return "15+"


def spread_bucket(bb, bl) -> str:
    from research.micro.events import tick_index
    if not bb or not bl:
        return "one side empty"
    k = round(tick_index(bl) - tick_index(bb))
    return "1 tick" if k <= 1 else "2 ticks" if k == 2 else "3-4 ticks" if k <= 4 else "5+ ticks"


def day_trades(lines) -> tuple[pd.DataFrame, dict]:
    """One row per (market, runner, side, bucket, price): the volume that traded against resting orders there, with
    the runner's SP; and the reconciliation of the day."""
    from research.greyhound import stream_replay as sr
    markets: dict = {}
    last_trd: dict = {}
    vol = defaultdict(float)                    # (mid, sid, side, t_bucket, spread, price) -> volume
    first = defaultdict(float)                  # (mid, sid) -> volume already traded when the runner was first seen
    inc = defaultdict(float)                    # (mid, sid) -> every increment since
    unsigned = defaultdict(float)               # reason -> volume
    for t, mc in sr.events(lines):
        mid = str(mc.get("id"))
        m0 = markets.get(mid)
        before = {}
        if m0 is not None:                      # the book before this message, for the runners it changes
            for rc in mc.get("rc", []) or []:
                r0 = m0.runners.get(int(rc["id"]))
                if r0 is not None:
                    before[int(rc["id"])] = (r0.best_back()[0], r0.best_lay()[0])
        image = bool(mc.get("img"))
        mid, changed = sr.apply(markets, mc)
        m = markets[mid]
        off = m.off
        to_off = (off - t).total_seconds() if off is not None else None
        live = (not m.inplay) and str(m.defn.get("status", "OPEN")) == "OPEN"
        for sid in changed:
            r, key = m.runners[sid], (mid, sid)
            prev = last_trd.get(key)
            last_trd[key] = dict(r.trd)
            if prev is None:
                first[key] = float(sum(r.trd.values()))
                continue
            bb, bl = before.get(sid, (None, None))
            for p, v in r.trd.items():
                d = v - prev.get(p, 0.0)
                if d <= 1e-9:
                    continue
                inc[key] += d
                if image:
                    unsigned["image"] += d
                elif not live or to_off is None:
                    unsigned["in play or not open"] += d
                elif bb is not None and p <= bb + 1e-9:
                    vol[(mid, sid, "resting lay", t_bucket(to_off), spread_bucket(bb, bl), p)] += d
                elif bl is not None and p >= bl - 1e-9:
                    vol[(mid, sid, "resting back", t_bucket(to_off), spread_bucket(bb, bl), p)] += d
                else:
                    unsigned["inside the spread or no book"] += d
    rows, no_sp = [], 0.0
    for (mid, sid, side, tb, spr, p), v in vol.items():
        m = markets[mid]
        sp = m.bsp(sid)
        if not sp or sp <= 1.0:
            no_sp += v
            continue
        rows.append({"market": mid, "sid": sid, "mtype": str(m.defn.get("marketType") or "?"),
                     "venue": m.defn.get("venue"), "side": side, "t_bucket": tb, "spread": spr, "price": p,
                     "band": band(p), "vol": v, "sp": sp,
                     "ret": (1.0 - p / sp) if side == "resting lay" else (p / sp - 1.0)})
    ok = tot = 0
    for key, f0 in first.items():
        r = markets[key[0]].runners.get(key[1])
        if r is None or not r.tv:
            continue
        tot += 1
        ok += abs(f0 + inc[key] - r.tv) <= 0.01 * max(r.tv, 1.0) + 0.5
    signed = float(sum(vol.values()))
    notes = {"runners_with_volume": tot, "runners_reconciled": int(ok),
             "reconciled_share": round(ok / tot, 4) if tot else None, "signed_volume": round(signed, 2),
             "unsigned_volume": {k: round(v, 2) for k, v in unsigned.items()},
             "signed_volume_without_sp": round(no_sp, 2)}
    return pd.DataFrame(rows), notes


def cells(tr: pd.DataFrame, day: str, comm: float) -> pd.DataFrame:
    """The day's sums a cell: volume, volume x return gross and net of commission on a gain, trades, markets."""
    if not len(tr):
        return pd.DataFrame()
    t = tr.assign(g=tr.vol * tr.ret, n=tr.vol * np.where(tr.ret > 0, tr.ret * (1 - comm), tr.ret))
    c = t.groupby(KEYS, as_index=False).agg(vol=("vol", "sum"), gross=("g", "sum"), net=("n", "sum"),
                                              trades=("vol", "size"), markets=("market", "nunique"))
    return c.assign(day=day)


def read(c: pd.DataFrame, by: list[str], min_days: int = 5) -> list[dict]:
    """Volume-weighted maker return to the SP by ``by``, with the 90% interval by day."""
    from model.race_book import ratio_interval
    out = []
    if not len(c):
        return out
    days = sorted(c.day.unique())
    for key, g in c.groupby(by):
        d = g.groupby("day")[["vol", "gross", "net"]].sum().reindex(days, fill_value=0.0)
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        v = float(d.vol.sum())
        row.update({"days": int((d.vol > 0).sum()), "trades": int(g.trades.sum()), "volume": round(v, 0),
                    "gross_%": round(100 * float(d.gross.sum()) / v, 3) if v else None,
                    "net_%": round(100 * float(d.net.sum()) / v, 3) if v else None})
        if row["days"] >= min_days:
            lo, hi = ratio_interval(d.gross.to_numpy(), d.vol.to_numpy(), n=2000)
            nlo, nhi = ratio_interval(d.net.to_numpy(), d.vol.to_numpy(), n=2000)
            row["gross_90%"] = [round(100 * lo, 3), round(100 * hi, 3)] if np.isfinite(lo) else None
            row["net_90%"] = [round(100 * nlo, 3), round(100 * nhi, 3)] if np.isfinite(nlo) else None
        out.append(row)
    return out


def report(c: pd.DataFrame, notes: dict) -> dict:
    res = {"tag": TAG, "days": sorted(c.day.unique().tolist()) if len(c) else [], "reconciliation": notes, "reads": {}}
    for name, by in (("all", ["mtype", "side"]), ("by seconds to the off", ["mtype", "side", "t_bucket"]),
                     ("by price band", ["mtype", "side", "band"]), ("by spread before the trade", ["mtype", "side", "spread"]),
                     ("by seconds and spread", ["mtype", "side", "t_bucket", "spread"])):
        res["reads"][name] = read(c, by)
    return res


def markdown(res: dict) -> str:
    md = [f"## Maker markout census ({res['tag']})", "",
          f"Days: {len(res['days'])} ({res['days'][0] if res['days'] else 'no data found'} to "
          f"{res['days'][-1] if res['days'] else ''}). Maker return closed at the SP, % of stake, volume-weighted; "
          "net = 2% off each gain. Descriptive: no cell is a rule.", ""]
    for name, rows in res["reads"].items():
        if not rows:
            continue
        ks = [k for k in rows[0] if k not in ("days", "trades", "volume", "gross_%", "net_%", "gross_90%", "net_90%")]
        md += [f"### {name}", "", "| " + " | ".join(ks) + " | Days | Trades | Volume | Gross % | 90% | Net % | 90% |",
               "|" + "---|" * (len(ks) + 7)]
        for r in rows:
            md.append("| " + " | ".join(str(r[k]) for k in ks) + f" | {r['days']} | {r['trades']} | {r['volume']:,.0f} | "
                      f"{r['gross_%']} | {r.get('gross_90%') or ''} | {r['net_%']} | {r.get('net_90%') or ''} |")
        md.append("")
    md += ["### Reconciliation", "", "```", json.dumps(res["reconciliation"], indent=1), "```"]
    return "\n".join(md)


def main(argv=None) -> int:
    from betfair_recorder import bucket, s3_client
    from greyhound.costs import commission
    from research.greyhound import stream_replay as sr
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="first", default=str(FIRST_DAY))
    ap.add_argument("--to", dest="last", default=None)
    ap.add_argument("--out", default="out/maker_census")
    a = ap.parse_args(argv)
    s3, bk = s3_client(), bucket()

    def listing(prefix):
        out, tok = [], None
        while True:
            r = s3.list_objects_v2(Bucket=bk, Prefix=prefix, **({"ContinuationToken": tok} if tok else {}))
            out += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return out
            tok = r["NextContinuationToken"]

    d, last = date.fromisoformat(a.first), date.fromisoformat(a.last) if a.last else date.today() - timedelta(days=1)
    comm, all_cells, tot = commission(), [], defaultdict(float)
    per_day = {}
    while d <= last:
        lines = sr.load_lines(lambda k: s3.get_object(Bucket=bk, Key=k)["Body"].read(), listing, d)
        if lines:
            tr, notes = day_trades(lines)
            per_day[str(d)] = notes
            log.info("%s: %d lines, %d signed rows, %s", d, len(lines), len(tr), notes)
            c = cells(tr, str(d), comm)
            if len(c):
                all_cells.append(c)
            for k in ("runners_with_volume", "runners_reconciled", "signed_volume", "signed_volume_without_sp"):
                tot[k] += notes[k]
            for k, v in notes["unsigned_volume"].items():
                tot[f"unsigned: {k}"] += v
        else:
            log.info("%s: no data found (no stream lines)", d)
        del lines
        d += timedelta(days=1)
    c = pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame()
    notes = {k: round(v, 2) for k, v in tot.items()}
    if tot["runners_with_volume"]:
        notes["reconciled_share"] = round(tot["runners_reconciled"] / tot["runners_with_volume"], 4)
    res = report(c, {"total": notes, "per_day": per_day}) if len(c) else \
        {"tag": TAG, "note": "no data found: no stream days", "days": [], "reads": {}, "reconciliation": notes}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "report.md").write_text(markdown(res))
    if len(c):
        c.to_csv(out / "cells.csv", index=False)
    print(markdown({**res, "reconciliation": notes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
