"""The early lay's green-up orders as Betfair settled them (read-only; the owner, 14 Nov 2026: "Read the green-up
orders from Betfair and fix the closing order").

For each live day, every settled bet of the early lay (listClearedOrders, SETTLED, one record a bet, the ``el-``
references) on the markets its ledger laid, joined by market and selection:

- the lays matched (stake ``S``, average price ``L``), the green-up backs (``-g``: stake ``B``, average price ``G``,
  how many bets, the price asked against the price matched) and any back at the SP (``-s``);
- the cover, ``B x G / (S x L)``: 1 is a flat green-up, above 1 backs more than the lay's value;
- Betfair's profit on the selection (before commission), the P&L of a flat green-up at the same back price
  (``S x (1 - L / G)``), and the difference (what the over- or under-cover gained or lost at the result);
- the ledger's locked P&L beside it (the worse of the two outcomes).

Each market's net after commission at the account's greyhound rate. No order method is called. Writes
out/green_audit/{bets.csv, selections.csv, report.json, report.md}.

    python -m research.greyhound.green_audit --from 2026-11-03 --to 2026-11-14
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

EPS = 1e-9


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def kind(ref: str, side: str) -> str:
    """A bet's part in the early lay from its reference (el-<mmdd>-<market>-<selection>-<kind>) and side."""
    ref = str(ref or "")
    if side == "BACK" and ref.endswith("-g"):
        return "green"
    if side == "BACK" and ref.endswith("-s"):
        return "sp_back"
    if side == "LAY":
        return "lay"
    return "other"


def bets_frame(cleared: list[dict], day: date, ref: str = "el-") -> pd.DataFrame:
    tag = f"{ref}{day:%m%d}-"
    rows = []
    for c in cleared:
        r = str(c.get("customerOrderRef") or "")
        if not r.startswith(tag):
            continue
        rows.append({"day": f"{day}", "market_id": str(c.get("marketId")), "selection_id": int(c.get("selectionId", 0)),
                     "bet_id": str(c.get("betId") or ""), "ref": r, "side": c.get("side"),
                     "kind": kind(r, c.get("side")), "order_type": c.get("orderType"),
                     "price_requested": c.get("priceRequested"), "price_matched": c.get("priceMatched"),
                     "size": c.get("sizeSettled"), "profit": c.get("profit"), "outcome": c.get("betOutcome"),
                     "placed": c.get("placedDate"), "last_matched": c.get("lastMatchedDate"),
                     "settled": c.get("settledDate")})
    return pd.DataFrame(rows)


def selections(bets: pd.DataFrame, ledger: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row a (market, selection) laid and matched: lays, green-up backs, cover, Betfair's profit, the flat
    green-up's P&L at the same back price, and the ledger's locked P&L."""
    if bets.empty:
        return pd.DataFrame()
    b = bets.copy()
    for c in ("price_requested", "price_matched", "size", "profit"):
        b[c] = _num(b[c])
    b["value"] = b["size"] * b.price_matched
    locked = {}
    if ledger is not None and len(ledger):
        led = ledger[ledger.action == "green"].copy()
        led["locked_pnl"] = _num(led.locked_pnl)
        led["selection_id"] = _num(led.selection_id)
        locked = led.groupby([led.market_id.astype(str), led.selection_id.astype(int)]).locked_pnl.sum().to_dict()
    rows = []
    for (day, mid, sid), g in b.groupby(["day", "market_id", "selection_id"]):
        lay, grn, spb = g[g.kind == "lay"], g[g.kind == "green"], g[g.kind == "sp_back"]
        S = float(lay["size"].sum())
        if S <= EPS:
            continue
        L = float(lay.value.sum() / S)
        B = float(grn["size"].sum())
        G = float(grn.value.sum() / B) if B > EPS else None
        won = None
        if len(lay):
            won = int((lay.outcome.astype(str) == "LOSE").any())   # a lay loses when the dog wins
        flat = S * (1 - L / G) if G else None
        cover = B * G / (S * L) if G else None
        profit = float(g.profit.sum())
        asked = grn.dropna(subset=["price_requested"])
        rows.append({"day": day, "market_id": mid, "selection_id": sid, "lay_stake": round(S, 2), "lay_price": round(L, 4),
                     "green_bets": len(grn), "green_stake": round(B, 2), "green_price": round(G, 4) if G else None,
                     "green_better_than_asked": round(float(((asked.price_matched - asked.price_requested)
                                                             * asked["size"]).sum() / asked["size"].sum()), 4)
                     if len(asked) and asked["size"].sum() > 0 else None,
                     "sp_back_stake": round(float(spb["size"].sum()), 2), "cover": round(cover, 4) if cover else None,
                     "won": won, "betfair_profit": round(profit, 2), "flat_pnl": round(flat, 2) if flat is not None else None,
                     "cover_effect": round(profit - flat, 2) if flat is not None and not len(spb) else None,
                     "ledger_locked": locked.get((str(mid), int(sid)))})
    return pd.DataFrame(rows)


def summary(sel: pd.DataFrame, bets: pd.DataFrame, comm: float) -> dict:
    if sel.empty:
        return {"note": "no data found: no settled early lay bets"}
    per_mkt = bets.assign(profit=_num(bets.profit)).groupby(["day", "market_id"]).profit.sum()
    net = float((per_mkt - comm * per_mkt.clip(lower=0)).sum())
    g = sel[sel.green_bets > 0]
    flat = sel.dropna(subset=["cover_effect"])
    out = {"selections": len(sel), "lay_stake": round(float(sel.lay_stake.sum()), 2),
           "betfair_profit_gross": round(float(sel.betfair_profit.sum()), 2), "betfair_profit_net": round(net, 2),
           "commission": comm, "greened": len(g), "sp_or_held": int((sel.green_bets == 0).sum()),
           "green_bets_per_selection": round(float(g.green_bets.mean()), 2) if len(g) else None,
           "cover_quantiles": {str(q): round(float(g.cover.quantile(q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
           if len(g) else None,
           "over_covered_1pct": int((g.cover > 1.01).sum()), "under_covered_1pct": int((g.cover < 0.99).sum()),
           "flat_pnl": round(float(flat.flat_pnl.sum()), 2), "profit_where_flat_known": round(float(flat.betfair_profit.sum()), 2),
           "cover_effect": round(float(flat.cover_effect.sum()), 2),
           "ledger_locked": round(float(_num(sel.ledger_locked).sum()), 2),
           "by_day": {d: {"selections": len(x), "lay_stake": round(float(x.lay_stake.sum()), 2),
                          "betfair_profit": round(float(x.betfair_profit.sum()), 2),
                          "flat_pnl": round(float(_num(x.flat_pnl).sum()), 2),
                          "ledger_locked": round(float(_num(x.ledger_locked).sum()), 2),
                          "median_cover": round(float(_num(x.cover).median()), 4) if _num(x.cover).notna().any() else None}
                      for d, x in sel.groupby("day")}}
    return out


def report_md(out: dict) -> str:
    s = out.get("summary", {})
    if "note" in s:
        return "# Early lay green-up audit\n\n" + s["note"] + "\n"
    L = ["# Early lay green-up audit (Betfair's settled bets)", "",
         f"{s['selections']} dogs laid and matched, GBP{s['lay_stake']:.2f} laid; Betfair's profit "
         f"GBP{s['betfair_profit_gross']:+.2f} gross, GBP{s['betfair_profit_net']:+.2f} after commission at "
         f"{s['commission']:.0%} of each market's winnings. Greened {s['greened']}, closed at the SP or held "
         f"{s['sp_or_held']}; {s['green_bets_per_selection']} green-up bets a dog.", "",
         f"Cover (backs' value over the lay's): quantiles {s['cover_quantiles']}; over by more than 1%: "
         f"{s['over_covered_1pct']}, under by more than 1%: {s['under_covered_1pct']}.", "",
         f"Where greened in full: Betfair's profit GBP{s['profit_where_flat_known']:+.2f} against "
         f"GBP{s['flat_pnl']:+.2f} for a flat green-up at the same back prices (the cover's effect "
         f"GBP{s['cover_effect']:+.2f}); the ledger's locked P&L (the worse outcome) GBP{s['ledger_locked']:+.2f}.", "",
         "| Day | Dogs | Laid | Betfair profit | Flat green-up | Ledger locked | Median cover |", "|---|---|---|---|---|---|---|"]
    for d, x in s["by_day"].items():
        L.append(f"| {d} | {x['selections']} | {x['lay_stake']:.2f} | {x['betfair_profit']:+.2f} | {x['flat_pnl']:+.2f} | "
                 f"{x['ledger_locked']:+.2f} | {x['median_cover']} |")
    L += ["", f"Days without a live ledger: {', '.join(out.get('no_ledger', [])) or 'none'}. Markets not yet settled: "
          f"{out.get('unsettled_markets', 0)}."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from greyhound.early_lay import PREFIX, REF, commission
    from sources.common import Store
    from trading.exchange import BetfairData, LiveExchange
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="date_from", default="2026-11-03")
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="out/green_audit")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    d0 = date.fromisoformat(a.date_from)
    d1 = date.fromisoformat(a.date_to) if a.date_to else date.today()
    x = LiveExchange(BetfairData(source="greyhound"))
    x.login()
    all_bets, all_sel, no_ledger, unsettled = [], [], [], 0
    d = d0
    while d <= d1:
        raw = store.get(f"{PREFIX}/live/{d:%Y-%m-%d}/ledger.csv")
        if not raw:
            no_ledger.append(f"{d}")
            d += timedelta(days=1)
            continue
        led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
        mids = sorted(set(led[led.action == "lay"].market_id.dropna().astype(str)))
        bets = bets_frame(x.cleared(mids), d, REF) if mids else pd.DataFrame()
        unsettled += len(set(mids) - set(bets.market_id)) if len(bets) else len(mids)
        log.info("%s: %d markets laid, %d settled bets", d, len(mids), len(bets))
        if len(bets):
            all_bets.append(bets)
            all_sel.append(selections(bets, led))
        d += timedelta(days=1)
    bets = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    sel = pd.concat(all_sel, ignore_index=True) if all_sel else pd.DataFrame()
    out = {"from": f"{d0}", "to": f"{d1}", "no_ledger": no_ledger, "unsettled_markets": unsettled,
           "summary": summary(sel, bets, commission())}
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    bets.to_csv(o / "bets.csv", index=False)
    sel.to_csv(o / "selections.csv", index=False)
    (o / "report.json").write_text(json.dumps(out, indent=1, default=str))
    (o / "report.md").write_text(report_md(out))
    print(report_md(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
