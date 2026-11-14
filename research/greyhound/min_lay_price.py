"""A minimum lay price for the greyhound early lay (the owner, 13 Nov 2026: "Yes run the minimum-lay-price test").

The early lay lays the dogs our forecast says will drift, at 80% of the forecast SP, and greens up before the off
(greyhound/laytest.py ``PRIMARY``). On its first day on real money with fresh prices the matched lays under a price
of 5 lost and those at 5 and over made a little, and on the runners with a morning price a 20% forecast gap was
followed by almost no move to the SP under 5 (ledger ``greyhound-short-odds-research-1113``). Registered before this
ran (``greyhound-min-lay-price-1113``): does the forward test's own rule lose on its lays under ``CUT``?

The procedure is the forward test's, unchanged (``laytest.ledger``): the recorder's books re-timed, the 5 October
model's price, the forecast refitted for each day on the recorded days before it, every recorded day from the first
that has a forecast. The primary rule's matched lays are split by the price they matched at:

- ``all``: the rule as it trades;
- ``under``: the lays matched under ``CUT``;
- ``over``: the lays matched at ``CUT`` or over - the rule with the minimum lay price.

Each band's return on the money at risk after commission on each market's net, with its 90% interval (a bootstrap
over every scored day, ``limit_study.rule_table``); and the two readings the decision rests on, paired by day: the
``under`` band's net a day, and the ``over`` rule's return less the whole rule's. ``SENSITIVITY`` cuts are reported,
judged by nothing.

    python -m research.greyhound.min_lay_price --out out/min_lay_price
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CUT = 5.0
SENSITIVITY = (3.0, 4.0, 6.0, 8.0)


def _boot(day: pd.DataFrame, f, n_boot: int = 2000, seed: int = 0) -> list[float] | None:
    if not len(day):
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(day), (n_boot, len(day)))
    v = np.array([f(day.iloc[i]) for i in idx])
    v = v[np.isfinite(v)]
    return [round(float(np.quantile(v, 0.05)), 4), round(float(np.quantile(v, 0.95)), 4)] if len(v) else None


def split(orders: pd.DataFrame, days: list[str], cut: float = CUT, n_boot: int = 2000) -> dict:
    """The primary rule's matched lays (``orders``: already selected, ``filled`` true) by the price matched."""
    from greyhound import tradeout as to
    from research.greyhound import limit_study as ls
    o = orders.assign(race_date=orders.race_date.astype(str))
    o["band"] = np.where(o.price.to_numpy(float) < cut, "under", "over")
    out = {"cut": cut, "days": len(days), "bands": {}}
    for name, sub in (("all", o.assign(band="all")), ("under", o[o.band == "under"]), ("over", o[o.band == "over"])):
        t = ls.rule_table(sub, keys=("band",), days=days, n_boot=n_boot) if len(sub) else pd.DataFrame()
        out["bands"][name] = json.loads(t.iloc[0].to_json()) if len(t) else {"trades": 0}
    # paired by day: each band's net and risk a day (commission on each market's net within the band)
    m = to.settle(o, ("band",))
    m["race_date"] = m.race_date.astype(str)
    d = m.pivot_table(index="race_date", columns="band", values=["net", "risk"], aggfunc="sum").reindex(days).fillna(0.0)
    for c in (("net", "under"), ("net", "over"), ("risk", "under"), ("risk", "over")):
        if c not in d:
            d[c] = 0.0
    whole = to.settle(o.assign(band="all"), ("band",))
    whole["race_date"] = whole.race_date.astype(str)
    w = whole.groupby("race_date").agg(net=("net", "sum"), risk=("risk", "sum")).reindex(days).fillna(0.0)
    day = pd.DataFrame({"u_net": d[("net", "under")], "o_net": d[("net", "over")], "o_risk": d[("risk", "over")],
                        "w_net": w.net, "w_risk": w.risk})
    roi = lambda n, r: float(n.sum() / r.sum()) if r.sum() > 0 else np.nan
    out["under_net_a_day"] = {"mean": round(float(day.u_net.mean()), 4),
                              "interval_90": _boot(day, lambda g: float(g.u_net.mean()), n_boot),
                              "days_down": int((day.u_net < 0).sum()), "days_up": int((day.u_net > 0).sum())}
    diff = roi(day.o_net, day.o_risk) - roi(day.w_net, day.w_risk)
    out["over_less_all_roi"] = {"diff": round(diff, 5) if np.isfinite(diff) else None,
                                "interval_90": _boot(day, lambda g: roi(g.o_net, g.o_risk) - roi(g.w_net, g.w_risk),
                                                     n_boot)}
    return out


def verdict(r: dict) -> str:
    """The registered reading: adopt the minimum lay price when the lays under it lose (the 90% interval of their net
    a day wholly below zero) and the rule without them returns no less than the whole rule."""
    u, d = r["under_net_a_day"]["interval_90"], r["over_less_all_roi"]["diff"]
    if u is None or d is None:
        return "not judged: no matched lays in a band"
    return "PASS" if u[1] < 0 and d >= 0 else "FAIL"


def run(store, first: date, last: date, source: str = "retimed", n_boot: int = 2000) -> dict:
    from greyhound import laytest as lt
    from research.greyhound import limit_study as ls
    panel = ls.load(store, first, last, source=source)
    days = sorted(panel.race_date.astype(str).unique()) if len(panel) else []
    books = lt.ledger(panel, days) if len(panel) else {}
    res = {"from": f"{first}", "to": f"{last}", "source": source, "rule": lt.PRIMARY, "days_with_books": len(days),
           "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if not books:
        return dict(res, note="no data found: no day with a walk-forward forecast")
    scored = sorted(books)
    o = pd.concat([lt._select(books[d], lt.PRIMARY) for d in scored], ignore_index=True)
    res.update(first_scored_day=scored[0], last_scored_day=scored[-1], scored_days=len(scored),
               orders=int(len(o)), matched=int(o.filled.sum()))
    f = o[o.filled.astype(bool)]
    res["primary"] = split(f, scored, CUT, n_boot)
    res["verdict"] = verdict(res["primary"])
    res["sensitivity"] = {str(c): split(f, scored, c, n_boot) for c in SENSITIVITY}
    return res


def markdown(r: dict) -> str:
    if "primary" not in r:
        return f"## Greyhound early lay: a minimum lay price\n\n{r.get('note')}\n"
    cols = ["trades", "per_day", "risk", "net", "roi%", "roi_90%", "clv_mean%", "price_median"]
    lines = ["## Greyhound early lay: a minimum lay price", "",
             f"The forward test's rule on its own procedure, {r['first_scored_day']} to {r['last_scored_day']} "
             f"({r['scored_days']} days, books: {r['source']}); {r['orders']:,} orders, {r['matched']:,} matched. "
             f"Registered reading at a price of {CUT:g}: **{r['verdict']}**.", ""]

    def block(s: dict, title: str):
        lines.extend([f"### {title}", "", "| Lays | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)])
        for name in ("all", "under", "over"):
            b = s["bands"].get(name) or {}
            lines.append(f"| {name} | " + " | ".join(str(b.get(c)) for c in cols) + " |")
        u, d = s["under_net_a_day"], s["over_less_all_roi"]
        lines.extend(["", f"The lays under {s['cut']:g}: GBP{u['mean']:+.2f} a day, 90% {u['interval_90']}; down on "
                          f"{u['days_down']} days, up on {u['days_up']}. The rule without them less the whole rule: "
                          f"{d['diff']} of the money at risk, 90% {d['interval_90']}.", ""])

    block(r["primary"], f"The registered cut: {CUT:g}")
    for c, s in r["sensitivity"].items():
        block(s, f"Reported, judged by nothing: {float(c):g}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from greyhound import tradeout as to
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="first", default=None)
    ap.add_argument("--to", dest="last", default=None)
    ap.add_argument("--source", default="retimed", choices=["retimed", "recorder", "live"])
    ap.add_argument("--out", default="out/min_lay_price")
    a = ap.parse_args(argv)
    first = date.fromisoformat(a.first) if a.first else to.FIRST_BOOK_DAY
    last = date.fromisoformat(a.last) if a.last else datetime.now(timezone.utc).date() - timedelta(days=1)
    r = run(Store(), first, last, a.source)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(r, indent=1, default=str))
    Path(a.out + ".md").write_text(markdown(r))
    print(markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
