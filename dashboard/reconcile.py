"""The dashboard against the traders' own records (read-only): a pass/fail check.

- Horse racing, live and paper: each day's settled P&L against the session's own ``trading/<mode>/<day>/summary.json``
  (``settled_pnl``, after commission).
- Football, live and paper: the all-time P&L and settled bets against the settle step's
  ``sources/football/live/<mode>/summary.json``.

    python -m dashboard.reconcile            # exit 1 on a difference over GBP0.05
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.traders import TRADERS, S3Source, summarise, trader_days  # noqa: E402

TOL = 0.05


def main() -> int:
    src = S3Source()
    today = datetime.now(ZoneInfo("Europe/London")).date()
    bad = 0
    for t in TRADERS:
        if t.sport == "horse_racing":
            days, _ = trader_days(src, t, today, cache=False)
            rows = []
            for d, m in sorted(days.items()):
                raw = src.get(t.kind, f"{t.base}/{d}/summary.json")
                theirs = json.loads(raw).get("settled_pnl") if raw else None
                ours = float(m.loc[m.settled.astype(bool), "pnl"].sum())
                diff = None if theirs is None else ours - float(theirs)
                rows.append((d, ours, theirs, diff))
                if diff is not None and abs(diff) > TOL and d < today.isoformat():
                    bad += 1
            print(f"== {t.id}: {len(rows)} days")
            for d, o, th, df in rows:
                flag = "" if df is None or abs(df) <= TOL else "   <-- differs"
                print(f"   {d}  dashboard {o:+10.2f}  summary.json "
                      f"{'(none)' if th is None else f'{float(th):+10.2f}'}"
                      f"{'' if df is None else f'  diff {df:+.2f}'}{flag}")
        elif t.sport == "football":
            raw = src.get(t.kind, f"{t.base}/summary.json")
            days, _ = trader_days(src, t, today, cache=False)
            s = summarise(t, days, today)
            n_mk = sum(len(m) for m in days.values())
            n_set = sum(int(m.settled.astype(bool).sum()) for m in days.values())
            print(f"== {t.id}: {len(days)} days, {n_mk} markets backed, {n_set} settled; dashboard all time "
                  f"{s['all_time']['pnl']}")
            if raw:
                j = json.loads(raw)
                print(f"   summary.json: bets {j.get('bets')}, settled {j.get('settled')}, staked {j.get('staked')}, "
                      f"pnl {j.get('pnl')}, updated {j.get('updated_utc')}")
                if s["all_time"]["pnl"] is not None and abs(s["all_time"]["pnl"] - float(j.get("pnl") or 0)) > TOL:
                    bad += 1
                    print("   <-- differs")
            else:
                print("   summary.json: none")
    print(f"\nreconcile: {'PASS' if not bad else f'FAIL ({bad} differences)'} (closed days, tolerance GBP{TOL})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
