"""Build the traders' dashboard: ``traders.json`` and ``index.html``, locally and (``--upload``) to S3.

    python -m dashboard --out out/dashboard
    python -m dashboard --out out/dashboard --upload            # s3://$PRED_BUCKET/dashboard/{traders.json,index.html}
    python -m dashboard --root path/to/copy --date 2026-11-04   # a local copy laid out as <root>/{pred,capture}/<key>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.page import render  # noqa: E402
from dashboard.traders import LocalSource, S3Source, build  # noqa: E402


def summary_md(data: dict) -> str:
    """A plain table for the job summary: every trader, the six figures."""
    def g(v):
        return "–" if v is None else f"{v:+,.2f}"
    L = [f"### Traders, {data['today']} (built {data['generated_utc']})", ""]
    for s in data["sports"]:
        L += [f"#### {s['label']}", ""]
        if not s["traders"]:
            L += ["No data found: no trader for this sport.", ""]
            continue
        L += ["| Trader | Mode | Today P&L | Volume | Forecast today | Month P&L | Forecast month | All time |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for t in s["traders"]:
            if not t["has_data"]:
                L.append(f"| {t['name']} | {t['mode']} | no data found | | | | | |")
                continue
            d, m, a = t["today"], t["month"], t["all_time"]
            vol = "–" if d["volume"] is None else f"{d['volume']:,.2f}"
            L.append(f"| {t['name']} | {t['mode']} | {g(d['pnl'])} | {vol} | {g(d['forecast'])} | {g(m['pnl'])} | "
                     f"{g(m['forecast'])} | {g(a['pnl'])} |")
        L.append("")
    if data["errors"]:
        L += ["Not read: " + "; ".join(f"{e['trader']}: {e['error']}" for e in data["errors"]), ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/dashboard")
    ap.add_argument("--root", default=None, help="a local copy in place of S3")
    ap.add_argument("--date", default=None, help="the UK trading day (default today)")
    ap.add_argument("--upload", action="store_true", help="copy both files to s3://$PRED_BUCKET/dashboard/")
    ap.add_argument("--no-cache", action="store_true", help="re-read every day's ledger")
    ap.add_argument("--summary", default=None, help="append the markdown table here (e.g. $GITHUB_STEP_SUMMARY)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    src = LocalSource(a.root) if a.root else S3Source()
    data = build(src, date.fromisoformat(a.date) if a.date else None, cache=not a.no_cache)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    js = json.dumps(data, indent=1, default=str).encode()
    html = render(data).encode()
    (out / "traders.json").write_bytes(js)
    (out / "index.html").write_bytes(html)
    md = summary_md(data)
    print(md)
    if a.summary:
        with open(a.summary, "a") as f:
            f.write(md + "\n")
    if a.upload:
        src.put("pred", "dashboard/traders.json", js, "application/json")
        src.put("pred", "dashboard/index.html", html, "text/html; charset=utf-8")
        print(f"uploaded {src.where('pred', 'dashboard/index.html')} and traders.json")
    return 1 if data["errors"] and len(data["errors"]) == sum(len(s["traders"]) for s in data["sports"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
