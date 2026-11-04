"""Replay a held Cricsheet match through the live engine and the production bundle: the live path, end to end.

    python -m cricket.live.replay --root data/cricket/store --zip data/cricket/raw/all_json.zip --match 1426312

Prices before every delivery (or every ``--every``th), writes them to ``--out`` (CSV) with the time each quote
took, and prints a line an over. A test of the path and its speed, not an evaluation: one match scores nothing.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

from cricket.ingest.ball_feed import CricsheetReplay
from cricket.ingest.cricsheet import RAW_PREFIX, latest_raw_day
from cricket.live.engine import Engine
from cricket.model.production import load_bundle
from sources.common import Store


def raw_match(store: Store, match_id: str, zip_path: str | None) -> dict:
    zb = Path(zip_path).read_bytes() if zip_path else store.get(f"{RAW_PREFIX}/{latest_raw_day(store)}/all_json.zip")
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        return json.loads(z.read(f"{match_id}.json"))


def replay(engine: Engine, feed, every: int = 1) -> pd.DataFrame:
    rows, n = [], 0
    for ev in feed.events():
        if ev.kind == "delivery":
            if n % every == 0:
                t0 = time.perf_counter()
                q = engine.price()
                rows.append({"innings": q["innings"], "runs": q["runs"], "wickets": q["wickets"], "legal": q["legal"],
                             "target": q["target"], **{f"p_raw_{k}": v for k, v in q["p_raw"].items()},
                             **({f"p_{k}": v for k, v in q["p"].items()} if q["p"] else {}),
                             **({f"total_{k}": v for k, v in q["innings_total"].items()}
                                if "innings_total" in q else {}),
                             "quote_seconds": time.perf_counter() - t0})
            n += 1
        engine.on_event(ev)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--zip", help="a local all_json.zip (default: the store's latest)")
    ap.add_argument("--match", required=True)
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--version", default="latest")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    raw = raw_match(store, a.match, a.zip)
    eng = Engine(load_bundle(store, a.version), paths=a.paths)
    df = replay(eng, CricsheetReplay(a.match, raw), a.every)
    if a.out:
        df.to_csv(a.out, index=False)
    info = raw["info"]
    print(f"{a.match}: {info.get('event', {}).get('name')} {info['dates'][0]} {' v '.join(info['teams'])}; "
          f"winner {info['outcome'].get('winner')}; model {eng.b['version']}; {len(df)} quotes, "
          f"median {df.quote_seconds.median():.2f}s a quote")
    first = df[df.legal % 6 == 0].drop_duplicates(["innings", "legal"])
    print(first.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
