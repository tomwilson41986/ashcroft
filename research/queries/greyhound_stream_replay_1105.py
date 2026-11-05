"""Today's early-lay paper books replayed through the recorded Exchange Stream with queue-aware fills
(research/greyhound/stream_replay.py; read-only), beside the optimistic replay the running page uses (matched once
the price trades, no queue). Each lay at its ledger time, price and stake; fills from the volume traded at its price
after the money ahead of it (all of the traded volume, and half: in case Betfair counts both sides); closed at the
best back price 5 s before the off and at the SP."""
import io
import json
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.getcwd())
from betfair_recorder import bucket, s3_client                       # noqa: E402
from sources.common import Store                                      # noqa: E402
from greyhound import early_lay as el                                 # noqa: E402
from research.greyhound import stream_replay as sr                     # noqa: E402

DAY = date(2026, 11, 5)
s3, store = s3_client(), Store()


def listing(prefix):
    out, tok = [], None
    while True:
        kw = {"Bucket": bucket(), "Prefix": prefix, **({"ContinuationToken": tok} if tok else {})}
        r = s3.list_objects_v2(**kw)
        out += [o["Key"] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        tok = r["NextContinuationToken"]


lines = sr.load_lines(lambda k: s3.get_object(Bucket=bucket(), Key=k)["Body"].read(), listing, DAY)
print("stream lines:", len(lines))
parts = el.replay_books(DAY)
for book in ("paper", "paper-m10", "paper-short20", "paper-short10"):
    raw = store.get(f"{el.PREFIX}/{book}/{DAY:%Y-%m-%d}/ledger.csv")
    if not raw:
        print(f"\n==== {book}: no data found (no ledger)")
        continue
    led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    orders = el_orders = sr.ledger_orders(led, book)
    print(f"\n==== {book}: {len(orders)} lays")
    for vf in (1.0, 0.5):
        df = sr.replay(lines, sr.ledger_orders(led, book), vol_factor=vf, comm=0.02)
        print(f"queue-aware, traded volume x{vf}:", json.dumps(sr.summary(df)))
        if vf == 1.0 and len(df):
            m = df[df.matched > 0]
            print(m[["market", "sid", "price", "queue_at_place", "taken", "matched", "close_price", "pnl", "pnl_sp"]]
                  .head(25).to_string(index=False))
    lays = led[(led.action == "lay") & pd.to_numeric(led.price, errors="coerce").gt(1)]
    if len(lays):
        d = el.replay_on_books(lays, parts)
        mm = d[d.matched == True]                                      # noqa: E712
        print("optimistic (the page's replay):", json.dumps({"lays": len(d), "matched": len(mm),
              "pnl": round(float(pd.to_numeric(mm.pnl, errors="coerce").fillna(0).sum()), 2)}))
