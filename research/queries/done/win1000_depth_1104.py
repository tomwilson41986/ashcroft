"""Is there the depth for backs_win1000 (the owner, 4 Nov: "is there enough liquidity and depth for the win1000
model")? Read-only. 12 Oct - 2 Nov:

1. The real live trader (to win GBP400): each horse's winnings got against its GBP400, by price band and UK hour;
   each order, the money shown at the best back against what it asked, and against what to win GBP1,000 would ask.
2. The win1000 replay on its own book (paper/horses/shared/agents/backs_win1000): each horse's winnings against
   GBP1,000 (GBP500 a bet at most), by price band and hour; when the GBP4,000 day limit was reached, live and win1000.
"""
import gzip
import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
DAYS = [date(2026, 10, 12) + timedelta(i) for i in range(22)]
BANDS = [1, 3, 6, 11, 21, 51, 1001]
LABELS = ["<3", "3-6", "6-11", "11-21", "21-51", "51+"]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


live, agent = [], []
for d in DAYS:
    raw = get(f"trading/live/{d}/ledger.csv")
    if raw is not None:
        led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
        for c in ("matched", "avg_price", "asked", "back_size", "best_back", "selection_id", "target"):
            led[c] = pd.to_numeric(led.get(c), errors="coerce")
        led["day"] = str(d)
        live.append(led[led.event == "back"])
    raw = get(f"paper/horses/shared/agents/backs_win1000/{d}/fills.csv.gz")
    if raw is not None:
        f = pd.read_csv(io.BytesIO(gzip.decompress(raw)), low_memory=False, dtype={"market_id": str})
        f["day"] = str(d)
        agent.append(f)
live = pd.concat(live, ignore_index=True)
agent = pd.concat(agent, ignore_index=True)
for x in (live, agent):
    x["t"] = pd.to_datetime(x.ts, utc=True, errors="coerce")
    x["hour"] = x.t.dt.tz_convert("Europe/London").dt.hour
agent["avg_price"] = pd.to_numeric(agent.price, errors="coerce")
agent["matched"] = pd.to_numeric(agent.matched, errors="coerce")


def horses(fills, target):
    f = fills[fills.matched > 0].copy()
    f["won"] = f.matched * (f.avg_price - 1)
    h = f.groupby(["day", "market_id", "selection_id"]).agg(staked=("matched", "sum"), won=("won", "sum"),
                                                            first=("t", "min"), hour=("hour", "first"),
                                                            orders=("matched", "size")).reset_index()
    h["price"] = h.won / h.staked + 1
    h["want"] = np.minimum(target, 500.0 * (h.price - 1)) if target > 400 else np.minimum(target, 200.0 * (h.price - 1))
    h["got"] = (h.won / h.want).clip(upper=1.0)
    h["band"] = pd.cut(h.price, BANDS, labels=LABELS, right=False)
    return h


def table(h, by):
    return h.groupby(by, observed=True).agg(horses=("got", "size"), staked=("staked", "sum"),
                                            stake_a_horse=("staked", "mean"), got_of_target=("got", "mean"),
                                            full=("got", lambda g: (g >= 0.95).mean()),
                                            orders_a_horse=("orders", "mean")).round(3)


hl, ha = horses(live, 400.0), horses(agent, 1000.0)
print(f"== 12 Oct-2 Nov: live {hl.day.nunique()} days, win1000 replay {ha.day.nunique()} days")
for name, h in (("live (to win GBP400)", hl), ("win1000 replay (to win GBP1,000)", ha)):
    print(f"\n-- {name}: {len(h)} horses, GBP{h.staked.sum():,.0f} staked, GBP{h.staked.mean():,.2f} a horse; "
          f"winnings got {h.got.mean():.0%} of the target on average, the full target on {(h.got >= 0.95).mean():.0%}")
    print("   by price band")
    print(table(h, "band").to_string())
    print("   by UK hour of the first fill")
    print(table(h, "hour").to_string())

# each live order: the money at the best back against what it asked, and what GBP1,000 would ask
o = live[(live.asked > 0)].copy()
o["ask1000"] = np.minimum(500.0, o.asked * 2.5)
o["band"] = pd.cut(o.best_back, BANDS, labels=LABELS, right=False)
print(f"\n== the live orders: {len(o)}; filled in full {(o.matched >= o.asked - 0.01).mean():.0%}; "
      f"the best back showed at least the ask on {(o.back_size >= o.asked).mean():.0%}, at least 2.5x the ask "
      f"(to win GBP1,000) on {(o.back_size >= o.ask1000).mean():.0%}")
print(o.groupby("band", observed=True).agg(orders=("asked", "size"), asked=("asked", "median"),
                                           shown=("back_size", "median"), filled=("matched", "median"),
                                           full=("matched", lambda m: 0),
                                           ).drop(columns="full").round(2).to_string())
o["hour"] = o.t.dt.tz_convert("Europe/London").dt.hour
print(o.groupby("hour").agg(orders=("asked", "size"), asked=("asked", "median"), shown=("back_size", "median"),
                            shown_over_ask=("back_size", lambda s: 0)).drop(columns="shown_over_ask")
      .assign(cover_400=o.groupby("hour").apply(lambda g: (g.back_size >= g.asked).mean()).round(2),
              cover_1000=o.groupby("hour").apply(lambda g: (g.back_size >= g.ask1000).mean()).round(2))
      .round(2).to_string())


# the day limit: when GBP3,950 was staked
def limit_hit(fills):
    rows = []
    for d, g in fills[fills.matched > 0].sort_values("t").groupby("day"):
        c = g.matched.cumsum()
        hit = g.t[c >= 3950]
        rows.append({"day": d, "staked": round(c.iloc[-1], 0), "horses": g.groupby(["market_id", "selection_id"]).ngroups,
                     "races": g.market_id.nunique(),
                     "limit_at_uk": hit.iloc[0].tz_convert("Europe/London").strftime("%H:%M") if len(hit) else "-"})
    return pd.DataFrame(rows).set_index("day")


lh = limit_hit(live).add_prefix("live_").join(limit_hit(agent).add_prefix("w1000_"))
print("\n== the GBP4,000 day limit: staked, horses, races and when it was reached (UK)")
print(lh.to_string())
