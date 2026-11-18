"""Why the horse trader's days shrank after 8 Nov (the owner, 18 Nov: "Run 1"; read-only, nothing reaches Betfair).

10 Oct - 8 Nov: GBP210 a day on 38 races; 9-17 Nov: GBP36 a day on 20 races (the live-day record). For every live day:
1. The card: UK/IE WIN markets Betfair listed (betfair_live_markets, the recorder's catalogue), by country and code.
2. Priced: the races in the 06:00 predictions (s3 predictions/<day>.csv, market_id) on that card.
3. Traded: the races and horses the trader backed (s3 trading/live/<day>/ledger.csv, event 'back', matched > 0).
4. The edge at entry: each horse's first back against the BSP (the settle row): the median log of price over BSP
   (the entry margin) and the stake-weighted CLV (P/B - 1).
Then the two periods side by side, by code: flat (Betfair raceType 'Flat') and jumps (every other raceType), and the
flat split by all-weather venue (an assumption, a fixed venue list: Southwell, Wolverhampton, Kempton, Lingfield,
Newcastle, Chelmsford, Dundalk; Lingfield and Newcastle also stage turf and jumps, so the split is rough).
And for the priced races not traded on 9-17 Nov, what the trader's ledger says about them (event and reason counts).
"""
import gzip
import io
import os
import sqlite3
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
DAYS = [date(2026, 10, 10) + timedelta(days=i) for i in range((date(2026, 11, 17) - date(2026, 10, 10)).days + 1)]
SPLIT = "2026-11-09"
AW = ("southwell", "wolverhampton", "kempton", "lingfield", "newcastle", "chelmsford", "dundalk")
K = ["market_id", "selection_id"]


def csv(key):
    try:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), dtype={"market_id": str}, low_memory=False)


card = pd.read_sql_query("""SELECT DISTINCT race_date, market_id, venue, country, race_type, market_start_utc
                            FROM betfair_live_markets WHERE market_type = 'WIN' AND country IN ('GB', 'IE')
                            AND race_date >= '2026-10-10'""", conn)
card["market_id"] = card.market_id.astype(str)
card["code"] = np.where(card.race_type.fillna("").str.lower() == "flat", "flat", "jumps")
card.loc[(card.code == "flat") & card.venue.fillna("").str.lower().str.contains("|".join(AW)), "code"] = "flat AW"
print(f"card: {card.market_id.nunique():,} UK/IE WIN markets, {card.race_date.min()} to {card.race_date.max()}; "
      f"race_type values {card.race_type.value_counts(dropna=False).to_dict()}")

days, horses, untraded = [], [], []
for d in DAYS:
    ds = f"{d:%Y-%m-%d}"
    c = card[card.race_date == ds]
    pr = csv(f"predictions/{ds}.csv")
    led = csv(f"trading/live/{ds}/ledger.csv")
    priced = set(pr.market_id.dropna().astype(str)) if pr is not None and "market_id" in pr else set()
    pr_races = (pr.groupby(["venue", "race_time"]).ngroups if pr is not None and {"venue", "race_time"} <= set(pr)
                else np.nan)
    traded = set()
    if led is not None and not led.empty:
        for col in ("matched", "avg_price", "bsp", "selection_id"):
            led[col] = pd.to_numeric(led[col], errors="coerce")
        led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
        b = led[(led.event == "back") & (led.matched > 0)].sort_values("t")
        traded = set(b.market_id.astype(str))
        st = led[led.event == "settle"].sort_values("t").drop_duplicates(K, keep="last")[K + ["bsp"]]
        f1 = b.drop_duplicates(K).merge(st, on=K, how="left", suffixes=("", "_s"))
        f1["day"] = ds
        horses.append(f1[["day", "market_id", "selection_id", "matched", "avg_price", "bsp_s"]])
        if ds >= SPLIT:
            rest = led[led.market_id.astype(str).isin(priced - traded)]
            if not rest.empty:
                untraded.append(rest.assign(day=ds))
    days.append({"day": ds, "card": c.market_id.nunique(), "card_gb": int((c.drop_duplicates("market_id").country == "GB").sum()),
                 "card_ie": int((c.drop_duplicates("market_id").country == "IE").sum()),
                 "pred_file": pr is not None, "pred_races": pr_races, "pred_runners": len(pr) if pr is not None else 0,
                 "priced_on_card": len(priced & set(c.market_id)), "priced_ids": len(priced), "traded": len(traded)})

dd = pd.DataFrame(days)
dd = dd[(dd.card > 0) | dd.pred_file]
print("\n== 1-3. each day: races on Betfair's card, priced at 06:00, traded")
print(dd.to_string(index=False))
dd["period"] = np.where(dd.day < SPLIT, "A 10 Oct-8 Nov", "B 9-17 Nov")
print("\n   a day, by period (means)")
print(dd.groupby("period")[["card", "card_gb", "card_ie", "pred_races", "pred_runners", "priced_on_card", "traded"]]
      .mean().round(1).to_string())

h = pd.concat(horses, ignore_index=True)
h = h.merge(card.drop_duplicates("market_id")[["market_id", "code", "country"]], on="market_id", how="left")
h["period"] = np.where(h.day < SPLIT, "A 10 Oct-8 Nov", "B 9-17 Nov")
ok = (h.bsp_s > 1) & (h.avg_price > 1)
h = h[ok].copy()
h["lm"] = np.log(h.avg_price / h.bsp_s)
h["clvx"] = h.matched * (h.avg_price / h.bsp_s - 1)


def tab(g):
    return pd.Series({"horses": len(g), "a_day": len(g) / g.day.nunique(), "staked": g.matched.sum(),
                      "entry_margin_median": g.lm.median(), "clv_first_backs": g.clvx.sum() / g.matched.sum()})


print("\n== 4. first backs against the BSP (no non-runner reduction: an estimate), by period")
print(h.groupby("period").apply(tab).round(4).to_string())
print("\n   by period and code (code from Betfair's raceType; 'flat AW' by the venue list)")
print(h.groupby(["period", "code"]).apply(tab).round(4).to_string())
print("\n   by period and country")
print(h.groupby(["period", "country"]).apply(tab).round(4).to_string())
print("\n   the entry margin by week")
h["week"] = pd.to_datetime(h.day).dt.to_period("W-SUN").astype(str)
print(h.groupby("week").apply(tab).round(4).to_string())

print("\n   the card's mix by period (share of races)")
card["period"] = np.where(card.race_date < SPLIT, "A 10 Oct-8 Nov", "B 9-17 Nov")
mix = card.drop_duplicates("market_id").groupby(["period", "code"]).size().unstack(fill_value=0)
print((mix.div(mix.sum(axis=1), axis=0)).round(3).to_string())

if untraded:
    u = pd.concat(untraded, ignore_index=True)
    print(f"\n== 5. priced races not traded, 9-17 Nov: {u.market_id.nunique()} markets with ledger rows")
    print("   events:", u.event.value_counts().to_dict())
    if "reason" in u:
        print("   reasons (top 15):")
        print(u.reason.astype(str).str.slice(0, 90).value_counts().head(15).to_string())
else:
    print("\n== 5. priced races not traded, 9-17 Nov: no ledger rows for them (the trader read them and planned nothing)")
