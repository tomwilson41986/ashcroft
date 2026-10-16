"""The owner's ask (16 Oct, 18:15 UTC): audit and QA today's trading. Read-only.

Part 1, the audit: done/audit_1015.py and done/topups_1015.py with 16 Oct added (the day before its settle: each
horse's BSP from the recorded books where the settle has not written it).
Part 2, the QA of today's inputs:
- the 06:00 predictions: when written, runners and races, and the trader's markets that carried no model price;
- horseracebase's pause (no results after 13 Oct): today's runners that ran on 14 or 15 Oct (found in those days'
  Betfair catalogues), whose last run the model cannot see, and the money and CLV on them against the rest;
- the model's price against the BSP for the horses backed, day by day (has the model drifted short or long?);
- non-runners among the horses backed."""
import io
import os
import re
import sys

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse  # noqa: E402

DONE = os.path.join(os.getcwd(), "research", "queries", "done")
DAYS = ["2026-10-10", "2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"]
for name in ("audit_1015.py", "topups_1015.py"):
    src = open(os.path.join(DONE, name), encoding="utf-8").read()
    old = '"2026-10-14", "2026-10-15"]'
    assert old in src, name
    src = src.replace(old, '"2026-10-14", "2026-10-15", "2026-10-16"]')
    print(f"\n######## {name} with 16 Oct")
    if name == "topups_1015.py":
        # 16 Oct is not settled yet: the settle rows are its BSP and result; without them the day drops out, so the
        # BSP the books showed after the off stands in (the result is not needed for CLV)
        src = src.replace('    b = b[b.result.isin(["WINNER", "LOSER"]) & (b.bsp > 1)]',
                          '    if day == "2026-10-16" and not len(st):\n'
                          '        bk = []\n'
                          '        for n in ("books_trader", "books"):\n'
                          '            try:\n'
                          '                x = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=f"betfair_live/{day}/{n}.csv.gz")["Body"].read()), compression="gzip", low_memory=False, dtype={"market_id": str})\n'
                          '                bk.append(x)\n'
                          '            except Exception:\n'
                          '                pass\n'
                          '        x = pd.concat(bk, ignore_index=True)\n'
                          '        x["sp_actual"] = pd.to_numeric(x.sp_actual, errors="coerce")\n'
                          '        x["selection_id"] = pd.to_numeric(x.selection_id, errors="coerce")\n'
                          '        sp = x[x.sp_actual > 1].sort_values("polled_utc").groupby(K).sp_actual.last().rename("bsp_b").reset_index()\n'
                          '        b = b.merge(sp, on=K, how="left")\n'
                          '        b["bsp"], b["sp_settled"], b["result"] = b.bsp_b, b.mp, "OPEN"\n'
                          '        b["price_s"] = b.avg_price\n'
                          '    b = b[b.result.isin(["WINNER", "LOSER", "OPEN"]) & (b.bsp > 1)]')
        src = src.replace('    b = b.drop(columns=["bsp", "clv", "result"], errors="ignore").merge(hm[K + ["mp"]], on=K).merge(st, on=K)',
                          '    b = b.drop(columns=["bsp", "clv", "result"], errors="ignore").merge(hm[K + ["mp"]], on=K).merge(st, on=K, how="left" if day == "2026-10-16" else "inner")')
        src = src.replace('    b["price_s"] = b.avg_price * b.sp_settled / b.mp\n',
                          '    b["price_s"] = np.where(b.sp_settled.notna(), b.avg_price * b.sp_settled / b.mp, b.avg_price)\n')
    exec(compile(src, name, "exec"), {"__name__": "__main__"})

# ------------------------------------------------------------------------------------------------- the QA of inputs
print("\n######## QA of today's inputs")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
K = ["market_id", "selection_id"]


def csv(key):
    try:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(raw), compression="gzip" if key.endswith(".gz") else None, low_memory=False,
                       dtype={"market_id": str})


def catalogue(day):
    c = pd.concat([csv(f"betfair_live/{day}/{n}.csv.gz") for n in ("markets", "markets_trader")], ignore_index=True)
    if c.empty:
        return c
    c = c[c.get("market_type", pd.Series("WIN", index=c.index)).fillna("WIN").astype(str).str.upper().eq("WIN")]
    c = c[c.country.isin(["GB", "IE"])] if "country" in c.columns else c
    c["key"] = c.runner_name.map(normalise_horse)
    return c.drop_duplicates(K)


today = "2026-10-16"
pred = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=f"predictions/{today}.csv")["Body"].read()))
head = s3.head_object(Bucket=bucket, Key=f"predictions/{today}.csv")
print(f"predictions/{today}.csv written {head['LastModified']}: {len(pred)} runners, "
      f"{pred.groupby(['track', 'race_time']).ngroups} races")
pred["key"] = pred.horse_name.map(normalise_horse)
cat = catalogue(today)
print(f"Betfair GB/IE win markets today: {cat.market_id.nunique()} markets, {len(cat)} runners")
cat["priced"] = cat.key.isin(set(pred.key))
mk = cat.groupby("market_id").agg(venue=("venue", "first"), start=("market_start_utc", "first"),
                                  runners=("key", "size"), priced=("priced", "sum"))
print("markets with no model price at all:", int((mk.priced == 0).sum()),
      "| runners in priced markets without a price:", int((mk.runners - mk.priced)[mk.priced > 0].sum()))
if (mk.priced == 0).any():
    print(mk[mk.priced == 0].to_string())

ran = {}
for d in ("2026-10-14", "2026-10-15"):
    c = catalogue(d)
    ran[d] = set(c.key) if not c.empty else set()
    print(f"runners on Betfair on {d}: {len(ran[d])}")
pred["ran_14_15"] = pred.key.isin(ran["2026-10-14"] | ran["2026-10-15"])
print(f"today's priced runners that ran on 14 or 15 Oct (their last run unseen by the model): "
      f"{int(pred.ran_14_15.sum())} of {len(pred)}")

led = csv(f"trading/live/{today}/ledger.csv")
for c in ("matched", "avg_price", "selection_id", "edge"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
b = led[(led.event == "back") & (led.matched > 0)].copy()
b = b.merge(cat[K + ["key", "runner_name"]], on=K, how="left")
bk = pd.concat([csv(f"betfair_live/{today}/{n}.csv.gz") for n in ("books_trader", "books")], ignore_index=True)
bk["sp_actual"] = pd.to_numeric(bk.sp_actual, errors="coerce")
bk["selection_id"] = pd.to_numeric(bk.selection_id, errors="coerce")
bk = bk.sort_values("polled_utc")
sp = bk[bk.sp_actual > 1].groupby(K).sp_actual.last().rename("bsp")
stt = bk.groupby(K).runner_status.last().rename("status")
b = b.merge(pd.concat([sp, stt], axis=1).reset_index(), on=K, how="left")
b["clv"] = b.avg_price / b.bsp - 1
b["ran_14_15"] = b.key.isin(ran["2026-10-14"] | ran["2026-10-15"])
ok = b.bsp > 1
g = b[ok].groupby("ran_14_15").apply(lambda x: pd.Series({
    "horses": x.selection_id.nunique(), "staked": x.matched.sum(),
    "clv": (x.clv * x.matched).sum() / x.matched.sum()}))
print("\ntoday's backs by whether the horse ran on 14-15 Oct (CLV at the books' BSP and the matched price, before any "
      "non-runner reduction):")
print(g.round(4).to_string())
print("horses backed that were removed (non-runners):", int(b[b.status == "REMOVED"].selection_id.nunique()),
      "| backs with no BSP yet:", int((~ok).sum()))

print("\n== the model's price against the BSP for the horses backed, by day (median of log(model/BSP); below 0 means "
      "the model rated them shorter than the close)")
for d in DAYS:
    l = csv(f"trading/live/{d}/ledger.csv")
    if l.empty or "model_price" not in l.columns and "p_model" not in l.columns:
        continue
    for c in ("matched", "selection_id", "p_model", "p_market"):
        if c in l.columns:
            l[c] = pd.to_numeric(l[c], errors="coerce")
    bb = l[(l.event == "back") & (pd.to_numeric(l.matched, errors="coerce") > 0)].drop_duplicates(K)
    s = l[l.event == "settle"].drop_duplicates(K, keep="last")
    s = s.assign(bsp=pd.to_numeric(s.bsp, errors="coerce"))[K + ["bsp"]] if len(s) else pd.DataFrame(columns=K + ["bsp"])
    if not len(s):
        s = pd.concat([sp], axis=1).reset_index().rename(columns={"sp_actual": "bsp"}) if d == today else s
    bb = bb.merge(s, on=K, how="inner")
    bb = bb[(bb.bsp > 1) & (bb.p_model > 0)]
    lr = np.log((1 / bb.p_model) / bb.bsp)
    lm = np.log((1 / bb.p_market) / bb.bsp) if "p_market" in bb.columns else pd.Series(dtype=float)
    print(f"  {d}: {len(bb)} horses; model {lr.median():+.3f} (mean {lr.mean():+.3f}); the market's price at the first "
          f"back {lm.median():+.3f}; model shorter than the BSP on {float((lr < 0).mean()):.0%}")
