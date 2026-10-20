"""The owner's questions (20 Oct): should we drip feed into the market so the higher prices last longer? Price,
liquidity and depth, not time (the owner: "time isn't a dimension"). Read-only, every back fill of 10-19 Oct against
the trader's own books (it reads every market each poll, three levels each side).

1. What our fill does to the book. Each fill against the read it acted on and the reads of the same runner after it:
   did the best back fall, and does the price we took come back (the best back at or above our price again, and the
   money offered at it)? Fills that took the whole top level against those that left some of it: the gap between
   them is our own footprint.
2. The ladder: the trader takes only the best back (a limit at that price). The levels below it at the same read,
   where they still clear the bar (expected CLV >= 3% at their own price, p_pool x price - 1) and the owner's guard
   (at or over the predicted BFSP): their money and their CLV at the BSP. Then a rule that takes them, horse by horse
   through the day's fills in order, each fill and top-up capped by the stake the horse still wants, so the depth
   taken early replaces the top-ups taken after: its money and CLV x stake against the live fills'.
3. The other side of drip feeding: what a fill leaves on the level. The fills that did not take the whole level (so
   the price was still there) and what the next fill on the horse got.
4. The day limit (GBP4,000): the days it bound, and the backs refused after it (their expected CLV and the CLV at the
   BSP at the price shown): the money the top-ups spent that better prices could have had.
"""
import io
import os

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
DAYS = [f"2026-10-{d}" for d in range(10, 20)]
BAR = 0.03
BOOK = ["polled_utc", "source", "market_id", "selection_id", "status", "inplay", "runner_status", "back1", "back1_size",
        "back2", "back2_size", "back3", "back3_size", "lay1", "lay1_size", "market_total_matched", "sp_actual"]
NUM = ["back1", "back1_size", "back2", "back2_size", "back3", "back3_size", "lay1", "lay1_size", "market_total_matched",
       "sp_actual"]
NEXT = 5


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


fills, refused, limits = [], [], []
for day in DAYS:
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is None:
        continue
    led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ("matched", "avg_price", "bsp", "selection_id", "minutes_to_off", "best_back", "back_size", "edge",
              "target", "move", "p_model", "p_pool"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
    raw = get(f"betfair_live/{day}/books_trader.csv.gz")
    if raw is None:
        print("no trader books", day)
        continue
    bk = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=lambda c: c in BOOK, dtype={"market_id": str},
                     low_memory=False)
    for c in NUM:
        if c in bk.columns:
            bk[c] = pd.to_numeric(bk[c], errors="coerce")
    bk["selection_id"] = pd.to_numeric(bk.selection_id, errors="coerce")
    bk = bk.dropna(subset=["selection_id"])
    bk["selection_id"] = bk.selection_id.astype("float64")
    bk["t"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce")
    bk = bk.dropna(subset=["t"]).sort_values("t")
    sp = bk[bk.sp_actual > 1].groupby(["market_id", "selection_id"]).sp_actual.last().rename("B_book")
    ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
    pre = bk[(bk.status == "OPEN") & ~ip & bk.runner_status.fillna("ACTIVE").eq("ACTIVE") & (bk.back1 > 1)].copy()

    b = led[(led.event == "back") & (led.matched > 0)].sort_values("t").copy()
    b["day"] = day
    b["selection_id"] = b.selection_id.astype("float64")
    b["sxp"] = b.matched * b.avg_price
    b["k"] = b.groupby(["market_id", "selection_id"]).cumcount()
    hm = b.groupby(["market_id", "selection_id"]).agg(m=("matched", "sum"), sxp=("sxp", "sum")).reset_index()
    hm["Pm"] = hm.sxp / hm.m
    st = led[led.event == "settle"].sort_values("t").drop_duplicates(["market_id", "selection_id"], keep="last")[
        ["market_id", "selection_id", "result", "bsp", "avg_price"]].rename(columns={"avg_price": "Ps", "bsp": "B"})
    st["selection_id"] = st.selection_id.astype("float64")
    b = b.drop(columns=["bsp", "result"], errors="ignore").merge(hm[["market_id", "selection_id", "Pm"]],
                                                                  on=["market_id", "selection_id"]) \
        .merge(st, on=["market_id", "selection_id"], how="left").sort_values("t")
    # the read the trader acted on: its last read of the runner at or before the order
    j = pd.merge_asof(b, pre[["t", "market_id", "selection_id"] + NUM[:-1]].rename(columns={"t": "t_book"}),
                      left_on="t", right_on="t_book", by=["market_id", "selection_id"], direction="backward",
                      tolerance=pd.Timedelta("3min"))
    # the reads of the runner after the order: the next NEXT, and every one to the off
    grp = {k: g for k, g in pre.groupby(["market_id", "selection_id"])}
    after = {f"{c}_{n}": [] for n in range(1, NEXT + 1) for c in ("back1", "size_at")}
    ever, share_at, n_after = [], [], []
    for r in j.itertuples():
        g = grp.get((r.market_id, r.selection_id))
        price = r.avg_price
        if g is None:
            for n in range(1, NEXT + 1):
                after[f"back1_{n}"].append(np.nan)
                after[f"size_at_{n}"].append(np.nan)
            ever.append(np.nan), share_at.append(np.nan), n_after.append(0)
            continue
        g = g[g.t > r.t]
        n_after.append(len(g))
        at = np.zeros(len(g))
        for lv in (1, 2, 3):
            p_, s_ = g[f"back{lv}"].to_numpy(float), g[f"back{lv}_size"].to_numpy(float)
            at += np.where(p_ >= price - 1e-9, np.nan_to_num(s_), 0.0)
        for n in range(1, NEXT + 1):
            after[f"back1_{n}"].append(g.back1.iloc[n - 1] if len(g) >= n else np.nan)
            after[f"size_at_{n}"].append(at[n - 1] if len(g) >= n else np.nan)
        ever.append(float((g.back1.to_numpy(float) >= price - 1e-9).any()) if len(g) else np.nan)
        share_at.append(float((g.back1.to_numpy(float) >= price - 1e-9).mean()) if len(g) else np.nan)
    for c, v in after.items():
        j[c] = v
    j["back_ever"], j["share_reads_at"], j["reads_after"] = ever, share_at, n_after
    j = j.merge(sp, on=["market_id", "selection_id"], how="left")
    fills.append(j)

    # the day limit and the backs refused after it
    tot = b.matched.sum()
    cum = b.set_index("t").matched.cumsum()
    hit = cum[cum >= 3950].index.min() if (cum >= 3950).any() else pd.NaT
    limits.append({"day": day, "staked": tot, "limit_hit_uk": (hit.tz_convert("Europe/London").strftime("%H:%M")
                                                             if pd.notna(hit) else "")})
    if pd.notna(hit):
        sk = led[(led.event == "skip") & (led.t >= hit)].copy()
        sk["day"] = day
        sk["selection_id"] = sk.selection_id.astype("float64")
        sk = sk.merge(sp, on=["market_id", "selection_id"], how="left")
        refused.append(sk)

f = pd.concat(fills, ignore_index=True)
f = f[f.result.isin(["WINNER", "LOSER"]) & (f.B > 1) & (f.Ps > 1)].copy()
f["clv"] = f.avg_price * f.Ps / f.Pm / f.B - 1
f["locked"] = f.clv * f.matched
have = f.back1.notna()
print(f"fills 10-19 Oct settled with a BSP: {len(f):,}, GBP{f.matched.sum():,.2f}; with the read acted on: "
      f"{int(have.sum()):,}; CLV {f.locked.sum() / f.matched.sum():+.2%}")
f = f[have].copy()
f["took"] = f.matched / f.back1_size
f["cleared"] = f.took >= 0.95
f["kind"] = np.where(f.k == 0, "first", "top-up")
f["model_bfsp"] = f.best_back / np.exp(f.move)

print("\n== 1. what our fill does to the book (the reads of the runner after the fill)")
for n in range(1, NEXT + 1):
    f[f"d{n}"] = np.log(f[f"back1_{n}"] / f.back1)
    f[f"back_{n}"] = (f[f"back1_{n}"] >= f.avg_price - 1e-9).astype(float).where(f[f"back1_{n}"].notna())
rows = []
for (kind, cl), g in f.groupby(["kind", "cleared"]):
    row = {"fills": len(g), "staked": g.matched.sum(), "median_took": g.took.median(),
           "median_size_offered": g.back1_size.median()}
    for n in (1, 2, 3, 5):
        row[f"best_back_move_read{n}"] = g[f"d{n}"].mean()
        row[f"price_back_read{n}"] = g[f"back_{n}"].mean()
        row[f"money_at_price_read{n}"] = g[f"size_at_{n}"].median()
    row["price_back_ever"] = g.back_ever.mean()
    row["share_reads_at_price"] = g.share_reads_at.mean()
    row["clv"] = g.locked.sum() / g.matched.sum()
    rows.append(pd.Series(row, name=f"{kind}, {'took the whole level' if cl else 'left some'}"))
print(pd.DataFrame(rows).round(4).T.to_string())
print("\n   took the whole level, by the money offered at the best back (first backs and top-ups)")
x = f[f.cleared]
print(x.groupby(pd.cut(x.back1_size, [0, 5, 10, 20, 50, 100, 1e9]), observed=True).apply(lambda g: pd.Series({
    "fills": len(g), "staked": g.matched.sum(), "move_read1": g.d1.mean(), "price_back_read1": g.back_1.mean(),
    "price_back_read3": g.back_3.mean(), "price_back_ever": g.back_ever.mean(),
    "money_at_price_read1": g.size_at_1.median(), "clv": g.locked.sum() / g.matched.sum()})).round(4).to_string())
print("\n   by the money in the market (took the whole level)")
print(x.groupby(pd.cut(x.market_total_matched, [-1, 1000, 2500, 5000, 10000, 1e12]), observed=True).apply(
    lambda g: pd.Series({"fills": len(g), "move_read1": g.d1.mean(), "price_back_read1": g.back_1.mean(),
                         "price_back_ever": g.back_ever.mean(), "clv": g.locked.sum() / g.matched.sum()})
).round(4).to_string())

print("\n== 2. the ladder below the best back at the read acted on")
f["pp"] = f.p_pool
for lv in (2, 3):
    p_, s_ = f[f"back{lv}"], f[f"back{lv}_size"].fillna(0)
    f[f"edge{lv}"] = p_ * f.pp - 1
    f[f"ok{lv}"] = (p_ > 1) & (f[f"edge{lv}"] >= BAR) & (p_ >= f.model_bfsp - 1e-9)
    f[f"clv{lv}"] = p_ / f.B - 1
for lv in (2, 3):
    g = f[f.cleared & f[f"ok{lv}"]]
    print(f"   level {lv} clearing the bar and the guard where the whole top level was taken: {len(g):,} reads, "
          f"GBP{g[f'back{lv}_size'].sum():,.2f} offered (median GBP{g[f'back{lv}_size'].median():.2f}), "
          f"price {(g[f'back{lv}'] / g.back1 - 1).mean():+.1%} against the best back, CLV at the BSP "
          f"{(g[f'clv{lv}'] * g[f'back{lv}_size']).sum() / g[f'back{lv}_size'].sum():+.2%}")
g = f[f.cleared]
print(f"   (level 2 of every read that took the whole top level, bar or not: {len(g):,}, CLV at the BSP "
      f"{(g.clv2 * g.back2_size.fillna(0)).sum() / g.back2_size.fillna(0).sum():+.2%}; its expected CLV "
      f"{(g.edge2 * g.back2_size.fillna(0)).sum() / g.back2_size.fillna(0).sum():+.2%})")


def ladder_rule(f, depth_levels, bar):
    """Horse by horse through the fills in order: each fill as it was, capped by what the horse still wants, then the
    levels below the best back (when the top level was taken whole) down to `bar`, up to the same cap."""
    out = []
    for _, g in f.sort_values("t").groupby(["day", "market_id", "selection_id"]):
        held, lk, st_ = 0.0, 0.0, 0.0
        for r in g.itertuples():
            cap = max(0.0, max(r.target, r.matched) - held) if r.k > 0 else max(r.target, r.matched)
            take = min(r.matched, cap)
            held += take
            st_ += take
            lk += take * r.clv
            if take < r.matched - 1e-9 or not r.cleared:
                continue
            for lv in depth_levels:
                p_, s_ = getattr(r, f"back{lv}"), getattr(r, f"back{lv}_size")
                if not (p_ > 1 and s_ > 0) or p_ * r.pp - 1 < bar or p_ < r.model_bfsp - 1e-9:
                    break
                room = max(0.0, cap - take)
                if room < 2.0:
                    break
                d = min(s_, room)
                take += d
                held += d
                st_ += d
                lk += d * (p_ * r.Ps / r.Pm / r.B - 1)
        out.append((st_, lk))
    a = np.array(out)
    return a[:, 0].sum(), a[:, 1].sum()


live_s, live_l = f.matched.sum(), f.locked.sum()
print(f"\n   the live fills (this set): GBP{live_s:,.2f}, CLV {live_l / live_s:+.2%}, CLV x stake GBP{live_l:,.2f}")
for levels, bar, name in (((), BAR, "the live fills, each capped by what the horse still wanted"),
                          ((2,), BAR, "+ level 2 down to the bar"), ((2, 3), BAR, "+ levels 2-3 down to the bar"),
                          ((2, 3), 0.06, "+ levels 2-3 where they clear 6%"),
                          ((2, 3), 0.10, "+ levels 2-3 where they clear 10%")):
    s_, l_ = ladder_rule(f, levels, bar)
    print(f"   {name:58s} GBP{s_:,.2f}, CLV {l_ / s_:+.2%}, CLV x stake GBP{l_:,.2f}")

print("\n== 3. a fill that left money on the level: what the horse's next fill got")
f = f.sort_values("t")
f["next_price"] = f.groupby(["day", "market_id", "selection_id"]).avg_price.shift(-1)
f["next_clv"] = f.groupby(["day", "market_id", "selection_id"]).clv.shift(-1)
for cl, g in f.groupby("cleared"):
    g = g[g.next_price.notna()]
    print(f"   {'took the whole level' if cl else 'left some'}: {len(g):,} fills followed by another; next fill's "
          f"price against this one {np.log(g.next_price / g.avg_price).mean():+.2%} (at or above it "
          f"{(g.next_price >= g.avg_price - 1e-9).mean():.0%}); next fill's CLV {g.next_clv.mean():+.2%}")

print("\n== 4. the day limit (GBP4,000)")
print(pd.DataFrame(limits).round(2).to_string(index=False))
if refused:
    r = pd.concat(refused, ignore_index=True)
    r = r[r.error.astype(str).str.contains("limit|turnover|minimum stake", case=False, regex=True)]
    r = r.dropna(subset=["best_back", "B_book"])
    r = r[(r.B_book > 1) & (r.best_back > 1)]
    r = r.drop_duplicates(["day", "market_id", "selection_id"])
    r["clv"] = r.best_back / r.B_book - 1
    r["w"] = np.clip(400 / (r.best_back - 1), 15, 200)
    print(f"   backs refused after the limit: {len(r):,} horses; expected CLV mean {r.edge.mean():+.2%}; CLV at the BSP "
          f"at the price shown {r.clv.mean():+.2%}, staked as the rule would {(r.clv * r.w).sum() / r.w.sum():+.2%} "
          f"on GBP{r.w.sum():,.2f}")
    print(r.groupby("day").apply(lambda g: pd.Series({"horses": len(g), "rule_stake": g.w.sum(),
                                                      "clv": (g.clv * g.w).sum() / g.w.sum()})).round(3).to_string())
