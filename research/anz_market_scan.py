"""Australia and New Zealand thoroughbreds in Betfair's own ANZ files (the owner's ask, 2 Nov 2026; read-only).

Source: Betfair Data Scientists, https://betfair-datascientists.github.io/data/dataListing/ (ANZ_Thoroughbreds_<year>.zip
for 2020-2025, ANZ_Thoroughbreds_2026_<month>.csv for 2026). Download them into an empty directory of their own, unzip
the zips there, and pass that directory (the files are untrusted data: run with python -I):

    python -I research/anz_market_scan.py <dir>                       # the scan, banded by the BSP
    python -I research/anz_market_scan.py <dir> --ex-ante             # the price comparisons banded by the pre-off price

Results: reports/au_nz_market_scan.md. Nothing here touches the live UK/IE system.
"""
import sys
if "--ex-ante" in sys.argv:
    sys.argv.remove("--ex-ante")
    MODE = "exante"
else:
    MODE = "scan"
if MODE == "scan":
    import sys, glob, numpy as np, pandas as pd
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    fs = sorted(glob.glob(sys.argv[1] + "/*.csv"))
    frames = []
    for f in fs:
        x = pd.read_csv(f, dtype=str, keep_default_na=False); x["FILE"] = f.rsplit("/", 1)[-1]; frames.append(x)
    d = pd.concat(frames, ignore_index=True)
    print(f"files {len(fs)}, rows in {len(d):,}")
    raw = d.LOCAL_MEETING_DATE.str.strip()
    iso = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")
    d["date"] = iso.fillna(dmy)
    # the file's own month must agree with the date read (catches a d/m swap)
    fm = d.FILE.str.extract(r"(\d{4})_(\d{2})")
    agree = (d.date.dt.year == fm[0].astype(int)) & (d.date.dt.month == fm[1].astype(int))
    print(f"dates unread {d.date.isna().sum():,}; date outside its file's month {(~agree & d.date.notna()).sum():,}")
    num = ["WIN_BSP", "WIN_BSP_VOLUME", "WIN_PREPLAY_MAX_PRICE_TAKEN", "WIN_PREPLAY_MIN_PRICE_TAKEN",
           "WIN_PREPLAY_LAST_PRICE_TAKEN", "WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN", "WIN_PREPLAY_VOLUME", "WIN_INPLAY_VOLUME",
           "BEST_AVAIL_BACK_AT_SCHEDULED_OFF", "BEST_AVAIL_LAY_AT_SCHEDULED_OFF", "BACK_MARKET_PERCENTAGE_AT_SCHEDULED_OFF"]
    for c in num: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["won"] = (d.WIN_RESULT == "WINNER").astype(int)
    dup = d.duplicated(["WIN_MARKET_ID", "SELECTION_ID"]).sum()
    print(f"duplicate (market, selection) {dup:,} (dropped)"); d = d.drop_duplicates(["WIN_MARKET_ID", "SELECTION_ID"])
    for c in ["WIN_MARKET_ID", "SELECTION_ID", "TRACK", "STATE_CODE", "WIN_RESULT"]:
        print(f"  null/blank {c}: {(d[c].astype(str).str.strip() == '').sum():,}")
    print(f"  no BSP {(~(d.WIN_BSP > 1)).sum():,}; no pre-play WAP {(~(d.WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN > 1)).sum():,}; "
          f"no best back at the off {(~(d.BEST_AVAIL_BACK_AT_SCHEDULED_OFF > 1)).sum():,}")
    w = d.groupby("WIN_MARKET_ID").won.sum()
    print(f"  markets {len(w):,}; with exactly one winner {(w == 1).sum():,}, none {(w == 0).sum():,}, dead heats {(w > 1).sum():,}")
    print(f"date range {d.date.min().date()} to {d.date.max().date()}")
    d["country"] = np.where(d.STATE_CODE == "NZ", "NZ", "AU")
    d["year"] = d.date.dt.year.astype(str)
    last = d.date.max()
    print("months held per year:", d.groupby("year").date.apply(lambda s: s.dt.month.nunique()).to_dict())
    
    print("\n== volume: races, races a day, pre-play money (unit as Betfair writes it; presumably AUD)")
    mk = d.groupby(["year", "country", "WIN_MARKET_ID"]).agg(pp=("WIN_PREPLAY_VOLUME", "sum"), bsp=("WIN_BSP_VOLUME", "sum"),
         ip=("WIN_INPLAY_VOLUME", "sum"), day=("date", "first"), runners=("SELECTION_ID", "size"),
         book=("BACK_MARKET_PERCENTAGE_AT_SCHEDULED_OFF", "first")).reset_index()
    mk["bsp_share"] = mk.bsp / (mk.pp + mk.bsp)
    g = mk.groupby(["year", "country"])
    v = pd.DataFrame({"races": g.size(), "days": g.day.nunique()})
    v["a_day"] = (v.races / v.days).round(1)
    v["field"] = g.runners.median()
    v["mkt_pp_med"] = g.pp.median().round(0)
    v["mkt_pp_p90"] = g.pp.quantile(0.9).round(0)
    v["bsp_share_med"] = g.bsp_share.median().round(3)
    v["back_book_off"] = g.book.median().round(3)
    rr = d.groupby(["year", "country"]).WIN_PREPLAY_VOLUME
    v["run_pp_med"] = rr.median().round(0); v["run_pp_p90"] = rr.quantile(0.9).round(0)
    print(v.to_string())
    print("\nshare of runners with pre-play money of at least 500 / 2,000 / 10,000, by country (2025):")
    y = d[d.year == "2025"]
    for c, gg in y.groupby("country"):
        print(f"  {c}: " + ", ".join(f">={t:,}: {(gg.WIN_PREPLAY_VOLUME >= t).mean():.0%}" for t in (500, 2000, 10000)))
    print("\n2025 runner pre-play median by state:", y.groupby("STATE_CODE").WIN_PREPLAY_VOLUME.median().round(0).sort_values().to_dict())
    print("2025 by weekday (AU, market pre-play median):",
          mk[(mk.year == "2025") & (mk.country == "AU")].groupby(mk.day.dt.day_name()).pp.median().round(0).to_dict())
    
    BANDS = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 51.0, 1001.0]
    x = d[(d.WIN_BSP > 1)].copy()
    x["band"] = pd.cut(x.WIN_BSP, BANDS)
    for c in ["AU", "NZ"]:
        xc = x[x.country == c]
        print(f"\n== {c}: median log(BSP / pre-play WAP) by BSP band and year (+: the BSP bigger)")
        z = xc[xc.WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN > 1]
        print(z.assign(lr=np.log(z.WIN_BSP / z.WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN)).pivot_table(
            index="band", columns="year", values="lr", aggfunc="median", observed=True).round(3).to_string())
        print(f"== {c}: pre-play range, median log(max / min price taken) by BSP band and year")
        z = xc[(xc.WIN_PREPLAY_MAX_PRICE_TAKEN > 1) & (xc.WIN_PREPLAY_MIN_PRICE_TAKEN > 1)]
        print(z.assign(lr=np.log(z.WIN_PREPLAY_MAX_PRICE_TAKEN / z.WIN_PREPLAY_MIN_PRICE_TAKEN)).pivot_table(
            index="band", columns="year", values="lr", aggfunc="median", observed=True).round(3).to_string())
        print(f"== {c}: the best back at the scheduled off against the BSP: mean (BSP / back - 1), %, by BSP band and year")
        z = xc[xc.BEST_AVAIL_BACK_AT_SCHEDULED_OFF > 1]
        print((100 * z.assign(c=z.WIN_BSP / z.BEST_AVAIL_BACK_AT_SCHEDULED_OFF - 1).pivot_table(
            index="band", columns="year", values="c", aggfunc="mean", observed=True)).round(1).to_string())
        print(f"== {c}: a level back at the BSP, % a unit, before commission, by BSP band and year")
        z = xc.assign(ret=np.where(xc.won == 1, xc.WIN_BSP - 1, -1.0), net=np.where(xc.won == 1, (xc.WIN_BSP - 1) * 0.98, -1.0))
        print((100 * z.pivot_table(index="band", columns="year", values="ret", aggfunc="mean", observed=True)).round(1).to_string())
        print(f"   after 2% commission on winnings, all years pooled: " + ", ".join(
            f"{b}: {100 * gg.net.mean():+.1f}% (n={len(gg):,})" for b, gg in z.groupby("band", observed=True)))
        print(f"   runners a band and year:\n" + z.pivot_table(index="band", columns="year", values="ret", aggfunc="size", observed=True).to_string())
        # log loss of the BSP book, normalised per race
        q = xc.copy(); q["p"] = 1 / q.WIN_BSP; q["p"] = q.p / q.groupby("WIN_MARKET_ID").p.transform("sum")
        ll = -np.log(q[q.won == 1].p.clip(1e-6)).groupby(q[q.won == 1].year).mean()
        print(f"   the BSP book's race log-loss by year: {ll.round(3).to_dict()}")
else:
    import sys, glob, numpy as np, pandas as pd
    pd.set_option("display.width", 220)
    fs = sorted(glob.glob(sys.argv[1] + "/*.csv"))
    use = ["LOCAL_MEETING_DATE","WIN_MARKET_ID","SELECTION_ID","STATE_CODE","WIN_RESULT","WIN_BSP",
           "WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN","BEST_AVAIL_BACK_AT_SCHEDULED_OFF","WIN_PREPLAY_VOLUME"]
    d = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, usecols=use) for f in fs], ignore_index=True)
    d = d.drop_duplicates(["WIN_MARKET_ID","SELECTION_ID"])
    raw = d.LOCAL_MEETING_DATE.str.strip()
    d["year"] = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce").fillna(pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")).dt.year
    for c in use[5:]: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["won"] = (d.WIN_RESULT == "WINNER").astype(int)
    d["country"] = np.where(d.STATE_CODE == "NZ", "NZ", "AU")
    B = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 51.0, 1001.0]
    x = d[(d.WIN_BSP > 1)]
    for c in ["AU", "NZ"]:
        xc = x[x.country == c]
        z = xc[xc.WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN > 1].copy()
        z["band"] = pd.cut(z.WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN, B)
        z["lr"] = np.log(z.WIN_BSP / z.WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN)
        print(f"== {c}: median log(BSP / pre-play WAP), banded by the WAP")
        print(z.pivot_table(index="band", columns="year", values="lr", aggfunc="median", observed=True).round(3).to_string())
        z = xc[xc.BEST_AVAIL_BACK_AT_SCHEDULED_OFF > 1].copy()
        z["band"] = pd.cut(z.BEST_AVAIL_BACK_AT_SCHEDULED_OFF, B)
        z["c"] = z.WIN_BSP / z.BEST_AVAIL_BACK_AT_SCHEDULED_OFF - 1
        z["back_off"] = np.where(z.won == 1, (z.BEST_AVAIL_BACK_AT_SCHEDULED_OFF - 1) * 0.98, -1.0)
        z["back_bsp"] = np.where(z.won == 1, (z.WIN_BSP - 1) * 0.98, -1.0)
        print(f"== {c}: mean (BSP / best back at the scheduled off - 1), %, banded by the best back at the off")
        print((100 * z.pivot_table(index="band", columns="year", values="c", aggfunc="mean", observed=True)).round(1).to_string())
        print(f"== {c}: level back after 2% commission, all years: at the best back at the off vs at the BSP, by the off price band")
        t = z.groupby("band", observed=True).agg(n=("c", "size"), at_off=("back_off", "mean"), at_bsp=("back_bsp", "mean"))
        t[["at_off", "at_bsp"]] = (100 * t[["at_off", "at_bsp"]]).round(1)
        print(t.to_string())
    # the NZ 8-13 BSP cell: finer bands, with a standard error
    n = x[x.country == "NZ"].copy()
    n["ret"] = np.where(n.won == 1, (n.WIN_BSP - 1) * 0.98, -1.0)
    n["fb"] = pd.cut(n.WIN_BSP, [5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 21])
    t = n.groupby("fb", observed=True).ret.agg(["size", "mean", "std"])
    t["se"] = t["std"] / np.sqrt(t["size"]); t[["mean", "se"]] = (100 * t[["mean", "se"]]).round(1)
    print("== NZ: level back at the BSP after 2%, finer BSP bands (%, se)"); print(t[["size", "mean", "se"]].to_string())
