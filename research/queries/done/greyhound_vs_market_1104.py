"""Our greyhound prices against the market, race by race, 4 Nov (the owner's ask; read-only).

Our price: the live model's (sources/greyhound/live/predictions/<day>.csv, the log-BSP model from 11:32 UK), renormalised
over the dogs that ran. The market: Betfair's BSP (the settled books), and the book's mid (the average of the best back's
and best lay's implied chances) at the last read 5 and 1 minutes before the off, each normalised over the runners. A race
counts once its result is in. Per race: the chance each gave the winner, the log loss (lower is better), the
favourites, the rank agreement with the BSP. Printed, and the rows as CSV between the markers for the report."""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from betfair_recorder import _read_day_csv          # noqa: E402
from sources.common import Store                     # noqa: E402

DAY = pd.Timestamp("2026-11-04").date()
store = Store()
raw = store.get(f"greyhound/live/predictions/{DAY:%Y-%m-%d}.csv")
if not raw:
    raise SystemExit("no data found: no live predictions for the day")
pred = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
print("predictions:", len(pred), "runners,", pred.market_id.nunique(), "markets; objective",
      sorted(set(pred.get("model_objective", pd.Series(["?"])).astype(str))))
pred["selection_id"] = pd.to_numeric(pred.selection_id, errors="coerce")
pred = pred.dropna(subset=["market_id", "selection_id", "p_model"])

books, used = [], []
for name in ("books_greyhound_live.csv", "books_greyhound.csv"):
    b = _read_day_csv(DAY, name)
    if b is not None and len(b):
        books.append(b.assign(_file=name))
        used.append(f"{name} {len(b)} rows")
print("books:", "; ".join(used) or "no data found")
if not books:
    raise SystemExit("no data found: no recorded greyhound books for the day")
bk = pd.concat(books, ignore_index=True)
bk["market_id"] = bk.market_id.astype(str)
bk["selection_id"] = pd.to_numeric(bk.selection_id, errors="coerce")
bk["t"] = pd.to_datetime(bk.polled_utc, utc=True, format="ISO8601", errors="coerce")
for c in ("back1", "lay1", "sp_actual"):
    bk[c] = pd.to_numeric(bk[c], errors="coerce")

mk = None
for name in ("markets_greyhound_live.csv", "markets_greyhound.csv"):
    m = _read_day_csv(DAY, name)
    if m is not None and len(m):
        mk = m if mk is None else pd.concat([mk, m], ignore_index=True)
if mk is None:
    raise SystemExit("no data found: no market catalogue for the day")
mk["market_id"] = mk.market_id.astype(str)
off = (mk.drop_duplicates("market_id").set_index("market_id").market_start_utc
       .pipe(pd.to_datetime, utc=True, format="ISO8601", errors="coerce"))

# the result and the BSP: each runner's settled read (one with a BSP over one without, then the latest)
fin = bk[bk.source.astype(str) == "final"].copy()
fin = (fin.assign(_has=fin.sp_actual.notna()).sort_values(["_has", "t"])
       .drop_duplicates(["market_id", "selection_id"], keep="last"))
settled = set(fin[fin.runner_status.astype(str) == "WINNER"].market_id)
print("settled races with a winner in the books:", len(settled))

pre = bk[(bk.source.astype(str) != "final") & bk.back1.gt(1) & bk.lay1.gt(1)].copy()
pre["off"] = pre.market_id.map(off)


def mid_at(minutes: int) -> pd.Series:
    """Each runner's mid chance at the last book read at or before the off less ``minutes`` (within 3 min of it)."""
    cut = pre[(pre.t <= pre.off - pd.Timedelta(minutes=minutes)) &
              (pre.t >= pre.off - pd.Timedelta(minutes=minutes + 3))]
    last = cut.sort_values("t").drop_duplicates(["market_id", "selection_id"], keep="last")
    return last.set_index(["market_id", "selection_id"]).pipe(lambda d: 0.5 * (1 / d.back1 + 1 / d.lay1))


mid5, mid1 = mid_at(5), mid_at(1)

rows, runners, dropped = [], [], {}
for mid, f in fin[fin.market_id.isin(settled)].groupby("market_id"):
    ran = f[f.runner_status.astype(str).isin(["WINNER", "LOSER", "PLACED"])]
    p = pred[pred.market_id == mid].merge(ran[["selection_id", "runner_status", "sp_actual"]], on="selection_id")
    why = ("not priced by our model" if not (pred.market_id == mid).any() else
           "fewer than 2 of our runners matched the runners" if len(p) < 2 else
           "the winner not among our priced runners" if not (p.runner_status == "WINNER").any() else
           "a runner without a BSP yet" if p.sp_actual.isna().any() else None)
    if why:
        dropped.setdefault(why, []).append(mid)
        continue
    p["ours"] = p.p_model / p.p_model.sum()
    p["bsp_p"] = (1 / p.sp_actual) / (1 / p.sp_actual).sum()
    for col, s in (("mid5_p", mid5), ("mid1_p", mid1)):
        v = p.selection_id.map(lambda sid: s.get((mid, sid), np.nan))
        p[col] = v / v.sum() if v.notna().all() else np.nan
    p["won"] = (p.runner_status == "WINNER").astype(int)
    w = p[p.won == 1].iloc[0]
    fav_o, fav_m = p.loc[p.ours.idxmax()], p.loc[p.bsp_p.idxmax()]
    t = off.get(mid)
    rows.append({
        "off_uk": t.tz_convert("Europe/London").strftime("%H:%M") if pd.notna(t) else str(p.race_time.iloc[0]),
        "track": str(p.track.iloc[0]), "market_id": mid, "runners": len(p), "winner_trap": int(w.trap),
        "winner_bsp": round(float(w.sp_actual), 2), "our_price_winner": round(float(1 / w.ours), 2),
        "our_p_winner": round(float(w.ours), 3), "bsp_p_winner": round(float(w.bsp_p), 3),
        "mid1_p_winner": round(float(w.mid1_p), 3) if pd.notna(w.mid1_p) else None,
        "logloss_ours": round(float(-np.log(w.ours)), 3), "logloss_bsp": round(float(-np.log(w.bsp_p)), 3),
        "logloss_mid5": round(float(-np.log(w.mid5_p)), 3) if pd.notna(w.mid5_p) else None,
        "logloss_mid1": round(float(-np.log(w.mid1_p)), 3) if pd.notna(w.mid1_p) else None,
        "our_fav_trap": int(fav_o.trap), "our_fav_won": int(fav_o.won), "bsp_fav_trap": int(fav_m.trap),
        "bsp_fav_won": int(fav_m.won),
        "rank_corr": round(float(p.ours.rank().corr(p.bsp_p.rank())), 2),
        "mean_abs_log_gap": round(float(np.abs(np.log(p.ours / p.bsp_p)).mean()), 3),
        "ours_beat_bsp": int(w.ours > w.bsp_p)})
    runners.append(p.assign(market_id=mid))

print("races dropped:", {k: len(v) for k, v in dropped.items()} or "none")
for k, v in dropped.items():
    if k == "not priced by our model":
        names = mk[mk.market_id.isin(v)].drop_duplicates("market_id")
        print("   not priced:", "; ".join(f"{a} {b}" for a, b in zip(names.venue, names.market_start_utc)))
r = pd.DataFrame(rows)
if not len(r):
    raise SystemExit("no data found: no settled race with our price, a BSP for every runner and a winner")
r = r.sort_values("off_uk").reset_index(drop=True)
both = r.dropna(subset=["logloss_mid5", "logloss_mid1"])
R = pd.concat(runners, ignore_index=True)

print(f"\n== {len(r)} settled races, {len(R)} runners (from {r.off_uk.min()} to {r.off_uk.max()} UK)")
print(f"mean log loss on the winner: ours {r.logloss_ours.mean():.3f}; BSP {r.logloss_bsp.mean():.3f}"
      f" (ours minus BSP {r.logloss_ours.mean() - r.logloss_bsp.mean():+.3f}; lower is better)")
if len(both):
    print(f"   on the {len(both)} races with both mids: ours {both.logloss_ours.mean():.3f}; mid 5 min "
          f"{both.logloss_mid5.mean():.3f}; mid 1 min {both.logloss_mid1.mean():.3f}; BSP {both.logloss_bsp.mean():.3f}")
d = r.logloss_ours - r.logloss_bsp
print(f"   ours minus BSP a race: mean {d.mean():+.3f}, sd {d.std():.3f}, se {d.std() / np.sqrt(len(d)):.3f}")
print(f"races where we gave the winner more chance than the BSP: {int(r.ours_beat_bsp.sum())} of {len(r)}")
print(f"favourites won: ours {int(r.our_fav_won.sum())} of {len(r)}; the BSP's {int(r.bsp_fav_won.sum())} of {len(r)};"
      f" same favourite in {int((r.our_fav_trap == r.bsp_fav_trap).sum())} races")
print(f"rank agreement with the BSP (Spearman, a race): mean {r.rank_corr.mean():.2f}; mean |log(ours/BSP)| a runner "
      f"{np.abs(np.log(R.ours / R.bsp_p)).mean():.3f}")
# calibration of our chances against the outcome, by band
R["band"] = pd.cut(R.ours, [0, .1, .15, .2, .3, .45, 1])
cal = R.groupby("band", observed=True).agg(runners=("won", "size"), ours=("ours", "mean"), bsp=("bsp_p", "mean"),
                                          won=("won", "mean")).round(3)
print("\n== our chance by band: mean ours, mean BSP chance, the win rate\n" + cal.to_string())
by_track = r.groupby("track").agg(races=("market_id", "size"), ll_ours=("logloss_ours", "mean"),
                                 ll_bsp=("logloss_bsp", "mean"), beat=("ours_beat_bsp", "sum")).round(3)
print("\n== by track\n" + by_track.to_string())
print("\n== every race\n" + r.drop(columns=["market_id"]).to_string(index=False))
print("\n---RACES CSV---")
print(r.to_csv(index=False), end="")
print("---END RACES CSV---")
