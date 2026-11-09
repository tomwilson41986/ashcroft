"""Is a non-runner forecast by market worth building? (read-only, descriptive; ledger `removal-by-market-1109`)

reductions-1021 found the live backs' settled CLV sits where a runner is still to come out (no removal to come +1.38%,
one +7.0%, two +9.0%, 10-21 Oct), and replay-nr-bar-1021 closed with "a removal forecast by market (not by price) is
the only form that could work". Nothing has been built. Before building it, four questions on the trader's own books
(done/closing_model_by_hour_1007.py's reading: the book nearest each hour mark, every runner's expected CLV recomputed
as the trader does, the BSP, and Betfair's reduction factors for runners removed after the read), at the hours the
rule trades (06:00-10:00 UK), 2 Oct on:

1. Mechanical or selection? If EVERY runner (not only the rule's) closes better after the cut where a removal is still
   to come, the gain is in the reduction factor itself, not in what the rule picks.
2. Why? The removed runner's factor against its share of the book at its last read before it came out (and removals
   under 2.5%, which cut nothing).
3. Can it be seen coming at the read? The chance of a cut removal by field size, time to the off, removals already at
   the meeting; a logistic on those, walk-forward by day, scored by AUC on days it has not seen.
4. Would a tilt have paid? The rule's runners at their first crossing, by the forecast's third.

One descriptive pass, no rule fitted or proposed here; a runner is counted at every mark it is read at, so marks are
not independent, and the days are few. Nothing here touches the live trader.
"""
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
path = os.path.join(os.getcwd(), "research", "queries", "done", "closing_model_by_hour_1007.py")
src = open(path, encoding="utf-8").read()
head = src[:src.index("\nrows = []\n")]
old = "MARKS = list(range(8, 21))"
assert old in head
head = head.replace(old, "MARKS = [6, 7, 8, 9, 10]")
G = {"__name__": "removal_by_market", "__file__": path}
exec(compile(head, path, "exec"), G)
books, catalogue, predictions, settled, expected = (G[k] for k in ("books", "catalogue", "predictions", "settled",
                                                                    "expected"))
cfg, lim, rb, KEY, UK, MARKS, TOL = (G[k] for k in ("cfg", "lim", "rb", "KEY", "UK", "MARKS", "TOL"))
FIRST, LAST = G["FIRST"], G["LAST"]

rows, removed_rows = [], []
day = FIRST
while day <= LAST:
    bt = books(f"betfair_live/{day}/books_trader.csv.gz")
    cat = catalogue(day)
    prices = predictions(day, cat)
    bsp, rem = settled(day)
    if bt.empty or not prices or bsp.empty:
        print(f"{day}: the trader's books {len(bt)}, model prices {len(prices)}, BSPs {len(bsp)}: not read")
        day += timedelta(days=1)
        continue
    c1 = cat.drop_duplicates("market_id").set_index("market_id")
    offs = c1.market_start_utc.map(lambda s: pd.Timestamp(s)).to_dict()
    venue = c1.venue.astype(str).to_dict()
    country = c1.country.astype(str).to_dict() if "country" in c1 else {}
    bt = bt[bt.market_id.isin({m for m, _ in prices})]
    rem = rem[rem.market_id.isin(set(bt.market_id))].copy()
    rem["venue"] = rem.market_id.map(venue)
    part = []
    for h in MARKS:
        target = pd.Timestamp(day, tz=UK) + pd.Timedelta(hours=h)
        x = bt[(bt.t - target).abs() <= pd.Timedelta(minutes=TOL)]
        if x.empty:
            continue
        polls = x.groupby(["market_id", "polled_utc"], as_index=False).agg(t=("t", "first"))
        polls["gap"] = (polls.t - target).abs()
        pick = polls.sort_values("gap").drop_duplicates("market_id")[["market_id", "polled_utc"]]
        snap = x.merge(pick, on=["market_id", "polled_utc"])
        for mid, g in snap.groupby("market_id"):
            off, t = offs.get(mid), g.t.iloc[0]
            if off is None or (off - t) <= pd.Timedelta(minutes=15):
                continue
            e = expected(g, prices)
            if e is None:
                continue
            later = rem[(rem.market_id == mid) & (rem.removed_at > t)]
            ven = venue.get(mid)
            e["market_id"], e["t"], e["mark"], e["day"] = mid, t, h, str(day)
            e["to_off"] = (off - t).total_seconds() / 60.0
            e["field"] = len(e)
            e["venue"], e["ie"] = ven, int(country.get(mid) == "IE")
            e["cut_to_come"] = int((later.rf >= 2.5).sum())
            e["small_to_come"] = int((later.rf < 2.5).sum())
            e["reduction"] = min(float(later.rf[later.rf >= 2.5].sum()), 90.0) / 100.0
            e["out_at_meeting"] = int(((rem.venue == ven) & (rem.removed_at <= t)).sum())
            part.append(e)
    # each removal after the trader's first read: its factor against its share of the book at its last active read
    act = bt[bt.runner_status.fillna("ACTIVE").eq("ACTIVE") & (bt.back1 > 1)]
    for r in rem.itertuples():
        if pd.isna(r.removed_at):
            continue
        mine = act[(act.market_id == r.market_id) & (act.selection_id == r.selection_id)
                   & (act.t < r.removed_at)].sort_values("t")
        if mine.empty:
            continue
        last = mine.polled_utc.iloc[-1]
        s = act[(act.market_id == r.market_id) & (act.polled_utc == last)]
        with np.errstate(invalid="ignore", divide="ignore"):
            now = rb.market_now(s.back1.to_numpy(float), s.lay1.to_numpy(float))
        q = (1.0 / now) / (1.0 / now).sum()
        share = float(q[(s.selection_id == r.selection_id).to_numpy()][0])
        removed_rows.append({"day": str(day), "market_id": r.market_id, "rf": float(r.rf), "share": share,
                             "field": len(s), "mins_before_removal": (r.removed_at - mine.t.iloc[-1]).total_seconds() / 60,
                             "uk_hour": r.removed_at.tz_convert(UK).hour})
    if part:
        p = pd.concat(part).merge(bsp, on=KEY, how="left")
        rows.append(p)
    print(f"{day}: {sum(len(x) for x in part):,} runner-marks; removals in the trader's markets {len(rem)} "
          f"({int((rem.rf >= 2.5).sum())} with a cut)", flush=True)
    day += timedelta(days=1)

if not rows:
    raise SystemExit("nothing to read")
A = pd.concat(rows, ignore_index=True)
print(f"\nread: {len(A):,} runner-marks in {A.market_id.nunique():,} markets over {A.day.nunique()} days "
      f"({A.day.min()} .. {A.day.max()}); runners later removed (no BSP, the back is void): "
      f"{int((~(A.bsp > 1)).sum()):,}")
D = A[np.isfinite(A.back) & (A.back > 1) & np.isfinite(A.ev) & (A.bsp > 1)].copy()
D["clv"] = D.back / D.bsp - 1.0
D["clv_cut"] = D.back * (1.0 - D.reduction) / D.bsp - 1.0
D["band"] = np.where(D.cut_to_come == 0, np.where(D.small_to_come > 0, "0 cut, a small one", "none"),
                     np.where(D.cut_to_come == 1, "1 cut", "2+ cuts"))
BANDS = ["none", "0 cut, a small one", "1 cut", "2+ cuts"]
D["chosen"] = ((D.ev >= cfg.clv_bar) & (D.back >= D.model_price) & (D.back >= lim.min_price)
               & (D.back <= lim.max_price))
D["stake"] = np.where(D.chosen, np.clip(cfg.target / (D.back - 1.0), cfg.stake_floor, lim.max_stake), 0.0)


def day_interval(g, num, den=None, n=3000, seed=0):
    """A 90% interval by day bootstrap for mean(num), or sum(num)/sum(den)."""
    by = g.groupby("day")
    a = by[num].sum().to_numpy(float)
    b = by[den].sum().to_numpy(float) if den else by[num].size().to_numpy(float)
    if len(a) < 3 or b.sum() <= 0:
        return "n/a"
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(a), (n, len(a)))
    r = a[i].sum(1) / np.maximum(b[i].sum(1), 1e-12)
    return f"{np.percentile(r, 5):+.4f} to {np.percentile(r, 95):+.4f}"


print("\n== 1. every runner read, by the removals still to come in its market (level; 90% by day)")
for name, g in ((b, D[D.band == b]) for b in BANDS):
    if len(g):
        print(f"  {name:20s} {len(g):7,} runner-marks, {g.market_id.nunique():5,} markets: CLV at the back "
              f"{g.clv.mean():+.4f}, after the cut {g.clv_cut.mean():+.4f} ({day_interval(g, 'clv_cut')}), "
              f"median after the cut {g.clv_cut.median():+.4f}")
F1 = D.sort_values("t").drop_duplicates(KEY)
print("   the same, each runner once (its first read):")
for name, g in ((b, F1[F1.band == b]) for b in BANDS):
    if len(g):
        print(f"  {name:20s} {len(g):7,} runners: after the cut {g.clv_cut.mean():+.4f} ({day_interval(g, 'clv_cut')})")

C = D[D.chosen].copy()
C["sx"] = C.stake * C.clv_cut
print(f"\n== the rule's runners (expected CLV >= {cfg.clv_bar:.0%}, at or over our price), staked to win "
      f"GBP{cfg.target:.0f}: {len(C):,} runner-marks")
for name, g in ((b, C[C.band == b]) for b in BANDS):
    if len(g):
        print(f"  {name:20s} {len(g):6,} marks, GBP{g.stake.sum():9,.0f} ({g.stake.sum() / C.stake.sum():5.1%}): "
              f"expected {(g.stake * g.ev).sum() / g.stake.sum():+.4f}, after the cut "
              f"{g.sx.sum() / g.stake.sum():+.4f} ({day_interval(g, 'sx', 'stake')})")
N = D[~D.chosen]
print("   and the runners the rule did NOT choose (level, after the cut): "
      + "; ".join(f"{b} {N[N.band == b].clv_cut.mean():+.4f} (n {int((N.band == b).sum()):,})" for b in BANDS
                  if (N.band == b).any()))

print("\n== 2. the removed runners: Betfair's factor against the runner's share of the book at its last read")
R = pd.DataFrame(removed_rows)
if len(R):
    R["gain"] = (1.0 - np.where(R.rf >= 2.5, R.rf, 0.0) / 100.0) / (1.0 - R.share) - 1.0
    for name, g in (("with a cut (factor >= 2.5%)", R[R.rf >= 2.5]), ("under 2.5% (no cut)", R[R.rf < 2.5])):
        if len(g):
            print(f"  {name:28s} {len(g):5,} removals: factor mean {g.rf.mean():.2f}%, share of the book at the last "
                  f"read {100 * g.share.mean():.2f}% (median {100 * g.share.median():.2f}%), factor under the share on "
                  f"{(g.rf / 100.0 < g.share).mean():.0%}; what the others gain if the book closes at that share: "
                  f"mean {g.gain.mean():+.4f}, median {g.gain.median():+.4f}; last read a median "
                  f"{g.mins_before_removal.median():.0f} min before it came out")
    print("  removals by UK hour:", R.groupby("uk_hour").size().to_dict())
else:
    print("  no removal read")

print("\n== 3. can a cut removal be seen coming? each market at its first read of the day")
M = (D.sort_values("t").drop_duplicates("market_id")
     [["day", "market_id", "mark", "field", "to_off", "ie", "out_at_meeting", "cut_to_come", "small_to_come", "venue"]]
     .copy())
M["y"] = (M.cut_to_come > 0).astype(int)
print(f"  {len(M):,} markets over {M.day.nunique()} days: a cut removal still to come in {M.y.mean():.1%}; any removal "
      f"in {((M.cut_to_come + M.small_to_come) > 0).mean():.1%}")
M["field_band"] = pd.cut(M.field, [0, 6, 9, 12, 100], labels=["<=6", "7-9", "10-12", "13+"])
print(M.groupby("field_band", observed=True).agg(markets=("y", "size"), cut_to_come=("y", "mean")).round(3).to_string())
print(M.groupby("mark").agg(markets=("y", "size"), cut_to_come=("y", "mean")).round(3).to_string())
print(M.assign(out=M.out_at_meeting.clip(upper=2)).groupby("out").agg(markets=("y", "size"),
                                                                     cut_to_come=("y", "mean")).round(3).to_string())
mt = M.groupby(["day", "venue"]).y.agg(["sum", "size"])
M = M.merge(mt.rename(columns={"sum": "m_sum", "size": "m_n"}).reset_index(), on=["day", "venue"]).reset_index(drop=True)
oth = (M.m_sum - M.y) / np.maximum(M.m_n - 1, 1)
print(f"  the meeting: where another race at the meeting has a cut removal to come the chance is "
      f"{M.y[oth > 0].mean():.1%}; where none has, {M.y[oth == 0].mean():.1%} (hindsight: the others' removals are "
      f"not known at the read)")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

X_COLS = ["field", "log_to_off", "ie", "out_at_meeting"]
M["log_to_off"] = np.log(M.to_off.clip(lower=15))
days = sorted(M.day.unique())
M["p"] = np.nan
for d in days[10:]:
    tr, te = M[M.day < d], M[M.day == d]
    if tr.y.nunique() < 2:
        continue
    mu, sd = tr[X_COLS].mean(), tr[X_COLS].std().replace(0, 1)
    m = LogisticRegression(C=1.0, max_iter=1000).fit((tr[X_COLS] - mu) / sd, tr.y)
    M.loc[te.index, "p"] = m.predict_proba((te[X_COLS] - mu) / sd)[:, 1]
S = M.dropna(subset=["p"])
if len(S) and S.y.nunique() == 2:
    auc = roc_auc_score(S.y, S.p)
    rng = np.random.default_rng(0)
    ds = sorted(S.day.unique())
    boots = []
    for _ in range(1000):
        b = pd.concat([S[S.day == d] for d in rng.choice(ds, len(ds))])
        if b.y.nunique() == 2:
            boots.append(roc_auc_score(b.y, b.p))
    print(f"  logistic on {X_COLS}, walk-forward by day from day 11: {len(S):,} markets on {len(ds)} unseen days, "
          f"AUC {auc:.3f} (90% by day {np.percentile(boots, 5):.3f} to {np.percentile(boots, 95):.3f}); the forecast's "
          f"thirds: " + ", ".join(f"{g.y.mean():.1%}" for _, g in S.groupby(pd.qcut(S.p, 3, duplicates="drop"),
                                                                             observed=True)))

    print("\n== 4. the rule's runners at their first crossing, by the forecast's third (unseen days only)")
    F = C.sort_values("t").drop_duplicates(KEY).merge(S[["market_id", "p"]], on="market_id")
    if len(F) >= 30:
        F["third"] = pd.qcut(F.p.rank(method="first"), 3, labels=["low", "mid", "high"])
        for name, g in F.groupby("third", observed=True):
            print(f"  {name:5s} {len(g):5,} runners, GBP{g.stake.sum():8,.0f}: a cut removal came in "
                  f"{(g.cut_to_come > 0).mean():.1%}; CLV after the cut {g.sx.sum() / g.stake.sum():+.4f} "
                  f"({day_interval(g, 'sx', 'stake')})")
        print(f"  all   {len(F):5,} runners, GBP{F.stake.sum():8,.0f}: CLV after the cut "
              f"{F.sx.sum() / F.stake.sum():+.4f} ({day_interval(F, 'sx', 'stake')})")
    else:
        print(f"  too few of the rule's runners on unseen days ({len(F)})")
else:
    print("  too few unseen days with both outcomes for a forecast")
