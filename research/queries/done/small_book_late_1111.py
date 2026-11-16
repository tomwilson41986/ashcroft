"""The owner's small book near the off (11 Nov; read-only research, nothing reaches Betfair).

The brief: 5, 3 or 2 minutes before the off, back a subset of the field so that every horse of ours wins the same
amount - a 50/60/70/80/90% book - leaving out the horses we think are bad prices; the probabilities combine the
market's and ours (Benter). We lose the stake when a horse left out wins.

The data: the live record (betfair_live_marks, 1 Oct - 10 Nov 2026; the recorder's books are on the delayed key):
the books at off_5, off_3 and off_1 (no off_2 mark is stored), the BSP and the winner from the 'final' mark, and our
06:00 forecast from S3 predictions/<day>.csv. A race is kept at a mark when every runner active then has a back and
a lay, a forecast, a BSP, and the race has one winner and no runner removed after the mark.

1. Benter near the off: a conditional logit of the winner (and, separately, of the BSP book) on the market's
   log share at the mark and ours, walk-forward by day (each day scored by a fit on the days before, from day 8).
   Does our forecast add anything beside the market at T-5/T-3/T-1? Log-likelihood gain per race with its se.
2. The owner's book: the horses ordered by value (blend probability x best back), added while the book (sum of
   1/back) stays within the target; a dutch to an equal return (stake ~ 1/back); settled at the result after 5% on
   the race's net. Against it the same book built in market order (favourites first: no model).
3. The improvements, on the same races: (a) lay the horses left out at the best lay instead (the same bet, other
   prices); (b) the book set by the edge, not by a size: every horse with value >= 1 (+0%, +3%), and joint Kelly
   (model/portfolio.py race_kelly_exact); (c) back at the mark and close at the SP: the CLV; (d) the same stakes
   taken at the SP.
Each variant: races, return on stake with a 90% race-bootstrap interval, CLV, hit rate, and the deflated Sharpe of
its per-race returns with every variant here counted as a trial (research/overfit.py dsr_check).
"""
import io
import os
import sqlite3
import sys
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h, utc_iso_to_uk_hhmm  # noqa: E402
from model import race_book as rb  # noqa: E402
from model.portfolio import race_kelly_exact  # noqa: E402
from research.overfit import dsr_check  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
C = 0.05
D0, D1 = date(2026, 10, 1), date(2026, 11, 10)
MARKS = ["off_5", "off_3", "off_1"]
BOOKS = [0.5, 0.6, 0.7, 0.8, 0.9]
MIN_TRAIN_DAYS = 7
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


def preds():
    text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name")}
    out, d = [], D0
    while d <= D1:
        raw = get(f"predictions/{d.isoformat()}.csv")
        if raw is not None:
            x = pd.read_csv(io.BytesIO(raw), dtype=text, low_memory=False)
            x["file_day"] = d.isoformat()
            out.append(x)
        d += timedelta(days=1)
    p = pd.concat(out, ignore_index=True)
    for a, b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
        if a not in p.columns and b in p.columns:
            p[a] = p[b]
    p["race_date"] = p["race_date"].fillna(p["file_day"]).astype(str).str.slice(0, 10)
    p["predicted_bfsp"] = pd.to_numeric(p["predicted_bfsp"], errors="coerce")
    p["_k"] = (p.race_date + "|" + p.track.map(normalise_track) + "|" + p.race_time.map(race_time_to_24h).astype(str)
               + "|" + p.horse_name.map(normalise_horse))
    return p.drop_duplicates("_k", keep="last")[["_k", "predicted_bfsp"]]


# ===================================================================================== the data
mk = pd.read_sql_query(f"""SELECT race_date, market_id, selection_id, venue, market_start_utc, runner_name
                           FROM betfair_live_markets WHERE race_date >= '{D0}' AND race_date <= '{D1}'
                           AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'""", conn)
mk["_k"] = (mk.race_date + "|" + mk.venue.map(normalise_track) + "|" + mk.market_start_utc.map(utc_iso_to_uk_hhmm).astype(str)
            + "|" + mk.runner_name.map(normalise_horse))
mk = mk.merge(preds(), on="_k", how="left")
mk["selection_id"] = pd.to_numeric(mk.selection_id, errors="coerce")
mk = mk.drop_duplicates(["market_id", "selection_id"])
cols = "market_id, selection_id, mark, runner_status, back1, back1_size, lay1, lay1_size, market_total_matched, bsp"
mm = pd.read_sql_query(f"""SELECT {cols} FROM betfair_live_marks WHERE race_date >= '{D0}' AND race_date <= '{D1}'
                           AND mark IN ('final', {", ".join(repr(m) for m in MARKS)})""", conn)
mm["selection_id"] = pd.to_numeric(mm.selection_id, errors="coerce")
for c in ("back1", "back1_size", "lay1", "lay1_size", "market_total_matched", "bsp"):
    mm[c] = pd.to_numeric(mm[c], errors="coerce")
fin = mm[mm["mark"] == "final"][["market_id", "selection_id", "runner_status", "bsp"]].rename(
    columns={"runner_status": "final_status"})
print(f"cards: {mk.market_id.nunique():,} WIN markets, {len(mk):,} runners, {mk.race_date.min()} to {mk.race_date.max()}; "
      f"with a forecast {mk.predicted_bfsp.notna().mean():.1%}")
print(f"final marks: {fin.market_id.nunique():,} markets; winners {(fin.final_status == 'WINNER').sum():,}")


def race_frame(mark):
    """One row per runner active at the mark, in races kept whole; with the drop counts."""
    a = mm[(mm["mark"] == mark)].drop(columns=["bsp"])
    a = a[a.runner_status.fillna("ACTIVE").str.upper() == "ACTIVE"]
    d = a.merge(fin, on=["market_id", "selection_id"], how="left").merge(
        mk[["market_id", "selection_id", "race_date", "predicted_bfsp"]], on=["market_id", "selection_id"], how="left")
    n0 = d.market_id.nunique()
    bad = d.groupby("market_id").apply(lambda g: pd.Series({
        "no_book": bool(((g.back1 <= 1) | (g.lay1 <= 1) | g.back1.isna() | g.lay1.isna()).any()),
        "no_forecast": bool(g.predicted_bfsp.isna().any() | (g.predicted_bfsp <= 1).any()),
        "no_bsp": bool((g.final_status != "REMOVED").any() and (g.bsp.isna() | (g.bsp <= 1))[g.final_status != "REMOVED"].any()),
        "late_nr": bool((g.final_status == "REMOVED").any()),
        "winners": int((g.final_status == "WINNER").sum()),
        "small": len(g) < 3}))
    drop = bad.no_book | bad.no_forecast | bad.no_bsp | bad.late_nr | (bad.winners != 1) | bad.small
    print(f"\n{mark}: {n0:,} markets with a book; dropped: no two-sided book {int(bad.no_book.sum())}, no forecast "
          f"{int(bad.no_forecast.sum())}, no BSP {int(bad.no_bsp.sum())}, a runner removed after the mark "
          f"{int(bad.late_nr.sum())}, not one winner {int((bad.winners != 1).sum())}, under 3 runners "
          f"{int(bad.small.sum())}; kept {int((~drop).sum()):,} ({(~drop).mean():.1%})")
    d = d[d.market_id.isin(bad.index[~drop])].copy()
    d["won"] = (d.final_status == "WINNER").astype(float)
    d = d.sort_values(["race_date", "market_id", "selection_id"]).reset_index(drop=True)
    d["g"] = pd.factorize(d.market_id)[0]
    return d


def clogit(X, y, g, l2=1e-4):
    nr = g.max() + 1

    def loss(b):
        lqh = rb.group_log_softmax(X @ b, g)
        return (-float(np.sum(y * lqh)) / nr + 0.5 * l2 * float(b @ b),
                -((y - np.exp(lqh))[:, None] * X).sum(axis=0) / nr + l2 * b)

    b0 = np.zeros(X.shape[1])
    b0[0] = 1.0
    return minimize(loss, b0, jac=True, method="L-BFGS-B").x


def lq(price, g):
    return np.log(rb.normalise_within(1.0 / np.asarray(price, float), g))


def walk_forward(d):
    """Out-of-sample blend probabilities (winner-fitted) by day, from day MIN_TRAIN_DAYS + 1; the weights and the
    log-likelihood gain beside the market alone, scored on the same days."""
    d = d.copy()
    d["mid"] = rb.market_now(d.back1.to_numpy(), d.lay1.to_numpy())
    g = d.g.to_numpy()
    d["lq_m"], d["lq_f"] = lq(d.mid, g), lq(d.predicted_bfsp, g)
    d["p_mkt"] = np.exp(d.lq_m)
    d["p_blend"] = np.nan
    days = sorted(d.race_date.unique())
    ll_rows, w_rows = [], []
    for i, day in enumerate(days):
        if i < MIN_TRAIN_DAYS:
            continue
        tr, te = d[d.race_date < day], d[d.race_date == day]
        gtr = pd.factorize(tr.market_id)[0]
        gte = pd.factorize(te.market_id)[0]
        Xtr = tr[["lq_m", "lq_f"]].to_numpy()
        Xte = te[["lq_m", "lq_f"]].to_numpy()
        b = clogit(Xtr, tr.won.to_numpy(), gtr)
        b_m = clogit(Xtr[:, :1], tr.won.to_numpy(), gtr)
        lp = rb.group_log_softmax(Xte @ b, gte)
        lp_m = rb.group_log_softmax(Xte[:, :1] @ b_m, gte)
        d.loc[te.index, "p_blend"] = np.exp(lp)
        w = te.won.to_numpy().astype(bool)
        ll_rows.append(pd.DataFrame({"race": te.market_id.to_numpy()[w], "gain": lp[w] - lp_m[w]}))
        w_rows.append({"day": day, "market": b[0], "model": b[1], "market_alone": b_m[0]})
    ll = pd.concat(ll_rows) if ll_rows else pd.DataFrame(columns=["race", "gain"])
    # the BSP book: the same two inputs fitted to the SP shares (soft labels), on every race
    gall = d.g.to_numpy()
    q_bsp = rb.normalise_within(1.0 / d.bsp.to_numpy(), gall)
    b_close = clogit(d[["lq_m", "lq_f"]].to_numpy(), q_bsp, gall)
    return d[d.p_blend.notna()].copy(), ll, pd.DataFrame(w_rows), b_close


def pick_book(v, inv, target):
    """Indices added in order of v (descending) while the sum of 1/back stays within target (+2.5 points)."""
    take, tot = [], 0.0
    for i in np.argsort(-v, kind="stable"):
        if tot + inv[i] <= target + 0.025:
            take.append(i)
            tot += inv[i]
    return np.array(take, dtype=int)


def settle(stakes, price, won, comm=C):
    """Backs at price, one winner: the race's net after commission on a positive net."""
    gross = float(np.sum(stakes * (price - 1.0) * won) - np.sum(stakes * (1.0 - won)))
    return gross - comm * max(gross, 0.0)


def race_rows(d):
    """Every variant's per-race stake (or liability) and P&L; stakes normalised to 1 a race."""
    out = []
    for _, r in d.groupby("market_id", sort=False):
        back, lay, bsp = r.back1.to_numpy(), r.lay1.to_numpy(), r.bsp.to_numpy()
        won, p, pm = r.won.to_numpy(), r.p_blend.to_numpy(), r.p_mkt.to_numpy()
        inv = 1.0 / back
        day = r.race_date.iloc[0]
        base = {"race": r.market_id.iloc[0], "day": day, "field": len(r), "book_back": inv.sum(),
                "book_lay": (1.0 / lay).sum(), "matched": r.market_total_matched.max()}

        def add(name, idx, kind="back"):
            if not len(idx):
                return
            if kind == "back":
                s = np.zeros(len(r))
                s[idx] = inv[idx] / inv[idx].sum()           # an equal return on each horse backed
                pnl = settle(s, back, won)
                clv = float(np.sum(s[idx] * (back[idx] / bsp[idx] - 1.0)))
                sp = settle(s, bsp, won)                     # the same stakes taken at the SP
                out.append({**base, "variant": name, "n": len(idx), "book": inv[idx].sum(), "risk": 1.0,
                            "pnl": pnl, "clv": clv, "pnl_sp": sp, "hit": float(won[idx].sum() > 0),
                            "ev": float(np.sum(p[idx]) / inv[idx].sum() - 1.0)})
            else:                                            # lay the horses left out to an equal loss
                out_idx = np.setdiff1d(np.arange(len(r)), idx)
                if not len(out_idx):
                    return
                c = (1.0 / lay[out_idx]).sum()
                K = 1.0 / max(1.0 - c, 1e-9)                 # liability 1 a race
                l = K / lay[out_idx]
                gross = -float(np.sum(l * (lay[out_idx] - 1.0) * won[out_idx])) + float(np.sum(l * (1.0 - won[out_idx])))
                pnl = gross - C * max(gross, 0.0)
                clv = float(np.sum(l * (1.0 - lay[out_idx] / bsp[out_idx])))  # laid at lay, closed at the SP
                out.append({**base, "variant": name, "n": len(idx), "book": 1.0 - c, "risk": 1.0, "pnl": pnl,
                            "clv": clv, "pnl_sp": np.nan, "hit": float(won[idx].sum() > 0),
                            "ev": float(np.sum(p[idx]) - (1.0 - c)) / max(1.0 - c, 1e-9)})

        for B in BOOKS:
            add(f"owner_{int(B * 100)}", pick_book(p * back, inv, B))
            add(f"market_order_{int(B * 100)}", pick_book(-back, inv, B))
            add(f"lay_out_{int(B * 100)}", pick_book(p * back, inv, B), kind="lay")
        for bar in (0.0, 0.03):
            add(f"value_{int(bar * 100)}", np.flatnonzero(p * back * (1 - C) >= 1.0 + bar))
        # joint Kelly at the best back, after commission: the stakes it gives, normalised to 1
        f = race_kelly_exact(p, back, commission=C)
        if f.sum() > 1e-9:
            s = f / f.sum()
            idx = np.flatnonzero(s > 0)
            out.append({**base, "variant": "kelly", "n": len(idx), "book": inv[idx].sum(), "risk": 1.0,
                        "pnl": settle(s, back, won), "clv": float(np.sum(s * (back / bsp - 1.0))),
                        "pnl_sp": settle(s, bsp, won), "hit": float(won[idx].sum() > 0),
                        "ev": float(np.sum(s * (p * back - 1.0)))})
    return pd.DataFrame(out)


def summarise(rows, n_trials):
    res = []
    for name, g in rows.groupby("variant", sort=False):
        per = g.pnl.to_numpy() / g.risk.to_numpy()
        lo, hi = rb.ratio_interval(g.pnl.to_numpy(), g.risk.to_numpy())
        ds = dsr_check(per, n_trials=n_trials)
        res.append({"variant": name, "races": len(g), "horses": g.n.mean(), "book": g.book.mean(),
                    "hit": g.hit.mean(), "ev": g.ev.mean(), "roi": g.pnl.sum() / g.risk.sum(), "lo": lo, "hi": hi,
                    "clv": g.clv.mean(), "roi_sp": g.pnl_sp.mean() if g.pnl_sp.notna().any() else np.nan,
                    "dsr": ds["dsr"], "passed": ds["passed"] and lo > 0})
    return pd.DataFrame(res)


all_rows = []
for mark in MARKS:
    d = race_frame(mark)
    if d.empty:
        print("  nothing to score")
        continue
    scored, ll, w, b_close = walk_forward(d)
    nr = scored.market_id.nunique()
    print(f"  scored (walk-forward from day {MIN_TRAIN_DAYS + 1}): {nr:,} races, {len(scored):,} runners, "
          f"{scored.race_date.min()} to {scored.race_date.max()}")
    if len(ll):
        print(f"  Benter to the winner: our forecast beside the market at {mark}: log-likelihood gain "
              f"{ll.gain.mean() * 1000:+.1f} mnats a race (se {ll.gain.std() / np.sqrt(len(ll)) * 1000:.1f}, "
              f"{len(ll)} races); last fit's weights market {w.market.iloc[-1]:.3f}, model {w.model.iloc[-1]:.3f}")
    print(f"  the BSP book from the same two inputs (all races): market {b_close[0]:.3f}, model {b_close[1]:.3f}")
    print(f"  the market's book at the mark: back {scored.groupby('market_id').back1.apply(lambda x: (1 / x).sum()).mean():.3f}, "
          f"lay {scored.groupby('market_id').lay1.apply(lambda x: (1 / x).sum()).mean():.3f}")
    rows = race_rows(scored)
    rows["mark"] = mark
    all_rows.append(rows)

if all_rows:
    rows = pd.concat(all_rows, ignore_index=True)
    n_trials = rows.groupby(["mark", "variant"]).ngroups
    print(f"\n=== every variant ({n_trials} trials counted); return per unit staked (lay rows: per unit of liability), "
          f"5% on each race's net; interval 90% race bootstrap; pass = interval above 0 and deflated Sharpe >= 0.95")
    for mark, g in rows.groupby("mark", sort=False):
        print(f"\n{mark}")
        print(summarise(g, n_trials).round(4).to_string(index=False))
    print("\nby field size, owner_70 and value_3 (all marks):")
    for v in ("owner_70", "value_3"):
        g = rows[rows.variant == v]
        if len(g):
            print(f"  {v}: field size  ->", g.groupby(pd.cut(g.field, [2, 7, 11, 40])).apply(
                lambda x: f"{len(x)} races ROI {x.pnl.sum() / x.risk.sum():+.3f} CLV {x.clv.mean():+.3f}").to_dict())
    print("\nby day (owner_70 at off_3):")
    g = rows[(rows.variant == "owner_70") & (rows["mark"] == "off_3")]
    print(g.groupby("day").apply(lambda x: f"{len(x)} races {x.pnl.sum():+.2f}").to_string())
