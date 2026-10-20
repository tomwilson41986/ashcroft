"""The two findings of 20 Oct (benter-blend-1020, depth-refill-1020) through the trader's own code (read-only).

research/queries/done/stake_target_replay.py's machinery: every recorded book of the day, the live session replayed a
minute at a time from 06:15 UK, fills against the recorded sizes (an offer the replay takes stays taken), GBP4,000 a
day, GBP5,000 in the account, settled at Betfair's price files; the live config as it stands from 21 Oct (to win
GBP400, GBP15-200 a horse, top-ups to the stake, never under the predicted BFSP). Days 12-19 Oct. Arms:

- live: the served closing model (Jan-Mar morning WAPs, 0.77 market + 0.26 ours);
- the ladder: each back's limit set at the lowest of the three best backs still at expected CLV >= 3% at its own
  price and at or over our price, so it takes the levels below the best back at the same read;
- the spread blend: the closing model refitted on the trader's own books of the days before the replayed day (walk-
  forward, 1 Oct on), the market now and ours with weights that move with the spread (lay / back) and the spread
  itself; everything else the live code;
- the spread blend with the ladder;
- live with top-ups only at expected CLV >= 6% (topups-1020);
- the spread blend with the ladder and top-ups only at >= 6%.

The spread blend reads only what the trader's runner view carries (best back and lay, our price), so it could be
served without new data. Read the CLV x stake (the hedged backs' expected profit before commission); eight days of
results are mostly the noise of the BSP.
"""
import dataclasses
import io
import os
import sqlite3
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize

here = os.path.join(os.getcwd(), "research", "queries", "done")
sys.path.insert(0, os.getcwd())
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h, utc_iso_to_uk_hhmm  # noqa: E402
from model import race_book as rb  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
REPLAY_DAYS = [date(2026, 10, d) for d in range(12, 20)]
BAR, BALANCE = 0.03, 5000.0

# ------------------------------------------------------------------ the replay's machinery, as replay_1019 runs it
path = os.path.join(here, "stake_target_replay.py")
src = open(path, encoding="utf-8").read()
head = src[:src.index("\nrows = []\n")]
head = head.replace("policy={})", "policy=POLICY)")
head = head.replace('OUT = Path("out/stake_target_replay")', 'OUT = Path("out/blend_ladder_replay")')
old = "    cfg.target, cfg.limits.max_stake, cfg.limits.max_daily_turnover = target, max_stake, max_day\n"
assert old in head
head = head.replace(old, old + "    cfg.trade_from = '06:15'\n")
assert "day.day, 6, 30, tzinfo" in head
head = head.replace("day.day, 6, 30, tzinfo", "day.day, 5, 0, tzinfo")
g = {"__name__": "__main__", "__file__": path, "POLICY": {}}
sys.argv = ["blend_ladder_replay_1020.py"]
exec(compile(head, path, "exec"), g)
_UNIT, _orig = g["_UNIT"], g["_orig_plan_race"]
print(f"the live config replayed: target {_UNIT.target}, floor {_UNIT.stake_floor}, top-ups to {_UNIT.topup_to}, "
      f"guard min_move {_UNIT.min_move}, bar {_UNIT.clv_bar}")

# ------------------------------------------------------------------ the spread blend's data: the trader's books
s3, bucket = g["s3"], g["bucket"]
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


FIT_DAYS = [(date(2026, 10, 1) + timedelta(n)).isoformat() for n in range(19)]
text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name")}
pf = []
for day in FIT_DAYS:
    raw = get(f"predictions/{day}.csv")
    if raw is not None:
        x = pd.read_csv(io.BytesIO(raw), dtype=text, low_memory=False)
        x["file_day"] = day
        pf.append(x)
p = pd.concat(pf, ignore_index=True)
for a, b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
    if a not in p.columns and b in p.columns:
        p[a] = p[b]
p["race_date"] = p["race_date"].fillna(p["file_day"]).astype(str).str.slice(0, 10)
p["predicted_bfsp"] = pd.to_numeric(p["predicted_bfsp"], errors="coerce")
p["_k"] = (p.race_date + "|" + p.track.map(normalise_track) + "|" + p.race_time.map(race_time_to_24h).astype(str) + "|"
           + p.horse_name.map(normalise_horse))
p = p.drop_duplicates("_k", keep="last")[["_k", "predicted_bfsp"]]
mk = pd.read_sql_query(f"""SELECT race_date, market_id, selection_id, venue, market_start_utc, runner_name
                           FROM betfair_live_markets WHERE race_date >= '{FIT_DAYS[0]}' AND race_date <= '{FIT_DAYS[-1]}'
                           AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'""", conn)
mk["_k"] = (mk.race_date + "|" + mk.venue.map(normalise_track) + "|" + mk.market_start_utc.map(utc_iso_to_uk_hhmm).astype(str)
            + "|" + mk.runner_name.map(normalise_horse))
mk = mk.merge(p, on="_k", how="left")
mk["selection_id"] = pd.to_numeric(mk.selection_id, errors="coerce")
fin = pd.read_sql_query(f"""SELECT market_id, selection_id, runner_status AS final_status, bsp FROM betfair_live_marks
                            WHERE mark = 'final' AND race_date >= '{FIT_DAYS[0]}' AND race_date <= '{FIT_DAYS[-1]}'""", conn)
fin["selection_id"] = pd.to_numeric(fin.selection_id, errors="coerce")
mk = mk.merge(fin, on=["market_id", "selection_id"], how="left")
BOOK = ["polled_utc", "market_id", "selection_id", "status", "inplay", "runner_status", "back1", "lay1"]
parts = []
for day in FIT_DAYS:
    raw = get(f"betfair_live/{day}/books_trader.csv.gz")
    if raw is None:
        continue
    bk = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=lambda c: c in BOOK, dtype={"market_id": str},
                     low_memory=False)
    for c in ("back1", "lay1", "selection_id"):
        bk[c] = pd.to_numeric(bk[c], errors="coerce")
    ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
    bk = bk[(bk.status == "OPEN") & ~ip & bk.runner_status.fillna("ACTIVE").eq("ACTIVE")]
    reads = bk[["market_id", "polled_utc"]].drop_duplicates().sort_values(["market_id", "polled_utc"])
    reads = reads[reads.groupby("market_id").cumcount() % 10 == 0]
    bk = bk.merge(reads, on=["market_id", "polled_utc"])
    bk["day"] = day
    parts.append(bk)
L = pd.concat(parts, ignore_index=True).merge(mk[["market_id", "selection_id", "predicted_bfsp", "bsp", "final_status"]],
                                              on=["market_id", "selection_id"], how="left")
L["snap"] = L.market_id + "|" + L.polled_utc.astype(str)
ok = (L.back1 > 1) & (L.predicted_bfsp > 1) & (L.bsp > 1) & L.final_status.isin(["WINNER", "LOSER"])
L = L[ok.groupby(L.snap).transform("all") & (L.groupby("snap").snap.transform("size") >= 2)].copy()
L = L.sort_values(["day", "snap"]).reset_index(drop=True)
print(f"the blend's data: {L.snap.nunique():,} whole reads, {len(L):,} runner-reads, {L.day.min()} .. {L.day.max()}")


def spread_of(back, lay):
    back, lay = np.asarray(back, float), np.asarray(lay, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.log(np.where(lay > 1, lay, np.e * back) / back)
    return np.clip(np.nan_to_num(s, nan=1.0), 0, 1)


def design(lq_now, lq_model, sz):
    return np.column_stack([lq_now, lq_model, lq_now * sz, lq_model * sz, sz])


def fit_blend(tr):
    gtr = rb.race_codes(tr.snap)
    now = rb.market_now(tr.back1.to_numpy(float), tr.lay1.to_numpy(float))
    lqn = np.log(rb.normalise_within(1.0 / now, gtr))
    lqm = np.log(rb.normalise_within(1.0 / tr.predicted_bfsp.to_numpy(float), gtr))
    sp = spread_of(tr.back1, tr.lay1)
    mu, sd = float(sp.mean()), float(sp.std()) or 1.0
    X = design(lqn, lqm, (sp - mu) / sd)
    q = rb.normalise_within(1.0 / tr.bsp.to_numpy(float), gtr)
    nr = gtr.max() + 1

    def loss(b):
        lqh = rb.group_log_softmax(X @ b, gtr)
        return (-float(np.sum(q * lqh)) / nr + 0.5e-4 * float(b @ b),
                -((q - np.exp(lqh))[:, None] * X).sum(axis=0) / nr + 1e-4 * b)

    b0 = np.zeros(X.shape[1])
    b0[:2] = 0.5
    beta = minimize(loss, b0, jac=True, method="L-BFGS-B").x
    book = float(np.median(np.bincount(gtr, weights=1.0 / tr.bsp.to_numpy(float))))
    return {"beta": beta, "mu": mu, "sd": sd, "book": book}


BLEND = {"fit": None}
_served_rec = rb.race_expected_clv


def blend_rec(back, lay, model_price, matched, model, n_draws=50000, rng=None, market=None):
    """The expected CLV at the best back from the spread blend (in place of the served closing model)."""
    f = BLEND["fit"]
    if f is None:
        return _served_rec(back, lay, model_price, matched, model, n_draws=n_draws, rng=rng, market=market)
    back = np.asarray(back, float)
    now = rb.market_now(back, lay) if market is None else np.asarray(market, float)
    g0 = np.zeros(len(back), int)
    lqn = np.log(rb.normalise_within(1.0 / now, g0))
    lqm = np.log(rb.normalise_within(1.0 / np.asarray(model_price, float), g0))
    sz = (spread_of(back, lay) - f["mu"]) / f["sd"]
    q = np.exp(rb.group_log_softmax(design(lqn, lqm, sz) @ f["beta"], g0))
    inv = f["book"] * q
    return back * inv - 1.0, 1.0 / inv


rb.race_expected_clv = blend_rec
_cache: dict = {}


def plan(views, cfg, bank):
    key = (BLEND["tag"],) + tuple((v.selection_id, v.back, v.lay, v.last_traded, v.model_price, v.status) for v in views)
    if key not in _cache:
        _cache[key] = _orig(views, _UNIT, bank)
    size = {v.selection_id: v.back_size for v in views}
    out = []
    for d in _cache[key]:
        d2 = dataclasses.replace(d, target=min(d.target * cfg.target / _UNIT.target, cfg.limits.max_stake),
                                 back_size=size.get(d.selection_id, d.back_size))
        out.append(d2)
    return out


g["ts_mod"].plan_race = plan


class LadderSession(g["PolicySession"]):
    """The live session; with policy['ladder'] each back's limit is the lowest of the three best backs still at
    expected CLV >= the bar and at or over our price; with policy['bar'] nothing is backed under that expected CLV."""

    def execute(self, m, book, d, now):
        pol = self.policy
        if pol.get("bar") is not None and not (np.isfinite(d.edge) and d.edge >= pol["bar"]):
            return
        if pol.get("ladder") is not None and d.price and d.price > 1 and d.p_pool > 0:
            q = book.runners.get(d.selection_id)
            ours = d.price / np.exp(d.move)
            lim, size = None, 0.0
            for p_, s_ in (q.back if q else []):
                if p_ * d.p_pool - 1.0 >= pol["ladder"] and p_ >= ours - 1e-9:
                    lim, size = p_, size + s_
                else:
                    break
            if lim is not None and lim < d.price:
                d = dataclasses.replace(d, price=float(lim), back_size=float(size))
        super().execute(m, book, d, now)


class LevelExchange(g["ReplayExchange"]):
    """The replay's exchange, with what a back takes recorded at each level it filled (a back down the ladder takes
    the best back first)."""

    def back(self, market_id, selection_id, price, size, ref, min_fill: float = 2.0):
        if self.balance is not None and size > self.balance - self.open_stakes() + 0.005:
            return g["Fill"](market_id=market_id, selection_id=int(selection_id), side="BACK", order_type="LIMIT",
                             price=float(price), size=round(float(size), 2), status="FAILURE",
                             error="INSUFFICIENT_FUNDS (ERROR_IN_ORDER)")
        book = self._books.get(market_id)
        q = book.runners.get(int(selection_id)) if book else None
        f = g["PaperExchange"].back(self, market_id, selection_id, price, size, ref, min_fill)
        if f.status == "SUCCESS":
            left = f.matched
            for p_, s_ in (q.back if q else []):
                if p_ + 1e-9 < price or left <= 0:
                    break
                take = min(s_, left)
                self.data.taken[(market_id, int(selection_id), round(float(p_), 2))] += take
                left -= take
            self.fills.append((market_id, f.matched, self.data.now))
            self.peak = max(self.peak, self.open_stakes())
        return f


g["PolicySession"], g["ReplayExchange"] = LadderSession, LevelExchange

ARMS = [  # name, blend, policy
    ("live (served blend)", False, {}),
    ("live + the ladder", False, {"ladder": BAR}),
    ("spread blend", True, {}),
    ("spread blend + the ladder", True, {"ladder": BAR}),
    ("live, top-ups at >= 6%", False, {"topup_bar": 0.06}),
    ("spread blend + the ladder, top-ups at >= 6%", True, {"ladder": BAR, "topup_bar": 0.06}),
]
rows = []
for day in REPLAY_DAYS:
    D = g["load"](day)
    if D is None:
        continue
    g["D"] = D
    fit = fit_blend(L[L.day < day.isoformat()])
    print(f"  the spread blend for {day} (fitted on 1 Oct - the day before): weights {np.round(fit['beta'], 3)}, "
          f"book {fit['book']:.4f}")
    for name, blend, pol in ARMS:
        BLEND["fit"], BLEND["tag"] = (fit if blend else None), (f"blend{day}" if blend else "served")
        g["POLICY"] = dict(pol)
        r = g["replay"](day, 400.0, BALANCE, 200.0, 4000.0)
        r["arm"] = name
        rows.append(r)
        print({k: r.get(k) for k in ("day", "arm", "horses", "staked", "clv", "clv_x_stake", "first_clv",
                                     "topup_clv", "result_after_commission", "day_limit_reached", "last_back_uk")},
              flush=True)
if not rows:
    raise SystemExit("no day could be settled")
res = pd.DataFrame(rows)
pooled = res.groupby("arm", sort=False).agg(days=("day", "nunique"), horses=("horses", "sum"), staked=("staked", "sum"),
                                            clv_x_stake=("clv_x_stake", "sum"), first_staked=("first_staked", "sum"),
                                            topup_staked=("topup_staked", "sum"),
                                            result=("result_after_commission", "sum"),
                                            limit_days=("day_limit_reached", "sum"))
pooled["clv"] = pooled.clv_x_stake / pooled.staked
print("\n== the days together, GBP5,000 in the account (CLV x stake before commission; result after 2%)")
print(pooled.round(4).to_string())
print("\n== CLV x stake by day")
print(res.pivot_table(index="arm", columns="day", values="clv_x_stake", sort=False).round(0).to_string())
print("\n== CLV by day")
print(res.pivot_table(index="arm", columns="day", values="clv", sort=False).round(4).to_string())
print("\n== result after commission by day")
print(res.pivot_table(index="arm", columns="day", values="result_after_commission", sort=False).round(0).to_string())
