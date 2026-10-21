"""The owner's ask (21 Oct): "run the full replay with the non-runner bar" (reductions-1021). The rule live from 22 Oct
(GBP400 to win, GBP15-200 a horse, top-ups to the stake, the guard, the ladder, max price 200, at most 5 top-ups,
GBP4,000 a day, from 06:15), replayed 10-19 Oct on the trader's own books with each day's opening balance, against
the same rule with the non-runner bar: a back (first or top-up, each ladder level) only where its expected CLV after the
expected non-runner reduction, (1 + CLV) x E[factor] - 1, is at least 3% (and 5%). E[factor] is walk-forward: the
money-weighted reduction factor of the live backs of every day before, by the price of the back, from Betfair's
REMOVED runners. Every arm scored after the reductions (CLV x stake and the result), the honest figure; the live days
as Betfair settled them beside it (the rule live then was not this one)."""
import io
import os
import sys

import boto3
import numpy as np
import pandas as pd

path = os.path.join(os.getcwd(), "research", "queries", "done", "blend_ladder_replay_1020.py")
src = open(path, encoding="utf-8").read()
src = src.replace("REPLAY_DAYS = [date(2026, 10, d) for d in range(12, 20)]",
                  "REPLAY_DAYS = [date(2026, 10, d) for d in range(10, 21)]")
old = """head = head.replace(old, old + "    cfg.trade_from = '06:15'\\n")"""
assert old in src
old2 = 'head = head.replace("policy={})", "policy=POLICY)")\n'
assert old2 in src
src = src.replace(old2, old2 + 'head = head.replace(\'for n in ("books_trader", "books")], ignore_index=True)\', \'for n in BOOK_SOURCES], ignore_index=True)\', 1)\nassert "BOOK_SOURCES" in head\n', 1)
old3 = 'g = {"__name__": "__main__", "__file__": path, "POLICY": {}}'
assert old3 in src
src = src.replace(old3, 'g = {"__name__": "__main__", "__file__": path, "POLICY": {}, "BOOK_SOURCES": ("books_trader", "books")}', 1)
src = src.replace(old, """head = head.replace(old, old + "    cfg.trade_from = '06:15'\\n"
                    "    for k, v in POLICY.get('cfg', {}).items():\\n"
                    "        setattr(cfg.limits if k == 'max_price' else cfg, k, v)\\n")""")
src = src.replace("    fit = fit_blend(L[L.day < day.isoformat()])\n", "    fit = None\n")
assert """    D = g["load"](day)\n    if D is None:\n        continue\n    g["D"] = D\n""" in src
src = src.replace("""    D = g["load"](day)\n    if D is None:\n        continue\n    g["D"] = D\n""", "")
src = src.replace("""    print(f"  the spread blend for {day} (fitted on 1 Oct - the day before): weights {np.round(fit['beta'], 3)}, "
          f"book {fit['book']:.4f}")
""", "")
a, b = src.index("ARMS = ["), src.index("]\nrows = []")
src = src[:a] + '''OPEN = {"2026-10-10": 4888.75, "2026-10-11": 5472.86, "2026-10-12": 5000.0, "2026-10-13": 5159.15,
        "2026-10-14": 5469.67, "2026-10-15": 5505.0, "2026-10-16": 5189.34, "2026-10-17": 5155.53,
        "2026-10-18": 5227.34, "2026-10-19": 5126.3, "2026-10-20": 5465.69}


def as_live(day):
    """The rule live on the day, as config overrides and the day's cap a bet."""
    base = {"ladder": False, "min_move": -100.0, "max_price": 1000.0, "max_topups": None,
            "trade_from": "08:00" if day <= "2026-10-12" else "06:15"}     # the 06:15 start from 13 Oct
    if day >= "2026-10-19":
        return {**base, "stake_floor": 15.0, "topup_to": "stake"}, 200.0
    return {**base, "stake_floor": 0.0, "topup_to": "winnings"}, 300.0


def reduced(day, balance, cap):
    """The replay's fills scored after the non-runners' reduction factors (each fill's price cut by the factors of the
    runners removed after it was matched), from the replay's own saved ledger."""
    import io as _io
    raw = []
    for n in ("books_trader", "books"):
        b_ = get(f"betfair_live/{day}/{n}.csv.gz")
        if b_ is not None:
            raw.append(pd.read_csv(_io.BytesIO(b_), compression="gzip", low_memory=False, dtype={"market_id": str},
                                   usecols=lambda c: c in ("polled_utc", "market_id", "selection_id", "runner_status",
                                                           "adjustment_factor", "removal_utc")))
    bk = pd.concat(raw, ignore_index=True)
    bk = bk[bk.runner_status.eq("REMOVED")].copy()
    bk["selection_id"] = pd.to_numeric(bk.selection_id, errors="coerce")
    bk["af"] = pd.to_numeric(bk.adjustment_factor, errors="coerce").fillna(0)
    bk["rt"] = pd.to_datetime(bk.removal_utc, utc=True, errors="coerce").fillna(pd.to_datetime(bk.polled_utc, utc=True))
    rem = bk.sort_values("polled_utc").groupby(["market_id", "selection_id"]).agg(af=("af", "last"), rt=("rt", "min")).reset_index()
    led = pd.read_csv(g["OUT"] / f"ledger_{day}_400_{int(cap)}_4000_{balance}.csv", low_memory=False, dtype={"market_id": str})
    for c in ("matched", "avg_price", "bsp", "selection_id"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
    st = led[(led.event == "settle") & led.result.isin(["WINNER", "LOSER"]) & (led.bsp > 1)].drop_duplicates(
        ["market_id", "selection_id"], keep="last")[["market_id", "selection_id", "bsp"]]
    bb = led[(led.event == "back") & (led.matched > 0)].merge(st, on=["market_id", "selection_id"], suffixes=("_x", ""))
    fac = []
    for r_ in bb.itertuples():
        x = rem[(rem.market_id == r_.market_id) & (rem.selection_id != r_.selection_id) & (rem.rt > r_.t)]
        fac.append(float(np.prod(1 - x.af.to_numpy() / 100.0)) if len(x) else 1.0)
    bb["f"] = fac
    st2 = led[(led.event == "settle") & led.result.eq("WINNER")].drop_duplicates(["market_id", "selection_id"], keep="last")
    won = bb.merge(st2[["market_id", "selection_id"]], on=["market_id", "selection_id"])
    lost_on_winners = float((won.matched * (won.avg_price - np.maximum(1.01, won.avg_price * won.f))).sum())
    raw_ = (bb.matched * (bb.avg_price / bb.bsp - 1)).sum()
    red = (bb.matched * (np.maximum(1.01, bb.avg_price * bb.f) / bb.bsp - 1)).sum()
    return {"clv_x_stake_reduced": round(float(red), 2), "reduction_on_winners": round(lost_on_winners, 2), "clv_x_stake_unreduced_check": round(float(raw_), 2),
            "share_cut": round(float((bb.f < 0.9999).mean()), 3)}


BANDS = [1.0, 4.0, 8.0, 16.0, 40.0, 1001.0]


def band(p):
    return int(np.searchsorted(BANDS, float(p), side="right")) - 1


class NRSession(LadderSession):
    \"\"\"The live session with the non-runner bar: each back's expected CLV (and the ladder's, through p_pool) after the
    day's expected reduction factor for its price band, held to policy['nr_bar'].\"\"\"

    def execute(self, m, book, d, now):
        bar = self.policy.get("nr_bar")
        if bar is not None and d.price and d.price > 1 and np.isfinite(d.edge):
            ef = EF.get(band(d.price), 1.0)
            edge = (1.0 + d.edge) * ef - 1.0
            if edge < bar:
                return
            d = dataclasses.replace(d, p_pool=d.p_pool * ef, edge=edge)
            self.cfg.clv_bar = bar
        super().execute(m, book, d, now)


g["PolicySession"] = NRSession
EF = {}


def expected_factors(day):
    \"\"\"Money-weighted reduction factor of the live backs of every day from 30 Sep to the day before, by price band:
    each back's price cut by the factors of the runners removed after it was matched (Betfair's REMOVED rows).\"\"\"
    acc = {}
    d0 = date(2026, 9, 30)
    while d0 < day:
        for k, (a0, a1) in day_factors(d0.isoformat()).items():
            a = acc.setdefault(k, [0.0, 0.0])
            a[0] += a0
            a[1] += a1
        d0 += timedelta(days=1)
    return {k: v[0] / v[1] for k, v in acc.items() if v[1] > 0}


_DAYF = {}


def day_factors(dd):
    \"\"\"One day's live backs: by price band, the money times its reduction factor and the money (cached).\"\"\"
    import io as _io
    if dd in _DAYF:
        return _DAYF[dd]
    acc = _DAYF[dd] = {}
    raw = get(f"trading/live/{dd}/ledger.csv")
    if raw is None:
        return acc
    rem = []
    for n in ("books_trader", "books"):
        b_ = get(f"betfair_live/{dd}/{n}.csv.gz")
        if b_ is not None:
            rem.append(pd.read_csv(_io.BytesIO(b_), compression="gzip", low_memory=False, dtype={"market_id": str},
                                   usecols=lambda c: c in ("polled_utc", "market_id", "selection_id", "runner_status",
                                                           "adjustment_factor", "removal_utc")))
    if not rem:
        return acc
    bk = pd.concat(rem, ignore_index=True)
    if not {"runner_status", "adjustment_factor"} <= set(bk.columns):
        return acc
    if "removal_utc" not in bk:
        bk["removal_utc"] = np.nan
    bk = bk[bk.runner_status.eq("REMOVED")].copy()
    bk["selection_id"] = pd.to_numeric(bk.selection_id, errors="coerce")
    bk["af"] = pd.to_numeric(bk.adjustment_factor, errors="coerce").fillna(0)
    bk["rt"] = pd.to_datetime(bk.removal_utc, utc=True, errors="coerce").fillna(pd.to_datetime(bk.polled_utc, utc=True))
    bk = bk.groupby(["market_id", "selection_id"]).agg(af=("af", "last"), rt=("rt", "min")).reset_index()
    led = pd.read_csv(_io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ("matched", "avg_price", "selection_id"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
    bb = led[(led.event == "back") & (led.matched > 0) & (led.avg_price > 1)].dropna(subset=["t"])
    for r_ in bb.itertuples():
        x = bk[(bk.market_id == r_.market_id) & (bk.selection_id != r_.selection_id) & (bk.rt > r_.t)]
        f_ = float(np.prod(1 - x.af.to_numpy() / 100.0)) if len(x) else 1.0
        a = acc.setdefault(band(r_.avg_price), [0.0, 0.0])
        a[0] += r_.matched * f_
        a[1] += r_.matched
    return acc


ARMS = [("the rule from 22 Oct", False, {"sources": ("books_trader",), "nr_bar": None}),
        ("+ the non-runner bar at 3%", False, {"sources": ("books_trader",), "nr_bar": 0.03}),
        ("+ the non-runner bar at 5%", False, {"sources": ("books_trader",), "nr_bar": 0.05})''' + src[b:]
old = """        BLEND["fit"], BLEND["tag"] = (fit if blend else None), (f"blend{day}" if blend else "served")
        g["POLICY"] = dict(pol)
        r = g["replay"](day, 400.0, BALANCE, 200.0, 4000.0)
"""
assert old in src
src = src.replace(old, """        if name == ARMS[0][0]:
            EF.clear()
        g["BOOK_SOURCES"] = pol["sources"]
        D = g["load"](day)
        if D is None:
            continue
        g["D"] = D
        cap = 200.0
        if not EF:
            EF.update(expected_factors(day))
            print(f"  the expected reduction factor by price band for {day} (live backs 30 Sep - the day before): "
                  f"{ {BANDS[k]: round(v, 4) for k, v in sorted(EF.items())} }", flush=True)
        BLEND["fit"], BLEND["tag"] = None, f"live22_{day}"
        g["POLICY"] = {"cfg": {}, "nr_bar": pol["nr_bar"]}
        r = g["replay"](day, 400.0, OPEN.get(day.isoformat(), BALANCE), cap, 4000.0)
        r.update(reduced(day.isoformat(), OPEN.get(day.isoformat(), BALANCE), cap))
""")
src = src.replace('"out/blend_ladder_replay"', '"out/replay_nr_bar"')
src += r'''
# ---------------------------------------------------------------- the live days, as Betfair settled them
s3c = __import__("boto3").client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
live = []
for day in sorted(res.day.unique()):
    raw = s3c.get_object(Bucket=os.environ.get("PRED_BUCKET") or "ashcroft", Key=f"trading/live/{day}/ledger.csv")["Body"].read()
    d = pd.read_csv(__import__("io").BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ("matched", "avg_price", "bsp", "pnl", "commission", "clv"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    st = d[d.event == "settle"].drop_duplicates(["market_id", "selection_id"], keep="last")
    ok = st.result.isin(["WINNER", "LOSER"]) & (st.bsp > 1)
    s = st[ok]
    live.append({"day": day, "live_staked": st.matched.sum(), "live_clv": float((s.clv * s.matched).sum() / s.matched.sum()),
                 "live_clv_x_stake": float((s.clv * s.matched).sum()),
                 "live_after_commission": float(st.pnl.sum() - d[d.event == "commission"].commission.sum())})
lv = pd.DataFrame(live).set_index("day")
res["result_reduced"] = res.result_after_commission - 0.98 * res.reduction_on_winners
print("\n== each arm by day, after the reductions (CLV x stake before commission; the result after 2%, less the winners' cuts)")
print(res.pivot_table(index="arm", columns="day", values="clv_x_stake_reduced", sort=False).round(0).to_string())
print(res.pivot_table(index="arm", columns="day", values="result_reduced", sort=False).round(0).to_string())
tot = res.groupby("arm", sort=False).agg(staked=("staked", "sum"), horses=("horses", "sum"),
                                         clv_x_stake_at_fill=("clv_x_stake", "sum"),
                                         clv_x_stake_reduced=("clv_x_stake_reduced", "sum"),
                                         result_at_fill=("result_after_commission", "sum"),
                                         result_reduced=("result_reduced", "sum"),
                                         limit_days=("day_limit_reached", "sum"))
tot["clv_reduced"] = tot.clv_x_stake_reduced / tot.staked
print("\n== the arms together, 10-19 Oct, after the reductions")
print(tot.round(4).to_string())
L_ = lv.loc[lv.index.isin(res.day.unique())].sum()
print(f"\n   the live days as Betfair settled them (the rules live then): staked GBP{L_.live_staked:,.2f}, CLV x stake "
      f"GBP{L_.live_clv_x_stake:,.2f}, result GBP{L_.live_after_commission:,.2f}")
base = tot.iloc[0]
for arm, r_ in tot.iloc[1:].iterrows():
    print(f"   {arm} against the rule from 22 Oct: CLV x stake after reductions GBP{r_.clv_x_stake_reduced - base.clv_x_stake_reduced:+,.2f}, "
          f"result after reductions GBP{r_.result_reduced - base.result_reduced:+,.2f}, staked GBP{r_.staked - base.staked:+,.2f}")
'''
sys.argv = ["replay_nr_bar_1021.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
