"""The execution audit, part 2 (21 Oct; read-only): where the replay's extra pounds come from (replay-vs-live-1021:
live 0.40 of the replay). The rule live each day, 10-20 Oct, replayed twice: on both recorded book sources (the
trader's own reads and the recorder's snapshots, which come through Betfair's delayed key) as every replay so far, and on
the trader's own reads only; each scored at the price matched and after the non-runners' reduction factors (the replay
settles at the price matched; Betfair cuts it for a later withdrawal). Against the live day as Betfair settled it."""
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
    raw_ = (bb.matched * (bb.avg_price / bb.bsp - 1)).sum()
    red = (bb.matched * (np.maximum(1.01, bb.avg_price * bb.f) / bb.bsp - 1)).sum()
    return {"clv_x_stake_reduced": round(float(red), 2), "clv_x_stake_unreduced_check": round(float(raw_), 2),
            "share_cut": round(float((bb.f < 0.9999).mean()), 3)}


ARMS = [("both book sources", False, {"sources": ("books_trader", "books")}),
        ("the trader's books only", False, {"sources": ("books_trader",)})''' + src[b:]
old = """        BLEND["fit"], BLEND["tag"] = (fit if blend else None), (f"blend{day}" if blend else "served")
        g["POLICY"] = dict(pol)
        r = g["replay"](day, 400.0, BALANCE, 200.0, 4000.0)
"""
assert old in src
src = src.replace(old, """        g["BOOK_SOURCES"] = pol["sources"]
        D = g["load"](day)
        if D is None:
            continue
        g["D"] = D
        over, cap = as_live(day.isoformat())
        _UNIT.min_move, _UNIT.limits.max_price, _UNIT.stake_floor = -100.0, 1000.0, over["stake_floor"]
        BLEND["fit"], BLEND["tag"] = None, f"as_live_{day}"
        g["POLICY"] = {"cfg": over}
        r = g["replay"](day, 400.0, OPEN.get(day.isoformat(), BALANCE), cap, 4000.0)
        r.update(reduced(day.isoformat(), OPEN.get(day.isoformat(), BALANCE), cap))
""")
src = src.replace('"out/blend_ladder_replay"', '"out/replay_sources"')
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
print("\n== the replay on each book source against the live day (CLV x stake before commission; the live is after reductions)")
for arm, x in res.groupby("arm", sort=False):
    t = x.set_index("day")[["staked", "clv", "clv_x_stake", "clv_x_stake_reduced", "result_after_commission"]].join(lv)
    print(f"\n   {arm}")
    print(t.round(3).to_string())
    s_ = t.sum()
    print(f"   together: replay CLV x stake GBP{s_.clv_x_stake:,.2f}, after reductions GBP{s_.clv_x_stake_reduced:,.2f}, "
          f"result after commission GBP{s_.result_after_commission:,.2f}; live CLV x stake GBP{s_.live_clv_x_stake:,.2f}, "
          f"result GBP{s_.live_after_commission:,.2f}; live / replay {s_.live_clv_x_stake / s_.clv_x_stake:.2f} (after "
          f"reductions {s_.live_clv_x_stake / s_.clv_x_stake_reduced:.2f})")
'''
sys.argv = ["replay_sources_1021.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
