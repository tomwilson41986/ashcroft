"""The replay of the evening of 24 Oct (read-only): done/replay_1021.py with 12-24 Oct as the price files allow (each
racing day's file is archived the evening after). config_live now carries the ladder, the guard, nothing over 200 and
at most 5 top-ups a horse, so the replay switches each through the config (cfg.ladder from cfg_ladder, cfg.max_topups
from cfg_max_topups, None = no cap); never through LadderSession's own policies. Every arm is scored at the price
matched and after the non-runners' reduction factors (execution-audit-1021: the replay settles at the price matched;
Betfair cuts it for a later withdrawal). Arms:

- the best back only, the rule of 20 Oct (no ladder, no guard, max price 1000, no cap);
- the ladder at GBP4,000 a day with no cap (the rule of 21 Oct);
- the ladder at GBP4,000 a day with at most 5 top-ups (the rule from 22 Oct);
- the ladder at GBP6,000 a day with the cap.
"""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "blend_ladder_replay_1020.py")
src = open(path, encoding="utf-8").read()
src = src.replace("REPLAY_DAYS = [date(2026, 10, d) for d in range(12, 20)]",
                  "REPLAY_DAYS = [date(2026, 10, d) for d in range(12, 25)]")
old = """head = head.replace(old, old + "    cfg.trade_from = '06:15'\\n")"""
assert old in src, "trade_from"
src = src.replace(old, """head = head.replace(old, old + "    cfg.trade_from = '06:15'\\n"
                    "    cfg.ladder = bool(POLICY.get('cfg_ladder', True))\\n"
                    "    cfg.max_topups = POLICY.get('cfg_max_topups')\\n"
                    "    if POLICY.get('rule_of_20_oct'):\\n"
                    "        cfg.min_move, cfg.limits.max_price = -100.0, 1000.0\\n")""")
a, b = src.index("ARMS = ["), src.index("]\nrows = []")
src = src[:a] + '''def reduced(day, balance, max_day):
    """The replay's backs scored after the non-runners' reduction factors (each fill's price cut by the factors of the
    runners removed after it was matched), from the replay's own saved ledger; the result less the winners' cuts."""
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
    rem = bk.groupby(["market_id", "selection_id"]).agg(af=("af", "last"), rt=("rt", "min")).reset_index()
    led = pd.read_csv(g["OUT"] / f"ledger_{day}_400_200_{int(max_day)}_{balance}.csv", low_memory=False,
                      dtype={"market_id": str})
    for c in ("matched", "avg_price", "bsp", "selection_id"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
    sett = led[(led.event == "settle") & led.result.isin(["WINNER", "LOSER"]) & (led.bsp > 1)].drop_duplicates(
        ["market_id", "selection_id"], keep="last")[["market_id", "selection_id", "bsp", "result"]]
    bb = led[(led.event == "back") & (led.matched > 0)].drop(columns=["result"]).merge(
        sett, on=["market_id", "selection_id"], suffixes=("_x", ""))
    fac = []
    for r_ in bb.itertuples():
        x = rem[(rem.market_id == r_.market_id) & (rem.selection_id != r_.selection_id) & (rem.rt > r_.t)]
        fac.append(float(np.prod(1 - x.af.to_numpy() / 100.0)) if len(x) else 1.0)
    bb["f"] = fac
    bb["p_red"] = np.maximum(1.01, bb.avg_price * bb.f)
    red = float((bb.matched * (bb.p_red / bb.bsp - 1)).sum())
    won = bb[bb.result.eq("WINNER")]
    return {"clv_x_stake_reduced": round(red, 2),
            "winners_cut": round(float((won.matched * (won.avg_price - won.p_red)).sum()), 2)}


ARMS = [  # name, blend, policy, day limit
    ("best back only (the rule of 20 Oct)", False, {"cfg_ladder": False, "rule_of_20_oct": True}, 4000.0),
    ("the ladder, GBP4,000, no cap (the rule of 21 Oct)", False, {}, 4000.0),
    ("the ladder, GBP4,000, 5 top-ups (the rule from 22 Oct)", False, {"cfg_max_topups": 5}, 4000.0),
    ("the ladder, GBP6,000, 5 top-ups", False, {"cfg_max_topups": 5}, 6000.0),
''' + src[b:]
old = "    for name, blend, pol in ARMS:\n"
assert old in src
src = src.replace(old, "    for name, blend, pol, max_day in ARMS:\n")
old = """        BLEND["fit"], BLEND["tag"] = (fit if blend else None), (f"blend{day}" if blend else "served")\n"""
assert old in src
src = src.replace(old, old + """        old_rule = bool(pol.get("rule_of_20_oct"))
        _UNIT.min_move, _UNIT.limits.max_price = (-100.0, 1000.0) if old_rule else (0.0, 200.0)
        BLEND["tag"] = BLEND["tag"] + ("_20oct" if old_rule else "")
""")
old = '        r = g["replay"](day, 400.0, BALANCE, 200.0, 4000.0)\n'
assert old in src
src = src.replace(old, '        r = g["replay"](day, 400.0, BALANCE, 200.0, max_day)\n'
                  '        r.update(reduced(day.isoformat(), BALANCE, max_day))\n')
src = src.replace('"out/blend_ladder_replay"', '"out/replay_1024"')
src += '''
print("\\n== the spread of the days (result after commission)")
print(res.groupby("arm", sort=False).agg(mean_day=("result_after_commission", "mean"),
                                         sd_day=("result_after_commission", "std"),
                                         worst_day=("result_after_commission", "min")).round(2).to_string())
res["result_reduced"] = res.result_after_commission - 0.98 * res.winners_cut
print("\\n== after the non-runners' reductions (the honest replay figure; live has run at 0.80 of it, replay-sources-1021)")
tot = res.groupby("arm", sort=False).agg(days=("day", "nunique"), staked=("staked", "sum"),
                                         clv_x_stake_at_fill=("clv_x_stake", "sum"),
                                         clv_x_stake_reduced=("clv_x_stake_reduced", "sum"),
                                         result_at_fill=("result_after_commission", "sum"),
                                         result_reduced=("result_reduced", "sum"))
tot["clv_reduced"] = tot.clv_x_stake_reduced / tot.staked
print(tot.round(4).to_string())
print(res.pivot_table(index="arm", columns="day", values="result_reduced", sort=False).round(0).to_string())
print("\\n== 19-22 Oct (the live days of the new staking): staked, CLV, CLV x stake (at the fill and reduced), result")
print(res[res.day >= "2026-10-19"][["day", "arm", "horses", "staked", "clv", "clv_x_stake", "clv_x_stake_reduced",
                                    "result_after_commission", "result_reduced"]].to_string(index=False))
'''
sys.argv = ["replay_1024.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
