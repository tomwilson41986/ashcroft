"""The market-read replay (reports/discovery_1111.md 4.4, ledger market-read-replay-1112; read-only).

reliability-book-1112 found the closing model's read of the morning book (race_book.market_now: the back price when
the spread is wide) further from the BSP book than the plain middle of back and lay, and the middle with the book's
gap given to the widest quotes nearer still. Does either make the rule more money? The nightly replay (done/replay_1111.py:
the trader's own code on the recorded books, 12 Oct - 10 Nov, every back closed at the BSP, scored after the
non-runners' reductions), three arms on the rule as it stands (the ladder, GBP4,000 a day, 5 top-ups):
- the rule as it stands;
- the market read as the plain middle (no lay: the back);
- the market read as the middle with the book's gap shared by the square of each runner's half-spread (the planner
  holds no lay size, so the size term of reliability-book-1112 is left out).
Only the closing model's market input changes; prices backed, stakes, the guard and the limits are the rule's.
Reading: each new arm's result after the reductions less the rule's, by day; the 90% day-bootstrap interval above zero
and research/overfit.py's deflated Sharpe at 0.95 with 2 trials, on the 30 days.
"""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "blend_ladder_replay_1020.py")
src = open(path, encoding="utf-8").read()
src = src.replace("REPLAY_DAYS = [date(2026, 10, d) for d in range(12, 20)]",
                  "REPLAY_DAYS = [date(2026, 10, d) for d in range(12, 32)] + [date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4), date(2026, 11, 5), date(2026, 11, 6), date(2026, 11, 7), date(2026, 11, 8), date(2026, 11, 9), date(2026, 11, 10), date(2026, 11, 11)]")
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


import model.race_book as _rb
_NOW = _rb.market_now


def _mid(back, lay):
    back, lay = np.asarray(back, float), np.asarray(lay, float)
    has = np.isfinite(lay) & (lay > 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(has, 2.0 / (1.0 / back + 1.0 / np.where(has, lay, back)), back)


def _rel(back, lay):
    back, lay = np.asarray(back, float), np.asarray(lay, float)
    ok = np.isfinite(back) & (back > 1)
    has = ok & np.isfinite(lay) & (lay > 1)
    if not ok.all() or not has.any():
        return _mid(back, lay)
    m = np.where(has, (1.0 / back + 1.0 / np.where(has, lay, back)) / 2.0, 0.5 / back)
    v = np.where(has, ((1.0 / back - 1.0 / np.where(has, lay, back)) / 2.0) ** 2, np.nan)
    if not np.nansum(v) > 0:
        return _mid(back, lay)
    v = np.where(np.isfinite(v), v, np.nanmax(v))
    x = np.maximum(m + (1.0 - m.sum()) * v / v.sum(), 0.001)
    return 1.0 / (x / x.sum())


READS = {"now": _NOW, "mid": _mid, "rel": _rel}
ARMS = [  # name, blend, policy, day limit
    ("the rule as it stands", False, {"cfg_max_topups": 5}, 4000.0),
    ("market read: the plain middle", False, {"cfg_max_topups": 5, "market": "mid"}, 4000.0),
    ("market read: the middle, gap to the widest", False, {"cfg_max_topups": 5, "market": "rel"}, 4000.0),
''' + src[b:]
old = "    for name, blend, pol in ARMS:\n"
assert old in src
src = src.replace(old, "    for name, blend, pol, max_day in ARMS:\n")
old = """        BLEND["fit"], BLEND["tag"] = (fit if blend else None), (f"blend{day}" if blend else "served")\n"""
assert old in src
src = src.replace(old, old + """        old_rule = bool(pol.get("rule_of_20_oct"))
        _UNIT.min_move, _UNIT.limits.max_price = (-100.0, 1000.0) if old_rule else (0.0, 200.0)
        BLEND["tag"] = BLEND["tag"] + ("_20oct" if old_rule else "") + "_" + pol.get("market", "now")
        _rb.market_now = READS[pol.get("market", "now")]
""")
old = '        r = g["replay"](day, 400.0, BALANCE, 200.0, 4000.0)\n'
assert old in src
src = src.replace(old, '        r = g["replay"](day, 400.0, BALANCE, 200.0, max_day)\n'
                  '        r.update(reduced(day.isoformat(), BALANCE, max_day))\n')
src = src.replace('"out/blend_ladder_replay"', '"out/market_read_replay_1112"')
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
src += '''
import sys as _sys
_sys.path.insert(0, ".")
from research.overfit import dsr_check
piv = res.pivot_table(index="day", columns="arm", values="result_reduced", sort=False)
stk = res.pivot_table(index="day", columns="arm", values="staked", sort=False)
base = ARMS[0][0]
rng = np.random.default_rng(1112)
print("\\n== THE READING: each arm less the rule, result after the reductions, by day")
for name in [a[0] for a in ARMS[1:]]:
    d = (piv[name] - piv[base]).dropna()
    bs = [d.sample(len(d), replace=True, random_state=int(rng.integers(1 << 31))).sum() for _ in range(4000)]
    lo, hi = np.percentile(bs, [5, 95])
    chk = dsr_check(d.to_numpy(), n_trials=2)
    print(f"  {name}: {len(d)} days, sum {d.sum():+.2f} (90% {lo:+.2f} to {hi:+.2f}), mean a day {d.mean():+.2f}, "
          f"days ahead {int((d > 0).sum())}; staked {stk[name].sum():,.0f} against {stk[base].sum():,.0f}; "
          f"deflated Sharpe {chk.get('dsr')} ({'pass' if chk.get('passed') else chk.get('reason')})")
    print("   ", "PASS" if (lo > 0 and chk.get("passed")) else "does not pass")
'''
sys.argv = ["replay_1102.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
