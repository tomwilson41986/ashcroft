"""The replay of the evening of 19 Oct (read-only; done/replay_1018.py with 19 Oct in the dates), the strategy replay and the staking replay in one, 1-17 Oct (18 Oct
once its price file is in). From 18 Oct trading/config_live.json carries the owner's staking (a GBP15 floor, top-ups
to the stake, GBP200 a horse), so the old rule's arm sets stake_floor 0 and topup_to winnings on the replay's config
and on the planning unit; one arm runs the live code's own path (stake_floor and topup_to in the config) to check it
against the scheme as first replayed; and floors of GBP25, 30 and 40. Before: done/stake_scheme_replay_1017b.py: done/stake_scheme_replay_1017.py found to win
GBP400 with a GBP15 floor, the GBP200 cap and top-ups to the stake at the price now the best of the schemes (GBP5,776
over 1-15 Oct against the rule's GBP4,950). This takes the scheme apart - the floor with the old top-ups (to the
winnings), the new top-ups with no floor - and tries a GBP20 and GBP25 floor and a GBP150 cap, with 16 Oct in.
Before: how to stake (the owner's review of 17 Oct: "is fixed variable stakes correct? All options are open").

The realized positions of 30 Sep - 17 Oct (strategy_review_1017.py) said the to-win stakes put most of the money on
short prices, whose CLV is the weakest, and that level or edge-weighted stakes earn more a GBP; but they could not say
what the day's budget, the sizes offered and the top-ups make of another scheme. This replays each scheme a minute at a
time against the recorded books (research/queries/done/strategy_replay_1009.py's machinery: fills against the
recorded sizes, GBP4,000 a day, GBP5,000 in the account, the 06:15 opening, the no-volume closing model, Betfair's
settled prices): to win GBP400 capped at GBP300 (the rule) and at GBP200; level GBP25 and GBP40 a horse; in proportion
to the expected CLV at entry (GBP1,000 and GBP600 a unit of edge: a 5% edge GBP50 / GBP30), capped at GBP300; to win
GBP400 capped at GBP200 with a GBP15 floor. Each scheme sets the whole stake a horse may reach; its top-ups fill to it.
"""
import os
import sys

here = os.path.join(os.getcwd(), "research", "queries", "done")
os.makedirs("out", exist_ok=True)

# the replay's session, edited: the 06:15 opening, the clock from 05:00 UTC, and top-ups filling to the scheme's stake
st_src = open(os.path.join(here, "stake_target_replay.py"), encoding="utf-8").read()
old = "    cfg.target, cfg.limits.max_stake, cfg.limits.max_daily_turnover = target, max_stake, max_day\n"
assert old in st_src
st_src = st_src.replace(old, old + "    cfg.trade_from = FROM\n"
                        "    cfg.stake_floor = float(POLICY.get('stake_floor', 0.0))\n"
                        "    cfg.topup_to = POLICY.get('topup_to', 'winnings')\n")
old = "_UNIT.limits.max_stake = 1e12\n"
assert old in st_src
st_src = st_src.replace(old, old + "_UNIT.stake_floor, _UNIT.topup_to = 0.0, 'winnings'   # the plan uncapped and unfloored\n")
assert "day.day, 6, 30, tzinfo" in st_src
st_src = st_src.replace("day.day, 6, 30, tzinfo", "day.day, 5, 0, tzinfo")
old = "        super().execute(m, book, d, now)\n\n    def step"
assert old in st_src
st_src = st_src.replace(old, """        sch = self.policy.get("scheme")
        if sch and sch not in ("to_win", "floor_w") and pos is not None and pos.matched > 0 and d.price > 1.0:
            # the session tops a closing_clv horse up to the to-win target: the target that makes its top-up the
            # scheme's stake less what is matched
            keep = self.cfg.target
            self.cfg.target = (d.target - pos.matched) * (d.price - 1.0) + pos.matched * (pos.avg_price - 1.0)
            try:
                super().execute(m, book, d, now)
            finally:
                self.cfg.target = keep
            return
""" + old)
edited = os.path.join("out", "_stake_target_replay_scheme.py")
open(edited, "w", encoding="utf-8").write(st_src)

path = os.path.join(here, "strategy_replay_1009.py")
src = open(path, encoding="utf-8").read()
old = 'path = os.path.join(os.getcwd(), "research", "queries", "done", "stake_target_replay.py")'
assert old in src
src = src.replace(old, f"path = {os.path.abspath(edited)!r}")
old = 'date(2026, 10, 8): 5410.91}")'
assert old in src
src = src.replace(old, 'date(2026, 10, 8): 5410.91, date(2026, 10, 9): 4951.66, date(2026, 10, 10): 4888.75, '
                       'date(2026, 10, 11): 5472.86, date(2026, 10, 12): 5000.0, date(2026, 10, 13): 5159.15, '
                       'date(2026, 10, 14): 5469.67, date(2026, 10, 15): 5505.0, date(2026, 10, 16): 5189.34, '
                       'date(2026, 10, 17): 5155.53, date(2026, 10, 18): 5227.34, date(2026, 10, 19): 5126.3}")')
old = '"POLICY": {}, "UNTIL": "21:30"}'
assert old in src
src = src.replace(old, '"POLICY": {}, "UNTIL": "21:30", "FROM": "06:15"}')
old = "        d2.target = min(d.target * cfg.target / _UNIT.target, cfg.limits.max_stake)\n"
assert old in src
src = src.replace(old, """        pol = g["POLICY"]
        sch = pol.get("scheme", "to_win")
        if sch == "level":
            d2.target = min(pol["unit"], cfg.limits.max_stake)
        elif sch == "edge":
            d2.target = min(pol["k"] * max(float(d.edge), 0.0), cfg.limits.max_stake)
        elif sch in ("floor", "floor_w"):
            d2.target = min(max(d.target * cfg.target / _UNIT.target, pol["floor"]), cfg.limits.max_stake)
        else:   # the rule's own path, with the config's floor (0 unless the arm sets it, as live does from 19 Oct)
            d2.target = min(max(d.target * cfg.target / _UNIT.target, getattr(cfg, "stake_floor", 0.0)),
                            cfg.limits.max_stake)
""")
old = ("    for name, model, one_back, target, max_day, until in ARMS:\n"
       "        MODE[\"model\"] = model\n"
       "        g[\"POLICY\"] = {\"no_topups\": True} if one_back else {}\n")
assert old in src
src = src.replace(old, "    for name, model, one_back, target, max_day, until, cap, pol in ARMS:\n"
                       "        g[\"FROM\"] = \"06:15\"\n"
                       "        MODE[\"model\"] = model\n"
                       "        g[\"POLICY\"] = {\"no_topups\": True} if one_back else dict(pol or {})\n")
old = '        r = g["replay"](day, target, BALANCE, 300.0, max_day)\n'
assert old in src
src = src.replace(old, '        r = g["replay"](day, target, BALANCE, cap, max_day)\n')
a, b = src.index("ARMS = ["), src.index("]\nBALANCE")
src = src[:a] + '''ARMS = [  # name, model, no top-ups, target to win, day limit, until, cap a horse, policy (the scheme)
    ("the old rule: to win 400, cap 300, top-ups to the winnings", "novol", False, 400.0, 4000.0, "21:30", 300.0, None),
    ("the rule from 19 Oct by the live code (stake_floor 15, topup_to stake, cap 200)", "novol", False, 400.0, 4000.0,
     "21:30", 200.0, {"stake_floor": 15.0, "topup_to": "stake"}),
    ("the same as first replayed (floor 15, top-ups to the stake, cap 200)", "novol", False, 400.0, 4000.0, "21:30",
     200.0, {"scheme": "floor", "floor": 15.0}),
    ("cap 200, floor 25, top-ups to the stake", "novol", False, 400.0, 4000.0, "21:30", 200.0,
     {"scheme": "floor", "floor": 25.0}),
    ("cap 200, floor 30, top-ups to the stake", "novol", False, 400.0, 4000.0, "21:30", 200.0,
     {"scheme": "floor", "floor": 30.0}),
    ("cap 200, floor 40, top-ups to the stake", "novol", False, 400.0, 4000.0, "21:30", 200.0,
     {"scheme": "floor", "floor": 40.0}),
''' + src[b:]
src += '''

res = pd.DataFrame(rows)
res["result_after_commission"] = pd.to_numeric(res["result_after_commission"], errors="coerce")
by = res.groupby("arm").result_after_commission
out = pd.DataFrame({"days": by.size(), "total": by.sum(), "mean_day": by.mean(), "sd_day": by.std(),
                    "worst_day": by.min(), "losing_days": by.apply(lambda x: int((x < 0).sum()))})
out["mean_over_sd"] = out.mean_day / out.sd_day
out["growth_log"] = res.assign(lg=np.log1p(res.result_after_commission / 5000.0)).groupby("arm").lg.sum()
st = res.groupby("arm").agg(staked=("staked", "sum"), horses=("horses", "sum"), clv_x_stake=("clv_x_stake", "sum"))
st["clv"] = st.clv_x_stake / st.staked
print("\\n== each scheme over the days (result after commission, GBP5,000 in the account)")
print(out.join(st).sort_values("total", ascending=False).round(3).to_string())
'''
sys.argv = ["replay_1019.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
