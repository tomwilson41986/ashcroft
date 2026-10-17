"""How to stake (the owner's review of 17 Oct: "is fixed variable stakes correct? All options are open"; read-only).

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
st_src = st_src.replace(old, old + "    cfg.trade_from = FROM\n")
assert "day.day, 6, 30, tzinfo" in st_src
st_src = st_src.replace("day.day, 6, 30, tzinfo", "day.day, 5, 0, tzinfo")
old = "        super().execute(m, book, d, now)\n\n    def step"
assert old in st_src
st_src = st_src.replace(old, """        sch = self.policy.get("scheme")
        if sch and sch != "to_win" and pos is not None and pos.matched > 0 and d.price > 1.0:
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
                       'date(2026, 10, 17): 5155.53}")')
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
        elif sch == "floor":
            d2.target = min(max(d.target * cfg.target / _UNIT.target, pol["floor"]), cfg.limits.max_stake)
        else:
            d2.target = min(d.target * cfg.target / _UNIT.target, cfg.limits.max_stake)
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
    ("to win 400, cap 300 (the rule)", "novol", False, 400.0, 4000.0, "21:30", 300.0, None),
    ("to win 400, cap 200", "novol", False, 400.0, 4000.0, "21:30", 200.0, None),
    ("level 25 a horse", "novol", False, 400.0, 4000.0, "21:30", 300.0, {"scheme": "level", "unit": 25.0}),
    ("level 40 a horse", "novol", False, 400.0, 4000.0, "21:30", 300.0, {"scheme": "level", "unit": 40.0}),
    ("edge x 1000 (5% = GBP50), cap 300", "novol", False, 400.0, 4000.0, "21:30", 300.0, {"scheme": "edge", "k": 1000.0}),
    ("edge x 600 (5% = GBP30), cap 300", "novol", False, 400.0, 4000.0, "21:30", 300.0, {"scheme": "edge", "k": 600.0}),
    ("to win 400, cap 200, floor 15", "novol", False, 400.0, 4000.0, "21:30", 200.0, {"scheme": "floor", "floor": 15.0}),
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
sys.argv = ["stake_scheme_replay_1017.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
