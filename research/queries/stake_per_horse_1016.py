"""The owner's question (15 Oct): what is the optimal stake per horse? Read-only. Re-run the evening of 16 Oct with
15 Oct (the losing day) and 16 Oct in, and the top-up policies beside the caps: no top-up after a 5% drift, and a top-up
only where the expected CLV is 5% or more (the first back keeps the 3% bar); done/stake_per_horse_1015.py before.

The live rule stakes each horse to win GBP400 (before commission) at the price it is backed at, first back and top-ups
together, and never more than GBP300 a horse (limits.max_stake: the trader tops a horse up to max_stake less what is
matched), under GBP4,000 a day. The replay of research/queries/done/strategy_replay_1014.py (every recorded book a minute
at a time, fills against the recorded sizes, Betfair's settled prices; 1-14 Oct, 15 Oct's price file comes tomorrow)
with the cap a horse and the target to win varied, and one more policy: no top-up once the horse's price has drifted
more than 5% above its average matched price (the audit of 15 Oct: top-ups into a drift lost 7.9% over 10-15 Oct).
GBP5,000 in the account; the 06:15 opening; the no-volume closing model, as live."""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "strategy_replay_1009.py")
src = open(path, encoding="utf-8").read()
old = 'date(2026, 10, 8): 5410.91}")'
assert old in src
src = src.replace(old, 'date(2026, 10, 8): 5410.91, date(2026, 10, 9): 4951.66, date(2026, 10, 10): 4888.75, '
                       'date(2026, 10, 11): 5472.86, date(2026, 10, 12): 5000.0, date(2026, 10, 13): 5159.15, '
                       'date(2026, 10, 14): 5469.67, date(2026, 10, 15): 5505.0, date(2026, 10, 16): 5189.34}")')
old = '"    cfg.trade_until = UNTIL\\n")'
assert old in src
src = src.replace(old, '"    cfg.trade_until = UNTIL\\n    cfg.trade_from = FROM\\n")\n'
                       'head = head.replace("day.day, 6, 30, tzinfo", "day.day, 5, 0, tzinfo")\n'
                       'assert "day.day, 5, 0, tzinfo" in head\n'
                       # the drift policy: no top-up once the price is more than DRIFT above the average matched
                       'old_nt = "            if pol.get(\\"no_topups\\"):\\n                return\\n"\n'
                       'assert old_nt in head\n'
                       'head = head.replace(old_nt, old_nt + "            if pol.get(\\"drift\\") is not None and '
                       'd.price > pos.avg_price * (1.0 + pol[\\"drift\\"]):\\n                return\\n")')
old = '"POLICY": {}, "UNTIL": "21:30"}'
assert old in src
src = src.replace(old, '"POLICY": {}, "UNTIL": "21:30", "FROM": "06:15"}')
old = ("    for name, model, one_back, target, max_day, until in ARMS:\n"
       "        MODE[\"model\"] = model\n"
       "        g[\"POLICY\"] = {\"no_topups\": True} if one_back else {}\n")
assert old in src
src = src.replace(old, "    for name, model, one_back, target, max_day, until, cap, pol in ARMS:\n"
                       "        g[\"FROM\"] = \"06:15\"\n"
                       "        MODE[\"model\"] = model\n"
                       "        g[\"POLICY\"] = {\"no_topups\": True} if one_back else "
                       "dict(pol or {})\n")
old = '        r = g["replay"](day, target, BALANCE, 300.0, max_day)\n'
assert old in src
src = src.replace(old, '        r = g["replay"](day, target, BALANCE, cap, max_day)\n')
a, b = src.index("ARMS = ["), src.index("]\nBALANCE")
src = src[:a] + '''ARMS = [  # name, model, no top-ups, target to win, day limit, until, cap a horse, policy
    ("to win 400, cap 300 (the rule)", "novol", False, 400.0, 4000.0, "21:30", 300.0, None),
    ("to win 400, cap 200", "novol", False, 400.0, 4000.0, "21:30", 200.0, None),
    ("to win 400, cap 150", "novol", False, 400.0, 4000.0, "21:30", 150.0, None),
    ("to win 600, cap 300", "novol", False, 600.0, 4000.0, "21:30", 300.0, None),
    ("to win 400, cap 300, no top-up after a 5% drift", "novol", False, 400.0, 4000.0, "21:30", 300.0, {"drift": 0.05}),
    ("to win 400, cap 200, no top-up after a 5% drift", "novol", False, 400.0, 4000.0, "21:30", 200.0, {"drift": 0.05}),
    ("to win 400, cap 300, top-ups at 5%", "novol", False, 400.0, 4000.0, "21:30", 300.0, {"topup_bar": 0.05}),
    ("to win 400, cap 200, top-ups at 5%", "novol", False, 400.0, 4000.0, "21:30", 200.0, {"topup_bar": 0.05}),
    ("to win 400, cap 300, one back a horse", "novol", True, 400.0, 4000.0, "21:30", 300.0, None),
''' + src[b:]
# after the replay's own tables: each arm's days as a series (the spread of the result, the worst day, growth)
src += '''

res = pd.DataFrame(rows)
res["result_after_commission"] = pd.to_numeric(res["result_after_commission"], errors="coerce")
by = res.groupby("arm").result_after_commission
out = pd.DataFrame({"days": by.size(), "total": by.sum(), "mean_day": by.mean(), "sd_day": by.std(),
                    "worst_day": by.min(), "losing_days": by.apply(lambda x: int((x < 0).sum()))})
out["mean_over_sd"] = out.mean_day / out.sd_day
out["growth_log"] = res.assign(lg=np.log1p(res.result_after_commission / 5000.0)).groupby("arm").lg.sum()
st = res.groupby("arm").agg(staked=("staked", "sum"), horses=("horses", "sum"))
print("\\n== each arm over the days (result after commission, GBP5,000 in the account)")
print(out.join(st).sort_values("total", ascending=False).round(3).to_string())
'''
sys.argv = ["stake_per_horse_1016.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
