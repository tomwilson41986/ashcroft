"""The owner's question (20 Oct): should the GBP4,000 day limit go? "I don't like leaving money and profit behind."
(read-only). done/blend_ladder_replay_1020.py's replay (the trader's own code a minute at a time, 12-19 Oct, fills
against the recorded sizes, settled at Betfair's price files) with GBP5,500 in the account (the account closed 19 Oct
at GBP5,465.69), so without the day limit the account's own money is the limit: a back larger than the money not
yet back from its races is refused, and the session holds new backs until it is. Arms: the live rule, and the live rule
with the ladder (blend-ladder-replay-1020: the best arm), each at GBP4,000, GBP6,000 and no day limit. Before (1-9 Oct, the rule of then, GBP5,000):
no limit made CLV x stake GBP1,686 against GBP2,207 with the limit (strategy-replay-1010)."""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "blend_ladder_replay_1020.py")
src = open(path, encoding="utf-8").read()
a, b = src.index("ARMS = ["), src.index("]\nrows = []")
src = src[:a] + '''NOLIMIT = 1000000.0
ARMS = [  # name, blend, policy, day limit
    ("live, GBP4,000 a day", False, {}, 4000.0),
    ("live, GBP6,000 a day", False, {}, 6000.0),
    ("live, no day limit", False, {}, NOLIMIT),
    ("live + the ladder, GBP4,000 a day", False, {"ladder": BAR}, 4000.0),
    ("live + the ladder, GBP6,000 a day", False, {"ladder": BAR}, 6000.0),
    ("live + the ladder, no day limit", False, {"ladder": BAR}, NOLIMIT),
''' + src[b:]
old = "    for name, blend, pol in ARMS:\n"
assert old in src
src = src.replace(old, "    for name, blend, pol, max_day in ARMS:\n")
old = '        r = g["replay"](day, 400.0, BALANCE, 200.0, 4000.0)\n'
assert old in src
src = src.replace(old, '        r = g["replay"](day, 400.0, BALANCE, 200.0, max_day)\n')
old = "BAR, BALANCE = 0.03, 5000.0"
assert old in src
src = src.replace(old, "BAR, BALANCE = 0.03, 5500.0")
src = src.replace('"out/blend_ladder_replay"', '"out/day_limit_replay"')
src += '''
print("\\n== the spread of the days (result after commission) and the money held for want of funds")
print(res.groupby("arm", sort=False).agg(mean_day=("result_after_commission", "mean"),
                                         sd_day=("result_after_commission", "std"),
                                         worst_day=("result_after_commission", "min"),
                                         horses_held=("horses_held", "sum"),
                                         held_never_backed=("held_never_backed", "sum"),
                                         peak_open=("peak_open", "max")).round(2).to_string())
print("\\n== staked by day")
print(res.pivot_table(index="arm", columns="day", values="staked", sort=False).round(0).to_string())
print("\\n== last back (UK) by day")
print(res.pivot_table(index="arm", columns="day", values="last_back_uk", aggfunc="first", sort=False).to_string())
'''
sys.argv = ["day_limit_replay_1020.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
