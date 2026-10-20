"""The rule candidates of the review of 20 Oct (ledger review-1020), through the trader's own code (read-only).

done/blend_ladder_replay_1020.py's replay (12-19 Oct as the price files allow, a minute at a time, fills against the
recorded sizes and each level's own money, GBP5,000 in the account, GBP4,000 a day), on the live config as it stands
for 21 Oct: the ladder, the guard (never under our predicted BFSP), nothing over 200, to win GBP400 with a GBP15 floor
and a GBP200 cap. The review found:

- the closing model's expected CLV is over-confident where it is small: 10-20 Oct the fills expected 3-6% (GBP29,114,
  70% of the money) closed at -0.3% to +1.2%; 6-10% at +1.6% to +3.8%; over 10% at +15% to +56%;
- the top-ups carry 80% of the money at about +1%, the first backs +11.4%; a top-up at a price longer than the horse's
  average matched (the horse drifting) closed at -3.2% (topups-1020), at a shorter one +2.8%;
- under 5.0 is 40% of the money at about 0% CLV (today -14%).

Arms (price and depth, not time):
  live          the rule from 21 Oct;
  bar 6%        nothing backed under expected CLV 6% (first backs and top-ups);
  no drift      no top-up at a price longer than the horse's average matched so far;
  short bar     under 5.0 nothing under expected CLV 6%;
  all three     bar 6% with no drift (the short bar is then the same bar).
"""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "blend_ladder_replay_1020.py")
src = open(path, encoding="utf-8").read()
src = src.replace("REPLAY_DAYS = [date(2026, 10, d) for d in range(12, 20)]",
                  "REPLAY_DAYS = [date(2026, 10, d) for d in range(12, 21)]")
old = """        if pol.get("ladder") is not None and d.price and d.price > 1 and d.p_pool > 0:"""
assert old in src
src = src.replace(old, """        if pol.get("no_drift"):
            held = self.positions.get((m.market_id, d.selection_id))
            if held is not None and held.matched > 0 and d.price > held.avg_price + 1e-9:
                return
        if pol.get("short_under") is not None and d.price < pol["short_under"] and not (
                np.isfinite(d.edge) and d.edge >= pol["short_min_edge"]):
            return
""" + old)
a, b = src.index("ARMS = ["), src.index("]\nrows = []")
src = src[:a] + '''ARMS = [  # name, blend, policy (the ladder is the live config's own, not the replay's policy)
    ("live (the rule from 21 Oct)", False, {}),
    ("bar 6%", False, {"bar": 0.06}),
    ("no drift", False, {"no_drift": True}),
    ("short bar (under 5.0 at 6%)", False, {"short_under": 5.0, "short_min_edge": 0.06}),
    ("bar 6% + no drift", False, {"bar": 0.06, "no_drift": True}),
''' + src[b:]
src = src.replace('"out/blend_ladder_replay"', '"out/rule_candidates_replay"')
src += '''
print("\\n== the spread of the days (result after commission)")
print(res.groupby("arm", sort=False).agg(mean_day=("result_after_commission", "mean"),
                                         sd_day=("result_after_commission", "std"),
                                         worst_day=("result_after_commission", "min"),
                                         last_back=("last_back_uk", "max")).round(2).to_string())
'''
sys.argv = ["rule_candidates_replay_1020.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
