"""The replay of the evening of 20 Oct (read-only): done/blend_ladder_replay_1020.py's machinery (the trader's own code a
minute at a time against the recorded books, fills against each level's own money, GBP5,000 in the account, settled at
Betfair's price files), 12-20 Oct as the price files allow. The ladder is now the live code's own (Session._ladder,
cfg.ladder; config_live carries it, with the guard and nothing over 200), so the replay switches it through the
config (cfg.ladder from the policy's cfg_ladder) and never through LadderSession's own "ladder" policy. Arms:

- the best back only, the rule of 20 Oct (no ladder, no guard, max price 1000; the plan read the same way);
- the ladder at GBP4,000 a day (the rule from 21 Oct);
- the ladder at GBP6,000 a day;
- the ladder with at most 5 top-ups a horse (a horse with 6 fills takes no more).
"""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "blend_ladder_replay_1020.py")
src = open(path, encoding="utf-8").read()
src = src.replace("REPLAY_DAYS = [date(2026, 10, d) for d in range(12, 20)]",
                  "REPLAY_DAYS = [date(2026, 10, d) for d in range(12, 21)]")
old = """head = head.replace(old, old + "    cfg.trade_from = '06:15'\\n")"""
assert old in src, "trade_from"
src = src.replace(old, """head = head.replace(old, old + "    cfg.trade_from = '06:15'\\n"
                    "    cfg.ladder = bool(POLICY.get('cfg_ladder', True))\\n"
                    "    if POLICY.get('rule_of_20_oct'):\\n"
                    "        cfg.min_move, cfg.limits.max_price = -100.0, 1000.0\\n")""")
old = """        super().execute(m, book, d, now)


class LevelExchange"""
assert old in src, "execute end"
src = src.replace(old, """        mt = pol.get("max_topups")
        key = (m.market_id, d.selection_id)
        if mt is not None:
            if not hasattr(self, "_nfills"):
                self._nfills = {}
            if self._nfills.get(key, 0) >= mt + 1:
                return
            had = self.positions.get(key)
            before = had.matched if had is not None else 0.0
        super().execute(m, book, d, now)
        if mt is not None:
            now_held = self.positions.get(key)
            if now_held is not None and now_held.matched > before + 1e-9:
                self._nfills[key] = self._nfills.get(key, 0) + 1


class LevelExchange""")
a, b = src.index("ARMS = ["), src.index("]\nrows = []")
src = src[:a] + '''ARMS = [  # name, blend, policy, day limit
    ("best back only (the rule of 20 Oct)", False, {"cfg_ladder": False, "rule_of_20_oct": True}, 4000.0),
    ("the ladder, GBP4,000 (the rule from 21 Oct)", False, {}, 4000.0),
    ("the ladder, GBP6,000", False, {}, 6000.0),
    ("the ladder, at most 5 top-ups a horse", False, {"max_topups": 5}, 4000.0),
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
src = src.replace(old, '        r = g["replay"](day, 400.0, BALANCE, 200.0, max_day)\n')
src = src.replace('"out/blend_ladder_replay"', '"out/replay_1020"')
src += '''
print("\\n== the spread of the days (result after commission)")
print(res.groupby("arm", sort=False).agg(mean_day=("result_after_commission", "mean"),
                                         sd_day=("result_after_commission", "std"),
                                         worst_day=("result_after_commission", "min")).round(2).to_string())
print("\\n== 19-20 Oct (the live days of the new staking): staked, CLV, CLV x stake, result after commission")
print(res[res.day >= "2026-10-19"][["day", "arm", "horses", "staked", "clv", "clv_x_stake", "first_clv", "topup_clv",
                                    "result_after_commission"]].to_string(index=False))
'''
sys.argv = ["replay_1020.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
