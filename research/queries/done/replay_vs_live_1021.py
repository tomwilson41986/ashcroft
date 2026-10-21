"""The owner (21 Oct): "all i care about is an accurate P&L. Ensure you are not overstating it, either now or in the
modeling / history" (read-only). The live P&L is Betfair's own (the ledger's settled result matched Betfair's statement
to the penny, 14-20 Oct). The replays (done/blend_ladder_replay_1020.py and every replay since) score rules against
each other; how far do their pounds stand from Betfair's? Each day 10-20 Oct replayed with the rule that was live that
day (10-18 Oct: to win GBP400, GBP300 a bet, no floor, top-ups to the winnings; 19-20 Oct: the GBP15 floor, GBP200 a
horse, top-ups to the stake; no ladder, no guard, no cap on the price, no cap on the top-ups; GBP4,000 a day) and the
day's own opening balance, against the live day: money, CLV, CLV x stake, the result after commission. The gap is the
factor every replay's absolute pounds carry (it does not change their ranking of rules, which share it)."""
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
src = src.replace(old, """head = head.replace(old, old + "    cfg.trade_from = '06:15'\\n"
                    "    for k, v in POLICY.get('cfg', {}).items():\\n"
                    "        setattr(cfg.limits if k == 'max_price' else cfg, k, v)\\n")""")
src = src.replace("    fit = fit_blend(L[L.day < day.isoformat()])\n", "    fit = None\n")
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


ARMS = [("as live that day", False, {})''' + src[b:]
old = """        BLEND["fit"], BLEND["tag"] = (fit if blend else None), (f"blend{day}" if blend else "served")
        g["POLICY"] = dict(pol)
        r = g["replay"](day, 400.0, BALANCE, 200.0, 4000.0)
"""
assert old in src
src = src.replace(old, """        over, cap = as_live(day.isoformat())
        _UNIT.min_move, _UNIT.limits.max_price, _UNIT.stake_floor = -100.0, 1000.0, over["stake_floor"]
        BLEND["fit"], BLEND["tag"] = None, f"as_live_{day}"
        g["POLICY"] = {"cfg": over}
        r = g["replay"](day, 400.0, OPEN.get(day.isoformat(), BALANCE), cap, 4000.0)
""")
src = src.replace('"out/blend_ladder_replay"', '"out/replay_vs_live"')
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
rp = res.set_index("day")[["staked", "clv", "clv_x_stake", "result_after_commission"]].add_prefix("replay_")
cmp_ = rp.join(lv)
print("\n== the replay of the rule live each day against the live day (Betfair's settled result)")
print(cmp_.round(3).to_string())
t = cmp_.sum()
print(f"\n   together: replay GBP{t.replay_staked:,.0f} staked, CLV x stake GBP{t.replay_clv_x_stake:,.2f}, result after "
      f"commission GBP{t.replay_result_after_commission:,.2f}; live GBP{t.live_staked:,.0f}, CLV x stake "
      f"GBP{t.live_clv_x_stake:,.2f}, result GBP{t.live_after_commission:,.2f}")
print(f"   replay CLV {t.replay_clv_x_stake / t.replay_staked:+.2%} against live {t.live_clv_x_stake / t.live_staked:+.2%}; "
      f"live result / replay result {t.live_after_commission / t.replay_result_after_commission:.2f}; live CLV x stake / "
      f"replay CLV x stake {t.live_clv_x_stake / t.replay_clv_x_stake:.2f}")
'''
sys.argv = ["replay_vs_live_1021.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
