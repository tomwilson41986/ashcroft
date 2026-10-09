"""Which rule would have made the most, on the days we traded (the owner's question, 9 Oct; read-only).

research/queries/done/stake_target_replay.py's machinery (every recorded book of the day, the live rule replayed a
minute at a time against them, fills against the recorded sizes, the account's money as Betfair holds it, settled at
Betfair's price files), run 1-8 Oct (8 Oct once its price file is archived) with GBP5,000 in the account, over the
choices the live record points at:

- the closing model: as the trader read it (the volume model where the book carries each runner's matched money, the
  model fitted without volume where it does not), or the model fitted without volume on every book;
- top-ups: as live (the trader tops each horse up to its target), or one back a horse;
- money: GBP400 or GBP250 to win, GBP300 a bet, no day limit or GBP4,000 a day;
- the stop: 15 minutes before each off, or nothing new after 12:00 UK.

The recorded books mix the trader's reads (the live key's feed from 6 Oct 15:25 UK, each runner's matched money) and
the recorder's (the delayed key, none), so "as read" is the live trader's choice where it read the book. CLV x stake is
the expected profit of the hedged backs before commission; the result is after 2% commission.
"""
import copy
import os
import sys
from datetime import date

path = os.path.join(os.getcwd(), "research", "queries", "done", "stake_target_replay.py")
src = open(path, encoding="utf-8").read()
head = src[:src.index("\nrows = []\n")]
old = "date(2026, 10, 6): 1956.62}"
assert old in head and "policy={})" in head
head = head.replace(old, "date(2026, 10, 6): 1956.62, date(2026, 10, 7): 2577.13, date(2026, 10, 8): 5410.91}")
head = head.replace("policy={})", "policy=POLICY)")
head = head.replace('OUT = Path("out/stake_target_replay")', 'OUT = Path("out/strategy_replay")')
head = head.replace("    cfg.target, cfg.limits.max_stake, cfg.limits.max_daily_turnover = target, max_stake, max_day\n",
                    "    cfg.target, cfg.limits.max_stake, cfg.limits.max_daily_turnover = target, max_stake, max_day\n"
                    "    cfg.trade_until = UNTIL\n")
assert "cfg.trade_until = UNTIL" in head
sys.argv = ["strategy_replay_1009.py"]
g = {"__name__": "__main__", "__file__": path, "POLICY": {}, "UNTIL": "21:30"}
exec(compile(head, path, "exec"), g)

pd, np, strategy, ts_mod = g["pd"], g["np"], g["strategy"], g["ts_mod"]
_UNIT, _orig = g["_UNIT"], g["_orig_plan_race"]
MODE = {"model": "as_read"}
_cache: dict = {}


def plan(views, cfg, bank):
    """The race planned once a book and a model (uncapped), sized to each replay's target: with the model fitted
    without volume, every runner's matched money is read as none (as the delayed key's feed gives it)."""
    if MODE["model"] == "novol":
        views = [copy.copy(v) for v in views]
        for v in views:
            v.traded = 0.0
    key = (MODE["model"],) + tuple((v.selection_id, v.back, v.lay, v.last_traded, v.model_price, v.status,
                                    float(v.traded or 0.0) > 0) for v in views)
    if key not in _cache:
        _cache[key] = _orig(views, _UNIT, bank)
    size = {v.selection_id: v.back_size for v in views}
    out = []
    for d in _cache[key]:
        d2 = type(d)(**{**d.__dict__})
        d2.target = min(d.target * cfg.target / _UNIT.target, cfg.limits.max_stake)
        d2.back_size = size.get(d.selection_id, d.back_size)
        out.append(d2)
    return out


ts_mod.plan_race = plan

NOLIMIT = 1000000.0
ARMS = [  # name, model, no top-ups, target, day limit, until
    ("live rule (as read, top-ups, 400, no limit)", "as_read", False, 400.0, NOLIMIT, "21:30"),
    ("as read, one back, 400, no limit", "as_read", True, 400.0, NOLIMIT, "21:30"),
    ("as read, top-ups, 400, 4000/day", "as_read", False, 400.0, 4000.0, "21:30"),
    ("no-volume, top-ups, 400, no limit", "novol", False, 400.0, NOLIMIT, "21:30"),
    ("no-volume, one back, 400, no limit", "novol", True, 400.0, NOLIMIT, "21:30"),
    ("no-volume, top-ups, 400, 4000/day", "novol", False, 400.0, 4000.0, "21:30"),
    ("no-volume, one back, 400, 4000/day", "novol", True, 400.0, 4000.0, "21:30"),
    ("no-volume, one back, 250, 4000/day", "novol", True, 250.0, 4000.0, "21:30"),
    ("no-volume, top-ups, 400, no limit, to 12:00", "novol", False, 400.0, NOLIMIT, "12:00"),
    ("no-volume, one back, 400, no limit, to 12:00", "novol", True, 400.0, NOLIMIT, "12:00"),
]
BALANCE = 5000.0

rows = []
for day in g["DAYS"]:
    D = g["load"](day)
    if D is None:
        continue
    g["D"] = D
    for name, model, one_back, target, max_day, until in ARMS:
        MODE["model"] = model
        g["POLICY"] = {"no_topups": True} if one_back else {}
        g["UNTIL"] = until
        r = g["replay"](day, target, BALANCE, 300.0, max_day)
        r["arm"] = name
        rows.append(r)
        print({k: r.get(k) for k in ("day", "arm", "horses", "staked", "clv", "clv_x_stake", "first_clv",
                                     "topup_clv", "result_after_commission", "horses_held", "last_back_uk")},
              flush=True)
if not rows:
    raise SystemExit("no day could be settled")
res = pd.DataFrame(rows)
res.to_csv(g["OUT"] / "arms.csv", index=False)
pd.set_option("display.width", 250)
pooled = res.groupby("arm", sort=False).agg(days=("day", "nunique"), horses=("horses", "sum"),
                                            staked=("staked", "sum"), clv_x_stake=("clv_x_stake", "sum"),
                                            result=("result_after_commission", "sum"),
                                            held=("horses_held", "sum"))
pooled["clv"] = (pooled.clv_x_stake / pooled.staked).round(4)
print("\n== the days together, GBP5,000 in the account (CLV x stake before commission; result after 2%)")
print(pooled.round(2).to_string())
print("\n== CLV x stake by day")
print(res.pivot_table(index="arm", columns="day", values="clv_x_stake", sort=False).round(0).to_string())
print("\n== result after commission by day")
print(res.pivot_table(index="arm", columns="day", values="result_after_commission", sort=False).round(0).to_string())
