"""No backs where a cut removal is least likely: the lead of removal-by-market-1109 through the trader's code
(read-only; registered before the run: ledger `removal-tilt-replay-1109`).

removal-by-market-1109 (run 614, 2 Oct - 8 Nov) found Betfair's reduction factors fair on average (so a removal is no
free gain), but the rule's own runners close better where a cut removal is likelier: at their first crossing, on days
the forecast had not seen, the lowest third of markets by the forecast closed +1.46% after the cut on GBP66,149, the
middle +4.40% on GBP43,183, the top +5.99% on GBP36,685. The lowest third takes the most money (short prices, staked
to win GBP400) at the least edge, and the day limit binds on most days. Does the rule make more if it leaves those
markets alone and the day's GBP4,000 goes to the rest? A filter that drops money without redeploying it overstates
(replay-nr-bar-1021), so the test is the full replay.

The forecast, as registered: a logistic for "a runner with a reduction factor of 2.5% or more comes out after the
market's first read", on the field (active runners), the log minutes to the off and Ireland or not, fitted each
replay day on the first reads of every day before it (2 Oct on; done/removal_by_market_1109.py's table). Removals
already at the meeting are left out: the table showed nothing in them. At each decision the trader's own book gives
the three inputs; the arm sends no back when the forecast is under the lowest-third cut of the forecast on its own
training markets. Nothing in it is later than the decision.

Arms, both under the live rule (done/replay_1108.py's machinery; the ladder, the guard, 5 top-ups, GBP4,000 a day),
12 Oct to the last day with a price file:
- served: the rule as it stands;
- skip the low third: the same, with no back in a market under the cut.

Registered reading (before any figure of the replay is seen): CLV x stake after the non-runners' reduction factors a
day (`clv_x_stake_reduced`), the arm less served on the same day; it passes only with a positive sum, the 90% day-
bootstrap interval of the mean daily difference above zero, and a deflated Sharpe of the daily differences >= 0.95
with 2 trials counted (this arm, and the descriptive read that suggested it). A pass is a candidate for the owner,
not a change: nothing here touches the live trader or its config.
"""
import os
import runpy
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
here = os.path.join(os.getcwd(), "research", "queries", "done")

# ------------------------------------------------------------------ the forecast's table: each market's first read
print("== the forecast's table (done/removal_by_market_1109.py, run again)", flush=True)
TILT_M = runpy.run_path(os.path.join(here, "removal_by_market_1109.py"))["M"][
    ["day", "market_id", "field", "to_off", "ie", "y"]].copy()
print(f"\nthe forecast's table: {len(TILT_M):,} markets, {TILT_M.day.nunique()} days, a cut removal to come in "
      f"{TILT_M.y.mean():.1%}", flush=True)

path = os.path.join(here, "replay_1108.py")
text = open(path, encoding="utf-8").read()
a = text.index("ARMS = [  # name, blend, policy, day limit")
b = text.index("''' + src[b:]")
text = text[:a] + '''ARMS = [  # name, blend, policy, day limit
    ("served", False, {"cfg_max_topups": 5}, 4000.0),
    ("skip the low third", False, {"cfg_max_topups": 5, "skip_low": True}, 4000.0),
''' + text[b:]

TILT_CODE = '''
TILT = {"fit": None, "seen": set(), "skipped": set()}


def _tilt_minutes(a_, b_):
    a_, b_ = pd.Timestamp(a_), pd.Timestamp(b_)
    a_ = a_.tz_localize("UTC") if a_.tzinfo is None else a_.tz_convert("UTC")
    b_ = b_.tz_localize("UTC") if b_.tzinfo is None else b_.tz_convert("UTC")
    return (a_ - b_).total_seconds() / 60.0


def _tilt_x(field_, to_off_, ie_):
    return np.array([float(field_), float(np.log(max(float(to_off_), 15.0))), float(ie_)])


def tilt_fit(day_):
    """The removal forecast fitted on the first reads of the days before ``day_``, and its lowest-third cut."""
    from sklearn.linear_model import LogisticRegression
    tr = TILT_M[TILT_M.day < day_.isoformat()]
    if tr.day.nunique() < 10 or tr.y.nunique() < 2:
        return None
    X = np.vstack([_tilt_x(f_, t_, i_) for f_, t_, i_ in zip(tr.field, tr.to_off, tr.ie)])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    m_ = LogisticRegression(C=1.0, max_iter=1000).fit((X - mu) / sd, tr.y.to_numpy(int))
    b0, bb = float(m_.intercept_[0]), m_.coef_[0].astype(float)
    p = 1.0 / (1.0 + np.exp(-(b0 + ((X - mu) / sd) @ bb)))
    return {"mu": mu, "sd": sd, "b0": b0, "b": bb, "cut": float(np.quantile(p, 1.0 / 3.0)), "n": int(len(tr)),
            "days": int(tr.day.nunique())}


def tilt_p(f_, m, book, now):
    field_ = sum(1 for q_ in book.runners.values() if q_.status == "ACTIVE")
    x_ = (_tilt_x(field_, _tilt_minutes(m.start, now), 1.0 if str(m.country) == "IE" else 0.0) - f_["mu"]) / f_["sd"]
    return float(1.0 / (1.0 + np.exp(-(f_["b0"] + x_ @ f_["b"]))))


'''

old = 'sys.argv = ["replay_1102.py"]\n'
assert old in text, "the replay's last edit"
text = text.replace(old, '''import re as _re
src, _n = _re.subn(r"REPLAY_DAYS = .*\\n", "REPLAY_DAYS = [date(2026, 10, 12) + timedelta(n) for n in range(40)]\\n",
                   src, count=1)
assert _n == 1, "the replay's days"
for _old, _new in (
        ("class LadderSession(", TILT_CODE + "class LadderSession("),
        ('    def execute(self, m, book, d, now):\\n        pol = self.policy\\n',
         '    def execute(self, m, book, d, now):\\n'
         '        if self.policy.get("skip_low") and TILT["fit"] is not None:\\n'
         '            TILT["seen"].add(m.market_id)\\n'
         '            if tilt_p(TILT["fit"], m, book, now) < TILT["fit"]["cut"]:\\n'
         '                TILT["skipped"].add(m.market_id)\\n'
         '                return\\n'
         '        pol = self.policy\\n'),
        ('    g["D"] = D\\n',
         '    g["D"] = D\\n'
         '    TILT["fit"] = tilt_fit(day)\\n'
         '    print(f"  the removal forecast for {day}: " + ("none (too few days)" if TILT["fit"] is None else '
         'f"fitted on {TILT[\\'fit\\'][\\'n\\']:,} markets of {TILT[\\'fit\\'][\\'days\\']} days, lowest-third cut '
         '{TILT[\\'fit\\'][\\'cut\\']:.3f}, weights {np.round(TILT[\\'fit\\'][\\'b\\'], 3)}"), flush=True)\\n'),
        ('        r.update(reduced(day.isoformat(), BALANCE, max_day))\\n',
         '        r.update(reduced(day.isoformat(), BALANCE, max_day))\\n'
         '        r["tilt_seen"], r["tilt_skipped"] = len(TILT["seen"]), len(TILT["skipped"])\\n'
         '        print(f"    {name}: markets it would have backed in {r[\\'tilt_seen\\']}, left alone '
         '{r[\\'tilt_skipped\\']}", flush=True)\\n'
         '        TILT["seen"], TILT["skipped"] = set(), set()\\n'),
        ('"out/replay_1108"', '"out/removal_tilt_replay_1109"')):
    assert src.count(_old) == 1, _old
    src = src.replace(_old, _new)
sys.argv = ["removal_tilt_replay_1109.py"]
''')
old = 'exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})'
assert old in text, "the replay's run"
text = text.replace(old, 'G = {"__name__": "__main__", "__file__": path, "TILT_M": TILT_M}\n'
                         'exec(compile(src, path, "exec"), G)')

ns = {"__name__": "replay_1108_wrapped", "__file__": path, "TILT_M": TILT_M, "TILT_CODE": TILT_CODE}
exec(compile(text, path, "exec"), ns)
res = ns["G"]["res"]

# ------------------------------------------------------------------ the registered reading
from research.overfit import dsr_check  # noqa: E402

N_TRIALS = 2
arm = "skip the low third"
piv = res.pivot_table(index="day", columns="arm", values="clv_x_stake_reduced", sort=False)
stk = res.pivot_table(index="day", columns="arm", values="staked", sort=False)
left = res[res.arm == arm]
print("\n== the registered reading: CLV x stake after the reductions, the arm less the served rule, a day")
print(f"days replayed: {len(piv)} ({piv.index.min()} .. {piv.index.max()}); markets left alone "
      f"{int(left.tilt_skipped.sum()):,} of {int(left.tilt_seen.sum()):,} it would have backed in")
d = (piv[arm] - piv["served"]).dropna().to_numpy(float)
if len(d) < 2:
    print(f"too few days ({len(d)})")
else:
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), (5000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [5, 95])
    chk = dsr_check(d, N_TRIALS)
    passed = bool(d.sum() > 0 and lo > 0 and chk["passed"])
    print(f"{arm}: served GBP{piv['served'].sum():,.2f} on GBP{stk['served'].sum():,.0f}; the arm "
          f"GBP{piv[arm].sum():,.2f} on GBP{stk[arm].sum():,.0f}; difference GBP{d.sum():+,.2f} over {len(d)} days, "
          f"mean GBP{d.mean():+.2f} a day (90% {lo:+.2f} to {hi:+.2f}), ahead on {(d > 0).sum()} days; deflated Sharpe "
          f"{chk['dsr']} with {N_TRIALS} trials ({chk['reason'] or 'clears 0.95'}): "
          f"{'PASSES the registered reading' if passed else 'does NOT pass'}")
