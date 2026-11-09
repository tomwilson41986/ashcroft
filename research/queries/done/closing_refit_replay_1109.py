"""The closing model refitted on the trader's own books, re-tested through the trader's code (read-only; registered
before the run: ledger `closing-refit-replay-1109`).

The served closing model (data/models/closing_model_novol.json: 0.774 market + 0.263 ours) was fitted on January-March
2026 morning WAPs and is read on a live book. benter-blend-1020 refitted it on the trader's own books and it was ahead
on a one-back read; blend-ladder-replay-1020 then replayed it through the trader's code on 7 days (12-18 Oct) and it
was behind (CLV x stake GBP1,878 against GBP2,131), with 6-17 days to fit on. This is that test again, once, on the
record since: the same machinery (done/replay_1108.py, the rule as config_live.json stands), 12 Oct to the last day
with a price file, the refit walk-forward (each day fitted on 1 Oct to the day before, now up to 5 weeks).

Arms, all under the live rule (the ladder, the guard, at most 5 top-ups, GBP4,000 a day):
- served: the closing model as served;
- refit, all hours: done/blend_ladder_replay_1020.py's spread blend as it was (every 10th read of a market, any hour);
- refit, before 10:00: the same blend fitted only on reads before 10:00 UK, the hours the rule now trades.

Registered reading (before any figure is seen):
- the measure is CLV x stake after the non-runners' reduction factors, a day (`clv_x_stake_reduced`), each refit arm
  less the served arm on the same day;
- a refit arm passes only if the summed difference is positive, the 90% interval of the mean daily difference (day
  bootstrap) is above zero, and the deflated Sharpe of the daily differences is at least 0.95 with 3 trials counted
  (the two refits here and the 20 Oct trial of the same idea);
- a pass is a candidate for the owner, not a change: nothing here touches the live trader or its config.
"""
import os
import re
import sys

import numpy as np

here = os.path.join(os.getcwd(), "research", "queries", "done")
path = os.path.join(here, "replay_1108.py")
text = open(path, encoding="utf-8").read()

# the arms: the live rule on the served closing model and on the two refits
a = text.index("ARMS = [  # name, blend, policy, day limit")
b = text.index("''' + src[b:]")
text = text[:a] + '''ARMS = [  # name, blend (None, or the refit's training reads), policy, day limit
    ("served", None, {"cfg_max_topups": 5}, 4000.0),
    ("refit, all hours", "all", {"cfg_max_topups": 5}, 4000.0),
    ("refit, before 10:00", "pre10", {"cfg_max_topups": 5}, 4000.0),
''' + text[b:]

# after replay_1108's own edits of the machinery: the days, the refit's data through the record, a fit for each arm
old = 'sys.argv = ["replay_1102.py"]\n'
assert old in text, "the replay's last edit"
text = text.replace(old, '''import re as _re
src, _n = _re.subn(r"REPLAY_DAYS = .*\\n", "REPLAY_DAYS = [date(2026, 10, 12) + timedelta(n) for n in range(40)]\\n",
                   src, count=1)
assert _n == 1, "the replay's days"
for _old, _new in (
        ("for n in range(19)]", "for n in range(60)]"),
        ('    bk["day"] = day\\n',
         '    bk["day"] = day\\n'
         '    bk["uk_h"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce").dt.tz_convert("Europe/London").dt.hour\\n'),
        ("    fit = fit_blend(L[L.day < day.isoformat()])\\n",
         "    _tr = L[L.day < day.isoformat()]\\n"
         "    fits = {'all': fit_blend(_tr), 'pre10': fit_blend(_tr[_tr.uk_h < 10])}\\n"
         "    fit = fits['all']\\n"
         "    print(f\\"  fitted on {_tr.day.nunique()} days: {_tr.snap.nunique():,} reads, \\"\\n"
         "          f\\"{_tr[_tr.uk_h < 10].snap.nunique():,} before 10:00 UK; before 10:00 weights \\"\\n"
         "          f\\"{np.round(fits['pre10']['beta'], 3)}\\")\\n"),
        ('(fit if blend else None), (f"blend{day}" if blend else "served")',
         '(fits[blend] if blend else None), (f"blend{blend}{day}" if blend else "served")'),
        ('"out/replay_1108"', '"out/closing_refit_replay_1109"')):
    assert src.count(_old) == 1, _old
    src = src.replace(_old, _new)
sys.argv = ["closing_refit_replay_1109.py"]
''')
old = 'exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})'
assert old in text, "the replay's run"
text = text.replace(old, 'G = {"__name__": "__main__", "__file__": path}\nexec(compile(src, path, "exec"), G)')

ns = {"__name__": "replay_1108_wrapped", "__file__": path}
exec(compile(text, path, "exec"), ns)
res = ns["G"]["res"]

# ------------------------------------------------------------------ the registered reading
sys.path.insert(0, os.getcwd())
from research.overfit import dsr_check  # noqa: E402

N_TRIALS = 3
piv = res.pivot_table(index="day", columns="arm", values="clv_x_stake_reduced", sort=False)
stk = res.pivot_table(index="day", columns="arm", values="staked", sort=False)
print("\n== the registered reading: CLV x stake after the reductions, each refit less the served rule, a day")
print(f"days replayed: {len(piv)} ({piv.index.min()} .. {piv.index.max()})")
rng = np.random.default_rng(0)
for arm in ("refit, all hours", "refit, before 10:00"):
    d = (piv[arm] - piv["served"]).dropna().to_numpy(float)
    if len(d) < 2:
        print(f"{arm}: too few days ({len(d)})")
        continue
    boot = d[rng.integers(0, len(d), (5000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [5, 95])
    chk = dsr_check(d, N_TRIALS)
    passed = bool(d.sum() > 0 and lo > 0 and chk["passed"])
    print(f"{arm}: served GBP{piv['served'].sum():,.2f} on GBP{stk['served'].sum():,.0f}; the refit "
          f"GBP{piv[arm].sum():,.2f} on GBP{stk[arm].sum():,.0f}; difference GBP{d.sum():+,.2f} over {len(d)} days, "
          f"mean GBP{d.mean():+.2f} a day (90% {lo:+.2f} to {hi:+.2f}), ahead on {(d > 0).sum()} days; deflated Sharpe "
          f"{chk['dsr']} with {N_TRIALS} trials ({chk['reason'] or 'clears 0.95'}): "
          f"{'PASSES the registered reading' if passed else 'does NOT pass'}")
