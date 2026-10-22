---
name: day-report
description: The owner's standard day report on the live trading. Use whenever the owner asks how today is going, for every market, the current or expected profit or loss, the CLV or EV now, or a forecast for the day.
---

# The owner's day report (the standard approach, 22 Oct 2026)

The owner: "never give me inflated expected value, i need real figures that i can expect and forecast" and "all i
care about is an accurate P&L". Every report on the day's trading follows this, and nothing else is quoted.

## Run it

1. Write `research/queries/day_report_<mmdd>.py` (a suffix `_b`, `_c` for a second or third run that day):

   ```python
   """The owner's day report for <D Mon> (the standard approach: research/queries/done/day_report.py; read-only)."""
   import os
   import runpy
   import sys

   os.environ["REPORT_DAY"] = "<YYYY-MM-DD>"
   sys.argv = ["day_report.py"]
   runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "day_report.py"), run_name="__main__")
   ```

2. Commit, push to the development branch, watch the research-query run by its id, read the `day_report_<mmdd>` group
   of the job's log.
3. Move the query to `research/queries/done/` in the next commit (a root query reruns on every push).

## Report it: three figures, then every race

1. **Settled**: the races run and settled, as Betfair settled them, after commission. Once the day is closed, the
   account's funds against the day before are the figure (they rule over the ledger).
2. **The mark on the races to run**, after the non-runners' cuts: at the mid (the fair mark) and at the lays (what
   Betfair's Cash Out pays now). Say the mark is noisy while the books are thin (the mid and the lays far apart) and
   that more non-runners usually cut it further before the off.
3. **The forecast for the day**, with its range: on the whole record and on the last 7 days (both, plainly: the last
   7 days are the cautious guide while they run below the record), the 10th-90th range, and the record itself (days,
   total, mean a day, days up).

Then the table of every race: off, course, horses, staked, the mark at the mid, Cash Out, the forecast, the result.

## Never

- Never quote the trader's raw expected value (edge x stake), the CLV at the fill, or the mark at the best back as a
  figure to expect: on 10-21 Oct the expected value ran at 0.61 of the settled result over the record and about 0
  over the last 7 days (ledger `forecast-1022`), and the CLV at the fill halves after the cuts (`execution-audit-1021`).
- Never quote a replay's pounds without the live scale: replays are scored after the cuts, and live has run at about
  0.80 of that (`replay-sources-1021`).
- Never round a loss into a profit or leave out the Cash Out figure because it is negative.
