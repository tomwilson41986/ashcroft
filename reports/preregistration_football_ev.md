# Pre-registration: the expected-value rule, paper-traded at the recorded Betfair books

racing² football research. Written 2026-11-04T15:10Z by the modelling agent, at the owner's request ("set up a paper
trader for your recommended approach to maximise profit"), before any forward book was read. Code:
`research/open_close/football_ev_trial.py`, tests `tests/test_football_ev_trial.py`. The scorer reads the JSON block
at the end of this file, checks the frozen models' SHA-256 against it, and records this file's SHA-256 beside every
result.

## Why this rule

**Development evidence** (`reports/football_model_rethink.md`, "Expected-value rule, trade types and staking"):
expected value above 2%, greened at the close.

| Period | Return |
|---|---|
| 2024/25 | +2.30% (level stakes, a tick off fair) |
| 2025/26 | +0.99% |
| Aug-Sep 2026, Betfair's own prices, a tick worse each end | +0.74% |
| Aug-Sep 2026, at the traded price | +3.19% |

**What the evidence showed:**
- Holding to the result loses.
- The draw carries no edge.
- Stakes in proportion to expected value give the best return for the risk.
- The spread is worth more than the edge, hence resting orders.

**Selection risk.** About 130 variants have been read on 2025/26 and the Aug-Sep 2026 prices. This is development
evidence, and only the forward record judges the rule.

## The rule

**1. Signals.** Frozen stacks (`models/football/ev_v1`, SHA-256 below):
- the mean of a ridge and a LightGBM on the team features plus `home_adv`, `team_news`, `form_short` and `x_short`;
- fitted once on every same-book move from 2013/14 to the latest held;
- forecasting each match's home and away chance move from the collection moment (Friday 17:00 / Tuesday 13:00 UK) to
  the close.

The frozen calibration (`calibration.json`: the expected move = -0.00477 + 1.3897 x the forecast, fitted on the
2024/25 out-of-sample forecasts) turns the forecast into an expected move.

**2. Selection.** Each side's expected value at Betfair's early book (football-data's collection snapshot):
- the expected CLV at fair, 2% commission off a gain;
- less the round trip a tick off fair at each end.

Every side above 2% is traded the better way round: back first if the chance is expected to rise, lay first if it
is expected to fall. MATCH_ODDS home and away only, in the 22 divisions, early books at a best-back sum of 1.05 or
under.

**3. Entry: resting orders** (as `reports/preregistration_football_stack_resting.md`):
- posted a tick inside the spread at the first recorded book within 60 minutes after the collection moment;
- filled only when a later recorded book proves it;
- cancelled 90 minutes before the off.

**4. Exit.** Greened at the last two-sided recorded book within 30 minutes of the off. 2% commission on each market's
net.

**5. Stakes.** A paper bank of GBP1,000:
- each collection window, 25% of the bank is shared among the window's signals in proportion to their expected value;
- at most 10% of the bank goes on one signal;
- capital is a back's stake or a lay's liability;
- an unfilled order's capital comes back;
- the bank moves window by window.

**Reported beside the headline, not judged:**
- the crossing variant (taking the best price at the early book), to measure what resting is worth;
- the adverse selection of the fills;
- the bank path.

## The gate

Judged on the filled trades at the recorded books, from the first Friday collection on or after the first recorded
football book. The trial runs to the later of 300 trades and 90 days, at most 365 days. It passes only if all hold:

- the return's 95% lower bound (bootstrap by day) above zero;
- the mean CLV (fill against the close mid) above zero;
- both halves of the window above zero.

**Until it passes, no real money goes on this rule.** Real stakes need the owner's separate decision (`FOOTBALL_LIVE`
stays unset).

## The rule as the scorer reads it

```json
{
 "model": "stack_ridge_lgbm_ev_v1_frozen",
 "variant": "expected value above 2%, resting entry, greened at the close, stakes in proportion to expected value",
 "spec": "team_all+home_adv+team_news+form_short+x_short",
 "models_dir": "models/football/ev_v1",
 "model_sha256": {
  "spec.json": "95a3ce0edb74b5d371fc546fc67c3b0536d8ffb981a20fbd4619fdb436c258da",
  "gbm_h.txt": "16f8b06bdf8ff3d1550a34b4c4c053a4b14ce75092e3dd48994f2101c31a08c8",
  "gbm_a.txt": "8491ed6b58a5623f4b3100b1c30e1fd125a4199387a8093bd0517665e9264178",
  "calibration.json": "e1436b40b74e79c729f24b880af857e08d24d4f3ea9a0dbb5415115d938eee12"
 },
 "market_types": ["MATCH_ODDS"],
 "selections": ["H", "A"],
 "leagues_fd": ["B1", "D1", "D2", "E0", "E1", "E2", "E3", "EC", "F1", "F2", "G1", "I1", "I2", "N1", "P1",
                "SC0", "SC1", "SC2", "SC3", "SP1", "SP2", "T1"],
 "book_width": ["tight <1.02", "medium 1.02-1.05"],
 "ev_margin": 0.02,
 "ev_cost": "a tick off fair at each end at the early fair price",
 "commission": 0.02,
 "post": "one tick inside the spread on the order's side, else join the best price on the other side",
 "fill": "a later snapshot's best price on the other side at or through the order's price",
 "cancel_before_start_min": 90.0,
 "exit": "cross at the last two-sided book within 30 min of the off",
 "staking": {"bank_gbp": 1000.0, "window_budget": 0.25, "max_per_trade": 0.10,
             "capital": "a back's stake, a lay's liability", "weights": "expected value"},
 "headline": "return on the capital staked at the fill and the recorded close, 95% interval by day",
 "gate": {"min_bets": 300, "min_days": 90, "cap_days": 365, "roi_lo95_above": 0, "clv_mean_above": 0,
          "both_halves_above": 0},
 "start": "first Friday collection on or after the first recorded football book, not before written_utc",
 "written_utc": "2026-11-04T15:10Z"
}
```
