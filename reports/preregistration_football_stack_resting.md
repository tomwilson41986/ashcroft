# Pre-registration: the stacked move rule with resting orders, paper-traded at the recorded Betfair books

racing² football research. Written 2026-11-03T09:10Z by the modelling agent, before any forward book was read.
Companion to `reports/preregistration_football_stack.md` (the crossing rule). Code:
`research/open_close/football_stack_resting.py`, tests `tests/test_football_stack_resting.py`. The scorer reads the
JSON block at the end of this file and records this file's SHA-256 beside every result.

## Why a resting variant

The crossing rule's development result turns on the cost of the spread. At 0.04 on 2024/25 it shows +1.58% a tick
off fair and +3.82% at a symmetric book (`reports/football_runs/stack_threshold_sweep_2024.json`). The early lay
spread was never recorded, and roadmap phase 2 sets the kill line for crossing at a median takeable spread over 2
ticks.

A resting order does not pay the spread. In exchange it fills only when the price comes to it, which for a back
means the price has drifted against the forecast for a moment. This trial measures whether the saved spread is worth
more than that adverse selection.

**There is no development evidence for this variant.** football-data has no books between the early price and the
close, so fills cannot be simulated on history. Every choice below is either taken from the crossing rule or fixed
by design, not by looking.

## The rule

### Taken unchanged from the crossing rule

- **Signals:** the same frozen models (`models/football/stack_v1`, SHA-256 below, checked before scoring).
- **Threshold:** 0.04.
- **Markets and leagues:** Betfair MATCH_ODDS, home and away; the 22 divisions.
- **Book width:** the early book at a best-back sum of 1.05 or under.
- **Early moment:** the first recorded snapshot within 60 min after football-data's collection moment.
- **Commission:** 2% on each market's net.
- **Stake:** GBP10 notional.

### Entry: post at the early book

- **A back:** one tick above the best back if that is still below the best lay. Otherwise at the best lay, joining
  the backers queued there.
- **A lay:** one tick below the best lay if that is still above the best back. Otherwise at the best back.

### Fill: only when the recorded book proves it

**Back.** A resting back at P is filled at the first later snapshot, before the deadline, whose best back is at P or
longer. A lay offer at P or above can only rest once no back at P or below is left, ours included.

**Lay.** A resting lay at P is filled when the best lay is at P or shorter.

**Missed moves.** The recorder polls every 30 min the evening before, every 15 min on the day, and every 10 min for
50 min at the Friday and Tuesday collections. A move through P that reverses between two snapshots is missed. The
rule can therefore only undercount fills, never invent one.

### Deadline

Unfilled orders are cancelled 90 minutes before the market's start. Team sheets come out about 60 minutes before
kick-off, and a resting order left through them is a free option for whoever reads them first.

### Exit

As the crossing rule: crossed at the last two-sided book within 30 minutes of the off, after the fill. Greened.

### Headline

The return per stake unit after commission, with its **95%** bootstrap interval by day (2,000 draws). It is 95%
rather than 90% because two registered rules now read the same forward books. Bonferroni at two holds this pair's
chance of a false pass at the 5% one-sided level that each 90% interval gave alone.

## Reported beside the headline (not the decision)

- **Fill rate:** of the signals that reached a takeable recorded book, the share filled.
- **Adverse selection:** the right-way share (close mid against early mid, in the signal's direction) of filled
  signals against unfilled ones.
- **Time to fill:** the median minutes from posting to fill.
- **Close size:** the share of closes where the size shown would not cover the hedge.

## Start, window and gate

- **Start:** as the crossing rule, the first Friday collection on or after the first recorded football book, and not
  before this file was written.
- **Window:** until the later of the 300th **filled** trade and 90 days, capped at 365 days. Until then the gate reads
  "running".

**Gate.** The trial passes only if all of these hold:

- at least 300 fills;
- at least 90 days;
- the 95% lower bound of the return above zero;
- the mean CLV at the fill price above zero;
- both halves of the window above zero.

A pass licenses a decision by the owner on small real stakes. A fail retires the variant: the offset, the deadline and
the threshold are not re-tuned on the trial's books.

**Relation to the crossing rule.** The two are scored separately and gated separately. If both pass, the one with the
higher lower bound goes forward. Neither result changes the other's registration.

## The expectation stated in advance

**The rule is expected to fail.**

- A resting back fills when the price drifts toward it, which is when the forecast is going wrong. The fills' right-way
  share is expected to sit below the unfilled signals' and below the crossing rule's 65.9% on 2024/25.
- The spread saved at entry is paid again at the exit, which still crosses.
- **Fill rate:** expected to be well under half of the signals. Beyond 25% to 50% no number is offered, because no
  data exists to set one.

## Power

At a fill rate of 25% to 50%, the crossing rule's 416 takeable signals a season would give about 100 to 210 fills. 300
fills would take one and a half to three seasons, beyond the 365-day cap.

**This trial will most likely end at the cap, short of 300 fills, and read as a fail on the bets check.** That is
recorded now so it is not read later as bad luck. Its real value is the fill rate and the adverse-selection read,
which decide whether an execution model is worth building (roadmap 4.5).

```json
{
 "model": "stack_ridge_lgbm_team_all_frozen",
 "variant": "resting entry, crossing exit",
 "arm": "team_all",
 "models_dir": "models/football/stack_v1",
 "model_sha256": {
  "spec.json": "22e1073fb826d174406ac86ee541551095bdfc8d7b1874410fd7bec4351c7df0",
  "gbm_h.txt": "3b430d931eecaae7c071dce6b9a6e36431a16e592e492ec79d1a9b69703a97cf",
  "gbm_a.txt": "86072d0ccaa24f754b701ae03844b3169e660c50ead92dac801f56b3f828e9aa"
 },
 "market_types": ["MATCH_ODDS"],
 "selections": ["H", "A"],
 "leagues_fd": ["B1", "D1", "D2", "E0", "E1", "E2", "E3", "EC", "F1", "F2", "G1", "I1", "I2", "N1", "P1",
                "SC0", "SC1", "SC2", "SC3", "SP1", "SP2", "T1"],
 "book_width": ["tight <1.02", "medium 1.02-1.05"],
 "threshold": 0.04,
 "commission": 0.02,
 "notional_gbp": 10.0,
 "post": "one tick inside the spread on the order's side, else join the best price on the other side",
 "fill": "a later snapshot's best price on the other side at or through the order's price",
 "cancel_before_start_min": 90.0,
 "exit": "cross at the last two-sided book within 30 min of the off",
 "headline": "return at the fill and the recorded close, 95% interval by day",
 "gate": {"min_bets": 300, "min_days": 90, "cap_days": 365, "roi_lo95_above": 0, "clv_mean_above": 0,
          "both_halves_above": 0},
 "start": "first Friday collection on or after the first recorded football book, not before written_utc",
 "companion": "reports/preregistration_football_stack.md",
 "written_utc": "2026-11-03T09:10Z"
}
```
