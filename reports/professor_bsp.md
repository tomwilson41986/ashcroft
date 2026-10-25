# The Professor at bookmaker prices: can the tips show a profit at the Betfair SP? (25 Oct 2026)

The owner decided (25 Oct) on **bookmaker prices only** (`professor/config.json require_books: true`: a horse is
advised only at the price three or more major bookmakers offer) and asked that the tips **show a profit at BFSP**.
Two research queries answer whether the rule can: `research/queries/done/professor_bsp_1025.py` (run 40178868298) and
`professor_bsp_1025b.py` (run 40179026879). Ledger `professor-bsp-1025`.

## The bookmakers' margin is the constraint

The dry runs measured the bookmakers' book on our races at 114-124% (reports/professor_record.md) against about
100-102% for the Betfair SP. The backtest of 23 Oct stood the bookmakers' price in with haircuts of 5-15% on the
exchange's price; the queries here divide the exchange's morning price by the real margin (1.10-1.22).

## The rule as set (10% over the expected close, ranked by win probability, 5 a day) at the real margin

| Sample, margin 1.14-1.18 | Tips a day | Strike | Avg bookmaker price | ROI at the BSP | ROI at the bookmakers |
|---|---|---|---|---|---|
| Aug-Sep history | 3.5-3.8 | 8-10% | 11-12 | -11% to -6% | -27% to -17% |
| Oct history | 0.9-1.8 | 8-10% | 23-24 | -49% to -42% | -38% to -21% |
| Live, 1-22 Oct | 1.2-1.9 | 8-10% | 25-29 | -37% to -35% | -22% to -21% |

At bookmaker prices the 10% bar admits only the horses whose exchange price was far above the model's expected close:
long prices that, on the evidence, drift. The bookmakers' price is then 11-21% **under** the BSP on average, the strike
rate falls to about 10% and both returns are negative. The dry run of 23 Oct showed the same in miniature (three long
prices advised at 17-51, closing at 26-110).

## Does the model's choice beat the BSP at all?

Every horse the live rule backs (the candidates, 50/1 or under), held to the result at the BSP after 2%:

| Population | Horses | Strike | ROI at the BSP | Winners against the BSP's expectation |
|---|---|---|---|---|
| History, Aug-Oct | 8,548 | 8.6% | -4.6% (90% -12.3 to +3.4) | 0.989 |
| Live, 1-22 Oct | 1,819 | 13.2% | +6.3% (90% -10.0 to +23.4) | 1.014 |

As TRADING.md found, the model forecasts the BSP, not the result beyond it: its horses win as the BSP says. The one
exception in the history is the shortest: by the closing model's win probability, 0.2-0.3 won 1.03 of expectation
(+3.0% at the BSP, 563 horses) and over 0.3 won 1.10 (+7.6%, 226 horses).

## The short-priced rule, tested with the decision set first

The hypothesis from that table: a bookmaker-only rule on the short-priced horses (win probability at least 0.15-0.30,
the bookmakers' price at least 90-110% of the expected BSP) shows a profit at the BSP. Set before the run: a cell is
viable only if its ROI at the BSP **and** at the bookmakers' price are above zero on Aug-Sep, on Oct and on the live
record, with at least one tip a day on each. **No cell of the 40 passed.**

The nearest, all failing on one sample or on volume:

| Rule (margin 1.14) | Aug-Sep: tips a day, BSP ROI, books ROI | Oct | Live |
|---|---|---|---|
| p_win >= 0.25, books >= 95% of the expected BSP | 2.3, +10.1%, +2.4% | 2.3, +22.1%, +14.7% | 1.4, **-3.3%**, +11.2% |
| p_win >= 0.20, books >= 95% of the expected BSP | 4.2, +11.4%, +2.2% | 3.4, **-6.8%**, **-12.2%** | 2.4, +6.3%, +15.6% |
| p_win >= 0.30, books >= 95% (margin 1.18) | **0.5**, +5.8%, +4.5% | 0.6, +47.2%, +34.5% | 0.3, +60.7%, +73.7% |

The short-priced horses do win more than the BSP says in most of these cells (1.05-1.33 of expectation, strike
25-45%), which is what a profit at BFSP needs; but on 30-130 tips a sample the intervals span zero and the rule fails
its own test on one sample in every cell that gives a tip a day.

## What it means

- **The tips cannot be designed to show a profit at BFSP on this evidence.** A profit at the BSP needs horses that win
  more often than the BSP says; the model's horses do not, except perhaps the shortest, and that is not proven.
- **The rule as set is wrong for bookmaker prices.** At the real margin it picks drifting long shots and loses at both
  prices. The short-priced variant (win probability at least 0.25, the bookmakers' price at least 95% of the expected
  BSP) fits the owner's "win frequently" better (strike 30-40%) and is the one the samples favour, at 1-2 tips a day;
  it is unproven, and a change to the rule is the owner's call.
- **The record shows the profit at BFSP either way**: every tip is settled at the advised price and at the BSP
  (`python -m professor --settle`), and the running total is kept with it.
