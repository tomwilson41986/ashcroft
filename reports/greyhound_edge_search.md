# Greyhound edge search: the ledger

The owner, 4 Nov 2026: "keep working in loops until you define a true edge and profitable strategy"; "fundamental is
always just to get the most accurate closing price, optimise towards this"; "only come back when you have 5 profitable
strategies".

**What counts.** A rule fixed before its test (chosen on a development period, read on a later holdout it never saw),
filled as it really would be (at the Betfair SP or a limit-on-close; resting orders only through the queue-aware
replay, `research/greyhound/queue_fill.py`), with its holdout return's 90% lower bound (a bootstrap over days) above
zero and its development return above zero. Every rule tried is counted here: about one in twenty clears a one-sided
90% bound by chance.

| # | Family | Data, test | Rules tried | Passing | Verdict | Source |
|---|---|---|---|---|---|---|
| 1 | Place SP against Harville on the win price known before the off (the morning WAP) | price files, GB, 2021-26 walk-forward | 16 (2% comm.) | 1 at +3.60% (0.18 to 7.28), 1 in 16 = chance | no edge | research-query 41883691562 |
| 1b | Same on the win SP (look-ahead upper bound) | price files, 2023-26 | 16 | most | not a rule: chose on the place SP itself (look-ahead on both sides) | research-query 41883691562, 41884997778 |
| 2 | Place SP against Harville on the win market's projected SP / mid, 120-10 s before the off, settled at the place SP | recorded books 6 Oct - 3 Nov (3,530 races; the recorder's, ~40 s stale near the off) | 16 | 0 | no edge on stale books; to retest on the live-key books (5 s, projected SP) as days accrue | research-query 41884997778 |
| 3 | Limit-on-close against the morning WAP (back drifts, lay steamers), win and place | price files, dev 2016-21, holdout 2022-26 | 72 | 0 | no edge | research-query 41888602917 |
| 4 | Limit-on-close against our fair price (win and place, raw and tempered), log-BSP and softmax models | fast-loop 41888253783 (fitted on 2021-23, 784,294 runners priced 2024 - 3 Nov 2026); dev 2024, holdout 2025-26 | 160 | 0 | no edge: every win rule's holdout return negative; the SP is efficient against our pre-race price | research-query 41891640383 |
| 5 | The SP's bias by segment (track, trap, field, session) as limit-on-close thresholds | price files, dev 2016-21, holdout 2022-26 as one portfolio | pending | | | research-query 41889297892 |
| 6 | Place against win from Betfair's historic data at 1 and 15 minutes out (last traded prices), bet at the place SP; GB and AU | BASIC Jan-Sep 2026; dev Jan-Apr, holdout May-Sep | pending | | | |

## The closing price (the objective)

RMSE of log(BSP), recorder books re-timed, 14 Oct - 2 Nov (closing-price run 41884995327): first traded 0.319 (mid
0.635), 5 min 0.368 (0.709), 3 min 0.225 (0.337), 1 min 0.144 (0.158). The recorder's greyhound books carry no
projected SP; the live key's do (from 2 Nov), to be scored as their days accrue.
