# Pre-registration: the Australian early lay, closed at the BSP (4 Nov 2026)

Written and committed before the test is run. The owner's ask (4 Nov 2026): is there a strategy in what gate A found?
Gate A (`au/README.md`, ledger `au-gate-a-1104`) found that an Australian price an hour before the off sits under the
BSP on average (a back there closed at the SP: -2.31% CLV), and that the Betfair-only model orders which runners drift.
Every figure that suggested this rule came from pooled averages read after the fact. This document fixes one rule and
one gate before the rule is scored, so the result cannot be fitted to the data.

## The rule (primary)

- **Universe:** Australian thoroughbred WIN markets on Betfair, Jan 2024 to the bundle's cut (early Oct 2026): every
  runner that ran (WINNER or LOSER), with a BSP and with the Betfair-only model's out-of-sample price
  (`s3://ashcroft/sources/au/model_report_oos.parquet`, each year priced by a model fitted on the years before it).
- **When:** 60 minutes before the scheduled off. The price is the last traded price at that mark (`ltp_t_60`,
  `s3://ashcroft/sources/betfair_historic/horse_racing_win_<year>.parquet`). A runner with no trade by then is skipped.
- **Which:** lay where the price is under the model's expected BSP, `L x p_model < 1` (the model's chance says the
  price is too short), and `1.5 <= L <= 21`.
- **Stake and close:** lay 1 unit at L; back `L / BSP` units at the BSP, so the position is level whatever wins. Each
  bet's profit is `1 - L / BSP`.
- **Commission:** 2% of each market's net profit when positive (the owner's rate, TRADING.md).
- **Measure:** profit per unit laid, after commission, over all bets: the sum of the markets' net profits over the sum of
  units laid. 90% interval by bootstrap over markets (1,000 resamples).

## The gate (all must hold)

1. Primary profit per unit laid after commission above zero, with the 90% interval's lower end above zero.
2. Above zero in each calendar year: 2024, 2025 and 2026 (to the cut), point estimates.
3. At least 5,000 bets.

If any fails, the rule fails, and no variant below can pass it in its place.

## Reported beside it, not part of the gate

- Every runner in the price range, no model filter (does the model add anything to the market-only drift).
- The same rule at T-15.
- Prices over 21.
- Settled by the result instead of closed at the BSP: lay 1 unit, +1 if it loses, -(L-1) if it wins, the same commission.
- By price band.

## What a pass would and would not show

A pass says the traded prices of 2024-2026 left a margin for an early layer who closed at the BSP. It does not say a lay
offer would have been matched: last traded prices are not a book, and the data hold no volume. A pass leads to gate B,
recording the Australian books from their opening for 2-4 weeks, then a paper trial. No money is staked on this test.

## Addendum, 4 Nov 2026: the staleness check (written before it runs)

The primary rule passed its gate (au-lay-test run 41823377954: +2.42% a unit laid after commission). Three signs say
the margin may sit in prices no one could lay: laying every runner made more than the model's filter, the margin is
gone by T-15, and prices over 21 show +15%. This check asks whether the margin survives on fresh prices.

- **Measure:** for each runner, how long before the T-60 mark its last traded price last changed (the BASIC files
  record a price when it changes, so this is the longest the price can have stood; a trade at the same price later is
  not seen).
- **Test:** the primary rule, unchanged, restricted to runners whose T-60 price changed within the 5 minutes before the
  mark. Reported beside it: within 15 minutes, 15-60 minutes, over 60 minutes.
- **Reading, fixed now:** if the fresh-price subset (within 5 minutes) makes above zero after commission with its 90%
  interval above zero, the stale-price explanation is weakened and gate B is worth running. If it is at or below zero,
  the margin was in stale prices, and the strategy is dropped without gate B.
