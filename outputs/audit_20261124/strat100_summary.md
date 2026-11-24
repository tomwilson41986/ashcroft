# racing² - 100-strategy test, 24 Nov 2026 (ledger strat100-1124)

Registered before the run (ledger `strat100-1124`, 100 strategies, conventions K1-K10). Research only: nothing placed
a bet. The locked holdout (from 28 Mar 2026) was refused by the loaders. Every figure is from the three run logs:

| Data set | Run | Strategies run (incl. benchmarks) | Verdicts |
|---|---|---|---|
| a_model (model OOS predictions, SELECT 28 Dec 2024-30 Sep 2025, CONFIRM 1 Oct 2025-27 Mar 2026) | research-query 45453565990 | 18 | FAIL 15, benchmark (FAIL) 3 |
| a_market (Betfair price files, SELECT 2018-2023, CONFIRM 2024-27 Mar 2026) | research-query 45453503335 | 36 | FAIL 26, not read: too few 6, benchmark (FAIL) 4 |
| b_live (live record, SELECT 1-31 Oct 2026, CONFIRM 1-23 Nov 2026) | research-query 45453409678 | 56 | FAIL 45, not read: too few 6, benchmark (FAIL) 5 |

**No strategy passed.** Every strategy with a positive CONFIRM interval failed on the deflated Sharpe alone (n_trials
486; SR* 2.93 a_model, 0.92 a_market, 2.05 b_live, the spread taken from each builder's own Sharpes). No paired
strategy made more pounds a day than its benchmark with a deflated Sharpe above 0.95.

## The live rule and its variants (b_live; return per GBP, closed at the BSP after non-runner cuts)

| Strategy | CONFIRM 1-23 Nov | 90% interval | SELECT Oct | Net GBP CONFIRM | Pair vs benchmark (GBP a day, CONFIRM) |
|---|---|---|---|---|---|
| S001 the live rule replayed (benchmark) | +4.69% | +3.37 to +6.09 | +9.13% | +2,184.50 | - |
| BASE-B the live trader's real fills | +3.41% | +2.20 to +4.70 | +3.13% | +2,158.59 | - |
| S006 price at least 50% over our forecast | +10.87% | +8.38 to +13.04 | +18.15% | +1,675.07 | -23.16 (-43.69 to -3.16) |
| S003 bar 8% | +8.43% | +5.93 to +10.96 | +15.30% | +1,364.72 | -37.26 (-59.10 to -16.32) |
| S009 first hour only | +7.66% | +4.90 to +10.44 | +11.66% | +1,519.09 | -30.25 |
| S013 first backs only, no top-ups | +7.57% | +5.88 to +9.26 | +11.58% | +1,144.95 | -47.25 (-68.61 to -27.52) |
| S015 top-up bar 7% (real fills) | +7.47% | +5.41 to +9.56 | +7.94% | +1,891.43 | -11.62 (-33.30 to +8.33) |
| S014 top-ups only within 30 min of the first back | +5.37% | +3.90 to +6.97 | +9.79% | +1,997.39 | -8.51 (-18.34 to +1.09) |
| S019 target scaled by hours to the off (800/400/200) | +4.76% | +3.35 to +6.25 | +9.50% | +2,356.18 | +7.80 (+0.80 to +15.39), pair DSR 0.0 |
| S018 stake floor GBP50 at 21+ | +4.68% | +3.29 to +6.13 | +9.47% | +2,262.01 | +3.52 (-2.38 to +9.79) |
| S002 bar 0% | +2.91% | +1.76 to +4.11 | +6.64% | +1,850.49 | -15.18 |
| S017 no ladder | +3.44% | +2.33 to +4.60 | +8.60% | +1,420.49 | -34.73 |
| S025 the live backs held to the result | +1.97% | -7.82 to +12.20 | +4.65% | +1,245.14 | -39.72 |
| S027 the live backs laid in running at half the SP odds | -30.93% | -36.03 to -26.08 | -29.92% | -18,645.85 | -948.09 |

Reading: a tighter rule (a higher bar, a large gap, first backs only, a 7% bar on top-ups) earns 2-6 points more a
pound, but on fewer pounds, so it makes less money a day than the rule as it stands. Only scaling the target by the
hours to the off made more pounds a day (+GBP7.80, interval above zero), and it fails the deflated Sharpe.

## Other families (b_live)
Lay books: S037 +1.31% (+0.54 to +2.11) on liability, S041 +1.94%, S039 +1.50%, S040 +1.46%; none passes. Place and
other-place: none above zero at CONFIRM except S049 +0.34% (-1.40 to +2.16). Momentum, model-free controls, signal
votes and single looks (S012 2h before the off -3.12%) lose.

## Price files and model predictions (a_market, a_model; no non-runner cuts on the files, cost about 4.66 points live)
- S053 the rule at the morning WAP, 3% bar: CONFIRM +8.74% (+8.24 to +9.26). Every variant made fewer pounds a day.
- S060 rule in the steamer model's top 30%: +11.44% (+10.58 to +12.30) on 3,919 bets, pair -160 GBP a day.
- S067 morning false favourite lay closed at the BSP: +3.74% (+3.05 to +4.43); held to the result +1.30% (-2.22 to +4.77).
- S082 back headgear removed at the BSP: CONFIRM +26.77% (+14.01 to +40.53) but SELECT +1.33% (-4.55 to +7.41); a
  likely data artefact, not an edge.
- S077 lay the steamer model's bottom tenth in the morning: +3.20% (+2.82 to +3.59), SELECT +4.66%.
- In-running back-to-lay and lay-to-back: all negative (upper bounds).

Caveat: the runs used dsr_check before the 24 Nov ME-02 fix, which only makes the check stricter; no verdict can change
to a pass.
