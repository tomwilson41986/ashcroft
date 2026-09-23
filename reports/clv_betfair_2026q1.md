# Closing-line value on Betfair's own prices, 1 Jan – 21 Mar 2026

The first test of the early-price edge on real exchange prices: the walk-forward BSP
forecast of the deployed recipe (`logit_norm_prob`, the v4 out-of-sample run) against
Betfair's morning volume-weighted price (`morningwap`), both scored against the BSP. The
prices are Betfair's historic files for UK and Irish win markets, 31 Dec 2025 – 21 Sep 2026,
loaded 23 Sep 2026 (86,121 runners; 99.5% linked to `race_results`; Betfair's BSP equals the
BSP horseracebase recorded on all 85,333 linked runners). The out-of-sample predictions end
on 21 Mar, so the test covers 19,549 runners in 2,233 races.

## The rule, fixed before looking

Back where the model's price is at least 22% shorter than the morning price
(`ln(morningwap / predicted_bfsp) >= 0.2`), with at least £100 matched in the morning. Take
the morning price and close at BSP. Net CLV per unit = `morningwap / bsp − 1`, less 5%
commission only when positive. Pass: the 90% race-bootstrap interval is above zero.

## Result: it passes

| Selection | Bets | Net CLV (90% CI) |
|---|---|---|
| Every runner with £100+ matched in the morning | 15,331 | −1.41% (−1.89 to −0.93) |
| Model ≥ 11% shorter than the morning price | 6,046 | +3.42% (+2.31 to +4.58) |
| **Model ≥ 22% shorter (the rule)** | **4,535** | **+4.83% (+3.54 to +6.09)** |
| Model ≥ 35% shorter | 3,315 | +6.76% (+5.13 to +8.46) |
| Model ≥ 22% *longer* | 4,393 | −7.05% (−7.89 to −6.21) |

Adding the model to the morning price improves the BSP forecast out of sample, each month
scored with coefficients fitted on the months before it: RMSE 0.469 → 0.444 (February), 0.503 →
0.470 (March); January has too little before it to fit on. The fitted weight on the model, beside
a curved morning-price term, is 0.33 (95% CI 0.31–0.35). R1 (the `log_bfsp` recipe) gives the
same picture: +5.05% (+3.63 to +6.56) on its 3,754 bets under the rule, and +9.70% on outsiders
against −0.68% for every runner in that band.

## The checks

* **Not the same-day leak.** Training features for later races include earlier results from
  the same day (QA review C2), which could let the model "predict" moves the market makes
  after those results. The edge is larger where that cannot happen: +8.62% (+4.46 to +12.85)
  in the first race at each meeting, +8.13% (+3.37 to +13.16) in the first three races of the day, +4.75%
  (+3.32 to +6.32) from race ten onwards.
* **Not the morning price's own bias.** Fitting that bias (a quadratic in the log price) never
  implies a 22% shortening, so it selects nothing on its own.
* **It lives in the outsiders.** Morning price 16+: +9.52% (+7.19 to +12.03) on 2,151 bets,
  against +1.02% (−0.75 to +2.70) for every runner in that band and −12.81% (−16.16 to −9.44)
  where the model says *longer*. Below 16 the rule is within noise (+0.3% to +0.9%).
* **Every month.** January +2.98% (+1.09 to +5.00), February +3.78% (+1.58 to +6.22), March
  +8.97% (+6.13 to +11.57).

## What it is, and what it is not

* **A trading edge, not a value bet.** The model picks horses the market goes on to back, and
  those close too short: held to settlement the rule returned −4.2% (−15.2 to +7.9) at the morning
  price and −10.3% (−21.8 to +2.8) at BSP, against −2.9% for every runner at BSP. The gain comes
  from taking the early price and trading out before the off.
* **Small stakes.** The median runner the rule selects had £491 matched in the whole morning
  (interquartile range £243–£1,035).
* **Priced at an average.** `morningwap` is the volume-weighted average of the morning's
  matched bets. The rule both selects on it and fills at it; real fills through the morning
  will be somewhat worse. Trading out exactly at BSP is not possible either — the lay has to
  be sized before BSP is known — so a real green-up happens at the last prices before the off.
* **Eleven weeks.** The retrain's evaluation (run 27, through 17 Sep) will extend the overlap to
  nearly nine months; rerun this then.
* **Offline, not live.** These forecasts come from result rows. The live card currently reaches
  the model with part of its features missing (QA review C1), so live forecasts are worse than
  these until that is fixed.

Reproduce: `python scripts/clv_betfair.py --predictions <oos_predictions.csv> --db horse_racing.db`
(or `--extract` with the `bf-import` artifact's `extract.csv.gz`). The v4 predictions are the
six-variant head-to-head's `oos_predictions_v4_logit.csv`.
