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
* **Eleven weeks here; the out-of-sample test is below.** Run 27 timed out; run 28 produced the
  forecasts for April–September.
* **Offline, not live.** These forecasts come from result rows. The live card currently reaches
  the model with part of its features missing (QA review C1), so live forecasts are worse than
  these until that is fixed.

Reproduce: `python scripts/clv_betfair.py --predictions <oos_predictions.csv> --db horse_racing.db`
(or `--extract` with the `bf-import` artifact's `extract.csv.gz`). The v4 predictions are the
six-variant head-to-head's `oos_predictions_v4_logit.csv`.

## Out of sample: 1 April – 6 September 2026

The test was pre-registered before its forecasts existed: `reports/preregistration_clv_oos.md` on the research branch, #74.
- **Rule:** unchanged.
- **Window:** it never set the rule.
- **Forecasts:** run 28, `train-bfsp.yml` `evaluate_only`.
  - Walk-forward monthly folds from 2025-12-11 to 2026-09-06; the last full fold ends 6 September.
  - Built on the **repaired** features of #74: day-lagged priors, EPF, the lengths parser and the debut flag.
  - The rule was fixed on the forecaster *before* those repairs. That difference is stated, not hidden.

| Selection | Bets | Net CLV (90% CI) |
|---|---|---|
| Every runner with £100+ matched in the morning | 43,091 | −2.05% (−2.28 to −1.82) |
| Model ≥ 11% shorter | 16,453 | +2.47% (+1.95 to +3.02) |
| **Model ≥ 22% shorter (the rule)** | **12,389** | **+3.54% (+2.86 to +4.19)** |
| Model ≥ 35% shorter | 9,009 | +5.21% (+4.42 to +6.05) |
| Model ≥ 22% *longer* | 12,971 | −7.75% (−8.20 to −7.29) |

**It passes:** the interval is above zero.

- **By month:** April +0.96% (−0.42 to +2.46), May +3.60%, June +4.00%, July +5.68%, August +3.71%, September (to the 6th, 492 bets) +2.71% (−1.10 to +6.83).
- **Adding the model to the morning price** lowers the BSP forecast's RMSE in every month, e.g. 0.459 → 0.430 in May. Its weight is 0.33 (0.31–0.34).
- **By morning price band,** the rule is ahead of every runner in every band:
  - 1–4: +1.4%;
  - 4–8: +2.5%;
  - 8–16: +1.3%;
  - 16+: +5.7% (every runner in the band −2.0%).
- **Held to settlement** it loses: −2.3% at the morning price and −6.8% at BSP, against −3.8% for every runner at BSP. It is a trading edge, as before.

From the same forecasts, January – 21 March gives +6.10% (+4.70 to +7.66) on 4,348 bets. The v4 forecaster had given +4.83% on the same weeks.

### The caveat that has to be settled before any money is risked

The same-day check was pre-registered to be reported whatever it showed. The edge has **moved to the later races**:

| Races | Net CLV (90% CI) |
|---|---|
| First race at each meeting | +0.55% (−1.24 to +2.45), n=1,764 |
| First three races of the day | −0.05% (−2.48 to +2.51), n=1,106 |
| Race ten onwards | +4.96% (+4.18 to +5.75), n=9,097 |

By off time (post hoc):

| Off | 13:00 | 14:00 | 15:00 | 16:00 | 17:00 | 18:00 | 19:00 | 20:00 |
|---|---|---|---|---|---|---|---|---|
| Net CLV | −0.0% | +0.9% | +3.3% | +2.9% | +6.6% | +4.1% | +4.0% | +8.1% |

In January–March the first races carried the largest edge (+8%). Two readings fit the new pattern.

1. **The morning price gets staler the longer it sits before the off.** A fundamentals forecast then has more to correct in an evening race. This reading is legitimate and tradeable.
2. **The backtest forecast knows race-day facts a morning bettor does not.** It is built from result rows, so it sees:
   - the final field after late non-runners;
   - the going on the day;
   - the jockey who actually rode.

   These facts pile up in later races. This reading is not tradeable.

Late non-runners are part of the story, but not all of it:
- In races where the morning book of the runners who ran is short (< 0.98, 462 bets), the rule makes +9.5% to +27%.
- Excluding those races it still makes **+3.25% (+2.60 to +3.93)** on 11,927 bets. The rise with off time is the same there.

The going and jockey changes are not yet separated.

**The deciding test** is the live 06:00 record (`s3://…/predictions/<date>.csv`), written from the morning card before the morning window.
- A positive CLV there settles reading 1.
- A null is ambiguous: those forecasts came from a model on history frozen at 22 March and are missing part of the live card (QA C1). The retrained model has to run live first.

### The live 06:00 record: null, and it cannot decide

The query is `research/queries/done/live_record_clv.py` on #74. It applies the same rule to the forecasts the 06:00 job actually wrote.
- The bucket holds files only from 14 July: 48 days, 16,378 runners to 6 September.
- **The rule makes −3.20% (−4.26 to −2.13)** on 4,772 trades, against −2.99% for every runner.
- The live forecast does not separate the market's later movers at all:
  - forecast ≥ 11%, 22% or 35% shorter: −3.15%, −3.20%, −3.11%;
  - forecast ≥ 22% longer: −2.98%.

This measures a broken pipeline, not the trade:
- **History:** the live model ran on history frozen at 22 March, so its form was four to six months out of date.
- **Card features (QA C1):** the live card reaches it without about a fifth of its features: distance, race type, surface, sex, sire, dam's sire, career runs, claims, the race's OR spread and track direction. Each missing category is scored as the first entry in the vocabulary.

So the backtest pass stands, **unconfirmed at bet time**. The deciding test is forward:
1. fix the live feature path (C1);
2. run the retrained model on current history;
3. measure this rule on the 06:00 forecasts as the weeks come in.

At about 80 trades a day, three weeks would separate +3.5% from zero.
