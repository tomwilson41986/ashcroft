# Pre-registration: the early-price trade, forward on the live path

Written 23 Sep 2026, before the fixed live path has served a single card. No forecast
this test will score exists yet.

## Why a forward test

The rule passed out of sample on April–September 2026 (`reports/preregistration_clv_oos.md`):
+3.54% net CLV (90% +2.86% to +4.19%) on 12,389 trades. But the edge sits in later races,
and two readings fit that:

1. **The morning price is staler before later races.** That edge would be tradeable.
2. **The backtest forecast knows race-day facts.** It was built from result rows, which
   carry the going as it was at the off, the jockey who actually rode and only the horses
   that ran. That edge would not be tradeable.

Only forecasts made in the morning can tell the two apart. The 06:00 record from July to
September was made by a broken pipeline:
- history frozen at 22 March;
- a card missing a fifth of its features (QA C1).

So its null (−3.20%) decides nothing.

## The rule, unchanged from #73 (`scripts/clv_betfair.py`, defaults)

- **Forecast:** the 06:00 job's `predicted_bfsp`, as written to `s3://…/predictions/<date>.csv`.
- **Back** a runner at Betfair's morning volume-weighted price when
  `ln(morningwap / forecast) ≥ 0.2` and at least £100 was matched in the morning.
- **Close** at BSP. Net CLV per unit is `morningwap / BSP − 1`, less 5% commission when
  positive.
- The join from forecast to price and result is the one in
  `research/queries/done/live_record_clv.py`: track, 24-hour off time and horse name,
  each normalised. Only its date window changes.

## The window

- **It starts on the first 06:00 run after all of the following are deployed.** The date is
  written to `reports/research_ledger.jsonl` on the day it happens.
  - the card fill (`model/card_enrich.py`);
  - a model retrained on the repaired features;
  - history that is current to the day before. The nightly ingestion fix keeps it current.
- **It ends after 28 race days**, or when 1,500 rule trades are reached, whichever is
  later, capped at 8 weeks.
  - Power: the out-of-sample interval implies a standard error of about 0.4% at 12,389
    trades. At 1,500 trades that is about 1.1%, which detects a true edge of +3.5% with
    90% one-sided power. An edge of half that would probably be missed, and the report
    will say so.
- **Days are excluded only if the 06:00 file is missing or was written after 09:00 UK
  time.** The reason is logged for every excluded day.

## Passes if

- the 90% race-bootstrap interval of the mean net CLV is above zero; and
- the mean is positive in both halves of the window, split by date.

## Reported, whatever the outcome

- The headline, its interval and the halves.
- The rule by race of the day (first three; race ten onwards) and by off hour. This is the
  split that separates the two readings.
- **The same days scored from the backtest's forecast**, rebuilt from result rows by
  `evaluate_oos.py` (walk-forward, same recipe). This is the direct test of reading 2:
  - both forecasts over the same runners with the same prices;
  - if the backtest's forecast gains and the live one does not, the gap is race-day facts.
- **What changed between 06:00 and the off**, from the morning card, which the 06:00 job
  now keeps (`s3://…/racecards/<date>_<HHMM>.csv`):
  - how often the going changed;
  - how often a rider was replaced;
  - how often the field lost a runner.
  - The rule's edge is reported with and without the races where any of these changed.
- **The trade at the price the job actually saw**, the best back price in the snapshot
  taken at prediction time. The morning WAP averages trades that may precede the forecast,
  so this second figure is the executable one. It is secondary, not the criterion.
- Races whose morning book, over the horses that ran, is short (late non-runners, whose
  reduction factors the CLV formula ignores), reported separately as before.

## What each outcome means

- **Pass:** the edge is available to a forecast made at 06:00. The next step is a
  staking trial on the executable price, with its own pre-registration.
- **Fail, with the backtest's forecast passing on the same days:** reading 2. The
  backtest's edge came from race-day facts, and the trade is abandoned.
- **Both fail:** the edge has gone, or was never there.

## Amendment, 24 Sep 2026, before any forward card has been served

**Secondary endpoint: the rank-1 subset.** These are the rule's trades on each race's rank-1 runner, the model's highest win probability (lowest forecast price). It is scored with the same statistics and bootstrap as the primary, but it is not the criterion. It is the direct test of the original goal, a profitable rank-1 selection, taken as a trade rather than held.

For reference only, from windows already scored:
- **January–March** (the window that set the rule): +1.01% (90% −0.95% to +3.01%) on 579 trades.
- **April–September** (post hoc): +3.16% (+2.05% to +4.29%) on 1,661 trades.
- **Every rank-1 held to settlement at the morning price** loses in both windows: −7.7% and −2.8%.

## What the test needs that is not yet in place (24 Sep)

Neither item changes the rule or the criteria. Without them the test cannot be scored.
- **Betfair's historic price files for the forward days.** These supply the morning WAP and BSP. GitHub's runners are refused (HTTP 403, QA M9), so the files must be downloaded over a UK connection and imported with `betfair-prices-import.yml`, as they were for 1 Jan – 22 Sep.
- **`BETFAIR_APP_KEY` set for the 06:00 job.** Without it the job cannot match markets, and records no exchange price at prediction time. On 23 Sep it logged "Could not fetch Betfair markets: BETFAIR_APP_KEY must be set". This is needed for the executable-price secondary, not for the primary.

## Start date (24 Sep 2026)

The fixed path merged on 24 September. That day's 06:00 run had already served on the old path, so **the forward window opens with the 06:00 run of 25 September 2026.**
