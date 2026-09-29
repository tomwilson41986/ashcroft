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

## Amendment, 24 Sep 2026 (21:45 UTC), before the window opens: the model it opens on

At the owner's instruction, the window opens on a model retrained since this document was written. **The 06:00 run of 25 September serves the price model on 535 features:** the 501 of the model served on 24 September, plus the card-safe intent block (20) and the freshness block (14).
- **Artefact:** train-bfsp run 31 (commit 86ca911), trained through 2026-09-22, feature hash `9bc95363f3181433`, 3000 rounds.
- **On its own 60-day holdout** (24 Jul – 22 Sep 2026, 21,491 runners), against the model it replaces on the same rows: MAE of the target 0.4936 → 0.4866, RMSE 0.6374 → 0.6282.
- **Walk-forward, Jan–Mar 2026** (research-loop iteration 27): the rule +5.74% on the old model, +5.85% on this one.

**Nothing in the rule, the criteria or the window changes.** The rule reads the 06:00 job's `predicted_bfsp`, whichever model makes it. Two consequences are fixed now, before any forward card is served:
- **The backtest's forecast** for the forward days (the reading-2 comparison) is rebuilt with this 535-feature recipe, so the two forecasts compared are the same model.
- **A later model may replace this one during the window only under strict conditions.** It must pass the research loop's decision rule on the development window, and it can never be swapped in mid-day. Every change is written to the ledger with its first 06:00 date. The report gives the headline for each model's days as well as for the whole window. The criterion stays the whole window.

## Amendment, 25 Sep 2026 (07:10 UTC): what day 1 ran on

**The 06:00 run of 25 September served the 501-feature model, not the 535.** The artefact was committed on 24 September, but PR #75, which puts it on the default branch, merged at 06:44 UTC on 25 September, after that day's scheduled run.
- **Run 195** (scheduled, 06:13–06:26 UTC): the 501 model. It wrote `racecards/2026-09-25_0613.csv` and `predictions/2026-09-25.csv` (528 runners, 44 races).
- **Run 196** (dispatched 06:46 UTC on the merged commit eeb5dec): the 535 model, same card, 528 of 528 runners priced. It wrote `racecards/2026-09-25_0646.csv`, and **rewrote `predictions/2026-09-25.csv` at 06:57:40 UTC (07:57 UK).** That is inside the rule that excludes a day only when the file is written after 09:00 UK.
- **Day 1 is scored on run 196's file, the 535 model,** so the whole window runs on one model as the previous amendment intends.
- **The 501's day-1 forecast is not kept as a file, but can be rebuilt.** Run 195's card is kept at `racecards/2026-09-25_0613.csv` and the model is deterministic. It is not scored.
- **No bets were affected.** No bet had been placed when the file was rewritten (execute.yml runs from 10:00 UTC), and `BETFAIR_APP_KEY` is unset, so the job records no exchange prices at prediction time.

## Amendment, 25 Sep 2026 (11:40 UTC), before the 26 September card: the model from day 2

At the owner's instruction of 25 September ("add the form windows into the main model"), **the 06:00 run of 26 September serves the price model on 615 features:** the 535 served on 25 September plus the 80 form windows (`model/form_windows.py`), fitted at 6000 rounds instead of 3000.
- **Artefact:** train-bfsp run 32 (commit 24df459), trained through 2026-09-22 (697,903 runs from 2021-01-01), feature code hash `f879b10c8369467a`, 6000 rounds. Verified before commit (`scripts/verify_model.py`): PASS (`reports/model_verify_615.md`): every feature built by the live path, books of 1, log prices correlated 0.993 with the 535's on the last fortnight's 5,225 runners.
- **Why it qualifies:** it passed the research loop's decision rule on the development window.
  - Iteration 29 (served recipe): −0.0073 in the price-forecast error against the 535 (90% CI −0.0083 to −0.0064), Brier skill against the market +0.0022 (+0.0011 to +0.0034).
  - Iteration 36 (served recipe, both at 6000 rounds): −0.0088 (−0.0099 to −0.0078); the rule at 22% shorter +6.06% → +7.70% on January–March.
- **The change is between days,** as the amendment of 24 September requires. Day 1 (25 September) stays scored on the 535, run 196's file. From day 2 the window runs on the 615.
- **Reporting:** the headline, its interval and the halves are given for the 535's day and the 615's days separately, as well as for the whole window. The whole window is still the criterion.
- **The backtest's forecast** for the forward days (the reading-2 comparison) is rebuilt with each day's model: the 535 recipe for 25 September, the 615 at 6000 rounds from 26 September.

**Nothing else changes:** not the rule, not the criteria, not the window's length or its exclusions.

## Amendment, 28 Sep 2026 (13:00 UTC), before the 29 September card: the model from day 5

At the owner's instruction of 28 September ("switch to the best model"), **the 06:00 run of 29 September serves the complete-careers pair gated with race_xent**, in place of the 615 (days 2-4, 26-28 September).
- **Artefact:** three boosters served as one (`data/models/bfsp_ensemble.json`), each on 993 features built on the history from 2018 and fitted on rows from 2021, trained through 2026-09-22, all verified before commit (`scripts/verify_model.py`: PASS):
  - the main: the combined 968s5xh with the Kalman rating, handicap angles and travel (train-bfsp run 54; `reports/model_verify_968s5xh_h18.md`);
  - the within-race partner (run 55; `reports/model_verify_968s5xd_h18.md`), averaged with the main;
  - race_xent (run 56; `reports/model_verify_race_xent_h18.md`), gated to the pair's leaders: its log price takes weight 1 for the pair's ranks 1-3 in each race, 0.5 for ranks 4-7 and 0 from rank 8, each race renormalised to a book of 1.
- **Why it qualifies:** on the development window (27 Sep 2025 - 31 Mar 2026, 53,910 runners) it prices the BSP at 0.3926 mean absolute log error against the 615's 0.4358, and against the best model before it (0.3966) −0.0040 (−0.0048 to −0.0031), with Brier skill against the market resolved better; the rule, paired by race bootstrap against that model on Jan-Mar 2026, +12.65% against +10.70%, +1.95 points (+1.21 to +2.75) (research query run 36401419654). The gate's weights were set before its bets were scored.
- **Checked on the 06:00 path:** the committed files dry-run on 28 September's card (predict-now runs 32 and the gated run after it): every runner priced, books of 1. The history the job builds starts in 2018 (the members' training summaries), which needs swap on the runner: `predict.yml` gains the opt-in swap step of the dry runs and a 60-minute limit.
- **The change is between days**, as the amendment of 24 September requires. Days 1-4 stay scored on the models that served them (the 535 on 25 September, the 615 from 26 to 28 September); from 29 September the window runs on this model. The headline, its interval and the halves are reported for each model's days as well as for the whole window, which is still the criterion.

**Nothing else changes:** not the rule, not the criteria, not the window's length or its exclusions.

## Record, 29 Sep 2026 (07:10 UTC): day 5 written by a dispatched run

**The scheduled 06:00 run of 29 September wrote nothing; a dispatched run wrote the day's file at 07:03 UTC (08:03 UK), inside the rule.**
- **Run 200** (scheduled, 06:13-06:14 UTC): horseracebase's login page came back without its form, so the job could not log in, priced no card and wrote no file. It still ended green.
- **Run 201** (dispatched 06:42 UTC on the same commit, 5023faa): logged in at once, wrote `racecards/2026-09-29_0642.csv` (375 runners in 38 races) and `predictions/2026-09-29.csv` at 07:03:11 UTC. Every runner was priced by the gated three, on the history from 2018.
- **Day 5 is scored on run 201's file.** Nothing else changes. PR #78 makes the login try again after 30 s, 60 s and 120 s, and makes a morning that prices nothing fail red.

## Amendment, 29 Sep 2026 (11:45 UTC), before the 30 September card: horses the card lists that are not running

**From the 06:00 run of 30 September the job leaves out every horse the card lists that is not running.** The owner found non-runners still priced on 29 September.
- **What horseracebase does:** it does not always drop a withdrawn horse from its card. It can keep the horse with the jockey blanked (a link to `jockeys.php?id=0`, sometimes with the claim left beside it: "(7)"). It also lists Irish reserves as RESERVE until they get into the race.
- **The effect:** priced as runners, both took their share of the race's book from the horses that ran, so every other price in those races was too long by that share.
- **Days 1-5 in the record** (research query `records_not_running_0925_0929.py`, run 36562950937; each day's file against the card it was priced on):

  | Day | Reserves priced | Blank jockeys priced | Races affected |
  |---|---|---|---|
  | 25 Sep | 33 | 0 | 12 |
  | 26 Sep | 3 | 0 | 1 |
  | 27 Sep | 3 | 1 (Out On Friday, Curragh 5:30) | 2 |
  | 28 Sep | 15 | 1 (Popeye Doyle, Wolverhampton 4:22) | 7 |
  | 29 Sep | 3 | 0 | 1 |

  - That is 23 races in all. Such horses held 7.4% of an affected race's book on average and 20.4% at most.
  - The research ledger (`non-runners-blank-jockey-0929`) has the detail.
- **The change:** `daily_predictions.declared_runners` drops both kinds on the HTML and the CSV card and counts each race's field again (PR #78).
  - Training's rows are the horses that ran, so the card now matches what the model was trained on.
- **Scoring:** days 1-5 stay scored on their files as written; the report gives the headline with and without the races whose book held a reserve or a blank jockey.

**Nothing else changes:** not the rule, not the criteria, not the window's length or its exclusions.
