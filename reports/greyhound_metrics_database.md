# The greyhound metrics database

The owner, 6 Nov 2026: "how could we build a database with the most advanced set of greyhound metrics in the world?
... work on all steps that are possible now. Do not stop until complete."

## What was built

`greyhound/pit` is a point-in-time feature store. Each table holds one row per dog per race, and every column is
known before that race's day. The tables are written to the sources store with a meta file (code hash, coverage and
fit notes). The registry of all 82 columns, with definitions and sources, is
`reports/greyhound_metrics_registry.md`. Builds run nightly in `greyhound-pit.yml` (read-only), and each table plugs
into the fast loop as a block, `pit_<table>`.

| Table | What it holds | First full build (run 42241050233) |
|---|---|---|
| ids | Each GBGB run's Betfair market and selection id (slot plus name match). A key table, not a feature. | 1,458,038 exact and 4,928 near matches. Betfair's greyhound selection ids belong to a market, not a dog: 37,106 of 40,421 matched dogs carry more than one, and 97 ids carry more than one dog. |
| ability | Kalman levels of speed, early pace and finishing speed, each with its spread, plus the chance of running the field's best figure. | Fitted by predictive likelihood: q = 0.05 a day, r = 16.2 lengths squared (speed). Race noise is 18 times r. |
| trouble | Each comment code priced in lengths (ridge regression, refitted yearly on earlier years only), the dog's trouble history, and a level on clean figures. | 84% coverage. The first year has no prices. |
| connections | The trainer's A/E against the SP, overall and in today's situation (back from a break, grade move, new yard, debut). | Trainer columns cover about 1% of runs (see the trainer finding below). The situation flags cover 100%. |
| market | The dog's and the yard's record against the SP and the Betfair SP, and the exchange's price against the bookmakers'. | The Betfair SP is present for 62% of runs. |
| breeding | Sire and dam progeny: speed, pace, finishing, A/E, and stamina by distance band. | 98-100% coverage. |
| life | Age, weight against the dog's history (card-unsafe), trials, grade path, and experience at the track and distance. | 97-100% coverage. Days since the last trial: 6%. |
| weather | NOAA hourly readings at the off (nearest station) and each track's learned weather variant. | Covers 2014 to about August 2025 only (see weather below). |

Supporting work:
- GBGB is backfilled to January 2014; the API serves nothing earlier.
- A sourced reference table of the 33 GBGB track names since 2014 (`research/greyhound/reference/`).
- Weather comes from NOAA's public-domain station data, chosen over Open-Meteo, whose free tier is non-commercial.

## What the data revealed

- **GBGB serves the CURRENT trainer, owner and season on every past result.**
  - In a 2026 fetch, none of 10,144 dogs shows a second trainer.
  - 2,648 of 16,313 dated seasons fall after their own race.
  - The database therefore uses the trainer only where GBGB was read within 4 days of the race (from the raw file's write time). That is 39 of 3,240 days at the first build, growing nightly.
  - The season is never a column.
- **The live model's trainer features read the same field.** These are `trainer_win`, `trainer_ae` and the intent block `ti_`. A pre-registered test (`trainer_*.json`, fast-loop runs 42241670921 and 42249534140) scored the model on the days read at the time (6 Oct to 5 Nov), trained on the leaky history, with and without the trainer features:
  - BFSP target: dropping them made the forecast worse. RMSE of log(BSP) went from 0.4108 to 0.4139 (+0.0031, 90% +0.0015 to +0.0047).
  - Win target: no clear change. Log loss +0.00055 (−0.00049 to +0.0016).
  - So they stay, by the fixed rule. The trainer's identity carries real signal, and the live model reads the same history.
- **NOAA's public weather archive stopped in 2025.** ISD and ISD-Lite stopped on 30 Aug 2025 and Global Hourly on 4 Oct 2025; there are no 2026 files. Weather for 2026 needs an account with Copernicus ERA5 (free, and its licence allows commercial use) or the Met Office DataHub.

## The scores, both targets (pre-registered: pit_win.json, pit_bfsp.json; test 2026-06-01 to 2026-11-05)

Each family is added alone to the live recipe's card-safe features, then all together. The 90% intervals are by day.

| Added | BFSP: RMSE log(BSP) | Change | Win: log-loss (softmax) | Change |
|---|---|---|---|---|
| live recipe | 0.4130 | - | 0.44420 | - |
| ability | 0.4132 | +0.0002 (-0.0005 to +0.0010) | 0.44447 | +0.00026 (-0.0002 to +0.0007) |
| trouble | 0.4117 | -0.0013 (-0.0020 to -0.0005) | 0.44460 | +0.00040 (0.0000 to +0.0008) |
| connections | 0.4121 | -0.0009 (-0.0016 to -0.0002) | 0.44444 | +0.00024 (-0.0002 to +0.0007) |
| market | 0.4123 | -0.0007 (-0.0015 to +0.0001) | 0.44444 | +0.00023 (-0.0002 to +0.0007) |
| breeding | 0.4127 | -0.0003 (-0.0011 to +0.0005) | 0.44446 | +0.00025 (-0.0003 to +0.0007) |
| life | 0.4119 | -0.0011 (-0.0018 to -0.0003) | 0.44441 | +0.00021 (-0.0003 to +0.0007) |
| **all** | **0.4101** | **-0.0029 (-0.0038 to -0.0020)** | 0.44445 | +0.00024 (-0.0002 to +0.0007) |
| weather | 0.4130 | 0 (no readings in 2026) | 0.44420 | 0 |

Sources: fast-loop runs 42255735836 (BFSP) and 42253527161 (Win).

**Verdict by the rule fixed beforehand (a gain on both targets):**
- No family passes yet.
- The database clearly improves the closing-price forecast. The −0.0029 is about the size of the weigh-in block's accepted gain (−0.0032, batch 9).
- It adds nothing to the result forecast.

Two caveats:
1. The test window had been read by earlier batches.
2. The first build had no pre-2018 years, so the ability and trouble fits used the earliest fifth of 2018-26. The rebuild with 2014-17 fits them on the years before 2018, as designed.

**Next gate:** `research/greyhound/experiments/pit_forward.json`, registered 6 Nov before its days exist. It scores the whole database against the live recipe on days after 6 Nov only, once 30 exist (from 7 Dec 2026), on both targets.

## Waiting on the owner

- **Weather for 2026:** a Copernicus or Met Office account.
- **Irish results:** Greyhound Racing Ireland publishes rich results under a CC BY 4.0 licence, but its site terms need written consent for automated collection (pr@grireland.ie).
- **Australia:** every state body reserves its results. GRV's Topaz split-time API needs a betting Betfair Australia account.
- **Seeding:** not in GBGB's results, and the API has no card endpoint. It would come from the GBGB website's cards, whose terms need checking first.
- **The data runner:** `vars.DATA_RUNNER` has never taken a job, so `other-data.yml` has been queued and then cancelled every night since 3 Nov.
