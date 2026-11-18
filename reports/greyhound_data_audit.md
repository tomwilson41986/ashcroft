# Greyhound data audit (Phase 0 of the improve-validate-test brief, 18 Nov 2026)

racing². Every figure from greyhound-api-research.yml run 44331099305 (task data-audit, research/greyhound/data_audit.py at commit aa6c64f; artifact greyhound-api-research-data-audit-8). Read-only.

## GBGB results (the training data), by year

Drops are greyhound.data.clean's, in its order: no dog id or date; no position on a run; no parseable off time; a race with no winner (void). Field size counts the races clean keeps.

| Year | Rows in | Races in | Dup race+dog | Null dog id | Drop no id | Drop no position | Drop no off time | Void races (rows) | Rows out | Races out | 6-runner races | 5-runner | 4 or fewer | SP missing (kept) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2014 | 337,573 | 56,861 | 246 | 4,100 | 4,100 | 4,918 | 686 | 161 (934) | 326,935 | 55,614 | 49,269 | 5,914 | 420 | 34 |
| 2015 | 433,547 | 109,939 | 335 | 2,952 | 2,952 | 3,825 | 88,582 | 152 (892) | 337,296 | 57,425 | 50,360 | 6,361 | 567 | 7 |
| 2016 | 431,459 | 110,247 | 348 | 3,809 | 3,809 | 3,446 | 89,647 | 170 (996) | 333,561 | 56,896 | 49,494 | 6,655 | 647 | 16 |
| 2017 | 430,253 | 108,724 | 117 | 3,885 | 3,885 | 3,844 | 87,130 | 179 (1,052) | 334,342 | 56,967 | 48,943 | 7,062 | 569 | 6 |
| 2018 | 438,671 | 107,417 | 37 | 1,964 | 1,964 | 4,196 | 79,172 | 201 (1,161) | 352,178 | 59,742 | 53,894 | 5,449 | 398 | 12 |
| 2019 | 442,709 | 110,298 | 79 | 1,659 | 1,659 | 4,929 | 81,912 | 172 (1,005) | 353,204 | 60,006 | 53,895 | 5,463 | 648 | 4 |
| 2020 | 335,982 | 88,404 | 60 | 522 | 522 | 3,368 | 85,138 | 127 (758) | 246,196 | 41,538 | 38,684 | 2,687 | 167 | 793 |
| 2021 | 377,516 | 98,693 | 108 | 547 | 547 | 3,967 | 78,353 | 193 (1,140) | 293,509 | 49,587 | 45,898 | 3,402 | 287 | 521 |
| 2022 | 371,982 | 96,315 | 200 | 1,841 | 1,841 | 5,637 | 74,938 | 230 (1,356) | 288,210 | 49,154 | 42,904 | 5,811 | 439 | 378 |
| 2023 | 386,392 | 98,024 | 726 | 8,368 | 8,368 | 8,565 | 70,702 | 237 (1,355) | 297,402 | 52,204 | 38,104 | 12,505 | 1,595 | 64 |
| 2024 | 384,406 | 94,897 | 853 | 12,717 | 12,717 | 12,000 | 62,464 | 240 (1,348) | 295,877 | 53,063 | 33,213 | 17,397 | 2,453 | 6 |
| 2025 | 358,132 | 87,389 | 688 | 13,987 | 13,987 | 11,962 | 56,095 | 237 (1,314) | 274,774 | 49,818 | 28,151 | 19,352 | 2,315 | 49 |
| 2026 | 294,486 | 72,348 | 527 | 13,291 | 13,291 | 9,556 | 46,506 | 139 (760) | 224,373 | 40,988 | 21,441 | 17,685 | 1,862 | 14 |

Date ranges: every year runs 1 Jan to 31 Dec except 2026 (1 Jan to 2026-11-17).

## Betfair price files (GB win) and the match to GBGB runs

| Year | Price rows | Markets | BSP null share | Dup date+off+track+trap | GBGB runs (non-trial) | Runs with a BSP | Share | Races fully priced | Races |
|---|---|---|---|---|---|---|---|---|---|
| 2014 | 222,548 | 37,520 | 0.0842 | 1 | 326,935 | 0 | 0.0 | 0 | 55,614 |
| 2015 | 230,262 | 39,108 | 0.0878 | 0 | 337,296 | 0 | 0.0 | 0 | 57,425 |
| 2016 | 216,298 | 40,625 | 0.1377 | 0 | 333,561 | 0 | 0.0 | 0 | 56,896 |
| 2017 | 462,170 | 50,150 | 0.6292 | 0 | 334,342 | 0 | 0.0 | 0 | 56,967 |
| 2018 | 522,643 | 60,998 | 0.4506 | 1 | 352,178 | 16,463 | 0.0467 | 2,806 | 59,742 |
| 2019 | 303,706 | 51,172 | 0.0 | 0 | 353,201 | 20,189 | 0.0572 | 3,401 | 60,005 |
| 2020 | 235,549 | 39,618 | 0.0 | 0 | 246,196 | 15,543 | 0.0631 | 2,607 | 41,538 |
| 2021 | 287,571 | 48,563 | 0.0 | 2 | 293,499 | 160,421 | 0.5466 | 27,034 | 49,581 |
| 2022 | 287,309 | 48,956 | 0.0 | 0 | 288,210 | 203,741 | 0.7069 | 34,612 | 49,154 |
| 2023 | 296,301 | 51,914 | 0.0 | 0 | 297,402 | 278,519 | 0.9365 | 48,833 | 52,204 |
| 2024 | 292,021 | 52,283 | 0.0 | 0 | 295,877 | 289,637 | 0.9789 | 51,865 | 53,063 |
| 2025 | 270,827 | 49,035 | 0.0 | 0 | 274,774 | 269,783 | 0.9818 | 48,851 | 49,818 |
| 2026 | 223,654 | 40,827 | 0.0 | 0 | 224,373 | 222,505 | 0.9917 | 40,585 | 40,988 |

## Recorded books, prices served

- Recorder (delayed key, books_greyhound.csv): 44 days, 6 Oct to 18 Nov 2026. Live key (books_greyhound_live.csv): 16 days, 3 Nov to 18 Nov 2026 (run log; per-day rows, removed runners and settled markets are in the artifact's report.json).
- Served predictions: 44 days, 2026-10-06 to 2026-11-18. The live recipe's out-of-sample prices (stack_mlsitd and its _mem set): 37 days, 2026-10-06 to 2026-11-11.

## Findings and anomalies (flagged, not yet fixed)

1. **The Betfair BSP join fails before 2021.** The price files hold 216k-523k GB win rows a year from 2014, but the share of GBGB runs matched to a BSP is 0% in 2014-2017, 5-6% in 2018-2020, 55% in 2021, 71% in 2022, then 94-99% from 2023. Any BSP-target model can only learn from 2021 on (the live recipe fits from June 2024, so it is not affected). The cause (track names, off times or trap numbers in the early files) is not established: no data found on it here.
2. **Fields are shrinking.** Six-runner races were 90% of races kept in 2018 (53,894 of 59,742) and 52% in 2026 (21,441 of 40,988); five-runner races rose from 9% to 43%. Vacant traps are now the norm, not the exception: field size and trap-vacancy effects deserve their own check.
3. **Duplicate (race, dog) rows: 527-853 a year since 2023** (37-246 a year before). The count includes rows with no dog id (pandas treats two missing ids in a race as equal), which clean drops anyway; how many are true duplicates of a real dog in one race is not separated here. greyhound.data.clean has no de-duplication step, so any true duplicates enter training twice. To check (the next audit run counts duplicates among non-null dog ids only).
4. **Missing dog ids and positions are rising**: null dog id 522-1,964 rows a year to 2022, 8,368-13,987 from 2023; no position 3,368-5,637 to 2022, 8,565-12,000 from 2023. clean drops them; what they are (vacant traps listed, reserves, data loss) is not established.
5. **Rows with no off time** (46,506-85,138 a year) are the largest drop; greyhound.data notes that GBGB serves trials with no off time, consistent with zero trial rows surviving clean. Not checked row by row.
6. **Void races**: 127-240 races a year dropped (no winner).
7. **Rule 4 / reduction factors are recorded but never applied** (reports/greyhound_pipeline_map.md).
8. **The out-of-sample prices of the live recipe end on 11 Nov 2026**: studies after that date fall back to the served predictions, a different model on some days.

