# Metric audit on the real database (25 Sep 2026)

*The owner's ask: "test every metric calculation — is it correct?" Unit tests check the arithmetic on hand-built races. This checks the real data the arithmetic runs on and every built feature. Source: `research/queries/metric_audit.py`, research-query run 38. Part A covers `race_results` from 2021-01-01 to 2026-03-31 (635,433 rows, 67,844 races). Part B covers the cached feature matrix (634 features on 634,743 rows).*

## The raw fields are sound

| Check | Result |
|---|---|
| Field size (NFP divides by N − 1) | `number_of_runners` equals the race's rows in 99.99% of races; no race has a placing beyond its field |
| Race time (RSR, the time figure) | one value per race in 100% of races, so it is the winning time; median 13.15 s a furlong |
| Lengths beaten | `total_dst_bt` parses for 100% of placed non-winners; winners read as 0; median 8.1 lengths, 99th percentile 95.5 (the cumulative distance behind the winner, as the metrics assume) |
| Non-finishers | 6.1% of rows: PU 28,353, F 5,494, UR 3,756, BD 429, RR 271, RO 184, VOI 154, SU 146, REF 80, CO 37 |
| Stall placement | filled on 100% of flat runs in every year since 2021 |

## What is wrong

1. **WOA is a copy of WAX.**
   - `_calc_wax_woa_cwo` sets `WOA_raw = WAX_raw`. The code comment says WOA was meant to compare with "the actual field average".
   - So eight served features duplicate eight others exactly: the career WOA of the horse, trainer, jockey and trainer–jockey pair, and the four WOA ranks.
   - The model learns nothing wrong from them, but "WOA" is one of the proprietary metrics and it is not implemented. It needs a definition of the average it is measured against.
2. **Four ranks point the wrong way for missing values.**
   - Every within-race rank is computed highest first, with missing values placed last (`custom_metrics.py`, `rank(ascending=False, na_option="bottom")`). For measures where lower is better, that puts a runner with no figure beside the best in the race:
     - `rLB`: career lengths beaten;
     - `rConsistency`: the standard deviation of NFP;
     - `rDistApt`: distance from the preferred trip;
     - `rGoingPref`: distance from the preferred going.
   - The evidence: career lengths beaten has a rank correlation of +0.349 with log BSP, and its within-race rank only −0.099.
3. **`track_draw_bias` is empty on every row.**
   - A served feature that carries nothing: 0% coverage, no distinct value.
   - Its code also shifts by row within the track rather than by day. If it were filled, a race would read the results of earlier races on the same day, which the 06:00 card does not have.
4. **Exact duplicates beyond WOA**, harmless but worth knowing:
   - `rNFP` = `horseNFPrank`;
   - `course_experience` = `horse_track_runs`;
   - `course_avg_nfp` = `horse_track_nfp`;
   - `class_change` = `it_class_change`;
   - `lead_prob_rank` = `rLeadProb`;
   - `rDrawBiasAlign` = `rDrawFieldAdj`.
   The form windows repeat four engine features by construction (`fw_nfp_car` = career NFP, `fw_nfp_l1` = last-run NFP, `fw_rsr_car`, `fw_rsr_l1`).
5. **Void races are read as non-finishes** (154 rows marked VOI). A void race is not a run. This is minor.

## Career NFP, which the owner expected to count for more

Career NFP is strongly tied to the price on its own: a rank correlation of −0.453 with log BSP, against −0.453 for `fw_nfp_car` (the same number) and −0.491 for last-run NFP. It shows little gain in the model because a dozen other features carry the same information (its rank, its windows, the last-3/5/10 totals). The grouped test is what measures it: withholding the whole NFP family (57 features) costs +0.0048 in the price error, 90% CI +0.0032 to +0.0063 (iteration 34).

## What happens next

- The four rank directions: `model/blocks/race_relative_wide.py` (iteration 37) already reads all four sources as z-scores and gaps to the best, which keep a missing value missing and do not depend on the direction. If it passes, the broken ranks can be withheld. A WOA needs the owner's definition of the average it is measured against.
- `track_draw_bias` would be rebuilt by day, not by row, or removed from the served list.
- These are engine changes to the served features. Each goes through the research loop's decision rule before it is served.
