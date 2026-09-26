# Which model serves from 27 September: the recommendation

*Written 26 Sep 2026 for the owner's decision. Nothing changes without the owner's word; if there is no answer before 06:00 UTC on 27 Sep, the 615 keeps serving.*

## Recommendation

**Serve the 944 from the 06:00 run of 27 September.** If the owner prefers a smaller step, the 859 is verified and dry-run clean as well.

The 944 is the 615 served today plus six blocks that each passed the research loop's decision rule on the served recipe, less the two sire counts found faulty. In order of adoption:
- within-race readings (60)
- time figure, wide within-race readings and exposure (171)
- collateral form, meaning today's runners' direct meetings (7)
- bookings (6)
- form lines (10)
- form variants (77)

**Trained longer: serve the 7000-round 944.** Iteration 63 found the 6000-round cap was costing a little. The same 944 grown to 9000 rounds was −0.0014 better (−0.0018 to −0.0009) than at 6000. Its first 6000 trees are the same fit, so no seed noise is in that number.

The 944 was then trained at 7000 rounds (train-bfsp run 38, artifact `bfsp-model-38`), the middle of the rounds early stopping chose.
- **Verify: PASS.** The same 944 features, every block built as served, books of 1. Its log prices correlate 0.9939 with the 615's (the 6000-round 944: 0.9942).
- **Dry run on today's card (predict-now run 13): clean.**
  - 569 runners in 53 races, every one priced, books of 1.
  - 16 minutes from load to prices.
  - Against the 6000-round 944 on the 48 races with an unchanged field: correlation 0.9972, the same top pick in 43.

If the 944 is chosen, this is the file. The model file is 84 MB, under GitHub's 100 MB limit.

## The evidence, on the served recipe

All at 6000 rounds, on the development window of 27 Sep 2025 to 31 Mar 2026: 53,910 runners in 5,923 races. The locked holdout was not read.

| model | what it adds | price-forecast error (mean abs log) | step, paired (90% CI) | early-price rule, Jan–Mar 2026 |
|---|---|---|---|---|
| the 615 (served now) | — | 0.4358 | — | +7.70% |
| the 675 | within-race readings | 0.4280 | −0.0078 | +7.65% |
| the 853 | time figure, wide readings, exposure, collateral form | — | −0.0053 (−0.0064 to −0.0041) | +7.59% to +7.83% |
| the 859 | bookings | 0.4197 | −0.0041 (−0.0051 to −0.0032) | +8.09% |
| **the 944** | form lines, form variants, less the two sire counts | **0.4156** | **−0.0040 (−0.0049 to −0.0031)** | **+8.44% (+7.17 to +9.82)** |

- Each row's step is measured against the row above, on the same runners (iterations 36, 42, 48 and 61). The 853 was fitted as a variant only, so the table gives no level for it.
- The rule's figure moves by about a point between fits of the same model: the served recipe's seed floor is about ±0.003 on the error and a point on the rule.

**The 944 against the 859 (iteration 61):**
- The error improvement is resolved in every rank band.
- Brier skill against the market improves by +0.0017 (+0.0007 to +0.0026). Its probabilities are measurably closer to the results, not only to the price.
- The early-price rule is positive in every month from December to March (+6.9% to +10.6%).
- It is positive in every morning-price band, including 1-4 (+3.1%), where the earlier models were not.
- The gain holds in each part of the window: September to December 2025 −0.0033 (−0.0046 to −0.0021), December to March −0.0051 (−0.0065 to −0.0037).

## The checks

- **Verify (train-bfsp run 37): PASS.**
  - 944 features, 6000 rounds, trained through 22 Sep on 697,903 runs.
  - The same engine code as the served model.
  - Every block built on the matrix exactly as the 06:00 path builds it.
  - Books of 1.
  - On the last fortnight its log prices correlate 0.9942 with the 615's.
- **Dry run on today's card (predict-now run 11): clean.**
  - 584 runners in 53 races, every one priced, books of 1.
  - 16 minutes from load to prices; the limit is 45.
  - Against this morning's 615: correlation 0.9859, the same top pick in 46 of 53 races.
- **Parity, the live path against training.**
  - The overnight check (research query, two March days) found the within-race readings, time figure, wide readings, exposure, collateral form and bookings identical runner for runner.
  - The 944's own check (form lines and form variants included): **clean** (research query run 36230342697). On a perfect card every feature of every block the 944 reads (form lines and form variants included) equals training's, runner for runner, and its prices are identical (correlation 1.0000, the same top pick in every race). As the 06:00 job runs, with the card's own gaps (debutants' sires, rail moves, sex, claims), its prices stay within 0.031 in mean |Δlog| of training's (the 615: 0.035), the same top pick in 96% of races (the 615: 92%).
- **The sire counts.** `sire_runners` and `damsire_runners` are misaligned within each sire in the engine: a lookahead in training, and a different value at 06:00. The 944 leaves them out, at no cost (iteration 61: −0.0005). The 615 serving today still reads them. The engine fix waits for the next engine rebuild.

## The top pick: the goal of a profitable rank 1

The 944's rank 1, its highest win probability, on January to March 2026 (2,530 races), settled three ways:

| | every race | when also ≥ 22% shorter than the morning price (605) |
|---|---|---|
| backed at the morning price, traded out at BSP | **+1.19% (+0.44 to +1.96)** | **+5.18% (+3.33 to +7.17)** |
| held to the result at the morning price | −6.10% | −10.67% |
| held to the result at BSP | −7.56% | −16.51% |

The top pick makes money as a trade against the price, not as a bet held to the result. That holds for every model so far, because the model forecasts the price. The market's favourite, for scale, loses −1.44% traded out.

The pre-registered forward test measures exactly the traded version, from the 06:00 run's own prices.

## An option: the three-seed average

Averaging three fits at different seeds beat each single fit by −0.0013 to −0.0040 (iteration 46). It would also make the served forecast repeatable: one fit differs from another by the choice of CPU alone (the fit-hardware-noise entry in the ledger). It needs three boosters of 72–84 MB each.

Compressed, a 7000-round booster is 33 MB with gzip or 25 MB with xz. A compressed booster loads in 2–3 seconds and predicts identically. The options:
- about 76 MB in git per retrain (xz), or
- release files, plus a change to how the 06:00 job fetches the model.

This is a storage decision for the owner, not needed for 27 Sep.

## What deploying involves (only with the owner's word)

1. Commit the chosen artefact to `data/models` on the branch, with its verify report. For the 944, that is the 7000-round file (train-bfsp run 38, artifact `bfsp-model-38`). For the 958, it is its own run's file, once its checks pass.
2. Add an amendment to `reports/preregistration_clv_forward.md`. It records the change between days, as the pre-registration requires, the evidence above, and its first 06:00 date. The forward report then gives the 615's days and the 944's days separately as well as the whole window.
3. Merge PR #77 to the default branch before 06:00 UTC on 27 Sep. The 06:00 job reads `data/models` from the default branch; nothing else in the merge changes what is served.

## A third option, if it is ready in time: the 958

The 958 is the 944 plus connection windows: the trainer's and jockey's recent runners over a window ladder, with their career NFP ranked in the race. The specification lists these ranks as trainerNFPrank and jockeyNFPrank, but the engine never built them.

- **On the screen (iteration 67):** −0.0060 (−0.0076 to −0.0046) beyond form lines, the largest gain since the form windows.
- **On the 944 itself, at the served recipe (iteration 68):** −0.0040 (−0.0050 to −0.0032).
  - Every rank band is resolved.
  - Each part of the window is resolved: September to December −0.0043, December to March −0.0037.
  - Neither Brier skill nor concordance against the market is worse.
  - The early-price rule is **+9.60% (+8.23 to +10.99)** against the 944's +8.41% on the same runs.
- **Training:** started at 11:24 UTC at 7000 rounds, due about 13:15.
- **Parity: clean** (research query run 36236954512). On a perfect card and on the 06:00 card with its own gaps, every connection-window feature equals training's, runner for runner. The 944 control repeated its earlier check exactly.
- **Checks still to pass:** verify, and a dry run on today's card.

If every check passes, the 958 is the stronger recommendation; if any fails, or it is late, the 944 stands. Collateral form was tested beside it and added nothing (−0.0036 with it, against −0.0040 without).

**A note on repeat fits.** GitHub's runners are not all the same CPU. The same fit on a different processor is not identical to the bit. The effect on the error was measured at −0.0001 (−0.0009 to +0.0006), far below every step in the table above.

## In progress, not needed for this decision

- A slower learning rate (0.02) at 9000 rounds read −0.0018 against the 6000-round 944, but it hit its cap again. It is left for the next recipe test at a higher cap.
- Connection grains: the connections' last fortnight, race code, course and pairing. They added nothing beyond connection windows (−0.0008, unresolved) and are retired.
