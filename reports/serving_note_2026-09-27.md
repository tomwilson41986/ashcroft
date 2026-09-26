# Which model serves from 27 September: the recommendation

*Written 26 Sep 2026 for the owner's decision, last updated 17:25 UTC. Nothing changes without the owner's word; if there is no answer before 06:00 UTC on 27 Sep, the 615 keeps serving.*

## Recommendation

**Serve the 958 from the 06:00 run of 27 September** (train-bfsp run 39, artifact `bfsp-model-39`). Every check has passed:
- the test on the served recipe against the 944
- verify
- the parity of its new block between the 06:00 path and training
- a dry run on today's card

**Directly against the 615 it would replace** (same 53,910 runners, both at 6000 rounds): the price-forecast error falls by −0.0243 (90% CI −0.0261 to −0.0229), in every rank band. Its Brier skill against the market improves by +0.0048 (+0.0031 to +0.0065), and the winner-vs-loser concordance by +0.0032 (+0.0011 to +0.0053), both intervals clear of zero. The early-price rule goes from +7.70% to +9.60%.

If the owner prefers a smaller step, the 7000-round 944 (run 38, `bfsp-model-38`) is verified and dry-run clean. The 859 is a further fallback.

**Deploy it with the card pedigree fix** (found 26 Sep afternoon; see "The 06:00 card's missing pedigree" below). The 06:00 card gave debutants no sire, damsire or sex, so every model so far priced them a mean 0.26 in log terms (about 30%) away from the price its training features give. The fix reads the pedigree the card does carry, and it applies to whichever model serves.

**The Huber loss does not change the recommendation** (iteration 76, 16:00 UTC). On the screen it beat the squared-error loss by −0.0029 on the 958's features (iteration 74). On the served recipe the gap closes: −0.0003 (−0.0010 to +0.0006), unresolved. Only the runners over 50 on the BSP improve (−0.0050); the runners at 8 to 16 are a little worse. Its trading readings are lower too: the early-price rule makes +9.20% against the 958's +9.60%, and the top pick traded out makes +1.14% against +1.45%. The 958 trained with it (train-bfsp run 40) passed verify, but it is not a candidate.

**The next candidate after the 958: the debut market block** (iteration 80, 17:15 UTC). How the market has priced each yard's debutants and lightly raced runners (`model/blocks/debut_market.py`, 10 features), beside the 958's features at the served recipe: −0.0032 (−0.0040 to −0.0024) on the price, every rank band better, and −0.0067 in maiden, novice and bumper races, where the error is largest. That was measured under the Huber loss. Iteration 81 measures it under the served squared-error loss, and it would need its parity check and verify before it could serve, so it is not part of tomorrow's recommendation.

What each adds:
- **The 944** is the 615 served today plus six blocks that each passed the research loop's decision rule on the served recipe, less the two sire counts found faulty. In order of adoption:
  - within-race readings (60)
  - time figure, wide within-race readings and exposure (171)
  - collateral form, meaning today's runners' direct meetings (7)
  - bookings (6)
  - form lines (10)
  - form variants (77)
- **The 958** is the 944 plus connection windows (14): the trainer's and jockey's recent runners over a window ladder (career, last 20, last 100), with their career NFP ranked in the race. The specification lists those two ranks as trainerNFPrank and jockeyNFPrank; the engine never built them. In the 958 they come 9th and 10th of 958 features by gain.

Both are trained at 7000 rounds, where early stopping lands (iterations 63 and 70).

## The evidence, on the served recipe

All at 6000 rounds, on the development window of 27 Sep 2025 to 31 Mar 2026: 53,910 runners in 5,923 races. The locked holdout was not read.

| model | what it adds | price-forecast error (mean abs log) | step, paired (90% CI) | early-price rule, Jan–Mar 2026 |
|---|---|---|---|---|
| the 615 (served now) | — | 0.4358 | — | +7.70% |
| the 675 | within-race readings | 0.4280 | −0.0078 | +7.65% |
| the 853 | time figure, wide readings, exposure, collateral form | — | −0.0053 (−0.0064 to −0.0041) | +7.59% to +7.83% |
| the 859 | bookings | 0.4197 | −0.0041 (−0.0051 to −0.0032) | +8.09% |
| the 944 | form lines, form variants, less the two sire counts | 0.4156 | −0.0040 (−0.0049 to −0.0031) | +8.44% (+7.17 to +9.82) |
| **the 958** | connection windows | **0.4115** | **−0.0040 (−0.0050 to −0.0032)** | **+9.60% (+8.23 to +10.99)** |

- Each row's step is measured against the row above, on the same runners (iterations 36, 42, 48, 61 and 68). The 853 was fitted as a variant only, so the table gives no level for it.
- The rule's figure moves by about a point between fits of the same model. The served recipe's seed floor is about ±0.003 on the error and a point on the rule.

**The 958 against the 944 (iteration 68):**
- The error improvement is resolved in every rank band, and in each part of the window: September to December −0.0043 (−0.0056 to −0.0030), December to March −0.0037 (−0.0050 to −0.0023).
- Neither Brier skill nor concordance against the market is worse.
- On the same runs, the early-price rule is +9.60% against the 944's +8.41%. It is positive and resolved in every morning-price band: 1–4 +3.34%, 4–8 +3.57%, 8–16 +3.04%, 16+ +17.14%. It is also positive in every month from January to March (+8.5% to +11.6%).
- The first race at a meeting (+9.67%) and races 10 and later on the day (+10.25%) show no same-day leak.

**The 944 against the 859 (iteration 61):**
- The error improvement is resolved in every rank band, and in each part of the window: September to December −0.0033, December to March −0.0051.
- Brier skill against the market improves by +0.0017 (+0.0007 to +0.0026).
- The early-price rule is positive in every month from December to March (+6.9% to +10.6%) and in every morning-price band.

**Trained longer.** For the 944, the trees beyond 6000 were worth −0.0014 (iteration 63). For the 958, early stopping lands at 4,300 to 6,700 rounds and the extra trees are worth −0.0003 (iteration 70). A slower learning rate (0.02, about 9,000 rounds) is a further −0.0010 on the price. It does not show in the rule (+9.27% against +9.66%) or the top pick, so it is left for the next retrain.

## The checks

**The 958 (train-bfsp run 39):**
- **Verify: PASS.**
  - 958 features, 7000 rounds, trained through 22 Sep on 697,903 runs.
  - Every block, connection windows included, built on the matrix exactly as the 06:00 path builds it.
  - Books of 1.
  - Log prices correlate 0.9936 with the 615's over 9–22 Sep.
- **Parity: clean** (research query run 36236954512). On a perfect card and on the 06:00 card with its own gaps, every connection-window feature equals training's, runner for runner. The 944, priced as a control, repeated its earlier check exactly.
- **Dry run on today's card (predict-now run 14): clean.**
  - 568 runners in 53 races, every one priced, books of 1.
  - 16 minutes from load to prices; the limit is 45.
  - Against the 7000-round 944 on the 52 races with an unchanged field: correlation 0.9949, the same top pick in 46.

**The 944 (runs 37 at 6000 rounds, 38 at 7000):**
- **Verify: PASS** for both.
- **Dry runs clean.**
  - Run 11: 584 runners.
  - Run 13: 569 runners, against the 6000-round file correlation 0.9972.
- **Parity: clean** (research query run 36230342697). On a perfect card its prices equal training's exactly. With the card's own gaps (debutants' sires, rail moves, sex, claims), they are within 0.031 in mean |Δlog| (the 615: 0.035), with the same top pick in 96% of races (the 615: 92%).

**The sire counts.** `sire_runners` and `damsire_runners` are misaligned within each sire in the engine: a lookahead in training, and a different value at 06:00. The 944 and the 958 leave them out, at no cost (iteration 61: −0.0005). The 615 serving today still reads them. The engine fix waits for the next engine rebuild.

**Repeat fits.** GitHub's runners are not all the same CPU, and the same fit on a different processor is not identical to the bit. The effect on the error was measured at −0.0001 (−0.0009 to +0.0006), far below every step in the table.

## The top pick: the goal of a profitable rank 1

The top pick is the model's rank 1, its highest win probability. Here it is with the 958's features, on January to March 2026 (2,529 races), settled three ways:

| | every race | when also ≥ 22% shorter than the morning price (610) |
|---|---|---|
| backed at the morning price, traded out at BSP | **+1.45% (+0.68 to +2.22)** | **+5.69% (+3.74 to +7.63)** |
| held to the result at the morning price | −3.92% | −0.87% |
| held to the result at BSP | −6.50% | −8.87% |

The 944's top pick on the same runs traded out at +1.05%, and +5.31% with the rule. Where the top pick is not the morning favourite, it traded out at +4.88% (755 races). The market's favourite, for scale, loses −1.44% traded out.

The top pick makes money as a trade against the price, not as a bet held to the result. That holds for every model so far, because the model forecasts the price. The pre-registered forward test measures exactly the traded version, from the 06:00 run's own prices.

## An option: the three-seed average

Averaging three fits at different seeds beat each single fit by −0.0013 to −0.0040 (iteration 46). It would also make the served forecast repeatable: one fit differs from another by the choice of CPU alone (the fit-hardware-noise entry in the ledger). It needs three boosters of 72–84 MB each.

Compressed, a 7000-round booster is 33 MB with gzip or 25 MB with xz. It loads in 2–3 seconds and predicts identically. The options:
- about 76 MB in git per retrain (xz), or
- release files, plus a change to how the 06:00 job fetches the model.

This is a storage decision for the owner, not needed for 27 Sep.

## The 06:00 card's missing pedigree (found and fixed 26 Sep)

The morning card is the HTML page, which has no pedigree column. A horse with earlier runs gets its sire, damsire and sex from its own history. A debutant has none, so it reached the model with them missing, though training always had them.

- **The size of it.** On the parity days (28 and 25 March, 745 runners), the 54 runners without a sire on the card were priced a mean 0.257 in log terms away from training's price for them, against 0.016 for the rest. That was 56% of the whole difference between the 06:00 path and training.
- **Where the pedigree is.** Every horse name on the card carries a tooltip: "Bay, Male, Stallion - Harry Angel (IRE), Dam - Twist n Shake". On 26 Sep all 568 runners had one, all 31 debutants included. There is no damsire in it.
- **The fix.** The scraper reads the tooltip. A horse with history keeps its history's pedigree, and the tooltip's sire agreed with history's on all 537 horses with both. A horse without history takes:
  - the tooltip's sire, spelled as history spells it;
  - the damsire from the dam's other offspring, or her own sire where she raced;
  - Filly or Mare from "Female" by age, and for "Male" the sex history's debutants of that age and race type most often are.
- **The dry run** (the 958 on 26 Sep's card, before and after, the same 568 runners):
  - sire filled for 31 debutants, damsire for 24, sex for 31;
  - the debutants' prices moved by a mean 0.28 in log terms, the well-bred ones shorter (a Frankel debutant 27.6 to 15.0);
  - the other runners moved by 0.009, and the top pick changed in 1 race of 53.
- **The parity check** (the March days, research query run 36247323034):
  - With the old card fill, the 958 was priced a mean 0.035 in log terms from training (debutants 0.273), with the same top pick in 92.0% of races.
  - With the fix it is 0.014 (debutants 0.075), with the same top pick in 98.7% of races. The 944 goes from 0.031 to 0.012, and the 615 from 0.035 to 0.016.
  - What is left are the card's other gaps: rail moves, jockeys' claims, geldings since the last run, and first foals' damsires.

It changes what the 06:00 job serves for debutants, for any model, so it goes out only with the owner's word. It is in PR #77 with the model.

## What deploying involves (only with the owner's word)

1. Commit the chosen artefact to `data/models` on the branch, with its verify report.
   - For the 958: train-bfsp run 39, artifact `bfsp-model-39`.
   - For the 944: the 7000-round file, run 38, artifact `bfsp-model-38`.
2. Add an amendment to `reports/preregistration_clv_forward.md`. It records the change between days, as the pre-registration requires, the evidence above, and its first 06:00 date. The forward report then gives the 615's days and the new model's days separately, as well as the whole window.
3. Merge PR #77 to the default branch before 06:00 UTC on 27 Sep. The 06:00 job reads `data/models` from the default branch. The only other change to what it serves is the card pedigree fix above, which the amendment records with the model.

## Tested today and not carried

These were each screened against form lines plus connection windows (reports/research_ledger.jsonl):
- **Connection grains** (the connections' last fortnight, race code, course and pairing): −0.0008, unresolved.
- **Connection windows against the field:** +0.0003, unresolved.
- **Sire windows** (the progeny's recent form): +0.0017, resolved worse.
- **Recency weighting of the training rows:** worse for the outsiders.
- **Collateral form through common opponents:** nothing beyond connection windows.

The remaining price error is where public form is thin: maiden, novice and bumper races (0.52 against 0.36 in handicaps), outsiders, and big fields. Ireland's higher error is mostly its greater share of those races.
