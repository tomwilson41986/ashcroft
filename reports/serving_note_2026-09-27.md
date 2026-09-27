# Which model serves next: the recommendation

*Written 26 Sep 2026 for the owner's decision, last updated 27 Sep 03:10 UTC. Nothing changes without the owner's word; without an answer before a 06:00 UTC run, the 615 keeps serving.*

## Recommendation

**Serve the 968s from the first 06:00 run after the owner's word** (train-bfsp run 44, artifact `bfsp-model-44`). It is the 968's features (the 958 plus the debut market block) fitted at the slow recipe: learning rate 0.02 for 10,000 rounds, with leaves of at least 200 runners. It is one booster, committed compressed: 38.6 MB with xz, 123 MB plain, which is over GitHub's 100 MB file limit. The 06:00 job reads the compressed file (commit 673d23c); it decompresses to the verified booster byte for byte and predicts identically. Every check has passed:
- the tests on the served recipe (iterations 81 and 85)
- verify (`reports/model_verify_968s.md`): 968 features, books of 1, log prices correlating 0.9936 with the 615's
- the parity of its features between the 06:00 path and training (the 968's features, checked for the 968; the recipe changes only the fit)
- a dry run on 27 Sep's card (predict-now run 20): 241 runners in 23 races, every one priced, books of 1, 12.5 minutes from load to prices, no warnings

**Against the 958**, the previous recommendation (same 53,910 runners): the price-forecast error falls by −0.0060 (90% CI −0.0069 to −0.0051). Brier skill against the market improves by +0.0014 (+0.0006 to +0.0022) and concordance by +0.0028 (+0.0011 to +0.0042), both resolved better.

**Against the 615 it would replace:** −0.0308 (−0.0325 to −0.0292), in every rank band. Brier skill +0.0064 (+0.0047 to +0.0081) and concordance +0.0057 (+0.0035 to +0.0081), both clear of zero. The early-price rule goes from +7.70% to +9.92%, and the top pick traded out makes +1.43%.

Fallbacks, each verified and dry-run clean: the 958 (run 39, `bfsp-model-39`), then the 7000-round 944 (run 38).

**Deploy it with the card pedigree fix** (found 26 Sep afternoon; see "The 06:00 card's missing pedigree" below). The 06:00 card gave debutants no sire, damsire or sex, so every model so far priced them a mean 0.26 in log terms (about 30%) away from the price its training features give. The fix reads the pedigree the card does carry, and it applies to whichever model serves.

**The next step: two slow fits served as one** (iteration 86, 02:45 UTC). The slow extra-trees 968 and the slow Huber 968, averaged, price the window at 0.4011: −0.0039 (−0.0046 to −0.0032) on the 968s alone. The early-price rule reads +10.34% and the model's weight beside the morning price 0.452, both the highest yet. Both are training for serving (train-bfsp runs 45 and 46). Each then needs its verify, and the pair a dry run. Serving them needs the owner's word on storing two boosters of about 40 MB each (xz).

**The Huber loss alone does not change the recommendation.** On the 958's features at the served recipe it was level with squared error: −0.0003 (−0.0010 to +0.0006), unresolved (iteration 76). At the slow recipe it is −0.0017 on the 968s (iteration 86), within reach of the refit noise. Its value is as a partner (above).

**What the 968s adds to the 958:**
- **the debut market block** (iterations 80 and 81): how the market has priced each yard's debutants and lightly raced runners (`model/blocks/debut_market.py`, 10 features). Beside the 958's features at the served recipe under squared error: −0.0018 (−0.0028 to −0.0008), concordance +0.0021, resolved better. Parity is clean: every debut market feature is identical on the 06:00 card and in training (research query run 36266481019);
- **the slow recipe** (iteration 85): −0.0041 (−0.0048 to −0.0034) on the same features, with Brier skill resolved better. A refit at another seed moves the slow recipe's error by 0.0007 (iteration 86), so the step is six times the refit noise.

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

All at 6000 rounds (the 968s at learning rate 0.02 to a 12,000-round cap, stopping at 9,549 to 10,058), on the development window of 27 Sep 2025 to 31 Mar 2026: 53,910 runners in 5,923 races. The locked holdout was not read.

| model | what it adds | price-forecast error (mean abs log) | step, paired (90% CI) | early-price rule, Jan–Mar 2026 |
|---|---|---|---|---|
| the 615 (served now) | — | 0.4358 | — | +7.70% |
| the 675 | within-race readings | 0.4280 | −0.0078 | +7.65% |
| the 853 | time figure, wide readings, exposure, collateral form | — | −0.0053 (−0.0064 to −0.0041) | +7.59% to +7.83% |
| the 859 | bookings | 0.4197 | −0.0041 (−0.0051 to −0.0032) | +8.09% |
| the 944 | form lines, form variants, less the two sire counts | 0.4156 | −0.0040 (−0.0049 to −0.0031) | +8.44% (+7.17 to +9.82) |
| the 958 | connection windows | 0.4115 | −0.0040 (−0.0050 to −0.0032) | +9.60% (+8.23 to +10.99) |
| the 968 | the debut market block | 0.4091 | −0.0018 (−0.0028 to −0.0008), against iteration 81's 958 | +9.43% |
| **the 968s** | **the slow recipe: learning rate 0.02, leaves of 200, about 10,000 rounds** | **0.4050** | **−0.0041 (−0.0048 to −0.0034)** | **+9.92% (+8.63 to +11.26)** |

- Each row's step is measured against the row above, on the same runners (iterations 36, 42, 48, 61, 68, 81 and 85). The 853 was fitted as a variant only, so the table gives no level for it.
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

**The 968s (train-bfsp run 44):**
- **Verify: PASS** (`reports/model_verify_968s.md`).
  - 968 features, 10,000 rounds at learning rate 0.02, trained through 22 Sep on 697,903 runs.
  - Every drop-in block, the debut market block included, built on the matrix exactly as the 06:00 path builds it.
  - Books of 1. Log prices correlate 0.9936 with the 615's over 9–22 Sep.
- **Parity: the 968's** (research query run 36266481019): every debut market feature is identical on the 06:00 card and in training, and the 968's card prices sit a mean |Δlog| of 0.0122 from training's. The 968s reads the same features through the same path.
- **Dry run on 27 Sep's card (predict-now run 20): clean.**
  - 241 runners in 23 races, every one priced, books of 1, no warnings.
  - 12.5 minutes from load to prices; the limit is 45.
  - Against the extra-trees 968 averaged with its Huber twin on the same card (run 19): correlation 0.9974, the same top pick in all 23 races.
- **The compressed file:** it decompresses to the verified booster byte for byte, and loaded alone it predicts identically (max difference 0.0 on 2,000 rows). It loads in 4.9 seconds.

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

## The slow recipe and its partners (iterations 85 and 86)

Fitted at learning rate 0.02 (to about 10,000 rounds) with leaves of at least 200 runners, one fit of the 968's features prices the development window better than anything served or proposed so far, including the 968 averaged with its Huber twin, and it needs one booster, not two:

| on the window (53,910 runners) | mean absolute log error | against the 968 | early-price rule | rank 1 traded out |
|---|---|---|---|---|
| the 968 (lr 0.03, one fit) | 0.4091 | — | +9.43% | +1.29% |
| the 968 and its Huber twin, averaged | 0.4062 | −0.0029 | | |
| **the 968 at lr 0.02, leaves of 200 (one fit)** | **0.4050** | **−0.0041 (−0.0048 to −0.0034)**, Brier skill resolved better | **+9.92%** | **+1.43%** |
| the same averaged with the extra-trees 968 (lr 0.03) | 0.4034 | −0.0057 | | |
| the extra-trees 968 at the slow recipe (one fit, iteration 86) | 0.4036 | −0.0055; against the slow 968 −0.0014 (−0.0023 to −0.0005), but concordance resolved worse | | |
| **the slow 968 and the slow extra-trees 968, averaged** | **0.4019** | **−0.0072**; against the slow 968 −0.0031 (−0.0035 to −0.0026), Brier skill and concordance level | | |
| the same with a third fit at another seed | 0.4015 | −0.0076; against the slow 968 −0.0035 (−0.0039 to −0.0031) | | |
| the Huber 968 at the slow recipe (one fit) | 0.4033 | −0.0058; against the slow 968 −0.0017 (−0.0024 to −0.0009) | +10.28% | +1.35% |
| **the slow extra-trees 968 and the slow Huber 968, averaged** | **0.4011** | **−0.0080**; against the slow 968 −0.0039 (−0.0046 to −0.0032) | **+10.34%** | +1.50% |
| the slow 968, extra-trees and Huber fits, averaged | 0.4011 | −0.0080 | | |

The 968 at the slow recipe (the 968s, train-bfsp run 44) has since passed verify and its dry run: it is the recommendation above, and it raises no storage question.

Iteration 86 (02:30 UTC, 27 Sep) found its best partner. At the slow recipe, a fit at another seed moves the error by only 0.0007 (at lr 0.03, up to 0.0010), so a second seed adds little; extremely randomised split points add more. Alone, the slow extra-trees fit ranks winners over losers slightly worse (concordance −0.0015, −0.0031 to −0.00002), so it serves only beside the slow 968, where the pair prices the window at 0.4019. Its Huber fit is a better partner still: the slow extra-trees and slow Huber fits average to 0.4011, as good as any three, with the early-price rule at +10.34% and the model's weight beside the morning price at 0.452, both the highest yet. Both are training for serving (train-bfsp runs 45 and 46, 11,000 rounds, not published). The pair would need two boosters, the storage question below.

## An option: several fits served as one

A single fit is one draw of a noise the paired test does not see. Fitted again with the same recipe and seed on other machines, the 958 prices the window with the same error: −0.0005 (−0.0011 to +0.0002). But each runner's price moves by about 7%, three quarters of what another seed moves it (iteration 81). Averaging the prices of several fits removes most of that. Against the single 958 (iteration 81, the same runners):

| served as one | price error (90% CI) | Brier skill | early-price rule | top pick traded out |
|---|---|---|---|---|
| the 958, one fit | — | — | +9.46% (+9.33% to +9.60% across three seeds) | +1.57% (+1.00% to +1.57%) |
| two seeds | −0.0021 (−0.0026 to −0.0017) | +0.0006, resolved | +9.57% | +1.42% |
| **three seeds** | **−0.0026 (−0.0030 to −0.0021)**, every rank band | +0.0003 | **+10.04% (+8.66 to +11.48)** | +1.35% |
| the 958 and its Huber twin | −0.0026 (−0.0030 to −0.0022), against iteration 68's fit | +0.0006, resolved | +9.18% | +1.48% |
| three seeds and the Huber twin | −0.0032 (−0.0037 to −0.0027) | +0.0005 | | |
| **the 968 and its Huber twin** | **−0.0048 (−0.0057 to −0.0039)**, against the single 958 | +0.0006, resolved; concordance +0.0025, resolved | | |

Every average passes the decision rule. The rule's and the top pick's readings move between single seeds by as much as between the averages (the top pick's by half a point), so the price error is the steadier measure. The 958's Huber twin (train-bfsp run 40) is already trained and verified, so the two-loss average could serve with no new training.

It needs two to four boosters of 72–84 MB each. Compressed, a 7000-round booster is 33 MB with gzip or 25 MB with xz, and it loads in 2–3 seconds with identical predictions. The options:
- about 25 MB in git per booster per retrain (xz), or
- S3 beside the database, with a manifest in the repository naming each file and its checksum. The 06:00 job already reads S3.

This is a storage decision for the owner, not needed for 27 Sep.

The serving path is ready and inert until a manifest is committed: `bfsp_ensemble.json` in `data/models`, naming each member's directory, makes the 06:00 job serve the members as one. Each member is priced by its own target's rule, and their prices are averaged as the research loop averages arms (the geometric mean, renormalised per race; `predict_bfsp_today.AveragedBooster`, tested in `tests/test_averaged_serving.py`). `predict-now.yml` dry-runs a candidate averaged with a second model (`twin_run`, `twin_artifact`). The 968's Huber twin (train-bfsp run 42, `bfsp-model-42`) has passed verify: 968 features, the Huber loss, 7000 rounds, books of 1, log prices correlating 0.9931 with the 615's. The pair's dry run on 26 Sep's card is clean (predict-now run 18): the averaged model loaded both members; 566 runners in 53 races, every one priced, books of 1, 17 minutes from load to prices. Against the 968 alone, correlation 0.9990 and the same top pick in 51 of 53 races. The pair is ready to serve, with the owner's word on the change and on storing two boosters.

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
   - For the 968s: train-bfsp run 44, artifact `bfsp-model-44`, as `bfsp_model.lgb.xz` (38.6 MB), deleting the 615's `bfsp_model.lgb` in the same commit. A test refuses the two side by side, since the plain file would be read.
   - For the 958: train-bfsp run 39, artifact `bfsp-model-39`.
   - For the 944: the 7000-round file, run 38, artifact `bfsp-model-38`.
2. Add an amendment to `reports/preregistration_clv_forward.md`. It records the change between days, as the pre-registration requires, the evidence above, and its first 06:00 date. The forward report then gives the 615's days and the new model's days separately, as well as the whole window.
3. Merge PR #77 to the default branch before 06:00 UTC. The 06:00 job reads `data/models` from the default branch. The only other change to what it serves is the card pedigree fix above, which the amendment records with the model.

## Tested today and not carried

These were each screened against form lines plus connection windows (reports/research_ledger.jsonl):
- **Connection grains** (the connections' last fortnight, race code, course and pairing): −0.0008, unresolved.
- **Connection windows against the field:** +0.0003, unresolved.
- **Sire windows** (the progeny's recent form): +0.0017, resolved worse.
- **Recency weighting of the training rows:** worse for the outsiders.
- **Collateral form through common opponents:** nothing beyond connection windows.
- **Whole careers** (iteration 83): each horse's runs before 2021, restored exactly from the database, add nothing against four refits of the 958 (−0.0008 to +0.0002), not even for the runners who have them (+0.0010). Fitting only on rows from 2022 is worse (+0.0009 to +0.0019), so the matrix keeps its 2021 start.
- **The pedigree as the market priced it** (iteration 84, two seeds each beside the 968): −0.0006 (−0.0012 to −0.0000) on the averages, and the early-price rule lower in every reading (+9.26% against +9.66%). The yard already carries what the market pays for in an unknown horse.
- **The conditions in the race name** (auction, sales, EBF, mares', restricted, classified): small biases on a few hundred runners each, worth about 0.0001 in all; not built.

The remaining price error is where public form is thin: maiden, novice and bumper races (0.52 against 0.36 in handicaps), outsiders, and big fields. Ireland's higher error is mostly its greater share of those races.
