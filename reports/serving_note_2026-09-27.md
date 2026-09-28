# Which model serves next: the recommendation

*Written 26 Sep 2026 for the owner's decision, last updated 28 Sep 05:45 UTC. Nothing changes without the owner's word; without an answer before a 06:00 UTC run, the 615 keeps serving.*

## Recommendation

**Serve the pair with the three blocks from the first 06:00 run after the owner's word**: the combined 968s5xh with the Kalman rating, handicap angles and travel (train-bfsp run 52, artifact `bfsp-model-52`) beside the within-race partner with the same blocks (run 53, `bfsp-model-53`), their prices averaged geometrically and renormalised per race. Two boosters, 88.4 MB with xz. Every check has passed:
- the tests (iteration 97; 53,910 runners, 27 Sep 2025 to 31 Mar 2026): **0.3966** mean absolute log error against the BSP, the best of any model trained for serving; against the pair without the blocks (runs 50 + 51, 0.3981) −0.0016 (−0.0021 to −0.0010), concordance resolved better, Brier skill level; the same pair at seed 7 prices 0.3969;
- **against the 615 it would replace: −0.0393 (−0.0411 to −0.0376)**, Brier skill against the market +0.0074 (+0.0054 to +0.0093) and concordance +0.0065 (+0.0042 to +0.0088), both resolved better;
- the bets (research query run 36369570799, which reproduces the pair's own line exactly): the early-price rule **+10.70%** (+9.33 to +12.04; the 615 +7.70%), the top pick backed at the morning price and traded out at BSP **+2.01%** (+1.25 to +2.78), the top pick held to the result −3.07%;
- verify: run 52 PASS (`reports/model_verify_968s5xh_all3.md`) and run 53 PASS (`reports/model_verify_968s5xd_all3.md`): 993 features each, every drop-in block built as the live path builds it, books of 1, log prices correlating 0.9918 and 0.9915 with the 615's; each decompresses from xz byte for byte;
- a dry run as a pair on 28 Sep's card (predict-now run 29): both members loaded, **468 runners in 45 races priced (every runner on the card)**, books of 1, 14 min 37 s from load to prices, peak memory 14.6 GB of the runner's 16 GB. The day's prices: `reports/predictions_2026-09-28_pair_all3.xlsx`.

Serving it needs the owner's word on the model and on storing two boosters (the manifest layout the 06:00 path already reads, as staged for runs 50 + 51). Fallbacks, each verified and dry-run clean: the pair without the blocks (runs 50 + 51, 0.3981), the 968s5xh alone (run 50, 0.4001), the 968s5 (run 47, 0.4029), the 958 (run 39). Deploy any of them with the card pedigree fix (below).

**Complete careers: the next step, which needs swap at 06:00** (iterations 99, 100 and 102). Every model so far builds its features on the history from 2021, so each horse's, yard's and rider's record starts there. Built on the history from 2018 and fitted on the same rows from 2021, the 968s5xh with the three blocks prices the window at **0.3968 against 0.3984, −0.0015 (−0.0023 to −0.0007)** (iteration 100, in single precision, which costs nothing: −0.0004 against double precision on the same rows, iteration 99). Beside the current partner (its careers still cut at 2021) the pair prices **0.3946**, −0.0019 (−0.0023 to −0.0015) on the recommended pair's 0.3966, Brier skill level, concordance −0.0010 (−0.0020 to +0.0000). Its bets, though, are level with the recommended pair's (the rule +10.81% against +10.70%, paired +0.11, −0.25 to +0.46; the top pick +0.03): the longer history sharpens the price more than the trade. Fitting the 2018-2020 rows as well adds nothing resolvable (iteration 100: −0.0005, −0.0011 to +0.0002), so rows from 2021 stay. Iteration 102 (research-loop run 36381991232) puts the partner on the same history and scores the pair with its bets, and iteration 103 (run 36384069890) the race_xent partner, which trades better (below); both members are training for serving meanwhile (train-bfsp runs 54 and 55, train only). Serving them means the 06:00 job builds 1.05m rows of history instead of 0.70m: without swap the runner runs out of memory four minutes into the engine (predict-now run 27); with 16 GB of swap it priced the card in 26 minutes against the job's 45-minute limit (predict-now run 28). So the 06:00 job would need the dry run's opt-in swap step: a change to `predict.yml`, for the owner's word with the model.

## How the recommendation got here (26-28 Sep)

Until 28 Sep 05:30 UTC the recommendation was the 968s5xh alone; the record of that, and of each step since, follows.


**The 968s5xh** (the recommendation until the pair below was verified and dry-run) (train-bfsp run 50, artifact `bfsp-model-50`). It is the 968's features (the 958 plus the debut market block) at the slow recipe with extremely randomised split points under the Huber loss: learning rate 0.02 for 12,000 rounds, leaves of at least 500 runners, `extra_trees`, the Huber loss. It is one booster, committed compressed as `bfsp_model.lgb.xz` (44.2 MB with xz; the 06:00 job reads a lone compressed booster, commit 673d23c); the file decompresses to the verified booster byte for byte. Every check has passed:
- the tests (iterations 91 and 93): it prices the development window at 0.4001, against 0.4029 for the 968s5 (the same features and recipe without extra trees or the Huber loss), −0.0028 (−0.0037 to −0.0019), four times the slow recipe's refit noise, with neither Brier skill nor concordance worse; the same fit at another seed prices it at 0.4005; the early-price rule +10.16%, the top pick traded out +1.78%;
- verify (`reports/model_verify_968s5xh.md`): PASS, 968 features, books of 1, log prices correlating 0.9919 with the 615's;
- the parity of its features between the 06:00 path and training (the 968's features, checked for the 968; the recipe changes only the fit);
- a dry run on 27 Sep's card (predict-now run 25): 217 runners in 23 races, every one priced, books of 1, 16 minutes from load to prices, the engine's usual warnings only; against the 968s5 on the same card, correlation 0.9904 and the same top pick in 19 of 23 races; against the pair at leaves of 500, 0.9972 and 21 of 23.

**Against the 958**, the recommendation of 26 Sep (same 53,910 runners): the price-forecast error falls by −0.0109 (90% CI −0.0120 to −0.0099). Brier skill against the market improves by +0.0022 (+0.0011 to +0.0032) and concordance by +0.0030 (+0.0010 to +0.0048), both resolved better.

**Against the 615 it would replace:** −0.0358 (−0.0375 to −0.0341), Brier skill +0.0072 (+0.0055 to +0.0089) and concordance +0.0059 (+0.0035 to +0.0084), both clear of zero. The early-price rule goes from +7.70% to +10.16%, and the top pick traded out makes +1.78%.

Fallbacks, each verified and dry-run clean: the 968s5 (run 47, 0.4029), the 968s (run 44, leaves of 200), the 958 (run 39), then the 7000-round 944 (run 38).

**Deploy it with the card pedigree fix** (found 26 Sep afternoon; see "The 06:00 card's missing pedigree" below). The 06:00 card gave debutants no sire, damsire or sex, so every model so far priced them a mean 0.26 in log terms (about 30%) away from the price its training features give. The fix reads the pedigree the card does carry, and it applies to whichever model serves.

**The next step: the 968s5xh with the within-race partner** (iterations 92 and 94), trained, verified and dry-run as a pair. The within-race extra-trees fit (train-bfsp run 51, `bfsp-model-51`: the 968's features, the within-race demeaned log price as the target, 11,500 rounds) prices the window worse alone (0.4027) but errs differently from every logit fit; served beside the 968s5xh the two price it at **0.3981**, the best measured model:
- against the 968s5xh alone −0.0019 (−0.0025 to −0.0014), against the pair at leaves of 500 −0.0013 (−0.0020 to −0.0006), against the 968s5 −0.0047 (−0.0056 to −0.0039);
- against the 958 −0.0129 (−0.0140 to −0.0118) and against the 615 −0.0377 (−0.0395 to −0.0360), with Brier skill and concordance resolved better against both;
- scored in one run (iteration 94): the early-price rule +10.77%, the top pick traded out +1.72%, and the model's weight beside the morning price 0.462, the highest yet;
- the partner's verify (`reports/model_verify_968s5xd.md`): PASS, 968 features, books of 1, log prices correlating 0.9913 with the 615's; 42.4 MB with xz, decompressing byte for byte. The within-race target is adopted for serving as an average's member (`SERVABLE_TARGETS`, commit 49d236f);
- staged as a pair (the 968s5xh at `data/models`, the partner at `data/models/members/dml`, a manifest; 86.5 MB with xz), the 06:00 path loads it as one in 12 s;
- its dry run on 27 Sep's card (predict-now run 26): both members loaded, 217 runners in 23 races priced, books of 1, 17 minutes, the engine's usual warnings only; against the 968s5xh alone correlation 0.9985 and the same top pick in 22 of 23 races.

The Huber within-race partner prices the pair the same (0.3981; the rule +10.96%, the top pick traded out +1.55%), so it is not trained. **A third member is not worth serving**: the Huber fit at leaves of 500 (run 49, verified) takes the pair's price to 0.3974 offline, but its bets do not follow (research query run 36346258544, which reproduces the pair's own line exactly): the rule falls from +10.77% to +10.49% and the top pick traded out from +1.72% to +1.66%, the weight beside the morning price unchanged at 0.462; four members are level with the pair (+10.74%, +1.66%). Unequal weights for the pair gain 0.0002. Other within-race partners do no better (iteration 95): at leaves of 1,000 the pair prices 0.3980, without extra trees 0.3979, under Huber 0.3982, none resolved, so the trained partner (run 51) stays. Iteration 96's new blocks take it further (below).

Behind it, the pair at leaves of 500 (iteration 89: the extra-trees and Huber fits at leaves of 500, train-bfsp runs 48 and 49) prices the window at 0.3995, within 0.0006 of the 968s5xh alone, and is trained, verified and dry-run as a pair (predict-now run 23; 78.7 MB with xz). The pair at leaves of 200 (runs 45 and 46) stands behind that.

Serving any pair needs the owner's word on the model and on storing two boosters (a manifest in `data/models`).

**How the 968s5xh was found** (iteration 91). The slow recipe's extra-trees fit at leaves of 500 holds at another seed (0.4015 against 0.4017; run 48 alone dry-runs clean, predict-now run 24); at leaves of 1,000 it prices 0.4006; under the Huber loss 0.4001, against xt500 −0.0016 (−0.0022 to −0.0010), Brier skill resolved better. Its folds reached the 12,000-round cap without stopping; to a 16,000 cap they stop near 13,000 rounds, worth −0.0005 (iteration 93), within the refit noise: noted for its next retrain. Averages of the logit fits reach 0.3984 to 0.3989; the within-race partner takes the pair further.

**Leaves of 1,000** are no better than 500 on their own (iteration 89: −0.0007, the refit noise's size); with extra trees −0.0011 (−0.0017 to −0.0004), 0.4006, at the noise's edge (iteration 91). A slower learning rate (0.015) and other tree sizes (255 or 63 leaves) add nothing (iteration 88).

**Travel carries** (iteration 96; the three new blocks fitted on the 968s5xh's recipe, against it at 0.4001). A screen against the pair's own errors (research query run 36346258544) had read a Kalman rating on the performance figure and handicap angles strongest and travel, how far the yard sends a horse, weakest; the fits said the opposite:
- travel: 0.3989, −0.0012 (−0.0018 to −0.0005), in maidens, novices and bumpers −0.0025, in both countries; Brier skill and concordance not resolved worse; the rule +10.05%, the top pick traded out +1.56%;
- travel with the Kalman rating and handicap angles: **0.3984, −0.0017 (−0.0024 to −0.0011)**, 2.4 times the refit noise, Brier skill and concordance level; the rule +10.16% (the 968s5xh's), the top pick traded out +1.57% (the 968s5xh's +1.78%, within a fit's spread);
- the Kalman rating and handicap angles alone: −0.0002, Brier skill resolved worse;
- beside the trained within-race partner, which reads none of them, the pairs price 0.3975 and **0.3970** (against 0.3981), the best measured model, Brier skill and concordance level.

The blocks are built strictly from earlier days and identical at 06:00 and in training (research query run 36347217357).

**The within-race partner reads them too: the next pair** (iteration 97, research-loop run 36358454411):
- the partner with travel, the Kalman rating and handicap angles prices the window at 0.4013 alone against the trained partner's 0.4027, −0.0014 (−0.0021 to −0.0007), concordance resolved better; with travel alone 0.4018;
- the combined 968s5xh at another seed (7) prices 0.3990, −0.0011 (−0.0017 to −0.0004) on the 968s5xh (seed 42: −0.0017), so the blocks hold;
- **the 968s5xh with the three blocks beside the partner with them: 0.3966**, −0.0016 (−0.0021 to −0.0010) on the pair's 0.3981, 2.3 times the refit noise, concordance +0.0015 resolved better, Brier skill level; the best measured model;
- **its bets hold** (research query run 36369570799, which reproduces the pair's own line exactly): the early-price rule +10.70% (the pair's +10.77%), the top pick traded out **+2.01%** (+1.72%), held at the morning price −3.07% (−3.43%), the model's weight beside the morning price 0.464 (0.462). The same pair at seed 7, in one run: 0.3969, +10.73%, +1.88%.

Both members are in training for serving. The 968s5xh with the three blocks is done and verified (train-bfsp run 52: PASS, 993 features, every block built as the live path builds it, 44.2 MB with xz; reports/model_verify_968s5xh_all3.md); the partner with them trains as run 53 (12,000 rounds, about 05:30 UTC). When it is done it is verified, the two are staged as a pair and dry-run on the day's card, and the pair replaces the 968s5xh and run 51 as the recommendation, with the same storage question (two boosters, about 88 MB with xz). Until then the recommendation stands. Weighting the pair's members unequally gains nothing: the best weight, chosen on two folds and scored on the third, is 0.6 on the main and gains 0.00015, a fifth of the refit noise.

**Two more sources: not carried** (iteration 98, research-loop run 36361679525). How the bookmakers and the place market priced each horse, yard and rider before (the industry SP and the Betfair place SP of past runs, which no feature read) prices the window at 0.3990 on the combined 968s5xh against 0.3984 without it, +0.0006 (+0.0000 to +0.0012); with the same months of earlier years as well, 0.3983, level. Beside the partner the pairs price 0.3968 and 0.3964 against 0.3966, the second with its concordance resolved worse. The bet lines move by 0.1-0.2 points, inside their noise. Both blocks are retired. (The residual screen that chose the last blocks cannot rank candidates, so from here the fits decide.)

**A within-race loss: worse on the price, better on the trade** (iteration 101, research-loop run 36371702086; research queries 36383360553 and 36385429999). Squared error on log prices counts every runner alike, so the long shots, most of any field, carry most of the fit. A new target, race_xent, fits the market's within-race probabilities by their cross-entropy instead, so a runner counts for its share of the race. Fitted on the partner's recipe with the three blocks, it is **better for the runners the market fancies** (against the recommended pair, the pair's top three in each race are each priced 0.0087 closer, rank 1 −0.0106 to −0.0067) and **on the win probabilities** (Brier skill against the market +0.0022, +0.0009 to +0.0036), and **much worse for the long shots** (ranks 8 and beyond +0.0488), which dominate the mean log error the decision rule reads first: 0.4092 against 0.3966 overall. The trade follows the fancied runners. A paired race bootstrap (races resampled together for both models) against the recommended pair's early-price rule (+10.70%):

| served as | price error | the rule | vs the pair (90% CI) | units per 1,000 races | top pick traded out |
|---|---|---|---|---|---|
| the recommended pair | 0.3966 | +10.70% | — | — | +2.01% |
| the pair and race_xent (three members) | 0.3969 | +11.14% | +0.44 (+0.12 to +0.75) | +9.6 | +2.14% |
| the main and race_xent | 0.3987 | +11.09% | +0.38 (−0.03 to +0.81) | +10.8 | +2.07% |
| the demeaned-log partner and race_xent | 0.4001 | +11.46% | +0.75 (+0.32 to +1.22) | +17.0 | +2.37% |
| race_xent alone | 0.4092 | +11.90% | **+1.20 (+0.64 to +1.77)** | **+31.0** | +2.23% |

The rule's gain is in the morning-price bands 1-4, 4-8 and 8-16 (about 2 points in each), the 16+ band level. So which model serves is a choice between the price of every runner and the trade: **the three members keep the price level and trade better; race_xent alone trades best of anything measured and gives up the long shots' prices** (the prices the early-price rule rarely backs, but the workbook shows). race_xent is not servable yet (one line and a test, ready); iteration 103 (run 36384069890) fits it on the complete-careers history, and iteration 104 will test squared error weighted by each runner's share of the race, which may hold both, with race_xent at another seed.

**More training rows add nothing; complete careers carry the gain** (iterations 99 and 100). On the matrix built from 2018, the combined 968s5xh fitted on every row from 2018 prices the window at 0.3964, against 0.3968 fitted on rows from 2021 only: −0.0005 (−0.0011 to +0.0002), inside the refit noise, the bets level (the rule +10.41% against +10.33%), for 40-60% more fitting time. Both beat iteration 96's arm, whose careers were cut at 2021 (0.3984): the features carry the gain, not the rows. Single precision costs nothing (−0.0004 against double precision on the same rows). But the bets do not follow the price here: the complete-careers main beside the current partner (0.3946 on the price) trades level with the recommended pair (the rule +10.81% against +10.70%, +0.11, −0.25 to +0.46; the top pick +0.03). Iteration 102 fits the partner on the same history too; both members are trained for serving meanwhile (train-bfsp runs 54 and 55).

**Future form** (the owner's ask of 27 Sep: `model/blocks/future_form.py`, 64 features) **is built, checked and not carried** (iteration 90):
- its parity at 06:00 is clean: every feature identical on the card and in training (research query run 36301993859);
- read cell by cell against the best single fit's miss, it explains −0.0013 jointly, most in the rivals' later finishing positions from the last five races; deeper windows add nothing (research query runs 36303528074 and 36303839023);
- at the recipe that serves (learning rate 0.02, leaves of 500) it moves the price error by −0.0003 (−0.0011 to +0.0004), no more than a refit; the early-price rule +10.34% against +10.03%, within a fit's spread; the top pick traded out lower (+1.13% against +1.55%). At the faster recipe it was −0.0018 and −0.0006 against two refits of the 968;
- it does not stand in for form lines: without them concordance is resolved worse;
- the fitted model already holds what it knows, through form lines, form variants and connection windows. As a third member beside the pair at leaves of 500 it is worth −0.0006 (0.3989 against 0.3995).

**The Huber loss alone does not change the recommendation.** On the 958's features at the served recipe it was level with squared error: −0.0003 (−0.0010 to +0.0006), unresolved (iteration 76). At the slow recipe it is −0.0017 on the 968s (iteration 86), within reach of the refit noise. Its value is as a partner (above).

**What the 968s5xh adds to the 958:**
- **the debut market block** (iterations 80 and 81): how the market has priced each yard's debutants and lightly raced runners (`model/blocks/debut_market.py`, 10 features). Beside the 958's features at the served recipe under squared error: −0.0018 (−0.0028 to −0.0008), concordance +0.0021, resolved better. Parity is clean: every debut market feature is identical on the 06:00 card and in training (research query run 36266481019);
- **the slow recipe** (iteration 85): −0.0041 (−0.0048 to −0.0034) on the same features, with Brier skill resolved better. A refit at another seed moves the slow recipe's error by 0.0007 (iteration 86), so the step is six times the refit noise.
- **leaves of at least 500 runners** (iteration 87): −0.0021 (−0.0028 to −0.0014) on the slow recipe's leaves of 200, three times the refit noise.
- **extremely randomised trees under the Huber loss** (iteration 91): −0.0028 (−0.0037 to −0.0019) on leaves of 500 alone, four times the refit noise; at another seed 0.4005 against 0.4001 (iteration 93).

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
| the 968s | the slow recipe: learning rate 0.02, leaves of 200, about 10,000 rounds | 0.4050 | −0.0041 (−0.0048 to −0.0034) | +9.92% (+8.63 to +11.26) |
| the 968s5 | leaves of 500 | 0.4029 | −0.0021 (−0.0028 to −0.0014) | +10.03% (+8.73 to +11.33) |
| **the 968s5xh** | **extra trees, the Huber loss, 12,000 rounds** | **0.4001** | **−0.0028 (−0.0037 to −0.0019)** | **+10.16%** |

- Each row's step is measured against the row above, on the same runners (iterations 36, 42, 48, 61, 68, 81, 85, 87 and 91). The 853 was fitted as a variant only, so the table gives no level for it.
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

**The 968s5xh (train-bfsp run 50):**
- **Verify: PASS** (`reports/model_verify_968s5xh.md`).
  - 968 features, 12,000 rounds at learning rate 0.02 with leaves of at least 500 runners, extremely randomised split points and the Huber loss, trained through 22 Sep on 697,903 runs.
  - Every drop-in block built on the matrix exactly as the 06:00 path builds it. Books of 1. Log prices correlate 0.9919 with the 615's over 9–22 Sep.
- **Parity: the 968's**, as for the 968s5: the same features through the same path.
- **Dry run on 27 Sep's card (predict-now run 25): clean.** 217 runners in 23 races (the card fetched at 16:29), every one priced, books of 1, the engine's usual warnings only; 16 minutes from load to prices. Against the 968s5 on the same card: correlation 0.9904, the same top pick in 19 of 23 races; against the pair at leaves of 500: 0.9972, 21 of 23.
- **The compressed file:** 44.2 MB with xz (150.4 MB plain); it decompresses to the verified booster byte for byte.

**The 968s5 (train-bfsp run 47):**
- **Verify: PASS** (`reports/model_verify_968s5.md`).
  - 968 features, 10,500 rounds at learning rate 0.02 with leaves of at least 500 runners, trained through 22 Sep on 697,903 runs.
  - Every drop-in block built on the matrix exactly as the 06:00 path builds it. Books of 1. Log prices correlate 0.9932 with the 615's over 9–22 Sep.
- **Parity: the 968's**, as for the 968s: the same features through the same path.
- **Dry run on 27 Sep's card (predict-now run 22): clean.** 239 runners in 23 races (the card fetched at 08:14), every one priced, books of 1, no warnings; 10 minutes from load to prices. Against the 968s on the same card: correlation 0.9969, the same top pick in 20 of 23 races; against the slow pair at leaves of 200: 0.9976, 20 of 23.
- **The compressed file:** 38.2 MB with xz; it decompresses to the verified booster byte for byte.

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
| the 968 at lr 0.02, leaves of 500 (one fit, iteration 87: the 968s5) | 0.4029 | −0.0062; against the slow 968 −0.0021 (−0.0028 to −0.0014) | +10.03% | +1.55% |
| leaves of 1,000 (iteration 89) | 0.4022 | −0.0069; against leaves of 500 −0.0007, the refit noise | +10.36% | +1.19% |
| the extra-trees fit at leaves of 500 | 0.4017 | −0.0074 | +10.17% | +2.17% |
| the Huber fit at leaves of 500 | 0.4020 | −0.0071 | +9.94% | +1.32% |
| **the extra-trees and Huber fits at leaves of 500, averaged** | **0.3995** | **−0.0096**; against the pair at leaves of 200 −0.0016 (−0.0020 to −0.0012) | **+10.44%** | +1.63% |
| the extra-trees fit at leaves of 500, seed 7 (iteration 91) | 0.4015 | −0.0076; against seed 42 −0.0002 (−0.0009 to +0.0005), the refit noise | +10.06% | +1.76% |
| the extra-trees fit at leaves of 1,000 | 0.4006 | −0.0085; against leaves of 500 −0.0011 (−0.0017 to −0.0004) | +10.49% | +1.92% |
| **the extra-trees Huber fit at leaves of 500 (one fit)** | **0.4001** | **−0.0090**; against the 968s5 −0.0028 (−0.0037 to −0.0019), Brier and concordance not worse | **+10.16%** | **+1.78%** |
| the extra-trees Huber fit with the Huber fit at leaves of 500 (run 49), averaged | 0.3989 | −0.0102; against the pair at leaves of 500 −0.0006 (−0.0009 to −0.0003) | | |
| the pair at leaves of 500 with the extra-trees Huber fit (three members) | 0.3986 | −0.0105; against the pair −0.0009 (−0.0011 to −0.0007) | | |

The 968 at the slow recipe (the 968s, train-bfsp run 44) passed verify and its dry run; with leaves of 500 (the 968s5, run 47) it was the recommendation until the 968s5xh (run 50), and none of the three raises a storage question.

Iteration 86 (02:30 UTC, 27 Sep) found its best partner. At the slow recipe, a fit at another seed moves the error by only 0.0007 (at lr 0.03, up to 0.0010), so a second seed adds little; extremely randomised split points add more. Alone, the slow extra-trees fit ranks winners over losers slightly worse (concordance −0.0015, −0.0031 to −0.00002), so it serves only beside the slow 968, where the pair prices the window at 0.4019. Its Huber fit is a better partner still: the slow extra-trees and slow Huber fits average to 0.4011, as good as any three, with the early-price rule at +10.34% and the model's weight beside the morning price at 0.452, both the highest yet. Both are trained, verified and dry-run as a pair (train-bfsp runs 45 and 46, predict-now run 21; see the recommendation above). The pair needs two boosters, the storage question below.

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
   - For the pair with the three blocks (the recommendation): run 52's booster as `data/models/bfsp_model.lgb.xz` with its meta file, run 53's as `data/models/members/dml/bfsp_model.lgb.xz` with its meta file, and `data/models/bfsp_ensemble.json` naming the two members (88.4 MB with xz; dry-run as a pair, predict-now run 29), deleting the 615's `bfsp_model.lgb` in the same commit. The history the 06:00 path builds stays from 2021 (both members' training summaries say so).
   - For the 968s5xh alone: train-bfsp run 50, artifact `bfsp-model-50`, as `bfsp_model.lgb.xz` (44.2 MB), with its meta file, deleting the 615's `bfsp_model.lgb` in the same commit. A test refuses the two side by side, since the plain file would be read.
   - For the pair of the 968s5xh and the within-race partner: run 50's booster as `data/models/bfsp_model.lgb.xz` with its meta file, run 51's as `data/models/members/dml/bfsp_model.lgb.xz` with its meta file, and `data/models/bfsp_ensemble.json` naming the two members (86.5 MB with xz; staged and dry-run as a pair, predict-now run 26), deleting the 615's `bfsp_model.lgb` in the same commit.
   - For the 968s5: train-bfsp run 47, artifact `bfsp-model-47`, as `bfsp_model.lgb.xz` (38.2 MB), the same way.
   - For the 968s: train-bfsp run 44, artifact `bfsp-model-44`, as `bfsp_model.lgb.xz` (38.6 MB), the same way.
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
