# The syndicate papers: what is worth trying in our models

*Read in full on 25 Sep 2026: seven documents, every page, with equations and tables checked against the PDFs. The key numbers quoted here were checked again against the text.*

| # | Document | Setting | What it does |
|---|---|---|---|
| 1 | Bolton & Chapman 1986, *Management Science* 32(8) | 200 US races, tote | Multinomial logit on 10 handicapping variables. Rank-ordered ("exploded") to the first three places. |
| 2 | Benter 1994, *Efficiency of Racetrack Betting Markets* | Hong Kong tote, 3,198 races | A fundamental logit, then a second logit combining it with the public odds. Also ΔR², Harville with corrections, fractional Kelly. |
| 3 | Lessmann, Sung & Johnson 2009, *EJOR* | Goodwood, 556 races, SP | SVM decision value fed into a second-stage conditional logit, with race-wise standardisation. |
| 4 | Sung & Johnson 2007, *J. Prediction Markets* 1(1) | Wolverhampton, 1,675 races, SP | One-step vs two-step conditional logit; exploded logit with a pooling test for the depth. |
| 5 | Lessmann, Sung & Johnson 2007, *JPM* 1(3) | Goodwood, 556 races | LS-SVR stage 1, with hyperparameters chosen by NDCG@3 rather than MSE. |
| 6 | Silverman 2013, UCLA PhD thesis | Hong Kong | Hierarchical Bayesian speed model; L1 conditional logit with a "reverse frailty" market offset. |
| 7 | Leung & Leung 2023, *Acta Machina* (blog) | Hong Kong 1986–2023 | Re-runs Benter's public-odds model; fits Harville corrections per era. |

## What they agree on, and what it means for us

- **Two stages.** First a market-blind fundamental model, then a conditional logit that combines it with the market. Every paper from Benter on uses this, and we already have it (`scripts/outcome_model.py` modes offset/free, `train_stage_f.py`).
- **The market dominates, and our finding is Benter's own case.**
  - His tipster consensus reaches the same stand-alone R² as his nine-factor model (.1014 vs .1016). Against the public, though, it adds a ΔR² of .0002, against .0090 for the factor model.
  - For information like that he writes that the second stage "would always be virtually identical to the public estimate, thus never indicating an advantage bet" (p.190).
  - The fundamentals we have tested against the BSP are in that position: the BSP already holds them.
- **The market has kept learning.** Leung & Leung find the Hong Kong public's pseudo-R² rose from 0.1325 (1986–93) to 0.1863 (2016–23), as Benter predicted (p.196). The BSP is a harder benchmark than any of these papers faced.
- **None of the profit claims carries confidence intervals, and all rest on small holdouts** (156–565 races).
  - Two choose their thresholds on the holdout itself. Bolton & Chapman pick the p_min floor from the hold-out tables. Silverman picks the λ with the best ROI; his text says λ = 7, while his chart peaks near λ ≈ 3, and neighbouring λ values swing ROI by up to 0.2.
  - The returns are therefore not evidence against our null. What transfers is the methods.

## Worth trying, in order of value for effort

1. **Within-race standardisation.** *Testing now: iteration 35, `model/blocks/race_relative.py`.*
   - Source: Lessmann 2009, eq. 14. Race-wise z-scores beside database-wide ones beat either alone.
   - Why it matters here: a BSP is a within-race quantity, but our model sees each runner on its own, with within-race ranks for about 60 metrics. A rank keeps the order and throws away the distance.
2. **A horse-level time figure.** *Next to build.*
   - Speed ratings are Bolton & Chapman's largest standardised effect (AVESPRAT, 0.562). They are also Benter's "normalized times" and Silverman's whole Ch.2.
   - Our RSR is the race's winning time against a lag-safe standard for course × trip × going, so it is the same number for every runner in the race.
   - What's needed: the horse's own time (winning time plus lengths beaten converted to seconds at the trip), a daily going allowance per meeting, and the figure on the pounds scale. Then the same windows as the form block.
3. **Rank-ordered likelihood for the win model.**
   - Bolton & Chapman and Sung & Johnson explode the finishing order to the first two or three places. Standard errors fall 28% from depth 1 to 2, then another 17–19% to depth 3. They set the depth with a pooling test.
   - Why it matters here: a test for information beyond the BSP is short of exactly this kind of power.
   - Implement: a Plackett–Luce loss to depth 2–3, with the market as an offset in the outcome model.
   - Lessmann warns that placings behind the prize money are less reliable, so we would leave out runners whose comment says eased, not pushed or tailed off.
4. **Benter's DP6A preferences.**
   - What it is: distance (or going, or course) preference measured as the horse's residual performance against a baseline, fitted against similarity to today's conditions, and divided by its standard error.
   - Ours are raw means: "mean NFP at the trip". Benter's later version fits a quadratic in log distance with "tack points" as a prior, so it works with fewer than three runs.
5. **ΔR² over the market, and the market's R² by segment.** Two cheap diagnostics, from Benter's eq. 3–4 and Leung & Leung:
   - Report our mnats per race also as Benter's ΔR², so it is comparable with his .009–.018.
   - Map where the BSP is least informative (season, code, class, handicap, field size, price band). That is where fundamentals have the most room.
6. **A free coefficient on the market** (Benter's β; Sung & Johnson's one-step model).
   - This lets the win model correct any favourite–longshot bias in the BSP instead of taking it at face value.
   - The outcome model already has `--mode free`; rerun it on the current features.
7. **The place market.**
   - Method: convert win probabilities to place probabilities with Harville plus Benter's corrections for the lower places (γ .81, δ .65 in Hong Kong; Leung & Leung find them stable over 37 years). Compare with the place BSP.
   - Benter notes that when the public bets consistently across pools, a win overlay is a worse place bet.
   - This is a second market; the effort is moderate.
8. **Staking.**
   - Joint Kelly over every overlay in a race rather than the single best (Benter; Wong, via Leung & Leung).
   - Fractional Kelly at ½ to ⅓ (Benter: full Kelly makes drawdowns over 50% "a common occurrence").
   - Stakes capped by the morning volume. The rule's bets have a median of about £500 matched.

## Not worth trying

- **Silverman's "reverse frailty" offset.** exp(Xβ + 1 − 1/d) equals exp(Xβ − 1/d) within a race, so it is a fixed tilt against favourites and carries no market information. Its 36.7% return is the best of 100 noisy λ values on random, not chronological, splits.
- **SVM or LS-SVR as stage 1.** Their gain was over a linear logit. LightGBM already fits the non-linear interactions.
- **A hierarchical Bayesian speed regression by race profile** (Silverman Ch.2). It lost 12.15% in its own test.
- **Thresholds chosen on the holdout**, such as Bolton & Chapman's p_min.

## Errors found in the papers

- **Lessmann et al. 2007, eq. 13:** the sign is wrong as printed (it gives the winner −0.5).
- **Lessmann et al. 2007, reinvested returns:** "+112.20%" and "+172.48%" match their own log-wealth chart only as 112% and 172% *of* the bank, i.e. +12% and +72%.
- **Lessmann et al. 2009:** Figure 2 and Table 3 disagree for SVR/CL.
- **Silverman:** λ = 7 in the text against the chart's peak at λ ≈ 3.
- **Leung & Leung:** the Z statistics are non-standard (a within-bin SD divided by 25). Recomputed as binomial Z, the long-shot over-prediction is real in the later eras but much weaker than printed.
