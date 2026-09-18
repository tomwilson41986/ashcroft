# Market-blind staking: Kelly and per-race ranks

*This report now carries figures from two runs, and every table says which.*

- **Run 10** — 2024-05-31 to 2026-03-21, 215,760 runners, 23,191 races, 22 folds. The
  staking work in sections 1-5.
- **R1** (run 13, `prod_faithful`) — 2025-04-26 to 2026-03-21, 108,711 runners, 11,839
  races, 11 folds, trained from 2021-01-01, on the current recipe: plain L2 on log(BFSP)
  over every runner, no recency weighting, early stopping on a holdout that is never the
  scored fold, folds purged 30 days, and the price race-normalised to a book of exactly 1
  with no calibrator in the path. Sections 6 onward.

**Both predate the current recipe.** The six-variant head-to-head has since moved the
training target from `log_bfsp` to `logit_norm_prob` (`reports/h2h_summary.md`: paired
−0.0016 over 107,047 runners, confirmed in both halves of the window). The difference is
small — 0.35% of the error, concentrated in the tail — but it is a different model from
the one measured below, and nothing here has been re-run on it. Read every figure in this
report as describing the `log_bfsp` recipe, and expect the tail of the field to price
slightly better than these numbers say once the model is retrained.

They are different windows as well as different recipes, so figures do not transfer
across that boundary — which is exactly the mistake this rewrite exists to stop. Every
figure is net of 5% Betfair commission. Reproduce with
`python research_lab.py stake --predictions <the run's csv>`.

**R1 against the market**, for orientation before the detail: model log loss 0.2964
against the market's 0.2833, Brier 0.0867 against 0.0833, Brier skill **−0.0403**,
within-race concordance **0.6589** against the market's **0.6799**, expected calibration
error 0.0024 against 0.0022. The same story run 10 told: calibrated about as well as the
market, and separating races less well. The model's top pick returns **−2.84%**
(90% CI −5.42 to −0.31) on 11,839 bets at an average BSP of 4.35.

I predicted these would come in *worse* than run 10's −4.03%, on the grounds that early
stopping no longer happens on the scored fold and the folds are purged. They came in
better. That prediction was wrong, and the comparison it rested on was not sound anyway:
different period, different training window, half the sample. The two numbers should not
be set against each other at all.

## 0. The previous version of this report was measuring a leak

It claimed the model's top pick returned **+17.4%** at level stakes, t = +15.8, stable
across four time slices. That was not a strategy. Thirty-six of 504 deployed features
were reading the race they were predicting — among them `win_surprise`, which is `won`
multiplied by the log of the price. A guard that reverses the finishing order of a race
and rebuilds every feature now runs in the test suite and reports zero.

The same evaluation, same period, same 22 folds, same 215,760 rows, on the fixed code:

| model's top pick | leaky | fixed |
|---|---|---|
| win rate | 39.33% | **30.01%** |
| flat ROI | **+17.4%** | **−4.03%** |
| t-statistic | +15.84 | −3.41 |
| log R² on log(BFSP) | 0.8835 | 0.8028 |
| correlation | 0.940 | 0.896 |
| agreement with the favourite | 82.0% | 57.7% |

The cell that gave it away is the one where model and market disagree hardest. Take
every race's market favourite and split by where the model ranked it:

| market favourite, model ranked it | leaky | fixed |
|---|---|---|
| 1st too | 40.8% win, +11.4% ROI | 38.7% win, −2.8% ROI |
| 2nd or 3rd | 7.7% win, −67.5% ROI | 31.2% win, −2.5% ROI |
| **4th or worse** | **1.1% win, −96.2% ROI** | **23.6% win, −6.8% ROI** |

Favourites win about a third of their races. A model that could find the 1-in-94
favourite would be worth more than this one is. The middle column was the leak reading
the result and pushing losers down the order; the right column is a model that mildly
disagrees with the market and is mildly wrong when it does.

**Read the rest of this report as the honest picture. The numbers are worse and they
are real.**

## 1. What "without the market" can mean, and what it cannot

A Kelly stake is a function of two numbers: the probability you believe and the price
someone offers. Take the market out of the price as well as out of the probability and
the model prices its own bets at its own probabilities — implied odds equal to 1/p, edge
identically zero, stake zero. There is nothing to measure.

So the market is removed from everything it *can* be removed from: the selection, the
probability, and every filter. It supplies only the price the bet settles at, BSP net of
5% commission. Nothing below conditions on a price that is unknown when the bet is
struck; where a price filter appears it is the model's *own forecast* price, not the BSP.

## 2. The model is now worse than the market, and says so consistently

| | model | market (BSP) |
|---|---|---|
| log loss | 0.29583 | 0.28117 |
| Brier | 0.08618 | 0.08256 |
| **Brier skill vs market** | **−0.0439** | — |
| resolution | 0.00881 | 0.01184 |
| expected calibration error | 0.00217 | 0.00151 |
| **concordance (c-index)** | **0.657** | **0.682** |

Every line says the same thing: the market orders races better than we do and separates
them better than we do. That is the expected result for a model whose target *is* the
market's price, and the point worth making is that it is now **internally consistent**.
The leaky version claimed a concordance of 0.673 — still below the market's 0.682 —
while also claiming its top pick beat the favourite by twenty points. Both cannot be
true. Only one of them was.

Calibration is the one thing the model does well: the reliability table's largest gap
across ten equal bins is 0.005, and ECE of 0.0022 against the market's 0.0015 is close
to a dead heat. The model knows how sure it is. It just knows less than the market does.

## 3. Kelly

Full Kelly backs 119,802 of 215,760 runners — 55.5% of every field — at an average 3.47%
of bank. It claims a median edge of +48.5% per unit staked. It realises **−9.05%**.

| Kelly fraction | turnover (bank multiples) | ROI on turnover | log₁₀ final bank | races to halve | max drawdown |
|---|---|---|---|---|---|
| 1 | 4,152 | −6.97% | −604.3 | 5 | 100% |
| ½ | 2,076 | −6.97% | −201.7 | 70 | 100% |
| ¼ | 1,038 | −6.97% | −70.8 | 90 | 100% |
| ⅛ | 519 | −6.97% | −26.5 | 737 | 100% |
| 1/20 | 208 | −6.97% | −8.1 | 1,061 | 100% |

The promised log growth over the period is +1,541. The realised figure is **−1,392**.
That gap is the whole story of staking on a model with no edge: Kelly sizes on believed
edge, and believing a +48.5% edge that is really −9% turns a small per-bet loss into
certain ruin. Every fraction of Kelly, down to a twentieth, ends at a 100% drawdown.
Fractional Kelly slows the bleeding — it does not change the sign.

Flattening the probabilities towards 1/N (p ∝ p^λ) does not rescue it either: λ = 0 still
ends at 10^−350.

## 4. Per-race ranks

| model rank | n | win rate | avg BSP | flat ROI | 90% CI |
|---|---|---|---|---|---|
| 1 | 23,191 | 30.02% | 4.43 | −4.03% | −5.97% to −2.13% |
| 2 | 23,191 | 19.93% | 6.87 | −3.69% | −6.06% to −1.29% |
| 3 | 23,147 | 14.16% | 10.62 | −7.56% | −10.73% to −4.60% |
| 4 | 22,866 | 11.03% | 18.22 | −3.82% | −8.13% to +0.02% |
| 5 | 21,992 | 8.09% | 32.12 | −7.66% | −12.20% to −2.73% |
| 6 | 20,345 | 6.67% | 47.24 | −5.24% | −10.86% to +0.21% |

The ordering works — 30%, 20%, 14%, 11%, 8%, 7% is a monotone ladder, and the top three
win 64% of races between them. It is the *prices* that do not work. Every rank loses, and
the losses cluster around the 5% commission plus a couple of points. The model ranks
horses roughly as the market does and then pays the commission for the privilege.

Backing the top k in every race is negative at every k: −4.03%, −3.86%, −5.09%, −4.77%.

## 5. Where it is least bad

Four slices survive with a confidence interval that crosses zero. None of them clears it.

| slice | n | win rate | ROI | 90% CI | t |
|---|---|---|---|---|---|
| model #1, field ≥ 16 | 1,183 | 22.1% | **+6.53%** | −5.06% to +18.74% | +0.88 |
| model #1, forecast ≥ 8.0 | 520 | 13.1% | +0.78% | −20.12% to +24.17% | +0.06 |
| model #1, field ≥ 12 | 5,449 | 23.2% | −1.12% | −5.68% to +3.97% | −0.37 |
| model #1, field ≥ 12, forecast ≥ 6 | 1,868 | 15.7% | −0.91% | −10.45% to +8.91% | −0.15 |

The pattern is consistent and plausible: big fields and longer prices, where the market
is thinner and our disagreement with it is worth more. It is also where the sample is
smallest. The field ≥ 16 slice earns its +6.5% almost entirely in its final quarter
(+35.9%, 295 bets) after three quarters of −10.3%, −0.5% and +1.1%. That is one good
half-year, not an edge.

Treat these as the places to look next, not as strategies. The honest summary of this
table is *nothing here is significant*.

## 6. The price forecast, which is the actual objective

*This section is rewritten on **R1** (run 13, `prod_faithful`): the production-faithful
walk-forward on the current recipe — 108,711 runners, 11,839 races, 11 folds,
2025-04-26 to 2026-03-21, trained from 2021-01-01. Sections 1–5 above still describe
**run 10** (215,760 runners, 23,191 races, 22 folds, from 2024-05-31). The two are
different windows as well as different recipes, so figures are not comparable across the
boundary and each table below says which run it is.*

Forecasting BFSP well enough to get money on earlier at a better price is what the model
is for, so this is the section that matters.

| model rank (R1) | median forecast / actual BSP | forecast bias | mean abs log error |
|---|---|---|---|
| 1 | 0.9612 | −3.88% | 0.286 |
| 2 | 0.9371 | −6.29% | 0.342 |
| 3 | 0.9189 | −8.11% | 0.395 |
| 4 | 0.9004 | −9.96% | 0.433 |
| 5 | 0.8884 | −11.16% | 0.464 |
| 6 | 0.8808 | −11.92% | 0.494 |

**The column is called `forecast_bias_pct`, and the previous name was wrong.** It was
`clv_at_forecast_pct`, and this report read it as closing-line value. It is
`median(forecast / BSP − 1)` — a forecast against the close. A forecast is not a price
anyone offered, so reading it as CLV upgrades "the model is unbiased" into "we are
beating the close", which is a different and much stronger claim. No early price enters
this table.

**A correction to the previous version of this section.** It reported this quantity as
+0.02% at rank 1 and within a percent at every rank, and concluded the calibrator "has no
work to do". Both statements were about the wrong column. In run 10 `predicted_bfsp` was
the *calibrator's* output — the calibrator overwrote the race-normalised price — so a
near-zero rank bias was the calibrator doing precisely the job it was built for, not
evidence that the job was unnecessary. R1 serves the race-normalised price with the
calibrator out of the path, and the bias it was correcting is visible again: −3.9% at the
top pick widening to −11.9% by rank 6.

Two things are worth separating there, because they pull in different directions.

*The within-race shape is close to right.* Measured on the raw booster output — the clean
comparison the old calibrated-against-calibrated reading could not give — the within-race
slope is **1.0275** against a nominal 1.0. There is essentially no compression inside a
race for a calibrator to remove, which is what justified taking it out of the serving
path.

*The rank-conditioned level is not.* Some of the −3.9% to −11.9% gap is a selection
effect and would appear even for an unbiased model: conditioning on the model's own
ordering picks the runners it happened to price shortest, which is the winner's curse.
But the monotone widening down the ranks is larger than the within-race slope alone
predicts, and it is the part the calibrator used to absorb.

So the earlier conclusion — "keep the calibrator fitted and monitored, but it currently
has no work to do" — was too strong, and so was my restatement of it this morning. The
honest position is narrower: **the calibrator is not needed to fix within-race shape, and
removing it has a visible cost in rank-conditioned level.** Whether that cost matters
depends on the use. For ordering runners it does not; for deciding what price to take it
might. That is a decision to take with the numbers in view, not one this report should
make by assertion. `research_lab.py price-cal` now fits against `predicted_bfsp_raw`, so
it can be re-measured honestly whenever the question is asked.

Quantile coverage on R1: the q25 forecast comes in at or below the realised BSP 23.6% of
the time and the q75 72.3%, against nominal 25% and 75%.

## Segments, which a pooled number hides (R1)

Field size and price band were the only cuts this report used to make, and they are the
two the model is least likely to be interestingly wrong about. Top pick by race type,
R1, commission 5%, segments under 200 bets pooled into `(other)` so the rows add back to
the whole:

| race type | bets | win rate | ROI | 90% CI | market fav ROI |
|---|---|---|---|---|---|
| Handicap Flat | 5,635 | 26.8% | −3.92% | −8.57 to +0.17 | −4.37% |
| Handicap Hurdle | 1,469 | 26.3% | +5.88% | −3.00 to +15.31 | −1.20% |
| **Handicap Chase** | **1,161** | **26.6%** | **−10.57%** | **−17.54 to −2.37** | **−5.95%** |
| Non-Hcp Hurdle | 1,157 | 43.7% | −0.90% | −7.94 to +5.61 | +1.09% |
| Maiden | 948 | 39.4% | −0.52% | −8.85 to +7.69 | −2.87% |
| Novices | 713 | 42.9% | +1.15% | −8.25 to +10.95 | −9.02% |
| NH Flat | 351 | 30.2% | −9.68% | −26.29 to +5.18 | −11.47% |
| Non-Hcp Chase | 345 | 39.7% | −7.38% | −19.71 to +5.87 | −10.83% |

**Handicap chases are the one segment that replicates.** R1 has the model's top pick at
−10.6% there with an interval clear of zero, against −5.9% for the market's favourite in
the same races. Run 10, a different period and a different recipe, put the same segment
at −10.2% against the favourite's −2.2%. A finding that survives a change of both is
worth more than any of the single-slice results in section 5, and it points at something
specific: whatever the model is getting wrong, it is worst where handicapping and jumping
interact.

Everything else in the table has an interval spanning zero and should be read as noise
until it replicates too.

## 7. What this says about what to do next

- **Do not stake this model.** Not flat, not Kelly, not any fraction of Kelly. −4% on the
  top pick with a t of −3.4 is a real, measured loss, not noise.
- **The calibration and the price forecast are assets.** Accurate BFSP with honest
  quantiles is what a market-making or timing strategy needs; it is not what a
  pick-the-winner strategy needs.
- **Beating the market requires information the market does not have.** The concordance
  gap (0.657 vs 0.682) is the whole problem in one number, and no amount of staking
  cleverness closes it. That is an argument for the Stage F market-free model and for
  the data the framework says we lack — sectionals, in-running, ladder depth, sales and
  physicals — not for more feature engineering on what we already have.
- **The 166 opt-in features are unvalidated.** They are leak-free, which is necessary and
  not sufficient. Put them through the promotion protocol against *these* numbers.
