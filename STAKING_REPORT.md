# Market-blind staking: Kelly and per-race ranks

*Run 10, walk-forward out of sample, 2024-05-31 to 2026-03-21: 215,760 runners, 23,191
races, 22 folds. Every figure below is net of 5% Betfair commission. Reproduce with
`python research_lab.py stake --predictions data/oos_predictions.csv`.*

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

This is the part that works, and it is what the model is for: forecasting BFSP well
enough to get money on earlier at a better price.

| model rank | median forecast / actual BSP | mean abs log error | CLV at forecast |
|---|---|---|---|
| 1 | 1.0002 | 0.305 | +0.02% |
| 2 | 0.9952 | 0.365 | −0.48% |
| 3 | 1.0109 | 0.417 | +1.09% |
| 4 | 1.0031 | 0.455 | +0.31% |
| 5 | 0.9922 | 0.486 | −0.78% |
| 6 | 0.9940 | 0.510 | −0.60% |

**The 8% price bias on top picks is gone — it now reads 0.02%.** Worth being precise
about why: the bias was substantially the leak, not a modelling flaw the calibrator
fixed. On this run the calibrator adds nothing at all (mean absolute log error 0.4711
calibrated against 0.4710 raw), and the rank bias it was built to remove is already
within a percent at every rank. Keep `bsp_price_calibrator.json` fitted and monitored,
but it currently has no work to do.

Quantile coverage is close to exact: the q25 forecast comes in at or below the realised
BSP 26.5% of the time, the q75 74.3%, the median 50.7%.

So: the model forecasts the closing price accurately and without systematic bias, and it
has **no edge over that price**. Those are separate findings and only the second one is
disappointing. Closing-line value near zero means the forecast is honest; it also means
that, today, there is nothing to harvest by betting earlier on this signal alone.

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
