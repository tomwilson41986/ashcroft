# Market-blind staking: Kelly and per-race ranks

*Reproduce with `python research_lab.py stake --predictions data/oos_predictions.csv --bank-chart banks.png`.*

## 0. What "without the market" can mean, and what it cannot

A Kelly stake is a function of two numbers: the probability you believe and the price
someone offers. Take the market out of the price as well as out of the probability and
the model prices its own bets at its own probabilities — implied odds equal to 1/p, edge
identically zero, stake zero. There is nothing to measure.

So the market is removed from everything it *can* be removed from: the selection, the
probability, and every filter. It supplies only the price the bet settles at, BSP net of
5% commission. Nothing below conditions on a price that is unknown when the bet is
struck; where a price filter appears it is the model's *own forecast* price, not the BSP.

Two models are measured, because "our model" means two different things:

| | runners | races | period | market-free? |
|---|---|---|---|---|
| **Production BFSP model** (`train_bfsp.py`) | 253,532 | 27,223 | Jan 2024 – Feb 2026 | **No.** Trained on BSP as the target, with ~20 market-derived features. Market-informed, even though none of the staking rules below look at a price. |
| **Stage F** (`train_stage_f.py`, Blandford data) | 47,020 | 5,017 | Jul 2025 – Feb 2026 | **Yes.** Winner label, no market feature, no market target. |

## 1. Kelly on the production model

Full Kelly backs 141,026 of 253,532 runners — 55.6% of every field — at an average
3.75% of bank each. It claims a median edge of +54% per unit staked. It realises −1.0%.

| Kelly fraction | turnover (bank multiples) | ROI per unit staked | races to halve the bank | races to reach 1% of bank | log₁₀ final bank |
|---|---|---|---|---|---|
| 1 | 5,284 | −1.04% | 9 | 29 | −732.5 |
| ½ | 2,642 | −1.04% | 13 | 154 | −224.1 |
| ¼ | 1,321 | −1.04% | 21 | 364 | −68.3 |
| ⅛ | 660 | −1.04% | 171 | 963 | −20.7 |
| 1/20 | 264 | −1.04% | 417 | 11,759 | −4.4 |

Two things are worth separating here.

**The edge drag is small; the variance drag is bigger.** At 1/20 Kelly the strategy turns
over 0.97% of bank per race at −1.04% ROI, an arithmetic cost of 0.010% of bank per race.
Realised log growth is −0.037% per race. Roughly **three quarters of the loss is variance,
not edge** — the signature of betting a probability you do not have.

**Kelly sizing is better per unit staked than flat sizing, and far worse in outcome.**
Over the same 141,026 selections, flat stakes return −5.69% and Kelly-weighted stakes
−1.04%: Kelly correctly puts more money on the short-priced runners where the model is
least wrong. It then loses the bank anyway, because it is sizing on a claimed +54% edge
that does not exist.

Flattening the probabilities does not rescue it. Replacing p with p^λ renormalised per
race and re-running quarter-Kelly gives log growth −0.0105 at λ=1 and −0.0364 at λ=0 —
monotonically worse as the model is flattened toward 1/N. **The ordering is the asset;
the confidence is not the problem.**

## 2. Per-race ranks on the production model

Flat stakes, settled at BSP net of commission, 90% race-bootstrap CIs.

| model rank | n | win % | place % | avg BSP | flat ROI | 90% CI | ROI laying it |
|---|---|---|---|---|---|---|---|
| 1 | 27,223 | 30.0 | 57.3 | 4.71 | **−0.44%** | −2.12 to +1.44 | −6.71% |
| 2 | 27,223 | 19.8 | 46.4 | 7.31 | −2.43% | −4.78 to −0.02 | −5.67% |
| 3 | 27,170 | 14.2 | 38.1 | 11.04 | −5.11% | −8.00 to −1.88 | −3.43% |
| 4 | 26,861 | 10.9 | 31.7 | 19.15 | +0.60% | −5.22 to +7.97 | −9.78% |
| 5 | 25,857 | 8.3 | 26.1 | 32.49 | −5.27% | −10.20 to −0.09 | −3.86% |
| 6 | 23,942 | 6.5 | 22.0 | 49.24 | −8.23% | −13.62 to −2.54 | −0.93% |

The ranking is monotone in win rate and in price, which is the least a usable model must
do. Backing the top pick in every race is a −0.44% business. The reference points: a
random runner returns −5.73% and the market favourite −3.05% over the same races, on a
BSP book that averages 1.0016, so almost all of that is the 5% commission. The model's
top pick is 5.3 points better than a random bet and 2.6 points better than the
favourite, and is not distinguishable from break-even. Backing the top 2, 3 or 4 is
worse (−1.44%, −2.66%, −1.85%). Laying loses on every rank.

Rank 4's +0.60% is noise: its four chronological quarters run +24.7%, −6.6%, −4.9%,
−10.8%, i.e. one good half-year in early 2024 and nothing since.

## 3. Kelly and rank together

Quarter-Kelly restricted to one rank.

| rank | bets | mean stake | ROI | log₁₀ final bank | worst drawdown |
|---|---|---|---|---|---|
| 1 | 14,181 | 2.54% | +0.68% | −12.4 | 100% |
| 2 | 14,663 | 1.75% | −3.30% | −13.8 | 100% |
| 3 | 14,529 | 1.32% | −1.54% | −9.8 | 100% |
| 4 | 14,635 | 0.99% | +4.69% | −9.5 | 100% |
| 5 | 14,434 | 0.77% | −7.15% | −7.6 | 100% |
| 6 | 13,433 | 0.62% | −8.72% | −7.2 | 100% |

Rank 1 at quarter-Kelly has a *positive* ROI per unit staked (+0.68%) and still ends at
10⁻¹² of the starting bank. A 2.5%-of-bank average stake at a 30% strike rate is far
past the growth-optimal size for an edge this thin — the stake, not the selection, is
what destroys it.

## 4. Where the market-blind edge actually is

Both filters below are known before the off and use no market data: the number of
declared runners, and the model's own forecast BFSP.

| selection | n | win % | avg BSP | flat ROI | 90% CI | t | quarters (chronological ROI) |
|---|---|---|---|---|---|---|---|
| model #1, all races | 27,223 | 30.0 | 4.71 | −0.44% | −2.3 to +1.5 | −0.4 | −1.3, −3.1, −0.3, **+2.9** |
| model #1, field ≥ 12 | 6,430 | 23.4 | 6.30 | **+4.40%** | −0.7 to +9.4 | 1.5 | −3.0, +3.2, +7.7, **+9.8** |
| model #1, field ≥ 16 | 1,379 | 21.5 | 7.60 | **+14.77%** | **+2.1 to +29.0** | 1.8 | +11.0, −13.4, +32.0, +29.6 |
| model #1, forecast price ≥ 8 | 1,254 | 15.7 | 9.54 | +13.44% | −1.8 to +29.0 | 1.5 | +13.7, −2.1, +27.7, +14.5 |
| model #1, field ≥ 12 and forecast ≥ 6 | 2,916 | 16.5 | 8.28 | +5.71% | −2.7 to +15.1 | 1.1 | −3.2, −0.2, +15.3, +11.0 |

Model #1 by field size: +14.8% in 16+, +1.6% in 12–15, −0.3% in 9–11, −3.1% in 6–8,
−3.0% in fields of 5 or fewer. By the model's own forecast price: +15.1% at 8–12 (n=1,192),
+1.8% at 5–8 (n=8,033), and negative everywhere shorter.

Read this carefully. Only one cell's 90% CI clears zero, and it is the smallest
(n=1,379). None of them would survive a multiple-comparisons correction on its own. What
makes them worth taking seriously is that they agree with each other, they agree with the
disagreement analysis (model #1 when the market has it 4th or worse: +7.9%), and the
trend in every one of them is upward across the two years. **Big fields and longer prices
are where the market is thinnest and where a market-blind model has room to be right.**

At quarter-Kelly the field ≥ 12 rule ends at 5× bank and the field ≥ 16 rule at 11×, but
with 95% and 69% peak-to-trough drawdowns respectively. The edge is real enough to
compound; the sizing that Kelly proposes for it is not survivable.

## 5. The genuinely market-free model (Stage F)

5,017 UK/IRE Flat races, July 2025 to February 2026. No market input anywhere.

| | rank-1 win % | rank-1 flat ROI | 90% CI | full-Kelly log₁₀ final bank |
|---|---|---|---|---|
| Conditional logit | 27.2 | −1.95% | −6.6 to +2.7 | −213.0 |
| LightGBM race-softmax | 26.5 | −5.11% | −9.8 to −0.3 | −199.0 |

Top-k is negative for every k. The field ≥ 16 and forecast-price ≥ 8 cells are positive
(+3.3% and +9.7% for the conditional logit) but carry 50 to 180 bets, which is no
evidence at all. Over eight months a market-free model built on the Timeform-style feed
picks winners at the same rate the production model does and loses about the same 2–5%
to the price. This is the expected result: Stage F's McFadden R² is 0.093–0.098 against
the market's 0.172.

## 6. The price forecast, by rank

The model's stated job is to predict BFSP so money can go on earlier at a better price.
Measured on that:

| model rank | median forecast | median BSP | forecast ÷ actual | forecast above BSP | mean abs log error |
|---|---|---|---|---|---|
| 1 | 4.24 | 3.85 | 1.083 | 57.8% | 0.342 |
| 2 | 6.15 | 5.78 | 1.060 | 54.7% | 0.400 |
| 3 | 8.22 | 7.80 | 1.050 | 53.8% | 0.451 |
| 4 | 10.67 | 10.50 | 1.033 | 52.3% | 0.493 |
| 5 | 13.66 | 13.95 | 1.007 | 50.4% | 0.528 |
| 6 | 17.14 | 18.00 | 0.999 | 50.0% | 0.555 |

There is a systematic **+8% upward bias in the forecast price for the model's own top
pick**, decaying to zero by rank 5. The top pick closes shorter than the model says it
will, 58% of the time. **This has since been fixed** — see `model/price_calibration.py`
and `python research_lab.py price-cal`. Calibrating the price against realised BFSP,
walk-forward, takes the top pick's median forecast/actual ratio from 1.081 to 0.992 and
holds every rank inside ±1.4%, with a slightly lower mean absolute log error. The tables
in this report are from the uncalibrated forecast and are left as the record of the
defect. For an early-money strategy that is a conservative bias — it
demands a longer price than the close requires, so it under-fires rather than over-fires
— but it is 8% of edge left on the table at the front of the book, and it is a
retransformation bias in a log-target regression, which is correctable.

## 7. What this says to do

1. **Do not size by Kelly on these probabilities.** Any fraction from 1 to 1/20 loses the
   bank over 27,000 races. If a rule is bet at all, bet it flat or at a small fixed
   fraction, not proportional to a claimed edge the model does not have.
2. **The only market-blind rules worth live testing are the model's top pick in big
   fields, and at forecast prices of 8 or longer.** Both are consistent with the earlier
   disagreement result and both have improved year on year.
3. **Fix the +8% price bias on top picks** before any of it is used as an early-price
   trigger.
4. **Nothing here changes the main conclusion**: the model is less resolved than BSP
   (Brier skill −4.5%), so the value is in forecasting where BSP will land, not in
   beating it. Closing-line value remains the metric, and it needs the Betfair price
   files to measure.
