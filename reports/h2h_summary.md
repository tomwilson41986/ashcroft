# Which objective should the BFSP model be trained on?

*Five variants against the `v0_l2` default, all on the same cached feature matrix
(`start_date 2023-06-01`, `eval_from 2025-04-01`), the same 11 walk-forward folds, and
the same 107,047 runners over 11,671 races from 2025-04-26 to 2026-03-21. Paired by
runner; every interval is a 90% race-cluster bootstrap. Per-variant detail in
`h2h_<tag>.md`; the tool itself is validated against a known answer in
`h2h_validation_leak.md`.*

**Decision rule, fixed before the runs:** a variant replaces the default only if its
paired error interval excludes zero in its favour **and** neither Brier skill nor
winner-vs-loser concordance is worse by more than its own interval. Rank-1 ROI is
reported and never decisive — at this sample its interval is about ±2.7 points, wider
than any difference worth acting on.

## The verdicts

| variant | what it changes | paired Δ mean abs log err (90% CI) | verdict |
|---|---|---|---|
| `v1_l2_decay1` | recency weight `exp(−days/365)` | +0.0011 (+0.0004 to +0.0018) | default stands |
| `v2_profit` | objective weights by `sqrt(1/BFSP)` | +0.0147 (+0.0138 to +0.0156) | default stands |
| `v3_profit_decay1` | both of the above | +0.0151 (+0.0142 to +0.0160) | default stands |
| `v4_logit` | target = logit of the normalised probability | **−0.0016 (−0.0023 to −0.0009)** | **passes the rule** |
| `v5_demeaned` | target = within-race demeaned log price | +0.0004 (−0.0006 to +0.0013) | default stands |

Negative favours the variant. `v0_l2` reads 0.4633 mean absolute log error.

## The finding is in the rank table, not the headline

Paired change in error by the base model's rank in the race:

| variant | rank 1 | rank 2 | rank 3 | rank 8+ |
|---|---|---|---|---|
| `v1_l2_decay1` | −0.0005 | +0.0003 | −0.0012 | **+0.0024** |
| `v2_profit` | **−0.0022** | −0.0003 | +0.0018 | **+0.0316** |
| `v3_profit_decay1` | −0.0016 | 0.0000 | +0.0024 | **+0.0306** |
| `v4_logit` | −0.0009 | −0.0001 | −0.0003 | **−0.0035** |
| `v5_demeaned` | +0.0003 | −0.0013 | −0.0043 | **+0.0059** |

Four of the five trade the same way: **whatever they buy at the top of the market they
pay for, with interest, in the tail.** `v2_profit` is the clearest case and the most
honest about it — the objective explicitly weights a runner by `sqrt(1/BFSP)`, so a 2.0
shot carries more than four times the gradient of a 50.0 shot. It duly delivers the
largest rank-1 improvement of any variant (−0.0022, interval clear of zero) and the
largest tail degradation by an order of magnitude (+0.0316). It does exactly what it
says on the tin, and what it says on the tin is the wrong objective for a model asked to
price every runner.

`v4_logit` is the only variant off that frontier: it improves the tail (−0.0035, clear of
zero) *and* nudges rank 1, which is why it is the only one to pass. Fitting the logit of
the race-normalised probability optimises the quantity that actually gets served, since
the price is normalised to a book of 1 either way.

`v5_demeaned` nets to nothing overall but is not inert underneath: rank 3 improves
(−0.0043, clear of zero) and rank 8+ degrades (+0.0059, clear of zero). That is worth
knowing before anyone reads "no difference" as "no effect".

## Two of these runs could not have been measured before today

`v3_profit_decay1` was, until this morning, a byte-identical duplicate of `v2_profit`.
LightGBM applies a Dataset's weight inside each *built-in* objective's gradient
computation; a custom objective's gradients pass through untouched, so `--decay-rate`
was silently discarded whenever the profit objective was on — which was production's
default. After the fix the two runs differ on **92.1% of runners** by more than 1%
(median |log ratio| 0.071). The decay now reaches the model; in production it never did.

Both profit runs were also re-dispatched after a second fix. A custom objective starts
from zero rather than the mean, costing roughly 300 of 3000 rounds climbing to the
intercept — a tenth of the budget, charged to one side of the comparison for an
implementation detail. With `init_score` set, the fit is within 0.15% of the intercept at
round one. So these are the first numbers for that objective that describe the objective
rather than its handicap, and it still loses.

## What this says

The default stays. The one variant that passes, `v4_logit`, does so by 0.35% of the
error, and its concordance interval (−0.00215 to +0.00010) very nearly excludes zero on
the wrong side — it buys tail accuracy at some cost to ordering. Adopting it means a
retrain and a second deploy, which is a decision to take deliberately rather than on the
strength of a rule clearing by a hair.

What the exercise settles firmly is the question that prompted it: **training on every
runner rather than on the ones that pay is the right call, and the margin is not
subtle.** The profit-weighted objective is 0.0147 worse overall and 0.0316 worse in the
tail, measured on the same rows with the same folds.
