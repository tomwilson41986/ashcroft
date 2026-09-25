"""
Empirical-Bayes shrinkage for statistics resting on different sample sizes.

A horse that has run once and won has a career NFP of 1.0; so does one that has
won twenty from twenty. A course-and-trip draw cell with four races behind it
reads as confidently as one with four hundred. Every raw mean in the feature set
has this problem, and a tree model can only partly undo it by splitting on the
count as well.

The fix is to report the posterior mean instead of the raw one. For a mean of
n observations with total s, under a normal (or beta) prior centred on m with
the weight of k observations,

    posterior mean = (s + k * m) / (n + k)

so a statistic resting on few observations sits near its prior and one resting
on many sits near its own mean. k is the ratio of the within-unit variance to
the between-unit variance: large when units differ little from each other
(pool hard), small when they differ a lot (trust each unit's own record).
`method_of_moments_k` estimates it from the data.

Hierarchies chain the same step: a track x trip x going cell shrinks toward its
track x trip cell, which shrinks toward the track, which shrinks toward the
whole population (`shrink_chain`). Each level borrows strength from the one
above exactly as far as its own sample falls short.

Every function here is arithmetic on sums and counts the caller has already
made lag-safe (earlier days only). None of them looks at dates.
"""

from __future__ import annotations

import numpy as np

#: Bounds on an estimated prior weight, in observations. Below 0.5 the prior is
#: ignored even for a single observation; above 500 nothing a unit does moves it.
K_MIN, K_MAX = 0.5, 500.0


def shrunk_mean(total, count, prior, k):
    """(total + k * prior) / (count + k), elementwise; NaN where the prior is NaN.

    `count` may be an effective (decayed) count. With count 0 the answer is the
    prior itself."""
    total = np.asarray(total, dtype=float)
    count = np.asarray(count, dtype=float)
    prior = np.asarray(prior, dtype=float)
    total = np.where(np.isfinite(total), total, 0.0)
    count = np.where(np.isfinite(count), count, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total + k * prior) / (count + k)


def method_of_moments_k(unit_means, unit_counts, within_var=None) -> float:
    """The prior weight k = within-unit variance / between-unit variance.

    `unit_means` are units' raw means and `unit_counts` their sample sizes. The
    variance of the raw means overstates the spread between units by the
    sampling noise, E[within / n], which is subtracted. Units with fewer than two
    observations inform the between-unit spread only through that correction.
    With `within_var` unset it is estimated as the pooled variance implied by a
    bounded outcome ([0, 1]: mean * (1 - mean)).

    Estimate k on earlier data than it is applied to: a k fitted on the scored
    window carries a (small) piece of the future into every prior."""
    m = np.asarray(unit_means, dtype=float)
    n = np.asarray(unit_counts, dtype=float)
    ok = np.isfinite(m) & (n >= 1)
    m, n = m[ok], n[ok]
    if len(m) < 10:
        return 10.0
    grand = np.average(m, weights=n)
    if within_var is None:
        within_var = max(grand * (1.0 - grand), 1e-6)
    total_var = np.average((m - grand) ** 2, weights=n)
    noise = within_var * np.mean(1.0 / n)
    between = total_var - noise
    if between <= 0:
        return K_MAX
    return float(np.clip(within_var / between, K_MIN, K_MAX))


def shrink_chain(levels, ks, top_prior):
    """Shrink a statistic down a hierarchy, most general level first.

    `levels` is a list of (total, count) pairs, arrays aligned to the same rows,
    from the broadest cell (say, the track) to the narrowest (track x trip x
    going x stall band). `ks` gives each level's prior weight. Each level's
    estimate is the next level's prior:

        est_0 = shrunk(level 0 toward top_prior)
        est_i = shrunk(level i toward est_{i-1})

    Returns the estimate at every level (list of arrays), so a caller can serve
    the narrowest and also the gap between levels (what the narrow cell says
    beyond the broad one)."""
    if len(levels) != len(ks):
        raise ValueError("one prior weight per level")
    est = np.asarray(top_prior, dtype=float)
    out = []
    for (total, count), k in zip(levels, ks):
        est = shrunk_mean(total, count, est, k)
        out.append(est)
    return out
