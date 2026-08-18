# H2H Match Simulation — Hopewell Rock (IRE) v Opportunity

**10,000 simulated races** (seed 42, `scripts/simulate_h2h.py`)

| Horse | HRB id | Wins | Win % |
|-------|--------|------|-------|
| Hopewell Rock (IRE) | 397142 | 4,265 | 42.6% |
| **Opportunity** | 399032 | **5,735** | **57.4%** |

Seed-stability: Opportunity 57.0–58.5% across seeds {1, 7, 42, 123, 2026}.

## Method

1. Form scraped from the logged-in horseracebase pages (`h2h-fetch` workflow,
   HRB credentials from repo secrets).
2. Each career run converted to a performance figure on the OR (lbs) scale:
   `OR at the time − beaten lengths × lbs-per-length(distance)`, winners
   credited with their winning margin (capped +5 lbs). Unrated early-career
   runs dropped; beaten figures floored at OR − 20 (eased/tailed-off runs).
3. Race-day performance sampled per horse via recency-weighted bootstrap of
   its own figures (half-life 270 days) + N(0, 3 lbs) day noise.
4. Higher figure wins the simulated match; repeated 10,000×.

## Features generated

**Hopewell Rock (IRE)** — 4yo New Bay gelding (Boughey), OR 101.
7 rated runs; recency-weighted ability **93.9** (spread 7.0).
Best recent: won £62k Cl2 handicap at Goodwood 1 Aug 2026 (perf 103).

**Opportunity** — 4yo (Haggas), OR 110.
6 rated runs; recency-weighted ability **95.8** (spread 10.4).
Best recent: 2nd beaten 1.3L in £62k Gr3 at Goodwood 1 Aug 2026 (perf 108).

## Reality check

They actually met on **19 Jun 2026 at Ascot** (1m4f £62k Cl2 handicap):
Opportunity **won** (1/17); Hopewell Rock finished 6th, beaten 8L.
The official handicapper now has Opportunity 9 lbs clear (110 v 101),
which points the same way as the simulation.

Why only 57/43 rather than a landslide: the bootstrap draws from each
horse's whole recent form book, and Hopewell Rock's figures are tightly
clustered (spread 7 lbs) while Opportunity's include a Gr2 flop and
lighter early-2026 marks — so on a bad day for B and a good day for A,
A wins. On latest form alone (perf 108 v 103), Opportunity is clearly
the more likely winner, and the simulation agrees.

Full per-run figures and config: `data/h2h/simulation_result.json`.
