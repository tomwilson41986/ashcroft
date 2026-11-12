# racing² - football trading strategy candidates (12 Nov 2026)

**Result.** A registered model-free scan of 25 rules on the 2026 Betfair football files found **no lead** by its
own reading (same sign in both halves of the year, and the even months' 90% interval above zero at a worse price
than the last trade). Four patterns are worth a proper test; three of them only pay for an order that rests in the
book rather than one that crosses the spread. Nothing here trades, and nothing is a pass.

Source for every figure: `reports/football_runs/strategy_scan_1112.json`, written by
`python -m research.open_close.football_strategy_scan` on the local build of the 2026 warehouse (the same files as
the capture bucket; the local and bucket builds agreed on every model number in runs 43112672668 and 43167168711).
Registered before the run: ledger `football-strategy-scan-1112`. Literature (read 11 Nov; abstracts only unless
marked): Whelan, https://www.karlwhelan.com/Papers/Betfair.pdf (full text); Angelini, De Angelis and Singleton 2022
(full text, cris.unibo.it); Choi and Hui 2014, JEBO 107(B) 614-629; Moskowitz 2021, J. Finance 76(6) 3153-3209. All bets are backs held to the result, a unit staked, 2% commission on a win; 90% intervals from
resampled match days. "Worse price" is one tick down the ladder (in play, the close) or 1.15% off (pre-match).

## 1. Candidates, ranked

| # | Candidate | What the scan shows | Why it might be real | What would confirm it | Needs |
|---|---|---|---|---|---|
| 1 | **Rest a back on the leader at half time** (raw chance 0.85 to 0.99) | at the price +1.16% (+0.20 to +2.18) on 2,865 bets and +1.20% (+0.39 to +2.00) on 1,071; about zero one tick worse | last traded prices read a side that is ahead about a point low (platform document 10.5); Whelan: takers on strong second-half favourites earn small profits, makers gain 0.6% | fills at the last traded price or better in recorded in-play books | in-play book recording (not held: the BASIC files have no quotes) |
| 2 | **Rest a back on the leader late on** (minutes 75-90, chance 0.85 to 0.99) | at the price +0.33% (+0.00 to +0.68) on 30,420 bets and +0.28% (+0.04 to +0.52) on 16,750; -0.80% (-1.30 to -0.31) one tick worse (even months) | the same reading bias; the edge is under one tick, so only a maker keeps it | as 1 | as 1 |
| 3 | **Lay the pre-match drifter at T-60** (log odds out by 0.10 or more since T-360) | backers lost -8.83% (-14.36 to -2.81) on 2,539 bets at a mean price of 6.3; -16.30% (-26.83 to -5.55) on 894 busy rows; negative in both halves, the even months' interval below zero | unexplained, and against the literature: Moskowitz (2021, US sports) finds open-to-close moves partly reverse at the result, which would favour the drifter; no football exchange study found. Treat as the weakest prior of the four | the mirror cell was not registered as a lay: a registered lay test on the back years (2016-2025), then at quotes | the back years; liability sizing (a lay at 6 risks 5 to win 1) |
| 4 | **Back the pre-match favourite after it scores** (price up by 0.12 or more, bet 2 to 5 minutes on) | at the price +4.61% (+0.22 to +10.07) on 17,742 bets; one tick worse +1.39% (-2.33 to +6.20) (odd) and +5.62% (-1.38 to +15.71) (even): intervals span zero | under-reaction to an ordinary goal (Choi and Hui 2014); Angelini et al. find the favourite's goal over-rated on the EPL: the sign is disputed | the back years, with real goal times from API-Football in place of price jumps | the back years; goal events |
| 5 | **Lay the over/under 2.5 favourite at a thin close** | favourites in tiers C-D: -1.71% (-2.95 to -0.40) on 9,921; outsiders +1.12% (-1.85 to +3.71); closing slope 0.936 (se 0.023) in the efficiency study | a reverse favourite-longshot lean where the market is thin | at quotes: a thin market's last trade is the weakest price we hold | recorded over/under books |
| 6 | Team-news timing (backlog `lineup-timing`) and the spread-aware gate (backlog `spread-aware-gate`) | not in this scan: registered earlier, waiting on the recorded books | first of the closing-line model's next steps | their own registered tests | lineup publication times |

Reading the ranking. Candidates 1 and 2 are the same finding: a leader's price is a little too long, by less than
one tick. A taker gives it all back at the spread; a resting order might keep it. That needs in-play books, which
we do not record. Candidate 3 is the largest effect in the scan and the least expected: the literature points the other
way, and with 25 cells about one such result is expected by chance. It stands or falls on the back years.

## 2. What the scan rules out (for now)

- **Crossing the spread on any in-play leader**: every window and band is at or below zero one tick worse.
- **Backing the side that concedes** (the late longshot): -15.36% (-25.46 to -3.57). Whelan's robust in-play anomaly; the lay side of it is candidate 2.
- **Following pre-match steam to the result**, **the ratings against the price as a win bet**, **the Asian handicap's main line at the close**: no cell has both halves above zero.
- From the trade model (same day): the trade-out at the SP and model win bets, both below zero.

## 3. All 25 cells

| Cell | Bets | Mean price | At the price, all months | Worse price, odd months | Worse price, even months |
|---|---|---|---|---|---|
| in-play leader | half time | chance 0.70-0.85 | 4,722 | 1.30 | -0.25% (-1.57 to +1.06) | -1.12% (-2.70 to +0.48) | -0.89% (-2.92 to +0.90) |
| in-play leader | minutes 60-75 | chance 0.70-0.85 | 15,942 | 1.31 | -0.50% (-1.28 to +0.25) | -1.35% (-2.41 to -0.32) | -1.17% (-2.27 to -0.15) |
| in-play leader | minutes 75-90 | chance 0.70-0.85 | 28,485 | 1.32 | -0.07% (-0.63 to +0.50) | -0.68% (-1.44 to +0.08) | -0.95% (-1.69 to -0.13) |
| in-play leader | half time | chance 0.85-0.95 | 2,865 | 1.11 | +1.16% (+0.20 to +2.18) | +0.49% (-0.77 to +1.71) | +0.05% (-1.40 to +1.55) |
| in-play leader | minutes 60-75 | chance 0.85-0.95 | 8,593 | 1.12 | -0.56% (-1.24 to +0.12) | -0.82% (-1.72 to +0.07) | -2.02% (-2.90 to -1.13) |
| in-play leader | minutes 75-90 | chance 0.85-0.95 | 30,420 | 1.13 | +0.33% (+0.00 to +0.68) | -0.28% (-0.71 to +0.19) | -0.80% (-1.30 to -0.31) |
| in-play leader | half time | chance 0.95-0.99 | 1,071 | 1.04 | +1.20% (+0.39 to +2.00) | +0.66% (-0.24 to +1.50) | -0.24% (-1.81 to +1.08) |
| in-play leader | minutes 60-75 | chance 0.95-0.99 | 5,253 | 1.04 | -0.19% (-0.60 to +0.20) | -0.68% (-1.20 to -0.18) | -1.62% (-2.28 to -0.99) |
| in-play leader | minutes 75-90 | chance 0.95-0.99 | 16,750 | 1.04 | +0.28% (+0.04 to +0.52) | -0.50% (-0.81 to -0.16) | -0.83% (-1.15 to -0.51) |
| price jump | back the gainer | gainer the pre-match favourite | 17,742 | 2.97 | +4.61% (+0.22 to +10.07) | +1.39% (-2.33 to +6.20) | +5.62% (-1.38 to +15.71) |
| price jump | back the other side | gainer the pre-match favourite | 17,742 | 86.41 | -15.36% (-25.46 to -3.57) | -20.86% (-28.62 to -11.70) | -15.71% (-32.10 to +6.81) |
| price jump | back the gainer | gainer the pre-match outsider | 12,391 | 4.08 | +1.11% (-2.81 to +5.79) | +4.67% (-2.65 to +13.52) | -4.95% (-7.49 to -2.55) |
| price jump | back the other side | gainer the pre-match outsider | 12,391 | 56.23 | +0.26% (-12.01 to +16.57) | +1.09% (-19.23 to +27.98) | -5.80% (-17.03 to +6.76) |
| over/under 2.5 close | outsider (chance under 0.40) | tiers A-B | 3,679 | 3.34 | -0.51% (-4.35 to +3.58) | +1.69% (-3.80 to +7.20) | -4.67% (-10.34 to +0.81) |
| over/under 2.5 close | outsider (chance under 0.40) | tiers C-D | 9,921 | 3.24 | +1.12% (-1.85 to +3.71) | +1.89% (-2.08 to +6.33) | -1.93% (-5.53 to +1.44) |
| over/under 2.5 close | favourite (chance over 0.60) | tiers A-B | 3,679 | 1.51 | -1.56% (-3.61 to +0.42) | -3.52% (-6.39 to -0.79) | -0.95% (-3.73 to +1.92) |
| over/under 2.5 close | favourite (chance over 0.60) | tiers C-D | 9,921 | 1.49 | -1.71% (-2.95 to -0.40) | -2.88% (-4.90 to -0.94) | -1.85% (-3.57 to -0.21) |
| Asian handicap main half line close | the shorter side | 31,003 | 1.70 | -0.29% (-1.03 to +0.41) | -0.24% (-1.29 to +0.85) | -1.57% (-2.50 to -0.64) |
| Asian handicap main half line close | the longer side | 31,003 | 2.95 | +8.91% (-0.46 to +21.98) | +15.89% (-2.30 to +40.23) | -0.81% (-2.41 to +0.77) |
| pre-match momentum T-360 to T-60 | back the steamer | all rows | 2,351 | 4.22 | -4.17% (-9.04 to +0.81) | -8.32% (-15.47 to -0.36) | -1.99% (-8.73 to +5.28) |
| pre-match momentum T-360 to T-60 | back the steamer | 60+ price updates | 992 | 3.98 | -3.70% (-11.20 to +3.86) | -5.44% (-16.72 to +6.33) | -4.14% (-13.14 to +4.86) |
| pre-match momentum T-360 to T-60 | back the drifter | all rows | 2,539 | 6.30 | -8.83% (-14.36 to -2.81) | -5.50% (-13.18 to +1.90) | -14.43% (-22.67 to -5.20) |
| pre-match momentum T-360 to T-60 | back the drifter | 60+ price updates | 894 | 6.06 | -16.30% (-26.83 to -5.55) | -11.31% (-26.04 to +4.65) | -23.36% (-36.59 to -8.37) |
| rating gap at T-60 | back the side the ratings favour | all rows | 4,197 | 5.10 | -0.66% (-5.64 to +4.53) | +1.52% (-5.78 to +9.83) | -4.68% (-12.24 to +2.97) |
| rating gap at T-60 | back the side the ratings favour | 60+ price updates | 1,341 | 5.14 | -1.80% (-10.71 to +7.97) | -6.38% (-19.14 to +7.79) | +0.45% (-10.54 to +12.28) |

## 4. Next steps and what needs the owner

1. **The back years** (2016-2025): candidates 3 and 4 can be tested the day they are ingested, each registered first.
2. **In-play books**: candidates 1 and 2 need the football recorder to keep recording after the kick-off on a sample of matches. That is a change to the football trade job's request budget on the live key: the owner's call.
3. Nothing is proposed for paper or live. A candidate that passes its registered test at quotes becomes a registry entry at the research stage (`docs/agent_desk.md`).
