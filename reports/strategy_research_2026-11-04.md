# Strategy research, 4 Nov 2026: freshness, new jockeys, new sires, improvers and other systems

racing² research. Read-only: nothing here touches the live model or its jobs.

## The bar

The owner's criteria (4 Nov 2026): strategies that "hold up over large sample sizes, pass CHI tests, have positive
BFSP WAX - wins over BFSP expected wins - [and are] positive in the place market too - let's back both", staked to
win a set amount.

Fixed before any test figure was read (`research/strat/strat_lib.py`, `gate` and `gate_am`):

| | TRAIN (choose) | TEST (report) |
|---|---|---|
| At BSP | 2010-21: n >= 2,000; win A/E > 1, chi2 p < 0.001; place A/E > 1, p < 0.01 | 2022 - 3 Nov 2026: n >= 500; WAX > 0, chi2 p < 0.05; place A/E > 1; win+place ROI > 0 after 5% commission; win and place A/E both > 1 in 3+ years |
| At the morning price | 2018-21 (morning prices start 2018): n >= 1,000; win A/E > 1, p < 0.001; place A/E > 1, p < 0.01 | as above against the morning win and place prices, win p < 0.01 |

WAX = winners - sum(1/BSP). A/E = winners / sum(1/BSP). Place: placed within the places Betfair paid, against the
place BSP. Chi-square: Pearson goodness of fit, 1 df. Every signal reads runs on earlier days only. A non-finisher is
a loser (the first pass dropped them and showed chase A/E 1.2 everywhere: a bug, fixed). Controls (every runner)
read win and place A/E 1.00 at BSP in every code.

## Result

197 rule tests across six studies (A-F). Two strategies pass your gate, and they pass at different times of day; a
third finding is a list of horses to avoid:

**1. The quick return, backed in the morning (passes, both periods).** Any horse back within 1-4 days, all codes.

| | TRAIN 2018-21 | TEST 2022-26 |
|---|---|---|
| Runners with a morning price | 5,059 | 5,353 |
| Win A/E vs morning price (WAX) | 1.165 (p 2e-07) | 1.148 (+138 winners, p 6e-07) |
| Place A/E vs morning place price | 1.092 (p 6e-07) | 1.060 (p 0.002) |
| Win+place level ROI at morning prices | +11.1% | +8.2% |
| Years with win and place both > 1 | | 4 of 5 |

Every other runner reads 0.97-0.98 against its morning price (the morning is slightly over-bet); quick returners
read 1.15-1.17. Their morning price is on average 12.2% (train) / 9.4% (test) longer than the BSP, against -3.8% /
-3.4% for the rest: the market reprices them by the off. At BSP the same rule does not pass the test half (A/E 1.055,
p 0.051; place 1.013), so the edge is the morning price, not the horse.

Refinement: did not win last time. Test: win A/E_am 1.185 (+121 winners, p < 0.0001), place 1.083 (p 0.0002),
win+place +10.9%. Caveat: this split was suggested by an earlier table that showed both periods, so it is not
pre-registered. The parent rule is.

Jumps alone (hurdles + chases, 1-4 days) is the most stable: test win A/E_am 1.20, place 1.18 (p 0.0002), but its
2018-21 sample (912) is under the gate's 1,000 and its train win p is 0.04, so it is a strong candidate, not a pass.

**2. Timeform improvers, backed at BSP (passes at BSP; not at the morning price).** Flat/AW horses that ran a
career-best Timeform performance rating, by 5lb+, last time (Railway `APIData_Table2`).

| | TRAIN | TEST 2022-26 |
|---|---|---|
| 2014-21 / 2022-26, Flat only (study D) | n 36,759, A/E 1.09 (p 2e-12), place 1.06 | n 21,673, A/E 1.04 (p 0.021), place 1.02 (p 0.025), win+place +2.0%, 4/5 years |
| 2018-21 / 2022-26, Flat+AW, joined to race_results (study F) | n 25,914, A/E 1.10 (p < 1e-4), place 1.06 | n 31,083, A/E 1.04 (p 0.001), place 1.02 (p 0.006), win+place +0.5% |
| ... did not win last time | n 16,701, A/E 1.12, place 1.07 | n 19,581, A/E 1.05 (p 0.007), place 1.02 (p 0.03), win+place +2.0% |

Against the morning price it is flat (test A/E_am 1.01, p 0.32): these horses drift into the off, so the time to
back them is the BSP. Thin after commission: real, not big.

**3. Avoid: exposed horses and the 8-14 day gap.** Timeform rating flat over the last three runs: A/E 0.94 (train,
p < 1e-4), 0.98 (test); at the morning price 0.93 / 0.97. Back within 8-14 days, all codes: morning A/E 0.95 in both
periods (test n 69,388, p < 0.0001). Over-bet; not profitable to lay after commission and spread at these margins.

## Staking: to win GBP100 in the win market and GBP100 in the place market on every selection

Quick return, all codes, only where GBP50+ was matched in the morning (`strat_e_quick_return_1104.txt`, section 4):

| | Bets | Staked | P&L | ROI | Worst drawdown | Years |
|---|---|---|---|---|---|---|
| TEST 2022-26, morning prices | 5,353 | GBP718,396 | GBP24,457 (win 12,915, place 11,541) | +3.40% | GBP7,761 | -0.4, +3.4, +7.1, +0.1, +6.7% |
| TRAIN 2018-21, morning prices | 5,059 | GBP624,248 | GBP35,432 | +5.68% | GBP4,086 | |
| TEST 2022-26, BSP only | 6,147 | GBP831,488 | -GBP2,453 | -0.30% | GBP17,427 | |
| Jumps only, TEST, morning prices | 846 | GBP125,061 | GBP13,304 | +10.64% | GBP1,801 | +11.5, +13.1, +6.7, +15.6, +6.3% |

To-win staking returns less than level stakes (+3.4% against +8.2%): it puts most money on the shortest prices,
where the edge is thinnest. Worth testing a cap or level stakes before going live.

Capacity (2018+, 12,042 quick-return selections): 86.5% had GBP50+ matched in the morning, 57% GBP500+, median
GBP710. Median stake to win GBP100 at the morning price: GBP13.89.

## What did not hold up

* **Days since the last run, by code**, beyond 1-4 days: no other gap passes (`strat_a_freshness_improvers_1104.txt`).
* **New jockeys** (2,922 jockeys first seen 2012+, 662,737 rides): the market prices them correctly at every career
  stage (A/E 0.98-1.06, nothing consistent). An early A/E does not carry forward: Spearman -0.08 between rides 1-200
  and 201-500 (307 jockeys); the top fifth on early A/E (1.28) read 0.997 afterwards. What predicts a career is
  speed, not value: the fastest quarter to 200 rides went on to 2,000+ rides 60% of the time, the slowest 1%. No
  jockey rule passes. Watch-list (first ride 2023+, shrunk A/E > 1.2, place A/E > 1): Orla Tynan, Myla Coppins,
  Mr Rian Corcoran (`strat_new_jockeys_now.csv`), on the evidence a watch-list, not a bet.
* **New sires** (1,293 first-runner 2012+, 800,392 progeny runs): early indicators identify the good sires (first
  150 runs' winners-to-runners predicts the next 600's at Spearman 0.49) but not the value (early A/E to later A/E
  0.045): the market learns the sire as fast as the data does. No sire rule passes. Watch-list: King Of Change
  (`strat_new_sires_now.csv`).
* **Improvers on finishing position and official rating** (beat the market rank, OR up twice, positions improving):
  none pass. Timeform's performance ratings carry the improver signal; positions and OR do not.
* **Other systems** (first-time headgear, beaten favourite, class drop, course winner, trainer and jockey 14-day
  form, trip changes, code switches, claimers, Timeform top-rated): none pass. Timeform's top-rated is priced exactly
  (A/E 1.00).
* **A trap, recorded:** filtering on the BSP (favourite, under 2.0) "passes" the morning gate. That is hindsight: a
  horse that ends up short has shortened since the morning by construction. Every price filter in the passing rules
  uses the morning price.

## Next

1. A forward paper test of the quick return at the morning price, win and place, rules frozen as above, on
   `paper-models.yml` (never the live job), with a pre-registered pass mark.
2. Check overlap with the live trader: `days_since_lr_num` is already a model feature, so part of this may already be
   in its prices.
3. The Timeform improver at BSP as a candidate feature for the BFSP model rather than a standalone bet (the edge is
   ~2% and the model already reads the ratings it is built from).

## Sources

All numbers: `research/strat/results_2026-11-04/` (outputs as run). Code: `research/strat/strat_lib.py` (scoring and
gates), `research/queries/done/strat_a_*`, `strat_b_*`, `strat_c_*`, `strat_e_*`, `research/queries/strat_f_*`,
`research/strat/strat_d_timeform.py`, `research/strat/fetch_timeform.py`.

* HRB `race_results` + Betfair `betfair_prices` (win and place, BSP and morning WAP): 2,019,700 rows, 1 Jan 2010 to
  2 Nov 2026; 1,895,591 priced in races with a winner (the rest: no BSP or no winner). Research query runs
  41837391669 (A, B, C at 9b1486c), 41838216431 (E at 7f21575), 41839470293 (F at 7537fe2).
* Timeform: `horseracesbackend-production.up.railway.app/api/APIData_Table2`, GBR + IRE, one request per country and
  month, 310 files, 952,882 rows, 1 Jan 2014 to 3 Nov 2026, Flat and AW only (the table carries no jumps); 925,118
  priced in races with a winner. Places paid read off the place SPs (their implied probabilities sum to 2, 3 or 4).
  The flags joined to race_results matched 99.9% (464,374 of 464,739): `research/strat/data/timeform_flags_2014_2026.csv.gz`.
* Unverified: whether Timeform's `preRaceAdjustedRating` / `preRaceMasterRating` as stored are exactly the figures
  published before the off (they are named pre-race; the passing Timeform rule uses the last run's performance
  rating, which is published after that run, so it does not depend on this).
