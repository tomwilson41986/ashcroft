# Australia and New Zealand on Betfair: the market-only scan (2 Nov 2026)

The owner's ask (2 Nov 2026): run the free first step of the international plan (`reports/international_racing.md`
s.6, step 1) for Australia and New Zealand. Read-only; nothing loaded into `horse_racing.db`, nothing in the live
UK/IE system touched, no Betfair key used.

Two sources, each read in full:

- **[A] our price-file archive**: every `auswin` file at `s3://ashcroft/betfair_prices_raw/` (5,154 files, 20 Sep
  2012 to 1 Nov 2026), read by `research/queries/done/au_nz_price_files_1102.py`, research-query run 41447563359
  (https://github.com/tomwilson41986/ashcroft/actions/runs/41447563359).
- **[B] Betfair's own ANZ thoroughbred files** (Betfair Data Scientists,
  https://betfair-datascientists.github.io/data/dataListing/: `ANZ_Thoroughbreds_2020.zip` to `_2025.zip` and
  `ANZ_Thoroughbreds_2026_01.csv` to `_2026_10.csv`, downloaded 2 Nov 2026), read by `research/anz_market_scan.py`
  (the scan, and `--ex-ante` for the price comparisons banded by the pre-off price).

Money is in each file's own unit. [B] carries no currency label (presumably AUD, unverified). [A] and [B] define
"pre-play" money differently (the medians below differ), so the two are never set against each other or against GBP.

## The result

1. **New Zealand is not in our price files.** In 14 years of `auswin`, NZ appears on one day (8 races, 78 runners,
   2026) [A]. NZ prices come only from Betfair's ANZ files [B]: about 11 races a day, run on 162-227 days a year.
2. **The archive is deeper than we thought: Australia from Sep 2012, every day.** [A] holds 364-366 files a
   year from 2013. The "81 races a day" in the 1 Nov note is harness included: about 38 harness and 46 thoroughbred
   races a day (2025: 13,868 and 17,042) [A]. Harness money is thin (runner median 433 in 2025) [A]; the model is
   thoroughbred only, so harness is out.
3. **The two sources agree on the races.** AU thoroughbred races: [A] 17,517 / 17,129 / 17,042 against [B]
   17,496 / 17,093 / 17,041 for 2023 / 2024 / 2025, within 0.2%.
4. **Only the price files' BSP and results are usable for Australia, not their average prices.** The file's
   "morning" window holds 54-59% of the money and its average price sits far under the BSP (median log(BSP /
   MORNINGWAP) +0.19 in 2012, rising to +0.49 in 2026). The PPWAP drifts away from the BSP from 2022 (+0.04 to
   +0.27 by band in 2022-2026; only 19-29% of runners "shortened") [A]. Betfair's own ANZ files put the pre-play average within
   -0.011 to +0.028 of the BSP from 1 to 51 in every year [B]. So in the price files, Australian races (run in the
   UK night) do not fit the UK morning and pre-play windows. For any pre-off price, use [B] or Betfair Historic Data.
5. **Australia: no market-only edge, and the close is sharp.** A level back at the BSP loses after 2% commission in
   every band, pooled 2020-2026: -1.0% (1-2) to -3.5% (21-51), -11.0% over 51 [B]. In [A], back to 2012, no band is
   positive after commission in more than 4 of the 15 years. The BSP book's race log-loss improves every year: 1.804
   (2020) to 1.755 (2026) [B].
6. **In Australia, a back near the off is worse than the BSP in every band and year.** The best back available at
   the scheduled off sits under the BSP by a mean 0.1-4.6% from 2 to 51 (banded by that price), every year
   2020-2026. A level back after 2% at that price against at the BSP: -1.3% against -1.1% (2-3), -2.6% against -1.1%
   (8-13), -4.7% against -2.9% (21-51) [B]. Late backing adds nothing over the SP. As in the UK, any edge has to be
   in the early price, our model's forecast of where it closes, and the SP lay that closes the position. Neither
   free source holds an early price.
7. **Australian money is shrinking but still there.** Market pre-play median 70,130 (2020), 54,312 (2022), 36,149
   (2025), 32,904 (Jan-Oct 2026). Runner median 2,956 to 1,820 over the same years. The share matched at the BSP rose
   from 20.4% to 35.7% [B]. In 2025, 47% of AU runners had at least 2,000 matched pre-play and 13% at least 10,000.
   Victoria is the deepest state (runner median 2,397), New South Wales 1,466 [B].
8. **New Zealand is a thin, looser market.** Market pre-play median 7,021-11,651, runner median 253-601 (2020-2026).
   In 2025 only 9% of runners had 2,000 matched pre-play, none had 10,000 [B]. The BSP book is less sharp than
   Australia's (log-loss 1.93-1.99 against 1.76-1.80). The best back at the off sits under the BSP by 1.8-15.3% from
   5 to 51 [B]. A model has more room there, and the books have less money to fill a GBP15-200 back.
9. **One NZ cell is positive, and it is most likely chance.** A level back at a BSP of 8-13 made +3.7% after
   commission, pooled 2020-2026 (30,331 runners), and was positive before commission in each of the 7 years [B].
   But it is one of 16 band-country cells tested. Its neighbours lose (13-21: -4.0%; 5-8: -4.2%), and finer bands
   jump about (7-8 -6.5%, 8-9 +1.3%, 12-13 +9.6%, standard errors 2.8-4.9%). Not a rule. Only a pre-registered test
   on data not yet seen could make it one.

## 1. Data checks

| Check | [A] price files (auswin) | [B] ANZ files |
|---|---|---|
| Files | 5,154 read, 0 unreadable, 18 empty | 82 monthly files (6 zips unpacked + 10 monthly CSVs) |
| Rows in | 4,119,258 | 1,303,266 |
| Duplicates (market, selection) | 1,131, dropped | 2,558, dropped |
| Nulls in key fields | 116 blank MENU_HINT (no country: 13 races 2017-2020); none in ids, names, times | none in market, selection, track, state, result |
| No BSP | 57,378 | 364 |
| No result | 1,237 | 0 (only WINNER/LOSER rows) |
| Markets | | 132,750: 132,519 with one winner, 231 dead heats, none without a winner |
| Dates | 18 Sep 2012 to 31 Oct 2026 | 1 Jan 2020 to 31 Oct 2026 (Nov-Dec 2026 not yet published); every date inside its file's month |
| Date formats | `DD-MM-YYYY HH:MM` | ISO in some files, `D/MM/YYYY` in others: both read, none unread |

## 2. Races and money

**[B] by year and country** (pre-play money as Betfair writes it, presumably AUD):

| Year | AU races (a day) | AU market pre-play, median | AU runner, median / p90 | AU share at the BSP | NZ races (a day) | NZ market pre-play, median | NZ runner, median |
|---|---|---|---|---|---|---|---|
| 2020 | 16,904 (46.3) | 70,130 | 2,956 / 23,526 | 20.4% | 1,882 (11.6) | 7,162 | 253 |
| 2021 | 16,971 (46.9) | 64,953 | 3,043 / 23,170 | 24.8% | 2,391 (11.0) | 8,406 | 333 |
| 2022 | 17,141 (47.1) | 54,312 | 2,943 / 19,297 | 26.9% | 2,355 (10.7) | 9,139 | 428 |
| 2023 | 17,496 (48.1) | 45,686 | 2,306 / 15,823 | 27.4% | 2,436 (11.1) | 11,651 | 601 |
| 2024 | 17,093 (46.8) | 39,273 | 1,974 / 13,266 | 32.3% | 2,457 (10.8) | 8,937 | 435 |
| 2025 | 17,041 (46.8) | 36,149 | 1,782 / 12,206 | 33.6% | 2,480 (11.2) | 7,021 | 325 |
| 2026 (Jan-Oct) | 14,058 (46.2) | 32,904 | 1,820 / 10,864 | 35.7% | 2,045 (10.6) | 7,134 | 368 |

"Share at the BSP" is the median over markets of BSP money / (pre-play money + BSP money). The median field is 10
in both countries (11 in NZ to 2023). The back book at the scheduled off is a median 1.02 in AU and 1.03-1.04 in NZ.

2025, AU market pre-play median by weekday: Wednesday 48,221, Saturday 43,401, Thursday 35,520, Friday 33,279,
Tuesday 32,985, Sunday 29,771, Monday 29,161 [B].

**[A] AU thoroughbreds** (the files' own pre-play window): runner median 1,852 (2012), 1,521 (2015), 2,408 (2018),
4,046 (2022, the peak), 2,132 (2025), 2,280 (2026); market median 49,081, 36,514, 55,774, 71,997, 40,614, 39,474.

## 3. Prices

**The pre-play average against the BSP** [B], median log(BSP / pre-play WAP), banded by the WAP:

| WAP band | AU 2020 | AU 2023 | AU 2025 | AU 2026 | NZ 2020 | NZ 2023 | NZ 2025 | NZ 2026 |
|---|---|---|---|---|---|---|---|---|
| 1-2 | -0.006 | -0.010 | -0.007 | -0.006 | -0.013 | -0.033 | -0.023 | -0.036 |
| 3-5 | +0.003 | +0.002 | +0.000 | +0.000 | +0.011 | -0.024 | -0.002 | -0.002 |
| 8-13 | +0.012 | +0.019 | +0.011 | +0.021 | +0.030 | +0.013 | +0.022 | +0.015 |
| 21-51 | +0.018 | +0.027 | +0.017 | +0.027 | +0.049 | +0.058 | +0.067 | +0.054 |
| 51+ | +0.073 | +0.104 | +0.077 | +0.063 | +0.075 | +0.163 | +0.158 | +0.133 |

The pre-play average is dominated by the money matched near the off, so it is close to the BSP by construction. It
cannot show the early move. The range of prices matched pre-play is wide (AU median log(max / min) 0.27-0.32 at
1-2, 0.44-0.53 at 5-8, 1.08-1.25 over 51, banded by the BSP) [B]. So there is a move to catch, but these files do
not show when it happens.

**The best back at the scheduled off against the BSP** [B]. A level back after 2% commission, pooled 2020-2026, at
each price, banded by the best back at the off:

| Band | AU runners | AU at the off | AU at the BSP | NZ runners | NZ at the off | NZ at the BSP |
|---|---|---|---|---|---|---|
| 1-2 | 15,039 | -1.7% | -0.9% | 1,124 | -6.5% | -4.6% |
| 2-3 | 44,770 | -1.3% | -1.1% | 4,264 | -5.9% | -5.3% |
| 3-5 | 123,223 | -2.4% | -1.5% | 15,907 | -3.8% | -2.2% |
| 5-8 | 154,224 | -3.5% | -2.4% | 24,897 | -4.5% | -1.8% |
| 8-13 | 179,405 | -2.6% | -1.1% | 31,726 | -2.8% | +0.7% |
| 13-21 | 165,816 | -3.7% | -2.5% | 30,889 | -7.6% | -3.5% |
| 21-51 | 225,839 | -4.7% | -2.9% | 40,002 | -11.2% | -4.3% |
| 51+ | 222,433 | -16.7% | -11.9% | 20,677 | -21.1% | -3.5% |

**A level back at the BSP, banded by the BSP, after 2% commission** (% a unit):

| BSP band | [A] AU, years positive of 15 (2012-2026) | [B] AU 2020-2026 pooled | [B] NZ 2020-2026 pooled (runners) |
|---|---|---|---|
| 1-2 | 4 | -1.0% | -2.3% (1,090) |
| 2-3 | 1 | -1.5% | -3.8% (4,126) |
| 3-5 | 2 | -1.4% | -1.2% (15,091) |
| 5-8 | 0 | -2.6% | -4.2% (23,659) |
| 8-13 | 3 | -1.2% | +3.7% (30,331) |
| 13-21 | 3 | -1.9% | -4.0% (29,881) |
| 21-51 | 3 | -3.5% | -2.4% (39,673) |
| 51+ | 0 | -11.0% | -7.8% (25,642) |

"Years positive" counts the cells of [A]'s after-commission table above zero (0.0 not counted). Year-by-year cells
are in the run log (section 6 of the query output).

## 4. What it means for the plan

- **Australia stays first, and NZ goes with it as a small add-on.** The Australian market is deep enough: 47% of
  runners carry 2,000 pre-play, and a GBP15-200 back is small against that. But this scan cannot show whether that
  money is there hours before the off, when our UK edge is taken.
- **The gate is the early price.** Neither free source has one. Next, at no cost: the owner takes Betfair Historic
  Data's free BASIC Horse Racing plan (`reports/international_racing.md` s.6 step 2). Its first-traded and T-60
  prices against the BSP answer whether an Australian price moves the way a UK one does.
- **Our archive gives 14 years of Australian BSPs and results.** That is the target and benchmark a walk-forward
  model needs from 2012, once a form source (Punting Form, with its written consent) supplies the inputs.
- **NZ's looser book is the better place for a model to find an edge, and the worse place to fill it.** It is
  worth recording alongside Australia, but at a median 325-601 a runner, stakes there would stay near the GBP15
  floor.

## 5. Limits

- No early prices: the move from first price to BSP is unmeasured here.
- [A]'s average prices for Australia are unusable (point 4); only its BSP, results and race counts are used.
- The NZ 8-13 cell is a multiple-comparison result (point 9); it is reported, not acted on.
- [B]'s 2026 runs to 31 Oct; Nov-Dec 2026 is not yet published.
- Currency: [B] presumably AUD, [A] unlabelled; neither is converted.
