# The Professor's selection rule: the backtest (Phase 0, 23 Oct 2026)

The owner asked (23 Oct) for an automated tipster on the model, "The Professor" (reports/professor_design.md): the
horses "we expect to win frequently and shorten most", a high strike rate, "backing at higher bookmaker prices than we
think the final BFSP is going to be", at most 10 tips a day, the top two free, nothing over 50/1. This is the
backtest that sets the rule's parameters, before anything is published: research-query runs 39950331744
(`research/queries/done/professor_rule_1023.py`) and 39950972382 (`professor_rule_1023b.py`), ledger
`professor-rule-1023`.

## The rule, and the decision

Among the horses the syndicate holds at 09:55 UK (the live rule's own backs), advise each whose bookmaker price
clears the closing model's expected BSP by **10%**, **one a race**, ranked by the expected win probability
(`p_win`, the closing model's expected 1/BSP), **at most 5 a day**; ranks 1-2 are the free tips. No floor on `p_win`
beyond the ranking, and the owner's cap of 50/1.

| `professor/config.json` | value | why |
|---|---|---|
| `overlay_bar` | 0.10 | the implied return is flat or negative under 10%, positive and steady from 10% (below) |
| `min_p_win` | 0 | the rank by `p_win` already puts the short prices first; a floor only cuts the tips a day |
| `max_per_race` | 1 | the second horse of a race is always the longer price with the lower `p_win` |
| `max_tips` | 5 | ranks 6-10 return nothing: implied +2%, realised -10%, strike 11% (below) |
| `free_tips` | 2 | the owner's decision |
| `max_price` | 51.0 | the owner's decision (50/1) |

The owner asked for up to 10 a day. The cap is 5 because the 6th to 10th horses of a day add turnover and no return
on either window; the owner can raise it (`max_tips`) and the record will show what they add.

## What the rule is expected to make, and what it made

Every cell was scored held to the result at 1 point level stakes at a proxy for the bookmaker's price (no bookmaker
price is on record: the Betfair price in the morning stands in, with haircuts of 5-15% for a bookmaker's price under
the exchange), and at the BSP. The **BSP-implied ROI** (the mean of price / BSP - 1, the CLV) is what the tips return
if the horses win as the BSP says; the realised ROI differs from it by luck. On a few hundred tips the luck is the
larger part, so the parameters were chosen on the implied return, and the realised one is reported beside it.

| Population, price | Tips | A day | Strike (expected) | Avg price | Implied ROI | Realised ROI (90% CI) | Implied at 5% / 10% / 15% under the exchange |
|---|---|---|---|---|---|---|---|
| B. Aug-Sep history, morning WAP (the window the rule was set on) | 280 | 5.0 | 23.2% (21.2%) | 5.2 | **+4.8%** | +10.9% (-10 to +33) | +0.6% / -3.5% / -7.7% |
| B. Oct history, morning WAP (the holdout) | 104 | 5.0 | 23.1% (20.6%) | 6.6 | **+10.1%** | +18.9% (-21 to +57) | +5.6% / +1.1% / -3.4% |
| B. Aug-Oct | 384 | 5.0 | 23.2% (21.0%) | 5.6 | +6.2% | +13.1% (-4 to +30) | +2.0% / -2.3% / -6.5% |
| A. live record 1-22 Oct, the trader's own first price | 106 | 4.8 | 18.9% (19.0%) | 8.5 | **+27.5%** | +27.8% (-23 to +85) | +22.0% / +16.6% / +11.2% |
| A. live record, the Betfair price at 10:00 UK | 103 | 4.7 | 17.5% (16.9%) | 8.5 | **-6.4%** | -17.8% (-49 to +16) | -10.2% / -14.1% / -17.9% |

- **Free tips (ranks 1-2)**: strike 27% (expected 25%), average price 4.5, implied +2%, realised +15% over Aug-Oct
  (Aug +20%, Sep -3%, Oct +36%); on the live record +59% realised on 44 tips (implied +28%).
- **The industry SP loses 29-55% everywhere.** A follower who takes the SP, or a bookmaker's price near it, loses;
  the tip is the early price or nothing. Best Odds Guaranteed adds 0-1 point (the horses shorten).
- **The price the trader gets is not the price the public gets.** The live record at the trader's own first price
  reads +27% implied; at the Betfair price at 10:00 UK, after the morning's move (ours among it), -6%. The bookmaker
  price at 10:00 will sit between the exchange's and the haircuts. This is the open question only the proofing period
  answers; the rule's expected return at a real bookmaker price at 10:00 is **about 0 to +5%**, and the 20 Oct QA
  found the morning prices nearer the close since 14 Oct.

## What the grid showed

- **The rank by `p_win` is the rule.** The same cell ranked by the overlay or by the expected CLV picks the long
  prices (average 28 against 5.6): strike 5%, realised -12%, worst run 126 (B, Aug-Oct); on the live record -28% to
  +15% with a worst run of 27-29. The owner's "win frequently" is right: the shortest-priced qualifiers are the tips.
- **Where the value sits by rank** (B, Aug-Oct, the chosen bars, cap 10): rank 1 strike 31%, implied +7%; rank 2
  23%, -3%; rank 3 22%, +9%; ranks 4-5 20%, +9%; **ranks 6-10 11%, +2%, realised -10%** (average price 9.4). On the
  live record ranks 6-10 read +36% implied at the trader's price (average 18.8), the long end the trader buys well
  and the public cannot.
- **The overlay bar** (the 5-tip cells, Aug-Sep): at 0-5% the implied return is +2.4%; at 10% +4.8%; at 15-20%
  +4.2-5.7% with a lower strike (15-16%) and, on the holdout, fewer tips a day (4.5 and 3.0); at 25% nil. 10% keeps
  the strike and the volume.
- **No selection**: every candidate of the live rule held at the morning WAP, Aug-Oct, 110 a day, strike 8.6%,
  realised -2.3%, implied -1.2%, worst run 134, drawdown 530 points. The selection is the whole product.
- **The calibration of `p_win` on the candidates** (B): the horses the model expects to shorten win less than their
  `p_win` says at `p_win` under 0.2 (0.73-0.87 of expected: the candidates are, by construction, the horses the
  market has longer than we do) and as it says above 0.2 (0.96-1.04). The strike rates above are the realised ones.
- **The pre-registered decision rule** (the design: the highest realised ROI on Aug-Sep with the holdout's BSP ROI
  positive and the free strike at least 20%) picked two tips a day at a 20% overlay and `p_win` >= 12%: +49% realised
  on 105 tips, of which +5% implied and **+44% luck** (26 winners where 19 were expected). The cells beside it swing
  from -15% to +57%. The rule was revised in the second run to the implied return with at least 4 tips a day, which
  is the decision above.
- **By price band** (the chosen cell, B): under 4.0 strike 41%, realised +35%, implied +6%; 4-8 (three quarters of
  the tips) strike 20%, +4%, implied +5%; 8-16 strike 18%, +62%, implied +10%; over 16 three tips, all lost.

## The data

- **A, the live record**: 2,885 positions on 22 days (1-22 Oct), 2,784 settled with a BSP, 74% first backed by 09:55
  UK (2,071, the tipster's population), every one with a 10:00 mark, 69% with an industry SP. Held at the trader's
  first price with no selection: 83 a day, strike 13%, realised +24% (CI +8 to +42), implied +16%.
- **B, the history**: 2,935 whole races, 27,731 runners, 1 Aug - 21 Oct (the 04:00/06:00 files from S3 joined to
  Betfair's price files and race_results by track, time and horse); the live rule's candidates rebuilt with the
  no-volume closing model at the morning WAP (expected CLV >= 3%, at or over our price, <= 200): 10,837 horses on 77
  days, 141 a day, strike 7.0%; 8,449 at 50/1 or under.
- The bootstrap intervals are over tips; the free tips are 2 a day, so a month is about 50 tips and its realised ROI
  swings by +/- 40 points. The proofing period's record will be read on the implied return first.

## What follows

Phase 1 of the design: the package, the odds scrapers, the free Telegram channel and the proof page, with these
parameters in `professor/config.json`. The first thing the proofing period must show is the bookmaker price at 10:00
against the Betfair price at 10:00 and the trader's first price: that gap decides whether the public tips carry the
+5% the history implies or the -6% the 10:00 mark implies.
