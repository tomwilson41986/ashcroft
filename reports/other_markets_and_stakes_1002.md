# Other markets, and bigger stakes (the owner's questions of 2 Oct 2026)

The owner asked, the day after the first full live day:

1. now that the price files are archived, are there trading ideas in other markets: UK and Irish place markets,
   AvB (match bet) markets, other countries' racing;
2. should the to-win target go from GBP250 to GBP300, 350, 400 or 500, and is there the liquidity for it.

Research-query runs 36965830970, 36966868668, 36969508159 and 37070286949 (the stake replay settled on every horse); queries in `research/queries/done/`.

**In short.** The place market has one rule worth a forward test: it passes its gate at the account's 2%
commission, to be scored next from the recorded books 15 minutes before the off. Betfair offers no AvB match bets
on British and Irish racing (the census of 3 Oct's card, section 3). Other countries' racing has only Australia
with real liquidity, and no model of ours. The to-win target should stay at GBP250 while the account's money is
what binds; with more money the time of entry, not the size, is the lever (section 5).

## 1. What the archive holds (2 Oct, 04:46 UTC, after one night)

13,372 files in 16 markets. UK and Irish win and place (Betfair's "To Be Placed") go back to 31 Aug 2017, 3,319 days
each. The other twelve markets have their last eight days so far; from tonight the archive takes them from 2018,
newest first (betfair-prices.yml), before the older UK/IE files that are never loaded. Betfair dates each file by the
morning after its racing (`dwbfpricesukwin01102026.csv` holds 30 Sep's races), so a day's prices reach the archive the
evening after; the nightly load dates each race by its own start time, not the file's name.

| Market | Races a day | Traded a runner before the off (median) |
|---|---|---|
| UK win | 28 | GBP11,821 (morning GBP539) |
| UK place | 28 | GBP1,825 (morning GBP17) |
| Irish win | 11 | GBP4,743 (morning GBP100) |
| Irish place | 11 | GBP696 (morning GBP2) |
| Australia win | 81 | GBP1,180 pre-play, GBP1,825 "morning" (see below) |
| Australia place | 77 | GBP0 pre-play, GBP261 "morning" |
| Greyhounds win (UK and Australia) | 259 | GBP1,274 |
| France win | 16 | GBP256 |
| USA win | 69 | GBP300 |
| South Africa win | 10 | GBP615 |
| UAE, USA place | 0 | out of season or empty files |

Betfair publishes price files only for markets with a Betfair SP: win and place. **AvB match bets have no SP and no
files**; they can only be recorded live (section 3).

## 2. UK and Irish place markets: the place pocket passes its gate at 2%

On 24 Sep (ledger `place-pocket-gate`) one rule, chosen after looking at 2023-26Q1, was gated on 2021-22:

> back to place at the place BSP where the win BSP is 5.0 or shorter and the place chance the win BSPs imply
> (Harville-type ordering, exponents fitted on 2021-22 finishing orders) gives an expected return above 0.05 after
> commission

At 5% commission it failed (2021 -0.4%, pooled 2021-22 interval touching zero), and the ledger said the account's
commission rate would decide whether it was worth a forward test. The account pays 2% (1 Oct). Read again at 2%,
the same rule and the same criterion set before the first run (2021 and 2022 both positive, the pooled 2021-22 90%
interval above zero):

| Year | Bets | Return at the place BSP | 90% interval |
|---|---|---|---|
| 2021 | 4,030 | +0.3% | -1.9% to +2.4% |
| 2022 | 3,941 | +3.2% | +1.0% to +5.4% |
| **2021-22 (the gate)** | **7,971** | **+1.7%** | **+0.2% to +3.3%: PASSES** |
| 2023 | 3,971 | +2.4% | +0.1% to +4.6% |
| 2024 | 4,280 | +1.8% | -0.3% to +3.8% |
| 2025 | 4,482 | +3.2% | +1.2% to +5.2% |
| 2026 Q1 | 1,096 | +5.6% | +1.6% to +9.6% |

About 11 bets a day; every runner at win BSP 5.0 or shorter, for scale, returns -0.5% (2021-22). At 5% the same
years read 2021 -0.4%, 2022 +2.6%: the 3 points of commission are the difference.

The BSPs are known only at the off, so the rule as gated cannot be traded as it stands. At prices before the off
(Betfair's files inside the development window, 21 Jan - 31 Mar 2026; the holdout not read):

| Prices the rule reads | Bets | Paid at that price | Paid at the place BSP | Place volume on the runners chosen |
|---|---|---|---|---|
| Morning (MORNINGWAP) | 409 | +11.0% | +0.7% | median GBP6: not tradable |
| Pre-play (PPWAP, most of it the last hour) | 403 | +3.7% (-3.5 to +10.9) | +7.0% (-0.6 to +14.6) | median GBP5,280 |

Ten weeks cannot resolve the pre-play figures, but they point the same way. **The candidate**: 15 minutes before
the off (the owner's limit), read the win and place books, take the place chance from the win prices, and back at
the place SP (a MARKET_ON_CLOSE back on the place market) where the expected return at the place market's price is
above 5% after 2%. The recorder keeps the place books each minute in the last hour from 1 Oct, so this exact rule
can be scored on the days recorded before any money is put on it.

## 3. AvB markets

Not in the price files. `betfair_recorder.py --census` lists every market type Betfair offers on the next day's GB/IE
racing, each evening on the UK server (live-record.yml). Its first reading, on 2 Oct for 3 Oct's card (a Saturday at
Newmarket, Ascot and Gowran Park, among others), found five types and no match bets:

| Market type | Markets | e.g. |
|---|---|---|
| WIN | 44 | Newmarket: 1m2f Hcap |
| PLACE | 43 | Newmarket: To Be Placed |
| OTHER_PLACE | 65 | Newmarket: 2 TBP, 3 TBP, 4 TBP |
| EACH_WAY | 39 | Newmarket: Each Way |
| ANTEPOST_WIN | 2 | Sun Chariot Stakes |

Betfair does not offer AvB markets on ordinary British and Irish racing, so there is nothing to record or model. The
census keeps running each evening; if a match-bet market ever appears, the recorder can keep it, and the model already
prices the ordering (P(A finishes ahead of B) from our win chances and the finishing-order exponents above). The census
does show two place-type markets the programme had not listed: OTHER_PLACE (a fixed number of places, 2 to 4) and
EACH_WAY. Betfair's price files cover only the win and To Be Placed markets, so either of these would have to be
recorded first.

## 4. Other countries' racing

- **Liquidity**: only Australia trades enough to matter (GBP1,180 a runner pre-play, 81 races a day). France, the
  USA and South Africa trade a few hundred pounds a runner.
- **No model of our own**: our model is built on UK and Irish form (horseracebase), so other countries allow only
  market-only rules for now.
- **The market-only baseline** (`market_only_drift`, UK win 2023-26Q1, 280,120 runners, placed by the morning price,
  which is known at the time): there is no drift to catch. Morning favourites +0.5% CLV against the BSP, every
  other band negative, long shots -24%; only runners with GBP2,000+ traded in the morning show +1% before commission.
  The live rule's +9% CLV on 1 Oct is the model's, not a pattern of the market.
- **Australia and the greyhounds**: the files' morning price is below the BSP on 94-98% of runners, which no market
  does before a race: their racing runs overnight UK time and Betfair's "morning" window includes trading after the
  off. A study there must read the pre-play price, with the backfill from tonight.
- Australian racing runs from about 01:00 to 09:00 UK, while the UK server archives from 22:30 to 06:15 UTC; trading
  there would need the archive moved or another server.

## 5. Bigger stakes: keep GBP250 to win for now

1 Oct replayed minute by minute through the books the trader and the recorder kept: the same rule and closing model,
each target from GBP250 to GBP500, the account's money held as Betfair holds it (a back larger than the funds is
refused; a race's stakes come back after it is run), and each back filled against the book as recorded. Every replay
is settled on every horse from Betfair's price file for 1 Oct's racing (`dwbfprices*02102026`, archived 2 Oct 19:38
UTC; research-query run 37070286949), each back laid at the SP, 2% commission. The first reading (run 36969508159),
before the file was published, scored only the horses the real day backed; its answer stands, but with more money
the picture changed, below. One day: read the CLV, not the result.

**With the money the account had** (GBP1,480.11 at the start of 1 Oct):

| To win | Horses | Staked | CLV | CLV x stake | Result after 2% |
|---|---|---|---|---|---|
| **GBP250** | 125 | GBP2,936 | **+7.4%** | **GBP214** | +131.68 |
| GBP300 | 111 | GBP2,988 | +5.8% | GBP170 | +80.94 |
| GBP350 | 105 | GBP3,202 | +5.6% | GBP178 | +89.79 |
| GBP400 | 100 | GBP3,319 | +5.5% | GBP180 | +111.15 |
| GBP500 | 91 | GBP3,534 | +4.6% | GBP162 | +92.43 |

The GBP250 replay stands for the real day (124 horses, GBP2,733, +9.2%, +188.19). Every bigger target is worth less:
it spends the balance sooner, on fewer horses, and more of each horse's stake goes in as top-ups after the first
fill. Top-ups are taken at worse prices and earn less: on 1 Oct each horse's first fill made +13.8% CLV against +6.0%
for the top-ups, which were 2.8% shorter; on 2 Oct +6.1% against nothing (TRADING.md). At GBP500 two thirds of the
stake is top-ups.

**With more money, under the GBP4,000 day limit**, every target stakes the GBP4,000, and a bigger one spends it sooner:

| To win | Horses (GBP3,000 / GBP5,000 in the account) | CLV x stake (GBP3,000 / GBP5,000) |
|---|---|---|
| GBP250 | 157 / 157 | GBP159 / GBP159 |
| GBP300 | 133 / 133 | GBP182 / GBP182 |
| GBP350 | 124 / 119 | GBP212 / GBP177 |
| GBP400 | 110 / 100 | GBP214 / GBP212 |
| GBP500 | 94 / 81 | GBP271 / GBP179 |

Settled on every horse, the bigger targets now earn more than GBP250 here. The first reading, which left out the
horses only the replays backed, had the five level (GBP174-194). The reason is when the money goes, not how much goes on
each horse. With GBP4,000 to spend, GBP250 reaches the afternoon: its last back is at 20:02 UK, and the 35 horses the
real day never reached for want of funds earn little. A bigger target uses up the day limit by 12:23-15:37 UK, on the
morning's horses, and backs nothing later. Both live days say the same: horses first backed by 11:00 UK +6.8% CLV,
after 11:00 +0.5% (TRADING.md). So the lever is the entry time and the top-ups, not the size. Raising the target would
take the afternoon out only as a side effect, and only once the account holds GBP3,000 or more.

**With the limits lifted** (GBP600 a bet, GBP8,000 a day), GBP400 stakes GBP6,072 (CLV x stake GBP209) and GBP500
GBP7,365 (GBP227). The account needs GBP3,794 and GBP4,540 at the busiest moment, against GBP2,508 at GBP250. The stake
over GBP250's earns +2.4% and +2.0% CLV, before the 2% commission on winnings.

**Liquidity.** Not what stops a bigger target, but thin where the stakes are biggest. Most of the horses the rule
backs are long prices (82 of 128 on 1 Oct at 12.0 or more), and there the size at the best back (GBP11-14 at the
first poll on the delayed key) covers a GBP250 target at once and a GBP500 one after a top-up or two. At 2.5-7.0 the
target needs GBP52-123 against GBP14 shown, and is reached over many minutes. Over the day the book took 88% of horses
to within 5% of a GBP250 target, and 79-83% of a GBP400-500 one with the limits lifted.

**The recommendation.** Keep GBP250 to win while the account's money is what binds (GBP1,712.60 on 2 Oct). At the
account's balance GBP250 earns the most. With more money a bigger target would earn more, but for a reason a smaller
change gets at directly: it stops the trader before the afternoon's weaker entries. The next replays test that change
directly on the recorded days. They also test top-ups held to a higher bar than a new horse, and a new horse given the
money before a top-up when funds run short. The owner decides any change to the rule.
