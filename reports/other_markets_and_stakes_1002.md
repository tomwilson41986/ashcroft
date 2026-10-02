# Other markets, and bigger stakes (the owner's questions of 2 Oct 2026)

The owner asked, the day after the first full live day:

1. now that the price files are archived, are there trading ideas in other markets: UK and Irish place markets,
   AvB (match bet) markets, other countries' racing;
2. should the to-win target go from GBP250 to GBP300, 350, 400 or 500, and is there the liquidity for it.

Research-query runs 36965830970, 36966868668 and 36969508159; queries in `research/queries/done/`.

**In short.** The place market has one rule worth a forward test: it passes its gate at the account's 2%
commission, to be scored next from the recorded books 15 minutes before the off. AvB match bets are not in the price
files, so they must be recorded live; a census of tomorrow's market types runs each evening. Other countries' racing
has only Australia with real liquidity, and no model of ours. The to-win target should stay at GBP250.

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

Not in the price files. `betfair_recorder.py --census` (added today) lists every market type Betfair offers on the
next day's GB/IE racing, each evening on the UK server; once it shows the match-bet code and how many there are,
the recording keeps them. The model already prices the ordering: P(A finishes ahead of B) follows from our win
chances and the finishing-order exponents above, so a few weeks of recorded AvB prices and results are enough to
test it. AvB markets have no SP to lay off at, so a bet stands to the result: more variance per pound than the win
rule, which keeps the price move whatever the result.

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

1 Oct replayed minute by minute through the books the trader and the recorder kept (research-query run 36969508159):
the same rule and closing model, each target from GBP250 to GBP500, the account's money held as Betfair holds it (a
back larger than the funds is refused; a race's stakes come back after it is run), and each back filled against the
book as recorded. Betfair's price file for 1 Oct's racing is not yet published (it is dated 2 Oct), so every replay
is scored on the 122 horses the real day backed, from the real day's settled bets; a horse only a replay backed is
left out (7-10 horses at the day's balance, up to 38 with more money). One day: read the CLV, not the result.

**With the money the account had** (GBP1,480.11 at the start of 1 Oct; GBP1,668.30 now):

| To win | Horses | Staked | CLV | CLV x stake | Result after 2% |
|---|---|---|---|---|---|
| **GBP250** | 125 | GBP2,936 | **+8.0%** | **GBP215** | +157.88 |
| GBP300 | 111 | GBP2,988 | +6.4% | GBP172 | +112.52 |
| GBP350 | 105 | GBP3,202 | +6.2% | GBP181 | +127.31 |
| GBP400 | 100 | GBP3,319 | +6.1% | GBP183 | +154.14 |
| GBP500 | 91 | GBP3,534 | +5.0% | GBP158 | +136.20 |

The GBP250 replay stands for the real day (124 horses, GBP2,733, +9.2%, +188.19). Every bigger target is worth less:
it spends the balance sooner, on fewer horses, and more of each horse's stake goes in as top-ups after the first
fill. Top-ups are taken at worse prices: on the real day each horse's first fill made +13.8% CLV against +6.0% for
the top-ups, which were 2.6% shorter. At GBP500 two thirds of the stake is top-ups.

**With more money, under the GBP4,000 day limit**, every target stakes the GBP4,000, and a bigger one spends it sooner:

| To win | Horses | Last back (UK) | CLV x stake |
|---|---|---|---|
| GBP250 | 157 | 20:02 | GBP174 |
| GBP300 | 133 | 15:37 | GBP194 |
| GBP350 | 119 | 14:26 | GBP182 |
| GBP400 | 100 | 13:32 | GBP193 |
| GBP500 | 81 | 12:23 | GBP182 |

The same expected value spread over fewer horses: more variance for nothing. At GBP500 the afternoon's races go
unbacked. (Scored on the horses the real day backed: the GBP250 row leaves out 35 horses, GBP932 of stake, that the
real day never reached for want of funds; GBP500's leaves out 10. Betfair's file decides those this evening.)

**With the limits lifted** (GBP600 a bet, GBP8,000 a day), GBP400 stakes GBP6,072 and GBP500 GBP7,365 a day. The
account needs GBP3,794 and GBP4,540 at the busiest moment, against GBP2,508 at GBP250. The stake over GBP250's earns
about +2.5% CLV, against +5.7% on the first GBP4,000, before the 2% commission on winnings.

**Liquidity.** Not what stops a bigger target, but thin where the stakes are biggest. Most of the horses the rule
backs are long prices (82 of 128 on 1 Oct at 12.0 or more), and there the size at the best back (GBP11-14 at the
first poll on the delayed key) covers a GBP250 target at once and a GBP500 one after a top-up or two. At 2.5-7.0 the
target needs GBP52-123 against GBP14 shown, and is reached over many minutes. Over the day the book took 88% of horses
to within 5% of a GBP250 target, 79-83% of a GBP400-500 one with the limits lifted.

**The recommendation.** Keep GBP250 to win. A bigger target pays only with more money and a higher day limit, and
even then the extra stake earns half the CLV of the first. One day decides nothing finally: the replay is scored again
on every horse when Betfair publishes 1 Oct's file (this evening), and on each day the recorder keeps. With the
first fills earning twice the top-ups, a better use of short funds is likely to give a new horse priority over a
top-up when the money runs low; that is the next replay, not a change yet.
