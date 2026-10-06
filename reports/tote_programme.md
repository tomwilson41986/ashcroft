# The Tote against Betfair

The owner's ask (6 Oct 2026): integrate the Tote, all its horse-racing pools (win, place, exacta, trifecta,
swinger), and back on the Tote with a lay on Betfair when the prices part; "just access the tote prices and
compare to betfair". This note says what the comparison measures and how it is being built. Ledger `tote-design-1006`.

## The pools

From the Britbet horse racing pool rules (October 2025), the rules of the UK pools:

| Pool | Deduction | Races | Pays on |
|---|---|---|---|
| Win | 19.25% | 2+ runners | the winner |
| Place | 20% | 5+ runners | 2 places at 5-7 runners, 3 at 8-15, at 16+ 3 (4 in a handicap) |
| Exacta | 25% | 3+ runners | 1st and 2nd in order |
| Trifecta | 25% | 3+ runners | 1st, 2nd and 3rd in order |
| Swinger | 30% | 6+ runners | two horses both in the first three; a third of the net pool to each of 1st+2nd, 1st+3rd, 2nd+3rd |

A horse taken out before coming under orders is refunded: a Tote bet carries no reduction factor. Two terms
apply to bets placed direct at tote.co.uk, from articles of 2021-25 and to be confirmed against the current
terms from the UK: the **Tote Guarantee** (a win bet pays at least the industry SP) and **Tote+** (10% more on win
and place dividends over 1.20, 5% on the exotics).

## What is compared

**Win and place, against a Betfair lay.** Back S on the Tote and lay S/(1 - c) on Betfair at price L. If the horse
loses, the lay's winnings pay the Tote stake and the pair nets nothing. If it wins, the pair nets S x (D - b), where
D is the Tote's return per unit staked (after Tote+ and the guarantee) and b = 1 + (L - 1)/(1 - c) is the
break-even (10.18 at L = 10 and the owner's 2% commission). So the pair makes money only on winners, and only when
the Tote pays more than b. Laid at Betfair's SP, both prices are fixed only at the off: this is not an arbitrage
but a bet that the Tote pays more than the BSP. The Tote's own deduction (19.25%, less 10% back through Tote+)
means a horse the Tote's punters price as Betfair's do pays about 11% under the BSP; the pair pays when they back
it markedly less than Betfair does. How often that happens, how far ahead of the off it can be seen, and how much
the pool moves in the last minutes are what the record measures.

**Exacta, trifecta, swinger.** Betfair lists no such markets on GB/IE racing (the census of 6 Oct: win, place,
other place, each way), so these are compared with the fair dividend Betfair's win prices imply through
`model/ordering.py` (Harville, or Benter's flattened exponents once fitted on our results): a combination's
probability p against the Tote's dividend D for it, a unit bet's expected return being p x D. There is no lay; it
is a bet against the pool, and the deduction is 25-30%.

The arithmetic is `tote_compare.py` (tested in `tests/test_tote_compare.py`).

## How it is built

1. **The probe** (`tote_probe.py`, `tote-probe.yml`). tote.co.uk answers "Unavailable in Region" outside the UK, so
   it is read from the UK server, read-only: no login, nothing posted, robots.txt kept, at most 40 requests two
   seconds apart. It saves what the racecard and results pages serve and the addresses of the data behind them.
   The runner serves the trader, so the probe waits behind the day's session.
2. **The record.** From what the probe finds: the Tote's prices for every GB/IE race and pool through the day to
   the off, and the declared dividends, raw to S3 beside the Betfair record. On trading days it runs inside the
   trader's job (one runner), as the evening record does.
3. **The comparison.** Daily: each runner's Tote price against Betfair's best lay at the same minute, and the
   declared dividends against the BSP and the Betfair-implied fair dividends, by price band and time to the off.
4. **Betting.** Not before the record shows a gap that holds to the off. The Tote has no public betting API for
   account holders that we know of, and betting through the website by machine would put the account against
   the Tote's terms; the owner decides how, and whether, Tote bets are placed.

No login is needed for any of this. The owner's Tote password was shared in chat on 6 Oct: it is not stored or
used anywhere here, and should be changed.
