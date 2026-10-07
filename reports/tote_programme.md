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

## What the Tote serves (the probes of 6 Oct)

Three read-only probes from the UK server (`tote_probe.py`; runs 37442604100, 37449057994) found:

- **The site.** tote.co.uk is a single-page app: every page is the same shell. Its scripts read a REST API whose
  address and public client key come from `/config.js`, a file every visitor's browser is served. robots.txt
  disallows only `/account`. The key is read at run time and kept out of every file and log line.
- **`race-card/pools/today`.** Every pool open today, with its total, status and race. On 6 Oct there were 86 win,
  86 place, 68 exacta, 68 trifecta, 41 swinger and 18 quinella pools, plus placepots and jackpots. They cover 35
  GB/IE races (the same 35 Betfair priced), and the US and French meetings besides.
- **`race-card/pool/<id>`.** The pool and its race's runners.
  - The win pool gives each runner's approximate dividend from the pool, per unit staked (`baseWinStake`), and
    the figure the site shows (`winstake`). The site's figure is the better of the pool's dividend and the
    bookmakers' current show price: that is the Tote Guarantee. The pool's dividends summed to a book of
    124.8% (1/(1 − 19.25%) is 123.8%, and dividends are rounded down).
  - The place pool does the same (`basePlaceStake`, `placestake`).
  - The exacta, trifecta and swinger pools give their totals only: no combination's dividend before the off.
- **`race-card/race-results/<race ids>`.** The declared dividends (the site's results module). Its answer is read
  on the first evening of the record.

## How it is built

1. **The probe** (`tote_probe.py`, `tote-probe.yml`): done (above).
2. **The record** (`tote_recorder.py`, from the session of 7 Oct inside the trader's job, one runner). It is
   read-only, at most one request a second, about 1,000 requests and 2 MB a day:
   - every pool's total, every 10 minutes;
   - each GB/IE race's win and place pools at 60, 30, 15, 10, 5, 3, 2 and 1 minutes before the off and 2 and 5
     minutes after it;
   - the exacta, trifecta, swinger and quinella pools at 5 minutes before and after;
   - the declared dividends at 10 and 30 minutes after the off.

   The raw answers go to s3 `tote/live/<day>/`.
3. **The comparison.** Daily: each runner's Tote price against Betfair's best lay at the same minute, and the
   declared dividends against the BSP and the Betfair-implied fair dividends, by price band and time to the off.
4. **Betting.** Not before the record shows a gap that holds to the off. The Tote has no public betting API for
   account holders that we know of, and betting through the website by machine would put the account against
   the Tote's terms; the owner decides how, and whether, Tote bets are placed.

No login is needed for any of this. The owner's Tote password was shared in chat on 6 Oct: it is not stored or
used anywhere here, and should be changed.

## The first comparison (6 Oct, from 12:56 UTC)

`research/queries/done/tote_vs_betfair.py`, ledger `tote-compare-1006`. The record began with the restarted
trading session, so it covers 32 GB/IE races. Every one was matched to its Betfair win and place markets by course,
off time and cloth number.

- **Before the off the Tote is dear.** At every mark from an hour out to a minute out, the win pool's dividend is a
  median 26-30% below the break-even of a Betfair lay at the same minute, and the place pool's 25-36% below. The
  figure the site shows with the Tote Guarantee is 16-20% below. Only 9-18% of runners show any edge at a mark.
- **Late money moves the dividends.** The median win pool holds £339 an hour before the off, £2,488 five minutes
  before, £4,127 a minute before and £7,797 just after it: about half the pool arrives at the off. A runner's
  dividend at a minute before ends anywhere from 0.77x to 1.36x of it (the middle 80%), and a third fall 10% or more.
- **The edges seen before the off do not hold.** The pairs that showed more than 5% (back on the Tote, lay on
  Betfair at the same minute) lost 22% to 66% a unit once paid at the declared dividend: 20-42 pairs a mark, 3-7
  winners.
- **The guarantee.** The win pool's listed dividend is the guaranteed figure (at least the industry SP) in all 35
  races. Against the BSP, the guarantee is the one structural feature worth testing. That test, and the exotics
  against their fair dividends, need the BSPs, which the day record did not hold (ledger `recorder-bsp-1006`; the
  recorder reads them at the off from 7 Oct).

So far, no gap between the Tote and Betfair survives to the declared dividend. The comparison runs again on 7 Oct,
the first whole day with the BSPs.

## The first whole day (7 Oct), and 6 Oct with its BSPs

`tote_vs_betfair.py` again (ledger `tote-compare-1007`). 7 Oct is the first whole day on the record: 38 races, each
matched to its Betfair win and place markets, with Betfair's place terms recorded. 6 Oct's price files brought its
BSPs (628 runners), so the parts against the BSP ran on 6 Oct.

- **Before the off, the same picture as 6 Oct.** On 7 Oct the win pool's dividend is a median 27-28% below a Betfair
  lay's break-even at every mark, 18-22% with the guarantee. The place pool's is 27-36% below.
- **Late money moves the dividends.** A runner's dividend at the off is a median 1.05-1.08 times its dividend at a
  mark. The middle 80% runs from about 0.6-0.7 to 1.7-2.1 times, and about a third fall 10% or more.
- **The pairs still lose.** Backing on the Tote and laying on Betfair wherever the edge cleared 5% lost 6% to 58% a
  unit at every mark, and 6% to 48% with the guarantee (22-72 pairs a mark, 5-10 winners).
- **Against the BSP (6 Oct).**
  - The near-final win dividend is a median 22% below the break-even of a lay at the BSP. Only 4% of runners are
    above it.
  - On the 32 winners the pool's dividend was a median 0.825 of the BSP and the guaranteed one 0.912 (the industry
    SP 0.884). A Betfair back at the BSP pays about 0.98 of it after 2% commission.
  - The guarantee is real (the listed dividend is the paid figure in every race), but it does not lift the Tote to
    the BSP.
- **The exotics (6 Oct).** Against their fair value under the BSP's chances, the exacta paid a mean 0.78 (9 races)
  and the trifecta 0.75 (9 races). The swinger's 1.06 over 7 races (median 0.71) needs more days.

No gap between the Tote and Betfair survives to the result. 7 Oct's parts against the BSP come with 8 Oct's price
file: no read after the off has yet carried a BSP (ledger `bsp-at-off-1007`).
