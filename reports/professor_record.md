# The Professor: the record of the dry runs, and the bookmakers' margin (25 Oct 2026)

The owner asked (25 Oct) how the tips did and whether there are any for today. Nothing has been published: these
are the dry runs' tips, settled from the trader's own ledger (every tip is a horse the syndicate holds, so the
ledger's settle row carries its result and the Betfair SP: `python -m professor --settle`, professor-publish.yml run
40177634961). 1 point level stakes; the settled rows are in `reports/professor_record/`.

## The tips, settled

| Day (card read at) | Race | Horse | Tier | Advised | Result | BSP | CLV | Points at advised |
|---|---|---|---|---|---|---|---|---|
| 23 Oct (14:30) | 15:00 Cheltenham | Planters Punch | free | 5.50 | lost | 5.50 | 0% | -1.00 |
| 23 Oct | 18:30 Wolverhampton | Way To Dubai | free | 15.0 | lost | 12.50 | +20% | -1.00 |
| 23 Oct | 14:25 Cheltenham | Loriko | private | 17.0 | lost | 25.63 | -34% | -1.00 |
| 23 Oct | 14:00 Sligo | Beat The Devil | private | 29.0 | lost | 64.54 | -55% | -1.00 |
| 23 Oct | 14:10 Doncaster | Dark Thirty | private | 51.0 | lost | 110.0 | -54% | -1.00 |
| 24 Oct (11:45) | 15:30 Cheltenham | North Shore | free | 5.00 | **won** | 4.30 | +16% | **+4.00** |
| 24 Oct | 14:05 Doncaster | Hi Corbett Court | free | 5.50 | lost | 4.42 | +24% | -1.00 |
| 24 Oct | 19:00 Southwell | African Spirit | private | 8.00 | lost | 8.06 | -1% | -1.00 |
| 24 Oct | 17:30 Southwell | Wyvill | withheld | 8.00 | lost | 8.33 | -4% | (not published) |

| | Tips | Winners | Points at advised | At the BSP | Mean CLV |
|---|---|---|---|---|---|
| 23 Oct | 5 | 0 | -5.00 | -5.00 | -24.5% |
| 24 Oct | 3 | 1 | +2.00 | +1.23 | +13.3% |
| Both | 8 | 1 | -3.00 | -3.77 | -10.3% |

Eight tips say nothing about the rule's worth. Two things they do show: the afternoon card of 23 Oct picked long
prices that drifted (three of them to double their advised price at the off), and the late-morning card of 24 Oct
beat the SP on two of its three.

## Today, 25 Oct: no tips at 08:45 UK

- **Aintree is abandoned** (Racing Post's index flags all seven races): 23 of the trader's 85 horses are in them and
  their bets are void. From this commit the record says so.
- **The other 62 were priced** by Racing Post (22 races at Galway, Market Rasen and Wexford, 1,942 quotes) and **none
  cleared the bar**. The nearest: Livin On Luco (14:41 Market Rasen) at 11/4 against an expected SP of 3.49, +7.6%;
  then Character Testing and Beau De Maine (+2.6%, +2.5%). The median horse sat 27% under its expected close at the
  bookmakers and 31% under the trader's fill.

## Why so few: the bookmakers' margin

The three dry runs' snapshots show what the backtest could not: the bookmakers' book on our races.

| Card read at | Races | One bookmaker's book (median) | The best price of each horse | The advised price (3rd-best of the majors) |
|---|---|---|---|---|
| 23 Oct 14:30 | 33 | 124% | 118% | 122% |
| 24 Oct 11:45 | 31 | 116% | 112% | 114% |
| 25 Oct 08:45 | 22 | 122% | 115% | 121% |

The Betfair SP book is about 100-102%. So the price three major bookmakers offer sits, on the average horse, about
**14-21% under its fair closing price** before any move at all. The backtest stood the bookmakers' price in with a
haircut of 5-15% on the exchange's morning price, and at 15% the rule's implied return was already -6.5%
(reports/professor_rule.md). The real haircut is at the top of that range or past it. A horse clears 10% over the
expected close at the bookmakers only when it is priced some 30% better than the average runner: a few a day at
most, and on the dry runs' evidence often the long prices about to drift.

## What it means for the service

- **At bookmaker prices the rule is thin and the volume small**: 5, 3 and 0 tips on the three days, the expected
  return near zero by the backtest's own haircut table. This is the structural cost of the bookmakers' margin, not a
  matter of timing: today's card, read at 08:45, two hours after most of the fills, found the gap as wide as ever.
- **At the exchange price the rule has the edge the record shows** (the trader's fills: +27% implied on the live
  record). The owner's word (23 Oct): "you can use BF price. But really we want to find horses that are with bookmaker
  prices." The choice is the owner's: tips at the exchange price with the bookmakers shown beside it when they match
  (more tips, the edge kept, followers need an exchange account), or bookmaker-only tips (few, thin).
- Either way, the alert loop on the UK runner at 06:15 is the next test: it reads the bookmakers at the minute of
  each fill, which no dry run here has done.

## 26 Oct: the card at 09:25 UK, settled (research-query run 40349529269)

| Race | Horse | Tier | Advised | Result | BSP | CLV | Points at advised | At the BSP |
|---|---|---|---|---|---|---|---|---|
| 13:55 Redcar | Merci Monsieur | free | 9/4, 3.25 (9 books) | **won** | 3.30 | -2% | **+2.25** | +2.25 |
| 14:52 Hereford | Getarose | free | 3/1, 4.0 (9 books) | lost | 3.20 | +25% | -1.00 | -1.00 |
| 12:48 Hereford | God Of Fire | private | 11/2, 6.5 (4 books) | lost | 7.27 | -11% | -1.00 | -1.00 |

**+0.25 points** at the advised price and at the BSP; mean CLV +4%.

The short-priced shadow rule (win probability at least 0.25, the bookmakers' price at least 95% of the expected BSP;
recorded, never posted) had 3 winners from 5: Sharkeyboy (4/5, BSP 1.71), Kamille Au Kanada (11/8, BSP 1.48) and Karl
Des Tourelles (13/8, BSP 1.97) won; Our Wee Horse and Coolishael lost. **+1.80 points** at the advised price, +0.12
at the BSP, mean CLV +15%.

## The record so far (bookmaker prices, 1 point level)

| | Days | Tips | Winners | Points at advised | At the BSP | Mean CLV |
|---|---|---|---|---|---|---|
| Published rule | 3 (23, 24, 26 Oct; none on 25 Oct) | 11 | 2 | -2.75 | -3.51 | -6.3% |
| Short-priced shadow | 1 (26 Oct) | 5 | 3 | +1.80 | +0.12 | +14.7% |

Far too few to choose between the rules: the record keeps both, every day, until it can.
