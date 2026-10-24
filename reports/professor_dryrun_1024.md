# The Professor: the second dry run, 24 Oct 2026

The owner asked to try the dry run again (24 Oct). It ran earlier in the day than the first (11:45 UK against 14:30)
on the day's real ledger and Racing Post's bookmaker prices. Nothing was posted. The evidence is beside this report
in `reports/professor_dryrun_1024/`.

## What ran

- **The positions**: professor-publish.yml run 40087050172 read the live ledger: **130 horses** in 31 races
  (Cheltenham, Doncaster, Kelso, Southwell), 78 first backed in the 06:00 hour UK, 23 at 07:00, 15 at 08:00, 10 at
  09:00, 4 at 10:00. Every fact source joined all 130 this time (the predictions file, the catalogue's jockey,
  trainer and form, the card's race name and class): the fixes of the first dry run hold.
- **The scheduled 09:35 run did not fire**: GitHub runs a workflow's schedule only from the default branch, and this
  workflow is on the PR branch. It fires on its own once the PR is merged; until then a push to the package runs it.
- **The bookmaker prices**: read from this session (Racing Post refuses GitHub's runners) at 11:45-11:52 UK: all 31
  races, 2,790 quotes from 10 bookmakers, **120 of the 130 horses priced**. The other 10 are withdrawn: 9 carry
  Racing Post's own non-runner flag (from this commit the record says "a non-runner" instead of "no quotes").

## The card at 11:45 UK

| Rank | Tier | Race | Horse | Advised (bookmakers at it) | Trader's fill | Expected SP | Over the close | p_win | First backed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | free | 15:30 Cheltenham | North Shore | 4/1, 5.0 (4) | 4.70 | 4.51 | +11% | 0.22 | 08:14 |
| 2 | free | 14:05 Doncaster | Hi Corbett Court | 9/2, 5.5 (5) | 5.31 | 4.68 | +18% | 0.21 | 06:23 |
| 3 | private | 19:00 Southwell | African Spirit | 7/1, 8.0 (7) | 7.60 | 7.19 | +11% | 0.14 | 06:23 |
| - | withheld | 17:30 Southwell | Wyvill | 7/1, 8.0 (8) | 7.60 | 7.27 | +10% | 0.14 | 08:30 |

Wyvill qualified and fell in the random fifth withheld from publication (the design's measure of our own impact);
it is scored like a tip and posted nowhere. Three published tips where the cap allows five: only four horses of 130
still cleared the bar by late morning.

## The move, measured

The fates now carry each horse's bookmaker price (the price three or more major books offer) against the trader's
fill and against the closing model's expected BSP. For the 120 priced horses at 11:45 UK:

| Bookmakers' price against | Median | Middle half | Above it |
|---|---|---|---|
| The trader's fill | -24% | -37% to -13% | 6% of horses |
| The expected close | -21% | -33% to -8% | 11% of horses |

By the hour of the first back, the median gap to the expected close is -16% for the 06:00 hour and -21% to -26% for
07:00-10:00. Five or six hours after the fills, the bookmakers have taken the move on almost every horse: the price
the model forecasts is reached, and passed, before lunch.

This agrees with the first dry run (a median 26% under the close at 14:30) and with the backtest (the trader's own
price +27% implied, the 10:00 exchange price -6%). It is the case for the alert at the fill: a tip is worth what it
is worth in the minutes after the trader's back, at the price then. What the dry runs cannot yet show is whether
the bookmakers have priced the races at 06:15-08:00, when most fills happen; that needs the alert loop on the UK
server in the early morning, or an odds-only read at that hour.
