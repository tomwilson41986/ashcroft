# The Professor: the dry run of 23 Oct 2026

The owner asked (23 Oct): "you can use BF price. But really we want to find horses that are with bookmaker prices.
dry run this." Phase 1 of the design was built to that point (the package `professor/`, PR #167) and run on the day's
real ledger. Nothing was posted. The evidence is beside this report in `reports/professor_dryrun_1023/`: the tips,
every candidate's fate, the odds snapshot (3,528 bookmaker quotes) and two cards.

## What ran

1. **The ledger to candidates** (professor-publish.yml on GitHub's runner, runs 39956773681 and 39957319139, the
   artifact `professor-card-*`): the live trader's ledger copy in S3 read into one row a horse: **164 positions**
   on the day, 68 first backed in the 06:00 hour UK, 24 in the 07:00 hour, 72 from 10:00 (the trader keeps backing
   to 15 minutes before each off). The catalogue matched every one (runner, venue, start time; the metadata came in
   the later files and is taken from the first file that holds it from this commit); the predictions file and the
   card matched none on the first runs because the ledger's `off` is a UK clock time and was read as UTC (fixed:
   both now join by course, time and name as well as by the ids).
2. **The bookmaker prices**: Racing Post's odds comparison, one `/betting-odds/` page a race, read through
   Chromium. **GitHub's runners get a 403 page** from Racing Post (the raw page is in the artifact); this session
   reads it, and the UK server will. So the day's card was finished here at 14:30 UK: 33 of our 37 races still to
   run read, 3,528 quotes from 10 bookmakers (William Hill, Ladbrokes, bet365, Coral, Paddy Power, Betfair
   Sportsbook, Sky Bet, BoyleSports, Betfred, the Tote), a median of 10 quotes a horse, **144 of the 164 horses
   matched** by course, time and name (the 20 unmatched were in races already run).
3. **The rule** (`professor/config.json`): the price three or more major bookmakers offer, 10% over the closing
   model's expected BSP, one a race, ranked by the expected win probability, 5 a day, the top two free; the
   exchange price only where no bookmaker quotes the race and at least GBP50 sits at it.

## The card at 14:30 UK

| Rank | Tier | Race | Horse | Advised (bookmakers at it) | Exchange at the fill | Expected SP | Over the close | p_win | First backed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | free | 15:00 Cheltenham | Planters Punch | 9/2, 5.50 (8) | 4.70 | 4.08 | +35% | 0.25 | 06:24 |
| 2 | free | 18:30 Wolverhampton | Way To Dubai | 14/1, 15.0 (4) | 13.0 | 12.6 | +19% | 0.08 | 06:51 |
| 3 | private | 14:25 Cheltenham | Loriko | 16/1, 17.0 (4) | 15.0 | 14.2 | +19% | 0.07 | 06:24 |
| 4 | private | 14:00 Sligo | Beat The Devil | 28/1, 29.0 (6) | 27.0 | 25.9 | +12% | 0.04 | 12:29 |
| 5 | private | 14:10 Doncaster | Dark Thirty | 50/1, 51.0 (7) | 46.0 | 40.6 | +26% | 0.02 | 06:24 |

Every candidate's fate at that hour:

| Fate | Horses |
|---|---|
| qualifies (5 taken, 2 behind the one-a-race rule and the cap) | 7 |
| under the overlay bar | 135 |
| no bookmaker quotes (the race run) and too little money at the exchange price | 18 |
| over the maximum price | 4 |

**The 135 under the bar are the lesson.** By 14:30 UK the bookmakers' prices on our horses sat a median **26% under
the expected closing price**; 96% were below it and 84 had shortened by more than 20% past it (Beeverstown Boy,
Sligo 15:10: backed at 16.0 at 06:24, 5.0 with every bookmaker at 14:30). The move the model forecasts happens in
the morning, and a card read in the afternoon finds it gone. The five that still qualify are the drifters and the
long shots. This is the design's point made with the day's own prices: the tip is the alert at the fill, at the
price then; the 10:00 card is the summary.

Two things the write-ups show that the next run fixes: the horse line is bare ("Planters Punch.") because the
jockey and trainer came in the catalogue's later files (now taken from the first file that holds them), and the
exchange price a tip quotes is the one at the trader's fill, hours before the bookmaker price beside it (the alert
loop reads both at the same minute).

## What the dry run settles

- **The pipeline works on the real data**: the ledger, the catalogue, the bookmaker prices, the rule, the write-ups,
  the cards, the post log, all in a dry run with nothing sent.
- **Racing Post is a usable bookmaker-price source**, ten books a horse with the Best Odds Guaranteed and each-way
  rows, read in about 7 seconds a race. It refuses GitHub's runners, so the scrape runs on the UK server (the second
  runner registration, `PROFESSOR_RUNNER`), where Oddschecker's challenge can also be tried.
- **Market Form** (dc-network.co.uk, the owner's third source) is a GBP49.99-a-month membership with its live
  market behind a login and a separate data API; the slot waits for the owner's key.
- **The prices move early**, as the backtest said: at the fill, +27% implied on the live record; at 10:00, -6%; at
  14:30, a median 26% under the close. The alert loop (`python -m professor --alerts`) is written and is what the
  UK runner should run from 06:15.

## To run it

```bash
python -m professor --card --day 2026-10-23 --dry-run --only-ours --sources racingpost   # the day at a point in time
python -m professor --card --dry-run --candidates-file out/professor/2026-10-23/candidates.csv --only-ours
python -m professor --alerts --until 10:00                                                # the live loop (UK runner)
python -m professor --odds --day 2026-10-23                                               # the scrape alone
```
