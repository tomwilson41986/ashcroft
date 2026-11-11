# racing² - the small book near the off

11 Nov 2026. Read-only research. Source for every number: research-query run 42986836901
(`research/queries/done/small_book_late_1111.py`), ledger tag `small-book-late-1111` (registered before the run).

## Result

**No edge. 0 of 54 variants passed.** Our 06:00 forecast adds nothing to the market 1-5 minutes before the off,
and every way of building the book loses about the spread plus commission.

## The brief

5, 3 or 2 minutes before the off, back a subset of the field with stakes that win the same amount on each horse
(a 50-90% book), leaving out horses judged bad prices, ordered by a Benter blend of the market and our forecast.

## Data

- `betfair_live_marks` 1 Oct - 10 Nov 2026 (recorder books, delayed key), marks `off_5`, `off_3`, `off_1`
  (no 2-minute mark is stored); BSP and winner from the `final` mark; our forecast from S3 `predictions/<day>.csv`.
- Cards: 1,603 WIN markets, 16,565 runners, forecast on 97.4%.
- Kept per mark: 1,130-1,137 whole races (a two-sided book on every runner, a forecast, a BSP, one winner, no
  runner removed after the mark, 3+ runners). The marks table also holds PLACE markets, which those filters drop.
- Scored walk-forward from day 8: 883-887 races, 14 Oct - 10 Nov.

## 1. Does our forecast help near the off? (Benter, fitted to the winner)

| Mark | Log-likelihood gain vs market alone | Weights (market / model) | Fitted to the BSP book |
|---|---|---|---|
| T-5 | -0.9 mnats a race (se 1.0) | 0.932 / 0.075 | 0.974 / 0.013 |
| T-3 | -0.9 (se 0.7) | 0.958 / 0.052 | 0.973 / 0.018 |
| T-1 | -1.0 (se 0.9) | 1.030 / -0.021 | 0.991 / 0.004 |

By the last minutes the market has absorbed everything our morning forecast knows.

## 2. The owner's book (return on stake, after 5% on each race's net)

| Book | T-5 | T-3 | T-1 |
|---|---|---|---|
| 50% | -0.8% | -3.2% | -1.8% |
| 60% | -0.6% | -2.0% | -3.4% |
| 70% | -2.7% | -4.7% | -4.5% |
| 80% | -4.0% | -3.1% | -4.4% |
| 90% | -1.1% | -1.8% | -2.2% |

CLV against the BSP: -1.6% to -2.0% for every book. The best back price at the mark is worse than the SP on average.
The market's back book is 1.017-1.020 and lay book 0.983-0.984, so about 1.7% is lost to the spread alone.

## 3. The improvements

| Idea | Best case | Verdict |
|---|---|---|
| Market order (favourites first, no model) | -0.2% (T-1, 60%) | Same as the owner's order: the model adds nothing |
| Lay the horses left out | -0.05% (T-5, 90%); CLV -0.65% to -1.8% | Cheapest execution, still negative |
| Value-set book (+0%) | +7.3% at T-5 (90% interval -16.3% to +35.2%, 211 races) | Noise: negative at T-3 and T-1, DSR 0.02 |
| Value-set book (+3%) | -11% to -28% | Loses |
| Joint Kelly | -5% to -9% | Loses |
| Same stakes at the SP | 0.0% to +1.4% best; mostly negative | No spread, no edge |

## What it means

- The edge the live trader takes is in the morning, when our forecast still knows something the market does not.
  By T-5 that information is in the price.
- A late book would need a late signal of its own (the ladder, money flow, non-runner news), not the 06:00 forecast.
  The stream recorder now running beside the trader (from 10 Nov) could supply one; that is a new study.

## Limits

- 28 scored days only. The recorder's books are on the delayed key and lag the live market.
- Fills at the best price with no size check.
- No 2-minute mark.

## Addendum: a late signal from the Exchange Stream (11 Nov)

Source: research-query run 42993797636 (`research/queries/done/stream_late_signal_1111.py`), ledger
`stream-late-signal-1111`.

**Not tested: no stream data found.** S3 `betfair_stream/horse/` holds nothing for 10 or 11 Nov. The recorder's
wiring reached the base branch at 08:00 UTC on 11 Nov, after that morning's live-trade run had started, so its first
morning is 12 Nov. It stops at 10:30 UK, before the first race, so it will hold no book near any off.

To test money flow and ladder pressure near the off, the stream has to run to the last race. That changes the live
job (or adds a second read-only process on the live key) and needs the owner's decision. The query is ready to
re-run unchanged once a few days are held.
