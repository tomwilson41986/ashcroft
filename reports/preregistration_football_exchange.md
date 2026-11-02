# Pre-registration: football on Betfair, the early price to the close, in takeable books only

Written 2 Nov 2026 at 11:55 UTC by the modelling agent, after the development months (Jul 2024 to Jul 2026) were
scored and before any holdout figure (Aug 2026 to 1 Nov 2026) was computed. The development figures are in
`reports/open_close/football_runs/football_exchange_development.json`; the method and the reasons are in
`reports/open_close/model_football_exchange.md`. The holdout step (`python -m research.open_close.football_exchange
--holdout`) reads the JSON block at the end of this file, records the file's SHA-256 beside its result, and scores
only the rules named there (the other models are scored beside them for the record, not chosen from).

## What development showed, in one paragraph

On football-data.co.uk's Betfair early snapshot, the move to Betfair's close is strongly forecastable from the
bookmakers' early prices (R2 against no change 0.46-0.64 on match odds for the gap rules and the full models,
0.19-0.37 on over/under 2.5), and the hedged trade "pays" +30% to +52% a bet a tick worse at |forecast move| of
0.30. All of it sits in Betfair early books whose back prices sum to more than 1.05: in those books the back side
taken at Betfair's recorded early back price loses 8-23% a bet. In the books of 1.05 or under, the Pinnacle gap rule
earns -1.6% to +1.0% at the mid on match odds and loses 3.6-11.6% a tick worse at each end; on over/under 2.5 it
earns +0.3% to +1.4% at the mid and loses 1.0-1.8% a tick worse (the cells of 500 bets or more). That is a stale quote in an empty book, not a price anyone can take. The rule below is therefore confined
to books of 1.05 or under. Pinnacle's early prices are stale in football-data's files from 23 Jul 2025 (football-
data.co.uk data.php) and absent from February 2026 on, and today's fixtures file carries no Pinnacle column at all,
so no Pinnacle rule can be read on the holdout or traded on paper: the rule uses no Pinnacle feature.

## How the rule was chosen (on development rows only)

Among every market (match odds, over/under 2.5), every model and every threshold of the sweep (0, 0.02, 0.05,
0.10, 0.15, 0.20, 0.30), on the walk-forward development rows with a Betfair early book of 1.05 or under, with at
least 200 development bets: the best lower 90% bound of the return a tick worse at each end, after 2% commission on
each market's net. The models that read Pinnacle's price were left out, for the reason above. The ranking (90%
bootstrap by day, 2,000 draws):

| market | model | threshold | bets | days | at the mid | a tick worse each end (90%) | right way |
|---|---|---|---|---|---|---|---|
| over/under 2.5 | gbm_all (reads Pinnacle: excluded) | 0.05 | 220 | 127 | +4.47% | +1.28% (+0.37 to +2.22) | 73.6% |
| over/under 2.5 | **gbm_nopin** | **0.05** | 224 | 124 | +3.65% | **+0.47% (-0.60 to +1.56)** | 70.1% |
| match odds | gbm_nopin | 0.15 | 209 | 118 | +10.82% | +1.66% (-0.87 to +4.37) | 72.3% |
| over/under 2.5 | gbm_nopin | 0.02 | 2,454 | 297 | +1.68% | -0.80% (-1.07 to -0.51) | 61.3% |
| match odds | gbm_all (reads Pinnacle: excluded) | 0.10 | 449 | 169 | +8.63% | +0.06% (-1.52 to +1.77) | 71.7% |

No rule that can run without Pinnacle has a development lower bound above zero. The best is registered as the primary
rule so that the holdout and the paper trial read a fixed rule rather than a search; the match-odds candidate is
registered as a secondary read. **The expectation stated in advance: neither passes.** The development point
estimates are under 2% a bet a tick worse, on about one bet a week a market, with intervals that include zero;
split in halves by date, the match-odds cell is +3.5% (Jan-Oct 2025) and -1.8% (Nov 2025-Jul 2026).

## The rule (primary)

- **Market:** Betfair OVER_UNDER_25, both sides.
- **Leagues:** every league football-data publishes Betfair early prices for (the 22 main divisions: England 0-4
  and the National League, Scotland 0-3, Germany 1-2, Italy 1-2, Spain 1-2, France 1-2, the Netherlands, Belgium,
  Portugal, Turkey, Greece).
- **Book width:** the Betfair early back book (the sum of 1/back price over the market's selections) at 1.05 or
  under.
- **Model:** `gbm_nopin` (`research/open_close/football_exchange.py`): LightGBM on Betfair's own early book (log
  price, share, rank, gap to the favourite, the book's overround), the same moment's market average, maximum and
  Bet365 books and their log gaps to Betfair's fair price, the Dixon-Coles probability fitted on the matches before
  the match's week (`football.model.walk_forward`) and its gap, league, top-five flag, month, day of the week,
  kick-off hour, days from the collection to the kick-off, and the season's stage. Fixed settings (400 trees, rate
  0.03, 31 leaves, 100 rows a leaf), refitted every month on every month before it.
- **Signal:** the forecast y_hat of log(close / early) on fair prices (each book scaled to one). Forecast shorter by
  more than 0.05: back at the early price, lay at the close. Forecast longer by more than 0.05: lay at the early
  price, back at the close. Greened, so the result does not matter.
- **Stake:** GBP1 a selection in the backtest (the ROI is scale-free); GBP10 notional a selection on paper. No
  money: the owner's rule (2 Nov 2026) keeps the football trader on paper, off the live runner.

## Secondary read (reported, gated the same way, not the decision)

- Betfair MATCH_ODDS, the same leagues, book width 1.05 or under, `gbm_nopin`, |forecast move| over 0.15.

## The holdout read (once)

- Months 2026-08 to 2026-11 (to 1 Nov 2026) of the same panel, each month scored walk-forward by models fitted on
  every month before it (the development months and the earlier holdout months), as a live trader would refit.
- Headline: the return a tick worse at each end on Betfair's ladder, after 2% commission on each market's net, its
  90% bootstrap interval by day; beside it the return at the fair mid, the mean CLV at the taken price, the back side
  as quoted, bets a day, the worst drawdown, the right-way share, by book width, price band, league tier and month.
- The holdout is football-data's snapshot, not a recorded book: whatever it shows, a pass here is not a licence to
  trade, only to run the paper trial below.

## The paper trial (the real test)

Prices from the coordinator's Betfair football book recorder (best back and lay with sizes), never from
football-data:

- **The early moment:** the recorder's snapshot at football-data's collection time (Friday by 17:00 UK for Friday to
  Monday matches, Tuesday by 13:00 UK for Tuesday to Thursday; matches.php), the features from football-data's
  fixtures file collected that day (`football/raw/fixtures/<day>/fixtures.csv`: Avg, Max, B365 and BFE columns, all
  present on 2 Nov 2026) and the Dixon-Coles fit of that week. The book-width filter is applied to the recorder's
  own best-back book at that moment.
- **The fill:** a back at the recorded best back, a lay at the recorded best lay, only for the size shown at that
  price (GBP10 or less, else the trade is skipped and logged).
- **The close:** the last recorded two-sided book before the market turns in play: a back closed by a lay at the
  best lay, a lay closed by a back at the best back.
- **Window:** from the first Friday collection after the recorder is live, until at least 300 rule trades **and** at
  least 90 days, whichever is later, capped at 12 months (at the development rate of about 12 rule trades a month a
  market, 300 trades is about two seasons; the cap ends it sooner, and the report will say the trial was
  underpowered if so).
- **Passes only if all of:** (1) the 90% lower bound of the return at the taken prices, after 2% commission on each
  market's net, is above zero (bootstrap by day, 2,000 draws); (2) the mean CLV at the taken price (a back o/c - 1,
  a lay 1 - o/c, c the recorded close mid) is above zero; (3) both halves of the window, split by date, have a
  positive return.
- **Reported whatever the outcome:** the headline and its interval, the halves, by book width at the moment, by
  price band, by league tier, the skipped trades for size, and the share of signals whose book would have been wide
  (over 1.05) on the recorder's own book.

```json
{
 "model": "gbm_nopin",
 "market_types": ["OVER_UNDER_25"],
 "leagues": "all",
 "book_width": ["tight <1.02", "medium 1.02-1.05"],
 "threshold": 0.05,
 "stake_gbp": 1,
 "commission": 0.02,
 "headline": "a tick worse at each end",
 "gate": {"min_bets": 300, "min_days": 90, "roi_lo90_above": 0, "clv_mean_above": 0},
 "holdout_from": "2026-08",
 "written_utc": "2026-11-02T11:55Z",
 "secondary": [
  {"model": "gbm_nopin", "market_types": ["MATCH_ODDS"], "leagues": "all",
   "book_width": ["tight <1.02", "medium 1.02-1.05"], "threshold": 0.15}
 ],
 "thresholds_by_model": {
  "MATCH_ODDS": {"gap_rule_ps": 0.30, "gap_rule_avg": 0.30, "gap_ols_ps": 0.30, "gap_ols_avg": 0.30,
                 "gbm_base": 0.30, "gbm_nopin": 0.30, "gbm_all": 0.30},
  "OVER_UNDER_25": {"gap_rule_ps": 0.30, "gap_rule_avg": 0.30, "gap_ols_ps": 0.30, "gap_ols_avg": 0.30,
                    "gbm_base": 0.30, "gbm_nopin": 0.30, "gbm_all": 0.30}
 }
}
```

The `thresholds_by_model` are the thresholds each model chose on all development rows (wide books included) by
the repository's standard chooser (`research.open_close.run.choose_threshold`); they place the other models' holdout
figures for the record and choose nothing.
