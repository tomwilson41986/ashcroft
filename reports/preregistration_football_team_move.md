# Pre-registration: the team-feature move forecast, traded on Betfair's own prices, takeable books only

Written 2026-11-02T15:41Z by the modelling agent, after the development season (2024/25) was scored on Betfair's prices and before
any holdout figure (football-data matches from 1 Jul 2025 to 1 Nov 2026) was computed. Development figures:
`reports/football_runs/team_move_dev.json`. Method: `research/open_close/football_team_move.py` and
`reports/football_team_features.md`. The holdout step (`python -m research.open_close.football_team_move --holdout`)
reads the JSON block at the end of this file and records this file's SHA-256 beside its result.

## What development showed

On 7,599 matches of 2024/25 (15,198 home and away selections), the forecast fitted on Pinnacle's moves of 2013/14 to
2023/24 correlates 0.175 with Betfair's own early-to-close move (fair prices), and its top tenth points the right way
65.7% of the time. Traded a tick worse at each end with 2% commission on each market's net:

- every book: positive from |forecast| 0.03 (+1.76%, 90% +0.65 to +2.90, 1,843 bets) to 0.05 (+7.72%, 340 bets);
- but wide books (Betfair early back prices summing over 1.05) carry it: +7.16% at 0.03, +19.36% at 0.05, while the
  back side taken at Betfair's recorded early back price loses 8.4% and 5.7% there. A stale quote in an empty book;
- takeable books (1.05 or under): -0.40% at 0.03 (90% -1.15 to +0.40, 1,316 bets), +0.06% at 0.04 (-1.18 to +1.30,
  549), +0.87% at 0.05 (-1.49 to +3.16, 214). No lower bound above zero;
- medium books alone (1.02 to 1.05): +1.44% at 0.03 (90% +0.37 to +2.70, 479 bets). This is one cell of 18 (three widths,
  six thresholds) and was found by looking.

## The rule (primary), chosen by the exchange study's chooser

Among thresholds with at least 200 development bets in takeable books, the best lower 90% bound a tick worse: 0.03.

- **Market:** Betfair MATCH_ODDS, home and away selections (never the draw), the 22 main divisions football-data
  prices on Betfair.
- **Forecast:** least squares of each side's chance move (log close / open, de-vigged) on the team features
  (`football.feature_eval.FAMILIES["all"]`, standardised on the training rows), fitted on Pinnacle-to-Pinnacle matches
  of 2013/14 to 2024/25 (Pinnacle's early prices are stale from 23 Jul 2025). The price forecast is minus the chance
  forecast.
- **Book width:** Betfair's early back book at 1.05 or under.
- **Signal:** price forecast shorter by more than 0.03: back at the early price, lay at the close; longer by more than
  0.03: lay, then back. Greened.
- **Headline:** the return a tick worse at each end on fair prices, 2% commission on each market's net, its 90%
  bootstrap interval by day (2,000 draws).

## Secondary read (reported, gated the same way, not the decision)

- Medium books only (1.02 to 1.05), threshold 0.03. Flagged in advance as chosen after looking.

## The expectation stated in advance

The primary fails: its development point estimate is below zero.

## Gate

At least 300 holdout bets, the 90% lower bound of the return a tick worse above zero, and the mean CLV at the taken
price above zero. A pass here is not a licence to trade. football-data's Betfair prices are an untimed snapshot. A pass
licenses only a paper trial on the recorder's books with sizes, as `reports/preregistration_football_exchange.md` sets
out for its rule.

```json
{
 "model": "ols_team_features_all_fit_on_pinnacle",
 "market_types": ["MATCH_ODDS"],
 "selections": ["H", "A"],
 "book_width": ["tight <1.02", "medium 1.02-1.05"],
 "threshold": 0.03,
 "commission": 0.02,
 "headline": "a tick worse at each end",
 "gate": {"min_bets": 300, "roi_lo90_above": 0, "clv_mean_above": 0},
 "holdout_from_season": 2025,
 "written_utc": "2026-11-02T15:41Z",
 "secondary": [
  {"book_width": ["medium 1.02-1.05"], "threshold": 0.03, "note": "chosen after looking"}
 ]
}
```
