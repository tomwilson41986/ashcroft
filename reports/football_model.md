# The football model for Betfair (7 Oct 2026)

**Bottom line: the pipeline, the model and the trader are built and tested, but the model has no edge against the
market, so the trader runs on paper only.** A market-blind Dixon-Coles team model is measurably worse than the
opening price at every level: by season, division and market. Pooled with the open it adds nothing, and its 1X2 weight
comes out *negative*. Backing on the model alone loses 4-7% on turnover with CLV of about -2.5% over tens of thousands
of out-of-sample bets. This matches what `reports/all_markets_priorities.md` expected of football (the deepest market
and the sharpest close). Do not set `FOOTBALL_LIVE=yes` on this model.

Data: football-data.co.uk, 2005/06 to 6 Oct 2026, 38 leagues, 226,836 matches, fetched from the source itself (this
session's S3 keys were refused; the same code runs on S3 in `football-model.yml`). Predictions from 1 Jul 2012, each
match priced only from the matches before it.

## 1. What was built

| Layer | Where | What |
|---|---|---|
| Bronze | `sources/football/raw/`, `sources/openfootball/raw/`, `sources/api_football/raw/`, `sources/betfair_historic/raw/soccer/` | As published: football-data.co.uk seasons and (new) the weekly fixtures files by day; openfootball (CC0); API-Football by day, append-only (once `API_FOOTBALL_KEY` is set); Betfair BASIC soccer files (once the account holds the plan) |
| Silver | `football/matches.parquet`, `openfootball/matches.parquet`, `api_football/fixtures.parquet` | One row a match per source |
| Gold | `football/gold/` (`football/data.py`) | `matches` (stable `match_id`, kick-off in UTC, status, scores, match stats, benchmark open/close for 1X2, O/U 2.5 and AH, `record_version`), `matches_history`, `odds_long` (9.56m rows), `betfair_market_map`, `team_aliases`, `betfair_review` |
| DQ | `football/dq/latest.json` | Duplicates, impossible scores, season match counts, overround, Pinnacle after 23/07/2025, freshness, scores against openfootball, API-Football and Betfair settlement |
| Model | `football/ratings.py`, `football/model.py` | Pooled Dixon-Coles, weekly walk-forward, stage-2 pooling with the open, value rules scored by CLV |
| Trader | `football/live.py`, `football/live_config.json` | Betfair MATCH_ODDS / OVER_UNDER_25; paper unless `FOOTBALL_LIVE=yes`; its own pre-play close for CLV; settles from gold |
| Jobs | `football-model.yml`, `football-trade.yml`, `live-trade.yml` | Daily gold, fit and settle on GitHub's runners; the paper trader beside the horse session on the UK runner |

Departures from the design brief, deliberately:

- **No Lambda, Iceberg or Athena.** The repository already runs on GitHub Actions and Parquet in S3 (`sources.common.Store`).
  The gold build is idempotent instead. It rebuilds from bronze and raises `record_version` only where a score,
  status or kick-off changed, keeping the old row in `matches_history`. That gives what `MERGE INTO` would.
- **`match_id`** is a hash of competition, season, the two canonical team ids and the meeting number. The meeting number
  covers split leagues, where a side can host the same opponent twice in a season. A rescheduled match keeps its id
  (tested).
- **Understat is not wired in.** It is scrape-only and fragile, as the brief warns. The shots-on-target mix below
  covers most of what xG would add to a goals model.

### Data checks on the real data

- 4 played rows had no score (abandoned or void) and are now `NO_RESULT`. That status change went through versioning
  (`record_version` 2, old rows in history), which exercised the upsert.
- Season match counts flag real gaps in the source files: ESP2 and FRA2 2007/08, POR1 2007/08, TUR1 2006/07, FRA2
  2023/24 and 2025/26 (one match each). The COVID-curtailed 2019/20 seasons are labelled as such.
- **openfootball:** 71,363 matches linked (97% of those with a score). 188 disagree, 179 of them in League One/Two in
  2025/26, almost all Nov-Dec 2025, which points at openfootball's own files for those months. Which source is right
  has not been checked match by match: the flags sit in `dq/` for review.
- **Benchmarks:** the opening price is Pinnacle (91,766 matches) or Betfair Exchange (15,414). The close is Pinnacle to
  2024/25 (153,336) and Betfair Exchange after (15,069), as the brief recommends.

## 2. The model

Each country is fitted as one pool (England's five divisions together), so a promoted side carries its rating. Each
team's attack and defence are ridge-shrunk towards its division's mean, and the division offsets are learned from the
sides that move. Matches are weighted by exp(-ξ·days), and ρ is the Dixon-Coles low-score correction. The fitted
target is (1-a)·goals + a·c·shots-on-target, where c is the pool's goals per shot on target. Refitted every 7 days on
the 1,100 days before. A full run over every league from 2012/13 takes 2½ minutes on 4 cores.

Tuning used 2015/16 to 2018/19 only, on the five big countries' 22,456 matches, measured by the model's own 1X2
log-loss:

| ξ (per day) | ridge | SoT mix | 1X2 log-loss | O/U log-loss |
|---|---|---|---|---|
| 0.0012 | 1 | 0 | 1.0305 | 0.6881 |
| 0.0023 | 3 | 0 | 1.0285 | 0.6877 |
| 0.0035 | 3 | 0 | 1.0284 | 0.6890 |
| 0.0023 | 10 | 0 | 1.0287 | 0.6860 |
| **0.0023** | **3** | **0.3** | **1.0277** | 0.6861 |
| 0.0023 | 3 | 0.6 | 1.0284 | 0.6855 |

Chosen: ξ 0.0023 (half-life of about 300 days), ridge 3, shots-on-target mix 0.3. 2019/20 onwards was not looked at
during tuning.

## 3. Against the market (out of sample, 2012/13 to 6 Oct 2026)

Log-loss on the same matches, those with every forecast present. Lower is better.

| Market | Matches | Model | Open | Model pooled with open | Close |
|---|---|---|---|---|---|
| 1X2 | 92,766 | 1.0166 | 1.0017 | 1.0014 | 0.9983 |
| O/U 2.5 | 54,653 | 0.6838 | 0.6769 | 0.6769 | 0.6745 |

The model is behind the open in every season from 2014 to 2026; there is no season where it wins. The pooled forecast
gains 0.0003 on 1X2, which is nothing, and the close is better than every forecast. The information the close adds
arrives after the open, mostly team news and money, and the model cannot see it.

Pooling weights, each season fitted on the seasons before it:

| Season | 1X2 weight on the model | 1X2 weight on the open | O/U weight on the model | O/U weight on the open |
|---|---|---|---|---|
| 2014 | -0.071 | 1.087 | 0.108 | 0.933 |
| 2018 | -0.076 | 1.113 | 0.062 | 1.025 |
| 2022 | -0.070 | 1.110 | 0.039 | 1.043 |
| 2026 | -0.076 | 1.126 | 0.030 | 1.054 |

A negative weight means that where the model disagrees with the open, the result tends to go further the market's
way. The model's disagreements carry no information beyond the price. Its one positive weight, on over/under, fades
towards zero as the data grows.

### Value rules

Back an outcome where the expected return at the opening price clears the edge. "Open" is the benchmark book with no
commission; "bfe" is Betfair Exchange's own opening price after 2% commission. CLV is the expected return at the price
taken, were the close fair.

| Probability | Venue | Edge | Market | Bets | ROI | ± | CLV | ± | CLV > 0 |
|---|---|---|---|---|---|---|---|---|---|
| model | open | 5% | 1X2 | 89,186 | -6.8% | 0.6 | -2.49% | 0.03 | 36% |
| model | open | 5% | O/U 2.5 | 38,855 | -4.9% | 0.5 | -2.75% | 0.04 | 30% |
| model | open | 5% | Asian handicap | 53,125 | -4.0% | 0.4 | -2.30% | 0.04 | 32% |
| model | bfe | 5% | 1X2 | 12,281 | -9.2% | 1.6 | -2.10% | 0.10 | 40% |
| model | bfe | 5% | O/U 2.5 | 5,702 | -4.6% | 1.4 | -2.03% | 0.09 | 35% |
| model | open | 10% | 1X2 | 63,078 | -7.4% | 0.7 | -2.39% | 0.04 | 37% |
| pooled | open | 2% | 1X2 | 644 | +2.7% | 3.5 | -1.98% | 0.25 | 38% |
| pooled | bfe | 2% | 1X2 | 91 | +19.5% | 10.7 | -0.86% | 0.77 | 42% |
| pooled | either | 5% | any | 6 | | | | | |

The model-only rules lose decisively. The CLV standard errors are a few hundredths of a percent, and the close moves
against the model's side about two times in three. The pooled rules almost never bet: pooling hands the price back to
the market. Their few bets have negative CLV, so the positive ROI on 91 or 644 bets is noise, not a signal to act on.

By tier, model-only at a 5% edge, 2019/20 onwards:

| Tier | Bets | ROI | CLV |
|---|---|---|---|
| 1 | 45,774 | -7.3% | -2.57% |
| 2 | 26,960 | -5.1% | -2.82% |
| 3 | 8,309 | -6.7% | -2.91% |
| 4 | 7,878 | -2.8% | -3.15% |
| 5 (National League) | 6,473 | -2.5% | -4.17% |

The lower leagues do not rescue it. CLV gets worse down the pyramid: the closes there move more, on news the model
lacks.

Per league, the pooling refitted on each league's matches before 2019 and tested from 2019 moves log-loss by between
-2.2 and +2.9 per thousand matches, scattered both ways with no pattern (ITA2 +1.5 on 1X2 but -2.2 on O/U; TUR1 +2.9
on 1X2). No league is a candidate.

## 4. The trader and its switch

`football/live.py --trade` runs beside the horse session on the UK runner (`live-trade.yml`, to 21:30 UK), or on its
own by hand (`football-trade.yml`):

- It reads Betfair's MATCH_ODDS and OVER_UNDER_25 markets of the next 14 hours in GB, DE, IT, ES, FR, NL, BE, PT, TR
  and GR.
- It prices only league matches between two sides the ratings know (10 or more matches each, same division). Cups,
  women's, youth and reserve football are excluded.
- Each selection is decided once, at its first real book (back and lay within 10%) in the 6 hours before kick-off. It
  backs where the pooled chance's expected return after commission clears 5%: GBP2 level, fill-or-kill, GBP4 a match,
  GBP100 a day.
- It keeps reading each book until the market turns in play and records that close, so its CLV is measured against
  the Exchange's own close.
- On paper it uses the delayed key: Betfair permits no read-only use of live data.
- S3 `football/STOP` or `trading/STOP` stops new bets within a minute.

On this evidence it will make almost no paper bets: at the configured 5% edge the pooled rule fired 6 times in 14
seasons of backtest. Its use now is
the record, not the bets. Its daily file of Betfair pre-play closes for every in-scope match is the benchmark football
lacked. `--settle` (daily in `football-model.yml`) settles any bets against gold and keeps
`sources/football/live/<mode>/summary.json`.

**Gate before any real money:** a rule whose out-of-sample CLV against the Betfair close has a 90% lower bound above
zero over at least 1,000 bets, first in the walk-forward and then on paper. Nothing here meets it.

## 5. What might find an edge (next)

The close beats the open by 0.0034 in 1X2 log-loss: that is the information that arrives between the two. A football
edge, if there is one, is in seeing that information first, not in a better team rating. In order:

1. **Line-ups and team news:** API-Football's line-ups and injuries (Pro, US$19 a month), about an hour before
   kick-off. Price the confirmed eleven against the market in the minutes after the team sheet. This needs the key, and
   Betfair ADVANCED data to measure the price move around team news.
2. **The open-to-close move itself:** a model of the move from the open (drift by league, kick-off time, day of week,
   the weekend's results) on Betfair BASIC files from 2015, once the account holds Soccer.
   `betfair_historic.py` already fetches it, and `football/data.py` already maps it to matches.
3. **Second-tier markets:** corners and cards (football-data has the match stats from 2000/01), with books thin
   enough that a stats model might beat them. The census of every sport (`betfair_live/<day>/census_sports.json`) says
   whether they have the liquidity.
4. **xG (Understat, big five only):** worth one test as a ratings input. The shots-on-target mix bought 0.0008 of
   log-loss, and the gap to the open is 0.015.

## 6. The owner's to supply

1. **Betfair Historic Data:** the free BASIC plan for Soccer, at historicdata.betfair.com with the trading account (still
   outstanding from `all_markets_priorities.md`). It fills `betfair_market_map` and the Betfair closes for 2015
   onwards.
2. **`API_FOOTBALL_KEY`** (Pro, US$19 a month) only if item 1 of section 5 goes ahead. Until then
   `sources/api_football.py` fetches nothing and says so.
3. **football-data.co.uk:** its terms say the data is "intended for private individuals only". Written permission is
   needed before any model trained on it stakes the business's money.
