# racing² football data warehouse - data dictionary

Generated from the table contracts in `football/warehouse/` by `python -m football.warehouse dictionary`. Do not edit by hand.

Times are UTC. A price is Betfair's decimal price. `handicap` is 0.0 where a runner has none. Tables live under `sources/football/warehouse/<table>/` in the capture bucket.

| Table | Rows are | Key | Stored as |
|---|---|---|---|
| `bronze/raw` | One row a raw Betfair Historic Data file (a market's stream file, bz2 JSON lines), with its bytes | plan_path | one file a (year, month) partition |
| `bronze/index` | bronze/raw without the bytes: one row a raw file | plan_path | one file a (year, month) partition |
| `silver/market` | One row a market: what it is, when it ran, how it ended, and how much the file holds on it | market_id | one file a (year, month) partition |
| `silver/runner` | One row a runner of a market: a (selection, handicap) pair, its role, its result and its first, closing and last traded prices | market_id, selection_id, handicap | one file a (year, month) partition |
| `silver/definition` | One row each time a market's definition was published: its state at that moment | market_id, seq | one file a (year, month) partition |
| `silver/spell` | One row a spell of a market's life: a run of messages over which (in play, status) did not change | market_id, spell_no | one file a (year, month) partition |
| `silver/tick` | One row each time a runner's last traded price was published (BASIC plan: about once a minute a runner while it trades), with the market's state at that moment | market_id, selection_id, handicap, pt, seq | one file a (year, month) partition |
| `gold/event` | One row a Betfair event (a football match): sides, kick-off, markets, result, trading, the closing 1X2 | event_id | one file a (year, month) partition |
| `gold/checkpoint` | One row a runner a checkpoint: the price as of a fixed number of minutes before the scheduled kick-off (0: the close), with its age, the activity behind it and the state of its book | market_id, selection_id, handicap, cp_min | one file a (year, month) partition |
| `gold/implied` | One row an event a checkpoint: the market's chances (1X2, over 2.5, the handicap's main line), the goal expectancies they imply, the draw premium (rho) and the disagreement between the handicap and the 1X2 | event_id, cp_min | one file a (year, month) partition |
| `gold/path` | One row a match-odds or over/under runner a decision checkpoint (1440, 360, 120, 60, 15 minutes before the scheduled kick-off): what its last traded price did up to that moment, in the logit of the raw chance (x = ln(p / (1 - p)), p = 1 / price; positive is shortening) | market_id, selection_id, handicap, cp_min | one file a (year, month) partition |
| `gold/goal` | One row an inferred goal: when, which side, the score after it, how sure, and the price moves that say so | event_id, goal_no | one file a (year, month) partition |
| `gold/timeline` | One row a match that went in play: its inferred goals, the time each side led, its stoppages, half time, whether the in-play record can be used, whether the price reading agreed with how the markets settled, and how far the final list can be trusted | event_id | one file a (year, month) partition |
| `gold/inplay_minute` | One row a match a minute in play: the market's chances of home win, draw and away win as of that minute, their quality, the over 2.5 chance while open, the phase and football minute, and the inferred score | event_id, minute | one file a (year, month) partition |
| `gold/inplay` | One row a match that went in play: the time-averaged in-play chances and the points they are worth against the pre-match expectation, the half-time chances, the extremes, and when the result was decided | event_id | one file a (year, month) partition |
| `gold/move` | One row a runner a checkpoint: the move of its normalised chance from the checkpoint to the close, with the quality of both ends | market_id, selection_id, handicap, cp_min | one file a (year, month) partition |
| `gold/team` | One row a Betfair team id (the selection id of a side on the match-odds market) | team_id | a single file |
| `gold/team_rating` | One row a team a match: the team's market-implied rating before the match (from closes fixed before the match's UTC day only) and after it | team_id, event_id | a single file |
| `gold/rating_gap` | One row an event a checkpoint (1440, 360, 120, 60, 15 minutes before the scheduled kick-off, and 0: the close): what the two sides' ratings as of the checkpoint imply for the match, what the market says there, and the gap (ratings less market) | event_id, cp_min | a single file |
| `club/team_match` | One row a team a match: the market's expectation, the result against it, the in-play record and the ratings | team_id, event_id | a single file |
| `club/team_summary` | One row a team: its matches in the warehouse added up, its rating now, and its standing in its country | team_id | a single file |
| `club/country_strength` | One row a country code and category: the strength of its sides on the ratings' one scale, and what its matches look like to the market | country_code, category | a single file |
| `gold/link` | One row a Betfair event: the API-Football fixture it is, how the link was made, and our canonical match | event_id | a single file |
| `gold/team_link` | One row a Betfair team id: the API-Football team it most often links to, by a vote over its linked matches, and our canonical team id where its matches are canonical ones | team_id | a single file |
| `gold/goal_check` | One row a linked match that went in play and has API-Football's score or goal events: the goals inferred from prices (gold/goal) against them | event_id | a single file |

## `bronze/raw`

One row a raw Betfair Historic Data file (a market's stream file, bz2 JSON lines), with its bytes. Immutable: a file enters once; a second copy of the same path is kept only if it is longer (a market read before it settled).

Key: `plan_path`. Stored as one file a (year, month) partition, sorted by `plan_path`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `plan_path` | string | yes | The file's path from Betfair's plan folder on, e.g. BASIC/2026/Jan/29/35155405/1.252787713.bz2: the same name the Historic Data API gives it. |
| `event_id` | int64 | yes | Betfair's event id, from the path. |
| `market_id` | string | yes | Betfair's market id, from the path. |
| `day` | int16 | yes | Day of the month in the path. |
| `size` | int32 | yes | Bytes of the compressed file. |
| `sha256` | string | yes | SHA-256 of the compressed bytes. |
| `source` | string | yes | Where the file came from: the bundle's name or 'api'. |
| `data` | binary | yes | The file itself (bz2). |

## `bronze/index`

bronze/raw without the bytes: one row a raw file. What builds and checks read.

Key: `plan_path`. Stored as one file a (year, month) partition, sorted by `plan_path`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `plan_path` | string | yes | The file's path from Betfair's plan folder on, e.g. BASIC/2026/Jan/29/35155405/1.252787713.bz2: the same name the Historic Data API gives it. |
| `event_id` | int64 | yes | Betfair's event id, from the path. |
| `market_id` | string | yes | Betfair's market id, from the path. |
| `day` | int16 | yes | Day of the month in the path. |
| `size` | int32 | yes | Bytes of the compressed file. |
| `sha256` | string | yes | SHA-256 of the compressed bytes. |
| `source` | string | yes | Where the file came from: the bundle's name or 'api'. |

## `silver/market`

One row a market: what it is, when it ran, how it ended, and how much the file holds on it.

Key: `market_id`. Stored as one file a (year, month) partition, sorted by `market_id`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `market_id` | string | yes | Betfair's market id. |
| `event_id` | int64 | yes | Betfair's event id: one football match. A match's markets share it. |
| `event_type_id` | int32 |  | Betfair's sport id (1: soccer). |
| `market_type` | string | yes | MATCH_ODDS, OVER_UNDER_25, ASIAN_HANDICAP, ... |
| `market_name` | string |  | The market's display name. |
| `event_name` | string |  | The match as Betfair names it: 'Home v Away'. |
| `home_name` | string |  | The side named first in event_name (blank when the name has no ' v '). |
| `away_name` | string |  | The side named second in event_name. |
| `country_code` | string |  | Betfair's country code for the event; blank on many cups and internationals. |
| `timezone` | string |  | The market's display time zone. |
| `betting_type` | string |  | ODDS, ASIAN_HANDICAP_DOUBLE_LINE, ... |
| `number_of_winners` | int16 |  | Winners the market pays. |
| `market_base_rate` | float |  | Betfair's base commission rate on the market, per cent. |
| `turn_in_play_enabled` | bool |  | Whether the market was set to go in play. |
| `persistence_enabled` | bool |  | Whether bets could persist in play. |
| `bsp_market` | bool |  | Whether Betfair SP was offered (not on football). |
| `regulators` | string |  | Regulator codes, comma-joined. |
| `market_time` | timestamp[ms, tz=UTC] |  | The kick-off time in the last definition. Betfair often rewrites it to the actual kick-off as the market turns in play: use market_time_pre for the schedule. |
| `market_time_first` | timestamp[ms, tz=UTC] |  | The scheduled kick-off in the first definition. |
| `market_time_pre` | timestamp[ms, tz=UTC] |  | The scheduled kick-off as it stood before the off: the kick-off time in force in the last definition before the market turned in play (before it closed, where it never did). The schedule a trader knew; every 'minutes before the off' is measured from it. |
| `schedule_set_pt` | timestamp[ms, tz=UTC] |  | When market_time_pre was last set: the message from which the schedule stood unchanged to the off. A moment before it did not know this schedule. |
| `n_market_time_changes` | int16 |  | Times the kick-off time changed in the file (a reschedule, or Betfair's rewrite at the kick-off). |
| `open_date` | timestamp[ms, tz=UTC] |  | The event's open date in the last definition. |
| `suspend_time` | timestamp[ms, tz=UTC] |  | The scheduled suspend time in the last definition. |
| `settled_time` | timestamp[ms, tz=UTC] |  | When Betfair settled the market (the last definition's settledTime). |
| `first_pt` | timestamp[ms, tz=UTC] | yes | The file's first message. |
| `last_pt` | timestamp[ms, tz=UTC] | yes | The file's last message. |
| `in_play_pt` | timestamp[ms, tz=UTC] |  | The market's turn in play: the actual kick-off, to the minute. Where the file has more than one turn (a turn made by mistake and undone, a drop out of play in mid-match) it is the one nearest the kick-off time in force when it was made. |
| `n_inplay_turns` | int16 | yes | Times the market turned in play in the file: more than one marks a file to read with care. |
| `off_pt` | timestamp[ms, tz=UTC] |  | The off: in_play_pt, or market_time_pre for a market that never turned in play (some trade on through the match). A price before it is a pre-off price. |
| `closed_pt` | timestamp[ms, tz=UTC] |  | The first message with the market CLOSED. |
| `first_trade_pt` | timestamp[ms, tz=UTC] |  | The first traded price in the file. |
| `last_pre_trade_pt` | timestamp[ms, tz=UTC] |  | The last traded price before the off: the close's time. |
| `last_trade_pt` | timestamp[ms, tz=UTC] |  | The last traded price in the file. |
| `status_final` | string |  | The last definition's status (CLOSED once settled). |
| `in_play_final` | bool |  | Whether the last definition had the market in play. |
| `n_messages` | int32 | yes | Messages in the file. |
| `n_definitions` | int32 | yes | Messages that carried a market definition. |
| `n_ticks` | int32 | yes | Traded-price updates in the file (rows of silver/tick). |
| `n_ticks_pre` | int32 | yes | Of those, before the off. |
| `n_ticks_inplay` | int32 | yes | Of those, in play (the rest, if any, came after the scheduled kick-off of a market that never turned in play). |
| `n_ltp_invalid` | int32 | yes | Price updates outside Betfair's range (1.01 to 1000), left out of silver/tick: a void market is closed with 1.0 on every runner. |
| `n_runners` | int16 | yes | Runners in the last definition (an Asian handicap line counts once a side). |
| `n_winners` | int16 | yes | Runners settled WINNER. |
| `n_removed` | int16 | yes | Runners settled REMOVED. |
| `n_susp_inplay` | int16 | yes | Suspended spells while in play (a goal, a penalty, a review, half time). |
| `kickoff_delay_s` | int32 |  | in_play_pt less market_time_pre, seconds (blank if never in play). |
| `plan_path` | string | yes | The bronze file the row was read from. |
| `file_sha256` | string | yes | That file's SHA-256. |
| `parser_version` | int16 | yes | football.warehouse.parse.PARSER_VERSION that read it. |

## `silver/runner`

One row a runner of a market: a (selection, handicap) pair, its role, its result and its first, closing and last traded prices.

Key: `market_id`, `selection_id`, `handicap`. Stored as one file a (year, month) partition, sorted by `market_id`, `selection_id`, `handicap`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `market_id` | string | yes | Betfair's market id. |
| `event_id` | int64 | yes | Betfair's event id. |
| `market_type` | string | yes | The market's type. |
| `selection_id` | int64 | yes | Betfair's selection id. On match odds and handicaps it is the team's own id, the same in every market the team plays. |
| `handicap` | double | yes | The runner's handicap line; 0.0 where it has none. |
| `runner_name` | string |  | The selection's name. |
| `sort_priority` | int16 |  | Betfair's display order within the market. |
| `role` | string |  | home, away, draw, over or under: what the runner is in the match (blank if not resolved). |
| `status_final` | string |  | ACTIVE, WINNER, LOSER, REMOVED ... in the last definition. |
| `won` | int8 |  | 1 if settled WINNER, 0 if LOSER, blank otherwise (void, removed, unsettled). On an Asian handicap line a push (stakes returned) is settled LOSER on both sides. |
| `bsp` | double |  | The Betfair starting price, on the few football markets that have one (bsp_market on silver/market); blank otherwise. |
| `ltp_first` | double |  | The first traded price. |
| `ltp_last_pre` | double |  | The last traded price before the off (silver/market.off_pt): the close. |
| `ltp_last` | double |  | The last traded price in the file. |
| `n_ticks` | int32 | yes | Traded-price updates on the runner. |
| `n_ticks_pre` | int32 | yes | Of those, before the off. |
| `first_trade_pt` | timestamp[ms, tz=UTC] |  | The runner's first traded price. |
| `last_pre_trade_pt` | timestamp[ms, tz=UTC] |  | The runner's last traded price before the off. |
| `last_trade_pt` | timestamp[ms, tz=UTC] |  | The runner's last traded price. |
| `in_definition` | bool | yes | False for a (selection, handicap) that traded but is not in the last definition's runners. |

## `silver/definition`

One row each time a market's definition was published: its state at that moment. The history behind silver/market (reschedules, the turn in play, each suspension, the close).

Key: `market_id`, `seq`. Stored as one file a (year, month) partition, sorted by `market_id`, `seq`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `market_id` | string | yes | Betfair's market id. |
| `seq` | int32 | yes | The message's position in the file (0 first). |
| `pt` | timestamp[ms, tz=UTC] | yes | When Betfair published the message. |
| `status` | string |  | OPEN, SUSPENDED, CLOSED or INACTIVE. |
| `in_play` | bool |  | Whether the market was in play. |
| `bet_delay` | int16 |  | The in-play bet delay, seconds. |
| `number_of_active_runners` | int16 |  | Runners still active. |
| `market_time` | timestamp[ms, tz=UTC] |  | The scheduled kick-off as it stood. |
| `version` | int64 |  | Betfair's version of the market. |
| `complete` | bool |  | Betfair's 'complete' flag (all runners added). |
| `cross_matching` | bool |  | Whether cross matching was on. |
| `settled_time` | timestamp[ms, tz=UTC] |  | settledTime, once the market is settled. |

## `silver/spell`

One row a spell of a market's life: a run of messages over which (in play, status) did not change. An in-play SUSPENDED spell is a stoppage Betfair's traders made: a goal, a penalty, a red card, a review.

Key: `market_id`, `spell_no`. Stored as one file a (year, month) partition, sorted by `market_id`, `spell_no`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `market_id` | string | yes | Betfair's market id. |
| `event_id` | int64 | yes | Betfair's event id. |
| `market_type` | string | yes | The market's type. |
| `spell_no` | int32 | yes | The spell's order in the market (0 first). |
| `start_pt` | timestamp[ms, tz=UTC] | yes | The definition that began the spell. |
| `end_pt` | timestamp[ms, tz=UTC] | yes | The definition that ended it (the file's last message for the last spell). |
| `duration_s` | double | yes | end_pt less start_pt, seconds. |
| `in_play` | bool |  | Whether the market was in play during the spell. |
| `status` | string |  | The market's status during the spell. |

## `silver/tick`

One row each time a runner's last traded price was published (BASIC plan: about once a minute a runner while it trades), with the market's state at that moment. The atomic fact of the warehouse.

Key: `market_id`, `selection_id`, `handicap`, `pt`, `seq`. Stored as one file a (year, month) partition, sorted by `market_id`, `selection_id`, `handicap`, `pt`, `seq`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `market_id` | string | yes | Betfair's market id. |
| `event_id` | int64 | yes | Betfair's event id. |
| `market_type` | string | yes | The market's type. |
| `selection_id` | int64 | yes | Betfair's selection id. |
| `handicap` | double | yes | The runner's handicap line; 0.0 where it has none. |
| `pt` | timestamp[ms, tz=UTC] | yes | When Betfair published the price. |
| `seq` | int32 | yes | The message's position in the file: orders prices published at the same instant. |
| `ltp` | double | yes | The last traded price. |
| `in_play` | bool | yes | Whether the price was published in play: from the message of the market's turn in play (silver/market.in_play_pt) on. |
| `pre_off` | bool | yes | Whether the price was published before the off (silver/market.off_pt). Not in play and not pre-off: a market that never turned in play, trading after its scheduled kick-off. |
| `status` | string |  | The market's status then (OPEN or SUSPENDED; a price can be published with the suspension it caused). |
| `t_off_s` | int32 |  | Seconds from pt to the scheduled kick-off (silver/market.market_time_pre); negative after it. |
| `t_inplay_s` | int32 |  | Seconds since the market turned in play; blank before. |

## `gold/event`

One row a Betfair event (a football match): sides, kick-off, markets, result, trading, the closing 1X2.

Key: `event_id`. Stored as one file a (year, month) partition, sorted by `event_id`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `event_id` | int64 | yes | Betfair's event id. |
| `event_name` | string |  | 'Home v Away' as Betfair names the match. |
| `home_name` | string |  | The side named first. |
| `away_name` | string |  | The side named second. |
| `home_id` | int64 |  | The home side's Betfair selection id: the team's id across matches. |
| `away_id` | int64 |  | The away side's Betfair selection id. |
| `country_code` | string |  | Betfair's country code; blank on many cups and internationals. |
| `category` | string | yes | senior, women, youth or reserves, read from the sides' names ('(W)', 'U21', 'Youth', '(Res)', '(Am)', 'Jong', a name with 'II' or ending ' B', ' 2', ' Res'; Betfair cuts a long name off, so '(Res' at the end counts): approximate. A match is not senior when either side is not: a cup tie between a first team and a reserve side is 'reserves'. |
| `kickoff` | timestamp[ms, tz=UTC] |  | The scheduled kick-off as it stood before the off (the match-odds market's market_time_pre; else the earliest of the event's markets). |
| `kickoff_actual` | timestamp[ms, tz=UTC] |  | When the match-odds market turned in play (silver/market.in_play_pt). |
| `kickoff_delay_s` | int32 |  | kickoff_actual less kickoff, seconds. |
| `kickoff_shift_s` | int32 |  | The scheduled kick-off's change before the off, seconds (kickoff less the first definition's): a reschedule when large. |
| `schedule_set` | timestamp[ms, tz=UTC] |  | When the scheduled kick-off was last set (silver/market.schedule_set_pt): a moment before it did not know this schedule. |
| `inplay_turns` | int16 |  | Times the match-odds market turned in play: more than one marks a match to read with care (a turn made by mistake, a drop out of play, an abandonment). |
| `mo_market_id` | string |  | The match-odds market. |
| `ou_market_id` | string |  | The over/under 2.5 goals market. |
| `ah_market_id` | string |  | The Asian handicap market. |
| `n_markets` | int16 | yes | Markets held for the event. |
| `went_in_play` | bool | yes | Whether the match-odds market went in play. |
| `result` | string |  | H, D or A from the match-odds winner; blank if void or unsettled. |
| `over25` | int8 |  | 1 if Over 2.5 Goals won, 0 if Under won; blank if void or unsettled. |
| `void` | bool | yes | The match-odds market closed without a winner (abandoned, postponed). |
| `settled_time` | timestamp[ms, tz=UTC] |  | When Betfair settled the match-odds market. |
| `ft_pt` | timestamp[ms, tz=UTC] |  | When the match-odds market closed: full time, within a few minutes. |
| `mo_ticks_pre` | int32 |  | Match-odds price updates before the off: a liquidity proxy. |
| `mo_ticks_inplay` | int32 |  | Match-odds price updates in play. |
| `ou_ticks_pre` | int32 |  | Over/under price updates before the off. |
| `ou_ticks_inplay` | int32 |  | Over/under price updates in play. |
| `ah_ticks_pre` | int32 |  | Asian handicap price updates before the off. |
| `ah_ticks_inplay` | int32 |  | Asian handicap price updates in play. |
| `mo_susp_inplay` | int16 |  | In-play suspensions of the match-odds market. |
| `close_legs` | int8 |  | Match-odds runners with a price before the off (3: a full closing book). The off is the turn in play, or the scheduled kick-off of a market that never turned. |
| `close_book_sum` | double |  | Sum of 1 / closing price over the match-odds runners priced. |
| `close_age_s` | double |  | Age at the off of the stalest closing price, seconds. |
| `p_home` | double |  | The closing chance of a home win, normalised to the book (full books only). |
| `p_draw` | double |  | The closing chance of a draw. |
| `p_away` | double |  | The closing chance of an away win. |

## `gold/checkpoint`

One row a runner a checkpoint: the price as of a fixed number of minutes before the scheduled kick-off (0: the close), with its age, the activity behind it and the state of its book.

Key: `market_id`, `selection_id`, `handicap`, `cp_min`. Stored as one file a (year, month) partition, sorted by `market_id`, `cp_min`, `selection_id`, `handicap`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `market_id` | string | yes | Betfair's market id. |
| `selection_id` | int64 | yes | Betfair's selection id. |
| `handicap` | double | yes | The runner's handicap line; 0.0 where it has none. |
| `cp_min` | int32 | yes | Minutes before the scheduled kick-off; 0 is the close. |
| `event_id` | int64 | yes | Betfair's event id. |
| `market_type` | string | yes | The market's type. |
| `role` | string |  | home, away, draw, over or under. |
| `line` | double | yes | The book the runner belongs to: 0.0 for match odds and over/under; for an Asian handicap the home side's line (home at h and away at -h are one book). |
| `cp_time` | timestamp[ms, tz=UTC] | yes | The checkpoint's moment (for the close: the off, silver/market.off_pt). |
| `schedule_known` | bool | yes | Whether the schedule the checkpoint is measured from had been set by its moment (always true at the close). False: the kick-off was moved later on, so the checkpoint could not have been known then; keep such rows out of anything that claims to decide at the checkpoint. |
| `ltp` | double | yes | The last traded price as of the checkpoint. |
| `trade_pt` | timestamp[ms, tz=UTC] | yes | When that price was published. |
| `age_s` | double | yes | cp_time less trade_pt, seconds: how stale the price is. |
| `n_ticks_before` | int32 | yes | The runner's price updates up to the checkpoint. |
| `n_ticks_window` | int32 | yes | Of those, since the previous (earlier) checkpoint. |
| `book_n` | int16 | yes | Runners of the book with a price as of the checkpoint. |
| `book_complete` | bool | yes | Whether every runner of the book has one (3 for match odds, 2 for over/under and for a handicap line). |
| `book_sum` | double | yes | Sum of 1 / price over the book's priced runners. |
| `book_max_age_s` | double | yes | The age of the book's stalest price, seconds. |
| `p_raw` | double | yes | 1 / price. |
| `p_norm` | double |  | p_raw / book_sum when the book is complete; blank otherwise. |

## `gold/implied`

One row an event a checkpoint: the market's chances (1X2, over 2.5, the handicap's main line), the goal expectancies they imply, the draw premium (rho) and the disagreement between the handicap and the 1X2.

Key: `event_id`, `cp_min`. Stored as one file a (year, month) partition, sorted by `event_id`, `cp_min`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `event_id` | int64 | yes | Betfair's event id. |
| `cp_min` | int32 | yes | Minutes before the scheduled kick-off; 0 is the close. |
| `cp_time` | timestamp[ms, tz=UTC] |  | The checkpoint's moment. |
| `p_home` | double |  | The chance of a home win, normalised (complete 1X2 book only). |
| `p_draw` | double |  | The chance of a draw. |
| `p_away` | double |  | The chance of an away win. |
| `mo_book_sum` | double |  | Sum of 1 / price over the 1X2 book. |
| `mo_age_s` | double |  | The 1X2 book's stalest price, seconds. |
| `p_over` | double |  | The chance of over 2.5 goals, normalised (complete book only). |
| `ou_book_sum` | double |  | Sum of 1 / price over the over/under book. |
| `ou_age_s` | double |  | The over/under book's stalest price, seconds. |
| `ah_line` | double |  | The Asian handicap main line as the home side's handicap (-0.5: home gives half a goal): the complete line nearest evens. |
| `ah_p_home` | double |  | The home side's normalised chance on that line. |
| `ah_age_s` | double |  | That line's stalest price, seconds. |
| `ah_lines` | int16 |  | Asian handicap lines with both sides priced. |
| `total` | double |  | Expected goals in the match (home + away rate), from the over 2.5 chance. |
| `supremacy` | double |  | Expected goal difference (home - away rate), from the 1X2 at that total. |
| `lam_home` | double |  | The home side's goal expectancy. |
| `lam_away` | double |  | The away side's goal expectancy. |
| `rho` | double |  | The Dixon-Coles low-score correction the market's draw price implies at those rates (negative: more draws than independent Poisson). Blank where the draw chance is more than 0.02 from any such a market could show: the books are not prices of one moment, and the row is one to distrust. Within 0.02 of the band, its edge. |
| `sup_ah` | double |  | The supremacy the handicap's main line implies at the same total and rho (the usual rho, -0.08, where the row has none). |
| `sup_gap` | double |  | sup_ah less supremacy: the handicap market against the 1X2, in goals. |

## `gold/path`

One row a match-odds or over/under runner a decision checkpoint (1440, 360, 120, 60, 15 minutes before the scheduled kick-off): what its last traded price did up to that moment, in the logit of the raw chance (x = ln(p / (1 - p)), p = 1 / price; positive is shortening). Every column uses only ticks published at or before cp_time. A window is the last L before the checkpoint, cut at the runner's first trade where that is later. The feature library of the closing-price model: joins gold/checkpoint and gold/move on the key.

Key: `market_id`, `selection_id`, `handicap`, `cp_min`. Stored as one file a (year, month) partition, sorted by `market_id`, `cp_min`, `selection_id`, `handicap`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `market_id` | string | yes | Betfair's market id. |
| `selection_id` | int64 | yes | Betfair's selection id. |
| `handicap` | double | yes | The runner's handicap line: 0.0 on these markets. |
| `cp_min` | int32 | yes | The checkpoint: minutes before the scheduled kick-off (1440, 360, 120, 60 or 15). |
| `event_id` | int64 | yes | Betfair's event id. |
| `market_type` | string | yes | MATCH_ODDS or OVER_UNDER_25. |
| `role` | string |  | home, away, draw, over or under. |
| `cp_time` | timestamp[ms, tz=UTC] | yes | The checkpoint's moment. |
| `x` | double | yes | The logit of the raw chance at the checkpoint: ln(p / (1 - p)) with p = 1 / the last traded price as of cp_time (-ln(price - 1)). Every move below is a difference of this. |
| `ret_15m` | double |  | The change in x from the price as of 15 minutes before the checkpoint to the price at the checkpoint; 0 if no tick since; blank when the runner had no price that far back. |
| `ret_30m` | double |  | The change in x from the price as of 30 minutes before the checkpoint to the price at the checkpoint; 0 if no tick since; blank when the runner had no price that far back. |
| `ret_1h` | double |  | The change in x from the price as of 1 hour before the checkpoint to the price at the checkpoint; 0 if no tick since; blank when the runner had no price that far back. |
| `ret_2h` | double |  | The change in x from the price as of 2 hours before the checkpoint to the price at the checkpoint; 0 if no tick since; blank when the runner had no price that far back. |
| `ret_6h` | double |  | The change in x from the price as of 6 hours before the checkpoint to the price at the checkpoint; 0 if no tick since; blank when the runner had no price that far back. |
| `ret_24h` | double |  | The change in x from the price as of 24 hours before the checkpoint to the price at the checkpoint; 0 if no tick since; blank when the runner had no price that far back. |
| `ret_first` | double | yes | The change in x since the runner's first traded price (0 when that is its only one). |
| `hours_since_first` | double | yes | Hours from the runner's first traded price to the checkpoint: how much history the windows have to read. |
| `rv_1h` | double | yes | Realised volatility of the last hour: the root of the sum of the squared changes in x made by the ticks in (cp_time - 1 h, cp_time]; 0 when none changed it. |
| `rv_6h` | double | yes | The same over the last 6 hours (since the first trade where that is later). |
| `rv_24h` | double | yes | The same over the last 24 hours (since the first trade where that is later). |
| `n_ticks_15m` | int32 | yes | Price updates in the last 15 minutes: (cp_time - 15 min, cp_time]. Activity: BASIC has no volume. The first trade counts as an update. |
| `n_ticks_1h` | int32 | yes | Price updates in the last hour. |
| `n_ticks_6h` | int32 | yes | Price updates in the last 6 hours. |
| `n_ticks_24h` | int32 | yes | Price updates in the last 24 hours. |
| `slope_6h` | double |  | Trend: the least-squares slope of x on time over the last 6 hours, in x an hour; the price path is a step function and each price weighs by the time it stood. 0 when the price did not change in the window; blank when the runner's history covers under 15 minutes of it. |
| `slope_24h` | double |  | The same over the last 24 hours. |
| `slope_r2_6h` | double |  | The fit of slope_6h: the share of the path's time-weighted variance the line explains (0 to 1). Blank where slope_6h is, and when the price did not move in the window. |
| `slope_r2_24h` | double |  | The fit of slope_24h. |
| `er_6h` | double |  | Efficiency ratio of the last 6 hours: the absolute net change in x over the sum of the absolute changes (1: every tick the same way; near 0: a round trip). Blank when no tick changed the price in the window. |
| `er_24h` | double |  | The same over the last 24 hours. |
| `range_pos_24h` | double |  | Where x stands between the lowest and the highest x of the last 24 hours (the price as of the window's start included): 0 at the low, 1 at the high. Blank when the price did not move in the window. |
| `drawdown_24h` | double | yes | The highest x of the last 24 hours less the current x (0 or more): how far the chance has fallen from its high. |
| `runup_24h` | double | yes | The current x less the lowest x of the last 24 hours (0 or more): how far the chance has risen from its low. |
| `jump_abs_2h` | double | yes | The largest absolute change in x made by one tick in the last 2 hours; 0 when no tick changed the price. |
| `jump_sign_2h` | int8 | yes | That change's sign: 1 a shortening, -1 a drift, 0 when there was none. Of equal changes the latest is taken. |
| `jump_min_2h` | double |  | Minutes from that tick to the checkpoint; blank when there was none. |
| `twap_dev_1h` | double |  | Mean reversion: the time-weighted average x of the last hour less the current x (positive: the price now is below its recent average chance). Blank only when the runner's first trade is the checkpoint's own moment. |
| `twap_dev_6h` | double |  | The same against the last 6 hours' time-weighted average. |
| `last_dx` | double |  | The change in x made by the runner's latest tick (the one whose price stands at the checkpoint): the move a bid-ask bounce would give back. Blank when it has traded once. |
| `run_len` | int32 | yes | The run the latest tick ends: how many consecutive ticks moved x the same way, signed (3: three shortenings in a row; -2: two drifts). 0 when it has traded once, or when the latest tick repeated the price before it. |
| `tick_imb_1h` | double |  | Tick imbalance of the last hour: (ticks that raised x - ticks that lowered it) / (both). The tick test as a stand-in for signed order flow. Blank when no tick changed the price. |
| `tick_imb_6h` | double |  | The same over the last 6 hours. |
| `ac1_6h` | double |  | First-order autocorrelation of the tick changes of the last 6 hours: the sum of products of consecutive changes over the root of the product of their sums of squares (-1 to 1; negative: bounce, positive: drift). Blank with fewer than 5 pairs of consecutive changes. |
| `ac1_24h` | double |  | The same over the last 24 hours. |
| `roll_24h` | double |  | Roll's implied spread over the last 24 hours, in x: 2 * sqrt(-c) where c is the mean product of consecutive tick changes; 0 when c is positive. Blank with fewer than 5 pairs. |
| `book_catchup` | double |  | The move the rest of the book has made since this runner last traded, as the change in this runner's x that would restore the book's sum of raw chances to what it was at that trade: logit(p + d) - logit(p) with d = (the other runners' 1 / price as of this runner's last trade, summed) less (the same as of the checkpoint). 0 when no other runner traded since. Blank unless every other runner of the book (two for match odds, one for over/under) had a price by this runner's last trade. |

## `gold/goal`

One row an inferred goal: when, which side, the score after it, how sure, and the price moves that say so. Inferred from Betfair's in-play prices (match odds, over/under 2.5, Asian handicap) and reconciled with how the markets settled; not a record of the match.

Key: `event_id`, `goal_no`. Stored as one file a (year, month) partition, sorted by `event_id`, `goal_no`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `event_id` | int64 | yes | Betfair's event id. |
| `goal_no` | int16 | yes | The goal's order in the match, 1 first: the total after it. |
| `pt` | timestamp[ms, tz=UTC] | yes | The goal's moment as the data places it: the start of the suspension that marks it, else the first price message that shows the move (the goal is up to a minute before: the gap between two messages). Where timed is false, a placeholder: after the last price. |
| `pt_before` | timestamp[ms, tz=UTC] |  | The last price message before the move: the goal is after it. |
| `suspended` | bool | yes | Whether a suspension of the match's markets marks the goal (about one in four: a suspension shorter than the minute between messages is not in the files). |
| `timed` | bool | yes | False for a goal no price move shows (evidence 'settle'): the settlement says it was scored, the data does not say when; pt, elapsed_min and match_min then place it after the last price. |
| `elapsed_min` | double | yes | Minutes from the market's turn in play to pt, wall clock (half time included). |
| `half` | int8 | yes | 1 or 2: the half pt falls in, by the match's half-time estimate. |
| `match_min` | double | yes | The football minute: elapsed_min in the first half (at most 45), 45 plus the minutes since the restart in the second (at most 90). The restart is an estimate: see the module's measured minute errors. |
| `side` | string | yes | home or away: the scorer; blank where the moves do not say. |
| `confidence` | double | yes | How far to trust the goal, 0 to 1: the share of goals with this evidence, reached this way (the price reading agreed with the settlement, kept through a reconciliation, added by one), that matched a real goal on 2026's odd-numbered months. |
| `evidence` | string | yes | What says it is a goal. ou+ah: Under out, match odds, and the handicap margin stepped; ou: Under out and match odds, no handicap reading; ou-ah: Under out and match odds but the handicap margin did not step; ah+mo: the margin stepped and match odds moved, the over/under silent; mo: match odds alone; mo-ou: a very large match-odds move against a flat over/under; third: the over/under market stopped for good (the third goal); ah: a handicap step with no match-odds or over/under move; settle: no price move, the settlement alone. |
| `by_settlement` | bool | yes | The goal is in the list, or on this side, because the settlement asks for it: a sub-threshold move promoted, a side turned, a side given to a goal whose moves did not say, or a goal with no move at all (evidence 'settle'). |
| `home_score` | int16 |  | Home goals after this goal; blank from the first goal with no side. |
| `away_score` | int16 |  | Away goals after this goal; blank from the first goal with no side. |
| `mo_jump` | double |  | The match-odds move: the home price's logit move less the away price's (logit of 1 / price); positive towards the home side. |
| `prob_jump` | double |  | The summed absolute change of the three raw 1X2 chances. |
| `ou_jump` | double |  | Under's move out: the mean of Under's logit fall and Over's rise; positive for a goal; blank when neither traded in the window. |
| `ah_jump` | double |  | The handicap margin's step across the goal, goals; positive towards the home side; blank without a reading on both sides. |
| `home_before` | double |  | The home price before the move (median-filtered). |
| `home_after` | double |  | The home price after it. |
| `draw_before` | double |  | The draw price before the move. |
| `draw_after` | double |  | The draw price after it. |
| `away_before` | double |  | The away price before the move. |
| `away_after` | double |  | The away price after it. |
| `under_before` | double |  | The Under 2.5 price before the move. |
| `under_after` | double |  | The Under 2.5 price after it. |
| `over_before` | double |  | The Over 2.5 price before the move. |
| `over_after` | double |  | The Over 2.5 price after it. |

## `gold/timeline`

One row a match that went in play: its inferred goals, the time each side led, its stoppages, half time, whether the in-play record can be used, whether the price reading agreed with how the markets settled, and how far the final list can be trusted.

Key: `event_id`. Stored as one file a (year, month) partition, sorted by `event_id`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `event_id` | int64 | yes | Betfair's event id. |
| `usable` | bool | yes | The data rule: at least 60 price rows in play, no gap between two rows over 12 minutes, 100 to 140 minutes from the turn in play to the close, a settled result, and first in-play prices near the closing ones (a market that went in play late has lost its first goals). |
| `consistent` | bool |  | The price reading alone (before any reconciliation) passed every check the settlement allows: agrees_result, agrees_over25 and agrees_margin, those that are not blank, and (where a result or a margin is settled) every goal has a side. Blank when none can be made. |
| `reconciled` | bool | yes | The price reading disagreed with the settlement and the list was edited at least cost to agree with it (n_edits says how much). False with consistent false: no edit within reach agrees, the list is the price reading. |
| `reliability` | double |  | The share of matches like this one whose final score was exactly the real one on 2026's odd-numbered months (football-data.co.uk scores): by how the list was reached (price reading agreed; one edit; several; a goal with no price move) and its goals (0 to 2, 3, 4, 5 or more). Blank where usable is false, no agreement with the settlement was reached, a goal still has no side, or the third goal is not where the over/under market stopped (agrees_third false): not measured there, and such lists are right far less often. |
| `n_goals` | int16 | yes | Inferred goals, after reconciliation. |
| `home_goals` | int16 | yes | Inferred home goals. |
| `away_goals` | int16 | yes | Inferred away goals. |
| `n_side_unknown` | int16 | yes | Inferred goals with no side. |
| `n_edits` | int16 | yes | Edits the reconciliation made: goals removed, added, turned to the other side. |
| `n_unplaced` | int16 | yes | Of the goals, those no price move shows (evidence 'settle', timed false): their minute is not known. |
| `min_confidence` | double |  | The least sure goal's confidence; blank without goals. |
| `agrees_result` | bool |  | The price reading's goal difference has the sign of the settled 1X2 result; blank without a result. |
| `agrees_over25` | bool |  | The price reading's three or more goals is the settled Over 2.5; blank without one. |
| `final_margin` | int16 |  | The settled goal difference, home less away, read off the Asian handicap lines that won (the lines stop at four: 4 and -4 stand for four or more); blank without a settled handicap market. |
| `agrees_margin` | bool |  | The price reading's goal difference is final_margin; blank without one. |
| `agrees_third` | bool |  | The list's third goal is within three minutes of the moment the over/under market stopped for good (ou_closed_min: the real third goal is, on 94 per cent of matches). False: a goal before it is missing or false, and the score is right less than half the time (second review round: 48 of 110 such matches with a real score). Blank where the market did not stop early, the list has under three goals or its third has no minute. |
| `n_stoppages` | int16 | yes | In-play suspensions of the match's markets before the final one (overlapping suspensions of its markets count once). |
| `n_goal_stoppages` | int16 | yes | Of those, the ones that mark an inferred goal. |
| `n_other_stoppages` | int16 | yes | The rest: a penalty not scored, a red card, a review, an injury, a goal disallowed. |
| `n_disallowed` | int16 | yes | Goals undone by a later move back (Under sharply in again). |
| `first_goal_min` | double |  | match_min of the first goal the data can time (a goal with no minute has only a placeholder); blank where there is none. |
| `last_goal_min` | double |  | match_min of the last goal the data can time. |
| `min_home_leading` | double |  | Minutes of play the home side led (half time left out); blank where a goal has no side. A goal with timed false counts from the last price on. |
| `min_level` | double |  | Minutes of play the score was level. |
| `min_home_trailing` | double |  | Minutes of play the home side trailed. |
| `ht_found` | bool | yes | Whether a quiet stretch marks half time in the match's own data (else the restart is read from the minutes in play alone). |
| `ht_start_min` | double | yes | Elapsed minute half time began (an estimate). |
| `ht_end_min` | double | yes | Elapsed minute the second half began (an estimate). |
| `inplay_min` | double |  | Minutes from the turn in play to the match-odds market's close. |
| `n_rows` | int32 | yes | Price rows in play (messages with a match-odds or over/under price). |
| `max_gap_min` | double |  | The longest gap between two price rows (or the turn in play and the first, or the last and the close), minutes. |
| `start_jump` | double |  | The summed absolute change of the three 1X2 chances from the close to the first in-play row: large when the market went in play after a goal. |
| `ou_closed_min` | double |  | Elapsed minute the over/under market stopped for good, where that was in play and before full time and Over won: the third goal as Betfair settled it. |

## `gold/inplay_minute`

One row a match a minute in play: the market's chances of home win, draw and away win as of that minute, their quality, the over 2.5 chance while open, the phase and football minute, and the inferred score.

Key: `event_id`, `minute`. Stored as one file a (year, month) partition, sorted by `event_id`, `minute`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `event_id` | int64 | yes | Betfair's event id. |
| `minute` | int16 | yes | Whole minutes since the match-odds market went in play (wall clock: half time included); 0 is the kick-off. |
| `pt` | timestamp[ms, tz=UTC] | yes | The minute's moment. |
| `phase` | string | yes | first_half, half_time or second_half, by the match's half-time estimate (gold/timeline; the 2026 medians where it has none). |
| `match_min` | double | yes | The football minute: the elapsed minute in the first half (at most 45), 45 at half time, 45 plus the minutes since the restart in the second (at most 90). |
| `p_home` | double |  | The chance of a home win as of the minute, normalised to the three legs. |
| `p_draw` | double |  | The chance of a draw. |
| `p_away` | double |  | The chance of an away win. |
| `mo_legs` | int8 | yes | Match-odds legs with a price as of the minute. |
| `mo_book_sum` | double |  | Sum of 1 / price over the legs priced. |
| `mo_max_age_s` | double |  | The stalest leg's age at the minute, seconds. |
| `mo_stale_mass` | double |  | The share of the book's sum carried by stale legs: priced over five minutes ago, or before the latest goal the data can time. |
| `ok` | bool | yes | All three legs priced, the book's sum between 0.90 and 1.15, and no more than a tenth of it in stale legs: a reading to trust. |
| `p_over` | double |  | The chance of over 2.5 goals as of the minute, normalised to its book; blank once that market has stopped trading (ou_open false) or where a leg is missing or stale. |
| `ou_open` | bool | yes | Whether the over/under 2.5 market was still trading at the minute: before the suspension that ran into its close (the third goal, or full time), or its close where no suspension did. Its last prices are from before the goal that settled it. |
| `home_score` | int16 |  | Home goals so far as gold/goal infers them; blank where the match has no usable timeline, and from the first goal whose scorer the prices do not say. A goal the data cannot time (gold/goal.timed false) counts from the last price on. |
| `away_score` | int16 |  | Away goals so far, likewise. |
| `score_ok` | bool | yes | The score columns can be relied on minute by minute: the timeline is usable, its final score agrees with the settlement (as read, or after reconciliation), every goal has a minute and a side, and the third goal is where the over/under market stopped (gold/timeline.agrees_third not false). The flag is made with the match's result: a table cut down to score_ok matches is not a fair sample for testing the prices against results. |

## `gold/inplay`

One row a match that went in play: the time-averaged in-play chances and the points they are worth against the pre-match expectation, the half-time chances, the extremes, and when the result was decided.

Key: `event_id`. Stored as one file a (year, month) partition, sorted by `event_id`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `event_id` | int64 | yes | Betfair's event id. |
| `minutes` | int16 | yes | Minutes on the grid: from the kick-off to the match-odds market's close. |
| `n_ok` | int16 | yes | Minutes with an ok reading. |
| `filled` | bool | yes | At least thirty ok minutes: the summary columns are filled. |
| `avg_p_home` | double |  | The home win chance averaged over every minute of the match (a minute without an ok reading takes the last one before it; the close before the first). |
| `avg_p_draw` | double |  | The draw's, likewise. |
| `avg_p_away` | double |  | The away win's, likewise. |
| `xpts_pre_home` | double |  | The home side's expected points at the close: 3 p_home + p_draw. |
| `xpts_pre_away` | double |  | The away side's. |
| `xpts_inplay_home` | double |  | The home side's expected points averaged over the match: 3 avg_p_home + avg_p_draw. |
| `xpts_inplay_away` | double |  | The away side's. |
| `perf_home` | double |  | xpts_inplay_home less xpts_pre_home: how the match went for the home side against what the market expected of it; close to zero on average. |
| `perf_away` | double |  | The away side's. |
| `ht_p_home` | double |  | The home win chance at the middle of half time (an ok reading within three minutes of it). |
| `ht_p_draw` | double |  | The draw's at half time. |
| `ht_p_away` | double |  | The away win's at half time. |
| `max_p_home` | double |  | The home side's best moment: its highest win chance in play. |
| `min_p_home` | double |  | Its lowest. |
| `max_p_away` | double |  | The away side's highest win chance in play. |
| `min_p_away` | double |  | Its lowest. |
| `travel` | double |  | How far the chances moved: the sum over consecutive ok minutes of half the absolute change of the three chances (an eventful match travels far). |
| `fav_changes` | int16 |  | Times the most likely result changed between consecutive ok minutes. |
| `decided_min` | int16 |  | The first grid minute (whole minutes since the turn in play: wall clock, half time included) from which the eventual result's chance stayed at or above 0.90 to the end; blank where it never did or the match has no result. |
| `decided_match_min` | double |  | The football minute of decided_min (gold/inplay_minute's match_min: 45 through half time, 90 at most): the one to quote. |
| `result_min_p` | double |  | The lowest chance the eventual result had in play: how unlikely it once looked. |

## `gold/move`

One row a runner a checkpoint: the move of its normalised chance from the checkpoint to the close, with the quality of both ends. The training target of a closing-price model.

Key: `market_id`, `selection_id`, `handicap`, `cp_min`. Stored as one file a (year, month) partition, sorted by `cp_min`, `market_id`, `selection_id`, `handicap`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `market_id` | string | yes | Betfair's market id. |
| `selection_id` | int64 | yes | Betfair's selection id. |
| `handicap` | double | yes | The runner's handicap line; 0.0 where it has none. |
| `cp_min` | int32 | yes | The checkpoint: minutes before the scheduled kick-off (never 0). |
| `event_id` | int64 | yes | Betfair's event id. |
| `market_type` | string | yes | The market's type. |
| `role` | string |  | home, away, draw, over or under. |
| `line` | double | yes | The book: 0.0, or the home side's handicap line. |
| `is_main` | bool | yes | Asian handicap: the market's main line at the checkpoint (nearest evens among the lines freshly traded). True for match odds and over/under. |
| `kickoff` | timestamp[ms, tz=UTC] |  | The scheduled kick-off. |
| `cp_time` | timestamp[ms, tz=UTC] | yes | The checkpoint's moment. |
| `ltp_cp` | double | yes | The price at the checkpoint. |
| `ltp_close` | double | yes | The closing price. |
| `p_cp` | double | yes | The normalised chance at the checkpoint. |
| `p_close` | double | yes | The normalised chance at the close. |
| `y_log` | double | yes | ln(p_close / p_cp): the move, in log chance. |
| `y_logit` | double | yes | logit(p_close) - logit(p_cp): the move, in log odds. |
| `price_ratio` | double | yes | ltp_cp / ltp_close: above 1 when the price shortened. |
| `cp_book_sum` | double | yes | The book sum at the checkpoint. |
| `cp_age_s` | double | yes | The runner's own price age at the checkpoint, seconds. |
| `cp_book_age_s` | double | yes | The stalest price of its book at the checkpoint, seconds. |
| `close_book_age_s` | double | yes | The stalest price of its book at the close, seconds. |
| `n_ticks_before` | int32 | yes | The runner's price updates up to the checkpoint. |
| `n_ticks_after` | int32 | yes | Its price updates between the checkpoint and the close. |
| `cp_fresh` | bool | yes | The checkpoint's book is fresh enough (the screen's checkpoint side only): known at the checkpoint, so a trading rule may select on it. |
| `usable` | bool | yes | cp_fresh and a fresh closing book: a reliable label. The close's freshness is not known at the checkpoint: use it to choose training and scoring rows, never to choose trades. |
| `mo_ticks_pre` | int32 |  | The match's pre-off match-odds price updates, the whole pre-off period (not known at the checkpoint: for reporting, not for selecting trades). |
| `tier` | string | yes | The match's liquidity tier from mo_ticks_pre: A, B, C or D (for reporting; use n_ticks_before to select at the checkpoint). |
| `schedule_known` | bool | yes | Whether the schedule the checkpoint is measured from had been set by its moment (gold/checkpoint): false rows could not have been decided at the checkpoint. |
| `kickoff_shift_s` | int32 |  | The scheduled kick-off's change before the off, seconds (a reschedule when large); schedule_known is the row-level test. |
| `country_code` | string |  | Betfair's country code. |
| `category` | string |  | senior, women, youth or reserves. |
| `won` | int8 |  | Whether the runner won, where the role says so from the event's result (match odds and over/under); blank otherwise. |

## `gold/team`

One row a Betfair team id (the selection id of a side on the match-odds market).

Key: `team_id`. Stored as a single file, sorted by `team_id`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `team_id` | int64 | yes | Betfair's selection id for the team. |
| `team_name` | string |  | The name it is most often given. |
| `n_names` | int32 | yes | Distinct names seen for the id. |
| `n_events` | int32 | yes | Matches held for the team. |
| `n_home` | int32 | yes | Of those, at home. |
| `n_played` | int32 | yes | Of those, that went in play. |
| `country_code` | string |  | The country code its home matches most often carry. |
| `first_kickoff` | timestamp[ms, tz=UTC] |  | Its first match held. |
| `last_kickoff` | timestamp[ms, tz=UTC] |  | Its last match held. |

## `gold/team_rating`

One row a team a match: the team's market-implied rating before the match (from closes fixed before the match's UTC day only) and after it. A team's latest rating is its last row's.

Key: `team_id`, `event_id`. Stored as a single file, sorted by `team_id`, `kickoff`, `event_id`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `team_id` | int64 | yes | Betfair's team id (the side's selection id on match odds). |
| `event_id` | int64 | yes | Betfair's event id: the match. |
| `kickoff` | timestamp[ms, tz=UTC] | yes | The scheduled kick-off. |
| `is_home` | bool | yes | Whether the team is the side named first. |
| `opponent_id` | int64 | yes | The other side's team id. |
| `country_code` | string |  | Betfair's country code for the match; blank on many cups and internationals. |
| `as_of` | timestamp[ms, tz=UTC] | yes | The moment the rating before the match is as of: the start of the UTC day of the kick-off. Only matches that went in play before it are in the rating. |
| `n_prior` | int32 | yes | Matches behind the rating before this one: the team's matches of the 400 days before as_of whose close was fixed before it. 0: the team has no rating of its own (it is new, or back after more than 400 days). |
| `days_since` | double |  | Days from the team's last rated match, however old, to this kick-off; blank if it has none. |
| `att` | double | yes | Attack before the match: (strength + tempo) / 2, in log goals. |
| `def` | double | yes | Defence before the match: (strength - tempo) / 2, in log goals (higher concedes less). |
| `strength` | double | yes | att + def before the match: what decides the ratio of the two sides' goal expectancies. A difference of 0.1 is about 10 per cent on that ratio. Only differences between teams mean anything; the scale's centre is the average team Betfair lists. With n_prior = 0 it is the opponent's (0 if both sides are new). |
| `tempo` | double | yes | att - def before the match: what the team adds to the log of the product of the two goal expectancies (high: its matches are priced for goals). |
| `att_sd` | double | yes | Standard deviation of att (conditional on the other teams' ratings). |
| `def_sd` | double | yes | Standard deviation of def (conditional). |
| `strength_sd` | double | yes | Standard deviation of strength, conditional on the other teams' ratings: NOISE_S over the root of the decayed count of matches behind it plus the prior. |
| `tempo_sd` | double | yes | Standard deviation of tempo (conditional). |
| `pool_teams` | int32 | yes | Teams on the same scale as this one before the match: those joined to it by chains of the rated matches behind the rating (itself included; 1: none). Strengths compare within a pool only: women's football, many youth leagues and unlinked lower leagues are pools of their own, each centred on its own average. |
| `linked` | bool | yes | Whether the two sides were on one scale going into the match (in one pool): the opponent's rating then compares with this side's. False: the two numbers are from different scales (or one side has none). |
| `rated` | bool | yes | Whether this match's close enters later ratings: it went in play with a complete closing 1X2 and over/under book that one market could have made (gold/implied gives it a rho). |
| `after_as_of` | timestamp[ms, tz=UTC] | yes | The moment the rating after the match is as of: the start of the UTC day after the kick-off. |
| `n_after` | int32 | yes | Matches behind the rating after the match. |
| `att_after` | double |  | Attack after the match; blank while the team has no rated match. |
| `def_after` | double |  | Defence after the match. |
| `strength_after` | double |  | Strength after the match. |
| `tempo_after` | double |  | Tempo after the match. |
| `end_as_of` | timestamp[ms, tz=UTC] | yes | The moment of the table's last solve: the start of the UTC day after the last rated close held. The same on every row. |
| `strength_end` | double |  | The team's strength in that last solve (the same on each of its rows): every team's from one solve, so these are the ones to rank teams on. strength_after is as of the day after each match, on a scale that drifts as teams join. Blank for a team with no rated match in the 400 days before. |
| `pool_teams_end` | int32 | yes | Teams on the team's scale in the last solve (itself included). |

## `gold/rating_gap`

One row an event a checkpoint (1440, 360, 120, 60, 15 minutes before the scheduled kick-off, and 0: the close): what the two sides' ratings as of the checkpoint imply for the match, what the market says there, and the gap (ratings less market).

Key: `event_id`, `cp_min`. Stored as a single file, sorted by `event_id`, `cp_min`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `event_id` | int64 | yes | Betfair's event id. |
| `cp_min` | int32 | yes | Minutes before the scheduled kick-off; 0 is the close. |
| `cp_time` | timestamp[ms, tz=UTC] |  | The checkpoint's moment. |
| `kickoff` | timestamp[ms, tz=UTC] | yes | The scheduled kick-off. |
| `home_id` | int64 | yes | The home side's team id. |
| `away_id` | int64 | yes | The away side's team id. |
| `as_of` | timestamp[ms, tz=UTC] | yes | The moment the ratings are as of: the start of the checkpoint's UTC day. |
| `n_home` | int32 | yes | Matches behind the home side's rating at the checkpoint (0: none of its own, the opponent's is used). |
| `n_away` | int32 | yes | Matches behind the away side's rating at the checkpoint. |
| `linked` | bool | yes | Whether the two sides' ratings are on one scale at the checkpoint: a chain of rated matches joins them. False: each is rated within its own pool (or one has no rating), and the rating columns say little about the match. |
| `rt_home_adv` | double | yes | The home advantage in force (global plus the event's country), in log rate ratio. |
| `rt_log_ratio` | double | yes | The ratings' log(lam_home / lam_away): home advantage + home strength - away strength. |
| `rt_log_ratio_sd` | double | yes | Its standard deviation from the two sides' conditional standard deviations. |
| `rt_lam_home` | double | yes | The home goal expectancy the ratings imply. |
| `rt_lam_away` | double | yes | The away goal expectancy the ratings imply. |
| `rt_total` | double | yes | The ratings' expected goals in the match. |
| `rt_supremacy` | double | yes | The ratings' expected goal difference (home - away). |
| `rt_p_home` | double | yes | The home win chance at those rates (Dixon-Coles score model with the market's usual low-score correction RHO). |
| `rt_p_draw` | double | yes | The draw chance at those rates. |
| `rt_p_away` | double | yes | The away win chance at those rates. |
| `rt_p_over` | double | yes | The chance of over 2.5 goals at those rates. |
| `mk_log_ratio` | double |  | The market's log(lam_home / lam_away) at the checkpoint (gold/implied); blank without complete 1X2 and over/under books. |
| `mk_total` | double |  | The market's expected goals at the checkpoint. |
| `mk_supremacy` | double |  | The market's expected goal difference at the checkpoint. |
| `mk_p_home` | double |  | The market's home win chance at the checkpoint (complete 1X2 only). |
| `mk_p_draw` | double |  | The market's draw chance at the checkpoint. |
| `mk_p_away` | double |  | The market's away win chance at the checkpoint. |
| `mk_p_over` | double |  | The market's chance of over 2.5 goals at the checkpoint. |
| `gap_log_ratio` | double |  | rt_log_ratio less mk_log_ratio: positive when the ratings make the home side stronger than the price does. |
| `gap_supremacy` | double |  | rt_supremacy less mk_supremacy, in goals. |
| `gap_total` | double |  | rt_total less mk_total, in goals. |
| `gap_logit_home` | double |  | logit(rt_p_home) less logit(mk_p_home): the gap on the scale of gold/move.y_logit for the home runner (needs only the 1X2 book). |
| `gap_logit_over` | double |  | logit(rt_p_over) less logit(mk_p_over): the gap on the over 2.5 runner's scale. |

## `club/team_match`

One row a team a match: the market's expectation, the result against it, the in-play record and the ratings.

Key: `team_id`, `event_id`. Stored as a single file, sorted by `team_id`, `kickoff`, `event_id`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `team_id` | int64 | yes | Betfair's team id. |
| `event_id` | int64 | yes | Betfair's event id. |
| `kickoff` | timestamp[ms, tz=UTC] |  | The scheduled kick-off. |
| `is_home` | bool | yes | Whether the team was the home side. |
| `opponent_id` | int64 |  | The other side's team id. |
| `team_name` | string |  | The team's name in this match. |
| `opponent_name` | string |  | The opponent's. |
| `country_code` | string |  | The match's country code (Betfair's; blank on many cups). |
| `category` | string |  | senior, women, youth or reserves. |
| `p_win` | double |  | The team's chance of winning at the close. |
| `p_draw` | double |  | The draw's chance at the close. |
| `p_loss` | double |  | The team's chance of losing at the close. |
| `xpts_pre` | double |  | Expected points at the close: 3 p_win + p_draw. |
| `lam_for` | double |  | The goals the market expected the team to score. |
| `lam_against` | double |  | The goals it expected the team to concede. |
| `move_24h` | double |  | The move of the team's win chance from 24 hours before the kick-off to the close, in log odds: positive when the market came for the team. |
| `move_6h` | double |  | The same from 6 hours before. |
| `move_60m` | double |  | The same from 60 minutes before (team news is out by then). |
| `result` | string |  | W, D or L; blank if the match has no result. |
| `points` | int8 |  | 3, 1 or 0. |
| `goals_for` | int16 |  | Goals scored as gold/timeline infers them; blank unless the timeline is usable, its final score agrees with the settlement (as read, or after reconciliation), every goal has a side and the third goal is where the over/under market stopped. |
| `goals_against` | int16 |  | Goals conceded, likewise. |
| `score_reliability` | double |  | How often a timeline like this one had exactly the real score (gold/timeline.reliability): filter on it where exact scores matter. The goal difference is far safer than the two counts: the usual error is a goal each side missed late in a high-scoring match. |
| `pts_vs_xpts` | double |  | points less xpts_pre: the result against the expectation. |
| `gd_vs_exp` | double |  | Goal difference less the expected one (lam_for - lam_against). |
| `xpts_inplay` | double |  | Expected points averaged over every minute of the match (gold/inplay); blank where the match traded too little in play. |
| `perf` | double |  | How the match went against the expectation: xpts_inplay less xpts_pre, less what sides of that pre-match expectation average on it (0.0591 - 0.0440 xpts_pre: the reading leans against strong sides; fitted on 2026's odd months). |
| `luck` | double |  | pts_vs_xpts less perf: what the result gave beyond the run of the match. |
| `ht_p_win` | double |  | The team's win chance at half time. |
| `max_p_win` | double |  | Its highest win chance in play. |
| `min_p_win` | double |  | Its lowest. |
| `min_leading` | double |  | Minutes the team spent ahead (inferred timeline; blank where a goal of the match has no minute). |
| `min_level` | double |  | Minutes level. |
| `min_trailing` | double |  | Minutes behind. |
| `travel` | double |  | How far the match's chances moved in play (gold/inplay): an eventful match. |
| `decided_min` | double |  | The football minute from which the result was settled in the market's eyes (gold/inplay.decided_match_min: its chance stayed at or above 0.90 to the end). |
| `strength` | double |  | The team's market-implied strength going into the match (gold/team_rating); blank until it has a match behind it. |
| `strength_sd` | double |  | That rating's standard deviation. |
| `n_prior` | int32 |  | Earlier matches behind the rating. |
| `pool_teams` | int32 |  | Teams on the same scale as this one going in: strengths compare only within a pool (gold/team_rating). |
| `opp_strength` | double |  | The opponent's strength going in; blank where the two sides were not on one scale then (gold/team_rating.linked false). |
| `strength_after` | double |  | The team's strength once this match's closing price is known. |

## `club/team_summary`

One row a team: its matches in the warehouse added up, its rating now, and its standing in its country.

Key: `team_id`. Stored as a single file, sorted by `team_id`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `team_id` | int64 | yes | Betfair's team id. |
| `team_name` | string |  | The name it is most often given. |
| `country_code` | string |  | The country code its home matches most often carry. |
| `category` | string |  | senior, women, youth or reserves: its matches' most common. |
| `matches` | int32 | yes | Matches with a result. |
| `first_kickoff` | timestamp[ms, tz=UTC] |  | Its first match with a result. |
| `last_kickoff` | timestamp[ms, tz=UTC] |  | Its last. |
| `wins` | int32 | yes | Matches won. |
| `draws` | int32 | yes | Matches drawn. |
| `losses` | int32 | yes | Matches lost. |
| `ppm` | double |  | Points a match. |
| `xpts_pre_pm` | double |  | Expected points a match at the close. |
| `pts_vs_xpts_pm` | double |  | Points a match beyond the expectation. |
| `pts_vs_xpts_se` | double |  | Its standard error: read the figure against this. |
| `n_inplay` | int32 | yes | Matches with an in-play record. |
| `perf_pm` | double |  | In-play performance a match (club/team_match.perf): how its matches went against the pre-match expectation, every minute counting. It describes a run of matches; it has not been shown to persist for a team. Over the n_inplay matches with an in-play record: with luck_pm it adds up to the points against expectation of those matches, which is not pts_vs_xpts_pm where some matches have no record. |
| `perf_se` | double |  | Its standard error. |
| `luck_pm` | double |  | Luck a match (pts_vs_xpts less perf): results beyond the run of play. |
| `luck_se` | double |  | Its standard error. |
| `xgf_pm` | double |  | Goals the market expected it to score a match (all matches with rates). |
| `xga_pm` | double |  | Goals the market expected it to concede a match. |
| `fav_share` | double |  | Share of its matches it started as the likelier winner. |
| `move_24h_pm` | double |  | The market's mean move on the team over the last 24 hours, log odds: persistently positive means the early price undervalues it. |
| `opp_strength_pm` | double |  | Its opponents' mean strength going into its matches (those on its own scale at the time): the difficulty of its matches. |
| `strength` | double |  | Its market-implied strength in the table's last solve (gold/team_rating.strength_end: every side's from the same day, the one to rank on); blank for a side with no rated match in the 400 days before. |
| `ppm_last5` | double |  | Points a match over its last five matches. |
| `perf_last5` | double |  | In-play performance a match over its last five with a record. |
| `ppm_last10` | double |  | Points a match over its last ten. |
| `perf_last10` | double |  | In-play performance a match over its last ten with a record. |
| `pool_id` | int64 |  | The rating pool it is in: the smallest team id among the teams its matches of the last 400 days join it to. Strengths compare within a pool only. |
| `pool_size` | int32 |  | Teams in that pool. |
| `ranked` | bool | yes | It is in a ranking: ten or more matches, a rating, a match in the last 120 days of the data, and a pool of twenty or more teams. |
| `country_rank` | int32 |  | Its place by strength among the ranked sides of its country, category and pool (1 is the strongest). |
| `country_teams` | int32 |  | Sides in that ranking. |
| `pct_overall` | double |  | The share of the ranked sides of its category and pool it is stronger than. |

## `club/country_strength`

One row a country code and category: the strength of its sides on the ratings' one scale, and what its matches look like to the market.

Key: `country_code`, `category`. Stored as a single file, sorted by `country_code`, `category`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `country_code` | string | yes | Betfair's country code. |
| `category` | string | yes | senior, women, youth or reserves. |
| `teams` | int32 | yes | Ranked sides of the country in the category's largest pool (the one scale the strength columns are on). |
| `strength_top` | double |  | The strongest side's strength. |
| `strength_top5` | double |  | The mean of the five strongest; blank with fewer than five. |
| `strength_median` | double |  | The median side's. |
| `strength_mean` | double |  | The mean. |
| `matches` | int32 | yes | Matches with a result under the country code. |
| `total_mean` | double |  | Goals the market expected a match. |
| `home_edge_mean` | double |  | The home side's expected goals less the away side's, a match. |
| `p_home_mean` | double |  | The mean closing chance of a home win. |
| `home_rate` | double |  | The share of matches the home side won. |
| `draw_rate` | double |  | The share drawn. |
| `over25_rate` | double |  | The share with three or more goals. |

## `gold/link`

One row a Betfair event: the API-Football fixture it is, how the link was made, and our canonical match. Competition, season, line-ups, goal minutes and xG join to the prices through it. Events that found no fixture are kept, with method 'none'.

Key: `event_id`. Stored as a single file, sorted by `event_id`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `event_id` | int64 | yes | Betfair's event id. |
| `linked` | bool | yes | Whether a fixture was found. |
| `method` | string | yes | How: exact (both names equal after normalisation), alias (a reviewed alias), team_map (both Betfair team ids mapped to the fixture's teams), team_anchor (one side mapped, the team's only fixture at that kick-off), fuzzy (both names similar, unique), none. |
| `score` | double |  | How sure, 0 to 1: 1 for exact and alias; the weaker side's name similarity for fuzzy; the weaker mapped side's vote share for team_map and team_anchor. |
| `apif_fixture_id` | int64 |  | API-Football's fixture id. |
| `apif_league_id` | int32 |  | API-Football's league id: the competition. |
| `apif_league` | string |  | The league's name (where the fixtures table gives it). |
| `apif_league_country` | string |  | The league's country or region as API-Football names it. |
| `season` | int16 |  | API-Football's season: the year it starts. |
| `apif_home_team_id` | int64 |  | API-Football's team id of the side Betfair names first (the event's home side, whichever way round the fixture lists them). |
| `apif_away_team_id` | int64 |  | API-Football's team id of the side Betfair names second. |
| `apif_home_name` | string |  | API-Football's name for the event's home side. |
| `apif_away_name` | string |  | API-Football's name for the event's away side. |
| `apif_kickoff` | timestamp[ms, tz=UTC] |  | The fixture's kick-off. |
| `kickoff_diff_min` | double |  | Betfair's scheduled kick-off less the fixture's, minutes. |
| `swapped` | bool |  | True when the fixture lists the sides the other way round (its home side is Betfair's away side): flip API-Football's home/away tables before joining. |
| `apif_ft_home` | int16 |  | API-Football's goals after 90 minutes for the event's home side. |
| `apif_ft_away` | int16 |  | ... for the event's away side. |
| `result_agrees` | bool |  | Whether that score gives the result Betfair settled (blank when either is missing). False on an exact link marks a data error to review; other methods never keep one. |
| `over25_agrees` | bool |  | Whether that score's total (three goals or more) is the over/under 2.5 Betfair settled (blank when either is missing). As result_agrees: False only on exact links. |
| `match_id` | string |  | Our canonical match id (football.data.make_match_id) where the fixture is one of the 38 football-data competitions' matches. |
| `competition_id` | string |  | Our competition id (ENG1 ...): the canonical match's, else the one API-Football's league id maps to (a play-off football-data does not hold). |

## `gold/team_link`

One row a Betfair team id: the API-Football team it most often links to, by a vote over its linked matches, and our canonical team id where its matches are canonical ones. Teams with no linked match are kept, blank.

Key: `team_id`. Stored as a single file, sorted by `team_id`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `team_id` | int64 | yes | Betfair's selection id for the team (gold/team). |
| `team_name` | string |  | The name Betfair most often gives it. |
| `apif_team_id` | int64 |  | The API-Football team id most of its linked matches give. |
| `apif_team_name` | string |  | API-Football's name for that team. |
| `n_linked` | int32 | yes | Its matches linked to a fixture. |
| `n_agree` | int32 | yes | Of those, the matches that give apif_team_id. |
| `share` | double |  | n_agree / n_linked. |
| `n_apif_ids` | int32 | yes | Distinct API-Football team ids its matches gave (more than one: Betfair gives one id to two clubs of the same name, or a wrong link). |
| `low_agreement` | bool | yes | The share is under 0.8: the map is not used to link events and should not be used to join team tables. |
| `canonical_team_id` | string |  | Our team id ('ENG:arsenal') most of its canonical matches give. |
| `canonical_share` | double |  | The share of its canonical matches that give it. |

## `gold/goal_check`

One row a linked match that went in play and has API-Football's score or goal events: the goals inferred from prices (gold/goal) against them. Counts against the 90-minute score; minutes against the goal events, the half-time break allowed for (API-Football's minute is the match clock, an inferred goal's is the wall clock).

Key: `event_id`. Stored as a single file, sorted by `event_id`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `event_id` | int64 | yes | Betfair's event id. |
| `apif_fixture_id` | int64 | yes | API-Football's fixture id. |
| `clock` | string | yes | What the wall clock was measured from: 'actual' (the market's turn in play) or 'fixture' (the fixture's scheduled kick-off: a late start is allowed for). |
| `usable` | bool |  | gold/timeline's data rule for the match (enough price rows, no long gap, in play from the kick-off): judge the inference on these. Blank without gold/timeline. |
| `has_events` | bool | yes | Whether API-Football's events are held for the fixture. |
| `n_inferred` | int16 | yes | Goals inferred from prices. |
| `n_untimed` | int16 | yes | Of those, goals the prices cannot time (gold/goal.timed false: the settlement says they were scored, their moment is a placeholder): in the count and the score, left out of the pairing with goal events. |
| `n_apif` | int16 |  | API-Football's goal events (normal, penalty, own goal; not a missed penalty) within the 90 minutes; blank without events. |
| `n_ft` | int16 |  | Goals in API-Football's 90-minute score; blank without one. |
| `count_diff` | int16 | yes | n_inferred less the true count (n_ft, else n_apif). |
| `count_agrees` | bool | yes | count_diff is 0. |
| `score_agrees` | bool |  | Whether the inferred goals give the true score, side by side (the 90-minute score, else the goal events'). False when the count differs; blank when the count agrees and an inferred goal, or a goal event, names no side. |
| `n_matched` | int16 | yes | Inferred goals paired with a goal event (in order, within 16 minutes). |
| `n_within_tol` | int16 | yes | Of those, pairs whose football minute agrees within 2: gold/goal's match_min against the middle of the real goal's minute (the wall clock's error where a goal has no match_min). |
| `n_side_known` | int16 | yes | Pairs where both sources name the scoring side (API-Football lists an own goal under the side it counts for). |
| `n_side_agree` | int16 | yes | Of those, pairs that name the same side. |
| `minute_mae` | double |  | Mean absolute minute error of the pairs on the wall clock (in the second half only what lies outside a lag of 15 to 25 minutes). |
| `minute_err_max` | double |  | The largest absolute minute error. |
| `match_min_mae` | double |  | Mean absolute error of gold/goal's match_min over the pairs, against the middle of the real goal's minute (capped at 45 and 90 as match_min is). |
| `ht_lag_min` | double |  | Median of wall clock less match clock over the second-half pairs: the half-time break plus first-half stoppage this match shows. |
