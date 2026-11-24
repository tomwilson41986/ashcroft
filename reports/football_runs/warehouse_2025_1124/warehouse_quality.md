**Warehouse quality: PASS** (50 checks; hard failures: 0; soft: 0)

| Check | Result | Value |
|---|---|---|
| bronze partitions held | pass | 20 |
| silver built from the current bronze by the current parser | pass | all |
| files in = read + unreadable + superseded | pass | all |
| each silver partition's reconciliation | pass | all |
| traded prices within 1.01 to 1000 | pass | 0 |
| price updates outside Betfair's range, left out of silver/tick | pass | 3 |
| share of files that could not be read | pass | {'files_in': 305485, 'unreadable': 0, 'superseded': 0} |
| silver/market: market_id unique across partitions | pass | 305485 |
| silver/runner: every row's market exists | pass | all |
| silver/definition: every row's market exists | pass | all |
| silver/spell: every row's market exists | pass | all |
| markets whose kick-off month is their partition's | pass | {'share': 0.9966, 'kickoff_min': '1970-01-21 10:04:13+00:00', 'kickoff_max': '2050-06-12 10:00:00+00:00'} |
| gold built from its current inputs | pass | all |
| gold/event: key unique in every partition | pass | 101522 |
| gold/event: every partition built | pass | 20 |
| gold/checkpoint: key unique in every partition | pass | 16158376 |
| gold/checkpoint: every row's market exists | pass | all |
| gold/checkpoint: every partition built | pass | 20 |
| gold/implied: key unique in every partition | pass | 1111616 |
| gold/implied: every partition built | pass | 20 |
| gold/path: key unique in every partition | pass | 1860839 |
| gold/path: every row's market exists | pass | all |
| gold/path: every partition built | pass | 20 |
| gold/goal: key unique in every partition | pass | 250509 |
| gold/goal: every partition built | pass | 20 |
| gold/timeline: key unique in every partition | pass | 96692 |
| gold/timeline: every partition built | pass | 20 |
| gold/inplay_minute: key unique in every partition | pass | 11230228 |
| gold/inplay_minute: every partition built | pass | 20 |
| gold/inplay: key unique in every partition | pass | 96681 |
| gold/inplay: every partition built | pass | 20 |
| gold/move: key unique in every partition | pass | 10463663 |
| gold/move: every row's market exists | pass | all |
| gold/move: every partition built | pass | 20 |
| gold/team: key unique | pass | 12532 |
| gold/team_rating: key unique | pass | 203028 |
| gold/rating_gap: key unique | pass | 486414 |
| club/team_match: key unique | pass | 203028 |
| club/team_summary: key unique | pass | 12285 |
| club/country_strength: key unique | pass | 271 |
| gold/event: event_id unique across partitions | pass | 101522 |
| gold/event: one row for every event in silver/market | pass | 101522 |
| gold/event: matches that went in play and have a result | pass | 0.99871 |
| closing 1X2 chances against results (mean chance, rate) | pass | {'matches': 78830, 'home': (0.4373, 0.4345), 'draw': (0.2481, 0.2486), 'away': (0.3146, 0.3169)} |
| gold/checkpoint: no price published after its checkpoint | pass | all |
| gold/checkpoint: a complete book's chances sum to one | pass | all |
| gold/checkpoint: every row on its grid (the schedule before the off; the close at the off) | pass | all |
| gold/event: the closing 1X2 is the checkpoint table's close | pass | 98129 |
| gold/implied: totals positive, supremacy inside the total | pass | 872358 |
| gold/implied: 1X2 chances sum to one | pass | 1067576 |
