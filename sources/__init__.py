"""Historic data for the prospective models beyond UK/IE horse racing (the owner's ask, 3 Oct 2026).

One module a source, each with the same two halves:

- ``fetch``: the source's raw files, as published, into s3://$CAPTURE_BUCKET/sources/<source>/raw/..., one file at a
  time, resumable (what the store already holds is not fetched again, except the recent days or the current season,
  which the source may still revise), within a time budget;
- ``build``: the raw files read into one tidy table a year (or season), s3://$CAPTURE_BUCKET/sources/<source>/<table>_
  <year>.parquet, which the research reads.

| Module | Source | For the model of | Needs |
|---|---|---|---|
| ``gbgb`` | api.gbgb.org.uk (the Greyhound Board of Great Britain's results) | UK greyhounds | nothing |
| ``football`` | football-data.co.uk (results, match stats, opening and closing odds incl. Pinnacle and Betfair) | football | nothing |
| ``tennis`` | Jeff Sackmann's tennis_atp / tennis_wta (match stats); tennis-data.co.uk (results with closing odds) | tennis | nothing |
| ``puntingform`` | api.puntingform.com.au (Australian form and results) | Australian racing | ``PUNTINGFORM_API_KEY`` |

Betfair's own historic prices for these sports are ``betfair_historic.py`` (Betfair's Historic Data service), and its
live books ``betfair_recorder.py``: both need a Betfair login, so they run on the UK server.

``python source_data.py --source gbgb --fetch --build --max-minutes 300``
"""

#: the quick sources first: a shared --max-minutes then leaves the rest to the greyhound backfill
SOURCES = ("football", "tennis", "puntingform", "gbgb")
