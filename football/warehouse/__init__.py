"""The racing² football data warehouse: Betfair's own price history, tick by tick, as analysis-ready tables.

One store, three layers (docs/football_data_platform.md):

- **bronze**: the raw Betfair Historic Data stream files exactly as served, one row a file with its bytes, hash and
  origin (``bronze.py``). Immutable; everything else can be rebuilt from it.
- **silver**: what the files say, typed and one fact a row (``parse.py``, ``silver.py``): ``market`` (one row a
  market), ``runner`` (a market's selections, with the result), ``definition`` (every change of a market's state),
  ``spell`` (the open / suspended / in-play spells between them) and ``tick`` (every traded price, with the market's
  state at that moment and the time to the off).
- **gold**: tables shaped for analysis and models (``gold*.py``): the match (event) dimension with its teams, the
  pre-off checkpoint grid, implied goal expectancies, inferred goals, market-implied team ratings, the move panel a
  closing-price model trains on, and the club analytics marts.

Parquet under ``sources/football/warehouse/`` in the capture bucket (or a local folder), partitioned by the year and
month of the file's own date, one file a partition, sorted by key; DuckDB reads it in place (``catalog.py``).
Every build is idempotent: a partition is rebuilt only when its input or the code's version changed
(``_manifest.json`` beside each table).

    python -m football.warehouse ingest --zip 2026.zip --source owner_drive_2026_11_11
    python -m football.warehouse build            # silver, then gold, for the partitions that changed
    python -m football.warehouse check            # the data-quality report (non-zero exit on a failed check)
"""

ROOT = "football/warehouse"
