"""The greyhound model (the owner's ask, 5 Oct 2026): GBGB results -> lag-safe metrics -> a market-blind LightGBM
win model, scored against the Betfair SP. See reports/greyhound_model.md.

- ``greyhound.data``: the GBGB run table (sources/gbgb/runs_<year>.parquet, local or S3) cleaned for modelling;
- ``greyhound.metrics``: the metrics engine, every feature from runs before the race only;
- ``greyhound.model``: walk-forward fitting, race-normalised probabilities, scoring against the SP and the BSP.
"""
