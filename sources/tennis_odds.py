"""tennis-data.co.uk's results with closing odds, the half of ``sources.tennis`` its Cloudflare refuses GitHub's
runners: run from the UK server (betfair-prices.yml), built into ``tennis/odds_matches.parquet`` there (and again by each
``source_data.py --source tennis --build``)."""

from sources.tennis import build_odds as build, fetch_odds as fetch  # noqa: F401

__all__ = ["build", "fetch"]
