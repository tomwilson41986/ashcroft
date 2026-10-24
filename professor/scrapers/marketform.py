"""Market Form (DC Network, dc-network.co.uk), the third source the owner named (23 Oct): a membership product whose
live market and odds data sit behind a login and a data API ("DC Data Services"). Nothing is fetched until the owner
holds a key (`MARKETFORM_API_KEY`) and the API's shape is known; this is the slot, returning no quotes and a status
line, so the day runs on the other sources meanwhile."""
from __future__ import annotations

import logging
import os
from datetime import date

from professor.scrapers.base import Quote

log = logging.getLogger(__name__)
NAME = "marketform"


class MarketForm:
    name = NAME

    def __init__(self, browser=None):
        self.key = os.getenv("MARKETFORM_API_KEY", "")
        self.raw: dict[str, str] = {}

    def day_quotes(self, day: date | str) -> list[Quote]:
        if not self.key:
            log.info("marketform: no MARKETFORM_API_KEY; the source is a slot until the owner holds one")
        else:
            log.info("marketform: a key is set but the API is not wired yet")
        return []
