"""What is Australian about the Australian model: the jurisdiction's settings, kept apart from the UK/IE code.

Only what the data in hand shows is filled in; each later stage adds its part (au/README.md).
"""

from __future__ import annotations

import re

#: the Betfair price-file market and the country in its meeting names ('WAGGA (AUS) 28TH FEB')
PRICE_FILE_MARKET = "auswin"
COUNTRY = "AUS"

#: the S3 store keys (under s3://$CAPTURE_BUCKET/sources/, sources.common.Store)
TARGETS_KEY = "au/bsp_targets.parquet"
TARGETS_REPORT_KEY = "au/bsp_targets_report.json"
TRACK_NAMES_KEY = "au/track_names.json"         # Betfair's short meeting names to full names (au/data.py)

#: Australian race times in UK hours (reports/international_racing.md s.4.1, from the ANZ files): about 00:20-10:30
#: UK, so every Betfair job for Australia needs a runner free overnight (au/README.md, stage 0)
UK_HOURS = ("00:00", "11:00")

#: Betfair's harness markets sit in the same files as the thoroughbreds; their names carry the gait
HARNESS_RE = re.compile(r"\b(?:Pace|Trot)\b", re.I)


def race_family(race_type) -> str:
    """The race-type code in Betfair's market name ('R3 1200m Hcap' -> 'Hcap') as a family: group, listed, class,
    maiden, handicap, age, other. Descriptive only: no ordering is claimed until Punting Form's class and benchmark
    fields (stage 1) give one."""
    t = str(race_type or "").strip().lower()
    if re.fullmatch(r"grp\s*[123]|g[123]", t):
        return "group"
    if t.startswith("listed") or t == "lr":
        return "listed"
    if re.fullmatch(r"cl\s*\d+|class\s*\d+", t):
        return "class"
    if t.startswith("mdn") or t.startswith("maiden"):
        return "maiden"
    if t.startswith("hcap") or re.fullmatch(r"bm\s*\d+", t):
        return "handicap"
    if re.fullmatch(r"[23]yo", t):
        return "age"
    return "other"
