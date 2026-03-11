"""Match HRB horse names to Betfair selection IDs.

HRB and Betfair use different naming conventions (country suffixes,
punctuation, spacing).  This module normalises both sides and matches
by race grouping (track + time) using exact match first, then fuzzy
matching for any remaining unmatched runners.
"""

import logging
import re

import pandas as pd
from thefuzz import fuzz

log = logging.getLogger(__name__)

# Known track name mappings: HRB name -> Betfair venue name
TRACK_ALIASES = {
    "Kempton": "Kempton",
    "Kempton (AW)": "Kempton",
    "Lingfield": "Lingfield",
    "Lingfield (AW)": "Lingfield",
    "Newcastle": "Newcastle",
    "Newcastle (AW)": "Newcastle",
    "Wolverhampton": "Wolverhampton",
    "Wolverhampton (AW)": "Wolverhampton",
    "Chelmsford": "Chelmsford City",
    "Chelmsford City": "Chelmsford City",
    "Dundalk": "Dundalk",
    "Dundalk (AW)": "Dundalk",
}

FUZZY_THRESHOLD = 90


def normalise_name(name: str) -> str:
    """Normalise a horse name for matching.

    * Lowercase
    * Strip country suffixes like ``(IRE)``, ``(GB)``, ``(FR)``, ``(USA)``
    * Remove apostrophes and non-alphanumeric characters (except spaces)
    * Collapse multiple spaces
    """
    s = name.lower().strip()
    s = re.sub(r"\s*\((ire|gb|fr|usa|aus|ger|ity|jpn|can|brz)\)\s*", "", s)
    s = re.sub(r"['\u2019]", "", s)  # apostrophes
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalise_track(track: str) -> str:
    """Normalise a track name for grouping."""
    clean = track.strip()
    return TRACK_ALIASES.get(clean, clean).lower()


def normalise_time(time_str: str) -> str:
    """Normalise race time to ``HH:MM`` format."""
    s = str(time_str).strip()
    # Handle "14.30" -> "14:30"
    s = s.replace(".", ":")
    # Ensure HH:MM
    parts = s.split(":")
    if len(parts) == 2:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
    return s


class HorseNameMatcher:
    """Match HRB declared runners to Betfair market selections."""

    def match_runners(
        self,
        hrb_runners: pd.DataFrame,
        bf_catalogues: list[dict],
    ) -> pd.DataFrame:
        """Match HRB runners to Betfair selection IDs.

        Args:
            hrb_runners: DataFrame with columns ``horse_name``,
                ``track``, ``race_time``.
            bf_catalogues: List of catalogue dicts from
                :meth:`BetfairClient.list_market_catalogue`.

        Returns:
            Copy of *hrb_runners* with added columns:
            ``selection_id``, ``market_id``, ``bf_runner_name``.
            Unmatched runners have ``None`` values.
        """
        result = hrb_runners.copy()
        result["selection_id"] = None
        result["market_id"] = None
        result["bf_runner_name"] = None

        # Build a lookup: (normalised_track, HH:MM) -> catalogue
        bf_lookup = {}
        for cat in bf_catalogues:
            venue = normalise_track(cat["venue"])
            start = cat["market_start_time"]
            if hasattr(start, "strftime"):
                race_time = start.strftime("%H:%M")
            else:
                race_time = normalise_time(str(start))
            key = (venue, race_time)
            bf_lookup[key] = cat

        matched = 0
        fuzzy_matched = 0
        unmatched_runners = []

        for idx, row in result.iterrows():
            hrb_track = normalise_track(str(row.get("track", "")))
            hrb_time = normalise_time(str(row.get("race_time", "")))
            hrb_name = normalise_name(str(row.get("horse_name", "")))

            # Find the matching Betfair market
            cat = bf_lookup.get((hrb_track, hrb_time))
            if cat is None:
                unmatched_runners.append(
                    f"  No market: {row.get('horse_name')} "
                    f"({row.get('track')} {row.get('race_time')})"
                )
                continue

            # Try exact match first
            found = False
            for runner in cat["runners"]:
                bf_norm = normalise_name(runner["runner_name"])
                if hrb_name == bf_norm:
                    result.at[idx, "selection_id"] = runner["selection_id"]
                    result.at[idx, "market_id"] = cat["market_id"]
                    result.at[idx, "bf_runner_name"] = runner["runner_name"]
                    matched += 1
                    found = True
                    break

            if found:
                continue

            # Fuzzy match
            best_score = 0
            best_runner = None
            for runner in cat["runners"]:
                bf_norm = normalise_name(runner["runner_name"])
                score = fuzz.ratio(hrb_name, bf_norm)
                if score > best_score:
                    best_score = score
                    best_runner = runner

            if best_runner and best_score >= FUZZY_THRESHOLD:
                result.at[idx, "selection_id"] = best_runner["selection_id"]
                result.at[idx, "market_id"] = cat["market_id"]
                result.at[idx, "bf_runner_name"] = best_runner["runner_name"]
                fuzzy_matched += 1
            else:
                unmatched_runners.append(
                    f"  No match: {row.get('horse_name')} "
                    f"(best: {best_runner['runner_name'] if best_runner else '?'}, "
                    f"score={best_score})"
                )

        total = len(result)
        unmatched_count = total - matched - fuzzy_matched
        log.info(
            f"Matched {matched + fuzzy_matched}/{total} runners "
            f"(exact={matched}, fuzzy={fuzzy_matched}, "
            f"unmatched={unmatched_count})"
        )

        if unmatched_runners:
            log.warning(
                f"Unmatched runners ({len(unmatched_runners)}):\n"
                + "\n".join(unmatched_runners[:20])
            )

        return result
