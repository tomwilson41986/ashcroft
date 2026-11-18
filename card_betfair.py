"""The day's card from Betfair's catalogue: the fallback when horseracebase.com gives no card (the owner, 18 Nov 2026:
"build the fallback").

On 15 and 17 Nov 2026 the night-before run failed at horseracebase.com's bot check ('One moment, please...', no CSRF
token, the login refused four times). The morning run logs in the same way, so a blocked morning would leave the
trader with no prices. GitHub's runners cannot reach Betfair either (the login answers 403), so the card is read from
what the UK server already records: tomorrow's GB/IE win markets with their runner metadata, written through the night
by betfair-prices.yml's overnight record to s3://<bucket>/betfair_live/<day>/markets_overnight.csv.gz (the day's own
records, markets.csv.gz, where they exist), and the runners Betfair has removed from the books beside it.

The card is written in the HRB card's form (daily_predictions.scrape_racecard_html), so model/card_enrich.py and
everything after it read it unchanged:

    race_date, race_time ('1:50', the card's 12-hour form), track (the results table's name), race_name (the
    market name with Betfair's abbreviations written out: '2m4f Hcap Chs' -> '2m4f Handicap Chase'), race_distance
    ('2m4f'), stall (the draw, else the saddlecloth, as the HRB card), horse_name, horse_age, pounds, jockey_name and
    trainer (in the results table's 'Surname, Given' form, matched on history), official_rating, headgear (the HRB
    codes), days_since_lr, odds (Betfair's forecast price), number_of_runners

What Betfair does not publish stays empty and card_enrich fills it as it does for any card gap: race class, prize
money, the pedigree of a horse with no history. The going is not published either: it is taken from the latest
results at the same track for the same code (flat or jumps) within GOING_DAYS days, and an all-weather flat race
reads 'Standard'; with nothing recent it is left empty. Every card row says where it came from (card_source).
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

UK = ZoneInfo("Europe/London")
GOING_DAYS = 14
HISTORY_YEARS = 3
CARD_FILES = ("markets_overnight.csv.gz", "markets.csv.gz", "markets_evening.csv.gz")
BOOK_FILES = ("books_overnight.csv.gz", "books.csv.gz")

# Betfair's market-name abbreviations, written out so the race type reads as it does from HRB's race names
# (card_enrich.race_signature)
_ABBREV = [
    (r"\bHcap\b", "Handicap"), (r"\bHrd\b", "Hurdle"), (r"\bChs\b", "Chase"), (r"\bNov\b", "Novices"),
    (r"\bMdn\b", "Maiden"), (r"\bI?NHF\b", "NH Flat"), (r"\bBeg\b", "Beginners"), (r"\bHunt\b", "Hunters"),
    (r"\bClm\b|\bClaim\b", "Claiming"), (r"\bSell\b|\bSlr\b", "Selling"), (r"\bStks\b", "Stakes"),
    (r"\bGrp\b", "Group"), (r"\bList\b", "Listed"), (r"\bJuv\b", "Juvenile"), (r"\bMares\b", "Mares"),
]
_DIST = re.compile(r"^\s*(\d+m(?:\d+(?:\.\d+)?f)?|\d+(?:\.\d+)?f)\b", re.I)
_TITLES = {"mr", "mrs", "miss", "ms", "capt", "captain", "dr", "sir", "lady", "lord", "major", "col", "lt"}
# Betfair's WEARING words -> the HRB card's headgear codes (model/blocks/gear.py reads them lower-cased)
_GEAR = [("blink", "Blnk"), ("cheek", "CkPc"), ("visor", "Vsor"), ("hood", "Hood"), ("eye", "Eye"),
         ("tongue", "TT")]
_AW_VENUES = ("southwell", "wolverhampton", "kempton", "lingfield", "newcastle", "chelmsford", "dundalk")


def _alnum(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s if s is not None else "").lower())


def expand_race_name(name) -> str:
    """'2m4f Hcap Chs' -> '2m4f Handicap Chase'."""
    out = str(name or "")
    for rx, word in _ABBREV:
        out = re.sub(rx, word, out, flags=re.I)
    return out


def distance_from_name(name) -> str | None:
    """The trip at the front of Betfair's market name: '2m4f Hcap Chs' -> '2m4f', '6f Nov Stks' -> '6f'."""
    m = _DIST.match(str(name or ""))
    return m.group(1) if m else None


def headgear_from_wearing(w) -> str | None:
    """Betfair's WEARING text as the HRB card's codes, space separated: 'cheekpieces, tongue strap' -> 'CkPc TT'."""
    t = str(w or "").lower()
    if not t or t in ("nan", "none"):
        return None
    codes = [code for word, code in _GEAR if word in t]
    return " ".join(codes) if codes else None


def card_time(start_utc) -> str | None:
    """Betfair's start time (UTC) as the HRB card writes it: UK time, 12-hour, no leading zero ('13:50' -> '1:50')."""
    try:
        t = pd.Timestamp(start_utc)
    except (ValueError, TypeError):
        return None
    if pd.isna(t):
        return None
    t = (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).tz_convert(UK)
    h = t.hour % 12 or 12
    return f"{h}:{t.minute:02d}"


def forecast_odds(num, den) -> float:
    try:
        n, d = float(num), float(den)
    except (TypeError, ValueError):
        return np.nan
    return n / d + 1.0 if d > 0 else np.nan


# --------------------------------------------------------------------------------------------------------------------
# Names: Betfair writes 'Harry Skelton', the results table 'Skelton, Harry' (or 'Skelton, H')
# --------------------------------------------------------------------------------------------------------------------

def _hist_keys(name) -> tuple:
    """A results-table name as its parts' (surname, first initial): 'Russell, Lucinda / Scudamore, M' ->
    (('russell', 'l'), ('scudamore', 'm'))."""
    keys = []
    for part in str(name or "").split("/"):
        part = re.sub(r"\(\d{1,2}\)\s*$", "", part).strip()
        if "," in part:
            sur, given = part.split(",", 1)
        else:
            bits = part.split()
            sur, given = (bits[-1], " ".join(bits[:-1])) if bits else ("", "")
        given = [g for g in re.split(r"[\s.]+", given) if g and _alnum(g) not in _TITLES]
        keys.append((_alnum(sur), _alnum(given[0])[:1] if given else ""))
    return tuple(sorted(keys))


def _betfair_part_keys(part) -> list[tuple]:
    """Every (surname, first initial) a Betfair name could be: 'Nico de Boinville' -> ('boinville', 'n'),
    ('deboinville', 'n'); 'Mr O. McGill' -> ('mcgill', 'o')."""
    toks = [t for t in re.split(r"[\s]+", str(part).strip()) if t and _alnum(t) not in _TITLES]
    if not toks:
        return []
    if len(toks) == 1:
        return [(_alnum(toks[0]), "")]
    out = []
    for k in range(1, len(toks)):
        out.append((_alnum("".join(toks[k:])), _alnum(toks[0])[:1]))
    return out


class NameIndex:
    """History's names by their parts' (surname, initial): the most recent wins a tie."""

    def __init__(self, names: pd.DataFrame):
        """names: columns name, last (the latest race date), n (rides or runners)."""
        self.by_keys: dict[tuple, tuple] = {}
        self.by_surname: dict[str, tuple] = {}
        for r in names.dropna(subset=["name"]).itertuples(index=False):
            keys = _hist_keys(r.name)
            score = (str(r.last), int(r.n))
            if keys and (keys not in self.by_keys or score > self.by_keys[keys][0]):
                self.by_keys[keys] = (score, r.name)
            if len(keys) == 1:
                sur = keys[0][0]
                if sur not in self.by_surname or score > self.by_surname[sur][0]:
                    self.by_surname[sur] = (score, r.name)

    def match(self, betfair_name) -> str | None:
        if not isinstance(betfair_name, str) or not betfair_name.strip():
            return None
        parts = [p for p in re.split(r"\s+&\s+|\s+and\s+", betfair_name.strip()) if p.strip()]
        options = [_betfair_part_keys(p) for p in parts]
        if not all(options):
            return None
        best = None
        from itertools import product
        for combo in product(*options):
            hit = self.by_keys.get(tuple(sorted(combo)))
            if hit and (best is None or hit[0] > best[0]):
                best = hit
        return best[1] if best else None


# --------------------------------------------------------------------------------------------------------------------
# History reads (the results table): names, tracks and the latest going
# --------------------------------------------------------------------------------------------------------------------

def history_lookups(db_path: str, day: date) -> dict:
    """The names, track names and recent going the card is written in, from race_results before ``day``."""
    since = (day - timedelta(days=365 * HISTORY_YEARS)).isoformat()
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        jockeys = pd.read_sql_query(
            "SELECT jockey_name AS name, MAX(race_date) AS last, COUNT(*) AS n FROM race_results "
            "WHERE race_date >= ? AND race_date < ? GROUP BY jockey_name", conn, params=(since, day.isoformat()))
        trainers = pd.read_sql_query(
            "SELECT trainer AS name, MAX(race_date) AS last, COUNT(*) AS n FROM race_results "
            "WHERE race_date >= ? AND race_date < ? GROUP BY trainer", conn, params=(since, day.isoformat()))
        tracks = pd.read_sql_query(
            "SELECT track, MAX(race_date) AS last FROM race_results WHERE race_date >= ? GROUP BY track", conn,
            params=(since,))
        going = pd.read_sql_query(
            "SELECT track, race_type, race_date, race_time, going_description FROM race_results "
            "WHERE race_date >= ? AND race_date < ? AND going_description IS NOT NULL AND going_description != ''",
            conn, params=((day - timedelta(days=60)).isoformat(), day.isoformat()))
    return {"jockeys": NameIndex(jockeys), "trainers": NameIndex(trainers), "tracks": tracks, "going": going}


def track_name(venue, tracks: pd.DataFrame) -> str:
    """Betfair's venue as the results table names it: 'Bangor-on-Dee' -> 'Bangor', 'Newmarket' -> the Newmarket
    course raced most recently. The venue itself when the table has nothing like it."""
    v = _alnum(venue)
    if not v or tracks is None or tracks.empty:
        return venue
    t = tracks.assign(k=tracks.track.map(_alnum))
    exact = t[t.k == v]
    if not exact.empty:
        return exact.sort_values("last").track.iloc[-1]
    near = t[t.k.map(lambda k: bool(k) and (k.startswith(v) or v.startswith(k)))]
    return near.sort_values("last").track.iloc[-1] if not near.empty else venue


def _is_jumps_type(race_type) -> bool:
    return bool(re.search(r"hurdle|chase|nh flat|hunter|bumper", str(race_type or ""), re.I))


def going_estimate(track: str, jumps: bool, day: date, going: pd.DataFrame) -> tuple[str | None, str]:
    """The going for a race: the latest recorded at the track for the same code within GOING_DAYS days, and where
    it came from. An all-weather flat track reads 'Standard' (its most recent flat going was all-weather)."""
    if going is None or going.empty:
        return None, "none"
    g = going[going.track.map(_alnum) == _alnum(track)]
    g = g[g.race_type.map(_is_jumps_type) == jumps]
    if g.empty:
        return None, "none"
    g = g.sort_values(["race_date", "race_time"])
    last = g.iloc[-1]
    aw = bool(re.search(r"standard|slow", str(last.going_description), re.I))
    if not jumps and aw:
        return "Standard", f"all-weather ({last.race_date})"
    if (day - date.fromisoformat(str(last.race_date)[:10])).days <= GOING_DAYS:
        return str(last.going_description), f"latest at the track ({last.race_date})"
    return None, "none recent"


# --------------------------------------------------------------------------------------------------------------------
# The card
# --------------------------------------------------------------------------------------------------------------------

def card_from_catalogue(cat: pd.DataFrame, day: date, lookups: dict | None = None,
                        removed: set | None = None) -> pd.DataFrame:
    """The recorder's catalogue rows (betfair_recorder.MARKET_FIELDS) as an HRB-form card for ``day``: UK/IE win
    markets that start on the day (UK), one row a runner, the removed ones left out."""
    if cat is None or cat.empty:
        return pd.DataFrame()
    c = cat.copy()
    c = c[(c.get("market_type", "WIN").fillna("WIN") == "WIN") & c.country.isin(["GB", "IE"])]
    start = pd.to_datetime(c.market_start_utc, utc=True, errors="coerce")
    c = c[start.dt.tz_convert(UK).dt.date == day]
    if "recorded_utc" in c.columns:
        c = c.sort_values("recorded_utc")
    c = c.drop_duplicates(["market_id", "selection_id"], keep="last")
    if removed:
        c = c[~c.apply(lambda r: (str(r.market_id), str(int(float(r.selection_id)))) in removed, axis=1)]
    if c.empty:
        return pd.DataFrame()
    lk = lookups or {}
    tracks = lk.get("tracks")
    venue_map = {v: track_name(v, tracks) for v in c.venue.dropna().unique()}
    jk, tr = lk.get("jockeys"), lk.get("trainers")

    def named(index, s):
        if index is None:
            return s
        m = {v: index.match(v) for v in s.dropna().unique()}
        return s.map(lambda v: m.get(v) or v)

    stall = pd.to_numeric(c.get("stall_draw"), errors="coerce")
    cloth = pd.to_numeric(c.get("cloth_number"), errors="coerce")
    card = pd.DataFrame({
        "race_date": day.isoformat(),
        "race_time": c.market_start_utc.map(card_time),
        "track": c.venue.map(venue_map),
        "race_name": c.market_name.map(expand_race_name),
        "race_distance": c.market_name.map(distance_from_name),
        "stall": stall.fillna(cloth),
        "horse_name": c.runner_name.astype(str).str.strip(),
        "horse_age": pd.to_numeric(c.get("age"), errors="coerce"),
        "pounds": pd.to_numeric(c.get("weight_value"), errors="coerce"),
        "jockey_name": named(jk, c.get("jockey")),
        "trainer": named(tr, c.get("trainer")),
        "official_rating": pd.to_numeric(c.get("official_rating"), errors="coerce"),
        "headgear": c.get("wearing").map(headgear_from_wearing) if "wearing" in c.columns else None,
        "days_since_lr": pd.to_numeric(c.get("days_since_last_run"), errors="coerce"),
        "odds": [forecast_odds(n, d) for n, d in zip(c.get("forecast_num"), c.get("forecast_den"))],
        "market_id": c.market_id.astype(str),
        "selection_id": pd.to_numeric(c.selection_id, errors="coerce").astype("Int64"),
        "betfair_race_type": c.get("race_type"),
        "card_source": "betfair",
    })
    card["number_of_runners"] = card.groupby("market_id").horse_name.transform("size")
    going = lk.get("going")
    est = {}
    for (track, mkt), g in card.groupby(["track", "market_id"]):
        rt = g.betfair_race_type.iloc[0]
        jumps = str(rt or "").lower() not in ("flat", "")
        est[mkt] = going_estimate(track, jumps, day, going)
    card["going_description"] = card.market_id.map(lambda m: est[m][0])
    card["going_source"] = card.market_id.map(lambda m: est[m][1])
    return card.reset_index(drop=True)


def _s3_csv(s3, bucket: str, key: str) -> pd.DataFrame | None:
    try:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001 - a missing file is a normal answer here
        return None
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), dtype={"market_id": str, "selection_id": str}, low_memory=False)


def removed_runners(books: pd.DataFrame | None) -> set:
    """(market_id, selection_id) of every runner whose latest recorded book says REMOVED."""
    if books is None or books.empty or "runner_status" not in books.columns:
        return set()
    b = books.sort_values("polled_utc").drop_duplicates(["market_id", "selection_id"], keep="last")
    b = b[b.runner_status.astype(str).str.upper() == "REMOVED"]
    return {(str(m), str(int(float(s)))) for m, s in zip(b.market_id, b.selection_id) if str(s) not in ("", "nan")}


def fetch_card(day: date, db_path: str | None = None, s3=None, bucket: str | None = None) -> pd.DataFrame:
    """The day's card from the UK server's recorded catalogue in S3, or an empty frame (and the reason logged)."""
    if s3 is None:
        import boto3
        s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION") or "eu-west-2")
    bucket = bucket or os.environ.get("CAPTURE_BUCKET") or os.environ.get("ULTRA_BETTING_S3_BUCKET") or "ashcroft"
    cats = [df for f in CARD_FILES if (df := _s3_csv(s3, bucket, f"betfair_live/{day:%Y-%m-%d}/{f}")) is not None]
    if not cats:
        log.error(f"Betfair card: no recorded catalogue for {day} under s3://{bucket}/betfair_live/{day:%Y-%m-%d}/")
        return pd.DataFrame()
    cat = pd.concat(cats, ignore_index=True)
    removed = set()
    for f in BOOK_FILES:
        removed |= removed_runners(_s3_csv(s3, bucket, f"betfair_live/{day:%Y-%m-%d}/{f}"))
    lookups = {}
    if db_path and os.path.exists(db_path):
        try:
            lookups = history_lookups(db_path, day)
        except Exception as e:  # noqa: BLE001 - names stay Betfair's, the going empty: a poorer card, not none
            log.warning(f"Betfair card: history lookups failed ({e}); names and going left as Betfair gives them")
    card = card_from_catalogue(cat, day, lookups, removed)
    if card.empty:
        log.error(f"Betfair card: the recorded catalogue holds no GB/IE win market on {day}")
        return card
    jk = lookups.get("jockeys")
    log.warning(
        f"Betfair card for {day}: {len(card)} runners in {card.market_id.nunique()} races "
        f"({len(removed)} removed runners left out); going estimated on "
        f"{card.going_description.notna().mean():.0%} of runners"
        + (f"; jockeys matched to history {card.jockey_name.str.contains(',', na=False).mean():.0%}" if jk else ""))
    return card
