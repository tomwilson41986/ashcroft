"""The bookmaker prices for the day: the sources run in turn (each in its own try), the quotes kept as the day's
snapshot (and the raw pages for the record), and matched to the syndicate's horses by course, UK time and name.

`scrape(day)` reads every race of the day (the dry run and the 10:00 card); `scrape_races(day, wanted)` only the
races named (the alert loop, one race at a time). The snapshot is one row a horse a bookmaker a source
(professor/odds/<day>/snapshot_<HHMM>.csv), the raw pages gzipped beside it.
"""
from __future__ import annotations

import gzip
import logging
from datetime import date, datetime, timezone

import pandas as pd

from professor import store
from professor.candidates import Candidate
from professor.scrapers.base import Browser, Quote, norm_name, norm_venue

log = logging.getLogger(__name__)
try:
    from zoneinfo import ZoneInfo
    _LONDON = ZoneInfo("Europe/London")
except Exception:  # noqa: BLE001
    _LONDON = timezone.utc


def _alias(name: str, cfg: dict) -> str:
    return cfg.get("bookmaker_aliases", {}).get(name, name)


def frame(quotes: list[Quote], cfg: dict) -> pd.DataFrame:
    df = pd.DataFrame([q.row() for q in quotes])
    if df.empty:
        return pd.DataFrame(columns=["source", "bookmaker", "venue", "race_hhmm", "horse", "price", "bog", "ew_terms",
                                     "fetched_utc", "race_key", "_venue", "_horse"])
    df["bookmaker"] = df["bookmaker"].map(lambda b: _alias(str(b), cfg))
    df["_venue"] = df["venue"].map(norm_venue)
    df["_horse"] = df["horse"].map(norm_name)
    return df


def scrape(day: date | str, cfg: dict, sources: tuple[str, ...] = ("racingpost", "oddschecker", "marketform"),
           wanted: set[tuple[str, str]] | None = None, keep_raw: bool = True) -> tuple[pd.DataFrame, dict]:
    """Every source's quotes for the day (or for the races `wanted`: (normalised venue, HH:MM)), and a status a
    source. Nothing raised: a dead source is a line in the status."""
    quotes: list[Quote] = []
    status: dict = {}
    raw: dict[str, str] = {}
    with Browser() as b:
        if "racingpost" in sources:
            from professor.scrapers.racingpost import RacingPost
            rp = RacingPost(b)
            try:
                races = rp.day_races(day)
                if wanted is not None:
                    races = [r for r in races if (norm_venue(r["venue"]), r["hhmm"]) in wanted]
                races = [r for r in races if not r.get("abandoned") and not r.get("result")]
                n = 0
                for r in races:
                    try:
                        q = rp.race_quotes(r)
                        quotes.extend(q)
                        n += 1
                    except Exception as e:  # noqa: BLE001
                        log.warning("racingpost %s %s: %s", r["hhmm"], r["venue"], str(e)[:100])
                status["racingpost"] = f"{n} of {len(races)} races read, {sum(q.source == 'racingpost' for q in quotes)} quotes"
            except Exception as e:  # noqa: BLE001
                status["racingpost"] = f"failed: {str(e)[:120]}"
            raw.update({f"racingpost_{k}": v for k, v in rp.raw.items()})
        if "oddschecker" in sources:
            from professor.scrapers.oddschecker import Oddschecker
            oc = Oddschecker(b)
            try:
                by_venue: dict[str, list[str]] = {}
                for q in quotes:
                    by_venue.setdefault(q.venue, [])
                    if q.race_hhmm not in by_venue[q.venue]:
                        by_venue[q.venue].append(q.race_hhmm)
                n0 = len(quotes)
                for venue, times in list(by_venue.items())[:1]:        # a probe: one meeting a run until it is seen to pass
                    quotes.extend(oc.meeting_quotes(day, venue, times[:2]))
                status["oddschecker"] = ("the challenge stood; raw page kept" if oc.blocked
                                         else f"{len(quotes) - n0} quotes")
            except Exception as e:  # noqa: BLE001
                status["oddschecker"] = f"failed: {str(e)[:120]}"
            raw.update({f"oddschecker_{k}": v for k, v in oc.raw.items()})
    if "marketform" in sources:
        from professor.scrapers.marketform import MarketForm
        try:
            quotes.extend(MarketForm().day_quotes(day))
            status["marketform"] = "no key: a slot" if not __import__("os").getenv("MARKETFORM_API_KEY") else "not wired"
        except Exception as e:  # noqa: BLE001
            status["marketform"] = f"failed: {str(e)[:120]}"
    df = frame(quotes, cfg)
    if keep_raw:
        snapshot(day, df, raw, status)
    return df, status


def snapshot(day: date | str, df: pd.DataFrame, raw: dict[str, str], status: dict) -> str:
    when = datetime.now(timezone.utc).strftime("%H%M")
    k = f"professor/odds/{day}/snapshot_{when}.csv"
    out = store.out_dir(day)
    store.write_csv(k, df.drop(columns=[c for c in ("_venue", "_horse") if c in df.columns]), out / f"snapshot_{when}.csv")
    store.write_json(f"professor/odds/{day}/status_{when}.json", status, out / f"odds_status_{when}.json")
    (out / "raw").mkdir(exist_ok=True)
    for name, html in raw.items():
        if html:
            body = gzip.compress(html.encode())
            (out / "raw" / f"{name}_{when}.html.gz").write_bytes(body)
            store.put_bytes(f"professor/odds/{day}/raw/{name}_{when}.html.gz", body, "application/gzip")
    return k


def match(df: pd.DataFrame, cands: list[Candidate]) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    """Each candidate's quotes {bookmaker: price} and BOG flags, matched by course, UK time and horse name; a quote
    from more than one source for the same bookmaker keeps the later fetch."""
    quotes: dict[tuple, dict] = {}
    bog: dict[tuple, dict] = {}
    if df is None or df.empty:
        return quotes, bog
    df = df.sort_values("fetched_utc")
    by = {(v, t, h): g for (v, t, h), g in df.groupby(["_venue", "race_hhmm", "_horse"], sort=False)}
    for c in cands:
        key = (norm_venue(c.venue), c.off_uk, norm_name(c.runner))
        g = by.get(key)
        if g is None:
            # the time as the other side may write it (Betfair's venue with a suffix, a name with a country)
            alt = [k for k in by if k[0] == key[0] and k[1] == key[1] and k[2].startswith(key[2][:8])]
            g = by.get(alt[0]) if alt else None
        if g is None:
            continue
        q = {r.bookmaker: float(r.price) for r in g.itertuples(index=False) if r.price and r.price > 1}
        quotes[(c.market_id, c.selection_id)] = q
        bog[(c.market_id, c.selection_id)] = {r.bookmaker: r.bog for r in g.itertuples(index=False)
                                              if r.bog is not None and r.bog == r.bog}
    return quotes, bog


def non_runners(df: pd.DataFrame, cands: list[Candidate]) -> set[tuple]:
    """Our horses a source flags as non-runners (Racing Post's page state), by course, UK time and name."""
    if df is None or df.empty:
        return set()
    nr = df[df["bookmaker"] == "NON-RUNNER"]
    keys = set(zip(nr["_venue"], nr["race_hhmm"], nr["_horse"]))
    return {(c.market_id, c.selection_id) for c in cands
            if (norm_venue(c.venue), c.off_uk, norm_name(c.runner)) in keys}
