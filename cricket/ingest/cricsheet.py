"""Cricsheet ingest (build spec 5.1): the raw zips untouched, then six tables, then the ingest report.

    python -m cricket.ingest.cricsheet --fetch --build --report            # S3 (sources/cricket/...)
    python -m cricket.ingest.cricsheet --root data/cricket/store --zip data/cricket/raw/all_json.zip --build --report

Raw files land at ``cricket/raw/cricsheet/<yyyy-mm-dd>/`` in the store (the download day, UTC) and are never
rewritten. The tables are a pure function of the zip: ``matches``, ``innings``, ``deliveries`` (one row per delivery),
``players``, ``venues``, ``people_register``. They go to ``cricket/tables/<name>.parquet`` and, with ``--db``, are
upserted on their keys (cricket/db/loaders.py), so a rerun of the same zip adds zero rows.

Cricsheet is ODC-By: the licence travels with the raw zip. What Cricsheet does not have is documented, not filled:
no ball tracking, no delivery timestamps, no fielding positions, no weather, no live feed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import logging
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from sources.common import Store, get, session

log = logging.getLogger(__name__)

BASE = "https://cricsheet.org"
RAW_FILES = {
    "all_json.zip": f"{BASE}/downloads/all_json.zip",
    "all_csv2.zip": f"{BASE}/downloads/all_csv2.zip",   # the "Ashwin" CSV format, kept raw only
    "people.csv": f"{BASE}/register/people.csv",
    "names.csv": f"{BASE}/register/names.csv",
}
DEFAULT_FILES = ["all_json.zip", "people.csv", "names.csv"]   # the CSV zip only with --with-csv
RAW_PREFIX = "cricket/raw/cricsheet"
TABLE_PREFIX = "cricket/tables"
# a wicket that ends a batter's innings and counts against the team; retired hurt / not out do not
NOT_TEAM_WICKETS = {"retired hurt", "retired not out"}


# ---------------------------------------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------------------------------------

def looks_valid(name: str, data: bytes) -> bool:
    """A zip starts with its local-file signature; a register CSV starts with its header. Cricsheet now and then
    answers a file's URL with an HTML page (seen 4 Nov 2026 for both register CSVs): that is never stored as raw."""
    if name.endswith(".zip"):
        return data[:4] == b"PK\x03\x04"
    if name.endswith(".csv"):
        return data.lstrip()[:11] == b"identifier,"
    return True


def fetch(store: Store, day: str | None = None, names: list[str] | None = None, tries: int = 3,
          pause: float = 10.0) -> dict[str, str]:
    """Download each raw file to ``cricket/raw/cricsheet/<day>/``, untouched. A file identical (sha256) to the last
    copy held is not stored again: ``latest.json`` names the day each file was last stored and its hash."""
    day = day or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    latest = json.loads(store.get(f"{RAW_PREFIX}/latest.json") or b"{}")
    have = store.listing(f"{RAW_PREFIX}/{day}/")
    s = session()
    out = {}
    for name in names or DEFAULT_FILES:
        key = f"{RAW_PREFIX}/{day}/{name}"
        if key in have:
            out[name] = "kept"
            continue
        r = None
        for attempt in range(tries):
            r = get(s, RAW_FILES[name], timeout=600)
            if r is None or looks_valid(name, r.content):
                break
            log.warning("%s: not a valid %s (attempt %d); again in %.0f s", name, name.rsplit(".", 1)[-1],
                        attempt + 1, pause)
            time.sleep(pause)
        if r is None:
            out[name] = "no data found (404)"
            continue
        if not looks_valid(name, r.content):
            out[name] = f"not stored: the reply was not a valid file after {tries} tries"
            continue
        digest = hashlib.sha256(r.content).hexdigest()
        if latest.get(name, {}).get("sha256") == digest:
            out[name] = f"unchanged since {latest[name]['day']}"
            continue
        store.put(key, r.content)
        latest[name] = {"day": day, "sha256": digest, "bytes": len(r.content)}
        out[name] = f"{len(r.content)} bytes"
    store.put(f"{RAW_PREFIX}/latest.json", json.dumps(latest, indent=1).encode())
    return out


def latest_raw_day(store: Store) -> str | None:
    """The day of the newest all_json.zip held (with its people.csv from the same or an earlier day)."""
    latest = json.loads(store.get(f"{RAW_PREFIX}/latest.json") or b"{}")
    return latest.get("all_json.zip", {}).get("day")


def _latest(store: Store, name: str, day: str) -> bytes | None:
    latest = json.loads(store.get(f"{RAW_PREFIX}/latest.json") or b"{}")
    d = latest.get(name, {}).get("day", day)
    return store.get(f"{RAW_PREFIX}/{d}/{name}")


# ---------------------------------------------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------------------------------------------

def _first(x):
    return x[0] if isinstance(x, list) and x else None


def _text(x):
    return None if x is None else str(x)


def parse_match(match_id: str, d: dict) -> dict[str, list[dict]]:
    """One Cricsheet JSON match to rows of matches, innings, deliveries, players. Pure: no I/O."""
    info, meta = d["info"], d.get("meta", {})
    reg = (info.get("registry") or {}).get("people", {})
    venue_reg = (info.get("registry") or {}).get("venues", {})
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    event = info.get("event", {}) or {}
    teams = info.get("teams", [None, None])
    toss = info.get("toss") or {}
    dates = info.get("dates", [])
    match = {
        "match_id": match_id,
        "data_version": meta.get("data_version"), "created": meta.get("created"), "revision": meta.get("revision"),
        "match_type": info.get("match_type"), "match_type_number": info.get("match_type_number"),
        "gender": info.get("gender"), "team_type": info.get("team_type"),
        "season": str(info.get("season")) if info.get("season") is not None else None,
        "start_date": dates[0] if dates else None, "end_date": dates[-1] if dates else None,
        "n_days": len(dates),
        "event_name": event.get("name"),
        # Cricsheet gives these as a number in some files and text in others: kept as text
        "event_match_number": _text(event.get("match_number")),
        "event_stage": _text(event.get("stage")), "event_group": _text(event.get("group")),
        "city": info.get("city"), "venue": info.get("venue"),
        "venue_id": venue_reg.get(info.get("venue")) if info.get("venue") else None,
        "team1": teams[0] if len(teams) > 0 else None, "team2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner"), "toss_decision": toss.get("decision"),
        "winner": outcome.get("winner"), "result": outcome.get("result"),
        "by_runs": by.get("runs"), "by_wickets": by.get("wickets"), "by_innings": by.get("innings"),
        "method": outcome.get("method"), "eliminator": outcome.get("eliminator"), "bowl_out": outcome.get("bowl_out"),
        "overs": info.get("overs"), "balls_per_over": info.get("balls_per_over"),
        "player_of_match": _first(info.get("player_of_match")),
        "missing": json.dumps(info.get("missing")) if info.get("missing") else None,
    }
    players = []
    for team, names in (info.get("players") or {}).items():
        for order, name in enumerate(names, 1):
            players.append({"match_id": match_id, "team": team, "list_order": order, "player": name,
                            "player_id": reg.get(name)})
    innings_rows, deliveries = [], []
    for inn_no, inn in enumerate(d.get("innings", []), 1):
        tgt = inn.get("target") or {}
        pen = inn.get("penalty_runs") or {}
        runs = wkts = legal = n_del = 0
        seen_over: Counter = Counter()
        for over in inn.get("overs", []):
            o = over["over"]
            rep = seen_over[o]
            seen_over[o] += 1
            for ball, dl in enumerate(over["deliveries"], 1):
                r, ex = dl["runs"], dl.get("extras", {}) or {}
                is_legal = not (ex.get("wides") or ex.get("noballs"))
                ws = dl.get("wickets", []) or []
                team_w = [w for w in ws if w.get("kind") not in NOT_TEAM_WICKETS]
                runs += r["total"]
                wkts += len(team_w)
                legal += int(is_legal)
                n_del += 1
                w1 = ws[0] if ws else {}
                w2 = ws[1] if len(ws) > 1 else {}
                deliveries.append({
                    "match_id": match_id, "innings": inn_no, "over": o, "ball": ball, "repeat": rep,
                    "legal": is_legal,
                    "batter": dl["batter"], "bowler": dl["bowler"], "non_striker": dl["non_striker"],
                    "batter_id": reg.get(dl["batter"]), "bowler_id": reg.get(dl["bowler"]),
                    "non_striker_id": reg.get(dl["non_striker"]),
                    "runs_batter": r["batter"], "runs_extras": r["extras"], "runs_total": r["total"],
                    "non_boundary": bool(r.get("non_boundary", False)),
                    "wides": ex.get("wides", 0), "noballs": ex.get("noballs", 0), "byes": ex.get("byes", 0),
                    "legbyes": ex.get("legbyes", 0), "penalty": ex.get("penalty", 0),
                    "wickets": len(ws), "team_wickets": len(team_w),
                    "wicket_kind": w1.get("kind"), "player_out": w1.get("player_out"),
                    "player_out_id": reg.get(w1.get("player_out")) if w1 else None,
                    "fielders": (json.dumps([f.get("name") for f in w1["fielders"]])
                                 if w1.get("fielders") else None),
                    "wicket2_kind": w2.get("kind"), "player_out2": w2.get("player_out"),
                    "has_review": "review" in dl, "has_replacement": "replacements" in dl,
                })
        innings_rows.append({
            "match_id": match_id, "innings": inn_no, "team": inn.get("team"),
            "super_over": bool(inn.get("super_over", False)), "declared": bool(inn.get("declared", False)),
            "forfeited": bool(inn.get("forfeited", False)),
            "target_runs": tgt.get("runs"), "target_overs": tgt.get("overs"),
            "penalty_pre": pen.get("pre"), "penalty_post": pen.get("post"),
            "absent_hurt": json.dumps(inn["absent_hurt"]) if inn.get("absent_hurt") else None,
            "miscounted_overs": json.dumps(inn["miscounted_overs"]) if inn.get("miscounted_overs") else None,
            "runs": runs, "wickets": wkts, "legal_balls": legal, "deliveries": n_del,
        })
    return {"matches": [match], "innings": innings_rows, "deliveries": deliveries, "players": players}


def manifest_counts(zip_bytes: bytes) -> pd.DataFrame:
    """Rows in, counted straight from the zip without the parser: matches and deliveries per match type and gender."""
    rows = Counter()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            d = json.loads(z.read(name))
            key = (d["info"].get("match_type"), d["info"].get("gender"))
            rows[key + ("matches",)] += 1
            rows[key + ("deliveries",)] += sum(len(o["deliveries"]) for i in d.get("innings", [])
                                               for o in i.get("overs", []))
    out = pd.DataFrame([{"match_type": k[0], "gender": k[1], "what": k[2], "n": v} for k, v in rows.items()])
    return out.pivot_table(index=["match_type", "gender"], columns="what", values="n", fill_value=0).reset_index()


SMALL_INTS = ["innings", "over", "ball", "repeat", "runs_batter", "runs_extras", "runs_total", "wides", "noballs",
              "byes", "legbyes", "penalty", "wickets", "team_wickets"]


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in SMALL_INTS:
        if c in df and df[c].notna().all():
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def build_tables(zip_bytes: bytes, people_csv: bytes | None = None, chunk: int = 1000,
                 names_csv: bytes | None = None) -> dict[str, pd.DataFrame]:
    """The tables, built a chunk of matches at a time so 12M deliveries fit a 16 GB runner."""
    keys = ("matches", "innings", "deliveries", "players")
    acc: dict[str, list[dict]] = {k: [] for k in keys}
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in keys}
    failures = []

    def flush():
        for k in keys:
            if acc[k]:
                parts[k].append(_frame(acc[k]))
                acc[k] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for i, name in enumerate(sorted(z.namelist())):
            if not name.endswith(".json"):
                continue
            mid = Path(name).stem
            try:
                part = parse_match(mid, json.loads(z.read(name)))
            except Exception as exc:  # a file that will not parse is listed, never dropped silently
                failures.append({"file": name, "error": repr(exc)})
                continue
            for k, v in part.items():
                acc[k].extend(v)
            if i % chunk == chunk - 1:
                flush()
    flush()
    t = {k: (pd.concat(parts[k], ignore_index=True) if parts[k] else pd.DataFrame()) for k in keys}
    for c in SMALL_INTS:
        if c in t["deliveries"]:
            t["deliveries"][c] = pd.to_numeric(t["deliveries"][c], downcast="integer")
    t["failures"] = pd.DataFrame(failures, columns=["file", "error"])
    m = t["matches"]
    t["venues"] = (m.groupby(["venue", "city"], dropna=False)
                   .agg(venue_id=("venue_id", "first"), matches=("match_id", "size"),
                        first_date=("start_date", "min"), last_date=("start_date", "max"))
                   .reset_index())
    # Postgres keys cannot be NULL: a venue Cricsheet gives no city for is keyed with an empty city
    t["venues"]["city"] = t["venues"]["city"].fillna("")
    if people_csv is not None:
        t["people_register"] = pd.read_csv(io.BytesIO(people_csv), dtype=str)
    if names_csv is not None:   # each person's other names (full names, other spellings): the vendor feeds' join
        t["people_names"] = pd.read_csv(io.BytesIO(names_csv), dtype=str).drop_duplicates()
    return t


# ---------------------------------------------------------------------------------------------------------------
# checks and report
# ---------------------------------------------------------------------------------------------------------------

DELIVERY_KEY = ["match_id", "innings", "over", "ball", "repeat"]


def checks(t: dict[str, pd.DataFrame], manifest: pd.DataFrame, db_counts: dict[str, int] | None = None) -> dict:
    d, m = t["deliveries"], t["matches"]
    out_counts = (d.merge(m[["match_id", "match_type", "gender"]], on="match_id")
                  .groupby(["match_type", "gender"]).size().rename("deliveries_out"))
    m_counts = m.groupby(["match_type", "gender"]).size().rename("matches_out")
    rec = manifest.set_index(["match_type", "gender"]).join(m_counts).join(out_counts).fillna(0).astype(int)
    rec = rec.rename(columns={"matches": "matches_in", "deliveries": "deliveries_in"})
    rec["ok"] = (rec.matches_in == rec.matches_out) & (rec.deliveries_in == rec.deliveries_out)
    dups = int(d.duplicated(DELIVERY_KEY).sum())
    nulls = {c: int(d[c].isna().sum()) for c in ("batter", "bowler", "runs_total")}
    res = {
        "reconciliation": rec.reset_index(),
        "duplicate_delivery_keys": dups,
        "duplicate_match_ids": int(m.match_id.duplicated().sum()),
        "nulls": nulls,
        "null_registry_ids": {c: int(d[c].isna().sum()) for c in ("batter_id", "bowler_id")},
        "date_range": (m.start_date.min(), m.end_date.max()),
        "failures": len(t["failures"]),
        "db_counts": db_counts,
    }
    res["pass"] = bool(rec.ok.all() and dups == 0 and res["duplicate_match_ids"] == 0
                       and not any(nulls.values()) and res["failures"] == 0
                       and (db_counts is None or (db_counts.get("deliveries") == len(d)
                                                  and db_counts.get("matches") == len(m))))
    return res


def report_md(res: dict, t: dict[str, pd.DataFrame], source: str, day: str, db_note: str) -> str:
    rec = res["reconciliation"]
    lines = [
        f"# Cricsheet ingest report - {day}",
        "",
        f"Source: `{source}` (cricsheet.org, ODC-By). Built by `python -m cricket.ingest.cricsheet`.",
        f"Check: **{'PASS' if res['pass'] else 'FAIL'}**",
        "",
        "## Rows in (zip manifest) vs rows out (tables)",
        "",
        "| match type | gender | matches in | matches out | deliveries in | deliveries out | ok |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rec.itertuples():
        lines.append(f"| {r.match_type} | {r.gender} | {r.matches_in:,} | {r.matches_out:,} | "
                     f"{r.deliveries_in:,} | {r.deliveries_out:,} | {'yes' if r.ok else 'NO'} |")
    lines.append(f"| **all** | | {rec.matches_in.sum():,} | {rec.matches_out.sum():,} | "
                 f"{rec.deliveries_in.sum():,} | {rec.deliveries_out.sum():,} | |")
    lines += [
        "",
        "## Integrity",
        "",
        f"- Duplicate delivery keys on (match_id, innings, over, ball, repeat): {res['duplicate_delivery_keys']}",
        f"- Duplicate match IDs: {res['duplicate_match_ids']}",
        f"- Nulls in batter / bowler / runs_total: {res['nulls']['batter']} / {res['nulls']['bowler']} / "
        f"{res['nulls']['runs_total']}",
        f"- Deliveries with no registry ID (batter / bowler): {res['null_registry_ids']['batter_id']:,} / "
        f"{res['null_registry_ids']['bowler_id']:,}",
        f"- Files that failed to parse: {res['failures']}",
        f"- Date range: {res['date_range'][0]} to {res['date_range'][1]}",
        "",
        "## Tables",
        "",
        "| table | rows |",
        "|---|---:|",
    ]
    for k, v in t.items():
        lines.append(f"| {k} | {len(v):,} |")
    lines += ["", "## Database", "", db_note, "",
              "## Gaps (documented, not filled)", "",
              "Cricsheet has no ball tracking, no delivery timestamps, no fielding positions, no weather and no live "
              "feed. Bowler type and venue coordinates: no data found in Cricsheet.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="a local folder as the store (default: S3, sources/ in the capture bucket)")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--with-csv", action="store_true", help="also fetch the CSV (Ashwin) zip, raw only")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--day", help="raw day folder to build from (default: the latest held)")
    ap.add_argument("--zip", help="build from this local all_json.zip instead of the store")
    ap.add_argument("--people", help="with --zip: the local people.csv")
    ap.add_argument("--names", help="with --zip: the local names.csv")
    ap.add_argument("--db", help="also upsert into this database (sqlite:///path or a postgresql:// URL)")
    ap.add_argument("--report-dir", default="reports/cricket")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = Store(root=a.root)
    if a.fetch:
        names = DEFAULT_FILES + (["all_csv2.zip"] if a.with_csv else [])
        print(json.dumps({"fetch": fetch(store, a.day, names)}))
    if not a.build:
        return 0
    if a.zip:
        zb = Path(a.zip).read_bytes()
        pb = Path(a.people).read_bytes() if a.people else None
        nb = Path(a.names).read_bytes() if a.names else None
        source, day = a.zip, a.day or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    else:
        day = a.day or latest_raw_day(store)
        if not day:
            print("no data found: no raw Cricsheet zip in the store", file=sys.stderr)
            return 1
        zb = store.get(f"{RAW_PREFIX}/{day}/all_json.zip")
        pb = _latest(store, "people.csv", day)
        nb = _latest(store, "names.csv", day)
        source = store.where(f"{RAW_PREFIX}/{day}/all_json.zip")
    t = build_tables(zb, pb, names_csv=nb)
    for name, df in t.items():
        store.put_parquet(f"{TABLE_PREFIX}/{name}.parquet", df)
    db_counts, db_note = None, "Not loaded (no `--db` given). Aurora is not provisioned: it waits on the owner's go."
    if a.db:
        from cricket.db.loaders import connect, upsert_tables
        con = connect(a.db)
        added, db_counts = upsert_tables(con, t)
        db_note = (f"Upserted into `{a.db.split('://')[0]}` database: rows added this run {added}; "
                   f"rows held {db_counts}.")
    res = checks(t, manifest_counts(zb), db_counts)
    if a.report:
        md = report_md(res, t, source, day, db_note)
        Path(a.report_dir).mkdir(parents=True, exist_ok=True)
        path = Path(a.report_dir) / f"ingest_{day}.md"
        path.write_text(md)
        store.put(f"cricket/reports/ingest_{day}.md", md.encode())
        print(md)
    return 0 if res["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
