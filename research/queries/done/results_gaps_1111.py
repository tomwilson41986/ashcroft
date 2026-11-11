"""Every gap between Betfair's record and race_results (the owner, 11 Nov: "look into all gaps"). Read-only.

1. Live record, 1 Oct on: each meeting (day, venue): Betfair WIN markets against the races race_results holds
   (distinct race times at that track that day). Meetings with no results, and meetings short of races.
2. The runners still unlinked after the re-match though their track's results are held: by Betfair's own settled
   status (REMOVED = a non-runner, expected), and for those that RAN (WINNER/LOSER): is the race itself missing from
   race_results (no race at that time), or is the horse there under another spelling (nearest name in that race)?
3. Betfair's price files, 2018 on: WIN runners with no race_results row by year and country; whole meetings
   unmatched, by year, and listed for 2025-26.
4. Days Betfair priced GB/IE racing on which race_results holds nothing at all, by year, and listed for 2025-26.
5. race_results meetings from 2024 with no Betfair price row at all (results we hold that the price files lack).
"""
import difflib
import sqlite3
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, ".")
import betfair_recorder as br  # noqa: E402
from betfair_prices import db_time_to_24h, normalise_horse, normalise_track, utc_iso_to_uk_hhmm  # noqa: E402

pd.set_option("display.width", 230)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_colwidth", 60)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
last = pd.read_sql_query("SELECT MAX(race_date) d FROM race_results", conn).d[0]
print("race_results to", last)

# ------------------------------------------------------------------ 1
m = pd.read_sql_query("SELECT * FROM betfair_live_markets WHERE race_date >= '2026-10-01' AND race_date <= ? "
                      "AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'", conn, params=[last])
m["tn"] = m.venue.map(normalise_track)
m["hhmm"] = m.market_start_utc.map(utc_iso_to_uk_hhmm)
rr = pd.read_sql_query("SELECT id, race_date, race_time, track, horse_name FROM race_results WHERE race_date >= '2026-10-01'", conn)
rr["tn"] = rr.track.map(normalise_track)
rr["hhmm"] = rr.race_time.map(db_time_to_24h)
rr["hn"] = rr.horse_name.map(normalise_horse)
bf_meet = m.groupby(["race_date", "tn"]).agg(venue=("venue", "first"), bf_races=("market_id", "nunique"),
                                             bf_runners=("market_id", "size"))
rr_meet = rr.groupby(["race_date", "tn"]).agg(rr_races=("hhmm", "nunique"), rr_runners=("id", "size"))
meet = bf_meet.join(rr_meet, how="left").fillna({"rr_races": 0, "rr_runners": 0})
print(f"\n1. live record: {len(meet)} meetings; with no results {int((meet.rr_races == 0).sum())}; "
      f"short of races {int(((meet.rr_races > 0) & (meet.rr_races < meet.bf_races)).sum())}")
print("   meetings with no results:")
print(meet[meet.rr_races == 0].reset_index()[["race_date", "venue", "bf_races", "bf_runners"]].to_string(index=False))
print("   meetings short of races (Betfair markets > races held):")
print(meet[(meet.rr_races > 0) & (meet.rr_races < meet.bf_races)].reset_index()[
    ["race_date", "venue", "bf_races", "rr_races", "bf_runners", "rr_runners"]].to_string(index=False))
days_bf = set(m.race_date.unique())
days_rr = set(rr.race_date.unique())
print("   days with Betfair markets and no race_results row at all:", sorted(days_bf - days_rr))

# ------------------------------------------------------------------ 2
out = []
for d, part in m[m.race_results_id.isna()].groupby("race_date"):
    x = br.match_markets(part.drop(columns=["race_results_id", "tn", "hhmm"]), conn, date.fromisoformat(d))
    out.append(x[x.race_results_id.isna()])
left = pd.concat(out)
left["tn"] = left.venue.map(normalise_track)
left["hhmm"] = left.market_start_utc.map(utc_iso_to_uk_hhmm)
fin = pd.read_sql_query("SELECT market_id, selection_id, runner_status FROM betfair_live_marks WHERE mark = 'final' "
                        "AND race_date >= '2026-10-01'", conn)
fin["selection_id"] = pd.to_numeric(fin.selection_id, errors="coerce")
left["selection_id"] = pd.to_numeric(left.selection_id, errors="coerce")
left = left.merge(fin, on=["market_id", "selection_id"], how="left")
left["status"] = left.runner_status.fillna("no final mark")
races = set(zip(rr.race_date, rr.tn, rr.hhmm))
meets = set(zip(rr.race_date, rr.tn))
left["meeting_held"] = [(d, t) in meets for d, t in zip(left.race_date, left.tn)]
left["race_held"] = [(d, t, h) in races for d, t, h in zip(left.race_date, left.tn, left.hhmm)]
print(f"\n2. still unlinked after the re-match: {len(left):,}")
print(left.groupby(["status", "meeting_held", "race_held"]).size().rename("runners").to_string())
ran = left[left.status.isin(["WINNER", "LOSER", "PLACED"])]
print(f"\n   ran on Betfair but unlinked: {len(ran):,}; meeting missing {int((~ran.meeting_held).sum())}, "
      f"race missing (meeting held) {int((ran.meeting_held & ~ran.race_held).sum())}, race held {int(ran.race_held.sum())}")
by_race = rr.groupby(["race_date", "tn", "hhmm"]).hn.apply(list).to_dict()
by_meet_times = rr.groupby(["race_date", "tn"]).hhmm.apply(lambda s: sorted(set(s.dropna()))).to_dict()
rows = []
for r in ran[ran.race_held].itertuples():
    names = by_race.get((r.race_date, r.tn, r.hhmm), [])
    near = difflib.get_close_matches(normalise_horse(r.runner_name), names, n=1, cutoff=0.75)
    rows.append({"race_date": r.race_date, "venue": r.venue, "time": r.hhmm, "betfair": r.runner_name,
                 "nearest_in_results": near[0] if near else "", "field_in_results": len(names)})
nm = pd.DataFrame(rows)
if len(nm):
    print(f"\n   race held, horse not found: {len(nm)}; with a near spelling in that race {int((nm.nearest_in_results != '').sum())}")
    print(nm.sort_values("nearest_in_results", ascending=False).head(40).to_string(index=False))
miss_race = ran[ran.meeting_held & ~ran.race_held].drop_duplicates(["race_date", "tn", "hhmm"])
print(f"\n   races Betfair ran that race_results lacks though the meeting is held: {len(miss_race)}")
for r in miss_race.head(40).itertuples():
    print(f"     {r.race_date} {r.venue} {r.hhmm}; results hold times {by_meet_times.get((r.race_date, r.tn), [])}")

# ------------------------------------------------------------------ 3
bp = pd.read_sql_query("""SELECT substr(race_date, 1, 4) y, country, COUNT(*) n, SUM(race_results_id IS NULL) unmatched
                          FROM betfair_prices WHERE LOWER(market_type) = 'win' AND race_date >= '2018-01-01'
                          AND race_date <= ? GROUP BY y, country""", conn, params=[last])
bp["share_unmatched"] = (bp.unmatched / bp.n).round(4)
print("\n3. Betfair price files, WIN runners with no race_results row:")
print(bp.to_string(index=False))
mt = pd.read_sql_query("""SELECT race_date, country, COALESCE(track_bf, menu_hint) course, COUNT(*) n,
                                 SUM(race_results_id IS NOT NULL) matched
                          FROM betfair_prices WHERE LOWER(market_type) = 'win' AND race_date >= '2018-01-01'
                          AND race_date <= ? GROUP BY race_date, country, course""", conn, params=[last])
dead = mt[mt.matched == 0]
print(f"   meetings in the price files: {len(mt):,}; wholly unmatched {len(dead):,}; by year:")
print(dead.groupby(dead.race_date.str[:4]).agg(meetings=("n", "size"), runners=("n", "sum")).to_string())
print("   wholly unmatched meetings, 2025-26:")
print(dead[dead.race_date >= "2025-01-01"].to_string(index=False))
part = mt[(mt.matched > 0) & (mt.matched / mt.n < 0.6) & (mt.race_date >= "2025-01-01")]
print(f"   meetings under 60% matched, 2025-26: {len(part)}")
print(part.head(60).to_string(index=False))

# ------------------------------------------------------------------ 4
d_bp = pd.read_sql_query("""SELECT race_date, COUNT(*) n FROM betfair_prices WHERE LOWER(market_type) = 'win'
                            AND race_date >= '2018-01-01' AND race_date <= ? GROUP BY race_date""", conn, params=[last])
d_rr = pd.read_sql_query("SELECT race_date, COUNT(*) n FROM race_results WHERE race_date >= '2018-01-01' GROUP BY race_date", conn)
gap = d_bp[~d_bp.race_date.isin(set(d_rr.race_date))]
print(f"\n4. days Betfair priced racing with no race_results row: {len(gap)}; by year:")
print(gap.groupby(gap.race_date.str[:4]).agg(days=("n", "size"), runners=("n", "sum")).to_string())
print("   2025-26:", gap[gap.race_date >= "2025-01-01"].race_date.tolist())
thin = d_bp.merge(d_rr, on="race_date", suffixes=("_bf", "_rr"))
thin = thin[(thin.n_rr < 0.6 * thin.n_bf) & (thin.race_date >= "2025-01-01")]
print(f"   days in 2025-26 where race_results holds under 60% of Betfair's runners: {len(thin)}")
print(thin.to_string(index=False))

# ------------------------------------------------------------------ 5
rm = pd.read_sql_query("""SELECT r.race_date, r.track, COUNT(*) n,
                                 SUM(EXISTS (SELECT 1 FROM betfair_prices b WHERE b.race_results_id = r.id)) priced
                          FROM race_results r WHERE r.race_date >= '2024-01-01' AND r.race_date <= '2026-10-31'
                          GROUP BY r.race_date, r.track""", conn)
none = rm[rm.priced == 0]
print(f"\n5. race_results meetings 2024 to Oct 2026: {len(rm):,}; with no Betfair price row {len(none)}; by track:")
print(none.groupby("track").agg(meetings=("n", "size"), runners=("n", "sum"), first=("race_date", "min"),
                                last=("race_date", "max")).sort_values("meetings", ascending=False).head(40).to_string())
