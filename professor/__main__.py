"""The Professor's command line.

    python -m professor --card  [--day D] [--until HH:MM] [--dry-run] [--no-scrape] [--sources racingpost]
        the day's card at a point in time: the ledger's positions by --until (UK, default now), the bookmaker prices,
        the selection, the write-ups and cards to out/professor/<day>/ (and S3 unless --dry-run), the posts
    python -m professor --alerts [--day D] [--until HH:MM] [--every 60]
        the live loop: the ledger read every --every seconds until --until UK, each new fill that qualifies posted
        at once, the free tips' day card posted at config publish_at
    python -m professor --odds [--day D]
        the scrape alone: the snapshot and the status
    python -m professor --replay --day D
        a past day from S3: the card as it would have stood at select_at, no posting

Posting needs PROFESSOR_LIVE=yes and no --dry-run; --test-channels sends to TELEGRAM_TEST_CHANNEL_ID instead.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from professor import odds as odds_mod
from professor import store
from professor.candidates import build, frame
from professor.publish import channels, post_day_card, post_tip
from professor.selection import alerts as select_alerts
from professor.selection import card as select_card
from professor.selection import load_config, tips_json

log = logging.getLogger("professor")
try:
    from zoneinfo import ZoneInfo
    LONDON = ZoneInfo("Europe/London")
except Exception:  # noqa: BLE001
    LONDON = timezone.utc


def uk_now() -> datetime:
    return datetime.now(LONDON)


def uk_at(day: date, hhmm: str) -> datetime:
    h, m = hhmm.split(":")
    return datetime(day.year, day.month, day.day, int(h), int(m), tzinfo=LONDON)


def _day(s: str | None) -> date:
    return date.fromisoformat(s) if s else uk_now().date()


def _wanted(cands) -> set[tuple[str, str]]:
    from professor.scrapers.base import norm_venue
    return {(norm_venue(c.venue), c.off_uk) for c in cands if c.off_uk}


def run_card(args) -> int:
    cfg = load_config()
    day = _day(args.day)
    until = uk_at(day, args.until) if args.until else None
    if args.candidates_file:
        from professor.candidates import from_frame
        cands = from_frame(pd.read_csv(args.candidates_file, dtype={"market_id": str}))
        if until is not None:
            cands = [c for c in cands if c.first_utc is not None and c.first_utc <= until]
    else:
        cands = build(day, until=until)
    out = store.out_dir(day)
    print(f"{day}: {len(cands)} positions in the ledger" + (f" by {args.until} UK" if args.until else ""))
    if not cands:
        return 0
    cdf = frame(cands)
    store.write_csv(store.key(day, "candidates.csv"), cdf, out / "candidates.csv")
    from professor.candidates import FACTS_STATUS
    (out / "facts_status.json").write_text(json.dumps(FACTS_STATUS, indent=1, default=str))
    quotes, bog, status, nrs = {}, {}, {}, set()
    if not args.no_scrape:
        df, status = odds_mod.scrape(day, cfg, sources=tuple(args.sources.split(",")), wanted=_wanted(cands) if args.only_ours else None)
        quotes, bog = odds_mod.match(df, cands)
        nrs = odds_mod.non_runners(df, cands)
        print("odds:", json.dumps(status), f"| {len(df)} quotes, {len(quotes)} of our horses matched")
    tips, fates = select_card(cands, cfg, quotes, bog, nrs)
    pd.DataFrame(fates).to_csv(out / "fates.csv", index=False)
    rows = tips_json(tips)
    for t, r in zip(tips, rows):
        r["odds_status"] = status
    store.write_json(store.key(day, "tips.json"), rows, out / "tips.json")
    chans = channels(cfg, test=args.test_channels)
    paid_open = os.getenv("PROFESSOR_PAID_OPEN", "").lower() == "yes"
    results = [post_tip(t, cfg, chans, paid_open) for t in tips]
    card = post_day_card(tips, cfg, chans, f"The Professor's free tips, {day:%a %d %b}")
    summary = {"day": str(day), "positions": len(cands), "quoted": len(quotes), "tips": len(tips),
               "withheld": sum(t.withheld for t in tips), "odds_status": status, "posts": results, "day_card": card,
               "live": os.getenv("PROFESSOR_LIVE", "").lower() == "yes" and not store.dry()}
    store.write_json(store.key(day, "summary.json"), summary, out / "summary.json")
    _print_tips(tips, fates)
    return 0


def _print_tips(tips, fates) -> None:
    print(f"\n{len(tips)} tips:")
    for t in tips:
        c = t.cand
        print(f"  {t.rank}. [{t.tier}] {c.off_uk} {c.venue} {c.runner}: advised "
              f"{t.advised:.2f} ({t.price_source}{', ' + str(len(t.bookmakers)) + ' books' if t.bookmakers else ''}), "
              f"exchange {t.advised_exchange:.2f} (GBP{t.exchange_size:.0f}), expected SP {c.exp_bsp:.2f}, "
              f"overlay {t.overlay:+.0%}, p_win {c.p_win:.2f}, backed {c.first_uk}")
    if fates:
        why = pd.Series([f["fate"] for f in fates]).value_counts()
        print("\nthe candidates' fates:")
        for k, n in why.items():
            print(f"  {n:4d}  {k}")


def run_alerts(args) -> int:
    cfg = load_config()
    day = _day(args.day)
    stop = uk_at(day, args.until or cfg.get("publish_at", "10:00"))
    chans = channels(cfg, test=args.test_channels)
    paid_open = os.getenv("PROFESSOR_PAID_OPEN", "").lower() == "yes"
    out = store.out_dir(day)
    tips, seen_status = [], {}
    quotes, bog = {}, {}
    posted_card = False
    print(f"{day}: alerts until {stop:%H:%M} UK, every {args.every}s")
    while True:
        now = uk_now()
        if store.stopped():
            print("STOP is set; no alerts")
            break
        cands = build(day, until=None, with_facts=True)
        new_keys = [(c.market_id, c.selection_id) for c in cands if (c.market_id, c.selection_id) not in quotes]
        if new_keys and not args.no_scrape:
            fresh = [c for c in cands if (c.market_id, c.selection_id) in set(new_keys)]
            try:
                df, status = odds_mod.scrape(day, cfg, sources=("racingpost",), wanted=_wanted(fresh), keep_raw=False)
                q, b = odds_mod.match(df, fresh)
                quotes.update(q)
                bog.update(b)
                for c in fresh:                       # a race without quotes is still read: the exchange fallback
                    quotes.setdefault((c.market_id, c.selection_id), q.get((c.market_id, c.selection_id)))
                seen_status = status
            except Exception as e:  # noqa: BLE001
                log.warning("scrape: %s", str(e)[:120])
        new, fates = select_alerts(cands, tips, cfg, {k: v for k, v in quotes.items() if v}, bog)
        for t in new:
            r = post_tip(t, cfg, chans, paid_open)
            tips.append(t)
            print(f"  {now:%H:%M} alert {t.rank} [{t.tier}{', withheld' if t.withheld else ''}] {t.cand.off_uk} "
                  f"{t.cand.venue} {t.cand.runner} {t.advised:.2f} ({t.price_source}) -> {r['posted'] or 'withheld'}")
        if tips:
            store.write_json(store.key(day, "tips.json"), tips_json(tips), out / "tips.json")
        if not posted_card and now >= uk_at(day, cfg.get("publish_at", "10:00")):
            card = post_day_card(tips, cfg, chans, f"The Professor's free tips, {day:%a %d %b}")
            print(f"  {now:%H:%M} day card -> {card}")
            posted_card = True
        if now >= stop:
            break
        time.sleep(args.every)
    summary = {"day": str(day), "tips": len(tips), "odds_status": seen_status, "stopped_at": uk_now().isoformat()}
    store.write_json(store.key(day, "summary.json"), summary, out / "summary.json")
    _print_tips(tips, [])
    return 0


def run_odds(args) -> int:
    cfg = load_config()
    day = _day(args.day)
    df, status = odds_mod.scrape(day, cfg, sources=tuple(args.sources.split(",")))
    print(json.dumps(status, indent=1))
    print(f"{len(df)} quotes on {df['_horse'].nunique() if len(df) else 0} horses in "
          f"{df.groupby(['_venue', 'race_hhmm']).ngroups if len(df) else 0} races")
    if len(df):
        print(df.groupby("bookmaker").size().sort_values(ascending=False).to_string())
    return 0


def run_settle(args) -> int:
    """Settle tips files (a glob: the dry runs' kept in reports/, or a day's tips.json) against their days' ledgers."""
    import glob as _glob
    from professor import settle as st
    files = sorted(_glob.glob(args.tips_file)) if args.tips_file else []
    if not files and args.day:
        local = store.out_dir(args.day) / "tips.json"
        if not local.exists():
            raw = store.get_bytes(store.key(args.day, "tips.json"))
            if raw:
                local.write_bytes(raw)
        files = [str(local)] if local.exists() else []
    if not files:
        print("no tips to settle")
        return 0
    frames = []
    for f in files:
        settled, summ = st.settle_file(f)
        if settled.empty:
            continue
        day = str(settled["day"].iloc[0])
        st.write(day, settled, summ)
        print(f"{f}:\n{st.report(settled, summ)}")
        frames.append(settled)
    if len(frames) > 1:
        allt = pd.concat(frames, ignore_index=True)
        print("\nall days:\n" + "\n".join(st.report(allt, st.summary(allt)).splitlines()[-2:]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="professor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--alerts", action="store_true")
    ap.add_argument("--odds", action="store_true")
    ap.add_argument("--settle", action="store_true", help="settle tips against the ledger's results and BSPs")
    ap.add_argument("--tips-file", help="--settle: a tips.json or a glob of them (default: the day's)")
    ap.add_argument("--replay", action="store_true", help="--card at select_at with no posting")
    ap.add_argument("--day")
    ap.add_argument("--until", help="UK HH:MM")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-scrape", action="store_true")
    ap.add_argument("--only-ours", action="store_true", help="scrape only the races we hold a horse in")
    ap.add_argument("--sources", default="racingpost,oddschecker,marketform")
    ap.add_argument("--test-channels", action="store_true")
    ap.add_argument("--candidates-file", help="the card from a saved candidates.csv instead of the ledger")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.dry_run or args.replay:
        os.environ["PROFESSOR_DRY"] = "1"
    if args.replay:
        args.card = True
        args.until = args.until or load_config().get("select_at", "09:55")
    if args.card:
        return run_card(args)
    if args.alerts:
        return run_alerts(args)
    if args.odds:
        return run_odds(args)
    if args.settle:
        return run_settle(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
