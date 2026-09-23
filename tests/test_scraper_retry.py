"""The nightly scrape asked for each day before it was published, and believed the answer.

`daily-results.yml` runs at 21:30 UTC with `--to today`. horseracebase has not
published the day by then, so the download comes back empty, and the scraper
logged that as `status='ok', rows_found=0` -- done. The resume point
(`MAX(scrape_date) WHERE status='ok'`) then moved past the day, and the per-day
skip refused to ask for it again even when a backfill named it. Every day from
18 to 22 September 2026 was lost that way, and every run reported success.

What these tests hold the scraper to:

- an empty answer for a recent day is 'pending', not done;
- a recent day already logged done-and-empty is asked again (the repair);
- a day with rows is never asked for twice;
- an empty answer for an old day still means no racing;
- the nightly invocation, which passes no --from, looks back far enough to
  heal itself -- including a day that a later day's results overtook.

horseracebase is faked; the database, the scrape loop and the CSV parser are real.
"""

import sqlite3
import sys
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import scraper

TODAY = date.today()


def ago(n: int) -> date:
    return TODAY - timedelta(days=n)


def _csv(d: date, runners: int = 2) -> str:
    head = "racedate,racetime,track,horse_name\n"
    return head + "".join(f"{d.isoformat()},2.30,Ascot,Horse {i}\n"
                          for i in range(runners))


@pytest.fixture
def hrb(monkeypatch, tmp_path):
    """A fake horseracebase. `published` maps a date to its CSV; any other
    date comes back empty, which is what an unpublished day looks like."""
    fake = SimpleNamespace(db=str(tmp_path / "horse_racing.db"),
                           published={}, asked=[])

    def download_csv(session, user_id, d):
        fake.asked.append(d)
        return fake.published.get(d)

    monkeypatch.setattr(scraper, "create_session", lambda: object())
    monkeypatch.setattr(scraper, "login", lambda session: "1")
    monkeypatch.setattr(scraper, "download_csv", download_csv)
    monkeypatch.setattr(scraper, "save_csv_file", lambda csv_text, d: None)
    monkeypatch.setattr(scraper, "REQUEST_DELAY", 0)
    return fake


def _seed(db: str, entries: dict[date, tuple[int, str]]) -> None:
    conn = scraper.init_db(db)
    conn.executemany(
        "INSERT OR REPLACE INTO scrape_log (scrape_date, rows_found, status) "
        "VALUES (?, ?, ?)",
        [(d.isoformat(), n, s) for d, (n, s) in entries.items()])
    conn.commit()
    conn.close()


def _log(db: str) -> dict[date, tuple[int, str]]:
    conn = sqlite3.connect(db)
    try:
        return {date.fromisoformat(d): (n, s) for d, n, s in conn.execute(
            "SELECT scrape_date, rows_found, status FROM scrape_log")}
    finally:
        conn.close()


def _results(db: str, d: date) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM race_results WHERE race_date = ?",
                            (d.isoformat(),)).fetchone()[0]
    finally:
        conn.close()


def _resume_point(db: str) -> date | None:
    conn = scraper.init_db(db)
    try:
        return scraper.get_last_scraped_date(conn)
    finally:
        conn.close()


def _nightly(monkeypatch, db: str) -> None:
    """What daily-results.yml runs: no --from, so the scraper resumes."""
    monkeypatch.setattr(sys, "argv", ["scraper.py", "--to", TODAY.isoformat(),
                                      "--no-backup", "--db", db])
    scraper.main()


# ---------------------------------------------------------------------------
# The day itself
# ---------------------------------------------------------------------------

def test_the_retry_window_boundary():
    t = date(2026, 9, 23)
    assert scraper.is_recent(t, t)
    assert scraper.is_recent(t - timedelta(days=scraper.RETRY_EMPTY_DAYS - 1), t)
    assert not scraper.is_recent(t - timedelta(days=scraper.RETRY_EMPTY_DAYS), t)


def test_an_unpublished_day_is_pending_not_done(hrb):
    scraper.scrape_date_range(TODAY, TODAY, hrb.db)

    assert _log(hrb.db)[TODAY] == (0, "pending")
    assert _resume_point(hrb.db) is None, (
        "an empty answer for today must not move the resume point past today")


def test_a_pending_day_is_picked_up_once_published(hrb):
    d = ago(1)
    scraper.scrape_date_range(d, d, hrb.db)          # first night: not up yet
    hrb.published[d] = _csv(d, runners=3)
    scraper.scrape_date_range(d, d, hrb.db)          # next night: it is

    assert _log(hrb.db)[d] == (3, "ok")
    assert _results(hrb.db, d) == 3


def test_a_recent_day_logged_done_and_empty_is_asked_again(hrb):
    """The exact state 18-22 September were left in, and the backfill that repairs it."""
    lost = [ago(k) for k in range(1, 6)]
    _seed(hrb.db, {d: (0, "ok") for d in lost})
    for d in lost:
        hrb.published[d] = _csv(d)

    scraper.scrape_date_range(min(lost), max(lost), hrb.db)

    assert sorted(hrb.asked) == sorted(lost)
    assert all(_log(hrb.db)[d] == (2, "ok") for d in lost)
    assert all(_results(hrb.db, d) == 2 for d in lost)


def test_a_day_with_rows_is_never_asked_for_twice(hrb):
    d = ago(2)
    hrb.published[d] = _csv(d)
    scraper.scrape_date_range(d, d, hrb.db)
    scraper.scrape_date_range(d, d, hrb.db)

    assert hrb.asked == [d]


def test_an_old_empty_day_is_still_no_racing(hrb):
    """Christmas Day, say. Old enough that empty is the answer, not a delay."""
    d = ago(scraper.RETRY_EMPTY_DAYS + 30)
    scraper.scrape_date_range(d, d, hrb.db)
    scraper.scrape_date_range(d, d, hrb.db)

    assert _log(hrb.db)[d] == (0, "ok")
    assert hrb.asked == [d], "a settled empty day must not be asked again"


# ---------------------------------------------------------------------------
# The nightly run, which resumes rather than being told where to start
# ---------------------------------------------------------------------------

def test_the_nightly_run_heals_the_lost_days_by_itself(hrb, monkeypatch):
    """Before the fix the resume point sat past the damage, so no amount of
    nightly runs would ever have gone back for it."""
    lost = [ago(k) for k in range(1, 6)]
    _seed(hrb.db, {ago(7): (40, "ok"), ago(6): (35, "ok"),
                   **{d: (0, "ok") for d in lost}})
    for d in lost:
        hrb.published[d] = _csv(d)

    _nightly(monkeypatch, hrb.db)

    log = _log(hrb.db)
    assert all(log[d] == (2, "ok") for d in lost)
    assert log[TODAY] == (0, "pending"), "today is not published at 21:30"
    assert ago(7) not in hrb.asked and ago(6) not in hrb.asked, (
        "the look-back must not re-download days already in")
    assert _resume_point(hrb.db) == ago(1)


def test_a_later_day_coming_in_does_not_bury_a_pending_one(hrb, monkeypatch):
    """Resuming after the latest day in would step over an earlier gap."""
    gap = ago(3)
    _seed(hrb.db, {gap: (0, "pending"), ago(2): (30, "ok"), ago(1): (28, "ok")})
    hrb.published[gap] = _csv(gap)

    _nightly(monkeypatch, hrb.db)

    assert _log(hrb.db)[gap] == (2, "ok")
    assert ago(2) not in hrb.asked and ago(1) not in hrb.asked


def test_a_pending_day_gets_its_final_answer_as_it_ages_out(hrb, monkeypatch):
    """A day with no racing is pending while recent, and settles as no racing
    on the last night it is in the window -- not pending forever."""
    quiet = ago(scraper.RETRY_EMPTY_DAYS)
    _seed(hrb.db, {quiet: (0, "pending"),
                   **{ago(k): (30, "ok") for k in range(1, scraper.RETRY_EMPTY_DAYS)}})

    _nightly(monkeypatch, hrb.db)

    assert _log(hrb.db)[quiet] == (0, "ok")
