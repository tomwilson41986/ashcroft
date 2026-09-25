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


def _header_only() -> str:
    """What the site sends for a day with no racing: the columns, no runners."""
    return "racedate,racetime,track,horse_name\n"


def test_an_old_day_whose_file_has_no_runners_is_no_racing(hrb):
    """Christmas Day, say: the site's own file, with nobody in it."""
    d = ago(scraper.RETRY_EMPTY_DAYS + 30)
    hrb.published[d] = _header_only()
    scraper.scrape_date_range(d, d, hrb.db)
    scraper.scrape_date_range(d, d, hrb.db)

    assert _log(hrb.db)[d] == (0, "ok")
    assert hrb.asked == [d], "a day the site says had no racing is not asked again"


def test_an_old_day_with_no_file_is_an_error_not_no_racing(hrb):
    """No file is a refusal, not an answer. Recording it as done is how 17 Sep
    2026 turned 179 racing days into 'no racing' in one run."""
    d = ago(scraper.RETRY_EMPTY_DAYS + 30)
    scraper.scrape_date_range(d, d, hrb.db)
    assert _log(hrb.db)[d] == (0, "error")

    hrb.published[d] = _csv(d, runners=3)
    scraper.scrape_date_range(d, d, hrb.db)
    assert _log(hrb.db)[d] == (3, "ok"), "an 'error' day is asked again by the next backfill"


def test_a_run_of_refusals_stops_the_scrape(hrb):
    """Seven old days, none available: stop after five rather than spend the
    allowance, and record none of them as done."""
    days = [ago(scraper.RETRY_EMPTY_DAYS + 40 + k) for k in range(7)]
    scraper.scrape_date_range(min(days), max(days), hrb.db)

    assert len(hrb.asked) == scraper.MAX_UNAVAILABLE_IN_A_ROW
    log = _log(hrb.db)
    assert not any(s == "ok" for _, s in log.values())
    assert set(log) == set(hrb.asked), "days not asked are not logged at all"


def test_a_refused_repair_never_marks_a_missing_day_done(hrb):
    """The 23 Sep repair, replayed: the site stopped sending files partway.
    Days we hold keep their rows; the day we lack stays open."""
    held, missing = ago(200), ago(201)
    hrb.published[held] = _csv(held, runners=6)
    scraper.scrape_date_range(held, held, hrb.db)
    hrb.published.clear()

    scraper.scrape_date_range(missing, held, hrb.db, recheck=True)

    log = _log(hrb.db)
    assert log[held] == (6, "ok") and _results(hrb.db, held) == 6
    assert log[missing] == (0, "error")


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
    """On the last night a day is in the window it gets a final answer: the
    site's empty file makes it no racing; still no file leaves it open."""
    quiet, refused = ago(scraper.RETRY_EMPTY_DAYS), ago(scraper.RETRY_EMPTY_DAYS)
    _seed(hrb.db, {quiet: (0, "pending"),
                   **{ago(k): (30, "ok") for k in range(1, scraper.RETRY_EMPTY_DAYS)}})
    hrb.published[quiet] = _header_only()

    _nightly(monkeypatch, hrb.db)
    assert _log(hrb.db)[quiet] == (0, "ok")

    _seed(hrb.db, {refused: (0, "pending")})
    del hrb.published[refused]
    _nightly(monkeypatch, hrb.db)
    assert _log(hrb.db)[refused] == (0, "error")


# ---------------------------------------------------------------------------
# The results report reads the table's own column names
# ---------------------------------------------------------------------------

def test_results_report_reads_the_results_it_reports_on(tmp_path):
    """It selected racetime/racedate -- the site's CSV headers -- from a table
    whose columns are race_time/race_date, and raised every night, hidden by
    the email step's continue-on-error."""
    import results_report

    db = str(tmp_path / "r.db")
    conn = scraper.init_db(db)
    conn.executemany("INSERT INTO race_results (race_date, race_time, track, horse_name, bfsp) "
                     "VALUES (?, ?, ?, ?, ?)",
                     [("2026-09-18", "1.20", "Ayr", "Diplomata", 4.2),
                      ("2026-09-18", "1.20", "Ayr", "Bai Tong", 9.0),
                      ("2026-09-17", "2.00", "Ayr", "Other Day", 3.0)])
    conn.commit()
    conn.row_factory = sqlite3.Row
    rows = results_report.get_results_for_date(conn, "2026-09-18")
    conn.close()

    assert {r["horse_name"] for r in rows} == {"Diplomata", "Bai Tong"}
    assert all(r["racetime"] == "1.20" for r in rows), "downstream reads the key 'racetime'"


# ---------------------------------------------------------------------------
# The account's downloads paused (25 Sep 2026)
# ---------------------------------------------------------------------------

PAUSE_PAGE = ("Data downloads temporarily unavailable An unusually high number of data download requests "
              "has been detected from this account. Access to data downloads has been temporarily paused to "
              "protect the service. Download access will become available again: Wednesday 30th September "
              "2026 at 13:53 The remainder of HorseRaceBase remains available.")


def test_the_pause_page_gives_its_end():
    from datetime import datetime
    assert scraper.paused_until(PAUSE_PAGE) == datetime(2026, 9, 30, 13, 53)
    assert scraper.paused_until("no time given") is None


def test_a_pause_stops_the_scrape_at_once_and_holds_until_it_ends(hrb, monkeypatch):
    """The first paused answer stops the run (no five more requests), records
    no day, and every later run asks nothing until the pause is over."""
    from datetime import datetime

    def paused(session, user_id, d):
        hrb.asked.append(d)
        raise scraper.DownloadsPaused(scraper.paused_until(PAUSE_PAGE), PAUSE_PAGE)

    monkeypatch.setattr(scraper, "download_csv", paused)
    monkeypatch.setattr(scraper, "uk_now", lambda: datetime(2026, 9, 25, 11, 0))
    days = [ago(k) for k in range(1, 4)]
    scraper.scrape_date_range(min(days), max(days), hrb.db)
    assert len(hrb.asked) == 1
    assert _log(hrb.db) == {}
    conn = scraper.init_db(hrb.db)
    assert scraper.pause_in_force(conn) == datetime(2026, 9, 30, 13, 53)
    conn.close()

    scraper.scrape_date_range(min(days), max(days), hrb.db)      # the next night
    assert len(hrb.asked) == 1, "nothing asked while paused"

    monkeypatch.setattr(scraper, "uk_now", lambda: datetime(2026, 9, 30, 14, 0))
    monkeypatch.setattr(scraper, "download_csv", lambda s, u, d: hrb.asked.append(d) or _csv(d))
    scraper.scrape_date_range(min(days), max(days), hrb.db)      # after it ends
    assert len(hrb.asked) == 4 and all(_results(hrb.db, d) == 2 for d in days)


def test_download_csv_turns_the_pause_page_into_the_pause(monkeypatch):
    class R:
        headers = {"Content-Type": "text/html; charset=UTF-8"}
        text = '<div style="width:700px"><h2>Data downloads temporarily unavailable</h2><p>' + PAUSE_PAGE + "</p></div>"

        def raise_for_status(self):
            pass

    class S:
        def post(self, url, data):
            return R()

    with pytest.raises(scraper.DownloadsPaused) as e:
        scraper.download_csv(S(), "1", ago(1))
    assert e.value.until.year == 2026 and e.value.until.hour == 13
