"""The night-before card (the owner, 4 Nov 2026: "Run predictions by night before"): HRB's 48-hour declarations,
one page a race, rewritten into the one-day page's layout and read by the one-day page's parser. The page shapes
are those scripts/hrb_card_probe.py read on 4 Nov (tomorrow's Chelmsford City card)."""

from datetime import date

import pytest

import daily_predictions as dp

INDEX = """<html><body><p>Thursday 5th November 2026</p>
<a href="?earlyraceid=101">1.57</a> <a href="?earlyraceid=101">Midnite Built Different Handicap (Div 1)</a>
<a href="?earlyraceid=102">2.30</a> <a href="?earlyraceid=102">Season 14 Has Begun Handicap</a></body></html>"""

RACE = """<html><body><table><tr><td><table><tr><td>menu</td></tr></table></td></tr></table>
<span>1.57 Chelmsford City (2 runners)</span>
<span>1m2f  (2200 yards) Class 6, Standard, 3yo+, Win: £3140 Avg OR : 51, Median OR : 50 Left Handed, Stall Positioning : Inside</span>
<table>
<tr><td></td><td>No.</td><td>Form</td><td>D</td><td>Horse</td><td>A/S</td><td>W/HG</td><td>Jockey</td><td>Trainer</td>
<td>Miles</td><td>Stall</td><td>OR</td><td>Odds</td><td>HRB</td><td>CD</td><td>S</td></tr>
<tr><td></td><td>10</td><td>8030354</td><td>3</td>
<td class="readablerowlink" title="Chestnut, Male, Stallion - Tasleet, Dam - Labise (IRE), Owner - Woodward, Jones And Astbury, Foal Date - 07-05-2023 "><a href="horses.php?id=555">Labiche</a></td>
<td>3yo G</td><td>9-1 Hood 1</td><td><a href="jockeys.php?id=9">Loughnane, Billy</a></td><td>Owen, J P</td><td>47</td><td>14</td><td>50</td><td>4/1</td><td>103</td><td></td><td>5</td><td></td></tr>
<tr><td></td></tr>
<tr><td></td><td>1</td><td>8465212</td><td>30</td>
<td title="Bay, Male, Stallion - Kodi Bear (IRE), Dam - Star Fire, Owner - L &amp; M Atkins, Foal Date - 04-03-2018 "><a href="horses.php?id=556">Jenson Benson (IRE)</a></td>
<td>8yo G</td><td>9-9 CkPc 1 TT</td><td><a href="jockeys.php?id=8">Mangan, Gina</a></td><td>Lycett, Shaun</td><td>134</td><td>3</td><td>55</td><td>5/1</td><td>134</td><td>CD</td><td>3</td><td></td></tr>
</table></body></html>"""


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, index=INDEX):
        self.index, self.got = index, []

    def get(self, url):
        self.got.append(url)
        if "showearly" in url:
            return _Resp(self.index)
        return _Resp(RACE if url.endswith("101") else RACE.replace("1.57 Chelmsford", "2.30 Chelmsford"))


def test_the_info_line_and_the_headgear_in_the_one_day_pages_words():
    assert dp.early_info_line("1m2f  (2200 yards) Class 6, Standard, 3yo+, Win: £3140 Avg OR : 51, Median OR : 50 "
                              "Left Handed, Stall Positioning : Inside") == "Class 6, 1m2f , Standard, 3yo+, Win: £3140"
    assert dp.early_info_line("7½f  (1702 yards) Class 4, Standard, 3yo+, Win: £4972 Avg OR : 71") \
        == "Class 4, 7½f , Standard, 3yo+, Win: £4972"
    assert dp.early_headgear(["CkPc", "1", "TT"]) == "CkPcTT" and dp.early_headgear(["Hood", "1"]) == "Hood"
    assert dp.early_headgear(["Hood", "Blnk"]) == "H+B" and dp.early_headgear([]) == ""
    assert dp.early_card_date(INDEX) == date(2026, 11, 5)


def test_tomorrows_card_reads_as_the_morning_page_would():
    s = _Session()
    df = dp.scrape_early_racecards(s, date(2026, 11, 5), pause=0)
    assert len([u for u in s.got if "earlyraceid" in u]) == 2               # one page a race, each once
    assert sorted(set(df.race_time)) == ["1:57", "2:30"] and len(df) == 4
    assert set(df.race_name) == {"Midnite Built Different Handicap (Div 1)", "Season 14 Has Begun Handicap"}
    r = df[df.race_time.eq("1:57")].set_index("horse_name")
    assert list(r.index) == ["Labiche", "Jenson Benson (IRE)"]
    a, b = r.loc["Labiche"], r.loc["Jenson Benson (IRE)"]
    assert a.track == "Chelmsford City" and a.race_class == "Class 6" and a.going_description == "Standard"
    # "1m2f " with its space, as the one-day parser reads the one-day page's "Class 6, 1m2f , Standard": the same
    # text the morning's card carries for a distance in miles
    one_day = dp.scrape_racecard_html(type("S", (), {"get": lambda self, u: _Resp(
        "<span>1.57 Chelmsford City(1 runners)</span><span>Class 6, 1m2f , Standard, 3yo+, Win: £3140</span>"
        "<table><tr>" + "".join(f"<td>{h}</td>" for h in dp._ONE_DAY_HEAD) + "</tr><tr><td>1</td><td>1</td>"
        "<td>3</td><td>X</td><td>3</td><td>9-1</td><td></td><td>J, A</td><td>T, B</td><td>1</td><td>1</td>"
        "<td>50</td><td>4/1</td><td>1</td><td></td><td>1</td></tr></table>")})(), date(2026, 11, 5))
    assert a.race_distance == one_day.race_distance.iloc[0] == "1m2f "
    assert a.prize_money == "3140" and a.race_date == "2026-11-05"
    assert a.race_name == "Midnite Built Different Handicap (Div 1)"
    assert a.horse_age == 3 and a.pounds == 9 * 14 + 1 and a.headgear == "Hood" and b.headgear == "CkPcTT"
    assert a.stall == 14 and a.official_rating == 50 and a.odds == pytest.approx(5.0) and a.days_since_lr == 3
    assert a.jockey_name == "Loughnane, Billy" and a.trainer == "Owen, J P" and a.number_of_runners == 2
    assert a.card_stallion == "Tasleet" and a.card_dam == "Labise (IRE)" and a.card_sex == "Male"


def test_a_48_hour_page_for_another_day_is_not_read_as_tomorrow():
    s = _Session(INDEX.replace("5th November", "6th November"))
    assert dp.scrape_early_racecards(s, date(2026, 11, 5), pause=0) is None
    assert not [u for u in s.got if "earlyraceid" in u]


def test_the_morning_card_still_comes_from_the_one_day_page(monkeypatch):
    called = []
    monkeypatch.setattr(dp, "uk_today", lambda: date(2026, 11, 5))
    monkeypatch.setattr(dp, "download_racecard_csv", lambda *a: None)
    monkeypatch.setattr(dp, "scrape_racecard_html", lambda s, d, **k: called.append(("one-day", d)) or None)
    monkeypatch.setattr(dp, "scrape_early_racecards", lambda s, d, **k: called.append(("early", d)) or None)
    dp.get_todays_runners(None, "u", date(2026, 11, 5))
    dp.get_todays_runners(None, "u", date(2026, 11, 6))
    assert called == [("one-day", date(2026, 11, 5)), ("early", date(2026, 11, 6))]


def test_the_night_before_run_never_writes_the_trader_s_file(monkeypatch):
    """pipeline.predict for another day writes under its own prefix only; the workflow sets it."""
    import yaml
    from pathlib import Path

    import pipeline.predict as pp
    monkeypatch.setenv("PREDICT_DATE", "2099-01-01")
    monkeypatch.delenv("PREDICTIONS_PREFIX", raising=False)
    with pytest.raises(SystemExit, match="its own PREDICTIONS_PREFIX"):
        pp.main()
    wf = yaml.safe_load((Path(__file__).resolve().parent.parent / ".github/workflows/predict-evening.yml").read_text())
    step = [s for s in wf["jobs"]["predict-evening"]["steps"] if s.get("name") == "Run predictions"][0]
    assert step["env"]["PREDICTIONS_PREFIX"] == "predictions_evening"
    assert step["env"]["RACECARDS_PREFIX"] == "racecards_evening"
    assert "PREDICT_DATE=$(TZ=Europe/London date -d tomorrow +%F)" in step["run"]
    assert [c["cron"] for c in wf[True]["schedule"]] == ["50 21 * * *"]
    assert "sync_dashboard" not in str(wf)
