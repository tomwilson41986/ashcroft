"""The predictions updated for the runners withdrawn since they were made (predict_bfsp_today.without_non_runners).

Mechanics only: small made-up cards, no model and no database."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from predict_bfsp_today import read_card, update_for_non_runners, without_non_runners


def _predictions() -> pd.DataFrame:
    rows = []
    for track, time, prices in (("Down Royal", "1:24", [2.5, 4.0, 5.0, 20.0]),
                                ("Wolverhampton", "6:30", [3.0, 3.0, 6.0])):
        p = 1.0 / np.array(prices)
        p = p / p.sum()
        for i, q in enumerate(p):
            rows.append({"race_date": "2026-09-28", "race_time": time, "track": track,
                         "horse_name": f"{track[:3]} {i} (IRE)", "number_of_runners": len(prices),
                         "predicted_bfsp": 1.0 / q, "predicted_win_prob_norm": q})
    df = pd.DataFrame(rows)
    df["model_rank"] = df.groupby(["track", "race_time"])["predicted_bfsp"].rank(method="min").astype(int)
    return df


def _card(pred: pd.DataFrame, drop=()) -> pd.DataFrame:
    return pred.loc[~pred["horse_name"].isin(drop), ["race_time", "track", "horse_name"]].reset_index(drop=True)


def test_a_non_runner_leaves_and_its_race_is_back_to_a_book_of_one():
    pred = _predictions()
    out, nr, unpriced = without_non_runners(pred, _card(pred, drop=["Dow 0 (IRE)"]))
    assert list(nr["horse_name"]) == ["Dow 0 (IRE)"] and len(unpriced) == 0
    race = out[out["track"] == "Down Royal"]
    assert len(race) == 3 and race["predicted_win_prob_norm"].sum() == pytest.approx(1.0)
    assert np.allclose(race["predicted_bfsp"] * race["predicted_win_prob_norm"], 1.0)
    assert set(race["number_of_runners"]) == {3}
    assert list(race.sort_values("predicted_bfsp")["model_rank"]) == [1, 2, 3]


def test_the_rest_of_the_race_keeps_its_order_and_rises_by_the_non_runner_s_share():
    pred = _predictions()
    share = pred.loc[pred["horse_name"] == "Dow 0 (IRE)", "predicted_win_prob_norm"].item()
    out, nr, _ = without_non_runners(pred, _card(pred, drop=["Dow 0 (IRE)"]))
    assert nr["share_of_book"].item() == pytest.approx(share)
    before = pred[(pred["track"] == "Down Royal") & (pred["horse_name"] != "Dow 0 (IRE)")].set_index("horse_name")
    after = out[out["track"] == "Down Royal"].set_index("horse_name")
    assert np.allclose(after["predicted_win_prob_norm"], before["predicted_win_prob_norm"] / (1 - share))


def test_a_race_that_lost_no_one_is_left_exactly_as_it_was():
    pred = _predictions()
    out, _, _ = without_non_runners(pred, _card(pred, drop=["Dow 3 (IRE)"]))
    cols = ["horse_name", "number_of_runners", "predicted_bfsp", "predicted_win_prob_norm", "model_rank"]
    pd.testing.assert_frame_equal(out.loc[out["track"] == "Wolverhampton", cols].reset_index(drop=True),
                                  pred.loc[pred["track"] == "Wolverhampton", cols].reset_index(drop=True))


def test_a_race_missing_from_the_card_is_not_a_field_of_non_runners():
    pred = _predictions()
    card = _card(pred)
    out, nr, _ = without_non_runners(pred, card[card["track"] != "Wolverhampton"])
    assert len(nr) == 0 and len(out) == len(pred)


def test_the_card_s_spelling_of_the_time_and_name_is_matched_loosely():
    pred = _predictions()
    card = _card(pred, drop=["Wol 2 (IRE)"])
    card["race_time"] = card["race_time"].str.replace(":", ".")
    card["horse_name"] = card["horse_name"].str.upper().str.replace(" ", "  ")
    card["track"] = " " + card["track"].str.lower()
    out, nr, unpriced = without_non_runners(pred, card)
    assert list(nr["horse_name"]) == ["Wol 2 (IRE)"] and len(unpriced) == 0 and len(out) == len(pred) - 1


def test_a_runner_the_predictions_never_priced_is_reported_not_invented():
    pred = _predictions()
    card = pd.concat([_card(pred), pd.DataFrame([{"race_time": "6:30", "track": "Wolverhampton",
                                                  "horse_name": "Late Reserve"}])], ignore_index=True)
    out, nr, unpriced = without_non_runners(pred, card)
    assert list(unpriced["horse_name"]) == ["Late Reserve"] and len(nr) == 0 and len(out) == len(pred)


def test_two_non_runners_in_a_race_and_one_in_another():
    pred = _predictions()
    out, nr, _ = without_non_runners(pred, _card(pred, drop=["Dow 0 (IRE)", "Dow 2 (IRE)", "Wol 0 (IRE)"]))
    assert len(nr) == 3 and len(out) == len(pred) - 3
    books = out.groupby("track")["predicted_win_prob_norm"].sum()
    assert np.allclose(books, 1.0)
    assert out.groupby("track")["number_of_runners"].first().to_dict() == {"Down Royal": 2, "Wolverhampton": 2}


def test_only_one_day_at_a_time():
    pred = _predictions()
    pred.loc[0, "race_date"] = "2026-09-27"
    with pytest.raises(ValueError):
        without_non_runners(pred, _card(pred))


def test_the_update_writes_the_predictions_and_the_non_runners_beside_them(tmp_path):
    pred = _predictions()
    base = tmp_path / "predictions.csv"
    pred.to_csv(base, index=False)
    nr = update_for_non_runners(str(base), str(tmp_path / "now.csv"), _card(pred, drop=["Wol 1 (IRE)"]))
    now = pd.read_csv(tmp_path / "now.csv", dtype={"race_time": str})
    assert len(now) == len(pred) - 1 and list(nr["horse_name"]) == ["Wol 1 (IRE)"]
    assert list(pd.read_csv(tmp_path / "now_non_runners.csv")["horse_name"]) == ["Wol 1 (IRE)"]
    assert now["race_time"].tolist()[0] == "1:24" and now["model_rank"].dtype.kind == "i"


def test_the_saved_html_card_is_read_as_the_06_00_fetch_reads_it(tmp_path):
    html = """<html><body><span>1.20 Curragh(3 runners)</span><span>Class 1, 6f , Good, 2yo, Win: £10,000</span>
    <span>Some Maiden</span>
    <table><tr><td>No.</td><td>Form</td><td>Days</td><td>Horse</td><td>Age</td><td>Weight</td><td>Headgear</td>
    <td>Jockey</td><td>Trainer</td><td>Stall</td><td>OR</td><td>Odds</td></tr>
    <tr><td>6</td><td>none</td><td>-</td><td><a href="horses.php?id=1">Shake It Out</a></td><td>2</td><td>9-7</td>
    <td></td><td>Whelan, R P</td><td>Cotter, Kieran P</td><td>10</td><td>0</td><td>50/1</td></tr>
    <tr><td>7</td><td>2</td><td>17</td><td><a href="horses.php?id=2">Known Horse</a></td><td>2</td><td>9-7</td>
    <td></td><td>Coen, Ben M</td><td>Murphy, Daniel</td><td>8</td><td>0</td><td>10/1</td></tr></table>
    </body></html>"""
    page = tmp_path / "onedayracecards.html"
    page.write_text(html, encoding="utf-8")
    card = read_card(str(page), date(2026, 9, 28))
    assert card[["race_time", "track", "horse_name"]].values.tolist() == [
        ["1:20", "Curragh", "Shake It Out"], ["1:20", "Curragh", "Known Horse"]]


# --- horses the card still lists that are not running ------------------------------------------

@pytest.mark.parametrize("jockey,status", [
    ("Lordan, W M", "runner"), ("Cusack, C(7)", "runner"), ("Tucker, Charlie(7)", "runner"),
    ("", "withdrawn"), ("   ", "withdrawn"), ("(7)", "withdrawn"), (" (3) ", "withdrawn"),
    (None, "withdrawn"), (float("nan"), "withdrawn"), ("NR", "withdrawn"),
    ("RESERVE", "reserve"), ("Reserve", "reserve"),
])
def test_a_card_row_s_jockey_says_whether_it_runs(jockey, status):
    from daily_predictions import card_status
    assert card_status(jockey) == status


def _card_with_jockeys(pred: pd.DataFrame, jockeys: dict) -> pd.DataFrame:
    card = _card(pred)
    card["jockey_name"] = [jockeys.get(h, "Rider, A") for h in card["horse_name"]]
    card["number_of_runners"] = card.groupby(["track", "race_time"])["horse_name"].transform("size")
    return card


def test_a_horse_listed_without_a_jockey_is_a_non_runner():
    pred = _predictions()
    card = _card_with_jockeys(pred, {"Dow 1 (IRE)": "", "Dow 2 (IRE)": "(7)"})
    out, nr, unpriced = without_non_runners(pred, card)
    assert set(nr["horse_name"]) == {"Dow 1 (IRE)", "Dow 2 (IRE)"} and len(unpriced) == 0
    assert set(nr["why"]) == {"no jockey on the card"}
    race = out[out["track"] == "Down Royal"]
    assert len(race) == 2 and race["predicted_win_prob_norm"].sum() == pytest.approx(1.0)


def test_a_reserve_not_yet_in_is_left_out_and_said_to_be_one():
    pred = _predictions()
    card = _card_with_jockeys(pred, {"Wol 2 (IRE)": "RESERVE"})
    card = card[card["horse_name"] != "Dow 3 (IRE)"]
    _, nr, _ = without_non_runners(pred, card)
    assert dict(zip(nr["horse_name"], nr["why"])) == {"Dow 3 (IRE)": "off the card",
                                                      "Wol 2 (IRE)": "reserve, not in"}


def test_the_field_is_counted_again_without_them():
    from daily_predictions import declared_runners
    pred = _predictions()
    card = declared_runners(_card_with_jockeys(pred, {"Dow 0 (IRE)": "", "Wol 1 (IRE)": "RESERVE"}))
    assert len(card) == len(pred) - 2
    assert card.groupby("track")["number_of_runners"].first().to_dict() == {"Down Royal": 3, "Wolverhampton": 2}


def test_a_card_that_names_no_jockeys_at_all_is_left_as_it_is():
    from daily_predictions import declared_runners
    pred = _predictions()
    card = _card(pred).assign(jockey_name="")
    assert len(declared_runners(card)) == len(card)


_ROW = ("<tr><td>{no}</td><td>1</td><td>9</td><td><a href='horses.php?id={no}'>{horse}</a></td><td>3</td>"
        "<td>9-2</td><td></td><td><a href='jockeys.php?id={jid}'>{jockey}</a>{claim}</td><td>Some, Trainer</td>"
        "<td>{no}</td><td>70</td><td>5/1</td></tr>")


def _html_card(rows) -> str:
    head = ("<tr><td>No.</td><td>Form</td><td>Days</td><td>Horse</td><td>Age</td><td>Weight</td>"
            "<td>Headgear</td><td>Jockey</td><td>Trainer</td><td>Stall</td><td>OR</td><td>Odds</td></tr>")
    body = "".join(_ROW.format(no=i + 1, horse=h, jid=j, jockey=jk, claim=c) for i, (h, j, jk, c) in enumerate(rows))
    return ("<html><body><span>1.55 Cork(4 runners)</span><span>Class 1, 5f , Yielding, 2yo, Win: £10,000</span>"
            f"<span>A Maiden</span><table>{head}{body}</table></body></html>")


def test_the_card_page_leaves_out_a_horse_whose_jockey_is_blanked(tmp_path):
    from daily_predictions import scrape_racecard_html
    html = _html_card([("Runs Today", 3865, "Lordan, W M", ""), ("Sapphire Sun (IRE)", 0, "", ""),
                       ("Its Life", 0, "", " (7)"), ("Watercraft (IRE)", 0, "RESERVE", "")])

    class _Page:
        text = html
        def raise_for_status(self): pass

    class _Saved:
        def get(self, url): return _Page()

    card = scrape_racecard_html(_Saved(), date(2026, 9, 29))
    assert list(card["horse_name"]) == ["Runs Today"] and list(card["number_of_runners"]) == [1]
    kept = scrape_racecard_html(_Saved(), date(2026, 9, 29), keep_withdrawn=True)
    assert len(kept) == 4
    page = tmp_path / "onedayracecards.html"
    page.write_text(html, encoding="utf-8")
    pred = pd.DataFrame({"race_date": "2026-09-29", "race_time": "1:55", "track": "Cork",
                         "horse_name": list(kept["horse_name"]), "predicted_bfsp": [2.0, 4.0, 8.0, 8.0]})
    pred["predicted_win_prob_norm"] = (1 / pred["predicted_bfsp"]) / (1 / pred["predicted_bfsp"]).sum()
    out, nr, _ = without_non_runners(pred, read_card(str(page), date(2026, 9, 29)))
    assert list(out["horse_name"]) == ["Runs Today"] and out["predicted_bfsp"].item() == pytest.approx(1.0)
    assert list(nr["why"]) == ["no jockey on the card", "no jockey on the card", "reserve, not in"]
