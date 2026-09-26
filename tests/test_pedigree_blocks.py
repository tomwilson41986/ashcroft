"""model/blocks/dam_line.py and model/blocks/sire_apt.py by hand: breeding read from earlier days only."""
import numpy as np
import pandas as pd

from model.blocks import dam_line, sire_apt


def _card():
    rows = []

    def run(day, horse, dam, sire, place, n=6):
        d = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day)
        rows.append(dict(raceid=f"{d.date()}|York|{day}", race_date=d, race_time="2.30", track="York",
                         horse_name=horse, dam=dam, stallion=sire, dam_stallion="DS", placing_numerical=place,
                         number_of_runners=n, official_rating=70, horse_age=4, dist_furlongs=8.0,
                         going_description="Good", prize_money=5000, major="", race_name="Maiden",
                         race_class="Class 5", race_type="Maiden", surface_type="Turf"))
    # the dam M1's first two foals win and run second; M2's foal is always last
    for k, (h, dam, place) in enumerate([("a1", "M1", 1), ("a2", "M1", 2), ("b1", "M2", 6), ("a1", "M1", 1),
                                         ("b1", "M2", 6)]):
        run(k * 7, h, dam, "S1", place)
    run(60, "a3", "M1", "S1", np.nan)                    # today: a third M1 foal on debut
    run(60, "b2", "M2", "S1", np.nan)                    # and a second M2 foal
    df = pd.DataFrame(rows)
    df["career_runs"] = df.groupby("horse_name").cumcount()          # runs before this one
    df["won"] = (df["placing_numerical"] == 1).astype(float).where(df["placing_numerical"].notna())
    return df


def test_the_dam_line_reads_the_siblings_and_not_the_horse():
    out = dam_line.build(_card()).set_index(["race_date", "horse_name"])
    today = pd.Timestamp("2025-03-02")
    a3, b2 = out.loc[(today, "a3")], out.loc[(today, "b2")]
    assert a3["sib_count"] == 2 and b2["sib_count"] == 1              # siblings that have run, not runs
    assert a3["dam_winners"] >= 1 and b2["dam_winners"] == 0
    assert a3["sib_mean_nmfp"] > b2["sib_mean_nmfp"]
    assert a3["dam_sr_shrunk"] > b2["dam_sr_shrunk"]
    # a horse is not its own sibling: on a1's second run (22 Jan) the only sibling that has run is a2
    second = out.loc[(pd.Timestamp("2025-01-22"), "a1")]
    assert second["sib_count"] == 1


def test_a_days_own_results_move_neither_block():
    df = _card()
    blanked = df.copy()
    last = blanked["race_date"] == blanked["race_date"].max()
    for col in ("placing_numerical", "won"):
        blanked.loc[last, col] = np.nan
    flipped = df.copy()
    flipped.loc[last, "placing_numerical"] = [1.0, 2.0]
    flipped["won"] = (flipped["placing_numerical"] == 1).astype(float)
    for block in (dam_line, sire_apt):
        a = block.build(df).loc[last, block.FEATURES]
        for other in (blanked, flipped):
            b = block.build(other).loc[last, block.FEATURES]
            pd.testing.assert_frame_equal(a, b, check_exact=True)
