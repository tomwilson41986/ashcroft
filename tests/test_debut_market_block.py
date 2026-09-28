"""model/blocks/debut_market.py by hand: how the market priced a yard's debutants and lightly raced runners."""
import numpy as np
import pandas as pd

from model.blocks import debut_market as dm


def _race(day, time, runners, result=True):
    """runners: (horse, trainer, jockey, career_runs, bsp)."""
    d = pd.Timestamp("2025-03-01") + pd.Timedelta(days=day)
    return [dict(race_date=d, race_time=time, track="York", raceid=f"{d.date()}|York|{time}", horse_name=h,
                 trainer=t, jockey_name=j, career_runs=r, race_type="Maiden", surface_type="Turf",
                 number_of_runners=len(runners), bfsp=b if result else np.nan,
                 placing_numerical=float(i + 1) if result else np.nan)
            for i, (h, t, j, r, b) in enumerate(runners)]


def _frame():
    rows = _race(0, "2.30", [("a", "T1", "J1", 0, 2.0), ("f", "T2", "J2", 0, 6.0), ("x", "T3", "J3", 5, 6.0),
                             ("y", "T3", "J3", 5, 6.0)])
    rows += _race(1, "2.30", [("b", "T1", "J1", 0, 6.0), ("p", "T3", "J3", 7, 2.0), ("q", "T3", "J3", 7, 6.0)])
    rows += _race(2, "2.30", [("c", "T1", "J2", 0, None), ("d", "T2", "J3", 1, None), ("e", "T4", "J9", 0, None),
                              ("z", "T3", "J3", 9, None)], result=False)
    return pd.DataFrame(rows)


def test_the_yards_debutants_as_the_market_rated_them():
    out = dm.build(_frame()).set_index("horse_name")
    w0, w1 = 2 ** (-2 / dm.HALFLIFE), 2 ** (-1 / dm.HALFLIFE)          # day 2 looking back at days 0 and 1
    mkt_a, mkt_f = np.log(0.5 * 4), np.log(1 / 6 * 4)                   # 2.0 and 6.0 in a book of 1 over four
    mkt_b = np.log((1 / 6) / (1 / 6 + 1 / 2 + 1 / 6) * 3)                 # 6.0 among 6.0, 2.0, 6.0
    pop = (w0 * (mkt_a + mkt_f) + w1 * mkt_b) / (2 * w0 + w1)           # every debutant on earlier days
    t1 = (w0 * mkt_a + w1 * mkt_b + dm.K * pop) / (w0 + w1 + dm.K)
    assert np.isclose(out.loc["c", "dm_tr_debut_mkt"], t1)
    assert np.isclose(out.loc["c", "dm_tr_debut_n"], w0 + w1)
    assert np.isclose(out.loc["e", "dm_tr_debut_mkt"], pop)              # a yard never seen: all debutants' level
    # T2's one debutant went off 6.0: its level is below the population's; T1's above
    assert out.loc["d", "dm_tr_debut_mkt"] < pop < out.loc["c", "dm_tr_debut_mkt"]
    # the rider: J2 rode T2's debutant f; today it rides c
    j2 = (w0 * mkt_f + dm.K * pop) / (w0 + dm.K)
    assert np.isclose(out.loc["c", "dm_jk_debut_mkt"], j2)


def test_each_runner_reads_its_own_stage_and_the_field_its_share():
    out = dm.build(_frame()).set_index("horse_name")
    assert np.isclose(out.loc["c", "dm_stage_mkt"], out.loc["c", "dm_trc_debut_mkt"])       # a debutant
    assert np.isclose(out.loc["d", "dm_stage_mkt"], out.loc["d", "dm_tr_second_mkt"])       # a second run
    assert np.isnan(out.loc["z", "dm_stage_mkt"])                                           # an exposed horse
    assert np.isclose(out.loc["c", "dm_race_debut_share"], 2 / 4)                           # c and e of four
    assert np.isclose(out.loc["c", "dm_race_unexposed_share"], 3 / 4)                       # and d


def test_the_day_itself_never_counts():
    out = dm.build(_frame()).set_index("horse_name")
    assert out.loc["a", "dm_tr_debut_n"] == 0 and out.loc["f", "dm_tr_debut_n"] == 0       # day 0: nothing before
    df = _frame()
    df.loc[df.horse_name == "b", "bfsp"] = 1.5                                              # day 1's price changed
    again = dm.build(df).set_index("horse_name")
    assert np.isclose(again.loc["b", "dm_tr_debut_mkt"], out.loc["b", "dm_tr_debut_mkt"])   # b's own day: unmoved
    assert not np.isclose(again.loc["c", "dm_tr_debut_mkt"], out.loc["c", "dm_tr_debut_mkt"])   # day 2 reads it
