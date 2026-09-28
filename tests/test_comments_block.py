"""The comments block against the module it reads the classes from, and by hand.

model/comment_features.add_comment_features is the reference: lagged by the
horse's own runs with pandas shift and rolling. The block computes the same
readings with cumulative sums over earlier days; on a history with one run a
day per horse the two agree. Synthetic frames test mechanics only.
"""
import numpy as np
import pandas as pd

from model.blocks import comments as block
from model.comment_features import add_comment_features

TEXTS = ["led, kept on", "hampered 2f out, stayed on", "slowly away, in rear", "keen, prominent, weakened",
         "raced wide, one pace", "not knocked about, shaped with promise", "pulled up", "won easily",
         "lost a shoe, tailed off", "switched right, ran on", None, ""]


def _history(seed=3, days=40, horses=25):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        day = pd.Timestamp("2025-03-01") + pd.Timedelta(days=d)
        runners = rng.choice(horses, int(rng.integers(5, 10)), replace=False)
        n = len(runners)
        finishers = n - int(rng.random() < 0.3)
        cum = 0.0
        for rank, h in enumerate(runners):
            fin = rank < finishers
            if fin and rank > 0:
                cum += float(rng.choice([0.2, 1.0, 3.0, 6.0]))
            rows.append({"raceid": f"r{d}", "race_date": day, "race_time": "2.30", "track": "York",
                         "horse_name": f"h{h}", "number_of_runners": n,
                         "placing_numerical": rank + 1 if fin else np.nan,
                         "total_dst_bt": ("0" if rank == 0 else f"{cum:.2f}") if fin else "",
                         "comment": TEXTS[int(rng.integers(0, len(TEXTS)))] if fin else "pulled up"})
    return pd.DataFrame(rows)


def test_the_block_matches_the_reference_on_one_run_a_day():
    df = _history()
    ref, _ = add_comment_features(df)
    got = block.build(df)
    pairs = {f"cm_lr_{k}": f"LR_c_{k}" for k in block.CLASSES}
    pairs.update({f"cm_l{w}_{k}": f"L{w}_c_{k}" for w in block.WINDOWS for k in block.CLASSES})
    pairs.update({"cm_excuse_close": "LR_excuse_close", "cm_excuse_nfp": "LR_excuse_nfp",
                  "cm_noexcuse_poor": "LR_noexcuse_poor", "cm_tender_close": "LR_tender_close",
                  "cm_finished_well_nfp": "LR_finished_well_nfp", "cm_runs_since_trouble": "runs_since_trouble"})
    for mine, theirs in pairs.items():
        np.testing.assert_allclose(got[mine].to_numpy(float), ref[theirs].to_numpy(float),
                                   rtol=0, atol=1e-12, equal_nan=True, err_msg=mine)
    # the reference scores a non-finisher's easy win as 0; the block leaves it unknown
    known = got["cm_easy_win"].notna()
    np.testing.assert_array_equal(got.loc[known, "cm_easy_win"], ref.loc[known, "LR_easy_win"])
    assert known.sum() > 50


def test_by_hand_and_a_same_day_duplicate_reads_nothing_of_the_first():
    rows = [("2025-01-01", "2.00", "A", 1, "hampered, stayed on", "1.5"),
            ("2025-01-08", "2.00", "A", 3, "led, weakened", "4"),
            ("2025-01-15", "2.00", "A", 2, "keen, kept on", "0.5"),
            ("2025-01-15", "4.00", "A", 1, "won easily", "0"),       # a duplicate day in the feed
            ("2025-01-22", "2.00", "A", np.nan, "", "")]              # today: nothing known yet
    df = pd.DataFrame(rows, columns=["race_date", "race_time", "horse_name", "placing_numerical", "comment",
                                     "total_dst_bt"])
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["raceid"] = [f"r{i}" for i in range(len(df))]
    df["track"] = "York"
    df["number_of_runners"] = 8
    out = block.build(df).reset_index(drop=True)
    # the 15th's two rows read the 1st and the 8th only
    for i in (2, 3):
        assert out.loc[i, "cm_lr_weakened"] == 1.0 and out.loc[i, "cm_lr_trouble"] == 0.0
        assert out.loc[i, "cm_l3_trouble"] == 0.5 and out.loc[i, "cm_runs_since_trouble"] == 2.0
        assert out.loc[i, "cm_l3_keen"] == 0.0
    # today reads the latest row of the 15th (the 4.00) as its last run, and all four in the window
    assert out.loc[4, "cm_lr_easy_win"] == 1.0 and out.loc[4, "cm_easy_win"] == 1.0
    assert out.loc[4, "cm_l6_keen"] == 0.25 and out.loc[4, "cm_l3_trouble"] == 0.0
    assert out.loc[4, "cm_runs_since_trouble"] == 4.0
    # the first run has no history
    assert out.loc[0, block.FEATURES].isna().all()
