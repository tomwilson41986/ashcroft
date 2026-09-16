"""The opt-in feature blocks are wired into training, and what they emit is legal.

Each block was built in isolation by a different worker. This exercises the
chain `train_bfsp.py` actually runs — run primitives, then pedigree, then
connections, then the odds block — and checks the two properties that matter at
the seam: the names a block promises are really on the frame, and none of them
describes the race being predicted.
"""

import numpy as np
import pandas as pd
import pytest

from model.connections import add_connection_features
from model.odds_metrics import ODDS_FEATURES, add_odds_metrics
from model.pedigree import add_pedigree_features
from model.primitives import add_run_primitives


def _history(n_days=40, races_per_day=3, runners=8, seed=0):
    """A small card history with real sires, dams, yards and prices."""
    rng = np.random.default_rng(seed)
    sires = [f"SIRE{i}" for i in range(6)]
    dams = [f"DAM{i}" for i in range(30)]
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for r in range(races_per_day):
            winner = int(rng.integers(0, runners))
            order = list(rng.permutation([i for i in range(runners) if i != winner]))
            pos = {winner: 1, **{h: k + 2 for k, h in enumerate(order)}}
            for i in range(runners):
                rows.append({
                    "race_date": day.strftime("%Y-%m-%d"), "race_time": f"{1 + r}.30",
                    "track": f"T{r}", "raceid": f"{day.date()}_{r}",
                    "horse_name": f"h{rng.integers(0, 120)}",
                    # a yard and a sire deliberately double up inside races
                    "trainer": f"TR{i % 4}", "jockey_name": f"J{i % 5}",
                    "stallion": sires[i % len(sires)], "dam": dams[i % len(dams)],
                    "dam_stallion": sires[(i + 1) % len(sires)],
                    "placing_numerical": pos[i], "number_of_runners": runners,
                    "total_dst_bt": "0" if pos[i] == 1 else f"{pos[i] - 1}",
                    "dist_furlongs": 6.0 + r, "going_description": "Good",
                    "race_type": "Flat", "race_code": "F", "race_class": "4",
                    "comment": "led throughout" if i % 3 == 0 else "held up in rear",
                    "comptime_numeric": 72.0 + r, "stall": i + 1,
                    "official_rating": 70 + (i * 2), "won": pos[i] == 1,
                    "bfsp": float(2 + i), "horse_age": 4, "horse_sex": "G",
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def built():
    d = add_run_primitives(_history())
    d, ped = add_pedigree_features(d)
    d, con = add_connection_features(d)
    d = add_odds_metrics(d)
    return d, ped, con


def test_every_promised_feature_is_actually_on_the_frame(built):
    d, ped, con = built
    assert len(ped) > 20 and len(con) > 20
    for name, feats in (("pedigree", ped), ("connections", con)):
        missing = [c for c in feats if c not in d.columns]
        assert not missing, f"{name} promised {missing} and did not deliver them"
        non_numeric = [c for c in feats if not pd.api.types.is_numeric_dtype(d[c])]
        assert not non_numeric, f"{name} emitted non-numeric {non_numeric}"
    assert [c for c in ODDS_FEATURES if c in d.columns], "the odds block produced nothing"


def test_no_block_emits_a_feature_that_describes_the_race_being_predicted(built):
    from train_bfsp import assert_no_post_race_features

    d, ped, con = built
    assert_no_post_race_features(ped)
    assert_no_post_race_features(con)
    assert_no_post_race_features([c for c in ODDS_FEATURES if c in d.columns])


def test_the_blocks_compose_with_the_production_feature_list(built):
    """The wiring extends one list; a name collision would silently shadow."""
    from train_bfsp import ALL_FEATURE_COLS, assert_no_post_race_features

    d, ped, con = built
    combined = list(ALL_FEATURE_COLS) + ped + con
    assert len(set(combined)) == len(combined), "a block reuses a production feature name"
    assert_no_post_race_features(combined)


def test_connection_and_pedigree_features_do_not_vary_inside_a_race(built):
    """A yard and a sire both run two horses in every race here, so a row-wise
    lag would show one of them the other's result."""
    d, ped, con = built
    for col in ("trainer_sr_shrunk", "sire_nmfp_shrunk"):
        if col not in d.columns:
            continue
        key = "trainer" if col.startswith("trainer") else "stallion"
        spread = d.groupby(["raceid", key])[col].nunique(dropna=True)
        assert spread.max() <= 1, f"{col} varies within one race for one {key}"
