"""The point-in-time harness (the owner's model brief, 18 Nov 2026, Phase 1): every served feature is under a leak
test, and the leak test itself catches a leak planted on purpose.

The leak tests already in the repository:
- tests/test_leakage.py: the deployed feature list (train_bfsp.ALL_FEATURE_COLS) built twice on a synthetic card,
  the last race's result reversed the second time; no feature of that race may move;
- tests/test_feature_blocks.py: every block in model/blocks, a day's results scrambled or blanked; none of that
  day's features may move.

Here: (1) the served model's features, all three members, are each in one of those two sets; (2) the comparison
those tests use flags a feature that reads its own race's result and one lagged by row instead of by race, and not a
feature lagged correctly; (3) the served features the synthetic card leaves empty (so their leak test passes without
looking) are named, so a new one is noticed. Synthetic frames test mechanics only; nothing is trained or evaluated.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from tests import test_leakage as L
from model import blocks

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEMBERS = ["data/models", "data/models/members/dml", "data/models/members/rx"]
#: served features the synthetic card of test_leakage leaves empty, so the flip test cannot see them move
EMPTY_ON_THE_CARD = {"front_sustainability", "track_draw_bias", "horse_draw_pref", "fr_runs_since_break",
                     "fr_break_len"}


def _served(member: str) -> list:
    path = ROOT / member / "bfsp_model_meta.json"
    if not path.exists():
        pytest.skip(f"{path} not in this checkout")
    return json.loads(path.read_text())["feature_cols"]


def _block_features() -> set:
    out = set()
    for name in blocks.names():
        out |= set(blocks.load(name).FEATURES)
    return out


def moved(base: pd.DataFrame, flipped: pd.DataFrame, cols: list) -> list:
    """The comparison of test_leakage: on the last race, the features that differ once its result is reversed."""
    last = sorted(base["raceid"].unique())[-1]
    key = ["raceid", "horse_name"]
    b = base[base["raceid"] == last].sort_values(key).reset_index(drop=True)
    f = flipped[flipped["raceid"] == last].sort_values(key).reset_index(drop=True)
    assert not b["placing_numerical"].equals(f["placing_numerical"]), "the fixture did not change the result"
    return sorted(c for c in cols if not np.allclose(b[c].fillna(-999.0), f[c].fillna(-999.0), equal_nan=True))


@pytest.mark.parametrize("member", MEMBERS)
def test_every_served_feature_is_under_a_leak_test(member):
    import train_bfsp as T
    covered = set(T.ALL_FEATURE_COLS) | _block_features()
    missing = [c for c in _served(member) if c not in covered]
    assert not missing, f"{len(missing)} served features have no leak test: {missing[:20]}"


@pytest.fixture(scope="module")
def deployed():
    return L._deployed_features_for(False), L._deployed_features_for(True)


def test_the_served_features_empty_on_the_card_are_the_known_five(deployed):
    base, _ = deployed
    blockcols = _block_features()
    served = [c for c in _served("data/models") if c not in blockcols]
    empty = {c for c in served if c in base.columns and base[c].isna().all()}
    assert empty == EMPTY_ON_THE_CARD, (
        f"newly empty on the card (their leak test would pass without looking): {sorted(empty - EMPTY_ON_THE_CARD)}; "
        f"no longer empty: {sorted(EMPTY_ON_THE_CARD - empty)}")


def _plant(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    order = d.sort_values(["race_date", "race_time", "horse_name"]).index
    d = d.loc[order]
    # reads its own race's result: the race's mean finishing position less the horse's
    d["leak_own_race"] = d.groupby("raceid")["placing_numerical"].transform("mean") - d["placing_numerical"]
    # lagged by row, not by race: on a key every runner of the race shares, the previous row is a rival
    d["leak_row_lag"] = d.groupby("track")["placing_numerical"].transform(lambda s: s.shift(1).expanding().mean())
    # lagged correctly: the horse's own previous run
    d["clean_prev_run"] = d.groupby("horse_name")["placing_numerical"].shift(1)
    return d


def test_the_leak_check_flags_planted_leaks_and_passes_a_clean_lag(deployed):
    base, flipped = deployed
    b, f = _plant(base), _plant(flipped)
    hits = moved(b, f, ["leak_own_race", "leak_row_lag", "clean_prev_run"])
    assert hits == ["leak_own_race", "leak_row_lag"], hits


def test_a_planted_leak_in_a_block_is_flagged_by_the_block_test():
    """The block test scrambles a day's results and compares that day's features; a block that reads its own day's
    finishing positions fails it, a block lagged by day passes."""
    from tests import test_feature_blocks as F
    from model.custom_metrics import CustomMetricsEngine
    base = F._history()
    rng = np.random.default_rng(5)
    run = lambda f: CustomMetricsEngine().calculate_all(f)          # noqa: E731
    a, b = run(base), run(F._scrambled(base, rng))

    def leaky(df):
        race = df["race_date"].astype(str) + df["track"] + df["race_time"].astype(str)
        return pd.to_numeric(df["placing_numerical"], errors="coerce").groupby(race).transform("rank")

    def clean(df):
        pos = pd.to_numeric(df["placing_numerical"], errors="coerce")
        day = pd.to_datetime(df["race_date"])
        by_day = pos.groupby([df["horse_name"], day]).mean()
        prev = by_day.groupby(level=0).shift(1)
        return pd.Series(prev.reindex(pd.MultiIndex.from_arrays([df["horse_name"], day])).to_numpy(), index=df.index)

    for fn, should_move in ((leaky, True), (clean, False)):
        x = a.assign(feat=fn(a)).set_index(F.KEY)["feat"].sort_index()
        y = b.assign(feat=fn(b)).set_index(F.KEY)["feat"].sort_index()
        day = x.index.get_level_values("race_date") == F.FLIP_DAY
        same = np.allclose(x[day].fillna(-999.0), y[day].fillna(-999.0))
        assert same != should_move, fn.__name__


def test_a_meeting_is_never_split_between_training_and_the_fold():
    """Folds cut by date (model/bfsp_model.fold_masks), so every runner of a meeting (date and course) is on one side:
    training (purged before the fold), the fold, or neither."""
    from model.bfsp_model import TrainConfig, fold_masks
    days = pd.date_range("2025-01-01", "2025-06-30", freq="D")
    df = pd.DataFrame([(d, t, h) for d in days for t in ("York", "Kempton") for h in range(6)],
                      columns=["race_date", "track", "h"])
    cfg = TrainConfig()
    train, val = fold_masks(df["race_date"], "2025-04-01", "2025-07-01", cfg)
    assert not (train & val).any()
    side = pd.Series(np.where(train, "train", np.where(val, "fold", "none")))
    assert (side.groupby([df["race_date"], df["track"]]).nunique() == 1).all()
    assert df.loc[train, "race_date"].max() < pd.Timestamp("2025-04-01") - pd.Timedelta(days=int(cfg.purge_days))
