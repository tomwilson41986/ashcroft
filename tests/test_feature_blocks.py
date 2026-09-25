"""Every drop-in block in model/blocks, through the same tests, automatically.

A block is found by being a module in model/blocks; nothing registers it. Each is
built on the metrics engine's output for a synthetic history -- three courses,
flat, all-weather and jumps, several goings and stall positions, non-finishers,
recurring horses, trainers and jockeys -- and must:

- keep the contract (FEATURES added, none of them post-race, the row count kept);
- move none of a day's features when that day's results, margins, comments and
  prices are scrambled, or blanked as they are on the 06:00 card;
- move some later day's features when they are (the block reads history at all;
  a block with no results-driven feature says so with READS_RESULTS = False);
- give each row the same values whatever order the rows arrive in;
- produce something: no feature entirely empty over the history.

Synthetic frames test mechanics only; nothing is trained or evaluated on them.
"""
import numpy as np
import pandas as pd
import pytest

from model import blocks
from model.custom_metrics import CustomMetricsEngine

KEY = ["race_date", "track", "race_time", "horse_name"]
RESULTS = ["placing_numerical", "place", "total_dst_bt", "distbt", "comment", "comptime",
           "comptime_numeric", "bfsp", "bfsp_place"]
COMMENTS = ["led, kept on", "made all", "prominent, weakened", "tracked leaders, one pace",
            "chased leaders, no extra", "mid-division, stayed on", "held up, headway 2f out",
            "in rear, never dangerous", "towards rear, ran on late", "pulled up"]
TRACKS = [("York", "Turf", "Flat", 6.0, "Handicap"), ("Kempton", "Standard", "Flat", 7.0, "Handicap"),
          ("Cheltenham", "Turf", "Hurdle", 16.0, "Handicap Hurdle")]
FLIP_DAY = pd.Timestamp("2025-02-20")


def _history(n_days=60, seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)
        for t, (track, surface, rtype, dist, rname) in enumerate(TRACKS):
            if rng.random() < 0.3 and day != FLIP_DAY:        # every course races on the flip day
                continue
            n = int(rng.integers(6, 11))
            horses = rng.choice(70, n, replace=False) + 100 * t      # each course its own pool
            order = rng.permutation(n)
            finishers = n - int(rng.random() < 0.3)                  # now and then one pulls up
            going = rng.choice(["Good", "Soft", "Good to Firm", "Heavy"]) if surface == "Turf" else "Standard"
            cum = 0.0
            for rank, i in enumerate(order):
                h = int(horses[i])
                fin = rank < finishers
                if fin and rank > 0:
                    cum += float(rng.choice([0.1, 0.5, 1.0, 2.0, 4.0, 9.0]))
                rows.append({
                    "race_date": day, "race_time": f"{1 + t}.{15 * (d % 3):02d}", "track": track,
                    "horse_name": f"h{h}", "jockey_name": f"j{int(rng.integers(0, 15))}",
                    "trainer": f"T{h % 9}", "stall": i + 1 if rtype == "Flat" else np.nan,
                    "draw": i + 1 if rtype == "Flat" else np.nan, "number_of_runners": n,
                    "dist_furlongs": dist, "race_distance": f"{int(dist)}f", "yards": int(dist * 220),
                    "going_description": going, "race_type": rname, "race_code": "F" if rtype == "Flat" else "H",
                    "race_class": str(int(rng.integers(2, 7))), "race_name": rname, "major": "N",
                    "race_restrictions_age": "3yo+", "prize_money": str(int(rng.integers(3, 30)) * 1000),
                    "placing_numerical": rank + 1 if fin else np.nan, "place": str(rank + 1) if fin else "PU",
                    "total_dst_bt": ("0" if rank == 0 else f"{cum:.2f}") if fin else "",
                    "distbt": "" if rank == 0 or not fin else "1",
                    "comment": COMMENTS[-1] if not fin else COMMENTS[int(rng.integers(0, len(COMMENTS) - 1))],
                    "comptime_numeric": 72.0 + rng.normal(), "comptime": "1:12.00",
                    "official_rating": 60 + h % 40, "median_or": 75, "max_or_in_race": 99,
                    "pounds": 120 + int(rng.integers(0, 20)), "jockeys_claim": 0, "card_no": i + 1,
                    "odds": float(rng.uniform(2, 30)), "fav": "", "bfsp": float(rng.uniform(1.5, 60)),
                    "bfsp_place": float(rng.uniform(1.1, 10)), "plcs_paid": 3, "bf_plcs_paid": 3,
                    "horse_age": 3 + h % 6, "horse_sex": "G", "days_since_lr": int(rng.integers(7, 90)),
                    "career_runs": 5 + h % 20, "stallion": f"S{h % 7}", "dam": f"D{h % 11}",
                    "dam_stallion": f"S{(h + 3) % 7}", "surface_type": surface, "horse_prizewin": "1000",
                    "headgear": "b" if h % 5 == 0 else "", "rail_move": "", "track_direction": "L",
                    "stall_positioning": rng.choice(["Stands", "Far", "Centre"]) if rtype == "Flat" else "",
                    "horse_code": f"hc{h}",
                })
    return pd.DataFrame(rows)


def _scrambled(df: pd.DataFrame, rng) -> pd.DataFrame:
    out = df.copy()
    on = out["race_date"] == FLIP_DAY
    for _, idx in out[on].groupby(["track", "race_time"]).groups.items():
        idx = list(idx)
        p = rng.permutation(len(idx))
        out.loc[idx, RESULTS] = out.loc[idx, RESULTS].to_numpy()[p]
    return out


def _blanked(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.loc[out["race_date"] == FLIP_DAY, RESULTS] = np.nan
    return out


@pytest.fixture(scope="module")
def engine_frames():
    base = _history()
    assert (base["race_date"] == FLIP_DAY).sum() > 10
    rng = np.random.default_rng(5)
    run = lambda f: CustomMetricsEngine().calculate_all(f)          # noqa: E731
    return {"base": run(base), "scrambled": run(_scrambled(base, rng)), "blanked": run(_blanked(base))}


def _block_on(frame, name):
    out, cols = blocks.attach(frame.copy(), [name])
    return out.set_index(KEY)[cols].sort_index()


BLOCKS = blocks.names()


def test_the_package_has_blocks():
    assert "form_variants" in BLOCKS


@pytest.mark.parametrize("name", BLOCKS)
def test_the_contract(name, engine_frames):
    mod = blocks.load(name)
    assert mod.__doc__ and len(mod.__doc__.strip()) > 40, "say what the block measures"
    frame = engine_frames["base"]
    out, cols = blocks.attach(frame.copy(), [name])
    assert cols == list(mod.FEATURES) and set(cols) <= set(out.columns)
    assert len(out) == len(frame)
    assert not set(cols) & set(frame.columns), "a block adds columns, it does not overwrite the engine's"
    assert not set(cols) & blocks.post_race(name)
    empty = [c for c in cols if not np.isfinite(pd.to_numeric(out[c], errors="coerce")).any()]
    assert not empty, f"entirely empty on the history: {empty[:10]}"


@pytest.mark.parametrize("change", ["scrambled", "blanked"])
@pytest.mark.parametrize("name", BLOCKS)
def test_a_days_own_results_move_none_of_its_features(name, change, engine_frames):
    a = _block_on(engine_frames["base"], name)
    b = _block_on(engine_frames[change], name)
    day = a.index.get_level_values("race_date") == FLIP_DAY
    assert day.sum() > 10
    pd.testing.assert_frame_equal(a[day], b[day], check_exact=True)


@pytest.mark.parametrize("name", BLOCKS)
def test_later_days_read_the_days_results(name, engine_frames):
    if not getattr(blocks.load(name), "READS_RESULTS", True):
        pytest.skip("the block reads no results")
    a = _block_on(engine_frames["base"], name)
    b = _block_on(engine_frames["scrambled"], name)
    later = a.index.get_level_values("race_date") > FLIP_DAY
    moved = [c for c in a.columns
             if not np.array_equal(a.loc[later, c].to_numpy(float), b.loc[later, c].to_numpy(float), equal_nan=True)]
    assert moved, "no feature of any later day moved: the block does not read history"


@pytest.mark.parametrize("name", BLOCKS)
def test_the_row_order_does_not_matter(name, engine_frames):
    frame = engine_frames["base"]
    a = _block_on(frame, name)
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    b = _block_on(shuffled, name)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", [n for n in BLOCKS if getattr(blocks.load(n), "READS", None) is not None])
def test_reads_names_every_column_the_block_reads(name, engine_frames):
    """The live path copies only a block's READS; built on those alone it must give
    exactly what it gives on the whole frame (a column read but not declared comes
    back as NaN or a KeyError, and fails here)."""
    mod = blocks.load(name)
    frame, _ = blocks.attach(engine_frames["base"].copy(), list(getattr(mod, "AFTER", ())))
    assert all(c in frame.columns for c in mod.READS), [c for c in mod.READS if c not in frame.columns]
    full = mod.build(frame.copy())
    narrow = mod.build(frame[list(dict.fromkeys(mod.READS))].copy())
    for c in mod.FEATURES:
        np.testing.assert_array_equal(pd.to_numeric(full[c]).to_numpy(float),
                                      pd.to_numeric(narrow[c]).to_numpy(float), err_msg=c)


@pytest.mark.parametrize("name", BLOCKS)
def test_what_a_block_declares_for_serving_is_real(name):
    """ENGINE names engine flags, AFTER names blocks: a typo would leave the live
    path building a block on columns nobody made."""
    import inspect
    from model.custom_metrics import CustomMetricsEngine as E
    mod = blocks.load(name)
    flags = set(inspect.signature(E.__init__).parameters) - {"self"}
    assert set(getattr(mod, "ENGINE", ())) <= flags
    assert set(getattr(mod, "AFTER", ())) <= set(BLOCKS) - {name}
