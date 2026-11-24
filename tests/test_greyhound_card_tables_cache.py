"""greyhound/live.py card_table_columns: the card's table columns built once a day and read again while nothing they
depend on changes (the card, the history, the code, the stored constants and inputs)."""

from datetime import date

import pandas as pd

from greyhound import live, pit
from sources.common import Store


def card(removed=()):
    rows = [{"race_id": r, "dog_id": 10 * r + t, "trap": t, "market_id": f"1.{r}", "selection_id": 100 * r + t,
             "race_class": "A5", "distance_m": 480.0} for r in (1, 2) for t in (1, 2, 3, 4)]
    c = pd.DataFrame(rows)
    return c[~c.dog_id.isin(removed)].reset_index(drop=True)


def builder(calls):
    prefix = pit.module("ability").PREFIX

    def build(store, day, crd, fits, lag):
        calls.append(len(crd))
        k = crd[pit.KEY].copy()
        return {"ability": k.assign(**{f"{prefix}x": k.dog_id * 0.5, "other": 1.0})}
    return build, f"{prefix}x"


def test_built_once_then_read_while_nothing_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(pit, "stored_fits", lambda store, names: {})
    store, calls = Store(root=tmp_path), []
    build, col = builder(calls)
    day, c = date(2026, 11, 24), card()
    w1, how1, k1 = live.card_table_columns(store, day, c, c, "h1", build=build)
    w2, how2, k2 = live.card_table_columns(store, day, c, c, "h1", build=build)
    assert (how1, how2) == ("built", "cached") and k1 == k2 and calls == [8]
    pd.testing.assert_frame_equal(w1, w2)
    assert list(w1.columns) == pit.KEY + [col]                  # only the table's own prefixed columns
    assert (w1[col] == w1.dog_id * 0.5).all()


def test_a_removed_runner_a_new_day_of_results_or_a_new_input_builds_again(tmp_path, monkeypatch):
    monkeypatch.setattr(pit, "stored_fits", lambda store, names: {})
    store, calls = Store(root=tmp_path), []
    build, _ = builder(calls)
    day = date(2026, 11, 24)
    live.card_table_columns(store, day, card(), card(), "h1", build=build)
    c2 = card(removed=(12,))
    _, how, _ = live.card_table_columns(store, day, c2, c2, "h1", build=build)
    assert how == "built" and calls == [8, 7]
    _, how, _ = live.card_table_columns(store, day, c2, c2, "h2", build=build)
    assert how == "built" and len(calls) == 3
    store.put_parquet("greyhound_prices/prices_2026.parquet", pd.DataFrame({"a": [1]}))
    _, how, _ = live.card_table_columns(store, day, c2, c2, "h2", build=build)
    assert how == "built" and len(calls) == 4
    _, how, _ = live.card_table_columns(store, day, c2, c2, "h2", build=build)
    assert how == "cached" and len(calls) == 4


def test_the_key_moves_with_the_constants_and_not_with_row_order():
    day, c = date(2026, 11, 24), card()
    k = live.card_tables_key(day, c, "h", {})
    assert k == live.card_tables_key(day, c.iloc[::-1], "h", {})
    assert k != live.card_tables_key(day, c, "h", {"ability2": {"constants": 1}})
    assert k != live.card_tables_key(date(2026, 11, 25), c, "h", {})
