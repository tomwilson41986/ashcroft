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

    def build(store, day, crd, fits, lag, names=None):
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
    monkeypatch.setattr(live, "CARD_FIELD_FREE", {})                 # no table proven free of the field
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


def two_tables(calls):
    """``connections`` reads only the dog's own row; ``ability`` the field (its value is the race's size)."""
    cx, kf = pit.module("connections").PREFIX, pit.module("ability").PREFIX

    def build(store, day, crd, fits, lag, names=None):
        names = list(names or ("ability", "connections"))
        calls.append((len(crd), tuple(sorted(names))))
        k = crd[pit.KEY + ["race_id"]].iloc[:, :2].copy()
        size = crd.groupby("race_id").dog_id.transform("size").to_numpy()
        out = {}
        if "ability" in names:
            out["ability"] = k.assign(**{f"{kf}field": size.astype(float)})
        if "connections" in names:
            out["connections"] = k.assign(**{f"{cx}own": (k.dog_id * 2.0).to_numpy()})
        return out
    return build, f"{kf}field", f"{cx}own"


def proven(monkeypatch, tables=("connections",)):
    monkeypatch.setattr(pit, "stored_fits", lambda store, names: {})
    monkeypatch.setattr(live, "CARD_FIELD_FREE", {n: pit.code_hash(n) for n in tables})
    from greyhound import stack_model as sm
    monkeypatch.setattr(sm, "PIT_TABLES", ("ability", "connections"))


def test_a_non_runner_builds_only_the_tables_that_read_the_field(tmp_path, monkeypatch):
    proven(monkeypatch)
    store, calls = Store(root=tmp_path), []
    build, field, own = two_tables(calls)
    day = date(2026, 11, 24)
    w1, how, _ = live.card_table_columns(store, day, card(), card(), "h1", build=build)
    assert how == "built" and calls == [(8, ("ability", "connections"))]
    c2 = card(removed=(12,))
    w2, how, _ = live.card_table_columns(store, day, c2, c2, "h1", build=build)
    assert how == "partial: ability" and calls[-1] == (7, ("ability",))
    assert "_card_row" not in w2.columns and len(w2) == 7
    assert (w2[own] == w2.dog_id * 2.0).all()                           # read from the morning's build
    r1 = w2[w2.race_id == 1]
    assert (r1[field] == 3.0).all() and (w2.loc[w2.race_id == 2, field] == 4.0).all()   # built with the field now
    _, how, _ = live.card_table_columns(store, day, c2, c2, "h1", build=build)
    assert how == "cached" and len(calls) == 2
    c3 = card(removed=(12, 23))                                        # another one out: from either earlier build
    w3, how, _ = live.card_table_columns(store, day, c3, c3, "h1", build=build)
    assert how == "partial: ability" and (w3[own] == w3.dog_id * 2.0).all()
    assert (w3.loc[w3.race_id == 2, field] == 3.0).all()


def test_a_new_runner_a_changed_own_row_or_new_code_builds_everything(tmp_path, monkeypatch):
    proven(monkeypatch)
    store, calls = Store(root=tmp_path), []
    build, _, _ = two_tables(calls)
    day = date(2026, 11, 24)
    c1 = card(removed=(12,))
    live.card_table_columns(store, day, c1, c1, "h1", build=build)
    c2 = card()                                                         # a reserve in: a runner no build has
    _, how, _ = live.card_table_columns(store, day, c2, c2, "h1", build=build)
    assert how == "built"
    c3 = card(removed=(12,)).assign(trap=lambda x: x.trap.where(x.dog_id != 11, 6))   # a dog's own row moved
    _, how, _ = live.card_table_columns(store, day, c3, c3, "h1", build=build)
    assert how == "built"
    monkeypatch.setattr(live, "CARD_FIELD_FREE", {"connections": "an older hash"})   # proven at other code: not free
    c4 = card(removed=(12, 23))
    _, how, _ = live.card_table_columns(store, day, c4, c4, "h1", build=build)
    assert how == "built"


def test_the_tables_rebuilt_take_their_inputs_with_them(monkeypatch):
    monkeypatch.setattr(live, "CARD_FIELD_FREE", {n: pit.code_hash(n) for n in ("ability", "trials", "connections")})
    got = live.card_field_tables(["ability", "trials", "connections", "trials2", "ability2"])
    assert got == ["ability", "trials", "trials2", "ability2"]          # ability2 reads ability, trials and trials2
    monkeypatch.setattr(live, "CARD_FIELD_FREE", {})
    assert live.card_field_tables(["ability", "connections"]) == ["ability", "connections"]


def test_field_diff_finds_the_tables_that_read_the_field():
    kf, cx = pit.module("ability").PREFIX, pit.module("connections").PREFIX
    full = pd.DataFrame({"race_id": [1, 1, 1, 2, 2], "dog_id": [11, 12, 13, 21, 22],
                         f"{kf}a": [3.0, 3.0, 3.0, 2.0, 2.0], f"{cx}a": [1.0, None, 2.0, 5.0, 6.0]})
    less = full[full.dog_id != 12].assign(**{f"{kf}a": [2.0, 2.0, 2.0, 2.0]})
    r = live.field_diff("ability", full, less, {1})
    assert r["rows"] == 4 and r["rows_differ"] == 2 and r["rows_differ_untouched_races"] == 0 and not r["field_free"]
    assert live.field_diff("connections", full, less, {1})["field_free"]
    away = less.assign(**{f"{kf}a": [3.0, 3.0, 9.0, 2.0]})
    assert live.field_diff("ability", full, away, {1})["rows_differ_untouched_races"] == 1

