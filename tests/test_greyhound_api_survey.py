"""research/greyhound/api_survey.py: the field walk and the report (structure only; no API call)."""

import gzip
import json

from research.greyhound import api_survey as sv


def test_walk_counts_fill_and_nested_paths():
    acc = {}
    for r in [{"a": 1, "b": {"c": None}, "d": [{"e": "x"}, {"e": None}]}, {"a": None, "b": {"c": 2}, "d": []}]:
        sv.walk(r, "", acc)
    rows = {r["path"]: r for r in sv.table(acc)}
    assert rows["a"]["fill"] == 0.5 and rows["b.c"]["fill"] == 0.5 and rows["d[].e"]["seen"] == 2
    assert rows["d[].e"]["examples"] == ["x"]


def test_stored_survey_and_report(tmp_path):
    from sources.common import Store
    st = Store(root=tmp_path)
    race = {"race_id": 7, "track": {"track_id": 3, "name": "Romford"},
            "result": {"positions": [{"dog_id": 9, "dog_name": "Some Dog", "weight_kg": 31.2}]}}
    st.put("greyhoundapi/raw/results/GB/2026-10.json.gz", gzip.compress(json.dumps([race]).encode()))
    out = sv.stored(st, regions=("GB",))
    paths = {r["path"] for r in out["results"]["GB"]["2026-10"]["paths"]}
    assert "result.positions[].weight_kg" in paths
    md = sv.report_md({"stored": out, "tables": {}})
    assert "result.positions[].weight_kg" in md and sv.ids(st)["dog"] == 9
