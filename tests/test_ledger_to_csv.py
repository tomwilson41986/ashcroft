"""scripts/ledger_to_csv.py: one row a ledger line, the brief's columns, rebuilt the same on a rerun."""
import csv
import json

from scripts.ledger_to_csv import COLUMNS, main


def test_ledger_to_csv_is_one_row_a_line_and_idempotent(tmp_path):
    led = tmp_path / "ledger.jsonl"
    lines = [{"when": "2026-11-18T15:20Z", "kind": "preregistration", "tag": "a", "question": "q", "design": "d"},
             {"when": "2026-11-18T16:20Z", "kind": "result", "tag": "a", "run": "123", "result": {"x": 1},
              "verdict": "FAIL"}]
    led.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n")
    out = tmp_path / "experiments" / "log.csv"
    assert main(["--ledger", str(led), "--out", str(out)]) == 0
    first = out.read_text()
    assert main(["--ledger", str(led), "--out", str(out)]) == 0
    assert out.read_text() == first
    got = list(csv.DictReader(out.open()))
    assert [r["id"] for r in got] == ["a", "a"]
    assert list(got[0]) == COLUMNS
    assert got[1]["metrics"] == '{"x": 1}' and got[1]["decision"] == "FAIL" and got[1]["folds"] == "123"
