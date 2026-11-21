"""scripts/shadow_predict.py and predict-shadow.yml (the review of 20 Nov 2026, M1): the previous blend prices the
morning's card read-only, only on the days shadow.json names."""
import json
from datetime import date
from pathlib import Path

import yaml

from scripts import shadow_predict as S

ROOT = Path(__file__).resolve().parents[1]


def test_the_shadow_runs_only_on_the_named_days(tmp_path):
    assert not S.active(None, date(2026, 11, 22))
    p = tmp_path / "shadow.json"
    p.write_text(json.dumps({"commit": "abc123", "until": "2026-12-05"}))
    cfg = S.config(p)
    assert S.active(cfg, date(2026, 11, 22)) and S.active(cfg, date(2026, 12, 5))
    assert not S.active(cfg, date(2026, 12, 6)) and not S.active({"until": "2026-12-05"}, date(2026, 11, 22))
    assert S.config(tmp_path / "absent.json") is None


def test_the_shadow_never_writes_the_traders_file_and_holds_no_betfair_key():
    src = (ROOT / "scripts" / "shadow_predict.py").read_text()
    assert 'write_csv("predictions_shadow"' in src and 'write_csv("predictions",' not in src
    text = (ROOT / ".github" / "workflows" / "predict-shadow.yml").read_text()
    wf = yaml.safe_load(text)
    assert "BETFAIR" not in text and "TRADING_LIVE" not in text
    assert wf["jobs"]["shadow"]["runs-on"] == "ubuntu-latest"
