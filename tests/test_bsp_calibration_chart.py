"""Tests for the BSP calibration chart: it reads the query's printout as printed, and draws from it."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bsp_calibration_chart", ROOT / "scripts" / "bsp_calibration_chart.py")
chart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chart)
TXT = ROOT / "reports" / "bsp_calibration.txt"


def test_the_printout_parses_into_both_windows_and_every_season():
    d = chart.parse(TXT.read_text())
    for w, runners in (("10", 1_201_061), ("5", 608_540)):
        m = d[w]["measures"]
        assert int(m["runners"]) == runners
        assert m["r2_bins"] > m["r2_mcfadden"] > m["r2_efron"] > 0
        assert len(d[w]["bands"]) == len(chart.BANDS) == len(chart.BAND_BSP)
        assert len(d[w]["bins"]) == 20
        assert sum(b[2] for b in d[w]["bins"]) == runners           # the 20 groups hold every runner
        assert d[w]["bands"][0][0] == "0.0%-1%" and d[w]["bands"][-1][0] == "80%+"
    assert [s[0] for s in d["seasons"]] == [f"{y}-{str(y + 1)[2:]}" for y in range(2016, 2026)]
    assert sum(s[1] for s in d["seasons"]) == d["10"]["measures"]["races"]


def test_the_interval_on_winners_over_expected_is_the_binomial_one():
    lo, hi = chart.ae_interval(10_000, 0.10, 1.0)                    # 1,000 winners expected and won
    half = 1.96 * (0.9 / 1_000) ** 0.5
    assert lo == pytest.approx(1 - half) and hi == pytest.approx(1 + half)


def test_the_chart_is_drawn(tmp_path):
    out = tmp_path / "chart.png"
    chart.chart(chart.parse(TXT.read_text()), out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and out.stat().st_size > 50_000
