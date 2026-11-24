"""The football metric library says only what the code holds: every built entry names a file and a name in it."""

from __future__ import annotations

from football import metric_library as ml


def test_every_built_entry_is_in_the_code_and_the_document_is_current():
    assert len({(e[0], e[1]) for e in ml.E}) == len(ml.E)
    for fam, name, _what, _data, status, where in ml.E:
        assert status in ml.STATUS
        if status in ("built", "testing"):
            path, token = where
            p = ml.ROOT / path
            text = "".join(f.read_text() for f in p.glob("*.py")) if p.is_dir() else p.read_text()
            assert token in text, f"{fam} / {name}: {token!r} not in {path}"
        else:
            assert where is None
    assert (ml.ROOT / "docs" / "football_metric_library.md").read_text() == ml.render()
