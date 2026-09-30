"""scripts/email_text.py: a text file emailed to the owner; without credentials it sends nothing and exits 0."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import email_text  # noqa: E402


def test_the_body_is_the_file_as_plain_text_and_escaped_html():
    msg = email_text.build("Betfair backs", "1:57 Catterick <Bladey Lady> & co", "a@x.com", "b@x.com")
    plain, html_part = msg.get_payload()
    assert msg["Subject"] == "Betfair backs" and msg["To"] == "b@x.com"
    assert "<Bladey Lady>" in plain.get_payload(decode=True).decode()
    assert "&lt;Bladey Lady&gt; &amp; co" in html_part.get_payload(decode=True).decode()


def test_without_credentials_nothing_is_sent(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SMTP_USERNAME", "")
    monkeypatch.setenv("SMTP_PASSWORD", "")
    f = tmp_path / "t.txt"
    f.write_text("x")
    assert email_text.main(["--subject", "s", "--file", str(f)]) == 0
    assert "nothing emailed" in capsys.readouterr().out
