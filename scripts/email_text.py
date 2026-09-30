#!/usr/bin/env python3
"""Email a text file to the owner: the body is the file, in a fixed-width font.

    python scripts/email_text.py --subject "Betfair backs" --file out/scored.txt

SMTP_HOST (default smtp.gmail.com), SMTP_PORT (default 587, STARTTLS), SMTP_USERNAME and SMTP_PASSWORD; the email
goes to REPORT_EMAIL, else to SMTP_USERNAME. Without the credentials it says so and sends nothing (exit 0): a
notification must never fail the job that made the file. An unset secret arrives as an empty string, so every
default is taken with `or`, not with getenv's default.
"""

from __future__ import annotations

import argparse
import html
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


def build(subject: str, text: str, sender: str, to: str) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, sender, to
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(f"<pre style=\"font-family: Consolas, Menlo, monospace; font-size: 12px\">"
                        f"{html.escape(text)}</pre>", "html", "utf-8"))
    return msg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subject", required=True)
    ap.add_argument("--file", required=True)
    a = ap.parse_args(argv)
    user, password = os.getenv("SMTP_USERNAME") or "", os.getenv("SMTP_PASSWORD") or ""
    if not user or not password:
        print("SMTP_USERNAME / SMTP_PASSWORD not set: nothing emailed")
        return 0
    to = os.getenv("REPORT_EMAIL") or user
    text = Path(a.file).read_text()
    host, port = os.getenv("SMTP_HOST") or "smtp.gmail.com", int(os.getenv("SMTP_PORT") or "587")
    try:
        with smtplib.SMTP(host, port, timeout=60) as s:
            s.starttls()
            s.login(user, password)
            s.sendmail(user, [to], build(a.subject, text, user, to).as_string())
    except Exception as exc:                                   # a failed notification is reported, not fatal
        print(f"email not sent: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0
    print(f"emailed {a.file} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
