#!/usr/bin/env python3
"""Paper-trade today's UK and Irish win markets on the model's prices.

The trader reads Betfair's live markets and prices (read-only) and simulates every order against
the order book as it stands: nothing is placed on the exchange. The ledger it writes is the
forward test of the early-price edge at the prices actually on offer (TRADING.md).

    # the morning session: the 06:00 record's prices, the configured window, a ledger
    python auto_trade.py --until 11:05

    # a local predictions CSV in place of the 06:00 record (runners matched by course, time, name)
    python auto_trade.py --predictions out/predictions.csv --until 12:00

    # the evening: settle the day's ledger from the closed markets (winners, non-runners, BSPs)
    python auto_trade.py --settle

Settings: trading/config.json, then TRADING_* environment variables, then these flags.
The ledger and a summary go to out/trading/ and to s3://$ULTRA_BETTING_S3_BUCKET/trading/paper/<date>/.
The kill switch is the S3 object trading/STOP in the same bucket, or --kill-file.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import signal
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trading.config import load_config  # noqa: E402
from trading.exchange import BetfairData, PaperExchange, uk_day_window  # noqa: E402
from trading.matching import attach_ids  # noqa: E402
from trading.session import Session  # noqa: E402

log = logging.getLogger("auto_trade")
BUCKET = os.getenv("ULTRA_BETTING_S3_BUCKET", "ashcroft")
MODE = "paper"


def uk_today() -> date:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Europe/London")).date()


def uk_time_on(day: date, hhmm: str) -> datetime:
    from zoneinfo import ZoneInfo
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, h, m, tzinfo=ZoneInfo("Europe/London")).astimezone(timezone.utc)


def _s3():
    import boto3
    return boto3.client("s3")


def _prefix(day: date) -> str:
    return f"trading/{MODE}/{day:%Y-%m-%d}/"


def load_predictions(day: date, path: str | None) -> pd.DataFrame:
    """The day's model prices: a local CSV, or the 06:00 record s3://$BUCKET/predictions/<day>.csv."""
    if path:
        df = pd.read_csv(path, dtype={"race_time": str, "market_id": str})
    else:
        obj = _s3().get_object(Bucket=BUCKET, Key=f"predictions/{day:%Y-%m-%d}.csv")
        df = pd.read_csv(io.BytesIO(obj["Body"].read()), dtype={"race_time": str, "market_id": str})
    if "predicted_bfsp" not in df.columns:
        raise SystemExit("the predictions carry no predicted_bfsp column")
    log.info("Loaded %d runners' prices for %s", len(df), day)
    return df


def kill_switch(kill_file: str | None):
    state = {"at": None, "on": False}

    def check() -> bool:
        if kill_file and Path(kill_file).exists():
            return True
        now = datetime.now(timezone.utc)
        if state["at"] and (now - state["at"]).total_seconds() < 60:
            return state["on"]
        state["at"] = now
        try:
            _s3().head_object(Bucket=BUCKET, Key="trading/STOP")
            state["on"] = True
        except Exception:
            state["on"] = False
        return state["on"]
    return check


def read_ledger(day: date, local: Path) -> list[dict] | None:
    """The day's ledger, local or from S3; None where the day was not traded."""
    if local.exists():
        with local.open() as f:
            return list(csv.DictReader(f))
    try:
        obj = _s3().get_object(Bucket=BUCKET, Key=_prefix(day) + "ledger.csv")
        return list(csv.DictReader(io.StringIO(obj["Body"].read().decode())))
    except Exception as exc:
        log.warning("No ledger for %s (%s): the day was not traded", day, exc)
        return None


def publish(day: date, ledger: Path, summary: dict) -> None:
    try:
        s3 = _s3()
        if ledger.exists():
            s3.put_object(Bucket=BUCKET, Key=_prefix(day) + "ledger.csv", Body=ledger.read_bytes())
        s3.put_object(Bucket=BUCKET, Key=_prefix(day) + "summary.json",
                      Body=json.dumps(summary, indent=2, default=str).encode())
        log.info("Published s3://%s/%s", BUCKET, _prefix(day))
    except Exception as exc:
        log.warning("Could not publish to S3: %s", exc)


def email(summary: dict) -> None:
    try:
        from ultra_betting.reporting.daily_report import send_email_report
    except Exception:
        return
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary.items() if k != "config")
    send_email_report(f"<h3>Paper trading {summary['day']}</h3><table>{rows}</table>",
                      subject=f"Ashcroft paper trading {summary['day']}: {summary['bets_matched']} bets, "
                              f"settled {summary['settled_pnl']:+.2f}, CLV {summary['stake_weighted_clv']}")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None, help="UK racing day, YYYY-MM-DD (default today)")
    ap.add_argument("--config", default="trading/config.json")
    ap.add_argument("--predictions", default=None, help="a local predictions CSV instead of the 06:00 record")
    ap.add_argument("--until", default="21:30", help="stop at this UK time")
    ap.add_argument("--settle", action="store_true", help="settle the day's ledger and publish it")
    ap.add_argument("--out", default="out/trading")
    ap.add_argument("--kill-file", default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--staking", default=None)
    ap.add_argument("--trade-from", default=None)
    ap.add_argument("--trade-until", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    day = date.fromisoformat(a.date) if a.date else uk_today()
    cfg = load_config(a.config, overrides={"strategy": a.strategy, "staking": a.staking,
                                           "trade_from": a.trade_from, "trade_until": a.trade_until})
    log.info("Paper trading %s: strategy %s, staking %s, window %s-%s UK, close at BSP %s (%s)", day,
             cfg.strategy, cfg.staking, cfg.trade_from, cfg.trade_until, cfg.trade_out, cfg.trade_out_at)
    missing = [k for k in ("BETFAIR_USERNAME", "BETFAIR_PASSWORD", "BETFAIR_APP_KEY") if not os.getenv(k)]
    if missing:                                           # a notice, not a daily failure, until they are set
        log.warning("Not trading: %s not set (TRADING.md: what the owner provides)", ", ".join(missing))
        return {"day": str(day), "mode": MODE, "traded": False, "reason": f"missing {', '.join(missing)}"}
    exchange = PaperExchange(BetfairData(), bank=cfg.limits.bank)
    exchange.login()
    ledger = Path(a.out) / f"ledger_{MODE}_{day:%Y-%m-%d}.csv"

    if a.settle:
        rows = read_ledger(day, ledger)
        if rows is None:                                  # nothing traded: nothing to settle or send
            return {"day": str(day), "mode": MODE, "settled": False, "reason": "no ledger"}
        session = Session(exchange, pd.DataFrame(columns=["market_id", "selection_id", "predicted_bfsp"]), cfg, day,
                          ledger_path=str(ledger))
        session.restore(rows, exchange.markets(*uk_day_window(day), cfg.countries))
        session.settle_all()
        session.flush()
        summary = session.summary()
    else:
        preds = load_predictions(day, a.predictions)
        if not {"market_id", "selection_id"} <= set(preds.columns) or preds["market_id"].isna().any():
            preds = attach_ids(preds, exchange.markets(*uk_day_window(day), cfg.countries))
        session = Session(exchange, preds, cfg, day, kill=kill_switch(a.kill_file), ledger_path=str(ledger))

        def stop(signum, _frame):                             # a cancelled workflow: write what we have
            log.warning("Signal %s: writing the ledger", signum)
            session.shutdown()
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        summary = session.run(uk_time_on(day, a.until))
    summary["config"] = cfg.to_dict()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"summary_{MODE}_{day:%Y-%m-%d}.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Summary: %s", {k: v for k, v in summary.items() if k != "config"})
    publish(day, ledger, summary)
    if a.settle:
        email(summary)
    return summary


if __name__ == "__main__":
    main()
