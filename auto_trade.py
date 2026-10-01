#!/usr/bin/env python3
"""Trade today's UK and Irish win markets on the model's prices: on paper, or live on the owner's account.

On paper the trader reads Betfair's markets and prices and simulates every order against the book as it
stands: nothing is placed on the exchange. Live (--live, the owner's decision of 30 Sep 2026) it places the
orders for real: the owner's rule (trading/config_live.json), each back filled at once or not at all and laid at
the Betfair SP as soon as it is matched. Live trading needs the owner's switch, TRADING_LIVE=yes, as well as the
flag; without it the job says so and places nothing.

    # the morning session: the 06:00 record's prices, the configured window, a ledger
    python auto_trade.py --until 11:05                 # paper, trading/config.json
    TRADING_LIVE=yes python auto_trade.py --live --until 11:05     # live, trading/config_live.json

    # a local predictions CSV in place of the 06:00 record (runners matched by course, time, name)
    python auto_trade.py --predictions out/predictions.csv --until 12:00

    # the evening: settle the day's ledger from the closed markets (winners, non-runners, BSPs)
    python auto_trade.py --settle            # or --live --settle

Settings: the config file, then TRADING_* environment variables, then these flags. The ledger and a summary go
to out/trading/ and to s3://$ULTRA_BETTING_S3_BUCKET/trading/<paper|live>/<date>/; live, the ledger is copied
there after every order, and a session that starts again takes up the day from it and from Betfair's own record
of the orders, so no horse is backed twice. The kill switch is the S3 object trading/STOP in the same bucket, or
--kill-file: it stops new bets at once (the lays at SP, the hedges, stay).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import signal
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trading.config import load_config  # noqa: E402
from trading.exchange import BetfairData, LiveExchange, PaperExchange, uk_day_window  # noqa: E402
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


def _prefix(day: date, mode: str = MODE) -> str:
    return f"trading/{mode}/{day:%Y-%m-%d}/"


def book_recorder(day: date, env=None):
    """The day's record of every catalogue and book the session reads (betfair_recorder.py), with BETFAIR_RECORD=1
    (set on the UK runner): research data, never in the way of the trading."""
    env = os.environ if env is None else env
    if str(env.get("BETFAIR_RECORD", "")).strip() != "1":
        return None
    try:
        from betfair_recorder import DayRecorder
        return DayRecorder(day, tag="trader")             # its own files: the recorder runs beside the trader
    except Exception as exc:
        log.warning("Book recorder unavailable (%s): trading without a record of the books", exc)
        return None


def live_switch_on(env=None) -> bool:
    """The owner's switch for real orders: TRADING_LIVE=yes (a repository variable in the workflow)."""
    env = os.environ if env is None else env
    return str(env.get("TRADING_LIVE", "")).strip().lower() == "yes"


def ledger_copier(day: date, mode: str):
    """Copy the ledger to S3 whenever it has grown: live, every order is on record away from the runner."""
    state = {"n": -1}

    def copy(path: Path, n: int) -> None:
        if n == state["n"]:
            return
        _s3().put_object(Bucket=BUCKET, Key=_prefix(day, mode) + "ledger.csv", Body=Path(path).read_bytes())
        state["n"] = n
    return copy


def load_predictions(day: date, path: str | None, wait_until: datetime | None = None,
                     poll_seconds: int = 120, sleep=None) -> pd.DataFrame:
    """The day's model prices: a local CSV, or the 06:00 record s3://$BUCKET/predictions/<day>.csv, waited for
    until ``wait_until`` when the 06:00 run is late (it has written as late as 07:03 UTC)."""
    import time as _time
    if path:
        df = pd.read_csv(path, dtype={"race_time": str, "market_id": str})
    else:
        while True:
            try:
                obj = _s3().get_object(Bucket=BUCKET, Key=f"predictions/{day:%Y-%m-%d}.csv")
                break
            except Exception as exc:
                if wait_until is None or datetime.now(timezone.utc) >= wait_until:
                    raise
                log.warning("The 06:00 record for %s is not in S3 yet (%s); waiting", day, str(exc)[:120])
                (sleep or _time.sleep)(poll_seconds)
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


def read_ledger(day: date, local: Path, mode: str = MODE) -> list[dict] | None:
    """The day's ledger, local or from S3; None where the day was not traded."""
    if local.exists():
        with local.open() as f:
            return list(csv.DictReader(f))
    try:
        obj = _s3().get_object(Bucket=BUCKET, Key=_prefix(day, mode) + "ledger.csv")
        return list(csv.DictReader(io.StringIO(obj["Body"].read().decode())))
    except Exception as exc:
        log.warning("No ledger for %s (%s): the day was not traded", day, exc)
        return None


def publish(day: date, ledger: Path, summary: dict, mode: str = MODE) -> None:
    try:
        s3 = _s3()
        if ledger.exists():
            s3.put_object(Bucket=BUCKET, Key=_prefix(day, mode) + "ledger.csv", Body=ledger.read_bytes())
        s3.put_object(Bucket=BUCKET, Key=_prefix(day, mode) + "summary.json",
                      Body=json.dumps(summary, indent=2, default=str).encode())
        log.info("Published s3://%s/%s", BUCKET, _prefix(day, mode))
    except Exception as exc:
        log.warning("Could not publish to S3: %s", exc)


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def race_lines(ledger: list[dict]) -> list[dict]:
    """The day's settled races from the ledger: course and off, horses backed, staked, the winner if we held it,
    the stake-weighted CLV (our average price against the Betfair SP) and the P&L after commission."""
    races: dict = {}
    for r in ledger:
        if r.get("event") not in ("settle", "commission"):
            continue
        g = races.setdefault(str(r.get("market_id")), {"venue": r.get("venue") or "", "off": r.get("off") or "",
                                                       "horses": 0, "staked": 0.0, "pnl": 0.0, "won": [],
                                                       "clv_w": 0.0, "clv_s": 0.0})
        pnl = _f(r.get("pnl"))
        g["pnl"] += pnl if math.isfinite(pnl) else 0.0
        if r.get("event") == "commission":
            continue
        stake, clv = _f(r.get("matched")), _f(r.get("clv"))
        g["horses"] += 1
        g["staked"] += stake if math.isfinite(stake) else 0.0
        if r.get("result") == "WINNER":
            g["won"].append(str(r.get("runner") or ""))
        if math.isfinite(clv) and math.isfinite(stake) and stake > 0:
            g["clv_w"] += clv * stake
            g["clv_s"] += stake
    out = [{"market_id": mid, "venue": g["venue"], "off": g["off"], "horses": g["horses"],
            "staked": round(g["staked"], 2), "winner": ", ".join(g["won"]), "pnl": round(g["pnl"], 2),
            "clv": g["clv_w"] / g["clv_s"] if g["clv_s"] > 0 else None} for mid, g in races.items()]
    return sorted(out, key=lambda x: (x["off"], x["venue"]))


def running_totals(mode: str = MODE, s3=None) -> dict | None:
    """Every day's published summary under trading/<mode>/ in S3: the days, bets, stakes and settled P&L to date."""
    try:
        s3 = s3 or _s3()
        keys, token = {}, None
        while True:
            kw = {"Bucket": BUCKET, "Prefix": f"trading/{mode}/"}
            if token:
                kw["ContinuationToken"] = token
            page = s3.list_objects_v2(**kw)
            for o in page.get("Contents", []) or []:
                parts = o["Key"].split("/")
                if len(parts) == 4 and parts[3] == "summary.json":
                    keys[parts[2]] = o["Key"]
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
        tot = {"since": min(keys) if keys else None, "days": 0, "bets": 0, "staked": 0.0, "pnl": 0.0}
        for day in sorted(keys):
            got = json.loads(s3.get_object(Bucket=BUCKET, Key=keys[day])["Body"].read())
            tot["days"] += 1
            tot["bets"] += int(got.get("bets_matched") or 0)
            tot["staked"] += float(got.get("turnover") or 0.0)
            tot["pnl"] += float(got.get("settled_pnl") or 0.0)
        return tot if tot["days"] else None
    except Exception as exc:                              # the email goes without the running total
        log.warning("Running totals not read: %s", exc)
        return None


def account_funds(data) -> dict | None:
    """The account's funds, for the record: never a reason for a run to fail."""
    try:
        return data.funds()
    except Exception as exc:
        log.warning("account funds not read (%s)", str(exc)[:200])
        return None


def email(summary: dict, races: list[dict] | None = None, totals: dict | None = None) -> None:
    try:
        from ultra_betting.reporting.daily_report import send_email_report
    except Exception:
        return
    mode = str(summary.get("mode", MODE)).capitalize()
    clv = summary.get("stake_weighted_clv")
    clv_text = f"{clv:+.1%}" if isinstance(clv, (int, float)) else "n/a"
    try:                                                  # a failed email never costs the day's record
        head = (f"<h3>{mode} trading {summary['day']}: settled {summary['settled_pnl']:+,.2f} after commission"
                f"</h3><p>{summary['bets_matched']} bets matched, GBP{summary['turnover']:,.2f} staked, "
                f"{summary.get('races_settled', 0)} races settled; our prices against the Betfair SP: {clv_text} "
                f"(stake-weighted). Each back is laid at the SP, so a horse of ours that wins nets nothing: the "
                f"profit is the price move on the ones that lose.</p>")
        body = ""
        if races:
            def pct(v):
                return f"{v:+.1%}" if v is not None else "-"
            cells = "".join(
                f"<tr><td>{r['off']} {r['venue']}</td><td>{r['horses']}</td><td>{r['staked']:,.2f}</td>"
                f"<td>{r['winner'] or '-'}</td><td>{pct(r['clv'])}</td><td>{r['pnl']:+,.2f}</td></tr>"
                for r in races)
            body += ("<table border='1' cellpadding='4' cellspacing='0'><tr><th>Race</th><th>Horses</th>"
                     "<th>Staked</th><th>Our winner</th><th>CLV</th><th>P&amp;L</th></tr>" + cells + "</table>")
        if totals:
            body += (f"<p>Since {totals['since']}: {totals['days']} days, {totals['bets']} bets, "
                     f"GBP{totals['staked']:,.2f} staked, settled {totals['pnl']:+,.2f}.</p>")
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary.items() if k != "config")
        send_email_report(head + body + f"<p>The day's summary:</p><table>{rows}</table>",
                          subject=f"Ashcroft {mode.lower()} trading {summary['day']}: settled "
                                  f"{summary['settled_pnl']:+,.2f} on GBP{summary['turnover']:,.2f} "
                                  f"({summary['bets_matched']} bets), CLV {clv_text}")
    except Exception as exc:
        log.warning("Email not sent: %s", exc)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None, help="UK racing day, YYYY-MM-DD (default today)")
    ap.add_argument("--live", action="store_true",
                    help="place real orders on the owner's account (only with TRADING_LIVE=yes)")
    ap.add_argument("--config", default=None, help="default trading/config.json, or trading/config_live.json live")
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
    mode = "live" if a.live else MODE
    config = a.config or ("trading/config_live.json" if a.live else "trading/config.json")
    cfg = load_config(config, overrides={"strategy": a.strategy, "staking": a.staking,
                                         "trade_from": a.trade_from, "trade_until": a.trade_until})
    log.info("%s trading %s (%s): strategy %s, staking %s, window %s-%s UK, close at BSP %s (%s); limits: "
             "GBP%.0f a bet, GBP%.0f a day", mode.capitalize(), day, config, cfg.strategy, cfg.staking,
             cfg.trade_from, cfg.trade_until, cfg.trade_out, cfg.trade_out_at, cfg.limits.max_stake,
             cfg.limits.max_daily_turnover)
    missing = [k for k in ("BETFAIR_USERNAME", "BETFAIR_PASSWORD", "BETFAIR_APP_KEY") if not os.getenv(k)]
    if missing:                                           # a notice, not a daily failure, until they are set
        log.warning("Not trading: %s not set (TRADING.md: what the owner provides)", ", ".join(missing))
        return {"day": str(day), "mode": mode, "traded": False, "reason": f"missing {', '.join(missing)}"}
    if a.live and not a.settle and not live_switch_on():
        log.warning("Not trading live: the owner's switch TRADING_LIVE is not 'yes'; nothing placed")
        return {"day": str(day), "mode": mode, "traded": False, "reason": "TRADING_LIVE is not yes"}
    data = BetfairData(recorder=None if a.settle else book_recorder(day))
    # settling reads the closed markets only: the live book is settled through the read-only exchange
    exchange = LiveExchange(data) if (a.live and not a.settle) else PaperExchange(data, bank=cfg.limits.bank)
    exchange.login()
    ledger = Path(a.out) / f"ledger_{mode}_{day:%Y-%m-%d}.csv"
    funds_start = account_funds(data) if a.live else None   # read-only; what the account can bet, for the record
    if funds_start:
        log.info("Account funds: GBP%s available to bet, exposure GBP%s, exposure limit GBP%s",
                 funds_start.get("available"), funds_start.get("exposure"), funds_start.get("exposure_limit"))

    if a.settle:
        rows = read_ledger(day, ledger, mode)
        if rows is None:                                  # nothing traded: nothing to settle or send
            return {"day": str(day), "mode": mode, "settled": False, "reason": "no ledger"}
        session = Session(exchange, pd.DataFrame(columns=["market_id", "selection_id", "predicted_bfsp"]), cfg, day,
                          ledger_path=str(ledger))
        session.mode = mode
        session.restore(rows, exchange.markets(*uk_day_window(day), cfg.countries))
        session.settle_all()
        session.flush()
        summary = session.summary()
    else:
        preds = load_predictions(day, a.predictions, wait_until=uk_time_on(day, cfg.trade_until))
        markets = exchange.markets(*uk_day_window(day), cfg.countries)
        if not {"market_id", "selection_id"} <= set(preds.columns) or preds["market_id"].isna().any():
            preds = attach_ids(preds, markets)
        session = Session(exchange, preds, cfg, day, kill=kill_switch(a.kill_file), ledger_path=str(ledger),
                          on_flush=ledger_copier(day, mode) if a.live else None)
        if a.live:                                        # take up the day: the ledger, then Betfair's own record
            rows = read_ledger(day, ledger, mode)
            if rows:
                session.restore(rows, markets)
                log.info("Took up today's live ledger: %d rows, GBP%.2f matched", len(rows), session.state.turnover)
            session.markets = {m.market_id: m for m in markets}
            session.sync(exchange.orders([m.market_id for m in markets]))

        def stop(signum, _frame):                             # a cancelled workflow: write what we have
            log.warning("Signal %s: writing the ledger", signum)
            session.shutdown()
            if data.recorder is not None:
                data.recorder.maybe_upload(force=True)
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        summary = session.run(uk_time_on(day, a.until))
        if data.recorder is not None:                         # the day's books so far, to S3 before the email
            data.recorder.maybe_upload(force=True)
            summary["books_recorded"] = data.recorder.rows
    if a.live:
        summary["funds_start"], summary["funds_end"] = funds_start, account_funds(data)
    summary["config"] = cfg.to_dict()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"summary_{mode}_{day:%Y-%m-%d}.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Summary: %s", {k: v for k, v in summary.items() if k != "config"})
    publish(day, ledger, summary, mode)
    if a.settle or a.live:
        email(summary, races=race_lines(session.ledger), totals=running_totals(mode))
    return summary


if __name__ == "__main__":
    main()
