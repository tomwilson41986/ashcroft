#!/usr/bin/env python3
"""
Ashcroft Web Dashboard — FastAPI application.

Provides a web UI for viewing predictions, bets, live odds, and performance.

Usage:
    uvicorn web.app:app --reload --port 8000
    python -m web.app
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_DIR, "horse_racing.db")

app = FastAPI(title="Ashcroft", docs_url="/docs")

# CORS — allow Netlify frontend (and local dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://*.netlify.app",
    ],
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(SCRIPT_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(SCRIPT_DIR, "templates"))


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def rows_to_dicts(rows):
    return [dict(r) for r in rows]


def table_exists(conn, name):
    r = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return r is not None


# ---------------------------------------------------------------------------
# Template Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    today = date.today().isoformat()

    with get_db() as conn:
        # Today's bets summary
        bets_today = []
        if table_exists(conn, "bets"):
            bets_today = rows_to_dicts(conn.execute(
                "SELECT * FROM bets WHERE race_date = ? ORDER BY race_time, track",
                (today,),
            ).fetchall())

        # Overall P&L
        pnl_summary = {"total_bets": 0, "winners": 0, "net_pnl": 0, "roi_pct": 0, "bank": 1000}
        if table_exists(conn, "daily_pnl"):
            row = conn.execute(
                """SELECT SUM(num_bets) as total_bets, SUM(winners) as winners,
                   SUM(net_pnl) as net_pnl, SUM(total_staked) as total_staked
                   FROM daily_pnl"""
            ).fetchone()
            if row and row["total_bets"]:
                staked = row["total_staked"] or 1
                pnl_summary = {
                    "total_bets": row["total_bets"],
                    "winners": row["winners"],
                    "net_pnl": round(row["net_pnl"] or 0, 2),
                    "roi_pct": round((row["net_pnl"] or 0) / staked * 100, 2),
                }
            last = conn.execute(
                "SELECT bank_end FROM daily_pnl ORDER BY date DESC LIMIT 1"
            ).fetchone()
            pnl_summary["bank"] = round(last["bank_end"], 2) if last else 1000

        # Recent P&L for sparkline
        daily_pnl = []
        if table_exists(conn, "daily_pnl"):
            daily_pnl = rows_to_dicts(conn.execute(
                "SELECT * FROM daily_pnl ORDER BY date DESC LIMIT 30"
            ).fetchall())
            daily_pnl.reverse()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "today": today,
        "bets_today": bets_today,
        "pnl_summary": pnl_summary,
        "daily_pnl": daily_pnl,
    })


@app.get("/races", response_class=HTMLResponse)
async def races_page(request: Request, date: str = Query(default=None)):
    target = date or datetime.today().strftime("%Y-%m-%d")

    with get_db() as conn:
        # Get predictions (bets table has our predictions + odds)
        bets = []
        if table_exists(conn, "bets"):
            bets = rows_to_dicts(conn.execute(
                "SELECT * FROM bets WHERE race_date = ? ORDER BY race_time, track",
                (target,),
            ).fetchall())

        # Get latest odds snapshot for this date
        odds = []
        if table_exists(conn, "betfair_odds"):
            odds = rows_to_dicts(conn.execute(
                """SELECT bo.* FROM betfair_odds bo
                   INNER JOIN (
                       SELECT market_id, runner_name, MAX(snapshot_at) as latest
                       FROM betfair_odds WHERE race_date = ?
                       GROUP BY market_id, runner_name
                   ) latest ON bo.market_id = latest.market_id
                       AND bo.runner_name = latest.runner_name
                       AND bo.snapshot_at = latest.latest
                   ORDER BY bo.venue, bo.race_time, bo.best_back""",
                (target,),
            ).fetchall())

    # Group odds by venue + race_time
    races = {}
    for o in odds:
        key = f"{o['venue']}|{o['race_time']}"
        if key not in races:
            races[key] = {"venue": o["venue"], "race_time": o["race_time"], "runners": []}
        races[key]["runners"].append(o)

    # Merge bet info into odds
    bet_lookup = {}
    for b in bets:
        bkey = (b["horse_name"].strip().lower(), b["track"].strip().lower())
        bet_lookup[bkey] = b

    return templates.TemplateResponse("races.html", {
        "request": request,
        "target_date": target,
        "races": races,
        "bets": bets,
        "bet_lookup": bet_lookup,
    })


@app.get("/bets", response_class=HTMLResponse)
async def bets_page(
    request: Request,
    date: str = Query(default=None),
    status: str = Query(default=None),
):
    with get_db() as conn:
        if not table_exists(conn, "bets"):
            return templates.TemplateResponse("bets.html", {
                "request": request, "bets": [], "dates": [],
                "selected_date": date, "selected_status": status,
            })

        query = "SELECT * FROM bets WHERE 1=1"
        params = []
        if date:
            query += " AND race_date = ?"
            params.append(date)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY race_date DESC, race_time"

        bets = rows_to_dicts(conn.execute(query, params).fetchall())

        dates = rows_to_dicts(conn.execute(
            "SELECT DISTINCT race_date FROM bets ORDER BY race_date DESC"
        ).fetchall())

    return templates.TemplateResponse("bets.html", {
        "request": request,
        "bets": bets,
        "dates": [d["race_date"] for d in dates],
        "selected_date": date,
        "selected_status": status,
    })


@app.get("/performance", response_class=HTMLResponse)
async def performance_page(request: Request):
    with get_db() as conn:
        daily_pnl = []
        if table_exists(conn, "daily_pnl"):
            daily_pnl = rows_to_dicts(conn.execute(
                "SELECT * FROM daily_pnl ORDER BY date"
            ).fetchall())

        # Edge bucket stats
        edge_stats = []
        if table_exists(conn, "bets"):
            edge_stats = rows_to_dicts(conn.execute(
                """SELECT
                    CASE
                        WHEN edge_pct >= 20 THEN '20%+'
                        WHEN edge_pct >= 15 THEN '15-20%'
                        WHEN edge_pct >= 10 THEN '10-15%'
                        WHEN edge_pct >= 5 THEN '5-10%'
                        ELSE '<5%'
                    END as bucket,
                    COUNT(*) as total,
                    SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) as won,
                    SUM(net_pnl) as pnl
                   FROM bets WHERE status IN ('WON','LOST')
                   GROUP BY bucket ORDER BY MIN(edge_pct) DESC"""
            ).fetchall())

        # Summary stats
        summary = {}
        if table_exists(conn, "bets"):
            row = conn.execute(
                """SELECT COUNT(*) as total,
                   SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN status='LOST' THEN 1 ELSE 0 END) as losses,
                   SUM(net_pnl) as net_pnl,
                   SUM(stake) as staked,
                   AVG(edge_pct) as avg_edge,
                   AVG(predicted_bfsp) as avg_pred_bfsp
                   FROM bets WHERE status IN ('WON','LOST')"""
            ).fetchone()
            if row and row["total"] > 0:
                summary = {
                    "total_bets": row["total"],
                    "wins": row["wins"],
                    "losses": row["losses"],
                    "strike_rate": round(row["wins"] / row["total"] * 100, 1),
                    "net_pnl": round(row["net_pnl"] or 0, 2),
                    "roi": round((row["net_pnl"] or 0) / (row["staked"] or 1) * 100, 2),
                    "avg_edge": round(row["avg_edge"] or 0, 1),
                }

        # Best / worst days
        best_day = worst_day = None
        if daily_pnl:
            best_day = max(daily_pnl, key=lambda x: x["net_pnl"])
            worst_day = min(daily_pnl, key=lambda x: x["net_pnl"])

    return templates.TemplateResponse("performance.html", {
        "request": request,
        "daily_pnl": daily_pnl,
        "edge_stats": edge_stats,
        "summary": summary,
        "best_day": best_day,
        "worst_day": worst_day,
    })


# ---------------------------------------------------------------------------
# API Routes (JSON)
# ---------------------------------------------------------------------------

@app.get("/api/predictions/{target_date}")
async def api_predictions(target_date: str):
    with get_db() as conn:
        if not table_exists(conn, "bets"):
            return []
        rows = conn.execute(
            "SELECT * FROM bets WHERE race_date = ? ORDER BY race_time, track",
            (target_date,),
        ).fetchall()
    return rows_to_dicts(rows)


@app.get("/api/bets/all")
async def api_all_bets():
    """Get all bets across all dates."""
    with get_db() as conn:
        if not table_exists(conn, "bets"):
            return []
        rows = conn.execute(
            "SELECT * FROM bets ORDER BY race_date DESC, race_time"
        ).fetchall()
    return rows_to_dicts(rows)


@app.get("/api/bets/edge-stats")
async def api_edge_stats():
    """Get strike rate broken down by edge bucket."""
    with get_db() as conn:
        if not table_exists(conn, "bets"):
            return []
        rows = conn.execute(
            """SELECT
                CASE
                    WHEN edge_pct >= 20 THEN '20%+'
                    WHEN edge_pct >= 15 THEN '15-20%'
                    WHEN edge_pct >= 10 THEN '10-15%'
                    WHEN edge_pct >= 5 THEN '5-10%'
                    ELSE '<5%'
                END as bucket,
                COUNT(*) as total,
                SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) as won,
                SUM(net_pnl) as pnl
               FROM bets WHERE status IN ('WON','LOST')
               GROUP BY bucket ORDER BY MIN(edge_pct) DESC"""
        ).fetchall()
    return rows_to_dicts(rows)


@app.get("/api/bets/live/today")
async def api_live_bets():
    today = date.today().isoformat()
    with get_db() as conn:
        if not table_exists(conn, "bets"):
            return []
        rows = conn.execute(
            "SELECT * FROM bets WHERE race_date = ? AND status = 'PENDING' ORDER BY race_time",
            (today,),
        ).fetchall()
    return rows_to_dicts(rows)


@app.get("/api/bets/{target_date}")
async def api_bets(target_date: str):
    with get_db() as conn:
        if not table_exists(conn, "bets"):
            return []
        rows = conn.execute(
            "SELECT * FROM bets WHERE race_date = ? ORDER BY race_time",
            (target_date,),
        ).fetchall()
    return rows_to_dicts(rows)


@app.get("/api/pnl/daily")
async def api_daily_pnl():
    with get_db() as conn:
        if not table_exists(conn, "daily_pnl"):
            return []
        rows = conn.execute(
            "SELECT * FROM daily_pnl ORDER BY date"
        ).fetchall()
    return rows_to_dicts(rows)


@app.get("/api/pnl/summary")
async def api_pnl_summary():
    with get_db() as conn:
        if not table_exists(conn, "bets"):
            return {"total_bets": 0, "net_pnl": 0, "roi": 0, "bank": 1000}

        row = conn.execute(
            """SELECT COUNT(*) as total,
               SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) as wins,
               SUM(net_pnl) as net_pnl, SUM(stake) as staked,
               AVG(edge_pct) as avg_edge
               FROM bets WHERE status IN ('WON','LOST')"""
        ).fetchone()

        if not row or row["total"] == 0:
            return {"total_bets": 0, "net_pnl": 0, "roi": 0, "bank": 1000}

        bank = 1000
        if table_exists(conn, "daily_pnl"):
            last = conn.execute(
                "SELECT bank_end FROM daily_pnl ORDER BY date DESC LIMIT 1"
            ).fetchone()
            if last:
                bank = round(last["bank_end"], 2)

        return {
            "total_bets": row["total"],
            "wins": row["wins"],
            "strike_rate": round(row["wins"] / row["total"] * 100, 1),
            "net_pnl": round(row["net_pnl"] or 0, 2),
            "roi": round((row["net_pnl"] or 0) / (row["staked"] or 1) * 100, 2),
            "avg_edge": round(row["avg_edge"] or 0, 1),
            "bank": bank,
        }


@app.get("/api/odds/{market_id}")
async def api_odds_history(market_id: str):
    """Get odds movement time series for a specific market."""
    with get_db() as conn:
        if not table_exists(conn, "betfair_odds"):
            return []
        rows = conn.execute(
            """SELECT runner_name, best_back, best_lay, sp_near,
                      last_traded, snapshot_at
               FROM betfair_odds WHERE market_id = ?
               ORDER BY snapshot_at, runner_name""",
            (market_id,),
        ).fetchall()
    return rows_to_dicts(rows)


@app.get("/api/odds/latest/{target_date}")
async def api_latest_odds(target_date: str):
    """Get the most recent odds snapshot for a date."""
    with get_db() as conn:
        if not table_exists(conn, "betfair_odds"):
            return []
        rows = conn.execute(
            """SELECT bo.* FROM betfair_odds bo
               INNER JOIN (
                   SELECT market_id, runner_name, MAX(snapshot_at) as latest
                   FROM betfair_odds WHERE race_date = ?
                   GROUP BY market_id, runner_name
               ) l ON bo.market_id = l.market_id
                   AND bo.runner_name = l.runner_name
                   AND bo.snapshot_at = l.latest
               ORDER BY bo.venue, bo.race_time""",
            (target_date,),
        ).fetchall()
    return rows_to_dicts(rows)


# ---------------------------------------------------------------------------
# HTMX Partials (for live polling)
# ---------------------------------------------------------------------------

@app.get("/partials/bets-today", response_class=HTMLResponse)
async def partial_bets_today(request: Request):
    today = date.today().isoformat()
    with get_db() as conn:
        bets = []
        if table_exists(conn, "bets"):
            bets = rows_to_dicts(conn.execute(
                "SELECT * FROM bets WHERE race_date = ? ORDER BY race_time",
                (today,),
            ).fetchall())
    return templates.TemplateResponse("partials/bets_table.html", {
        "request": request, "bets": bets,
    })


@app.get("/partials/pnl-card", response_class=HTMLResponse)
async def partial_pnl_card(request: Request):
    with get_db() as conn:
        pnl = {"net_pnl": 0, "bank": 1000, "total_bets": 0}
        if table_exists(conn, "daily_pnl"):
            last = conn.execute(
                "SELECT bank_end, cumulative_pnl FROM daily_pnl ORDER BY date DESC LIMIT 1"
            ).fetchone()
            if last:
                pnl["bank"] = round(last["bank_end"], 2)
                pnl["net_pnl"] = round(last["cumulative_pnl"], 2)
    return templates.TemplateResponse("partials/pnl_card.html", {
        "request": request, "pnl": pnl,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
