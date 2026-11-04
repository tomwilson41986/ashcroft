#!/usr/bin/env python3
"""
Ashcroft Web Dashboard — FastAPI API backend.

All data is read from S3 (no local database required).
Deploy to Render/Railway/Fly.io with AWS credentials in env vars.

Usage:
    uvicorn web.app:app --reload --port 8000
    # then http://localhost:8000/traders: every trader's figures, refreshed every 60 s
    # (needs AWS credentials that read s3://$ULTRA_BETTING_S3_BUCKET/dashboard/traders.json)
"""

import os
from datetime import date

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from web.s3_store import (
    get_all_bets,
    get_bets_for_date,
    get_daily_pnl,
    get_edge_stats,
    get_odds_for_date,
    get_pnl_summary,
    get_traders,
    invalidate_cache,
)
from dashboard.page import render as render_traders
from web.performance import compute_performance_metrics, compute_performance_summary

app = FastAPI(title="Ashcroft", docs_url="/docs")

# CORS — local dev only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "service": "ashcroft-api"}


@app.get("/api/bets/all")
async def api_all_bets():
    return get_all_bets()


@app.get("/api/bets/edge-stats")
async def api_edge_stats():
    return get_edge_stats()


@app.get("/api/bets/live/today")
async def api_live_bets():
    today = date.today().isoformat()
    return [b for b in get_bets_for_date(today) if b.get("status") == "PENDING"]


@app.get("/api/bets/{target_date}")
async def api_bets(target_date: str):
    return get_bets_for_date(target_date)


@app.get("/api/pnl/daily")
async def api_daily_pnl():
    return get_daily_pnl()


@app.get("/api/pnl/summary")
async def api_pnl_summary():
    return get_pnl_summary()


@app.get("/api/odds/latest/{target_date}")
async def api_latest_odds(target_date: str):
    return get_odds_for_date(target_date)


@app.get("/api/model/performance")
async def api_model_performance():
    """Full model performance breakdown from all settled bets."""
    return compute_performance_metrics(get_all_bets())


@app.get("/api/model/performance/summary")
async def api_model_performance_summary():
    """Condensed key performance metrics only."""
    return compute_performance_summary(get_all_bets())


# ---------------------------------------------------------------------------
# The traders' dashboard (dashboard/; built every 10 minutes by trader-dashboard.yml)
# ---------------------------------------------------------------------------

_NO_STORE = {"Cache-Control": "no-store"}
_NO_TRADERS = ("No data found: s3://{bucket}/dashboard/traders.json is not there yet. "
               "It is written by trader-dashboard.yml (python -m dashboard --upload).")


@app.get("/traders", response_class=HTMLResponse)
async def traders_page():
    """Every trader's P&L, volume and forecasts, a tab a sport. The page re-reads /traders.json every 60 s."""
    data = get_traders()
    if data is None:
        from web.s3_store import S3_BUCKET
        return HTMLResponse(f"<!doctype html><title>racing² Traders</title><p>{_NO_TRADERS.format(bucket=S3_BUCKET)}</p>",
                            status_code=503, headers=_NO_STORE)
    return HTMLResponse(render_traders(data), headers=_NO_STORE)


@app.get("/traders.json")
async def traders_json():
    data = get_traders()
    if data is None:
        from web.s3_store import S3_BUCKET
        return JSONResponse({"error": _NO_TRADERS.format(bucket=S3_BUCKET)}, status_code=503, headers=_NO_STORE)
    return JSONResponse(data, headers=_NO_STORE)


@app.post("/api/cache/invalidate")
async def api_invalidate_cache():
    """Force refresh data from S3."""
    invalidate_cache()
    return {"status": "cache cleared"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
