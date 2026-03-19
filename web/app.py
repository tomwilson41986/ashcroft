#!/usr/bin/env python3
"""
Ashcroft Web Dashboard — FastAPI API backend.

All data is read from S3 (no local database required).
Deploy to Render/Railway/Fly.io with AWS credentials in env vars.

Usage:
    uvicorn web.app:app --reload --port 8000
"""

import os
from datetime import date

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from web.s3_store import (
    get_all_bets,
    get_bets_for_date,
    get_daily_pnl,
    get_edge_stats,
    get_odds_for_date,
    get_pnl_summary,
    invalidate_cache,
)

app = FastAPI(title="Ashcroft", docs_url="/docs")

# CORS — allow Netlify frontend and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5500", "http://127.0.0.1:5500"],
    allow_origin_regex=r"https://.*\.netlify\.app",
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
