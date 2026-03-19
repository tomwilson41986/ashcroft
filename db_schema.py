"""Database schema management for Ashcroft betting system.

Creates and manages tables for odds snapshots, bets, and P&L tracking.
"""

import logging
import os
import sqlite3

log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")


def ensure_tables(db_path: str = DB_PATH) -> None:
    """Create betting-related tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS betfair_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            selection_id INTEGER,
            race_date TEXT NOT NULL,
            race_time TEXT,
            venue TEXT,
            runner_name TEXT NOT NULL,
            best_back REAL,
            best_back_size REAL,
            best_lay REAL,
            best_lay_size REAL,
            sp_near REAL,
            sp_far REAL,
            last_traded REAL,
            total_matched REAL,
            runner_status TEXT,
            market_status TEXT,
            snapshot_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_bf_odds_date ON betfair_odds(race_date);
        CREATE INDEX IF NOT EXISTS idx_bf_odds_market ON betfair_odds(market_id);
        CREATE INDEX IF NOT EXISTS idx_bf_odds_runner ON betfair_odds(runner_name);
        CREATE INDEX IF NOT EXISTS idx_bf_odds_snapshot ON betfair_odds(snapshot_at);

        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_date TEXT NOT NULL,
            race_time TEXT,
            track TEXT NOT NULL,
            horse_name TEXT NOT NULL,
            market_id TEXT,
            selection_id INTEGER,
            predicted_bfsp REAL,
            predicted_win_prob REAL,
            betfair_back REAL,
            betfair_lay REAL,
            edge_pct REAL,
            bet_type TEXT DEFAULT 'BACK',
            stake REAL DEFAULT 0,
            price REAL,
            status TEXT DEFAULT 'PENDING',
            actual_bsp REAL,
            placing INTEGER,
            gross_pnl REAL,
            commission REAL,
            net_pnl REAL,
            created_at TEXT DEFAULT (datetime('now')),
            settled_at TEXT,
            UNIQUE(race_date, race_time, track, horse_name)
        );

        CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(race_date);
        CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);
        CREATE INDEX IF NOT EXISTS idx_bets_track ON bets(track);

        CREATE TABLE IF NOT EXISTS daily_pnl (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            num_bets INTEGER DEFAULT 0,
            winners INTEGER DEFAULT 0,
            losers INTEGER DEFAULT 0,
            voids INTEGER DEFAULT 0,
            total_staked REAL DEFAULT 0,
            gross_pnl REAL DEFAULT 0,
            commission REAL DEFAULT 0,
            net_pnl REAL DEFAULT 0,
            roi_pct REAL DEFAULT 0,
            cumulative_pnl REAL DEFAULT 0,
            bank_start REAL DEFAULT 1000,
            bank_end REAL DEFAULT 1000,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_daily_pnl_date ON daily_pnl(date);
    """)
    conn.commit()
    conn.close()
    log.info("Betting tables ensured")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_tables()
    print("Tables created successfully")
