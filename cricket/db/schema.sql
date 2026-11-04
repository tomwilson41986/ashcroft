-- racing² cricket trader: Cricsheet tables (build spec 5.1).
-- Written in the subset of SQL that PostgreSQL (Aurora Serverless v2, once the owner approves it) and SQLite (local
-- runs and tests) both accept, so one file serves both. Keys are Cricsheet's own: the match ID is the file name,
-- a delivery is (match_id, innings, over, ball, repeat). Loads are upserts on these keys (cricket/db/loaders.py).

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    data_version TEXT, created TEXT, revision INTEGER,
    match_type TEXT, match_type_number INTEGER, gender TEXT, team_type TEXT, season TEXT,
    start_date TEXT, end_date TEXT, n_days INTEGER,
    event_name TEXT, event_match_number TEXT, event_stage TEXT, event_group TEXT,
    city TEXT, venue TEXT, venue_id TEXT, team1 TEXT, team2 TEXT, toss_winner TEXT, toss_decision TEXT,
    winner TEXT, result TEXT, by_runs INTEGER, by_wickets INTEGER, by_innings INTEGER,
    method TEXT, eliminator TEXT, bowl_out TEXT, overs INTEGER, balls_per_over INTEGER,
    player_of_match TEXT, missing TEXT
);

CREATE TABLE IF NOT EXISTS innings (
    match_id TEXT NOT NULL, innings INTEGER NOT NULL, team TEXT,
    super_over BOOLEAN, declared BOOLEAN, forfeited BOOLEAN,
    target_runs INTEGER, target_overs REAL, penalty_pre INTEGER, penalty_post INTEGER,
    absent_hurt TEXT, miscounted_overs TEXT,   -- Cricsheet's own flag: overs the umpires miscounted
    runs INTEGER, wickets INTEGER, legal_balls INTEGER, deliveries INTEGER,
    PRIMARY KEY (match_id, innings)
);

CREATE TABLE IF NOT EXISTS deliveries (
    match_id TEXT NOT NULL, innings INTEGER NOT NULL, over INTEGER NOT NULL, ball INTEGER NOT NULL,
    repeat INTEGER NOT NULL,
    legal BOOLEAN,
    batter TEXT NOT NULL, bowler TEXT NOT NULL, non_striker TEXT,
    batter_id TEXT, bowler_id TEXT, non_striker_id TEXT,
    runs_batter INTEGER, runs_extras INTEGER, runs_total INTEGER NOT NULL, non_boundary BOOLEAN,
    wides INTEGER, noballs INTEGER, byes INTEGER, legbyes INTEGER, penalty INTEGER,
    wickets INTEGER, team_wickets INTEGER, wicket_kind TEXT, player_out TEXT, player_out_id TEXT, fielders TEXT,
    wicket2_kind TEXT, player_out2 TEXT, has_review BOOLEAN, has_replacement BOOLEAN,
    PRIMARY KEY (match_id, innings, over, ball, repeat)
);

CREATE TABLE IF NOT EXISTS players (
    match_id TEXT NOT NULL, team TEXT NOT NULL, list_order INTEGER, player TEXT NOT NULL, player_id TEXT,
    PRIMARY KEY (match_id, team, player)
);

CREATE TABLE IF NOT EXISTS venues (
    venue TEXT NOT NULL, city TEXT NOT NULL, venue_id TEXT, matches INTEGER, first_date TEXT, last_date TEXT,
    PRIMARY KEY (venue, city)
);

-- Cricsheet's people register: its ID to ESPNcricinfo, Opta, NV Play and the other providers' keys
CREATE TABLE IF NOT EXISTS people_register (
    identifier TEXT PRIMARY KEY, name TEXT, unique_name TEXT,
    key_bcci TEXT, key_bcci_2 TEXT, key_bigbash TEXT, key_cricbuzz TEXT, key_cricheroes TEXT, key_crichq TEXT,
    key_cricinfo TEXT, key_cricinfo_2 TEXT, key_cricinfo_3 TEXT, key_cricingif TEXT, key_cricketarchive TEXT,
    key_cricketarchive_2 TEXT, key_cricketworld TEXT, key_nvplay TEXT, key_nvplay_2 TEXT, key_opta TEXT,
    key_opta_2 TEXT, key_pulse TEXT, key_pulse_2 TEXT
);

-- Cricsheet's other names for each person (full names, other spellings): how a vendor feed's names are joined
CREATE TABLE IF NOT EXISTS people_names (
    identifier TEXT NOT NULL, name TEXT NOT NULL,
    PRIMARY KEY (identifier, name)
);
