"""Which Betfair price files are loaded, 15-23 Nov (the UK win files stopped at 17 Nov at the 22 Nov check; read-only)."""
import sqlite3

con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
cols = [r[1] for r in con.execute("PRAGMA table_info(betfair_prices_files)")]
print("betfair_prices_files columns:", cols)
for pat in ("ukwin", "ukplace", "irewin", "ireplace"):
    rows = con.execute(f"SELECT * FROM betfair_prices_files WHERE {cols[0]} LIKE ? ORDER BY {cols[0]}",
                       (f"%{pat}%2026%",)).fetchall()
    tail = [r for r in rows if any(f"{d}112026" in str(r[0]) for d in range(15, 24))]
    print(f"\n{pat}: {len(rows)} files in all; 15-23 Nov:")
    for r in tail:
        print("  ", r)
