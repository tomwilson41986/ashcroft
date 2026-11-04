"""Schema probe for the strategy studies of 4 Nov 2026: field values and coverage (read-only)."""
import sqlite3
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
c = sqlite3.connect("horse_racing.db")
q = lambda s: pd.read_sql_query(s, c)
print(q("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").name.tolist())
for t in ("race_results", "betfair_prices", "betfair_prices_files"):
    try:
        print(t, [r[1] for r in c.execute(f"PRAGMA table_info({t})")])
    except Exception as e:
        print(t, e)
print(q("SELECT race_code, race_type, surface_type, COUNT(*) n, MIN(race_date) a, MAX(race_date) b FROM race_results GROUP BY 1,2,3 ORDER BY n DESC LIMIT 40"))
print(q("""SELECT substr(race_date,1,4) y, COUNT(*) n, AVG(bfsp>1) bsp, AVG(odds>0) sp, AVG(days_since_lr IS NOT NULL) dslr,
          AVG(official_rating>0) orr, AVG(stallion IS NOT NULL AND stallion<>'') sire, AVG(jockeys_claim IS NOT NULL) claim
          FROM race_results GROUP BY y"""))
print(q("SELECT * FROM race_results ORDER BY race_date DESC LIMIT 3").T)
print(q("SELECT days_since_lr, COUNT(*) n FROM race_results WHERE days_since_lr<15 GROUP BY 1"))
print(q("SELECT place, placing_numerical, COUNT(*) n FROM race_results GROUP BY 1,2 ORDER BY n DESC LIMIT 15"))
print(q("SELECT jockeys_claim, COUNT(*) n FROM race_results GROUP BY 1 ORDER BY n DESC LIMIT 10"))
print(q("SELECT race_class, COUNT(*) n FROM race_results GROUP BY 1 ORDER BY n DESC LIMIT 15"))
try:
    print(q("SELECT * FROM betfair_prices LIMIT 3").T)
    print(q("""SELECT b.market_type, substr(r.race_date,1,4) y, COUNT(*) n FROM betfair_prices b JOIN race_results r ON r.id=b.race_results_id
             GROUP BY 1,2"""))
except Exception as e:
    print("betfair_prices", e)
