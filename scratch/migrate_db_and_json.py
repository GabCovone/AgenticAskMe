import sqlite3
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import history_db

# 1. Trigger init_history_db to add system_mode column if missing
history_db.init_history_db()

# 2. Update all existing rows to 'full'
conn = sqlite3.connect("agentic_history.db")
cursor = conn.cursor()
cursor.execute("UPDATE query_runs SET system_mode = 'full' WHERE system_mode IS NULL OR system_mode = ''")
conn.commit()

# Check count and columns
cursor.execute("PRAGMA table_info(query_runs)")
cols = [c[1] for c in cursor.fetchall()]
print("Updated Columns in agentic_history.db:", cols)

cursor.execute("SELECT COUNT(*), system_mode FROM query_runs GROUP BY system_mode")
print("Mode counts in DB:", cursor.fetchall())

# 3. Export to benchmark_30_runs.json with clean 1..30 indexing and system_mode='full'
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("SELECT * FROM query_runs ORDER BY id ASC")
rows = [dict(r) for r in cursor.fetchall()]
conn.close()

for idx, r in enumerate(rows):
    r["id"] = idx + 1
    r["system_mode"] = "full"

with open("benchmark_30_runs.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print("Updated benchmark_30_runs.json with 30 runs, system_mode=full, IDs 1..30.")
