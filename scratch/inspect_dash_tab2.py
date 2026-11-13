with open("Dashboard.py", "r", encoding="utf-8") as f:
    text = f.read()

# Find occurrences of get_history_stats or comparison
import re
matches = [m.start() for m in re.finditer(r"Confronto Diretto", text)]
print(f"Matches for 'Confronto Diretto': {len(matches)}")
if matches:
    idx = matches[0]
    print(text[idx-200:idx+600])

