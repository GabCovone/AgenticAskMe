import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    first_line = src.split("\n")[0] if src else "(empty)"
    print(f"Cell {idx} ({cell.get('cell_type')}): {first_line[:70]}")

