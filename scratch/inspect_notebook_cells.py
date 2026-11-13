import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb.get("cells", [])):
    src = "".join(cell.get("source", []))
    if "BENCHMARK SU 30 DOMANDE HOTPOTQA" in src or "RETRIEVAL_MODE" in src:
        print(f"--- Cell {idx} ---")
        print(src[:600])
        print("...")
    elif "history_db" in src or "benchmark_results" in src or "streamlit" in src.lower():
        print(f"--- Cell {idx} ---")
        print(src[:600])
        print("...")

