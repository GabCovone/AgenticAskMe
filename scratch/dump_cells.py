import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in [9, 10, 11]:
    if i < len(nb.get("cells", [])):
        print(f"================ CELL {i} ================")
        print("".join(nb["cells"][i]["source"]))

