import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in [11]:
    if i < len(nb.get("cells", [])):
        with open(f"scratch/cell_{i}.py", "w", encoding="utf-8") as out:
            out.write("".join(nb["cells"][i]["source"]))
        print(f"Cell {i} saved to scratch/cell_{i}.py")

