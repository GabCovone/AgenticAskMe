import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("".join(nb["cells"][10]["source"]))

