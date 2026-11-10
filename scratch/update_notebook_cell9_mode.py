import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][9]
source = "".join(cell["source"])

target = """            steps_trace=last_state.get("visited_nodes", []),
            reasoning=last_state.get("reasoning", ""),
            retrieved_context=last_state.get("retrieved_context", "")
        )"""

replacement = """            steps_trace=last_state.get("visited_nodes", []),
            reasoning=last_state.get("reasoning", ""),
            retrieved_context=last_state.get("retrieved_context", ""),
            system_mode=RETRIEVAL_MODE
        )"""

if target in source:
    source = source.replace(target, replacement)
    cell["source"] = [l + "\n" for l in source.split("\n")[:-1]] + [source.split("\n")[-1]]
    with open("notebook.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Updated cell 9 in notebook.ipynb with system_mode=RETRIEVAL_MODE.")
else:
    print("Target not found in cell 9.")

