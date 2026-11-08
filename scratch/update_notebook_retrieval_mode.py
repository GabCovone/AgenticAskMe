import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][9]
source = "".join(cell["source"])

target = """NUM_TESTS = 10  # Numero di campioni da testare nel benchmark
correct = 0
results_log = []

print(f"\\n--- AVVIO BENCHMARK SU {NUM_TESTS} DOMANDE HOTPOTQA --- ")"""

replacement = """NUM_TESTS = 10  # Numero di campioni da testare nel benchmark
correct = 0
results_log = []

# CONFIGURAZIONE MODALITÀ RETRIEVAL (Zero modifiche ai prompt):
# - 'full': Sistema Integrale (KB Locale + Ricerca Web Live)
# - 'baseline': Sistema Baseline (Solo Knowledge Base Locale di Wikipedia)
RETRIEVAL_MODE = "full"  # Imposta 'baseline' per vincolare la ricerca alla sola KB locale
Planner.set_retrieval_mode(RETRIEVAL_MODE)
print(f"Modalità di Retrieval Configurata: {Planner.get_retrieval_mode().upper()}")

print(f"\\n--- AVVIO BENCHMARK SU {NUM_TESTS} DOMANDE HOTPOTQA --- ")"""

state_target = """    state = {
        "original_query": question,
        "current_query": question,
        "retrieved_context": "",
        "num_refinement": 0,
        "num_planning": 0,
        "next_node": "",
        "feedback_history": [],
        "final_answer": ""
    }"""

state_replacement = """    state = {
        "original_query": question,
        "current_query": question,
        "retrieved_context": "",
        "num_refinement": 0,
        "num_planning": 0,
        "next_node": "",
        "feedback_history": [],
        "final_answer": "",
        "retrieval_mode": RETRIEVAL_MODE
    }"""

if target in source and state_target in source:
    source = source.replace(target, replacement)
    source = source.replace(state_target, state_replacement)
    cell["source"] = [l + "\n" for l in source.split("\n")[:-1]] + [source.split("\n")[-1]]
    with open("notebook.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully with RETRIEVAL_MODE toggle.")
else:
    print("Targets not matched directly.")

