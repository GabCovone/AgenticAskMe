import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][9]
source = "".join(cell["source"])

old_stream_code = """    final_answer = ""
    try:
        for output in app.stream(state, {"recursion_limit": 30}):
            for key, value in output.items():
                if "final_answer" in value and value["final_answer"]:
                    final_answer = value["final_answer"]"""

new_stream_code = """    final_answer = ""
    last_state = {}
    try:
        for output in app.stream(state, {"recursion_limit": 30}):
            for key, value in output.items():
                if isinstance(value, dict):
                    last_state.update(value)
                if "final_answer" in value and value["final_answer"]:
                    final_answer = value["final_answer"]"""

old_append_code = """    results_log.append({
        'question': question,
        'gold': gold_answer,
        'predicted': final_answer,
        'correct': is_correct,
        'f1': q_f1,
        'precision': q_prec,
        'recall': q_rec,
        'em': q_em
    })

    time.sleep(1) # Rispetto dei limiti di frequenza per la ricerca web"""

new_append_code = """    results_log.append({
        'question': question,
        'gold': gold_answer,
        'predicted': final_answer,
        'correct': is_correct,
        'f1': q_f1,
        'precision': q_prec,
        'recall': q_rec,
        'em': q_em
    })

    # Persistenza opzionale nel database storico locale SQLite
    try:
        import history_db
        history_db.log_query_run(
            query=question,
            final_answer=final_answer,
            gold_answer=gold_answer,
            is_correct=is_correct,
            f1=q_f1,
            precision=q_prec,
            recall=q_rec,
            em=q_em,
            accuracy=1.0 if is_correct else 0.0,
            num_refinement=last_state.get("refinement_count", last_state.get("num_refinement", 0)),
            num_planning=last_state.get("planning_hops", last_state.get("num_planning", 0)),
            steps_trace=last_state.get("visited_nodes", []),
            reasoning=last_state.get("reasoning", ""),
            retrieved_context=last_state.get("retrieved_context", "")
        )
    except Exception:
        pass

    time.sleep(1) # Rispetto dei limiti di frequenza per la ricerca web"""

if old_stream_code in source and old_append_code in source:
    source = source.replace(old_stream_code, new_stream_code)
    source = source.replace(old_append_code, new_append_code)
    cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]
    with open("notebook.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully with history_db persistence.")
else:
    print("Code blocks not found directly, checking differences...")

