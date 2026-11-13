import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import history_db

transcript_path = r"C:\Users\gabri\.gemini\antigravity\brain\e1c8ba1c-edd9-404c-ac00-858544a03c26\.system_generated\logs\transcript_full.jsonl"

baseline_log = ""
with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("step_index") == 2167:
            baseline_log = obj.get("content", "")
            break

print(f"Baseline log length: {len(baseline_log)}")
with open("scratch/raw_baseline_step2167.txt", "w", encoding="utf-8") as f:
    f.write(baseline_log)

test_blocks = re.split(r"\[Test\s+(\d+)/30\]", baseline_log)
print(f"Number of test block parts: {len(test_blocks)}")

parsed_baseline = []
for i in range(1, len(test_blocks), 2):
    test_num = int(test_blocks[i])
    block = test_blocks[i+1]
    
    # Domanda
    m_q = re.search(r"Domanda:\s*(.*?)\r?\n", block)
    question = m_q.group(1).strip() if m_q else ""
    
    # Gold Answer
    m_g = re.search(r"Gold Answer:\s*(.*?)\r?\n", block)
    gold = m_g.group(1).strip() if m_g else ""
    
    # Risposta Agente
    m_a = re.search(r"Risposta Agente:\s*(.*?)\r?\n", block)
    pred = m_a.group(1).strip() if m_a else ""
    
    # Valutazione
    m_val = re.search(r"Valutazione:\s*(CORRETTA|ERRATA).*?\(F1=([\d\.]+),\s*EM=([\d\.]+)\)", block)
    is_corr = (m_val.group(1) == "CORRETTA") if m_val else False
    f1 = float(m_val.group(2)) if m_val else 0.0
    em = float(m_val.group(3)) if m_val else 0.0
    
    # N. Refinements e N. Plannings
    ref_count = len(re.findall(r"--- \[REFINER\] Inizio Ottimizzazione", block))
    plan_count = len(re.findall(r"--- \[PLANNER\] Inizio Esecuzione Piano", block))
    
    # Steps trace
    steps = []
    for line in block.split("\n"):
        line = line.strip()
        if line.startswith(("[VALIDATOR]", "[REFINER]", "[PLANNER]")):
            if any(k in line for k in ["Instradamento", "Output finale", "Sintesi", "Proposta", "Esecuzione single_retrieval", "Avvio ciclo ReAct"]):
                steps.append(line)
                
    # Reasoning
    reasoning_lines = []
    for line in block.split("\n"):
        line = line.strip()
        if any(marker in line for marker in ["[REASONING]:", "[ReAct Thought", "[Synthesis Reasoning]:", "[REFINER REASONING]:", "[Adaptive-RAG Reasoning]:"]):
            reasoning_lines.append(line)
    reasoning_text = "\n".join(reasoning_lines)
    
    # Retrieved context
    ctx_lines = []
    for line in block.split("\n"):
        line = line.strip()
        if any(marker in line for marker in ["[Observation]:", "[PLANNER] Retrieved:"]):
            ctx_lines.append(line)
    context_text = "\n\n".join(ctx_lines)
    
    # Precision e Recall
    metrics = history_db.compute_qa_metrics(pred, gold, question=question)
    prec = metrics["precision"]
    rec = metrics["recall"]
    acc = 1.0 if is_corr else 0.0
    
    parsed_baseline.append({
        "num": test_num,
        "question": question,
        "gold": gold,
        "predicted": pred,
        "is_correct": is_corr,
        "f1": f1,
        "em": em,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "ref_count": ref_count,
        "plan_count": plan_count,
        "steps": steps,
        "reasoning": reasoning_text,
        "context": context_text
    })

print(f"Parsed {len(parsed_baseline)} baseline tests.")
correct_count = sum(1 for t in parsed_baseline if t["is_correct"])
avg_f1 = sum(t["f1"] for t in parsed_baseline) / len(parsed_baseline) if parsed_baseline else 0
avg_em = sum(t["em"] for t in parsed_baseline) / len(parsed_baseline) if parsed_baseline else 0
print(f"Baseline Results: {correct_count}/{len(parsed_baseline)} ({correct_count/len(parsed_baseline)*100:.1f}%) Correct")
print(f"Avg F1: {avg_f1*100:.2f}%, Avg EM: {avg_em*100:.2f}%")

for t in parsed_baseline[:5]:
    print(f"Test {t['num']}: Q: {t['question'][:30]} | Pred: {t['predicted'][:20]} | Gold: {t['gold']} | Correct: {t['is_correct']}")

