import re

with open("scratch/raw_baseline_step2167.txt", "r", encoding="utf-8") as f:
    text = f.read()

parts = re.split(r"\[Test\s+(\d+)/30\]", text)
for i in range(1, len(parts), 2):
    test_idx = parts[i]
    block = parts[i+1]
    has_val = "Valutazione:" in block
    has_pred = "Risposta Agente:" in block
    print(f"Test {test_idx}: has_pred={has_pred}, has_val={has_val}, block_len={len(block)}")

