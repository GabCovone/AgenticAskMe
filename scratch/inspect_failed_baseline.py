import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratch.parse_baseline_step2167 import parsed_baseline

for t in parsed_baseline:
    if not t["is_correct"]:
        print(f"Test {t['num']}: Q: {t['question'][:50]} | Pred: {t['predicted'][:35]} | Gold: {t['gold']} | F1: {t['f1']}")
