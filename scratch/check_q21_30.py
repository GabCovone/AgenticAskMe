import json

with open("benchmark_30_runs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i in range(20, 30):
    print(f"Test {i+1}: Q: {data[i]['query']} | Gold: {data[i]['gold_answer']}")

