with open("scratch/raw_baseline_step2167.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
for i, l in enumerate(lines[-25:]):
    print(f"{len(lines)-25+i}: {repr(l)}")

