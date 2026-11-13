with open("scratch/raw_baseline_step2167.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Length of raw text:", len(text))
last_1000 = text[-1000:]
print("LAST 1000 CHARS OF STEP 2167:")
print(last_1000)

