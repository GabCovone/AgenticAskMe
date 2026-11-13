import re

with open('scratch/raw_baseline_log.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tests = re.findall(r'\[Test (\d+)/30\]', text)
print(f"Found test markers: {len(tests)} -> {tests}")

# Let's check some samples
test_blocks = re.split(r'\[Test \d+/30\]', text)
print(f"Number of test blocks: {len(test_blocks)}")

if len(test_blocks) > 1:
    print("\n--- SAMPLE BLOCK 1 (first 500 chars) ---")
    print(test_blocks[1][:500])
    print("\n--- SAMPLE BLOCK 1 (last 500 chars) ---")
    print(test_blocks[1][-500:])

