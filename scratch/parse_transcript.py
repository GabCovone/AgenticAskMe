import json
import os

transcript_path = r'C:\Users\gabri\.gemini\antigravity\brain\e1c8ba1c-edd9-404c-ac00-858544a03c26\.system_generated\logs\transcript_full.jsonl'
if not os.path.exists(transcript_path):
    print("Transcript full not found, checking transcript.jsonl...")
    transcript_path = r'C:\Users\gabri\.gemini\antigravity\brain\e1c8ba1c-edd9-404c-ac00-858544a03c26\.system_generated\logs\transcript.jsonl'

with open(transcript_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT':
            content = data.get('content', '')
            if 'Modalità di Retrieval Configurata: BASELINE' in content or 'AVVIO BENCHMARK SU 30 DOMANDE HOTPOTQA' in content:
                print(f"Matched line {idx}, content len: {len(content)}")
                with open('scratch/raw_baseline_log.txt', 'w', encoding='utf-8') as out:
                    out.write(content)
                print("Saved to scratch/raw_baseline_log.txt")
                break

