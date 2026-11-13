import json

transcript_path = r'C:\Users\gabri\.gemini\antigravity\brain\e1c8ba1c-edd9-404c-ac00-858544a03c26\.system_generated\logs\transcript_full.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT' and data.get('step_index', 0) >= 2000:
            content = data.get('content', '')
            step = data.get('step_index')
            tf = data.get('truncated_fields')
            print(f'Line {idx}, step {step}, len={len(content)}, truncated={tf}, preview={content[:60]}...')

