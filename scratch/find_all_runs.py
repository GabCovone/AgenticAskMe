import json

transcript_path = r"C:\Users\gabri\.gemini\antigravity\brain\e1c8ba1c-edd9-404c-ac00-858544a03c26\.system_generated\logs\transcript_full.jsonl"

runs_found = []
with open(transcript_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        obj = json.loads(line)
        if obj.get("type") == "USER_INPUT":
            content = obj.get("content", "")
            if "AVVIO BENCHMARK SU 30 DOMANDE HOTPOTQA" in content:
                mode_str = "Unknown"
                if "Modalità di Retrieval Configurata: BASELINE" in content:
                    mode_str = "BASELINE"
                elif "Modalità di Retrieval Configurata: FULL" in content or "FULL" in content:
                    mode_str = "FULL"
                else:
                    # check if local kb or web search was used
                    if "duckduckgo" in content.lower() or "ddg" in content.lower() or "ricerca web" in content.lower():
                        mode_str = "FULL (detected web search)"
                    else:
                        mode_str = "FULL or default"
                runs_found.append((idx, obj.get("step_index"), mode_str, len(content)))

print(f"Total benchmark logs found in user inputs: {len(runs_found)}")
for r in runs_found:
    print(r)

