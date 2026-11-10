import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 10: Export & Download
cell_export = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# =====================================================================\n",
        "# ESPORTAZIONE E DOWNLOAD DEL DATABASE E DEL FILE JSON\n",
        "# =====================================================================\n",
        "# Esporta tutti i test eseguiti (inclusa la modalità 'system_mode') e scarica lo zip\n",
        "\n",
        "import sqlite3, json, zipfile\n",
        "from google.colab import files\n",
        "\n",
        "# 1. Estrai tutti i record da agentic_history.db\n",
        "conn = sqlite3.connect(\"agentic_history.db\")\n",
        "conn.row_factory = sqlite3.Row\n",
        "rows = [dict(r) for r in conn.cursor().execute(\"SELECT * FROM query_runs ORDER BY id ASC\").fetchall()]\n",
        "conn.close()\n",
        "\n",
        "# 2. Salva in formato JSON con system_mode\n",
        "with open(\"benchmark_history.json\", \"w\", encoding=\"utf-8\") as f:\n",
        "    json.dump(rows, f, indent=2, ensure_ascii=False)\n",
        "\n",
        "# 3. Crea l'archivio ZIP con DB e JSON per evitare blocchi del browser\n",
        "with zipfile.ZipFile(\"benchmark_results.zip\", \"w\") as zipf:\n",
        "    zipf.write(\"agentic_history.db\")\n",
        "    zipf.write(\"benchmark_history.json\")\n",
        "\n",
        "print(f\"Archiviati con successo {len(rows)} record (modalità: full/baseline).\")\n",
        "print(\"Avvio download automatico di 'benchmark_results.zip'...\")\n",
        "files.download(\"benchmark_results.zip\")\n"
    ]
}

# Cell 11: Launch Streamlit in Colab with localtunnel & CORS disabled
cell_streamlit = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# =====================================================================\n",
        "# AVVIO DELLA DASHBOARD STREAMLIT IN GOOGLE COLAB (Tunnel Diretto)\n",
        "# =====================================================================\n",
        "# 1. Installa Streamlit e LocalTunnel per tunneling sicuro con WebSockets\n",
        "!pip install -q streamlit\n",
        "!npm install -g localtunnel > /dev/null\n",
        "\n",
        "import subprocess, sys, time, urllib.request\n",
        "\n",
        "# 2. Recupera l'IP pubblico di Colab (necessario come password su LocalTunnel)\n",
        "try:\n",
        "    colab_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()\n",
        "except Exception:\n",
        "    colab_ip = \"N/A\"\n",
        "\n",
        "# 3. Avvia Streamlit disabilitando CORS e XSRF (indispensabili su Colab)\n",
        "process = subprocess.Popen([\n",
        "    sys.executable, \"-m\", \"streamlit\", \"run\", \"Dashboard.py\",\n",
        "    \"--server.port\", \"8501\",\n",
        "    \"--server.headless\", \"true\",\n",
        "    \"--server.enableCORS\", \"false\",\n",
        "    \"--server.enableXsrfProtection\", \"false\"\n",
        "])\n",
        "time.sleep(4)\n",
        "\n",
        "print(\"\\n\" + \"=\"*65)\n",
        "print(f\"🔑 PASSWORD LOCALTUNNEL (Inserisci questo IP): {colab_ip}\")\n",
        "print(\"=\"*65)\n",
        "print(\"Clicca sul link sottostante ('your url is: ...') per aprire la Dashboard:\\n\")\n",
        "\n",
        "# 4. Espone la porta su URL pubblico\n",
        "!npx localtunnel --port 8501\n"
    ]
}

# Replace cell 10 and append cell 11
if len(nb["cells"]) >= 11:
    nb["cells"][10] = cell_export
    nb["cells"] = nb["cells"][:11] + [cell_streamlit]
else:
    nb["cells"] = nb["cells"][:10] + [cell_export, cell_streamlit]

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Updated notebook.ipynb with export cell (10) and streamlit cell (11). Total cells: {len(nb['cells'])}")

