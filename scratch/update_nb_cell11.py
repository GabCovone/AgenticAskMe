import json

cell_11_code = [
    "# =====================================================================\n",
    "# AVVIO DELLA DASHBOARD STREAMLIT IN GOOGLE COLAB (Con Cloudflare Tunnel)\n",
    "# =====================================================================\n",
    "# Metodo ultra-affidabile: Cloudflare Tunnel non richiede password IP,\n",
    "# non soffre dei timeout 504 di LocalTunnel e gestisce i WebSockets perfettamente.\n",
    "\n",
    "!pip install -q streamlit\n",
    "\n",
    "# 1. Installa Cloudflared (tunnel pubblico sicuro ad alta velocita)\n",
    "!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb\n",
    "!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1\n",
    "\n",
    "import subprocess, sys, time, re\n",
    "\n",
    "# 2. Termina eventuali istanze orfane precedenti\n",
    "subprocess.run([\"pkill\", \"-f\", \"streamlit\"], capture_output=True)\n",
    "subprocess.run([\"pkill\", \"-f\", \"cloudflared\"], capture_output=True)\n",
    "time.sleep(1)\n",
    "\n",
    "# 3. Avvia Streamlit in background con parametri ottimizzati per Colab\n",
    "st_proc = subprocess.Popen([\n",
    "    sys.executable, \"-m\", \"streamlit\", \"run\", \"Dashboard.py\",\n",
    "    \"--server.port\", \"8501\",\n",
    "    \"--server.headless\", \"true\",\n",
    "    \"--server.enableCORS\", \"false\",\n",
    "    \"--server.enableXsrfProtection\", \"false\"\n",
    "])\n",
    "time.sleep(3)\n",
    "\n",
    "# 4. Avvia il tunnel Cloudflare verso la porta 8501\n",
    "tunnel_proc = subprocess.Popen(\n",
    "    [\"cloudflared\", \"tunnel\", \"--url\", \"http://localhost:8501\"],\n",
    "    stdout=subprocess.PIPE,\n",
    "    stderr=subprocess.STDOUT,\n",
    "    text=True\n",
    ")\n",
    "\n",
    "print(\"\\n\" + \"=\"*70)\n",
    "print(\"ATTIVAZIONE TUNNEL CLOUDFLARE PER LA DASHBOARD...\")\n",
    "for line in iter(tunnel_proc.stdout.readline, \"\"):\n",
    "    if \"trycloudflare.com\" in line:\n",
    "        url_match = re.search(r\"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com\", line)\n",
    "        if url_match:\n",
    "            print(f\"\\nLINK DIRETTO DASHBOARD STREAMLIT:\")\n",
    "            print(f\"👉 {url_match.group(0)}\")\n",
    "            print(\"=\"*70 + \"\\n\")\n",
    "            print(\"Clicca sul link qui sopra per aprire la Dashboard completa in una nuova scheda!\")\n",
    "            break\n"
]

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"][11]["source"] = cell_11_code

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook cell 11 updated with Cloudflared tunnel successfully!")

