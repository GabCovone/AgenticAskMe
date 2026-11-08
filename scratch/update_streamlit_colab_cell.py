import json
import sys

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

last_cell = nb["cells"][-1]
last_cell["source"] = [
    "# =====================================================================\n",
    "# AVVIO DELLA DASHBOARD STREAMLIT IN GOOGLE COLAB\n",
    "# =====================================================================\n",
    "# 1. Installazione automatica di Streamlit (se non presente nel runtime Colab)\n",
    "!pip install -q streamlit\n",
    "\n",
    "import subprocess\n",
    "import sys\n",
    "import time\n",
    "\n",
    "# 2. Avvia il server Streamlit in background con l'interprete Python corrente\n",
    "process = subprocess.Popen([\n",
    "    sys.executable, \"-m\", \"streamlit\", \"run\", \"Dashboard.py\",\n",
    "    \"--server.port\", \"8501\",\n",
    "    \"--server.headless\", \"true\"\n",
    "])\n",
    "time.sleep(4)\n",
    "\n",
    "# 3. Esponi la porta 8501 in un nuovo tab del browser\n",
    "try:\n",
    "    from google.colab import output\n",
    "    print(\"\\n--- DASHBOARD AVVIATA IN GOOGLE COLAB ---\")\n",
    "    print(\"Clicca sul pulsante/link generato sottostante per visualizzare la Dashboard:\")\n",
    "    output.serve_kernel_port_as_window(8501)\n",
    "except ImportError:\n",
    "    print(\"\\n--- ESECUZIONE LOCALE RILEVATA ---\")\n",
    "    print(\"Sul tuo computer locale, esegui da PowerShell: streamlit run Dashboard.py\")\n"
]

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated last cell in notebook.ipynb successfully.")

