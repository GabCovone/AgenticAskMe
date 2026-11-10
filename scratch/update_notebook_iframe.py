import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

last_cell = nb["cells"][-1]
last_cell["source"] = [
    "# =====================================================================\n",
    "# AVVIO DELLA DASHBOARD STREAMLIT IN GOOGLE COLAB\n",
    "# =====================================================================\n",
    "!pip install -q streamlit\n",
    "\n",
    "import subprocess\n",
    "import sys\n",
    "import time\n",
    "\n",
    "process = subprocess.Popen([\n",
    "    sys.executable, \"-m\", \"streamlit\", \"run\", \"Dashboard.py\",\n",
    "    \"--server.port\", \"8501\",\n",
    "    \"--server.headless\", \"true\"\n",
    "])\n",
    "time.sleep(4)\n",
    "\n",
    "try:\n",
    "    from google.colab import output\n",
    "    print(\"\\n--- DASHBOARD ATTIVA IN GOOGLE COLAB ---\")\n",
    "    # Visualizzazione interattiva incorporata direttamente nella cella del notebook\n",
    "    output.serve_kernel_port_as_iframe(8501, height=900)\n",
    "except ImportError:\n",
    "    print(\"\\n--- ESECUZIONE LOCALE RILEVATA ---\")\n",
    "    print(\"Sul tuo computer locale, esegui da PowerShell: streamlit run Dashboard.py\")\n"
]

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated notebook with serve_kernel_port_as_iframe.")

