import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# =====================================================================\n",
        "# AVVIO DELLA DASHBOARD STREAMLIT IN GOOGLE COLAB\n",
        "# =====================================================================\n",
        "# NOTA: 'streamlit run Dashboard.py' è un comando da terminale/shell.\n",
        "# Per aprirlo da Google Colab, usiamo subprocess e serve_kernel_port:\n",
        "\n",
        "import subprocess\n",
        "import time\n",
        "\n",
        "# Avvia il server Streamlit in background sulla porta 8501\n",
        "process = subprocess.Popen([\n",
        "    \"streamlit\", \"run\", \"Dashboard.py\",\n",
        "    \"--server.port\", \"8501\",\n",
        "    \"--server.headless\", \"true\"\n",
        "])\n",
        "time.sleep(3)\n",
        "\n",
        "# Apri la Dashboard in un nuovo tab o finestra\n",
        "try:\n",
        "    from google.colab import output\n",
        "    print(\"\\n--- DASHBOARD AVVIATA IN GOOGLE COLAB ---\")\n",
        "    print(\"Clicca sul link sottostante o controlla il popup per visualizzare la Dashboard:\")\n",
        "    output.serve_kernel_port_as_window(8501)\n",
        "except ImportError:\n",
        "    print(\"\\n--- ESECUZIONE LOCALE RILEVATA ---\")\n",
        "    print(\"Sul tuo computer locale (PowerShell), esegui semplicemente:\")\n",
        "    print(\"  streamlit run Dashboard.py\")\n"
    ]
}

nb["cells"].append(new_cell)

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Added Streamlit launch cell to notebook.ipynb (total cells: {len(nb['cells'])}).")

