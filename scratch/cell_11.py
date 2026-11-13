# =====================================================================
# AVVIO DELLA DASHBOARD STREAMLIT IN GOOGLE COLAB (Tunnel Diretto)
# =====================================================================
# 1. Installa Streamlit e LocalTunnel per tunneling sicuro con WebSockets
!pip install -q streamlit
!npm install -g localtunnel > /dev/null

import subprocess, sys, time, urllib.request

# 2. Recupera l'IP pubblico di Colab (necessario come password su LocalTunnel)
try:
    colab_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()
except Exception:
    colab_ip = "N/A"

# 3. Avvia Streamlit disabilitando CORS e XSRF (indispensabili su Colab)
process = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "Dashboard.py",
    "--server.port", "8501",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false"
])
time.sleep(4)

print("\n" + "="*65)
print(f"🔑 PASSWORD LOCALTUNNEL (Inserisci questo IP): {colab_ip}")
print("="*65)
print("Clicca sul link sottostante ('your url is: ...') per aprire la Dashboard:\n")

# 4. Espone la porta su URL pubblico
!npx localtunnel --port 8501
