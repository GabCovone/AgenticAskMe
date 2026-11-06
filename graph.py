import os
import functools
import warnings
import logging

# Disattiva tutti i warning di deprecazione e runtime (incluso datetime.utcnow() di jupyter_client su Python 3.13)
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

import importlib
from typing import TypedDict, List, Literal, Optional
from langgraph.graph import StateGraph, START, END
import state as _state_mod
importlib.reload(_state_mod)
from state import GraphState

# --- CONFIGURAZIONE CENTRALE DEL MODELLO ---
# Modello predefinito: Mistral-Nemo-Instruct (12B Q5_K_M) per ragionamento avanzato,
# oppure Gemma-2-9B-it (come citato nel README per hardware con VRAM limitata).
DEFAULT_REPO_ID = os.getenv("LLM_REPO_ID", "bartowski/Mistral-Nemo-Instruct-2407-GGUF")
DEFAULT_FILENAME = os.getenv("LLM_FILENAME", "Mistral-Nemo-Instruct-2407-Q5_K_M.gguf")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.getenv("LLM_LOCAL_DIR", os.path.join(REPO_ROOT, "models"))
LOCAL_PATH = os.path.join(LOCAL_DIR, DEFAULT_FILENAME)

# Temperatura impostata a 0.3 per favorire migliore flessibilità generativa e recupero da errori
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

_cached_llm = None

def _preload_cuda_libraries():
    """
    Risolve il problema di linking dinamico su Linux / Google Colab (libcuda.so.1 / libcudart.so.12 / libcublas.so.12)
    precaricando le librerie condivise del driver e di runtime nella tabella globale con ctypes.RTLD_GLOBAL.
    """
    import sys
    if not sys.platform.startswith("linux"):
        return

    import os, glob, ctypes

    search_dirs = [
        "/usr/lib64-nvidia",
        "/usr/local/cuda/compat",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12/compat",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu/nvidia",
        "/usr/lib64",
        "/usr/local/cuda/lib64/stubs",
    ]

    # 1. Pacchetti Python nvidia.* installati con pip
    for pkg in ["nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cuda_nvrtc", "nvidia.cuda_cupti"]:
        try:
            mod = __import__(pkg, fromlist=["__path__"])
            p = os.path.join(mod.__path__[0], "lib")
            if os.path.isdir(p) and p not in search_dirs:
                search_dirs.append(p)
        except Exception:
            pass

    # 2. Percorsi standard CUDA e driver di Linux / Google Colab
    search_dirs.extend(glob.glob("/usr/lib64-nvidia*"))
    search_dirs.extend(glob.glob("/usr/local/cuda*/lib64"))
    search_dirs.extend(glob.glob("/usr/local/cuda*/compat"))
    search_dirs.extend(glob.glob("/usr/lib/x86_64-linux-gnu/nvidia*"))

    # Crea un symlink di compatibilità in /tmp/cuda_compat se esiste libcuda.so ma non libcuda.so.1
    compat_dir = "/tmp/cuda_compat"
    try:
        os.makedirs(compat_dir, exist_ok=True)
        if compat_dir not in search_dirs:
            search_dirs.insert(0, compat_dir)
        libcuda_target = os.path.join(compat_dir, "libcuda.so.1")
        if not os.path.exists(libcuda_target):
            for d in search_dirs:
                if not os.path.isdir(d) or d == compat_dir:
                    continue
                candidates = glob.glob(os.path.join(d, "libcuda.so*"))
                real_cands = [c for c in candidates if os.path.exists(os.path.realpath(c))]
                if real_cands:
                    try:
                        os.symlink(real_cands[0], libcuda_target)
                        break
                    except Exception:
                        pass
    except Exception:
        pass

    # Aggiorna anche os.environ['LD_LIBRARY_PATH'] con tutti i percorsi validi trovati
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    valid_dirs = [os.path.abspath(d) for d in search_dirs if os.path.isdir(d)]
    new_parts = [d for d in valid_dirs if d not in existing_ld.split(":")]
    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts) + (":" + existing_ld if existing_ld else "")

    targets = [
        "libcuda.so.1",
        "libcuda.so",
        "libcublasLt.so.12",
        "libcublasLt.so",
        "libcublas.so.12",
        "libcublas.so",
        "libcudart.so.12",
        "libcudart.so",
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
    ]
    loaded = []
    for d in valid_dirs:
        for target in targets:
            fpath = os.path.join(d, target)
            if os.path.exists(fpath) and target not in loaded:
                try:
                    ctypes.CDLL(fpath, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(target)
                except Exception:
                    pass

    if loaded:
        print(f"[INIT] Librerie CUDA precaricate con successo: {', '.join(loaded)}")
    elif sys.platform.startswith("linux"):
        print("[INIT WARNING] Nessuna libreria driver CUDA (libcuda.so.1) trovata. Se sei su Google Colab, assicurati che il tipo di runtime sia impostato su 'T4 GPU' (Runtime -> Cambia tipo di runtime).")

def get_llama_model(model_path: Optional[str] = None, temperature: float = DEFAULT_TEMPERATURE):
    """
    Inizializza e restituisce un'istanza singleton del modello LLM.
    Configurato con temperature=0.3 per bilanciare creatività nel refinement e precisione nel JSON parsing.
    Verifica l'integrità del file GGUF prima del caricamento (riscarica se incompleto o corrotto).
    """
    global _cached_llm
    if _cached_llm is not None:
        return _cached_llm
        
    _preload_cuda_libraries()

    # Cerca il file in percorsi noti e ne verifica l'integrità
    candidate_paths = [
        model_path,
        LOCAL_PATH,
        os.path.join(REPO_ROOT, "models", DEFAULT_FILENAME),
        os.path.join(os.getcwd(), "models", DEFAULT_FILENAME),
        os.path.join("/content", "models", DEFAULT_FILENAME),
        os.path.join("/content", "AgenticAskMe", "models", DEFAULT_FILENAME),
        f"./models/{DEFAULT_FILENAME}"
    ]
    
    valid_target_path = None
    MIN_VALID_SIZE = 5 * 1024 * 1024 * 1024  # Almeno 5 GB per Mistral-Nemo-Q5 (pesa ~8.7 GB)

    for p in candidate_paths:
        if not p:
            continue
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            fsize = os.path.getsize(abs_p)
            if fsize >= MIN_VALID_SIZE:
                valid_target_path = abs_p
                break
            else:
                print(f"[INIT WARNING] File modello incompleto o corrotto rilevato in {abs_p} ({fsize / (1024**2):.1f} MB). Rimozione in corso...")
                try:
                    os.remove(abs_p)
                except Exception:
                    pass

    if not valid_target_path:
        from huggingface_hub import hf_hub_download
        print(f"[INIT] Download del modello GGUF ({DEFAULT_FILENAME}) da HuggingFace ({DEFAULT_REPO_ID})...")
        os.makedirs(LOCAL_DIR, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=DEFAULT_REPO_ID,
            filename=DEFAULT_FILENAME,
            local_dir=LOCAL_DIR
        )
        valid_target_path = os.path.abspath(downloaded)

    print(f"[INIT] Caricamento del modello GGUF in memoria da {valid_target_path} (temperature={temperature})...")
    try:
        from langchain_community.chat_models import ChatLlamaCpp
        last_error = None
        for gpu_layers, ctx_len in [(-1, 8192), (33, 8192), (0, 4096)]:
            tier_name = (
                "GPU Completo (n_gpu_layers=-1)" if gpu_layers == -1
                else (f"GPU Parziale (n_gpu_layers={gpu_layers})" if gpu_layers > 0 else "CPU Fallback (n_gpu_layers=0)")
            )
            try:
                _cached_llm = ChatLlamaCpp(
                    model_path=valid_target_path,
                    n_gpu_layers=gpu_layers,
                    n_ctx=ctx_len,
                    temperature=temperature,
                    max_tokens=512,
                    repeat_penalty=1.05,
                    streaming=False,
                    verbose=False
                )
                print(f"[INIT] Modello LLM caricato con successo in modalità: {tier_name} (n_ctx={ctx_len})!")
                return _cached_llm
            except Exception as e_tier:
                last_error = e_tier
                print(f"[INIT WARNING] Tentativo {tier_name} fallito: {e_tier}")

        if last_error:
            err_str = str(last_error)
            if "libcuda.so" in err_str or "libllama.so" in err_str:
                print("\n" + "="*70)
                print("[ERRORE CRITICO: DRIVER GPU NON RILEVATO SU GOOGLE COLAB]")
                print("Llama-cpp non ha trovato 'libcuda.so.1' sul sistema.")
                print("La causa è che la sessione di Google Colab è attualmente impostata su CPU.")
                print("\nCOME RISOLVERE IN 2 CLICK:")
                print("1. Nel menu in alto di Colab clicca su: Runtime -> Cambia tipo di runtime")
                print("2. Sotto 'Acceleratore hardware' seleziona: T4 GPU")
                print("3. Clicca su 'Salva' e riesegui le celle a partire da !nvidia-smi.")
                print("="*70 + "\n")
            raise last_error
    except ImportError as e:
        print(f"[INIT WARNING] llama-cpp-python non disponibile nell'ambiente locale: {e}")
        raise

# Import dei nodi dopo la definizione di GraphState
from Validator import validator_node
from Refiner import refiner_node
from Planner import planner_node

# --- ROUTER CONDIZIONALE ---
def route_after_validator(state: GraphState) -> Literal["refinement_node", "planning_node", "__end__"]:
    """Legge lo stato aggiornato dal validatore e instrada al nodo successivo."""
    next_action = state.get("next_node")
    if next_action == "route_to_refinement":
        return "refinement_node"
    elif next_action == "route_to_planning":
        return "planning_node"
    else:
        return END

# --- COSTRUZIONE DEL GRAFO LANGGRAPH ---
def build_graph(llm_instance=None):
    """
    Costruisce e compila il grafo di orchestrazione LangGraph.
    Se llm_instance non è fornito, richiama get_llama_model().
    """
    shared_llm = llm_instance if llm_instance is not None else get_llama_model()

    workflow = StateGraph(GraphState)

    # Nodi principali con iniezione del modello condiviso
    workflow.add_node("validator_node", functools.partial(validator_node, llm=shared_llm))
    workflow.add_node("refinement_node", functools.partial(refiner_node, llm=shared_llm))
    workflow.add_node("planning_node", functools.partial(planner_node, llm=shared_llm))

    # Punto di ingresso: il flusso inizia sempre dal validatore
    workflow.add_edge(START, "validator_node")

    # Routing condizionale dal validatore
    workflow.add_conditional_edges(
        "validator_node",
        route_after_validator,
        {
            "refinement_node": "refinement_node",
            "planning_node": "planning_node",
            END: END
        }
    )

    # I nodi satellite riportano sempre il controllo al validatore per la chiusura del ciclo
    workflow.add_edge("refinement_node", "validator_node")
    workflow.add_edge("planning_node", "validator_node")

    app = workflow.compile()
    return app

if __name__ == "__main__":
    print("Modulo graph.py verificato. Eseguire notebook.ipynb o Dashboard.py per avviare il sistema.")
