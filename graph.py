import os
import functools
import warnings
import logging

# Disattiva tutti i warning di deprecazione e runtime (incluso datetime.utcnow() di jupyter_client su Python 3.13)
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

from typing import TypedDict, List, Literal, Optional
from langgraph.graph import StateGraph, START, END
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
    Risolve il problema di linking dinamico su Linux / Google Colab (libcudart.so.12 / libcublas.so.12)
    precaricando le librerie condivise di runtime nella tabella globale con ctypes.RTLD_GLOBAL.
    """
    import sys
    if not sys.platform.startswith("linux"):
        return

    import os, glob, ctypes

    search_dirs = []
    # 1. Pacchetti Python nvidia.* installati con pip
    try:
        import nvidia.cuda_runtime
        search_dirs.append(os.path.join(nvidia.cuda_runtime.__path__[0], "lib"))
    except Exception:
        pass
        
    try:
        import nvidia.cublas
        search_dirs.append(os.path.join(nvidia.cublas.__path__[0], "lib"))
    except Exception:
        pass

    # 2. Percorsi standard CUDA di Linux / Google Colab
    search_dirs.extend([
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/lib/x86_64-linux-gnu"
    ])
    search_dirs.extend(glob.glob("/usr/local/cuda-12.*/lib64"))

    targets = ["libcublasLt.so.12", "libcublas.so.12", "libcudart.so.12"]
    loaded = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
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
        try:
            _cached_llm = ChatLlamaCpp(
                model_path=valid_target_path,
                n_gpu_layers=-1,      # Offload completo su GPU (es. CUDA su Google Colab)
                n_ctx=4096,           # Finestra di contesto ampia per RAG multi-hop
                temperature=temperature,
                max_tokens=2048,
                repeat_penalty=1.15,
                verbose=False
            )
        except Exception as e_gpu:
            print(f"[INIT WARNING] Offload n_gpu_layers=-1 fallito ({e_gpu}). Tentativo con n_gpu_layers=33...")
            _cached_llm = ChatLlamaCpp(
                model_path=valid_target_path,
                n_gpu_layers=33,      # Offload parziale sicuro per GPU con VRAM ridotta
                n_ctx=4096,
                temperature=temperature,
                max_tokens=2048,
                repeat_penalty=1.15,
                verbose=False
            )
        return _cached_llm
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
