import os
import functools
from typing import TypedDict, List, Literal, Optional
from langgraph.graph import StateGraph, START, END

# --- STATO GLOBALE CONDIVISO DEL GRAFO ---
class GraphState(TypedDict):
    original_query: str
    current_query: str
    retrieved_context: str
    num_refinement: int
    num_planning: int
    next_node: str
    feedback_history: List[str]
    final_answer: str

# --- CONFIGURAZIONE CENTRALE DEL MODELLO ---
# Modello predefinito: Mistral-Nemo-Instruct (12B Q5_K_M) per ragionamento avanzato,
# oppure Gemma-2-9B-it (come citato nel README per hardware con VRAM limitata).
DEFAULT_REPO_ID = os.getenv("LLM_REPO_ID", "bartowski/Mistral-Nemo-Instruct-2407-GGUF")
DEFAULT_FILENAME = os.getenv("LLM_FILENAME", "Mistral-Nemo-Instruct-2407-Q5_K_M.gguf")
LOCAL_DIR = os.getenv("LLM_LOCAL_DIR", "./models")
LOCAL_PATH = os.path.join(LOCAL_DIR, DEFAULT_FILENAME)

# Temperatura impostata a 0.3 per favorire migliore flessibilità generativa e recupero da errori
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

_cached_llm = None

def get_llama_model(model_path: Optional[str] = None, temperature: float = DEFAULT_TEMPERATURE):
    """
    Inizializza e restituisce un'istanza singleton del modello LLM.
    Configurato con temperature=0.3 per bilanciare creatività nel refinement e precisione nel JSON parsing.
    """
    global _cached_llm
    if _cached_llm is not None:
        return _cached_llm
        
    target_path = model_path or LOCAL_PATH
    
    # Se il file locale non esiste, scarica da Hugging Face Hub
    if not os.path.exists(target_path):
        from huggingface_hub import hf_hub_download
        print(f"[INIT] Download del modello GGUF ({DEFAULT_FILENAME}) da HuggingFace...")
        os.makedirs(LOCAL_DIR, exist_ok=True)
        hf_hub_download(repo_id=DEFAULT_REPO_ID, filename=DEFAULT_FILENAME, local_dir=LOCAL_DIR)
        
    print(f"[INIT] Caricamento del modello GGUF in memoria (temperature={temperature})...")
    try:
        from langchain_community.chat_models import ChatLlamaCpp
        _cached_llm = ChatLlamaCpp(
            model_path=target_path,
            n_gpu_layers=-1,      # Offload completo su GPU (es. CUDA su Google Colab)
            n_ctx=4096,           # Finestra di contesto ampia per RAG multi-hop
            temperature=temperature,  # 0.3 per evitare collasso deterministico
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
