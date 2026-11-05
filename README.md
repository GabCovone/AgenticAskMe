# Agentic Information Retrieval (IR) System

This project implements an advanced **Agentic Information Retrieval (IR)** system based on Large Language Models (LLMs) and multi-agent orchestration.
The core goal is to overcome the structural limitations of traditional, unidirectional Retrieval-Augmented Generation (RAG) by introducing a cyclical, self-reflexive, and grounded multi-agent architecture.

The system manages complex, multi-hop information needs, evaluates query ambiguity, decomposes questions via Least-to-Most prompting, dynamically selects between a **Local Wikipedia Knowledge Base** and **Live Web Search**, and self-corrects using verbal reflection and critique tokens. Orchestration is powered by **LangGraph**, while inference is optimized for quantized LLMs (e.g. *Mistral-Nemo-Instruct* or *Gemma-2-9B-it*) running on GPU or Google Colab environments.

---

## 🏛️ System Architecture

The architecture is built upon a cyclic state graph (`GraphState`) shared across three specialized agent modules. To minimize latency and cognitive drift, the system couples module-level autonomy with centralized semantic routing:

```
                    +------------------------+
                    |    User Input Query    |
                    +------------------------+
                                |
                                v
                    +------------------------+ <----------------------+
                    |     Validator Node     |                        |
                    | (Adaptive + Self-RAG)  |                        |
                    +------------------------+                        |
                      /         |          \                          |
           (Refine)  /          |           \ (Finish)                |
                    v           v            v                        |
             +-----------+ +-----------+  [ END ]                     |
             |  Refiner  | |  Planner  | (Final Grounded Answer)      |
             +-----------+ +-----------+                              |
                    \           /                                     |
                     +---------+--------------------------------------+
```

### Execution Lifecycle:
1. **Input**: The user inputs a query (factoid, ambiguous, comparison, or multi-hop).
2. **Routing & Assessment (Validator)**: The Validator performs semantic classification (`query_complexity`), decides whether refinement is required, or routes straight to retrieval planning.
3. **Autonomous Query Refinement (Refiner)**: Decomposes, rewrites, or semantically expands the query without performing retrieval, preserving explicit named entities.
4. **Adaptive Retrieval Planning (Planner)**: Classifies the retrieval strategy, dynamically decides between the **Local Wikipedia KB** and **Live Web Search**, and runs an interactive **ReAct** (Reasoning + Acting) loop with observation previews.
5. **Grounded Synthesis & Quality Critique (Validator)**: The Validator evaluates the proposed answer against the retrieved evidence using explicit Self-RAG critique tokens (`is_context_relevant`, `is_query_answered`). If evidence is missing or hallucinated, it generates verbal critique (`Reflexion`) and redirects the agent.

---

## 📚 Academic Foundations & Literature Mapping

The system directly implements the state-of-the-art (SOTA) frameworks from the literature available in `Paper/`:

| Framework / Paper | Reference | Implementation in Code |
| :--- | :--- | :--- |
| **Agentic IR** | Zhang et al. (CIKM '24) & Wu et al. (2025) | Cyclic graph replacing passive one-shot RAG with autonomous reasoning, dynamic tools, and self-evaluation. |
| **PaperQA** | Lála et al. (2023) | Strict contextual grounding (`Planner.py` synthesis), avoiding hallucinations with explicit "insufficient information" fallbacks. |
| **Self-RAG** | Asai et al. (ICLR 2024) | Critique tokens `[IsRel]` (`is_context_relevant`) and `[IsSup]` (`is_query_answered`) in `Validator.py` to veto unsupported outputs. |
| **Reflexion** | Shinn et al. (NeurIPS 2023) | Episodic verbal memory buffer (`feedback_history`) passed across iterations to steer agents away from past failures. |
| **Least-to-Most Prompting** | Zhou et al. (ICLR 2023) | `decompose_query` tool in `Refiner.py` to break composite or comparison questions into atomic sub-questions. |
| **Rewrite-Retrieve-Read** | Ma et al. (EMNLP 2023) | `rewrite_query` tool in `Refiner.py` to strip conversational noise and optimize search keywords. |
| **Query2doc** | Wang et al. (EMNLP 2023) | `expand_query` in `Refiner.py` generating synthetic pseudo-documents to maximize sparse/dense retrieval recall. |
| **Adaptive-RAG** | Jeong et al. (NAACL 2024) | `AdaptiveStrategy` in `Planner.py` and `query_complexity` in `Validator.py` to adapt retrieval effort to question difficulty. |
| **ReAct** | Yao et al. (ICLR 2023) | Interleaving reasoning (`Thought:`) with tool calls (`Action:`) and environment data (`Observation:`) in `Planner.py`. |
| **IRCoT** | Trivedi et al. (ACL 2023) | Interleaving retrieval with Chain-of-Thought, accumulating `global_context` across sub-queries and iterations. |
| **Self-Ask** | Press et al. (EMNLP 2023) | Follow-up question generation to resolve missing intermediate entities before final synthesis. |

---

## 🧩 Modules in Detail

### 1. The Validator (`Validator.py`)
Acts as the conductor (input router) and final judge:
* **Self-RAG Critique Tokens**: Evaluates evidence relevance (`is_context_relevant` [IsRel]) and factual grounding (`is_query_answered` [IsSup]). Vetoes premature termination if the candidate answer contains ungrounded claims.
* **Reflexion Feedback**: Injects detailed verbal guidance into `feedback_history` upon retrieval failures, prompting the agent to revise search terms.
* **Sliding-Window Robust JSON Parser**: Resilient parser that scans for nested JSON objects and handles raw string arguments and malformed escapes seamlessly.

### 2. The Refiner (`Refiner.py`)
Responsible for **Autonomous Query Refinement**:
* **Decomposition Tool (`decompose_query`)**: Scomposes multi-entity or comparison questions into parallel atomic queries (Least-to-Most).
* **Rewriting Tool (`rewrite_query`)**: Cleans syntax, resolves noise, and retains explicit named entities (Rewrite-Retrieve-Read).
* **Semantic Expansion Tool (`expand_query`)**: Synthesizes pseudo-documents to boost retrieval vocabulary (Query2doc).
* **Atomic Preservation (`keep_query_unchanged`)**: Leaves simple queries intact when no refinement is necessary.

### 3. The Planner (`Planner.py`)
The operational retrieval engine:
* **Dual-Source Retrieval**:
  - `local_kb_search`: Searches the local Wikipedia Knowledge Base (SQLite FTS5 + BM25Okapi).
  - `web_search`: Live search on the open web via DuckDuckGo.
* **Autonomous Source Decision**: In `AdaptiveStrategy`, the agent evaluates whether the query requires encyclopedic knowledge (`local_kb`), live/recent information (`web`), or both (`both`).
* **Multi-Step ReAct Loop**: Coordinates search actions, logs intermediate thoughts, and handles observations step-by-step.
* **Cumulative Context (IRCoT)**: Preserves `global_context` across multi-hop reasoning passes.

### 4. Local Wikipedia Knowledge Base (`local_kb.py`)
* Embedded SQLite database (`wikipedia_local.db`) equipped with **FTS5 full-text indexing** and **BM25Okapi** ranking.
* Pre-seeded with curated Wikipedia articles spanning Computer Science, AI, Biographies, Science, Geography, Cinema, and Literature.
* In-memory index caching for sub-millisecond retrieval latency during agentic loops.
* Extensible via `add_article(title, text, category)` and batch HuggingFace dataset import (`build_index`).

### 5. Interactive Dashboard (`Dashboard.py`)
* Streamlit-based web dashboard offering:
  - Live query testing and execution tracing.
  - Visualization of Adaptive-RAG decisions and Self-RAG critique tokens.
  - Direct exploration and search in the local Wikipedia Knowledge Base.
  - Ingestion tool to add custom Wikipedia articles directly from the UI.

---

## ⚙️ Model Parameters & Temperature Setting

* **Default Temperature**: `0.3` (configurable via `TEMPERATURE=0.3`).
  - *Rationale*: A temperature of `0.1` can cause deterministic mode collapse and rigid query rewriting, hindering recovery when initial searches return zero matches. Setting the temperature to `0.3` provides the required flexibility for creative query expansion and alternative keyword formulation, while maintaining strict JSON structural adherence.
* **Model Choices**:
  - `bartowski/Mistral-Nemo-Instruct-2407-GGUF` (Q5_K_M, ~8.5 GB) - High reasoning capability.
  - `bartowski/gemma-2-9b-it-GGUF` (Q4_K_M, ~6.5 GB) - Fits within free Google Colab T4 GPU (15 GB VRAM).

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install langchain langgraph langchain-community huggingface-hub pydantic rank_bm25 datasets duckduckgo-search streamlit pandas
```
*For GPU acceleration on Google Colab / CUDA:*
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 -q
```

### 2. Initialize Local Wikipedia KB
```bash
python local_kb.py
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run Dashboard.py
```

### 4. Run Benchmark in Notebook
Open and run `notebook.ipynb` in Google Colab or your local Jupyter environment.
