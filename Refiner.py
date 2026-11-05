import json
import ast
import re
import sys
import warnings
import logging
from typing import List, Set, Optional, Dict, Any
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from state import GraphState

# Suppress runtime and deprecation warnings
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --- 1. DEFINIZIONE DEI TOOL DI REFINEMENT ---

@tool
def decompose_query(reflection: str, reasoning: str, sub_queries: List[str]) -> List[str]:
    """
    Applies the 'Least-to-Most Prompting' framework (Zhou et al., 2022) to decompose a complex query
    into simpler, logically sequential or parallel sub-questions.
    WHEN TO USE:
    - Queries that compare two or more entities (e.g. "Were X and Y born in the same country?").
    - Multi-hop questions with multiple independent facts needed to construct the final answer.
    ARGUMENTS:
    - reflection: Critique of previous attempts based on the Validator's feedback.
    - reasoning: Step-by-step reasoning explaining why and how the query is broken down.
    - sub_queries: A list of clean, self-contained sub-questions. Do NOT drop explicit named entities!
    """
    return sub_queries

@tool
def expand_query(reflection: str, reasoning: str, pseudo_document: str) -> str:
    """
    Applies the 'Query2doc' framework (Wang et al., 2023) for semantic query expansion.
    WHEN TO USE:
    - Queries that are short, telegraphic, or lack background vocabulary for BM25/retrieval.
    ARGUMENTS:
    - reflection: Critique of previous attempts.
    - reasoning: Explanation of which domain concepts need expansion.
    - pseudo_document: A synthetic, factual passage generated from model knowledge to enrich retrieval vocabulary.
    """
    return pseudo_document

@tool
def rewrite_query(reflection: str, reasoning: str, rewritten_query: str) -> str:
    """
    Applies the 'Rewrite-Retrieve-Read' framework (Ma et al., 2023) for syntactic and entity correction.
    WHEN TO USE:
    - Queries with conversational filler ("hey can you tell me...", "I was wondering..."), typos, or ambiguous syntax.
    ARGUMENTS:
    - reflection: Critique of past attempts.
    - reasoning: Justification for the rewrites.
    - rewritten_query: The cleaned, search-engine-optimized query. ALWAYS preserve explicit named entities!
    """
    return rewritten_query

@tool
def keep_query_unchanged(reflection: str, reasoning: str) -> str:
    """
    Use this tool when the query is already atomic, unambiguous, and ready for retrieval without modification.
    ARGUMENTS:
    - reflection: Acknowledgement of current state and Validator feedback.
    - reasoning: Justification for keeping the query in its current form.
    """
    return "UNCHANGED"


def extract_refiner_tool_calls(response_msg, content_resp: str) -> List[Dict[str, Any]]:
    """
    Extracts refiner tool calls via:
    1. LangChain native tool_calls
    2. Markdown fenced JSON
    3. Sliding-window JSON parser
    4. Text question extraction
    """
    tool_calls_list = []
    
    # 1. Native tool calls
    if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
        for tc in response_msg.tool_calls:
            args = tc.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if args:
                tool_calls_list.append({"name": tc.get("name"), "args": args})
        if tool_calls_list:
            return tool_calls_list

    # 2. Markdown fenced JSON
    md_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content_resp, re.DOTALL)
    for m in md_matches:
        try:
            parsed = json.loads(m.replace(": False", ": false").replace(": True", ": true"))
            if isinstance(parsed, dict):
                t_name = parsed.get("name")
                t_args = parsed.get("arguments", parsed.get("args", parsed.get("parameters", parsed)))
                if isinstance(t_args, str):
                    try:
                        t_args = json.loads(t_args)
                    except Exception:
                        pass
                if not isinstance(t_args, dict):
                    t_args = parsed
                if not t_name:
                    sub_keys = ["sub_queries", "queries", "subqueries", "sub-queries", "questions", "sub_questions", "subquestions"]
                    if any(k in t_args for k in sub_keys):
                        t_name = "decompose_query"
                    elif "rewritten_query" in t_args or "query" in t_args:
                        t_name = "rewrite_query"
                    elif "pseudo_document" in t_args or "expansion" in t_args:
                        t_name = "expand_query"
                    else:
                        t_name = "keep_query_unchanged"
                tool_calls_list.append({"name": t_name, "args": t_args})
        except Exception:
            pass
    if tool_calls_list:
        return tool_calls_list

    # 3. Sliding-window JSON parser
    i = 0
    while i < len(content_resp):
        if content_resp[i] == '{':
            brace_level = 0
            in_string = False
            escape = False
            for j in range(i, len(content_resp)):
                char = content_resp[j]
                if char == '"' and not escape:
                    in_string = not in_string
                if not in_string:
                    if char == '{':
                        brace_level += 1
                    elif char == '}':
                        brace_level -= 1
                if brace_level == 0:
                    try:
                        json_str = content_resp[i:j+1]
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            t_name = parsed.get("name")
                            t_args = parsed.get("arguments", parsed.get("args", parsed.get("parameters", parsed)))
                            if isinstance(t_args, str):
                                try:
                                    t_args = json.loads(t_args)
                                except Exception:
                                    pass
                            if not isinstance(t_args, dict):
                                t_args = parsed
                            
                            if not t_name:
                                sub_keys = ["sub_queries", "queries", "subqueries", "sub-queries", "questions", "sub_questions", "subquestions"]
                                if any(k in t_args for k in sub_keys):
                                    t_name = "decompose_query"
                                elif "rewritten_query" in t_args or "query" in t_args:
                                    t_name = "rewrite_query"
                                elif "pseudo_document" in t_args or "expansion" in t_args:
                                    t_name = "expand_query"
                                else:
                                    t_name = "keep_query_unchanged"
                                    
                            if t_name:
                                tool_calls_list.append({"name": t_name, "args": t_args})
                    except Exception:
                        pass
                    break
                if char == '\\':
                    escape = not escape
                else:
                    escape = False
        i += 1
    if tool_calls_list:
        return tool_calls_list

    # 4. Text question extraction fallback (e.g. "1. What is ... ? 2. What is ... ?")
    numbered_questions = re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]|\-|\*)\s*(.*?\?)", content_resp)
    if len(numbered_questions) >= 2:
        clean_qs = [q.strip() for q in numbered_questions if len(q.strip()) > 5]
        if clean_qs:
            tool_calls_list.append({
                "name": "decompose_query",
                "args": {
                    "reflection": "Extracted sub-queries from model response text.",
                    "reasoning": "Decomposed into numbered questions.",
                    "sub_queries": clean_qs
                }
            })

    return tool_calls_list


def heuristic_least_to_most(query: str) -> Optional[List[str]]:
    """
    Applies deterministic Least-to-Most decomposition (Zhou et al., 2022)
    when comparative or multi-entity queries are detected.
    E.g. 'Were X and Y of the same nationality?' ->
         ['What is the nationality of X?', 'What is the nationality of Y?']
    """
    clean_q = query.strip().rstrip("?").strip()
    pattern = r"^(?:Were|Did|Are|Is)\s+([A-Za-z\s\.]+?)\s+and\s+([A-Za-z\s\.]+?)\s+(?:of the same|share the same|born in the same|in the same)\s+([a-zA-Z\s]+)$"
    match = re.search(pattern, clean_q, re.IGNORECASE)
    if match:
        e1 = match.group(1).strip()
        e2 = match.group(2).strip()
        trait = match.group(3).strip()
        return [
            f"What is the {trait} of {e1}?",
            f"What is the {trait} of {e2}?"
        ]
        
    # Pattern 2: "Did X and Y work in the same field?"
    pattern2 = r"^(?:Did|Were|Are)\s+([A-Za-z\s\.]+?)\s+and\s+([A-Za-z\s\.]+?)\s+both\s+([a-zA-Z\s]+)$"
    match2 = re.search(pattern2, clean_q, re.IGNORECASE)
    if match2:
        e1 = match2.group(1).strip()
        e2 = match2.group(2).strip()
        trait = match2.group(3).strip()
        return [
            f"Did {e1} {trait}?",
            f"Did {e2} {trait}?"
        ]

    return None


# --- 2. NODO REFINER ---
def refiner_node(state: GraphState, llm) -> dict:
    num_ref = state.get("num_refinement", 0)
    print(f"\n--- [REFINER] Inizio Ottimizzazione Query (Ciclo {num_ref + 1}) ---")
    current_query_raw = state.get("current_query", state.get("original_query", ""))
    
    # Suddivisione in sub-query esistenti per consentire micro-raffinamento modulare
    queries = [q.strip() for q in current_query_raw.split("\n---\n") if q.strip()]
    if not queries:
        queries = [current_query_raw]
        
    tools = [decompose_query, expand_query, rewrite_query, keep_query_unchanged]
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = """You are the Autonomous Query Refinement Agent implementing SOTA query reformulation techniques:
- Least-to-Most Prompting (Zhou et al., 2022) via 'decompose_query'
- Rewrite-Retrieve-Read (Ma et al., 2023) via 'rewrite_query'
- Query2doc (Wang et al., 2023) via 'expand_query'

CORE OPERATIONAL RULES:
1. PRESERVE ENTITIES: Never delete or generalize named entities from the original query (e.g. keep explicit names like "Marie Curie", never replace with generic terms).
2. DECOMPOSITION: If the question asks to compare two entities, check a shared property, or involves multiple independent lookups, use 'decompose_query' to create clean sub-questions.
3. ATOMIC QUESTIONS: If a query is already simple and focused on a single entity, call 'keep_query_unchanged'.
4. NOISE REMOVAL: If conversational fluff is present, call 'rewrite_query'.
5. REFLEXION: Use the 'reflection' parameter to critique and address the Validator's feedback.

FEW-SHOT EXAMPLES:
Example 1 (Comparative query -> decompose_query):
Query: "Were Marie Curie and Pierre Curie born in the same country?"
Output:
```json
{
  "name": "decompose_query",
  "arguments": {
    "reflection": "The question asks to compare two distinct people; looking up both simultaneously causes mixed search results.",
    "reasoning": "Decomposing into two parallel atomic sub-queries for each person's birth country.",
    "sub_queries": [
      "What was the birth country of Marie Curie?",
      "What was the birth country of Pierre Curie?"
    ]
  }
}
```

Example 2 (Atomic factual query -> keep_query_unchanged):
Query: "When was Albert Einstein born?"
Output:
```json
{
  "name": "keep_query_unchanged",
  "arguments": {
    "reflection": "The query is already a single atomic factual question ready for retrieval.",
    "reasoning": "No decomposition or rewrite needed."
  }
}
```

CRITICAL INSTRUCTION: Output EXACTLY ONE RAW JSON tool call. Do not add conversational text."""

    feedback_history = state.get("feedback_history", [])
    feedback_text = ""
    if feedback_history:
        feedback_text = "\n\nFEEDBACK FROM VALIDATOR (You must explicitly address these critiques):\n- " + "\n- ".join(feedback_history)

    final_processed_queries: List[str] = []
    tools_used_global: Set[str] = set()

    for q_idx, q in enumerate(queries):
        print(f"[REFINER] Analisi sotto-query {q_idx+1}/{len(queries)}: '{q}'")
        user_prompt = f"""Global Original Query: '{state.get('original_query', '')}'
Current Sub-Query to refine: '{q}'{feedback_text}

Analyze the query and feedback, then issue the appropriate JSON tool call."""

        content_resp = ""
        try:
            response_msg = llm_with_tools.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            content_resp = getattr(response_msg, "content", str(response_msg))
            tool_calls_list = extract_refiner_tool_calls(response_msg, content_resp)
        except Exception as e:
            print(f"[REFINER ERROR] Errore inferenza per la sub-query: {e}")
            tool_calls_list = []

        # Heuristic Least-to-Most Fallback se nessun tool è stato estratto
        if not tool_calls_list:
            heuristic_subs = heuristic_least_to_most(q)
            if heuristic_subs:
                print(f"   [REFINER] [Least-to-Most Fallback] Scomposizione euristica applicata per query comparativa.")
                tool_calls_list = [{
                    "name": "decompose_query",
                    "args": {
                        "reflection": "Heuristic Least-to-Most decomposition of composite comparative query.",
                        "reasoning": "Splitting comparative query into independent sub-queries.",
                        "sub_queries": heuristic_subs
                    }
                }]

        if tool_calls_list:
            # Considera la prima chiamata tool valida emessa per la sotto-query
            first_tool = tool_calls_list[0]
            t_name = first_tool.get("name")
            if t_name not in ["decompose_query", "rewrite_query", "expand_query", "keep_query_unchanged"]:
                t_name = "keep_query_unchanged"
            tools_used_global.add(t_name)
            args = first_tool.get("args", {})
            print(f"   [REFINER REASONING]: {args.get('reasoning', 'No reasoning provided')}")
            print(f"   [REFINER TOOL CALL]: '{t_name}'")

            decomposed_queries = None
            rewritten_query = None
            expansion_text = None
            keep_unchanged = False

            is_comp = any(w in q.lower() for w in ["same", "both", "compare", "differ"]) or (" and " in q.lower())

            if t_name == "decompose_query":
                # La scomposizione è valida se la query corrente è composita o se è l'unica query globale
                if is_comp or len(queries) == 1:
                    sub_keys = ["sub_queries", "queries", "subqueries", "sub-queries", "questions", "sub_questions", "subquestions"]
                    raw_subs = None
                    for k in sub_keys:
                        if k in args:
                            raw_subs = args[k]
                            break
                    if isinstance(raw_subs, str):
                        try:
                            raw_subs = json.loads(raw_subs)
                        except Exception:
                            try:
                                raw_subs = ast.literal_eval(raw_subs)
                            except Exception:
                                raw_subs = [raw_subs]
                    if isinstance(raw_subs, list):
                        decomposed_queries = [str(s).strip() for s in raw_subs if str(s).strip()]
                        # Verifica che le sotto-query siano realmente atomiche e separate
                        if is_comp and decomposed_queries:
                            is_still_composite = any(
                                any(w in sq.lower() for w in ["same", "both", "what about", "as well as", "either", "neither", "similar", "differ"])
                                for sq in decomposed_queries
                            ) or any(" and " in sq.lower() for sq in decomposed_queries)
                            if is_still_composite:
                                heuristic_subs = heuristic_least_to_most(q)
                                if heuristic_subs:
                                    print("   [REFINER] Rilevata scomposizione non atomica. Sostituzione con Least-to-Most rigorosa.")
                                    decomposed_queries = heuristic_subs
                else:
                    # Se la sotto-query è già atomica, non riscomporla
                    keep_unchanged = True
            elif t_name == "rewrite_query":
                rewritten_query = args.get("rewritten_query", args.get("query"))
            elif t_name == "expand_query":
                expansion_text = args.get("pseudo_document", args.get("expansion", args.get("context")))
            elif t_name == "keep_query_unchanged":
                keep_unchanged = True

            # Controllo rigoroso Least-to-Most (Zhou et al., 2022):
            # Se la query è composita e il tool o il modello non hanno prodotto scomposizione valida,
            # applica deterministicamente la scomposizione Least-to-Most
            if is_comp and (keep_unchanged or not decomposed_queries):
                heuristic_subs = heuristic_least_to_most(q)
                if heuristic_subs:
                    print("   [REFINER] Applicazione garantita di Least-to-Most per query comparativa.")
                    decomposed_queries = heuristic_subs
                    keep_unchanged = False

            # Risoluzione delle modifiche
            if decomposed_queries and isinstance(decomposed_queries, list):
                final_processed_queries.extend(decomposed_queries)
            elif rewritten_query and isinstance(rewritten_query, str):
                final_processed_queries.append(rewritten_query.strip())
            elif expansion_text and isinstance(expansion_text, str):
                final_processed_queries.append(f"{q}\n[Expanded Context]: {expansion_text.strip()}")
            else:
                final_processed_queries.append(q)
        else:
            final_processed_queries.append(q)

    # Deduplicazione ordinata delle sotto-query per prevenire query ridondanti o duplicate
    seen = set()
    deduped_queries = []
    for sq in final_processed_queries:
        sq_clean = sq.strip()
        if sq_clean and sq_clean.lower() not in seen:
            seen.add(sq_clean.lower())
            deduped_queries.append(sq_clean)
    final_processed_queries = deduped_queries if deduped_queries else [current_query_raw]

    final_query_str = "\n---\n".join(final_processed_queries)
    if tools_used_global:
        print(f"[REFINER] Modifiche applicate con tool: {', '.join(tools_used_global).upper()}")
    else:
        print("[REFINER] Nessun tool chiamato, query mantenuta inalterata.")
    print(f"[REFINER] Output finale:\n{final_query_str}")

    return {
        "current_query": final_query_str,
        "num_refinement": num_ref + 1
    }
