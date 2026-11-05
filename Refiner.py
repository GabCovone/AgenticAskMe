import json
import ast
import sys
from typing import List, Set
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from graph import GraphState

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

        try:
            response_msg = llm_with_tools.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
        except Exception as e:
            print(f"[REFINER ERROR] Errore inferenza per la sub-query: {e}")
            final_processed_queries.append(q)
            continue

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

        # 2. Sliding window JSON parser fallback
        if not tool_calls_list:
            content_resp = getattr(response_msg, "content", str(response_msg))
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
                                parsed = json.loads(content_resp[i:j+1])
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
                                        if "sub_queries" in t_args or "queries" in t_args:
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
                            i = j
                            break
                        if char == '\\':
                            escape = not escape
                        else:
                            escape = False
                i += 1

        if tool_calls_list:
            decomposed_queries = None
            rewritten_query = None
            expansion_text = None
            keep_unchanged = False
            
            for tool_call in tool_calls_list:
                t_name = tool_call.get("name")
                tools_used_global.add(t_name)
                args = tool_call.get("args", {})
                print(f"   [REFINER REASONING]: {args.get('reasoning', 'No reasoning provided')}")
                print(f"   [REFINER TOOL CALL]: '{t_name}'")

                if t_name == "decompose_query":
                    decomposed_queries = args.get("sub_queries", args.get("queries"))
                    if isinstance(decomposed_queries, str):
                        try:
                            decomposed_queries = json.loads(decomposed_queries)
                        except Exception:
                            try:
                                decomposed_queries = ast.literal_eval(decomposed_queries)
                            except Exception:
                                decomposed_queries = [decomposed_queries]
                elif t_name == "rewrite_query":
                    rewritten_query = args.get("rewritten_query", args.get("query"))
                elif t_name == "expand_query":
                    expansion_text = args.get("pseudo_document", args.get("expansion", args.get("context")))
                elif t_name == "keep_query_unchanged":
                    keep_unchanged = True

            # Risoluzione gerarchica delle modifiche
            if decomposed_queries and isinstance(decomposed_queries, list):
                final_processed_queries.extend([str(sq).strip() for sq in decomposed_queries if str(sq).strip()])
            elif rewritten_query and isinstance(rewritten_query, str):
                final_processed_queries.append(rewritten_query.strip())
            elif expansion_text and isinstance(expansion_text, str):
                final_processed_queries.append(f"{q}\n[Expanded Context]: {expansion_text.strip()}")
            elif keep_unchanged:
                final_processed_queries.append(q)
            else:
                final_processed_queries.append(q)
        else:
            final_processed_queries.append(q)

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
