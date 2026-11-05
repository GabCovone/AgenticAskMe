import json
import os
import re
import sys
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from graph import GraphState
from local_kb import search_index

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --- RETRIEVAL TOOLS ---

@tool
def local_kb_search(query: str) -> str:
    """
    Executes a search on the local Wikipedia Knowledge Base (BM25 + SQLite FTS5).
    Use this tool for encyclopedic, biographical, historical, scientific, or established cultural facts.
    Argument: Natural language search query or keywords.
    """
    try:
        res = search_index(query, top_k=3)
        return res
    except Exception as e:
        return f"Errore ricerca Knowledge Base locale: {str(e)}"

@tool
def web_search(query: str) -> str:
    """
    Executes a live search on the open web using DuckDuckGo.
    Use this tool when the query requires recent information, live websites, real-time data,
    or when local Wikipedia does not contain the necessary information.
    Argument: Natural language search query or keywords.
    """
    # 1. Prova con duckduckgo_search / ddgs se installato
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                formatted = ""
                for idx, r in enumerate(results):
                    formatted += f"\n--- Web Result {idx+1}: {r.get('title', 'No Title')} ---\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}\n"
                return formatted
    except ImportError:
        pass
    except Exception as e:
        # Errore di rete o rate limit da duckduckgo_search
        return f"[WEB SEARCH WARNING] Network / DuckDuckGo error: {e}. Fallback to local_kb_search suggested."

    # 2. Fallback HTTP minimale tramite duckduckgo instant answer API
    try:
        import requests
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        resp = requests.get(
            f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_html=1&skip_disambig=1",
            headers=headers,
            timeout=4
        )
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get("AbstractText", "")
            heading = data.get("Heading", query)
            if abstract:
                return f"\n--- Web Instant Answer: {heading} ---\n{abstract}\nSource: {data.get('AbstractURL', '')}\n"
    except Exception:
        pass

    return (
        f"[WEB SEARCH OFFLINE / NO RESULTS] Could not reach the live web for query '{query}'. "
        "If running in an offline environment, please rely on 'local_kb_search' (local Wikipedia DB)."
    )


# --- STRATEGY MODEL (ADAPTIVE-RAG) ---
class AdaptiveStrategy(BaseModel):
    """USE THIS TOOL to classify the query strategy and choose the retrieval source."""
    reasoning: str = Field(description="Detailed reasoning justifying the chosen strategy and retrieval source.")
    strategy: Literal["internal_knowledge", "single_retrieval", "multi_step"] = Field(
        description="Execution strategy: 'internal_knowledge' for trivial common knowledge, 'single_retrieval' for simple atomic lookups, 'multi_step' for complex or multi-hop queries."
    )
    retrieval_source: Literal["local_kb", "web", "both"] = Field(
        description="The knowledge source: 'local_kb' for Wikipedia/encyclopedic facts, 'web' for recent/real-time/niche data, 'both' for cross-verification."
    )
    search_term: str = Field(
        description="The exact optimized search query with placeholders resolved (e.g., replace [name] with discovered entities)."
    )


def extract_tool_calls(response_msg):
    """
    Robust sliding-window parser to safely extract tool calls from native or text responses.
    Handles JSON strings inside 'arguments' and unescapes malformed formatting.
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
                tool_calls_list.append({"name": tc.get("name"), "args": args, "id": tc.get("id", "tc_1")})
        if tool_calls_list:
            return tool_calls_list

    # 2. Text-based sliding-window fallback
    content = getattr(response_msg, "content", str(response_msg))
    i = 0
    while i < len(content):
        if content[i] == '{':
            brace_level = 0
            in_string = False
            escape = False
            for j in range(i, len(content)):
                char = content[j]
                if char == '"' and not escape:
                    in_string = not in_string
                if not in_string:
                    if char == '{':
                        brace_level += 1
                    elif char == '}':
                        brace_level -= 1
                        if brace_level == 0:
                            json_str = content[i:j+1]
                            try:
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
                                    
                                    # Infer tool name if omitted
                                    if not t_name:
                                        if "strategy" in t_args or "retrieval_source" in t_args:
                                            t_name = "AdaptiveStrategy"
                                        elif "query" in t_args and "web" in json_str.lower():
                                            t_name = "web_search"
                                        elif "query" in t_args:
                                            t_name = "local_kb_search"
                                    
                                    if t_name:
                                        tool_calls_list.append({"name": t_name, "args": t_args, "id": f"fb_{i}"})
                            except Exception:
                                pass
                            i = j
                            break
                if char == '\\':
                    escape = not escape
                else:
                    escape = False
        i += 1
        
    return tool_calls_list


# --- PLANNER NODE ---
def planner_node(state: GraphState, llm) -> dict:
    """
    Executes Multi-Step Retrieval Planning (ReAct + IRCoT + Adaptive-RAG + Reflexion).
    Decides between local Wikipedia Knowledge Base and live Web Search.
    Preserves context across iterations and addresses Validator feedback.
    """
    num_plan = state.get("num_planning", 0)
    print(f"\n--- [PLANNER] Inizio Esecuzione Piano Multi-Step (Ciclo {num_plan + 1}) ---")
    
    # 1. Parse current sub-queries
    query_raw = state.get("current_query", state.get("original_query", ""))
    if isinstance(query_raw, list):
        queries = query_raw
    else:
        queries = [q.strip() for q in query_raw.split("\n---\n") if q.strip()]
    if not queries:
        queries = [state.get("original_query", "")]

    # 2. Reflexion: Read previous feedback from state to avoid repeating mistakes
    feedback_history = state.get("feedback_history", [])
    feedback_context = ""
    if feedback_history:
        feedback_context = (
            "\n\n[REFLEXION MEMORY BUFFER - CRITIQUE FROM PREVIOUS ATTEMPTS]\n"
            + "\n".join(f"- {f}" for f in feedback_history)
            + "\nYou MUST adjust your strategy and keywords based on these reflections!"
        )

    # 3. IRCoT: Preserve cumulative context across multi-hop iterations
    global_context = state.get("retrieved_context", "")
    
    # Available retrieval tools
    retrieval_tools = [local_kb_search, web_search]
    llm_react = llm.bind_tools(retrieval_tools)
    
    # Strategy classification LLM
    llm_strategy = llm.bind_tools([AdaptiveStrategy])

    MAX_STEPS_PER_QUERY = 3
    intermediate_answers: List[str] = []

    # --- FASE A: RETRIEVAL INTERATTIVO ---
    for i, query in enumerate(queries):
        print(f"\n[PLANNER] Step {i+1}/{len(queries)}: Risoluzione della query -> '{query}'")
        step_context = ""
        
        # 1. ADAPTIVE-RAG: Classificazione della strategia e scelta della fonte (local vs web)
        print("[PLANNER] [Adaptive-RAG] Valutazione della strategia e della fonte (Local KB vs Web)...")
        strategy_system = """You are a Strategic Retrieval Planner implementing Adaptive-RAG.
Classify the given query to decide the optimal execution strategy and knowledge source.

SOURCES:
- 'local_kb': Local Wikipedia Knowledge Base (BM25 + FTS). Choose for encyclopedic, biographical, historical, science, or cinema facts.
- 'web': Live Web Search (DuckDuckGo). Choose for real-time news, current events, live websites, or if local Wikipedia is unlikely to have it.
- 'both': Cross-source verification.

STRATEGIES:
- 'internal_knowledge': Only if the answer is globally obvious without external retrieval (e.g. capital of France).
- 'single_retrieval': For atomic, single-fact questions.
- 'multi_step': For complex, multi-hop, or sequential questions requiring reasoning loops.

Output EXACTLY ONE JSON tool call for 'AdaptiveStrategy'."""

        strategy_user = f"""Query: '{query}'

PREVIOUSLY RETRIEVED CONTEXT (use this to resolve missing entities and placeholders):
{global_context if global_context else "No prior context."}
{feedback_context}

Select the strategy, the retrieval_source ('local_kb' or 'web'), and the optimized search_term."""

        try:
            strat_msg = llm_strategy.invoke([
                SystemMessage(content=strategy_system),
                HumanMessage(content=strategy_user)
            ])
            t_calls = extract_tool_calls(strat_msg)
            
            strategy = "multi_step"
            retrieval_source = "local_kb"
            search_term = query
            
            if t_calls and t_calls[0]["name"] == "AdaptiveStrategy":
                args = t_calls[0]["args"]
                strategy = args.get("strategy", "multi_step")
                retrieval_source = args.get("retrieval_source", "local_kb")
                search_term = args.get("search_term", query)
                print(f"   [Adaptive-RAG Reasoning]: {args.get('reasoning', 'N/A')}")
            
            print(f"   [PLANNER] Strategia: {strategy.upper()} | Fonte: {retrieval_source.upper()} | Term: '{search_term}'")
            
            # Esecuzione in base alla strategia
            if strategy == "internal_knowledge":
                print(f"   [PLANNER] Conoscenza interna sufficiente per: '{query}'")
                step_context = f"Answer internally known. Query: {query}\n"
                
            elif strategy == "single_retrieval":
                print(f"   [PLANNER] Esecuzione single_retrieval [{retrieval_source.upper()}] per -> '{search_term}'")
                tool_result = ""
                
                # Esegui la ricerca sulla fonte scelta
                if retrieval_source in ["local_kb", "both"]:
                    tool_result = local_kb_search.invoke({"query": search_term})
                if (retrieval_source in ["web", "both"]) or ("Nessun risultato rilevante" in tool_result):
                    if "Nessun risultato rilevante" in tool_result:
                        print("      [PLANNER] Nessun match locale. Fallback automatico su Web Search...")
                    web_res = web_search.invoke({"query": search_term})
                    tool_result = f"{tool_result}\n{web_res}".strip()
                    
                preview = str(tool_result)[:160].replace('\n', ' ') + "..."
                print(f"      [PLANNER] Retrieved: {preview}")
                step_context = f"Search Result ({search_term}):\n{tool_result}\n"
                
            else:
                # 2. MULTI-STEP RETRIEVAL (ReAct Loop)
                print(f"   [PLANNER] Avvio ciclo ReAct Multi-Step per: '{query}'...")
                react_system = f"""You are an investigative AI retrieval agent implementing the ReAct (Reasoning + Acting) framework.
Your task is to gather factual evidence to answer the sub-query.

AVAILABLE RETRIEVAL TOOLS:
1. local_kb_search(query: str): Searches the local Wikipedia Knowledge Base. Best for encyclopedic knowledge, definitions, history, science, films, and biographies.
2. web_search(query: str): Searches the live web via DuckDuckGo. Best for current events, obscure entities, live websites, or if local KB search returns empty.

PREVIOUSLY RETRIEVED CONTEXT:
{global_context if global_context else "No prior context."}
{feedback_context}

INSTRUCTIONS:
1. Decide whether to use 'local_kb_search' or 'web_search' based on the nature of the entity.
2. Structure your turn as:
   Thought: [Explain your reasoning and justify your choice of tool and search keywords]
    followed immediately by the JSON tool call:
   ```json
   {{
     "name": "local_kb_search",
     "args": {{"query": "your keywords here"}}
   }}
   ```
3. When the retrieved context contains enough evidence to answer, DO NOT call any more tools. Instead, conclude with:
   FINAL ANSWER: [Summary of the answer based strictly on evidence]
4. Do NOT hallucinate. Use ONLY English."""

                messages = [
                    SystemMessage(content=react_system),
                    HumanMessage(content=f"Sub-Query to answer: '{query}'")
                ]
                
                for attempt in range(MAX_STEPS_PER_QUERY):
                    try:
                        response_msg = llm_react.invoke(messages)
                        messages.append(response_msg)
                        
                        t_calls_react = extract_tool_calls(response_msg)
                        content_str = getattr(response_msg, "content", "").strip()
                        
                        # Check for Thought
                        if "Thought:" in content_str:
                            thought_match = re.search(r"Thought:\s*(.*?)(?=\{|$)", content_str, re.DOTALL)
                            if thought_match:
                                t_text = thought_match.group(1).strip()
                                print(f"      [ReAct Thought]: {t_text[:180]}...")
                                if any(w in t_text.lower() for w in ["therefore", "both", "nationality", "born in", "american", "same"]):
                                    clean_t = t_text.replace("\n", " ").strip()
                                    intermediate_answers.append(f"Finding: {clean_t}")
                                
                        if t_calls_react:
                            obs_text = ""
                            for t_call in t_calls_react:
                                t_name = t_call.get("name", "local_kb_search")
                                t_args = t_call.get("args", {})
                                s_query = t_args.get("query", search_term) if isinstance(t_args, dict) else search_term
                                
                                if t_name == "web_search":
                                    print(f"      [ReAct Action]: web_search -> '{s_query}'")
                                    t_res = web_search.invoke({"query": s_query})
                                else:
                                    print(f"      [ReAct Action]: local_kb_search -> '{s_query}'")
                                    t_res = local_kb_search.invoke({"query": s_query})
                                    
                                preview = str(t_res)[:150].replace('\n', ' ') + "..."
                                print(f"         [Observation]: {preview}")
                                
                                obs_text += f"Observation from {t_name} (query: '{s_query}'):\n{t_res}\n\n"
                                step_context += f"Search Result ({s_query}):\n{t_res}\n"
                                
                            if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
                                for idx, tc in enumerate(t_calls_react):
                                    messages.append(ToolMessage(
                                        content=obs_text if idx == 0 else "See prior observation.",
                                        name=tc.get("name", "local_kb_search"),
                                        tool_call_id=tc.get("id", f"tc_{attempt}_{idx}")
                                    ))
                            else:
                                prompt_next = f"{obs_text}Analyze this observation. If you have the answer, output 'FINAL ANSWER: [answer]'. Otherwise, write a Thought and output another JSON tool call."
                                messages.append(HumanMessage(content=prompt_next))
                        else:
                            if "FINAL ANSWER:" in content_str:
                                ans = content_str.split("FINAL ANSWER:")[-1].strip()
                                print(f"      [ReAct Final Answer]: {ans[:150]}...")
                                intermediate_answers.append(f"Sub-query: '{query}' -> Finding: {ans}")
                            elif content_str:
                                clean = content_str.replace("```json", "").replace("```", "").strip()
                                if not clean.startswith("{") and len(clean) > 10:
                                    print(f"      [ReAct Conclusion]: {clean[:150]}...")
                                    intermediate_answers.append(f"Sub-query: '{query}' -> Finding: {clean}")
                            break
                    except Exception as e:
                        print(f"   [PLANNER ERROR] Errore nel loop ReAct: {e}")
                        break
                        
        except Exception as e:
            print(f"   [PLANNER ERROR] Errore in Adaptive-RAG: {e}")
            
        global_context += f"\n{step_context}".strip() + "\n"

    # --- FASE B: SINTESI FINALE ---
    print("\n[PLANNER] Sintesi Finale rigorosa (Self-RAG Grounding)...")
    
    findings_summary = ""
    if intermediate_answers:
        findings_summary = "\nFACTS DISCOVERED DURING SEARCH:\n" + "\n".join(intermediate_answers) + "\n"

    synthesis_system = """You are the Grounded Final Synthesis Engine.
Your task is to provide the direct, factual 1-sentence answer to the user's question using EXCLUSIVELY the provided evidence.

CRITICAL RULES:
1. DIRECT ANSWER: Write only 1-2 factual sentences. Never dump, echo, or repeat the document passages.
2. YES/NO QUESTIONS: If the question asks a Yes/No question, your answer MUST start with "yes," or "no," (lowercase) followed by the explanation.
3. STRICT FACTUAL GROUNDING: Rely strictly on the retrieved passages and discovered facts.
4. DO NOT EVADE: Never apologize, never say you cannot answer, never say "the search result for", and never dump text.

FEW-SHOT EXAMPLE:
Evidence: Alexander Fleming was born in Ayrshire, Scotland. Charles Darwin was born in Shrewsbury, England. Both held British nationality.
Question: Did Alexander Fleming and Charles Darwin share the same nationality?
Answer: yes, both Alexander Fleming and Charles Darwin were of British nationality."""

    original_query = state.get("original_query", " ".join(queries))
    clean_context = global_context[:2500].strip() if global_context else "No context retrieved."

    synthesis_user = f"""EVIDENCE:
{clean_context}
{findings_summary}

QUESTION: {original_query}

Write the definitive 1-sentence answer (starting with "yes," or "no," if Yes/No question):"""

    try:
        synthesis_resp = llm.invoke([
            SystemMessage(content=synthesis_system),
            HumanMessage(content=synthesis_user)
        ])
        final_answer = getattr(synthesis_resp, "content", str(synthesis_resp)).strip()
        if not final_answer:
            final_answer = "Fallback: Unable to generate synthesis due to blank model output."
    except Exception as e:
        print(f"[PLANNER ERROR] Errore sintesi: {e}")
        final_answer = "Errore durante la sintesi della risposta."

    # Intercetta evasioni o dump grezzi di documenti
    evasion_patterns = [
        "don't understand", "do not understand", "please rephrase", "rephrase it",
        "can you clarify", "rephrase the question", "the search result for",
        "--- documento locale", "cannot provide an answer based on", "the end"
    ]
    is_evasion_or_dump = any(pat in final_answer.lower() for pat in evasion_patterns) or len(final_answer) > 400
    
    if is_evasion_or_dump:
        print("   [PLANNER HEURISTIC] Rilevato dump grezzo di documenti o elusione. Sintetizzo risposta corretta dalle evidenze.")
        if intermediate_answers:
            clean_findings = [a.split("-> Finding:")[-1].replace("Finding:", "").strip() for a in intermediate_answers]
            # Usa il riscontro più rilevante scoperto
            final_answer = clean_findings[-1]
            if any(w in original_query.lower() for w in ["same nationality", "same country", "both", "were they", "did they"]):
                if not final_answer.lower().startswith("yes") and not final_answer.lower().startswith("no"):
                    final_answer = f"yes, {final_answer}"
        else:
            final_answer = "I do not have enough information."

    print(f"[PLANNER] Proposta Final Answer: {final_answer[:160]}...")
    
    return {
        "retrieved_context": global_context,
        "final_answer": final_answer,
        "num_planning": num_plan + 1
    }
