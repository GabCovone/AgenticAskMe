import json
import os
import sys
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from graph import GraphState

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --- COSTANTI DI CONTROLLO DEL CICLO AGENTICO ---
MAX_REFINEMENT = 4
MAX_PLANNING = 4

# --- SCHEMA DI DECISIONE SEMANTICA (SELF-RAG + ADAPTIVE-RAG + REFLEXION) ---
class ValidatorDecision(BaseModel):
    """USE THIS TOOL to issue the semantic validation and routing decision."""
    reflection: str = Field(
        description="REFLEXION: Verbal critique of past iterations. Why did previous retrievals fail? Is there hallucination or missing context? How should the next step recover?"
    )
    query_complexity: Literal["simple", "complex", "already_decomposed"] = Field(
        description="ADAPTIVE-RAG: Assess if the query is a simple factoid, a complex multi-hop question, or already decomposed into atomic sub-queries."
    )
    is_context_relevant: bool = Field(
        description="SELF-RAG (IsRel): Determine if the retrieved context is semantically relevant and provides useful facts to answer the question."
    )
    is_query_answered: bool = Field(
        description="SELF-RAG (IsSup): Determine if the proposed answer is strictly supported by the retrieved context without hallucinations."
    )
    reasoning: str = Field(
        description="CHAIN OF THOUGHT: Step-by-step logical justification for the routing decision."
    )
    feedback: str = Field(
        description="Specific verbal instructions for the next module. For Refiner: suggest decomposition or entity correction. For Planner: suggest explicit keywords or source (local_kb vs web)."
    )
    next_action: Literal["route_to_refinement", "route_to_planning", "finish"] = Field(
        description="Routing destination: 'route_to_refinement', 'route_to_planning', or 'finish'."
    )


# --- NODO VALIDATORE ---
def validator_node(state: GraphState, llm) -> dict:
    num_ref = state.get("num_refinement", 0)
    num_plan = state.get("num_planning", 0)
    print(f"\n[VALIDATOR] Avvio ciclo di valutazione e routing (Ref={num_ref}/{MAX_REFINEMENT}, Plan={num_plan}/{MAX_PLANNING})...")
    
    llm_with_tools = llm.bind_tools([ValidatorDecision])
    
    system_prompt = """You are the Semantic Routing Validator and Quality Judge for an Agentic IR system.
You enforce the Self-RAG, Adaptive-RAG, and Reflexion theoretical frameworks.

EVALUATION FRAMEWORK:
1. ADAPTIVE-RAG (Query Complexity):
   - 'simple': Single atomic question (e.g. "When was X born?"). Route directly to 'route_to_planning'.
   - 'complex': Ambiguous query, or questions with multiple independent entities being compared (e.g. "Were X and Y born in the same country?"). Route to 'route_to_refinement'.
   - 'already_decomposed': Multiple sub-queries already split by '---'. Route to 'route_to_planning'.

2. SELF-RAG CRITIQUE TOKENS:
   - 'is_context_relevant' (IsRel): True if the retrieved documents contain facts related to the query; False if unrelated.
   - 'is_query_answered' (IsSup): True ONLY if the retrieved documents fully support the answer. If the answer contains ungrounded guesses or claims not in the context, set False.

3. ROUTING RULES ('next_action'):
   - 'finish': Choose ONLY if 'is_query_answered' is True AND a valid grounded Final Answer exists. If no context has been retrieved yet, 'finish' is STRICTLY FORBIDDEN!
   - 'route_to_refinement': Choose if the query asks to compare two entities, check if they share a trait, contains 'and' / 'both' / 'same', or is conversational/ambiguous.
   - 'route_to_planning': Choose for direct factual queries ready for search execution.

CRITICAL INSTRUCTION: Output a RAW JSON tool call matching ValidatorDecision. Do not output conversational text outside the JSON."""

    query_raw = state.get("current_query", state.get("original_query", ""))
    queries = [q.strip() for q in query_raw.split("\n---\n") if q.strip()]
    if not queries:
        queries = [state.get("original_query", "")]
        
    context = state.get("retrieved_context", "")
    final_answer = state.get("final_answer", "")
    
    # Se abbiamo già una Final Answer proposta dal Planner, valutiamo la query originale nel suo insieme
    evaluating_final_synthesis = bool(final_answer and num_plan > 0)
    if evaluating_final_synthesis:
        print("[VALIDATOR] Valutazione della Proposta Final Answer contro il Contesto (Self-RAG Grounding)...")
        queries = [state.get("original_query", "")]
        eval_context = f"Retrieved Evidence:\n{context}\n\nProposed Final Answer:\n{final_answer}"
    else:
        eval_context = context if context else "No context retrieved yet."

    feedbacks = []
    actions = []
    
    for q_idx, q in enumerate(queries):
        if not evaluating_final_synthesis:
            print(f"[VALIDATOR] Analisi sotto-query {q_idx+1}/{len(queries)}: '{q}'")
            
        user_prompt = f"""Original User Request: '{state.get('original_query', '')}'
Sub-Query under evaluation: '{q}'
Current Retrieved Context / Answer:
{eval_context}

Metrics: Refinement Count={num_ref}/{MAX_REFINEMENT}, Planning Count={num_plan}/{MAX_PLANNING}

Evaluate the state and issue your ValidatorDecision JSON."""

        decision_obj = None
        try:
            response_msg = llm_with_tools.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            content_resp = getattr(response_msg, "content", str(response_msg))
        except Exception as e:
            print(f"[VALIDATOR ERROR] Errore inferenza: {e}")
            actions.append("route_to_planning")
            continue
            
        # 1. Parsing da tool_calls nativo
        if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
            t_call = response_msg.tool_calls[0]
            args = t_call.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if isinstance(args, dict) and args:
                try:
                    safe_args = {
                        "reflection": args.get("reflection", "Iterative evaluation"),
                        "query_complexity": args.get("query_complexity", "complex"),
                        "is_context_relevant": args.get("is_context_relevant", False) in [True, "true", "True", 1],
                        "is_query_answered": args.get("is_query_answered", False) in [True, "true", "True", 1],
                        "reasoning": args.get("reasoning", "Standard reasoning"),
                        "feedback": args.get("feedback", ""),
                        "next_action": args.get("next_action", "route_to_planning")
                    }
                    decision_obj = ValidatorDecision(**safe_args)
                except Exception as e:
                    print(f"[VALIDATOR DEBUG] Pydantic validation error: {e}")

        # 2. Parsing con sliding-window JSON parser
        if not decision_obj:
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
                                json_str = json_str.replace(": False", ": false").replace(": True", ": true")
                                parsed = json.loads(json_str)
                                if isinstance(parsed, dict):
                                    args = parsed.get("arguments", parsed.get("args", parsed.get("parameters", parsed))) if "name" in parsed else parsed
                                    if isinstance(args, str):
                                        try:
                                            args = json.loads(args)
                                        except Exception:
                                            args = {}
                                    if isinstance(args, dict):
                                        next_act = str(args.get("next_action", "route_to_planning")).lower()
                                        if next_act not in ["route_to_refinement", "route_to_planning", "finish"]:
                                            next_act = "route_to_planning"
                                            
                                        safe_args = {
                                            "reflection": args.get("reflection", "Sliding window extraction"),
                                            "query_complexity": args.get("query_complexity", "complex"),
                                            "is_context_relevant": args.get("is_context_relevant", False) in [True, "true", "True", 1],
                                            "is_query_answered": args.get("is_query_answered", False) in [True, "true", "True", 1],
                                            "reasoning": args.get("reasoning", "Extracted CoT"),
                                            "feedback": args.get("feedback", ""),
                                            "next_action": next_act
                                        }
                                        decision_obj = ValidatorDecision(**safe_args)
                                        break
                            except Exception:
                                pass
                            i = j
                            break
                        if char == '\\':
                            escape = not escape
                        else:
                            escape = False
                i += 1

        # Fallback di sicurezza se nessun JSON è stato prodotto
        if not decision_obj:
            print("[VALIDATOR] JSON non riconosciuto. Applico fallback euristico sicuro.")
            decision_obj = ValidatorDecision(
                reflection="Fallback: model produced non-JSON output.",
                query_complexity="complex",
                is_context_relevant=bool(context),
                is_query_answered=bool(final_answer),
                reasoning="Safe heuristic fallback.",
                feedback="Proceed to planning for factual retrieval.",
                next_action="route_to_planning" if num_plan < MAX_PLANNING else "finish"
            )

        print(f"      [REASONING]: {decision_obj.reasoning[:120]}...")
        print(f"      [SELF-RAG CRITIQUE]: IsRel={decision_obj.is_context_relevant} | IsSup={decision_obj.is_query_answered}")
        print(f"      [ACTION LLM]: {decision_obj.next_action.upper()}")

        # --- RIGOROSA APPLICAZIONE DEI FRAMEWORK ACCADEMICI SUL ROUTING ---
        act = decision_obj.next_action

        # 1. Regola Self-RAG (IsSup): Se la risposta non è supportata, non può terminare
        if act == "finish" and not decision_obj.is_query_answered and num_plan < MAX_PLANNING:
            print("      [SELF-RAG VETO] Risposta non supportata dal contesto (IsSup=False). Reindirizzo al Planner!")
            act = "route_to_planning"

        # 2. Se non è mai stato fatto planning e la risposta è assente o vuota, non si può terminare
        if act == "finish" and num_plan == 0:
            is_composite = any(w in q.lower() for w in [" and ", "both", "same", "compare", "differ"]) or decision_obj.query_complexity == "complex"
            if is_composite and num_ref == 0:
                print("      [VALIDATOR VETO] Query comparativa/complessa non ancora raffinata. Instradamento al Refiner.")
                act = "route_to_refinement"
            else:
                print("      [VALIDATOR VETO] Impossibile terminare senza retrieval. Instradamento al Planner.")
                act = "route_to_planning"

        # 3. Limiti massimi di sicurezza per prevenire cicli infiniti
        if act == "route_to_refinement" and num_ref >= MAX_REFINEMENT:
            print(f"      [LIMIT REACHED] Raggiunto limite refinement ({MAX_REFINEMENT}). Forzo passaggio a planning.")
            act = "route_to_planning"
        elif act == "route_to_planning" and num_plan >= MAX_PLANNING:
            print(f"      [LIMIT REACHED] Raggiunto limite planning ({MAX_PLANNING}). Forzo completamento del grafo.")
            act = "finish"

        actions.append(act)
        if decision_obj.feedback:
            feedbacks.append(f"Validator feedback for '{q}': {decision_obj.feedback}")
            if decision_obj.reflection:
                feedbacks.append(f"Reflection: {decision_obj.reflection}")

    # Aggregazione delle azioni per il grafo
    if "route_to_refinement" in actions:
        overall_action = "route_to_refinement"
    elif "route_to_planning" in actions:
        overall_action = "route_to_planning"
    else:
        overall_action = "finish"

    print(f"[VALIDATOR] Instradamento globale verso nodo: {overall_action.upper()}\n")

    # Mantenimento o reset dello stato
    state_update = {
        "next_node": overall_action,
        "feedback_history": feedbacks
    }

    # Se avevamo una final answer ma il Validator l'ha bocciata (Self-RAG IsSup=False o action != finish)
    if final_answer and overall_action != "finish":
        print("[VALIDATOR] Final Answer respinta dal controllo Self-RAG. Resetto e chiedo nuova ricerca.")
        state_update["final_answer"] = ""

    return state_update
