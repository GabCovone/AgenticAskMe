import json
import os
import re
import sys
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from state import GraphState

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


def extract_validator_decision(response_msg, content_resp: str) -> Optional[ValidatorDecision]:
    """
    Tolerant multi-stage parser for ValidatorDecision:
    1. LangChain native tool_calls
    2. Markdown ```json blocks
    3. Sliding-window JSON parser
    4. Text-based semantic classification
    """
    # 1. Native tool calls
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
                next_act = str(args.get("next_action", "route_to_planning")).lower().strip()
                if next_act not in ["route_to_refinement", "route_to_planning", "finish"]:
                    next_act = "route_to_planning"
                return ValidatorDecision(
                    reflection=str(args.get("reflection", "Native tool call critique")),
                    query_complexity=args.get("query_complexity", "complex"),
                    is_context_relevant=args.get("is_context_relevant", False) in [True, "true", "True", 1],
                    is_query_answered=args.get("is_query_answered", False) in [True, "true", "True", 1],
                    reasoning=str(args.get("reasoning", "Standard reasoning")),
                    feedback=str(args.get("feedback", "")),
                    next_action=next_act
                )
            except Exception:
                pass

    # 2. Markdown fenced JSON
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_resp, re.DOTALL)
    if md_match:
        try:
            raw_str = md_match.group(1).replace(": False", ": false").replace(": True", ": true")
            parsed = json.loads(raw_str)
            if isinstance(parsed, dict):
                args = parsed.get("arguments", parsed.get("args", parsed.get("parameters", parsed))) if "name" in parsed else parsed
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if isinstance(args, dict):
                    next_act = str(args.get("next_action", "route_to_planning")).lower().strip()
                    if next_act not in ["route_to_refinement", "route_to_planning", "finish"]:
                        next_act = "route_to_planning"
                    return ValidatorDecision(
                        reflection=str(args.get("reflection", "Fenced JSON critique")),
                        query_complexity=args.get("query_complexity", "complex"),
                        is_context_relevant=args.get("is_context_relevant", False) in [True, "true", "True", 1],
                        is_query_answered=args.get("is_query_answered", False) in [True, "true", "True", 1],
                        reasoning=str(args.get("reasoning", "Fenced JSON reasoning")),
                        feedback=str(args.get("feedback", "")),
                        next_action=next_act
                    )
        except Exception:
            pass

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
                        json_str = json_str.replace(": False", ": false").replace(": True", ": true")
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            args = parsed.get("arguments", parsed.get("args", parsed.get("parameters", parsed))) if "name" in parsed else parsed
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    args = {}
                            if isinstance(args, dict) and ("next_action" in args or "is_query_answered" in args or "reasoning" in args):
                                next_act = str(args.get("next_action", "route_to_planning")).lower().strip()
                                if next_act not in ["route_to_refinement", "route_to_planning", "finish"]:
                                    next_act = "route_to_planning"
                                return ValidatorDecision(
                                    reflection=str(args.get("reflection", "Sliding window extraction")),
                                    query_complexity=args.get("query_complexity", "complex"),
                                    is_context_relevant=args.get("is_context_relevant", False) in [True, "true", "True", 1],
                                    is_query_answered=args.get("is_query_answered", False) in [True, "true", "True", 1],
                                    reasoning=str(args.get("reasoning", "Extracted CoT")),
                                    feedback=str(args.get("feedback", "")),
                                    next_action=next_act
                                )
                    except Exception:
                        pass
                    break
                if char == '\\':
                    escape = not escape
                else:
                    escape = False
        i += 1

    # 4. Text-based semantic classification fallback
    lower_resp = content_resp.lower()
    if "route_to_planning" in lower_resp or "to planning" in lower_resp:
        return ValidatorDecision(
            reflection="Text fallback indicates routing to retrieval planning.",
            query_complexity="simple",
            is_context_relevant=False,
            is_query_answered=False,
            reasoning="Extracted planning intent from text.",
            feedback="Search for evidence.",
            next_action="route_to_planning"
        )
    elif "route_to_refinement" in lower_resp or "to refinement" in lower_resp:
        return ValidatorDecision(
            reflection="Text fallback suggests query refinement or decomposition.",
            query_complexity="complex",
            is_context_relevant=False,
            is_query_answered=False,
            reasoning="Extracted refinement intent from text.",
            feedback="Decompose or clarify query.",
            next_action="route_to_refinement"
        )
    elif "finish" in lower_resp or "answered" in lower_resp:
        return ValidatorDecision(
            reflection="Text fallback indicates answer is completed and verified.",
            query_complexity="simple",
            is_context_relevant=True,
            is_query_answered=True,
            reasoning="Extracted finish intent from text.",
            feedback="None.",
            next_action="finish"
        )

    return None


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
   - 'route_to_planning': Choose for direct factual queries ready for search execution, OR when queries are already decomposed into sub-queries. IF NO DOCUMENTS HAVE BEEN RETRIEVED YET AND THE QUERY IS READY, YOU MUST ROUTE TO 'route_to_planning' TO RETRIEVE FACTS! Never route to refinement simply because context is missing!
   - 'route_to_refinement': Choose ONLY at the very start (Turn 0: Refinement=0, Planning=0) IF AND ONLY IF the original query is a complex multi-entity comparison that needs decomposition into sub-queries. If the query is already an atomic question, DO NOT choose refinement!
   - 'finish': Choose ONLY if 'is_query_answered' is True AND a valid grounded Final Answer exists. If no context has been retrieved yet, 'finish' is STRICTLY FORBIDDEN!

CRITICAL INSTRUCTION: Output EXACTLY ONE RAW JSON tool call matching ValidatorDecision. Do not output text outside the JSON."""

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

        content_resp = ""
        try:
            response_msg = llm_with_tools.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            content_resp = getattr(response_msg, "content", str(response_msg))
            decision_obj = extract_validator_decision(response_msg, content_resp)
        except Exception as e:
            print(f"[VALIDATOR ERROR] Errore inferenza: {e}")
            decision_obj = None

        # Fallback intelligente e rigoroso se nessun JSON è stato prodotto
        if not decision_obj:
            print("[VALIDATOR] JSON non riconosciuto. Applico fallback euristico rigoroso.")
            if evaluating_final_synthesis:
                # Controlla se la risposta sintetizzata è valida e non un messaggio di errore/insufficienza
                failure_words = ["not enough information", "unable to", "cannot be answered", "errore", "fallback"]
                has_substance = len(final_answer.strip()) >= 10 and not any(w in final_answer.lower() for w in failure_words)
                is_supported = has_substance and bool(context)
                
                decision_obj = ValidatorDecision(
                    reflection="Evaluating synthesized answer against context.",
                    query_complexity="simple",
                    is_context_relevant=bool(context),
                    is_query_answered=is_supported,
                    reasoning="Synthesized answer is present and grounded in retrieved evidence." if is_supported else "Synthesized answer lacks sufficient grounding.",
                    feedback="Answer complete." if is_supported else "Retrieve more evidence for missing entities.",
                    next_action="finish" if is_supported else ("route_to_planning" if num_plan < MAX_PLANNING else "finish")
                )
            else:
                is_composite = any(w in q.lower() for w in ["same", "both", "compare", "differ"]) or (" and " in q.lower())
                next_act = "route_to_refinement" if (num_ref == 0 and is_composite) else "route_to_planning"
                decision_obj = ValidatorDecision(
                    reflection="Turn 0 evaluation: routing to refinement or planning.",
                    query_complexity="complex" if is_composite else "simple",
                    is_context_relevant=bool(context),
                    is_query_answered=False,
                    reasoning="Heuristic routing based on query complexity.",
                    feedback="Proceed with refinement/planning.",
                    next_action=next_act
                )

        print(f"      [REASONING]: {decision_obj.reasoning[:120]}...")
        print(f"      [SELF-RAG CRITIQUE]: IsRel={decision_obj.is_context_relevant} | IsSup={decision_obj.is_query_answered}")
        print(f"      [ACTION LLM]: {decision_obj.next_action.upper()}")

        # --- RIGOROSA APPLICAZIONE DEI FRAMEWORK ACCADEMICI SUL ROUTING ---
        act = decision_obj.next_action

        # 1. Regola Adaptive-RAG: Al turno 0 (nessun refinement ancora eseguito),
        # se la query è comparativa o composita (es. "Were X and Y born in same country?"),
        # deve essere instradata al Refiner per la scomposizione Least-to-Most (Zhou et al., 2022)
        if num_ref == 0 and num_plan == 0:
            is_composite = any(w in q.lower() for w in ["same", "both", "compare", "differ", "opposed to"]) or (" and " in q.lower())
            if is_composite or decision_obj.query_complexity == "complex":
                print("      [ADAPTIVE-RAG ROUTING] Rilevata query comparativa/complessa al turno 0. Instradamento al Refiner (Least-to-Most).")
                act = "route_to_refinement"

        # 2. Regola Post-Refinement: Se il refinement è già stato eseguito almeno una volta (num_ref > 0)
        # e non abbiamo ancora effettuato alcun ciclo di planning (num_plan == 0),
        # il sistema DEVE procedere a planning_node per il recupero delle evidenze!
        elif num_ref > 0 and num_plan == 0:
            print("      [POST-REFINEMENT ROUTING] Query già raffinata/decomposta. Instradamento obbligatorio al Planner per il recupero.")
            act = "route_to_planning"

        # 3. Regola Self-RAG (IsSup): Se la risposta non è supportata, non può terminare
        if act == "finish" and not decision_obj.is_query_answered and num_plan < MAX_PLANNING:
            print("      [SELF-RAG VETO] Risposta non supportata dal contesto (IsSup=False). Reindirizzo al Planner!")
            act = "route_to_planning"

        # 4. Se non è mai stato fatto planning e la risposta è assente o vuota, non si può terminare
        if act == "finish" and num_plan == 0:
            if num_ref == 0:
                act = "route_to_refinement"
            else:
                act = "route_to_planning"

        # 5. Limiti massimi di sicurezza per prevenire cicli infiniti
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

