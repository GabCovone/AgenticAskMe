import re
import sys
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from state import GraphState, invoke_structured_json

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --- COSTANTI DI CONTROLLO DEL CICLO AGENTICO ---
MAX_REFINEMENT = 4
MAX_PLANNING = 4


# --- SCHEMA DI DECISIONE SEMANTICA (SELF-RAG + ADAPTIVE-RAG + REFLEXION) ---
# Nota sull'ordine dei campi: reflection e reasoning precedono i token booleani e next_action
# per garantire Chain-of-Thought prima della decisione durante il campionamento GBNF.
class ValidatorDecision(BaseModel):
    """Semantic validation, Self-RAG critique, and Adaptive-RAG routing decision."""
    reflection: str = Field(
        description="REFLEXION (concise, 1 sentence): Verbal critique of past iterations, missing entities, or hallucinations."
    )
    reasoning: str = Field(
        description="CHAIN OF THOUGHT (concise, 1-2 sentences): Step-by-step justification for relevance, support, and routing."
    )
    query_complexity: Literal["simple", "complex", "already_decomposed"] = Field(
        description="ADAPTIVE-RAG: 'simple' for single-entity/bridge lookup, 'complex' for multi-entity comparison needing decomposition, 'already_decomposed' for atomic sub-queries."
    )
    is_context_relevant: bool = Field(
        description="SELF-RAG (IsRel): True if retrieved context contains factual evidence about the target entities."
    )
    is_query_answered: bool = Field(
        description="SELF-RAG (IsSup): True if the proposed answer directly resolves the question and is factually supported by retrieved evidence. For Yes/No questions, 'no' is True when evidence confirms the entities differ in the requested property. False if unsupported, circular, or wrong entity type."
    )
    feedback: str = Field(
        description="Actionable instructions for the next node (e.g. which entity or fact is still missing)."
    )
    next_action: Literal["route_to_refinement", "route_to_planning", "finish"] = Field(
        description="Routing target: 'route_to_refinement', 'route_to_planning', or 'finish'."
    )


def is_comparative_query(q: str) -> bool:
    """Detects parallel multi-entity comparative queries suited for Least-to-Most decomposition."""
    ql = q.strip().lower()
    if any(w in ql for w in ["same ", "both ", "compare ", "differ", "opposed to"]):
        return True
    if re.search(r"^(?:who|which\b[^,]*)\s+is\s+(?:older|younger|taller|shorter|larger|smaller|bigger|earlier|later)\b.+\bor\b", ql):
        return True
    if re.search(r"^(?:were|are|was|is|did|do|does)\s+.+\s+and\s+.+", ql):
        return True
    return False


def extract_compared_entities(original_query: str) -> list[str]:
    """Extracts the two primary entities in comparative questions (if applicable) to verify complete grounding."""
    clean_q = original_query.strip().rstrip("?").strip()
    patterns = [
        r"^(?:Were|Are|Was|Is)\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)\s+(?:located\s+in\s+the\s+same|in\s+the\s+same|of\s+the\s+same|born\s+in\s+the\s+same|share\s+the\s+same)\b",
        r"^(?:Are|Were|Was|Is|Did)\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)\s+both\b",
        r"^Who is (?:older|younger|taller|larger|bigger),\s*(.+?)\s+or\s+(.+?)$",
    ]
    for pat in patterns:
        m = re.search(pat, clean_q, flags=re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
    return []


def _has_valid_evidence(context: str) -> bool:
    ctx = (context or "").strip().lower()
    if len(ctx) < 40:
        return False
    has_doc_marker = any(
        m in ctx for m in ("--- wikipedia result", "--- web result", "--- documento locale", "title:", "search result")
    )
    if "nessun risultato" in ctx and not has_doc_marker:
        return False
    return True


def _text_fallback_decision(content_resp: str, evaluating_final_synthesis: bool) -> Optional[ValidatorDecision]:
    """Safe text fallback if a non-GBNF mock/custom LLM returns plain text."""
    # Remove schema field names so 'is_query_answered' never false-matches 'answered'
    sanitized = re.sub(r"is_query_answered|is_context_relevant|query_complexity|next_action", "", content_resp.lower())
    if "route_to_refinement" in content_resp.lower() or "to refinement" in sanitized:
        return ValidatorDecision(
            reflection="Text fallback indicates query decomposition/refinement.",
            reasoning="Extracted refinement routing intent.",
            query_complexity="complex",
            is_context_relevant=False,
            is_query_answered=False,
            feedback="Decompose comparative query into atomic sub-queries.",
            next_action="route_to_refinement",
        )
    if "route_to_planning" in content_resp.lower() or "to planning" in sanitized:
        return ValidatorDecision(
            reflection="Text fallback indicates retrieval planning.",
            reasoning="Extracted planning routing intent.",
            query_complexity="simple",
            is_context_relevant=False,
            is_query_answered=False,
            feedback="Retrieve factual evidence.",
            next_action="route_to_planning",
        )
    if evaluating_final_synthesis and re.search(r'\b"finish"\b|\bfinish\b', content_resp.lower()):
        return ValidatorDecision(
            reflection="Text fallback indicates synthesis evaluation completed.",
            reasoning="Extracted finish intent during final synthesis check.",
            query_complexity="simple",
            is_context_relevant=True,
            is_query_answered=True,
            feedback="Answer verified.",
            next_action="finish",
        )
    return None


# --- NODO VALIDATORE ---
def validator_node(state: GraphState, llm) -> dict:
    num_ref = state.get("num_refinement", 0)
    num_plan = state.get("num_planning", 0)
    print(f"\n[VALIDATOR] Avvio ciclo di valutazione e routing (Ref={num_ref}/{MAX_REFINEMENT}, Plan={num_plan}/{MAX_PLANNING})...")

    query_raw = state.get("current_query", state.get("original_query", ""))
    queries = [q.strip() for q in query_raw.split("\n---\n") if q.strip()]
    if not queries:
        queries = [state.get("original_query", "")]

    context = state.get("retrieved_context", "")
    final_answer = state.get("final_answer", "")
    original_query = state.get("original_query", "")

    evaluating_final_synthesis = bool(final_answer and num_plan > 0)
    if evaluating_final_synthesis:
        print("[VALIDATOR] Valutazione della Proposta Final Answer contro il Contesto (Self-RAG Grounding)...")
        queries = [original_query]
        system_prompt = """You are the Self-RAG Quality Judge and Grounding Evaluator (Asai et al., 2023).
Evaluate whether the PROPOSED FINAL ANSWER is strictly supported by the RETRIEVED EVIDENCE.

EVALUATION RULES:
1. is_context_relevant (IsRel): True if the retrieved evidence contains factual information about the entities in the question.
2. is_query_answered (IsSup):
   - True if the Proposed Final Answer directly answers the question AND every factual claim in it is supported by the Retrieved Evidence.
   - For polar questions (Yes/No): A negative answer starting with 'no' IS FULLY VALID and SUPPORTED (is_query_answered = True) when the evidence shows the two entities differ in the requested property (e.g. different neighborhoods: District Alpha vs District Beta).
   - For comparative questions involving two entities, BOTH entities must have supporting facts in the Retrieved Evidence.
   - False if the answer answers with the wrong type of entity (e.g. returns a person's name when asked for a government position or title), repeats the question circularly, or hallucinates facts not in the evidence.
3. next_action:
   - 'finish' when is_query_answered is True.
   - 'route_to_planning' when critical evidence is missing, the answer is the wrong entity type, or unsupported.

Return a single JSON object conforming strictly to the ValidatorDecision schema."""
    else:
        system_prompt = """You are the Semantic Routing Validator for an Agentic Information Retrieval system implementing Adaptive-RAG (Jeong et al., 2024), Self-RAG (Asai et al., 2023), and Reflexion (Shinn et al., 2023).

ROUTING RULES:
1. 'route_to_refinement': Choose ONLY at Turn 0 (Refinement=0, Planning=0) when the question compares two independent entities (e.g. "Were X and Y of the same...", "Are X and Y both...", "Who is older, X or Y?") and needs decomposition into sub-queries.
2. 'route_to_planning': Choose whenever no context has been retrieved yet and the query is atomic or a sequential bridge question, OR when sub-queries are already decomposed. NEVER choose 'finish' when no context has been retrieved!
3. 'finish': Forbidden before retrieval. Only valid when a grounded Final Answer already exists.

Return a single JSON object conforming strictly to the ValidatorDecision schema."""

    eval_context = context[:3000] if context else "No context retrieved yet."
    feedbacks = []
    actions = []

    for q_idx, q in enumerate(queries):
        if not evaluating_final_synthesis:
            print(f"[VALIDATOR] Analisi sotto-query {q_idx+1}/{len(queries)}: '{q}'")
            user_prompt = f"""Original User Request: '{original_query}'
Sub-Query under evaluation: '{q}'
Current Retrieved Context:
{eval_context}

Metrics: Refinement Count={num_ref}/{MAX_REFINEMENT}, Planning Count={num_plan}/{MAX_PLANNING}

Issue your ValidatorDecision JSON."""
        else:
            user_prompt = f"""QUESTION: '{original_query}'

RETRIEVED EVIDENCE:
{eval_context}

PROPOSED FINAL ANSWER:
{final_answer}

Verify strict factual support and issue your ValidatorDecision JSON."""

        decision_obj = None
        content_resp = ""
        try:
            decision_obj, content_resp = invoke_structured_json(
                llm,
                ValidatorDecision,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
            )
            if decision_obj is None and content_resp:
                decision_obj = _text_fallback_decision(content_resp, evaluating_final_synthesis)
        except Exception as e:
            print(f"[VALIDATOR ERROR] Errore inferenza: {e}")
            decision_obj = None

        # Fallback deterministico di sicurezza
        if not decision_obj:
            print("[VALIDATOR] Applico fallback deterministico di sicurezza.")
            if evaluating_final_synthesis:
                has_ev = _has_valid_evidence(context)
                decision_obj = ValidatorDecision(
                    reflection="Deterministic evaluation of synthesized answer against retrieved context.",
                    reasoning="Checking answer substance and evidence presence.",
                    query_complexity="simple",
                    is_context_relevant=has_ev,
                    is_query_answered=has_ev,
                    feedback="Answer verified." if has_ev else "Missing supporting evidence in retrieved context.",
                    next_action="finish" if has_ev else ("route_to_planning" if num_plan < MAX_PLANNING else "finish"),
                )
            else:
                comp = is_comparative_query(q) and len(queries) == 1
                next_act = "route_to_refinement" if (num_ref == 0 and num_plan == 0 and comp) else "route_to_planning"
                decision_obj = ValidatorDecision(
                    reflection="Initial routing assessment.",
                    reasoning="Routing based on query structure and retrieval state.",
                    query_complexity="complex" if comp else ("already_decomposed" if len(queries) > 1 else "simple"),
                    is_context_relevant=bool(context),
                    is_query_answered=False,
                    feedback="Decompose comparative query." if next_act == "route_to_refinement" else "Execute retrieval plan.",
                    next_action=next_act,
                )

        print(f"      [REASONING]: {decision_obj.reasoning[:140]}...")
        print(f"      [SELF-RAG CRITIQUE]: IsRel={decision_obj.is_context_relevant} | IsSup={decision_obj.is_query_answered}")
        print(f"      [ACTION LLM]: {decision_obj.next_action.upper()}")

        act = decision_obj.next_action

        # --- VERIFICA DETERMINISTICA DI GROUNDING SULLA FINAL ANSWER (SELF-RAG) ---
        if evaluating_final_synthesis:
            ans_stripped = final_answer.strip()
            ans_lower = ans_stripped.lower()
            ctx_lower = (context or "").lower()
            ctx_digits_norm = ctx_lower.replace(",", "")
            ans_digits_norm = ans_lower.replace(",", "")

            failure_words = [
                "unable to", "cannot", "don't know", "not enough", "enough information",
                "do not have", "don't have", "no evidence", "not possible", "cannot determine",
                "cannot be determined", "fallback", "errore durante", "not explicitly", "no specific",
                "i do not have", "unknown"
            ]
            # Concise entity/number/polar answers (e.g. 'yes', 'no', 'Novel Series', '15,000 seated') are valid (>=2 chars),
            # whereas echoed questions ending in '?' or failure phrases are rejected.
            is_echoed_question = ans_stripped.endswith("?")
            is_substantive = (
                len(ans_stripped) >= 2
                and not is_echoed_question
                and not any(w in ans_lower for w in failure_words)
            )
            has_evidence = _has_valid_evidence(context)

            # Per domande comparative a 2 entità, verifica che entrambe siano state recuperate nel contesto
            compared_ents = extract_compared_entities(original_query)
            missing_ents = []
            for ent in compared_ents:
                ent_tokens = [t.lower() for t in re.findall(r"\w{3,}", ent)]
                if ent_tokens and not any(tok in ctx_lower for tok in ent_tokens):
                    missing_ents.append(ent)

            # Verifica overlap dei token fattuali della risposta con il contesto
            is_polar_ans = bool(re.match(r"^(?:yes|no)\b", ans_lower))
            stop_ans = {
                "yes", "no", "the", "and", "both", "were", "was", "are", "is", "from", "with",
                "that", "this", "have", "been", "same", "located", "born", "nationality", "older", "younger"
            }
            orig_tokens = set(re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]{2,}", original_query.lower().replace(",", "")))
            ans_tokens = [
                w for w in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]{2,}", ans_digits_norm)
                if w not in stop_ans
            ]
            novel_ans_tokens = [w for w in ans_tokens if w not in orig_tokens]
            is_choice_q = " or " in original_query.lower()
            grounded_tokens = [w for w in ans_tokens if w in ctx_digits_norm]
            grounding_ratio = (len(grounded_tokens) / len(ans_tokens)) if ans_tokens else (1.0 if is_polar_ans and has_evidence else 0.0)

            # Rilevamento risposte circolari che ripetono il nome d'arte della domanda (es. 'Aladdin' per stage name 'Aladin')
            is_echoed_stage_name = False
            m_stage = re.search(r"\bstage name\s+([A-Za-z0-9]+)\b", original_query, flags=re.IGNORECASE)
            if m_stage:
                st_name = m_stage.group(1).lower()
                ans_core = re.sub(r"[^\w]", "", ans_lower)
                st_core = re.sub(r"[^\w]", "", st_name)
                if ans_core.startswith(st_core[:4]) and len(ans_core) <= len(st_core) + 2:
                    is_echoed_stage_name = True

            if not is_substantive or not has_evidence:
                print("      [SELF-RAG GROUNDING CHECK] Risposta elusiva, interrogativa o priva di evidenze. Reindirizzo al Planner.")
                decision_obj.is_query_answered = False
                decision_obj.feedback = "Previous answer was empty, interrogative, or lacked retrieved evidence. Retrieve missing facts via web_search."
                act = "route_to_planning"
            elif missing_ents:
                print(f"      [SELF-RAG GROUNDING CHECK] Evidenze mancanti per entità comparate: {missing_ents}. Reindirizzo al Planner.")
                decision_obj.is_query_answered = False
                decision_obj.feedback = f"Missing retrieved evidence for: {', '.join(missing_ents)}. Search explicitly for {', '.join(missing_ents)}."
                act = "route_to_planning"
            elif is_echoed_stage_name:
                print(f"      [SELF-RAG GROUNDING CHECK] La risposta '{ans_stripped}' ripete solo il nome d'arte invece dell'identità reale. Reindirizzo al Planner.")
                decision_obj.is_query_answered = False
                decision_obj.feedback = f"Answer '{ans_stripped}' merely repeats the stage name. Search for the real identity and name of the person."
                act = "route_to_planning"
            elif not is_polar_ans and ans_tokens and grounding_ratio == 0.0:
                print("      [SELF-RAG GROUNDING CHECK] Nessun token della risposta trovato nel contesto recuperato (possibile allucinazione). Reindirizzo al Planner.")
                decision_obj.is_query_answered = False
                decision_obj.feedback = f"Proposed answer '{ans_stripped}' is not grounded in retrieved text. Search for explicit supporting evidence."
                act = "route_to_planning"
            elif not is_polar_ans and not is_choice_q and not novel_ans_tokens:
                print("      [SELF-RAG GROUNDING CHECK] La risposta ripete solo parole della domanda senza aggiungere l'entità specifica. Reindirizzo al Planner.")
                decision_obj.is_query_answered = False
                decision_obj.feedback = f"Answer '{ans_stripped}' only repeats words already in the question. Perform Hop 2 search to find the specific missing entity/location."
                act = "route_to_planning"
            elif is_polar_ans and not missing_ents and has_evidence and (decision_obj.next_action == "finish" or decision_obj.is_query_answered):
                print(f"      [SELF-RAG GROUNDING CHECK] Risposta polare verificata con evidenze presenti per entrambe le entità.")
                decision_obj.is_query_answered = True
                decision_obj.is_context_relevant = True
                act = "finish"
            elif decision_obj.is_query_answered and (grounding_ratio >= 0.5 or is_polar_ans) and decision_obj.next_action == "finish":
                print(f"      [SELF-RAG GROUNDING CHECK] Risposta verificata e supportata dal contesto (grounding ratio: {grounding_ratio:.2f}).")
                decision_obj.is_context_relevant = True
                act = "finish"
            elif not is_polar_ans and novel_ans_tokens and all(w in ctx_digits_norm for w in novel_ans_tokens) and decision_obj.is_context_relevant and decision_obj.next_action == "finish":
                print(f"      [SELF-RAG GROUNDING CHECK] Risposta canonica verificata nel contesto (grounding ratio: {grounding_ratio:.2f}).")
                decision_obj.is_query_answered = True
                act = "finish"

        # 1. Regola Adaptive-RAG al Turno 0: instrada query comparative multi-entità al Refiner (Least-to-Most)
        if num_ref == 0 and num_plan == 0:
            if is_comparative_query(q):
                print("      [ADAPTIVE-RAG ROUTING] Rilevata query comparativa multi-entità al turno 0. Instradamento al Refiner (Least-to-Most).")
                act = "route_to_refinement"
            else:
                act = "route_to_planning"

        # 2. Regola Post-Refinement: dopo il refinement iniziale, procedi obbligatoriamente al recupero nel Planner
        elif num_ref > 0 and num_plan == 0:
            print("      [POST-REFINEMENT ROUTING] Query già raffinata/decomposta. Instradamento obbligatorio al Planner.")
            act = "route_to_planning"

        # 3. Regola Self-RAG (IsSup): se non supportata e sotto il limite di planning, torna al Planner
        if act == "finish" and not decision_obj.is_query_answered and num_plan < MAX_PLANNING:
            print("      [SELF-RAG VETO] Risposta non supportata dal contesto (IsSup=False). Reindirizzo al Planner!")
            act = "route_to_planning"

        # 4. Impedisci finish prematuro senza alcun ciclo di planning
        if act == "finish" and num_plan == 0:
            act = "route_to_refinement" if (num_ref == 0 and is_comparative_query(q)) else "route_to_planning"

        # 5. Limiti massimi contro loop infiniti
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

    if "route_to_refinement" in actions:
        overall_action = "route_to_refinement"
    elif "route_to_planning" in actions:
        overall_action = "route_to_planning"
    else:
        overall_action = "finish"

    print(f"[VALIDATOR] Instradamento globale verso nodo: {overall_action.upper()}\n")

    state_update = {
        "next_node": overall_action,
        "feedback_history": feedbacks,
    }

    if final_answer and overall_action != "finish":
        print("[VALIDATOR] Final Answer respinta dal controllo Self-RAG. Resetto e chiedo nuova ricerca.")
        state_update["final_answer"] = ""

    return state_update
