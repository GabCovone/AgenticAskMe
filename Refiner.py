import ast
import json
import logging
import re
import sys
import warnings
from typing import Any, Dict, List, Literal, Optional, Set
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from state import GraphState, invoke_structured_json

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
    Applies 'Least-to-Most Prompting' (Zhou et al., 2022) to decompose a complex comparative
    or multi-entity query into independent, self-contained atomic sub-questions.
    """
    return sub_queries


@tool
def expand_query(reflection: str, reasoning: str, pseudo_document: str) -> str:
    """
    Applies 'Query2doc' (Wang et al., 2023) for semantic query expansion on short or telegraphic queries.
    """
    return pseudo_document


@tool
def rewrite_query(reflection: str, reasoning: str, rewritten_query: str) -> str:
    """
    Applies 'Rewrite-Retrieve-Read' (Ma et al., 2023) to remove conversational noise while preserving named entities.
    """
    return rewritten_query


@tool
def keep_query_unchanged(reflection: str, reasoning: str) -> str:
    """
    Keeps the query unchanged when it is already atomic and ready for retrieval.
    """
    return "UNCHANGED"


# Schema GBNF piatto per garantire output JSON deterministico su ChatLlamaCpp
class RefinerDecision(BaseModel):
    """Structured query refinement decision selecting one of the four academic reformulation tools."""
    reflection: str = Field(
        description="Concise 1-sentence critique of current query complexity and Validator feedback."
    )
    reasoning: str = Field(
        description="Concise 1-2 sentence justification for decomposing, rewriting, expanding, or keeping the query."
    )
    tool_name: Literal["decompose_query", "rewrite_query", "expand_query", "keep_query_unchanged"] = Field(
        description="Selected refinement tool."
    )
    sub_queries: List[str] = Field(
        description="List of atomic self-contained sub-questions when tool_name is 'decompose_query' (empty list [] otherwise). Preserve exact entity spelling!"
    )
    rewritten_text: str = Field(
        description="Rewritten query or Query2doc pseudo-passage when tool_name is 'rewrite_query' or 'expand_query' (empty string '' otherwise)."
    )


def clean_subquery_text(s: str) -> str:
    """Strips markdown bullets, numbering, and sub-query labels from generated sub-questions."""
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"^\*+\s*", "", s).strip()
    s = re.sub(r"^\*?(?:sub-?query|question)\s*\d*\s*[:\*\-]*\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^\d+[\.\)]\s*", "", s).strip()
    s = re.sub(r"^[-*]\s*", "", s).strip()
    s = re.sub(r"\*+$", "", s).strip()
    return s.strip()


def heuristic_least_to_most_with_entities(query: str) -> tuple[Optional[List[str]], List[str]]:
    """
    Deterministic Least-to-Most decomposition (Zhou et al., 2022) for comparative multi-entity queries.
    Returns (sub_queries_or_None, [entity1, entity2]).
    """
    clean_q = query.strip().rstrip("?").strip()

    # Pattern 1: "Were/Are/Was/Is X and Y (located) in/of/born in/share the same [trait]?"
    pattern1 = (
        r"^(?:Were|Are|Was|Is)\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)\s+"
        r"(?:located\s+in\s+the\s+same|in\s+the\s+same|of\s+the\s+same|born\s+in\s+the\s+same|share\s+the\s+same)\s+([a-zA-Z\s]+)$"
    )
    match1 = re.search(pattern1, clean_q, re.IGNORECASE)
    if match1:
        e1 = match1.group(1).strip()
        e2 = match1.group(2).strip()
        trait = match1.group(3).strip()
        return [f"What is the {trait} of {e1}?", f"What is the {trait} of {e2}?"], [e1, e2]

    # Pattern 2: "Are/Were/Was/Is/Did X and Y both [predicate]?"
    pattern2 = r"^(?:Are|Were|Was|Is|Did)\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)\s+both\s+(.+)$"
    match2 = re.search(pattern2, clean_q, re.IGNORECASE)
    if match2:
        e1 = match2.group(1).strip()
        e2 = match2.group(2).strip()
        trait = match2.group(3).strip()
        orig_starter = clean_q.split()[0].capitalize()
        if orig_starter in ["Are", "Were", "Was", "Is"]:
            return [f"Is {e1} {trait}?", f"Is {e2} {trait}?"], [e1, e2]
        return [f"Did {e1} {trait}?", f"Did {e2} {trait}?"], [e1, e2]

    # Pattern 3: "Who/Which is older/younger/taller/larger/bigger, X or Y?"
    pattern3 = r"^(?:Who|Which\b[^,]*)\s+is\s+(older|younger|taller|shorter|larger|smaller|bigger),\s*(.+?)\s+or\s+(.+?)$"
    match3 = re.search(pattern3, clean_q, re.IGNORECASE)
    if match3:
        adj = match3.group(1).lower()
        e1 = match3.group(2).strip()
        e2 = match3.group(3).strip()
        if adj in ("older", "younger"):
            return [f"When was {e1} born?", f"When was {e2} born?"], [e1, e2]
        return [f"What is the {adj} attribute or size of {e1}?", f"What is the {adj} attribute or size of {e2}?"], [e1, e2]

    return None, []


def heuristic_least_to_most(query: str) -> Optional[List[str]]:
    subs, _ = heuristic_least_to_most_with_entities(query)
    return subs


# --- 2. NODO REFINER ---
def refiner_node(state: GraphState, llm) -> dict:
    num_ref = state.get("num_refinement", 0)
    print(f"\n--- [REFINER] Inizio Ottimizzazione Query (Ciclo {num_ref + 1}) ---")
    current_query_raw = state.get("current_query", state.get("original_query", ""))

    queries = [q.strip() for q in current_query_raw.split("\n---\n") if q.strip()]
    if not queries:
        queries = [current_query_raw]

    system_prompt = """You are the Autonomous Query Refinement Agent implementing:
- Least-to-Most Prompting (Zhou et al., 2022) via 'decompose_query'
- Rewrite-Retrieve-Read (Ma et al., 2023) via 'rewrite_query'
- Query2doc (Wang et al., 2023) via 'expand_query'
- Direct passthrough via 'keep_query_unchanged'

OPERATIONAL RULES:
1. EXACT ENTITY PRESERVATION: Copy every proper noun, band name, person name, film/album title, and place name character-for-character from the input query. NEVER alter or substitute entity words!
2. COMPARATIVE DECOMPOSITION: When the query compares two entities (e.g. "Were X and Y of the same...", "Are X and Y both...", "Who is older, X or Y?"), set tool_name='decompose_query' and output 2 self-contained atomic sub_queries, one per entity.
3. ATOMIC QUERIES: When a query already focuses on a single entity or sequential bridge lookup, set tool_name='keep_query_unchanged' with sub_queries=[] and rewritten_text=''.

Return a single JSON object conforming strictly to the RefinerDecision schema."""

    feedback_history = state.get("feedback_history", [])
    feedback_text = ""
    if feedback_history:
        feedback_text = "\n\nFEEDBACK FROM VALIDATOR:\n- " + "\n- ".join(feedback_history)

    final_processed_queries: List[str] = []
    tools_used_global: Set[str] = set()

    for q_idx, q in enumerate(queries):
        print(f"[REFINER] Analisi sotto-query {q_idx+1}/{len(queries)}: '{q}'")
        user_prompt = f"""Global Original Query: '{state.get('original_query', '')}'
Current Sub-Query to refine: '{q}'{feedback_text}

Issue your RefinerDecision JSON."""

        decision: Optional[RefinerDecision] = None
        content_resp = ""
        try:
            decision, content_resp = invoke_structured_json(
                llm,
                RefinerDecision,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
            )
        except Exception as e:
            print(f"[REFINER ERROR] Errore inferenza per la sub-query: {e}")
            decision = None

        heuristic_subs, expected_entities = heuristic_least_to_most_with_entities(q)

        # Fallback se il modello non ha restituito uno schema valido
        if decision is None:
            if heuristic_subs:
                print("   [REFINER] [Least-to-Most Fallback] Scomposizione deterministica applicata per query comparativa.")
                decision = RefinerDecision(
                    reflection="Comparative multi-entity query detected.",
                    reasoning="Splitting comparative query into independent atomic sub-queries.",
                    tool_name="decompose_query",
                    sub_queries=heuristic_subs,
                    rewritten_text="",
                )
            else:
                decision = RefinerDecision(
                    reflection="Query is atomic.",
                    reasoning="Keeping atomic query unchanged.",
                    tool_name="keep_query_unchanged",
                    sub_queries=[],
                    rewritten_text="",
                )

        t_name = decision.tool_name
        print(f"   [REFINER REASONING]: {decision.reasoning}")
        print(f"   [REFINER TOOL CALL]: '{t_name}'")

        is_comp = heuristic_subs is not None or any(
            w in q.lower() for w in ["same ", "both ", "compare ", "differ"]
        )

        if t_name == "decompose_query" and (is_comp or len(queries) == 1):
            cleaned_subs = [clean_subquery_text(s) for s in decision.sub_queries if clean_subquery_text(s)]
            joined_lower = " ".join(cleaned_subs).lower()

            # Verifica che la scomposizione sia atomica e che NESSUNA entità nominata sia stata alterata/persa
            still_composite = any(
                any(
                    w in sq.lower()
                    for w in [
                        "same ", "both ", "what about", "as well as", "either", "neither",
                        "who is older", "who is younger", "who is taller", "which is older", "which is larger"
                    ]
                )
                for sq in cleaned_subs
            ) or any(" and " in sq.lower() for sq in cleaned_subs)
            missing_expected_entity = any(ent.lower() not in joined_lower for ent in expected_entities)

            if (not cleaned_subs or len(cleaned_subs) < 2 or still_composite or missing_expected_entity) and heuristic_subs:
                print("   [REFINER] Verifica preservazione entità Least-to-Most: ripristino sotto-query atomiche esatte.")
                cleaned_subs = heuristic_subs

            if cleaned_subs:
                invoked_subs = decompose_query.invoke({
                    "reflection": decision.reflection,
                    "reasoning": decision.reasoning,
                    "sub_queries": cleaned_subs,
                })
                tools_used_global.add("decompose_query")
                final_processed_queries.extend(invoked_subs)
            else:
                keep_query_unchanged.invoke({"reflection": decision.reflection, "reasoning": decision.reasoning})
                tools_used_global.add("keep_query_unchanged")
                final_processed_queries.append(q)

        elif is_comp and heuristic_subs:
            # Garanzia Least-to-Most: se la query è comparativa ma il modello ha scelto keep_query_unchanged
            print("   [REFINER] Applicazione garantita di Least-to-Most per query comparativa.")
            invoked_subs = decompose_query.invoke({
                "reflection": decision.reflection,
                "reasoning": "Guaranteed Least-to-Most decomposition for comparative query.",
                "sub_queries": heuristic_subs,
            })
            tools_used_global.add("decompose_query")
            final_processed_queries.extend(invoked_subs)

        elif t_name == "rewrite_query" and decision.rewritten_text.strip():
            rw = clean_subquery_text(decision.rewritten_text)
            # Impedisci che una riscrittura cancelli entità nominate tra virgolette o codici
            quoted_orig = re.findall(r'["\']([^"\']+)["\']', q)
            if any(qo.lower() not in rw.lower() for qo in quoted_orig):
                rw = q
            invoked_rw = rewrite_query.invoke({
                "reflection": decision.reflection,
                "reasoning": decision.reasoning,
                "rewritten_query": rw,
            })
            tools_used_global.add("rewrite_query")
            final_processed_queries.append(invoked_rw)

        elif t_name == "expand_query" and decision.rewritten_text.strip():
            invoked_exp = expand_query.invoke({
                "reflection": decision.reflection,
                "reasoning": decision.reasoning,
                "pseudo_document": decision.rewritten_text.strip(),
            })
            tools_used_global.add("expand_query")
            final_processed_queries.append(f"{q}\n[Expanded Context]: {invoked_exp}")

        else:
            keep_query_unchanged.invoke({"reflection": decision.reflection, "reasoning": decision.reasoning})
            tools_used_global.add("keep_query_unchanged")
            final_processed_queries.append(q)

    # Deduplicazione ordinata delle sotto-query
    seen = set()
    deduped_queries = []
    for sq in final_processed_queries:
        sq_clean = clean_subquery_text(sq)
        if sq_clean and sq_clean.lower() not in seen:
            seen.add(sq_clean.lower())
            deduped_queries.append(sq_clean)
    final_processed_queries = deduped_queries if deduped_queries else [current_query_raw]

    final_query_str = "\n---\n".join(final_processed_queries)
    if tools_used_global:
        print(f"[REFINER] Modifiche applicate con tool: {', '.join(sorted(tools_used_global)).upper()}")
    else:
        print("[REFINER] Nessun tool chiamato, query mantenuta inalterata.")
    print(f"[REFINER] Output finale:\n{final_query_str}")

    return {
        "current_query": final_query_str,
        "num_refinement": num_ref + 1,
    }
