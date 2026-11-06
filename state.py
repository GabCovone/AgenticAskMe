import ast
import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar, TypedDict
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class GraphState(TypedDict):
    original_query: str
    current_query: str
    retrieved_context: str
    num_refinement: int
    num_planning: int
    next_node: str
    feedback_history: List[str]
    final_answer: str


def _clean_json_candidate(raw: str) -> str:
    """Repairs common minor LLM JSON syntax deviations (Python booleans, trailing commas)."""
    s = raw.strip()
    s = re.sub(r":\s*True\b", ": true", s)
    s = re.sub(r":\s*False\b", ": false", s)
    s = re.sub(r":\s*None\b", ": null", s)
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    return s


def extract_json_dicts(text: str) -> List[Dict[str, Any]]:
    """
    Extracts candidate JSON dictionaries from raw model output:
    1. Direct full-string JSON parse
    2. Markdown ```json ... ``` blocks
    3. Balanced-brace sliding window extraction
    """
    if not isinstance(text, str) or not text.strip():
        return []

    results: List[Dict[str, Any]] = []

    def _try_parse(candidate: str) -> Optional[Dict[str, Any]]:
        cleaned = _clean_json_candidate(candidate)
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(cleaned)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return None

    # 1. Direct parse (fast path for GBNF-constrained JSON output)
    direct = _try_parse(text)
    if direct is not None:
        return [direct]

    # 2. Markdown fenced blocks
    for match in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE):
        parsed = _try_parse(match)
        if parsed is not None:
            results.append(parsed)
    if results:
        return results

    # 3. Balanced brace scan
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            esc = False
            for j in range(i, n):
                ch = text[j]
                if ch == '"' and not esc:
                    in_str = not in_str
                if not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            parsed = _try_parse(text[i : j + 1])
                            if parsed is not None:
                                results.append(parsed)
                            i = j
                            break
                esc = (ch == "\\") and not esc
        i += 1

    return results


def unwrap_tool_payload(data: Dict[str, Any], schema_cls: Type[T]) -> Dict[str, Any]:
    """
    Unwraps nested {'name': ..., 'arguments': {...}} wrappers if present,
    and rejects echoed JSON Schema definitions (where field values are dicts with 'type'/'description').
    """
    payload = data
    if "name" in payload and any(k in payload for k in ("arguments", "args", "parameters")):
        inner = payload.get("arguments", payload.get("args", payload.get("parameters")))
        if isinstance(inner, str):
            nested = extract_json_dicts(inner)
            if nested:
                payload = nested[0]
        elif isinstance(inner, dict):
            payload = inner

    if "properties" in payload and isinstance(payload.get("properties"), dict) and "type" in payload:
        # Model echoed a JSON schema wrapper; reject if values are schema descriptors
        props = payload["properties"]
        if all(isinstance(v, dict) and "type" in v for v in props.values()):
            return {}

    schema_fields = set(schema_cls.model_fields.keys())
    cleaned: Dict[str, Any] = {}
    for k, v in payload.items():
        if k not in schema_fields:
            continue
        # Guard against echoed schema field descriptors like {"type": "string", "description": "..."}
        if isinstance(v, dict) and ("type" in v or "description" in v):
            continue
        cleaned[k] = v
    return cleaned


def invoke_structured_json(
    llm: Any,
    schema_cls: Type[T],
    messages: List[Any],
) -> tuple[Optional[T], str]:
    """
    Invokes `llm` with deterministic GBNF JSON Schema grammar enforcement (`response_format`)
    when supported by ChatLlamaCpp, with transparent fallback for generic/mock LLMs.
    Returns `(validated_model_or_None, raw_content_str)`.
    """
    json_schema = schema_cls.model_json_schema()
    response_msg = None

    # Ensure ChatLlamaCpp uses non-streaming completion so llama-cpp-python enforces GBNF schema
    if hasattr(llm, "streaming") and getattr(llm, "streaming", False) and hasattr(llm, "client"):
        try:
            llm.streaming = False
        except Exception:
            pass

    # 1. Primary path: GBNF grammar-constrained JSON generation in llama-cpp-python
    try:
        response_msg = llm.invoke(
            messages,
            response_format={
                "type": "json_object",
                "schema": json_schema,
            },
        )
    except TypeError:
        # Mock or custom Runnable that does not accept extra kwargs
        response_msg = llm.invoke(messages)
    except Exception:
        # Fallback if backend rejects schema grammar
        response_msg = llm.invoke(messages)

    raw_content = getattr(response_msg, "content", str(response_msg)) or ""

    # Check native tool_calls first if a mock/chat model returned them
    if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
        tc = response_msg.tool_calls[0]
        tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
        if isinstance(tc_args, str):
            extracted = extract_json_dicts(tc_args)
            tc_args = extracted[0] if extracted else {}
        if isinstance(tc_args, dict) and tc_args:
            cleaned = unwrap_tool_payload(tc_args, schema_cls)
            try:
                return schema_cls.model_validate(cleaned), raw_content
            except Exception:
                pass

    # Parse and validate JSON dicts from content
    for cand in extract_json_dicts(raw_content):
        cleaned = unwrap_tool_payload(cand, schema_cls)
        if not cleaned:
            continue
        try:
            return schema_cls.model_validate(cleaned), raw_content
        except Exception:
            continue

    return None, raw_content
