import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import GraphState, invoke_structured_json
import os
from local_kb import search_index

# Modalità di retrieval globale configurabile a runtime:
# - 'full' (default): Sistema Integrale (Adaptive-RAG con KB locale + Ricerca Web Live).
# - 'baseline': Sistema Baseline (Accesso consentito ESCLUSIVAMENTE alla KB Locale di Wikipedia).
RETRIEVAL_MODE: str = os.environ.get("AGENTIC_IR_MODE", "full").lower()


def set_retrieval_mode(mode: str) -> None:
    """
    Imposta a runtime la modalità di retrieval del sistema senza modificare alcun prompt:
    - 'full': abilita il sistema completo con Adaptive-RAG, KB locale e Web Search.
    - 'baseline': vincola rigidamente il sistema alla sola Knowledge Base locale di Wikipedia.
    """
    global RETRIEVAL_MODE
    if str(mode).lower() in ("baseline", "local", "local_only", "kb_only"):
        RETRIEVAL_MODE = "baseline"
    else:
        RETRIEVAL_MODE = "full"


def get_retrieval_mode() -> str:
    """Restituisce la modalità di retrieval attualmente attiva ('full' oppure 'baseline')."""
    return RETRIEVAL_MODE

warnings.filterwarnings("ignore")
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# In-memory HTTP response cache and polite rate-limiter for MediaWiki API
_WIKI_CACHE: Dict[str, dict] = {}
_LAST_WIKI_TS: float = 0.0
MIN_WIKI_INTERVAL: float = 0.55

WIKI_HEADERS = {
    "User-Agent": "AgenticAskMeIR/1.0 (https://github.com/GabCovone/AgenticAskMe; academic-ir-evaluation)",
    "Accept-Encoding": "gzip",
}


def clean_query_term(q: str) -> str:
    """
    Sanitizes search strings for BM25 and DuckDuckGo by stripping brackets, quotes,
    and boolean operators while preserving slashes, hyphens, and accented characters.
    """
    if not isinstance(q, str):
        return ""
    cleaned = re.sub(r"[\[\]\(\)\{\}\"']", " ", q)
    cleaned = re.sub(r"\b(AND|OR|NOT)\b", " ", cleaned)
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned if cleaned else q.strip()


def sanitize_preserving_quotes(q: str) -> str:
    """Keeps double quotes and parentheses intact for exact-entity extraction in Wikipedia API."""
    if not isinstance(q, str):
        return ""
    cleaned = re.sub(r"[\[\]\{\}]", " ", q)
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned


def _wiki_fetch_json(url: str, timeout: int = 8) -> dict:
    """Fetches JSON from MediaWiki API with caching, polite pacing, gzip support, and HTTP 429 retry."""
    global _LAST_WIKI_TS
    if url in _WIKI_CACHE:
        return _WIKI_CACHE[url]
    import gzip
    for attempt in range(3):
        elapsed = time.time() - _LAST_WIKI_TS
        if elapsed < MIN_WIKI_INTERVAL:
            time.sleep(MIN_WIKI_INTERVAL - elapsed)
        try:
            _LAST_WIKI_TS = time.time()
            req = urllib.request.Request(url, headers=WIKI_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
                if resp.info().get("Content-Encoding") == "gzip":
                    raw_bytes = gzip.decompress(raw_bytes)
                data = json.loads(raw_bytes.decode("utf-8"))
                _WIKI_CACHE[url] = data
                return data
        except urllib.error.HTTPError as he:
            if he.code == 429 and attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            return {}
        except Exception:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            return {}
    return {}


# --- RETRIEVAL TOOLS ---

@tool
def local_kb_search(query: str) -> str:
    """
    Executes a search on the local Wikipedia Knowledge Base (BM25 + SQLite FTS5).
    Contains 29 foundational seed articles on AI/CS and selected biographies.
    """
    cleaned_q = clean_query_term(query)
    try:
        return search_index(cleaned_q, top_k=3)
    except Exception as e:
        return f"Errore ricerca Knowledge Base locale: {str(e)}"


def extract_wikipedia_plan(query: str) -> Tuple[List[str], List[str]]:
    """
    Extracts:
    1. `exact_titles`: High-precision entity titles for batched MediaWiki `titles=` lookup
       (resolves exact pages, alphanumeric codes, and standard Wikipedia media disambiguations).
    2. `search_queries`: Ranked full-text search queries for MediaWiki `generator=search`.
    """
    raw = sanitize_preserving_quotes(query)
    ql = raw.lower()
    exact_titles: List[str] = []
    search_queries: List[str] = []

    def _add_title(t: str):
        t = t.strip(" .,'\"?!")
        if len(t) >= 2 and t not in exact_titles:
            exact_titles.append(t)

    def _add_sq(sq: str):
        sq = " ".join(sq.split()).strip()
        if len(sq) >= 2 and sq not in search_queries:
            search_queries.append(sq)

    is_film_ctx = any(w in ql for w in ["film", "movie", "comedy", "drama", "director", "directed", "starring", "portrayed", "actress", "actor"])
    is_music_ctx = any(w in ql for w in ["band", "album", "boy group", "girl group", "record label", "musical"])
    is_book_ctx = any(w in ql for w in ["novel", "book", "series", "author", "written by"])

    # 1. Alphanumeric / slash codes (e.g. '2014 S/S')
    for code in re.findall(r"\b\d{4}\s+[A-Za-z0-9/._-]+\b", raw):
        _add_title(code)
        _add_sq(f'"{code}"')
        if is_music_ctx:
            _add_sq(f'"{code}" album')

    # 2. Explicitly quoted titles (avoid matching possessive apostrophes like men's)
    quoted = re.findall(r'"([^"]+)"', raw) + re.findall(r"(?<!\w)'([^']+)'(?!\w)", raw)
    for qp in quoted:
        qp_clean = qp.strip()
        if not qp_clean:
            continue
        if is_film_ctx:
            _add_title(f"{qp_clean} (film)")
            _add_sq(f'"{qp_clean}" film')
        if is_music_ctx:
            _add_title(f"{qp_clean} (album)")
            _add_title(f"{qp_clean} (band)")
            _add_sq(f'"{qp_clean}" band')
        if is_book_ctx:
            _add_title(f"{qp_clean} (novel)")
        _add_title(qp_clean)
        _add_sq(f'"{qp_clean}"')

    # 3. Media work titles introduced by 'film X', 'novel X', 'album X' (handles titles containing 'and'/'of'/'the')
    media_matches = re.findall(
        r"\b(?:film|movie|novel|book|album)\s+([A-ZÀ-ÖØ-Þ][\w'.-]*(?:\s+(?:and|of|the|in|to|for|a|an|[A-ZÀ-ÖØ-Þ][\w'.-]*))*)",
        raw,
    )
    trail_stop = {"and", "of", "the", "in", "to", "for", "a", "an", "is", "was", "were", "that", "which", "where", "who"}
    media_titles: List[str] = []
    for mm in media_matches:
        parts = mm.split()
        while parts and parts[-1].lower() in trail_stop:
            parts = parts[:-1]
        if parts:
            mt = " ".join(parts)
            media_titles.append(mt)
            if is_film_ctx:
                _add_title(f"{mt} (film)")
            _add_title(mt)

    # 4. Comparative entity pairs ('Were/Are X and Y ...', 'Who is older, X or Y?')
    m_comp = re.search(
        r"^(?:Are|Were|Was|Is)\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)\s+(?:both|located|of the same|from|in the same)",
        raw,
        flags=re.IGNORECASE,
    )
    if m_comp:
        for ent in (m_comp.group(1), m_comp.group(2)):
            _add_title(ent)
            _add_sq(ent)

    m_choice = re.search(r"^(?:Who|Which\b[^,]*)\s+is\s+\w+,\s*(.+?)\s+or\s+(.+?)\??$", raw, flags=re.IGNORECASE)
    if m_choice:
        for ent in (m_choice.group(1), m_choice.group(2)):
            _add_title(ent)
            _add_sq(ent)

    # 5. Contiguous Unicode proper-noun spans (after stripping ONLY sentence-initial question word)
    q_body = re.sub(
        r"^(?:What|Who|Where|When|Why|How|Are|Were|Was|Is|Did|Do|Does|Can|Could|Which)\b\s*(?:the\s+)?",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    lead_trim = {"The", "A", "An", "In", "On", "At", "By", "From", "To", "Of", "With", "Both", "What", "Who", "Where", "When", "Which"}
    geo_bg = {
        "united states", "new york", "new york city", "south korea", "south korean",
        "united kingdom", "american", "british", "french", "english"
    }
    cap_spans = re.findall(
        r"\b[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ0-9_'.-]*(?:\s+[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ0-9_'.-]*)*",
        q_body,
    )
    proper_entities: List[str] = []
    for span in cap_spans:
        words = span.split()
        while words and words[0] in lead_trim:
            words = words[1:]
        if not words:
            continue
        phrase = " ".join(words).strip(" .,'\"?!")
        if len(phrase) < 2 or phrase.lower() in geo_bg:
            continue
        if phrase not in proper_entities:
            proper_entities.append(phrase)

    # Combine media work + character/person if both present
    multiword_anchors = [t.lower() for t in (media_titles + quoted) if len(t.split()) >= 2]
    filtered_proper: List[str] = []
    for pe in proper_entities:
        # Skip single-word fragments that are merely sub-words of an already extracted multi-word title
        if len(pe.split()) == 1 and any(pe.lower() in mw.split() for mw in multiword_anchors):
            continue
        filtered_proper.append(pe)
    proper_entities = filtered_proper

    for mt in media_titles:
        other_ents = [pe for pe in proper_entities if pe not in mt and mt not in pe]
        if is_film_ctx and other_ents:
            _add_sq(f'"{mt}" film {other_ents[0]}')
        if is_film_ctx:
            _add_sq(f'"{mt}" film')

    # Check how many content tokens exist before deciding how to rank bare 1-word proper nouns
    stopwords = {
        "what", "who", "where", "when", "why", "how", "are", "were", "was", "is", "did", "do", "does",
        "the", "a", "an", "and", "or", "in", "on", "at", "by", "for", "from", "to", "of", "with",
        "that", "this", "which", "both", "same", "their", "they", "its", "her", "his", "has", "have",
        "had", "been", "being", "can", "could", "many", "people", "held", "woman", "man", "known"
    }
    tokens = re.findall(r"[\w/-]+", raw)
    content = [t for t in tokens if t.lower() not in stopwords and len(t) >= 2]

    for pe in proper_entities:
        # A bare single-word name inside a long descriptive query (e.g. 'Aladin' with 6+ other predicates)
        # is too ambiguous for exact-title lookup without its descriptive predicates
        if len(pe.split()) == 1 and len(content) >= 6:
            continue
        _add_title(pe)
        if is_music_ctx and len(pe.split()) <= 3:
            _add_title(f"{pe} (band)")
        _add_sq(pe)

    # 6. Distinctive content-word combinations (crucial for descriptive entity-identification questions)
    if content:
        filler = {"told", "set", "narrating", "stories", "played", "home", "games", "based", "formed", "helped", "improve"}
        high_idf = [t for t in content if t.lower() not in filler]
        if len(high_idf) >= 3:
            if not exact_titles:
                search_queries.insert(0, " ".join(high_idf[:8]))
            else:
                _add_sq(" ".join(high_idf[:8]))
        _add_sq(" ".join(content[:8]))

    _add_sq(clean_query_term(raw))
    return exact_titles[:6], search_queries[:4]


def _is_disambiguation(title: str, extract: str) -> bool:
    tl = title.lower()
    el = extract[:240].lower()
    return "(disambiguation)" in tl or "may refer to:" in el or "may also refer to:" in el


def search_wikipedia_api(query: str, max_results: int = 3) -> str:
    """
    Two-stage MediaWiki retrieval with domain-mismatch and weak-overlap filtering:
    Stage 1: Single batched `titles=` request for exact proper-noun/work titles (resolving redirects).
    Stage 2: `generator=search` full-text search when Stage 1 is empty or misses domain predicates.
    """
    if not query or len(query.strip()) < 2:
        return ""

    exact_titles, search_queries = extract_wikipedia_plan(query)
    ql = query.lower()

    raw_words = set(w.lower() for w in re.findall(r"\w{3,}", query))
    stopwords = {
        "what", "who", "where", "when", "why", "how", "are", "were", "was", "is", "did", "do", "does",
        "the", "and", "for", "from", "with", "that", "this", "which", "both", "same", "their", "they",
        "its", "his", "her", "has", "have", "had", "was", "been", "into", "born"
    }
    key_terms = raw_words - stopwords

    film_domain_words = {"film", "movie", "comedy", "director", "directed", "starring", "portrayed"}
    needs_film_domain = bool(film_domain_words & raw_words)

    accepted_pages: List[Tuple[str, str]] = []
    seen_titles = set()
    had_disambiguation = False

    # --- STAGE 1: Batched Exact Title Lookup (1 HTTP call, resolves redirects cleanly) ---
    if exact_titles:
        enc_titles = urllib.parse.quote("|".join(exact_titles))
        url_batch = (
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1"
            f"&redirects=1&titles={enc_titles}&format=json"
        )
        data_b = _wiki_fetch_json(url_batch)
        pages_b = data_b.get("query", {}).get("pages", {})
        for p in pages_b.values():
            if "missing" in p:
                continue
            title = p.get("title", "").strip()
            extract = p.get("extract", "").strip()
            if not title or not extract:
                continue
            if _is_disambiguation(title, extract):
                had_disambiguation = True
                continue
            comb_words = set(w.lower() for w in re.findall(r"\w{3,}", f"{title} {extract}"))
            overlap = key_terms & comb_words
            # Reject single-token accidental title collision on long descriptive queries (e.g. 'Aladin' -> 'Aladdin')
            if len(key_terms) >= 5 and len(overlap) < 2 and len(title.split()) == 1:
                continue
            if title.lower() not in seen_titles:
                seen_titles.add(title.lower())
                accepted_pages.append((title, extract))

    # Check if Stage 1 satisfied domain constraints (e.g. prevent town 'Big Stone Gap, Virginia' from blocking 'Big Stone Gap (film)')
    stage1_satisfies_domain = True
    if needs_film_domain and accepted_pages:
        combined_stage1 = " ".join(f"{t} {e}" for t, e in accepted_pages).lower()
        if not any(fw in combined_stage1 for fw in ("film", "movie", "directed by", "starring", "screenplay")):
            stage1_satisfies_domain = False
            # Demote non-film town/homonym matches so the actual film page ranks first
            accepted_pages = []
            seen_titles.clear()

    # --- STAGE 2: Full-Text Generator Search ---
    # Runs if Stage 1 found nothing, hit a disambiguation page (e.g. 'Kiss and Tell'), or missed domain predicates
    if not accepted_pages or not stage1_satisfies_domain or had_disambiguation:
        stage2_pages: List[Tuple[str, str]] = []
        for sq in search_queries[:3]:
            if len(stage2_pages) >= max_results:
                break
            enc_sq = urllib.parse.quote(sq)
            url_gen = (
                f"https://en.wikipedia.org/w/api.php?action=query&format=json&generator=search"
                f"&gsrsearch={enc_sq}&gsrlimit={max_results}&prop=extracts&exintro=1&explaintext=1&redirects=1"
            )
            data_g = _wiki_fetch_json(url_gen)
            pages_g = data_g.get("query", {}).get("pages", {})
            if not pages_g:
                continue
            sorted_pages = sorted(pages_g.values(), key=lambda item: item.get("index", 999))
            for p in sorted_pages:
                title = p.get("title", "").strip()
                extract = p.get("extract", "").strip()
                if not title or not extract or _is_disambiguation(title, extract):
                    continue
                if title.lower() in seen_titles:
                    continue
                comb_words = set(w.lower() for w in re.findall(r"\w{3,}", f"{title} {extract}"))
                overlap = key_terms & comb_words
                exact_multiword_match = any(
                    len(et.split()) >= 2 and et.lower() in title.lower()
                    for et in exact_titles
                )
                # For long multi-constraint queries (>=6 key terms), require >=3 overlapping key terms
                # (unless a multi-word proper title matched), preventing weak homonyms from blocking Web Search
                if len(key_terms) >= 6 and len(overlap) < 3 and not exact_multiword_match:
                    continue
                if len(key_terms) >= 4 and len(overlap) < 2 and not exact_multiword_match:
                    continue
                if not overlap and key_terms:
                    continue
                seen_titles.add(title.lower())
                stage2_pages.append((title, extract))
                if len(stage2_pages) >= max_results:
                    break
            if stage2_pages:
                break
        # Prioritize disambiguated Stage 2 results ahead of generic Stage 1 pages when disambiguation occurred
        if had_disambiguation and stage2_pages:
            accepted_pages = stage2_pages + accepted_pages
        else:
            accepted_pages.extend(stage2_pages)

    if not accepted_pages:
        return ""

    formatted = ""
    for title, extract in accepted_pages[:max_results]:
        formatted += f"\n--- Wikipedia Result: {title} ---\nTitle: {title}\nSummary: {extract[:850]}\n"
    return formatted.strip()


def _search_ddgs(cleaned_q: str, max_results: int = 3) -> str:
    """Queries DuckDuckGo text search (`ddgs` or `duckduckgo_search`) when available."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(cleaned_q, max_results=max_results))
            if results:
                formatted = ""
                for idx, r in enumerate(results):
                    title = r.get("title", "No Title")
                    href = r.get("href", "")
                    body = r.get("body", "")
                    formatted += f"\n--- Web Result {idx+1}: {title} ---\nURL: {href}\nSnippet: {body[:350]}\n"
                return formatted.strip()
    except Exception:
        pass
    return ""


@tool
def web_search(query: str) -> str:
    """
    Executes a hybrid live search across English Wikipedia API and DuckDuckGo Web Search.
    Preserves quotes and exact entity titles, falling back to or enriching with live Web snippets
    when Wikipedia has no match or only a short stub lead (<350 chars).
    """
    if RETRIEVAL_MODE == "baseline":
        print("      [BASELINE MODE ACTIVE] Ricerca Web esterna disabilitata: reindirizzamento forzato alla KB locale.")
        return local_kb_search.invoke({"query": query})

    raw_q = sanitize_preserving_quotes(query)
    cleaned_q = clean_query_term(query)
    if not cleaned_q or len(cleaned_q) < 2:
        return "Query non valida o vuota."

    # 1. Primary Encyclopedic Search on English Wikipedia API (passing raw_q with quotes intact)
    wiki_res = search_wikipedia_api(raw_q, max_results=3)

    # 2. Adaptive Hybrid Web Fallback / Enrichment:
    #    - If Wikipedia returned nothing (e.g. non-Wikipedia person/consultant or contemporary entity), run full DDGS search.
    #    - If Wikipedia returned a very short stub (<400 chars total), supplement with 2 DDGS snippets for richer detail.
    if not wiki_res:
        ddg_res = _search_ddgs(cleaned_q, max_results=4)
        if ddg_res:
            return ddg_res
    elif len(wiki_res) < 400:
        ddg_supp = _search_ddgs(cleaned_q, max_results=2)
        if ddg_supp:
            return f"{wiki_res}\n\n{ddg_supp}"
        return wiki_res
    else:
        return wiki_res

    # 3. Minimal HTTP Instant Answer Fallback
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(
            f"https://api.duckduckgo.com/?q={requests.utils.quote(cleaned_q)}&format=json&no_html=1&skip_disambig=1",
            headers=headers,
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get("AbstractText", "")
            heading = data.get("Heading", cleaned_q)
            if abstract:
                return f"--- Web Instant Answer: {heading} ---\n{abstract}\nSource: {data.get('AbstractURL', '')}"
    except Exception:
        pass

    return f"Nessun risultato trovato sul web per la query: '{cleaned_q}'."


# --- SCHEMI GBNF PIATTI PER GENERAZIONE JSON DETERMINISTICA ---

class AdaptiveStrategy(BaseModel):
    """Adaptive-RAG classification of execution strategy and optimal retrieval source."""
    reasoning: str = Field(
        description="Concise 1-sentence justification for strategy and knowledge source selection."
    )
    strategy: Literal["internal_knowledge", "single_retrieval", "multi_step"] = Field(
        description="Execution strategy: 'single_retrieval' for atomic single-entity lookup, 'multi_step' for multi-hop/bridge questions."
    )
    retrieval_source: Literal["local_kb", "web", "both"] = Field(
        description="'local_kb' ONLY for the 29 local seed topics (Scott Derrickson, Ed Wood, Doctor Strange, Alan Turing, Einstein, Newton, Rowling, Curie, AI/CS); 'web' for all other entities."
    )
    search_term: str = Field(
        description="Optimized search query preserving all exact named entities, quotes, and alphanumeric titles."
    )


class ReActStep(BaseModel):
    """Single structured step of the ReAct (Reasoning + Acting) investigative loop."""
    thought: str = Field(
        description="Concise 1-2 sentence chain-of-thought analyzing prior observations, identifying any bridge entity discovered, and deciding the next search or final answer."
    )
    action: Literal["local_kb_search", "web_search", "final_answer"] = Field(
        description="Choose 'web_search' or 'local_kb_search' to retrieve missing facts, or 'final_answer' ONLY when the retrieved observations explicitly answer every hop of the sub-query."
    )
    action_input: str = Field(
        description="If action is a search tool, provide the exact search query targeting the bridge entity discovered in the previous observation. If action is 'final_answer', state the concise factual finding."
    )


class SynthesisOutput(BaseModel):
    """Structured Self-RAG grounded final synthesis."""
    reasoning: str = Field(
        description="Concise 1-2 sentence verification of the exact retrieved facts for each entity/hop (e.g. comparing exact birth years where earlier year = older, comparing exact neighborhoods vs city, or citing the exact role/title/number)."
    )
    final_answer: str = Field(
        description="Concise canonical answer ONLY. For yes/no questions, start with 'yes' or 'no'. For Who/What/Where/Which/How many questions, output ONLY the exact entity name, title, role, location, or number (NEVER mention the unchosen candidate in 'X or Y' questions, and NEVER start with 'yes' or 'no')."
    )


def preserve_hop0_anchors(original_subquery: str, candidate_query: str) -> str:
    """
    Prevents LLM query reformulation on Hop 0 from dropping explicit anchors
    (quoted titles like "Big Stone Gap", codes like '2014 S/S', or proper nouns like 'For Against').
    """
    if not candidate_query or not isinstance(candidate_query, str) or len(candidate_query.strip()) < 2:
        return original_subquery

    cand_low = candidate_query.lower()

    # Check alphanumeric codes (e.g. 2014 S/S)
    codes = re.findall(r"\b\d{4}\s+[A-Za-z0-9/._-]+\b", original_subquery)
    for c in codes:
        if c.lower() not in cand_low:
            return original_subquery

    # Check quoted titles (keep quotes intact!)
    quoted = re.findall(r'"([^"]+)"', original_subquery)
    for q in quoted:
        if f'"{q.lower()}"' not in cand_low:
            return original_subquery

    # Check exact proper-noun titles extracted from original_subquery
    exact_titles, _ = extract_wikipedia_plan(original_subquery)
    base_titles = [t for t in exact_titles if "(" not in t]
    if base_titles and not any(bt.lower() in cand_low for bt in base_titles):
        return original_subquery

    # Preserve 'film' qualifier if present in original_subquery
    if "film" in original_subquery.lower() and "film" not in cand_low and "movie" not in cand_low:
        return f"{candidate_query} film"

    return candidate_query


def is_polar_question(query: str) -> bool:
    """
    Strictly identifies Yes/No polar questions while excluding Wh-questions
    (such as 'What...', 'Who...', '... formed by who?', '... based in what city?', '... how many people?').
    """
    ql = query.strip().lower()
    if re.search(r"\b(?:what|who|whom|whose|where|when|which|how)\b", ql):
        return False
    if re.match(r"^(?:were|was|is|are|did|do|does|can|could|has|have|had)\b", ql):
        # Exclude 'Is X A or B?' alternative choice unless 'same' or 'both' is present
        if " or " in ql and not any(w in ql for w in ("same", "both", "either")):
            return False
        return True
    return False


def normalize_final_answer(original_query: str, raw_answer: str, context: str = "") -> str:
    """
    Cleans formatting artifacts and prevents polarity/comparison false positives:
    - Strips leaked 'Reasoning:' sections or markdown bolding.
    - Removes spurious leading 'yes,'/'no,' on Wh-/entity questions.
    - Extracts only the winning subject entity on 'Who/Which is <adj>, X or Y?' questions.
    - Programmatically safeguards mathematical age comparison (earlier birth year = older).
    - Safeguards verbatim seated capacity extraction against convention/patron hallucinations.
    - Safeguards position/title extraction when asked for government position vs person name.
    """
    ans = (raw_answer or "").strip()
    if "Final Answer:" in ans:
        ans = ans.split("Final Answer:")[-1].strip()
    if "\nReasoning:" in ans:
        ans = ans.split("\nReasoning:")[0].strip()
    ans = re.sub(r"^\*+\s*|\s*\*+$", "", ans).strip()
    ans = ans.strip("\"'")

    polar = is_polar_question(original_query)
    if not polar:
        # Remove accidental 'yes, ' or 'no, ' prefix on non-polar questions
        ans = re.sub(r"^(?:yes|no)\s*[,:\-]\s*", "", ans, flags=re.IGNORECASE).strip()

        # On 'Who/Which is <comparative>, X or Y?' choice questions, if model wrote 'X is older than Y', keep only 'X'
        m_choice = re.search(
            r"^(?:who|which\b[^,]*)\s+is\s+(older|younger|taller|shorter|larger|smaller|bigger|earlier),\s*(.+?)\s+or\s+(.+?)\??$",
            original_query.strip(),
            flags=re.IGNORECASE,
        )
        if m_choice:
            comp_type = m_choice.group(1).lower()
            cand_a = m_choice.group(2).strip()
            cand_b = m_choice.group(3).strip()

            # Mathematical age comparison safeguard: earlier year (smaller number) is older
            if comp_type in ("older", "younger") and context:
                def _find_birth_year(entity_name, text):
                    toks = [t for t in re.findall(r"\w{3,}", entity_name) if t.lower() not in ("the", "and")]
                    for tok in toks:
                        for m in re.finditer(rf"{re.escape(tok)}[\s\S]{{0,160}}?(?:\bborn\b|\b[A-Za-z]+\s+\d{{1,2}},\s*)(\d{{4}})\b", text, flags=re.IGNORECASE):
                            y = int(m.group(1))
                            if 1800 <= y <= 2025:
                                return y
                    return None

                year_a = _find_birth_year(cand_a, context)
                year_b = _find_birth_year(cand_b, context)
                if year_a and year_b and year_a != year_b:
                    if comp_type == "older":
                        return cand_a if year_a < year_b else cand_b
                    else:
                        return cand_a if year_a > year_b else cand_b

            m_than = re.match(
                r"^(.+?)\s+(?:is|was)\s+(?:older|younger|taller|shorter|larger|smaller|bigger|earlier|more\s+\w+)\s+than\b",
                ans,
                flags=re.IGNORECASE,
            )
            if m_than:
                ans = m_than.group(1).strip(" .,")

        # Seating capacity safeguard: verify exact seated capacity (e.g. 'X seated' / 'X seats')
        if re.search(r"\bseat(?:\s+how\s+many\b|\b.*capacity)", original_query, flags=re.IGNORECASE) and context:
            m_seated = re.search(r"\(?(\b\d[\d,]+\s+(?:seated|seats))\)?", context, flags=re.IGNORECASE)
            if m_seated:
                seated_str = m_seated.group(1)
                num_only = re.sub(r"[^\d]", "", seated_str)
                if num_only and num_only not in ans.replace(",", ""):
                    ans = seated_str

        # If model answered with 'Person served as Title' or 'held the position of Title', extract the title
        if re.search(r"\bwhat\s+(?:government\s+)?(?:position|office|role|title)\b", original_query, flags=re.IGNORECASE):
            m_held = re.search(
                r"(?:served\s+as|held\s+the\s+(?:position|office|role)\s+of|appointed\s+(?:as)?)\s+([A-Z][a-zA-Z]*(?:\s+(?:of|the|for|and|[A-Z][a-zA-Z]*))*)",
                ans,
                flags=re.IGNORECASE,
            )
            if m_held:
                extracted = m_held.group(1).strip()
                if len(extracted) >= 3:
                    ans = extracted

    else:
        # Ensure polar answer starts cleanly with lowercase 'yes' or 'no'
        m_pol = re.match(r"^(yes|no)\b(.*)$", ans, flags=re.IGNORECASE | re.DOTALL)
        if m_pol:
            ans = f"{m_pol.group(1).lower()}{m_pol.group(2)}".strip()

    return ans


# --- PLANNER NODE ---
def planner_node(state: GraphState, llm) -> dict:
    """
    Executes Multi-Step Retrieval Planning (Adaptive-RAG + ReAct + IRCoT + Reflexion)
    with GBNF-constrained JSON schemas and two-stage Wikipedia/Web hybrid retrieval.
    """
    num_plan = state.get("num_planning", 0)
    print(f"\n--- [PLANNER] Inizio Esecuzione Piano Multi-Step (Ciclo {num_plan + 1}) ---")

    query_raw = state.get("current_query", state.get("original_query", ""))
    if isinstance(query_raw, list):
        queries = [str(q).strip() for q in query_raw if str(q).strip()]
    else:
        queries = [q.strip() for q in str(query_raw).split("\n---\n") if q.strip()]
    if not queries:
        queries = [state.get("original_query", "")]

    feedback_history = state.get("feedback_history", [])
    feedback_context = ""
    if feedback_history:
        feedback_context = (
            "\n\n[REFLEXION MEMORY BUFFER - CRITIQUE FROM PREVIOUS ATTEMPTS]\n"
            + "\n".join(f"- {f}" for f in feedback_history[-4:])
            + "\nAdjust your search queries to retrieve any missing entities!"
        )

    global_context = state.get("retrieved_context", "")
    MAX_STEPS_PER_QUERY = 4
    intermediate_answers: List[str] = []

    seed_entities = [
        "scott derrickson", "ed wood", "doctor strange", "alan turing", "albert einstein",
        "isaac newton", "rowling", "harry potter", "marie curie", "christopher nolan",
        "inception", "geoffrey hinton", "yann lecun", "artificial intelligence", "agentic ai",
        "retrieval-augmented generation", "information retrieval", "large language model", "transformer"
    ]

    # --- FASE A: RETRIEVAL ADATTIVO E MULTI-HOP (REACT + IRCOT) ---
    for i, query in enumerate(queries):
        print(f"\n[PLANNER] Step {i+1}/{len(queries)}: Risoluzione della query -> '{query}'")
        step_context = ""
        has_seed = any(se in query.lower() for se in seed_entities)

        # 1. ADAPTIVE-RAG: Scelta strategia e sorgente con schema GBNF
        print("[PLANNER] [Adaptive-RAG] Valutazione della strategia e della fonte (Local KB vs Web)...")
        strategy_system = """You are a Strategic Retrieval Planner implementing Adaptive-RAG (Jeong et al., 2024).
Classify the sub-query to choose the execution strategy and retrieval source:
- 'local_kb': ONLY for questions directly about the 29 local seed topics (Scott Derrickson, Ed Wood, Doctor Strange, Alan Turing, Einstein, Newton, Rowling, Curie, Nolan, AI/CS).
- 'web': For ALL other people, bands, films, albums, books, venues, buildings, or external facts.
- 'single_retrieval': ONLY for simple single-entity atomic questions (e.g. 'When was X born?', 'What is the nationality of X?').
- 'multi_step': For bridge or multi-hop questions where an intermediate entity (e.g. director, actor, arena, group) must be discovered first.

Preserve all exact entity names, quotes, and codes in search_term. Return valid AdaptiveStrategy JSON."""

        strategy_user = f"""Sub-Query: '{query}'
Prior Context:
{global_context[-1200:] if global_context else "No prior context."}
{feedback_context}

Return your AdaptiveStrategy JSON."""

        strategy = "single_retrieval" if len(queries) > 1 else "multi_step"
        retrieval_source = "local_kb" if has_seed else "web"
        search_term = query

        try:
            strat_obj, _ = invoke_structured_json(
                llm,
                AdaptiveStrategy,
                [
                    SystemMessage(content=strategy_system),
                    HumanMessage(content=strategy_user),
                ],
            )
            if strat_obj is not None:
                if isinstance(strat_obj.strategy, str):
                    strategy = strat_obj.strategy
                if isinstance(strat_obj.retrieval_source, str):
                    src = strat_obj.retrieval_source
                    retrieval_source = "web" if (src == "local_kb" and not has_seed) else src
                if isinstance(strat_obj.search_term, str) and strat_obj.search_term.strip():
                    search_term = strat_obj.search_term.strip()
                print(f"   [Adaptive-RAG Reasoning]: {strat_obj.reasoning[:140]}")
        except Exception as e:
            print(f"   [PLANNER WARNING] Fallback Adaptive-RAG: {e}")

        # Regole strutturali Adaptive-RAG:
        # - Domande già decomposte dal Refiner in sotto-query atomiche (len > 1) usano single_retrieval simmetrico e veloce.
        # - Domande singole non decomposte (len == 1) usano multi_step per consentire il salto bridge Hop 1 -> Hop 2.
        if len(queries) > 1 and num_plan == 0:
            strategy = "single_retrieval"
        elif len(queries) == 1 and strategy != "multi_step":
            strategy = "multi_step"

        effective_hop0_query = query if num_plan == 0 else preserve_hop0_anchors(query, search_term)
        current_mode = state.get("retrieval_mode", RETRIEVAL_MODE).lower()
        effective_mode = "baseline" if current_mode in ("baseline", "local", "local_only", "kb_only") else "full"
        source_display = "LOCAL_KB (BASELINE)" if effective_mode == "baseline" else retrieval_source.upper()
        print(f"   [PLANNER] Modalità: {effective_mode.upper()} | Strategia: {strategy.upper()} | Fonte: {source_display} | Term: '{effective_hop0_query}'")

        try:
            if strategy == "single_retrieval":
                print(f"   [PLANNER] Esecuzione single_retrieval [{source_display}] per -> '{effective_hop0_query}'")
                tool_result = ""
                if effective_mode == "baseline":
                    print("      [BASELINE ACTIVE] Retrieval vincolato esclusivamente alla Knowledge Base locale.")
                    tool_result = local_kb_search.invoke({"query": effective_hop0_query})
                else:
                    if retrieval_source in ("local_kb", "both"):
                        tool_result = local_kb_search.invoke({"query": effective_hop0_query})
                    if retrieval_source in ("web", "both") or "Nessun risultato rilevante" in tool_result:
                        if tool_result and "Nessun risultato rilevante" in tool_result:
                            print("      [PLANNER] Nessun match locale. Fallback automatico su Web Search...")
                            tool_result = ""
                        web_res = web_search.invoke({"query": effective_hop0_query})
                        tool_result = f"{tool_result}\n{web_res}".strip() if tool_result else web_res

                tool_result = tool_result[:2400]
                preview = tool_result[:160].replace("\n", " ") + "..."
                print(f"      [PLANNER] Retrieved: {preview}")
                step_context = f"Search Result ({query}):\n{tool_result}\n"

                if tool_result and "Nessun risultato" not in tool_result:
                    summary_lines = [
                        l.strip() for l in tool_result.split("\n")
                        if l.strip() and not l.strip().startswith(("---", "Title:", "Category:", "URL:"))
                    ]
                    if summary_lines:
                        joined_sum = " ".join(summary_lines)[:450]
                        intermediate_answers.append(f"Sub-query: '{query}' -> Evidence: {joined_sum}")

            else:
                # 2. MULTI-STEP RETRIEVAL (Structured ReAct + IRCoT Loop)
                print(f"   [PLANNER] Avvio ciclo ReAct Multi-Step per: '{query}'...")
                react_system = f"""You are an investigative multi-hop retrieval agent implementing ReAct (Yao et al., 2022) and IRCoT (Trivedi et al., 2022).

AVAILABLE ACTIONS in `action`:
- 'local_kb_search': ONLY for the 29 local seed topics (Scott Derrickson, Ed Wood, Doctor Strange, Alan Turing, AI/CS).
- 'web_search': Searches full English Wikipedia and live Web. Use for ALL films, bands, albums, books, venues, people, and external facts.
- 'final_answer': Choose ONLY AFTER you have retrieved explicit evidence answering EVERY hop of the sub-query!

MULTI-HOP BRIDGE RULE:
- Hop 1: Search for the named work, organization, or entity mentioned in the question to discover the intermediate bridge entity (e.g. the director, author, founder, or venue).
- Hop 2 (BRIDGE ENTITY EXTRACTION): When Hop 1 reveals who directed, wrote, starred in, or created something (e.g. 'directed by Director A' or 'starring Actor B'):
  In Hop 2, your action MUST be 'web_search' and action_input MUST specifically target that discovered person or entity and the requested attribute!
  Example: if Hop 1 found 'directed by Director A', Hop 2 action_input MUST search for: 'Director A city of residence' or 'Director A based in'.
  If Hop 1 found 'starring Actor B', Hop 2 action_input MUST search for: 'Actor B public office' or 'Actor B title'.

EARLY STOPPING RULE (CRITICAL):
- If an Observation ALREADY explicitly answers the final question (e.g. explicitly naming the record label, founding organization, or exact capacity requested):
  STOP IMMEDIATELY! Set action='final_answer' and state the exact answer in action_input.
  DO NOT issue unnecessary redundant searches that introduce irrelevant distractor entities!

PRIOR CONTEXT:
{global_context[-1000:] if global_context else "No prior context."}
{feedback_context}

Return a single JSON object conforming strictly to the ReActStep schema."""

                messages = [
                    SystemMessage(content=react_system),
                    HumanMessage(content=f"Sub-Query to solve: '{query}'"),
                ]
                executed_queries: List[str] = []

                for attempt in range(MAX_STEPS_PER_QUERY):
                    step_obj, raw_react = invoke_structured_json(llm, ReActStep, messages)

                    if step_obj is None:
                        # Fallback automatico se il modello non produce JSON valido
                        if attempt == 0:
                            step_obj = ReActStep(
                                thought="Executing initial retrieval for sub-query anchors.",
                                action="local_kb_search" if has_seed else "web_search",
                                action_input=effective_hop0_query,
                            )
                        else:
                            break

                    print(f"      [ReAct Thought (Hop {attempt+1})]: {step_obj.thought[:180]}...")

                    # Impedisci final_answer al turno 0 prima di aver recuperato evidenze!
                    if step_obj.action == "final_answer" and attempt == 0 and not step_context:
                        step_obj = ReActStep(
                            thought=step_obj.thought,
                            action="local_kb_search" if has_seed else "web_search",
                            action_input=effective_hop0_query,
                        )

                    if step_obj.action == "final_answer":
                        ans_text = step_obj.action_input.strip() or step_obj.thought.strip()
                        print(f"      [ReAct Final Answer]: {ans_text[:160]}")
                        intermediate_answers.append(f"Sub-query: '{query}' -> Finding: {ans_text}")
                        break

                    # Determina la query di ricerca:
                    # - Al primo tentativo (Hop 0 del Ciclo 1), usa la query fedele per non perdere predicati o inserire rumore.
                    # - Dai salti successivi (Hop 2+), usa la query bridge formulata dall'agente sull'entità scoperta nell'Observation!
                    raw_s_query = step_obj.action_input.strip() if isinstance(step_obj.action_input, str) else ""
                    if attempt == 0 and num_plan == 0:
                        s_query = effective_hop0_query
                    elif attempt == 0:
                        s_query = preserve_hop0_anchors(query, raw_s_query or effective_hop0_query)
                    else:
                        s_query = raw_s_query if len(raw_s_query) >= 2 else effective_hop0_query

                    # Evita ripetizione identica della query precedente
                    if s_query.lower() in [eq.lower() for eq in executed_queries] and attempt > 0:
                        print("      [ReAct Dedup]: Query già eseguita; concludo raccolta per questa sotto-query.")
                        break
                    executed_queries.append(s_query)

                    if effective_mode == "baseline":
                        print(f"      [ReAct Action (Baseline Mode)]: local_kb_search vincolato -> '{s_query}'")
                        t_res = local_kb_search.invoke({"query": s_query})
                    elif step_obj.action == "local_kb_search" and has_seed:
                        print(f"      [ReAct Action]: local_kb_search -> '{s_query}'")
                        t_res = local_kb_search.invoke({"query": s_query})
                        if "Nessun risultato rilevante" in t_res:
                            print("         [PLANNER] Nessun match locale. Fallback automatico su web_search...")
                            t_res = web_search.invoke({"query": s_query})
                    else:
                        print(f"      [ReAct Action]: web_search -> '{s_query}'")
                        t_res = web_search.invoke({"query": s_query})

                    # Cap ampia (2400 char) per preservare tutti i 3 risultati Wikipedia senza troncamento prematuro
                    t_res = t_res[:2400]
                    preview = t_res[:160].replace("\n", " ") + "..."
                    print(f"         [Observation]: {preview}")

                    step_context += f"Search Result ({s_query}):\n{t_res}\n"
                    obs_prompt = (
                        f"Observation from search ('{s_query}'):\n{t_res}\n\n"
                        f"Carefully check if this Observation already answers '{query}', OR if it reveals an intermediate bridge entity "
                        f"(such as a person, actress, director, band, or arena name) that requires a Hop {attempt+2} 'web_search' to find the final answer. "
                        f"Issue your next ReActStep JSON."
                    )
                    # Registra prima la risposta assistant per rispettare rigorosamente l'alternanza user/assistant/user dei template Jinja (Mistral)
                    assistant_json = raw_react.strip() if (raw_react and raw_react.strip()) else step_obj.model_dump_json()
                    messages.append(AIMessage(content=assistant_json))
                    messages.append(HumanMessage(content=obs_prompt))
                    if len(messages) > 8:
                        messages = [messages[0], messages[1]] + messages[-4:]
                else:
                    # Se il loop ha eseguito tutti gli step di ricerca, chiedi una conclusione sintetica o registra evidenza
                    if step_context:
                        intermediate_answers.append(f"Sub-query: '{query}' -> Retrieved multi-hop evidence ready for synthesis.")

        except Exception as e:
            print(f"   [PLANNER ERROR] Errore durante l'esecuzione dello step: {e}")

        if step_context:
            global_context = f"{global_context}\n{step_context}".strip()

    # --- FASE B: SINTESI FINALE CON SCHEMA STRUTTURATO GBNF ---
    print("\n[PLANNER] Sintesi Finale rigorosa (Self-RAG Grounding)...")

    original_query = state.get("original_query", " ".join(queries))
    polar = is_polar_question(original_query)
    clean_context = global_context[-3400:].strip() if global_context else "No context retrieved."

    findings_summary = ""
    if intermediate_answers:
        findings_summary = "\nINTERMEDIATE FINDINGS:\n" + "\n".join(intermediate_answers) + "\n"

    if polar:
        answer_format_rule = (
            "This is a YES/NO comparative question. In `reasoning`, explicitly state the exact requested attribute found for Entity 1 "
            "and the exact requested attribute found for Entity 2 separately. "
            "Pay strict attention to spatial/categorical granularity: if the question asks about 'same neighborhood' or 'same province', sharing the same broad region or metropolitan city "
            "does NOT mean same neighborhood (e.g. District Alpha vs District Beta in City Gamma -> answer 'no')! "
            "Then in `final_answer`, start with 'yes,' or 'no,' (lowercase) followed by a concise 1-sentence factual justification."
        )
    else:
        answer_format_rule = (
            "This is a SPECIFIC ENTITY / ROLE / NUMBER / CHOICE question (NOT a yes/no question!). "
            "NEVER start `final_answer` with 'yes' or 'no'! "
            "In `reasoning`, verify the exact entity, title, role, location, company, or number from the evidence:\n"
            "- TARGET ROLE/POSITION: If the question asks for a position, office, or role ('What position was held by...'), output ONLY the official title or position (e.g. 'Minister of Foreign Affairs', 'Ambassador', 'Director'), NEVER the person's name.\n"
            "- AGE COMPARISON: For 'Who is older, X or Y?', compare exact birth years. The person born in the EARLIER year (smaller number, e.g. 1945 < 1958) was born first and is OLDER. In `final_answer`, output ONLY the older person's name.\n"
            "- SEATED CAPACITY: For venue seating questions, copy the exact seated capacity span from the evidence (e.g. '15,000 seated' or '12,500 seats'), NOT total standing or festival capacity.\n"
            "- LOCATION / RESIDENCE: Output the specific neighborhood/city span stated in the person's biography.\n"
            "- ORGANIZATION / COMPANY: Output the exact organization, record label, or parent company identified in the evidence.\n"
            "In `final_answer`, output ONLY the concise canonical answer span. Do NOT write a full comparative sentence!"
        )

    synthesis_system = f"""You are the Self-RAG Grounded Final Synthesis Engine.
Answer the user's question relying EXCLUSIVELY on the provided RETRIEVED EVIDENCE.

CRITICAL FORMATTING & GROUNDING RULES:
1. {answer_format_rule}
2. Rely strictly on facts present in RETRIEVED EVIDENCE. Do not invent dates or facts.
3. Return a single JSON object conforming strictly to the SynthesisOutput schema."""

    synthesis_user = f"""RETRIEVED EVIDENCE:
{clean_context}
{findings_summary}

QUESTION: {original_query}

Issue your SynthesisOutput JSON:"""

    final_answer = ""
    try:
        syn_obj, syn_raw = invoke_structured_json(
            llm,
            SynthesisOutput,
            [
                SystemMessage(content=synthesis_system),
                HumanMessage(content=synthesis_user),
            ],
        )
        if syn_obj is not None and isinstance(syn_obj.final_answer, str) and syn_obj.final_answer.strip():
            print(f"   [Synthesis Reasoning]: {syn_obj.reasoning[:180]}...")
            final_answer = syn_obj.final_answer.strip()
        elif syn_raw:
            final_answer = syn_raw.strip()
    except Exception as e:
        print(f"[PLANNER ERROR] Errore sintesi: {e}")
        final_answer = ""

    final_answer = normalize_final_answer(original_query, final_answer, context=clean_context)

    if not final_answer:
        final_answer = "Unable to determine answer from retrieved evidence."

    print(f"[PLANNER] Proposta Final Answer: {final_answer[:160]}")

    return {
        "retrieved_context": global_context,
        "final_answer": final_answer,
        "num_planning": num_plan + 1,
    }
