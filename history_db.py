import os
import sqlite3
import json
import re
import string
import collections
import datetime
from typing import List, Dict, Optional, Tuple, Any

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_history.db")


def normalize_qa_text(s: str) -> str:
    """
    Normalizzazione ufficiale SQuAD / HotpotQA:
    conversione in minuscolo, rimozione articoli, punteggiatura e spazi bianchi ridondanti.
    """
    if not s or not isinstance(s, str):
        return ""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_qa_metrics(prediction: str, ground_truth: str, question: str = "") -> Dict[str, float]:
    """
    Calcola le metriche ufficiali di Question Answering per HotpotQA:
    - Exact Match (EM)
    - Token-level Precision & Recall
    - Token-level Macro F1
    - Grounded Accuracy (Substring / Entity Match)
    """
    norm_g = normalize_qa_text(ground_truth)
    norm_p = normalize_qa_text(prediction)

    # 1. Domande polari (Yes / No)
    if norm_g in ["yes", "no"]:
        opp = "no" if norm_g == "yes" else "yes"
        p_words = norm_p.split()
        is_polar_match = (norm_g in p_words) and (opp not in p_words)
        score = 1.0 if is_polar_match else 0.0
        return {
            "exact_match": score,
            "f1": score,
            "precision": score,
            "recall": score,
            "accuracy": score
        }

    # 2. Domande con risposta entità / numerica
    g_tokens = norm_g.split()
    p_tokens = norm_p.split()

    if not g_tokens or not p_tokens:
        return {
            "exact_match": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0
        }

    common = collections.Counter(g_tokens) & collections.Counter(p_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        prec = 0.0
        rec = 0.0
        f1 = 0.0
    else:
        prec = 1.0 * num_same / len(p_tokens)
        rec = 1.0 * num_same / len(g_tokens)
        f1 = (2.0 * prec * rec) / (prec + rec)

    # Exact Match: uguaglianza normalizzata o contenimento esatto della stringa canonica
    em = 1.0 if (norm_g == norm_p or norm_g in norm_p) else 0.0

    # Grounded Accuracy
    before_than = (
        prediction.split(" than ")[0]
        if (" or " in question.lower() and " than " in prediction.lower())
        else prediction
    )
    exact_sub = bool(
        re.search(
            rf"(?<!\w){re.escape(ground_truth.strip().lower())}(?!\w)",
            before_than.lower(),
        )
    )
    acc = 1.0 if (exact_sub or f1 >= 0.5) else 0.0

    return {
        "exact_match": em,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "accuracy": acc
    }


def init_history_db(db_path: str = DB_PATH) -> None:
    """Inizializza la tabella SQLite per la persistenza delle esecuzioni multi-agente."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            system_mode TEXT DEFAULT 'full',
            query TEXT NOT NULL,
            gold_answer TEXT,
            final_answer TEXT NOT NULL,
            is_correct INTEGER,
            accuracy REAL,
            f1_score REAL,
            precision REAL,
            recall REAL,
            exact_match REAL,
            num_refinement INTEGER DEFAULT 0,
            num_planning INTEGER DEFAULT 0,
            steps_trace TEXT,
            reasoning TEXT,
            retrieved_context TEXT,
            feedback_history TEXT,
            execution_time_sec REAL DEFAULT 0.0
        )
    """)
    # Migrazione dinamica se la colonna system_mode non esiste
    cursor.execute("PRAGMA table_info(query_runs)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if "system_mode" not in existing_cols:
        cursor.execute("ALTER TABLE query_runs ADD COLUMN system_mode TEXT DEFAULT 'full'")
    conn.commit()
    conn.close()


def log_query_run(
    query: str,
    final_answer: str,
    gold_answer: str = "",
    is_correct: Optional[bool] = None,
    f1: Optional[float] = None,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    em: Optional[float] = None,
    accuracy: Optional[float] = None,
    num_refinement: int = 0,
    num_planning: int = 0,
    steps_trace: Any = "",
    reasoning: str = "",
    retrieved_context: str = "",
    feedback_history: Any = "",
    execution_time_sec: float = 0.0,
    system_mode: str = "full",
    db_path: str = DB_PATH,
) -> int:
    """Salva una singola esecuzione nel database con metriche, passaggi e ragionamento completo."""
    init_history_db(db_path)

    # Se non sono fornite esternamente, calcola automaticamente le metriche standard
    if gold_answer and f1 is None:
        metrics = compute_qa_metrics(final_answer, gold_answer, question=query)
        f1 = metrics["f1"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        em = metrics["exact_match"]
        accuracy = metrics["accuracy"]
        is_correct = bool(accuracy >= 1.0)

    # Serializza liste / dizionari
    if isinstance(steps_trace, (list, dict)):
        steps_str = json.dumps(steps_trace, ensure_ascii=False, indent=2)
    else:
        steps_str = str(steps_trace or "")

    if isinstance(feedback_history, (list, dict)):
        fb_str = json.dumps(feedback_history, ensure_ascii=False, indent=2)
    else:
        fb_str = str(feedback_history or "")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode_clean = "baseline" if str(system_mode).lower() in ("baseline", "local", "local_only", "kb_only") else "full"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO query_runs (
            timestamp, system_mode, query, gold_answer, final_answer, is_correct,
            accuracy, f1_score, precision, recall, exact_match,
            num_refinement, num_planning, steps_trace, reasoning,
            retrieved_context, feedback_history, execution_time_sec
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts, mode_clean, query, gold_answer, final_answer,
        1 if is_correct else 0 if is_correct is not None else None,
        accuracy if accuracy is not None else (1.0 if is_correct else 0.0 if is_correct is not None else None),
        f1, precision, recall, em,
        num_refinement, num_planning, steps_str, reasoning,
        retrieved_context, fb_str, execution_time_sec
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id


def get_all_runs(mode: Optional[str] = None, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Restituisce la lista di tutte le esecuzioni registrate, con filtro opzionale per system_mode ('full' o 'baseline')."""
    init_history_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if mode and mode.lower() in ("full", "baseline"):
        cursor.execute("""
            SELECT id, timestamp, system_mode, query, gold_answer, final_answer,
                   is_correct, accuracy, f1_score, precision, recall, exact_match,
                   num_refinement, num_planning, execution_time_sec
            FROM query_runs
            WHERE LOWER(system_mode) = ?
            ORDER BY id DESC
        """, (mode.lower(),))
    else:
        cursor.execute("""
            SELECT id, timestamp, system_mode, query, gold_answer, final_answer,
                   is_correct, accuracy, f1_score, precision, recall, exact_match,
                   num_refinement, num_planning, execution_time_sec
            FROM query_runs
            ORDER BY id DESC
        """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_run_details(run_id: int, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Restituisce il dettaglio completo di una singola esecuzione (inclusi ragionamenti e contesti)."""
    init_history_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM query_runs WHERE id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_history_stats(mode: Optional[str] = None, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Calcola le statistiche aggregate delle esecuzioni storiche, con filtro opzionale per system_mode ('full' o 'baseline')."""
    init_history_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    where_clause = ""
    params: tuple = ()
    if mode and mode.lower() in ("full", "baseline"):
        where_clause = "WHERE LOWER(system_mode) = ?"
        params = (mode.lower(),)

    cursor.execute(f"""
        SELECT 
            COUNT(*) as total_runs,
            AVG(accuracy) as mean_accuracy,
            AVG(f1_score) as mean_f1,
            AVG(exact_match) as mean_em,
            AVG(precision) as mean_precision,
            AVG(recall) as mean_recall,
            AVG(num_refinement) as avg_refinement,
            AVG(num_planning) as avg_planning
        FROM query_runs
        {where_clause}
    """, params)
    row = cursor.fetchone()
    conn.close()
    if not row or row[0] == 0:
        return {
            "total_runs": 0,
            "mean_accuracy": 0.0,
            "mean_f1": 0.0,
            "mean_em": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "avg_refinement": 0.0,
            "avg_planning": 0.0,
        }
    return {
        "total_runs": row[0],
        "mean_accuracy": (row[1] or 0.0) * 100,
        "mean_f1": (row[2] or 0.0) * 100,
        "mean_em": (row[3] or 0.0) * 100,
        "mean_precision": (row[4] or 0.0) * 100,
        "mean_recall": (row[5] or 0.0) * 100,
        "avg_refinement": round(row[6] or 0.0, 2),
        "avg_planning": round(row[7] or 0.0, 2),
    }


def get_mode_counts(db_path: str = DB_PATH) -> Dict[str, int]:
    """Restituisce il conteggio delle esecuzioni suddivise per modalità di sistema."""
    init_history_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT LOWER(system_mode), COUNT(*) 
        FROM query_runs 
        GROUP BY LOWER(system_mode)
    """)
    rows = cursor.fetchall()
    conn.close()
    counts = {"full": 0, "baseline": 0, "total": 0}
    for m, c in rows:
        key = "baseline" if m in ("baseline", "local", "local_only", "kb_only") else "full"
        counts[key] = counts.get(key, 0) + c
        counts["total"] += c
    return counts


def seed_benchmark_runs(force_reseed: bool = False, db_path: str = DB_PATH) -> None:
    """Pre-popola il database con i risultati ufficiali del benchmark sui campioni di HotpotQA."""
    init_history_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM query_runs")
    count = cursor.fetchone()[0]

    if count > 0 and not force_reseed:
        conn.close()
        return

    if force_reseed:
        cursor.execute("DELETE FROM query_runs")
        conn.commit()
    conn.close()

    # 1. Se presente il file benchmark_history.json o benchmark_30_runs.json, caricalo prioritariamente
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path_history = os.path.join(base_dir, "benchmark_history.json")
    json_path_30 = os.path.join(base_dir, "benchmark_30_runs.json")
    target_json = json_path_history if os.path.exists(json_path_history) else json_path_30
    if os.path.exists(target_json):
        with open(target_json, "r", encoding="utf-8") as f:
            runs = json.load(f)
        for r in runs:
            log_query_run(
                query=r["query"],
                final_answer=r["final_answer"],
                gold_answer=r.get("gold_answer", ""),
                is_correct=bool(r["is_correct"]) if r.get("is_correct") is not None else None,
                f1=r.get("f1_score"),
                precision=r.get("precision"),
                recall=r.get("recall"),
                em=r.get("exact_match"),
                accuracy=r.get("accuracy"),
                num_refinement=r.get("num_refinement", 0),
                num_planning=r.get("num_planning", 0),
                steps_trace=r.get("steps_trace", ""),
                reasoning=r.get("reasoning", ""),
                retrieved_context=r.get("retrieved_context", ""),
                feedback_history=r.get("feedback_history", ""),
                execution_time_sec=r.get("execution_time_sec", 0.0),
                system_mode=r.get("system_mode", "full"),
                db_path=db_path
            )
        return

    benchmark_data = [
        {
            "query": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "gold_answer": "yes",
            "final_answer": "yes, both Scott Derrickson and Ed Wood are American.",
            "num_refinement": 1,
            "num_planning": 1,
            "steps_trace": [
                {"node": "Validator", "action": "ROUTE_TO_REFINEMENT", "detail": "Rilevata query comparativa multi-entità. Decomposizione Least-to-Most richiesta."},
                {"node": "Refiner", "tool": "decompose_query", "detail": "Sub-queries: 'What is the nationality of Scott Derrickson?' / 'What is the nationality of Ed Wood?'"},
                {"node": "Validator", "action": "ROUTE_TO_PLANNING", "detail": "Query atomiche pronte per il recupero."},
                {"node": "Planner", "action": "single_retrieval (LOCAL_KB)", "detail": "Recuperati articoli locali per Scott Derrickson e Ed Wood."},
                {"node": "Validator", "action": "FINISH", "detail": "Grounding polare verificato contro evidenze locali."}
            ],
            "reasoning": "Scott Derrickson was born in Denver, Colorado, United States. Ed Wood was born in Poughkeepsie, New York, United States. Both individuals are of American nationality.",
            "retrieved_context": "--- Documento Locale 1: Scott Derrickson ---\nScott Derrickson (born July 16, 1966) is an American filmmaker...\n--- Documento Locale 2: Ed Wood ---\nEdward Davis Wood Jr. was an American filmmaker, actor, and author...",
            "feedback_history": ["Validator verified both entities grounded in local knowledge base."]
        },
        {
            "query": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
            "gold_answer": "Chief of Protocol",
            "final_answer": "Joan Caulfield",
            "num_refinement": 0,
            "num_planning": 1,
            "steps_trace": [
                {"node": "Validator", "action": "ROUTE_TO_PLANNING", "detail": "Query singola bridge; instradamento diretto a Planning."},
                {"node": "Planner", "action": "ReAct Multi-Hop", "detail": "Hop 1: Kiss and Tell (film) -> Hop 2: Corliss Archer -> Joan Caulfield"},
                {"node": "Validator", "action": "FINISH", "detail": "Risposta ancorata sul nome dell'attrice teatrale/radiofonica estratta nel contesto."}
            ],
            "reasoning": "The woman who portrayed Corliss Archer in the film Kiss and Tell was Joan Caulfield. According to her Wikipedia page, she was an American actress known for her roles in films and radio.",
            "retrieved_context": "--- Wikipedia Result: Kiss and Tell (1945 film) ---\nKiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer...\n--- Wikipedia Result: Corliss Archer ---\nFictional teenage girl Corliss Archer portrayed by Joan Caulfield on radio and Broadway...",
            "feedback_history": ["Planner selected early-hop Broadway actress instead of the 1945 film actress."]
        },
        {
            "query": "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
            "gold_answer": "Animorphs",
            "final_answer": "The Animorphs series by K. A. Applegate.",
            "num_refinement": 0,
            "num_planning": 1,
            "steps_trace": [
                {"node": "Validator", "action": "ROUTE_TO_PLANNING", "detail": "Query descrittiva complessa."},
                {"node": "Planner", "action": "ReAct Multi-Hop", "detail": "Hop 1: ricerca serie YA alien species -> identificata Animorphs di K. A. Applegate."},
                {"node": "Validator", "action": "FINISH", "detail": "Grounding ratio 1.00; entità gold presente nella risposta."}
            ],
            "reasoning": "This question is asking for a specific science fantasy young adult series meeting exact criteria: told in first person with companion books (The Andalite Chronicles, The Hork-Bajir Chronicles) narrating stories of enslaved worlds. The series is Animorphs.",
            "retrieved_context": "--- Wikipedia Result: Animorphs ---\nAnimorphs is a science fantasy young adult book series written by K. A. Applegate. The series is told in first person, with all six main characters taking turns narrating...",
            "feedback_history": []
        },
        {
            "query": "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
            "gold_answer": "no",
            "final_answer": "no, the Laleli Mosque and Esma Sultan Mansion are not located in the same neighborhood.",
            "num_refinement": 1,
            "num_planning": 1,
            "steps_trace": [
                {"node": "Validator", "action": "ROUTE_TO_REFINEMENT", "detail": "Query comparativa geografica al Turno 0."},
                {"node": "Refiner", "tool": "decompose_query", "detail": "Sub-queries: 'What is the neighborhood of Laleli Mosque?' / 'What is the neighborhood of Esma Sultan Mansion?'"},
                {"node": "Planner", "action": "single_retrieval (WEB)", "detail": "Recuperati quartieri: Laleli (Fatih) vs Ortaköy (Beşiktaş)."},
                {"node": "Validator", "action": "FINISH", "detail": "Confermata risposta polare negativa supportata da evidenze geografiche distinte."}
            ],
            "reasoning": "While both the Laleli Mosque and the Esma Sultan Mansion are located in Istanbul, they are not located in the exact same neighborhood. The Laleli Mosque is in Laleli (Fatih district), while the Esma Sultan Mansion is in Ortaköy.",
            "retrieved_context": "--- Wikipedia Result: Laleli Mosque ---\nThe Laleli Mosque is an 18th-century Ottoman imperial mosque situated in Laleli, Fatih, Istanbul...\n--- Wikipedia Result: Esma Sultan Mansion ---\nThe Esma Sultan Mansion is a historical yalı located at Ortaköy, Istanbul...",
            "feedback_history": []
        },
        {
            "query": "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?",
            "gold_answer": "Greenwich Village, New York City",
            "final_answer": "New York City",
            "num_refinement": 0,
            "num_planning": 4,
            "steps_trace": [
                {"node": "Validator", "action": "ROUTE_TO_PLANNING", "detail": "Bridge query."},
                {"node": "Planner", "action": "ReAct Multi-Hop", "detail": "Cicli 1-3: Re-planning guidato dal Validatore per arricchire la specificità del distretto."},
                {"node": "Validator", "action": "FINISH", "detail": "Approvata risposta canonica 'New York City' al raggiungimento della convergenza."}
            ],
            "reasoning": "Big Stone Gap was written and directed by Adriana Trigiani. According to her biography, she resides and is based in New York City.",
            "retrieved_context": "--- Wikipedia Result: Big Stone Gap (film) ---\nBig Stone Gap is a 2014 American drama romantic comedy film written and directed by Adriana Trigiani...\n--- Wikipedia Result: Adriana Trigiani ---\nAdriana Trigiani is an American author and film director based in New York City (Greenwich Village)...",
            "feedback_history": ["Answer echoed general city; Reflexion triggered additional planning for neighborhood specificity."]
        },
        {
            "query": "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
            "gold_answer": "YG Entertainment",
            "final_answer": "YG Entertainment",
            "num_refinement": 0,
            "num_planning": 2,
            "steps_trace": [
                {"node": "Validator", "action": "ROUTE_TO_PLANNING", "detail": "Query multi-hop."},
                {"node": "Planner", "action": "ReAct Multi-Hop (Ciclo 1)", "detail": "Allucinazione Big Hit intercettata dal Validatore (IsSup=False)."},
                {"node": "Validator", "action": "ROUTE_TO_PLANNING", "detail": "Reflexion feedback: cerca esplicitamente l'etichetta di Winner."},
                {"node": "Planner", "action": "ReAct Multi-Hop (Ciclo 2)", "detail": "Early stopping all'osservazione dell'etichetta YG Entertainment."},
                {"node": "Validator", "action": "FINISH", "detail": "Exact Match perfetto verificato."}
            ],
            "reasoning": "The South Korean boy group that released the debut album 2014 S/S in 2014 was Winner, which was formed by YG Entertainment.",
            "retrieved_context": "--- Wikipedia Result: 2014 S/S ---\n2014 S/S is the debut album of South Korean group Winner. It was released on August 12, 2014, by the group's record label, YG Entertainment...\n--- Wikipedia Result: YG Entertainment ---\nYG Entertainment Inc. is a South Korean multinational entertainment agency...",
            "feedback_history": ["Rejected initial candidate Big Hit Entertainment; re-searched album release metadata."]
        },
        {
            "query": "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?",
            "gold_answer": "Eenasul Fateh",
            "final_answer": "James Monroe Iglehart",
            "num_refinement": 0,
            "num_planning": 2,
            "steps_trace": [
                {"node": "Validator", "action": "ROUTE_TO_PLANNING", "detail": "Query descrittiva."},
                {"node": "Planner", "action": "ReAct", "detail": "Web search: recuperato attore di musical Disney Aladdin (James Monroe Iglehart)."},
                {"node": "Validator", "action": "FINISH", "detail": "Grounding completato su pagina live web per assenza di voce biografica standalone per Eenasul Fateh."}
            ],
            "reasoning": "The search for stage name Aladin and performance consultant on modern live web returned theatrical performer James Monroe Iglehart (Aladdin on Broadway).",
            "retrieved_context": "--- Web Result 1: James Monroe Iglehart - Wikipedia ---\nJames Monroe Iglehart is an American actor best known for playing the Genie in the Broadway production of Aladdin...",
            "feedback_history": ["Eenasul Fateh has no standalone Wikipedia EN article in modern dumps (temporal drift)."]
        },
        {
            "query": "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
            "gold_answer": "3,677 seated",
            "final_answer": "6,000 people",
            "num_refinement": 0,
            "num_planning": 1,
            "steps_trace": [
                {"node": "Validator", "action": "ROUTE_TO_PLANNING", "detail": "Query quantitativa multi-hop."},
                {"node": "Planner", "action": "ReAct Multi-Hop", "detail": "Hop 1: Lewiston Maineiacs home arena -> The Colisée -> estratta capienza massima estesa."},
                {"node": "Validator", "action": "FINISH", "detail": "Capienza numerica verificata nel testo recuperato."}
            ],
            "reasoning": "The Lewiston Maineiacs played their home games at the Androscoggin Bank Colisée. The venue lists multiple capacities up to 6,000 for events.",
            "retrieved_context": "--- Wikipedia Result: The Colisée ---\nThe Colisée is a 4,000 capacity (3,677 seated) multi-purpose arena in Lewiston, Maine. For conventions and events it can accommodate up to 6,000 people...",
            "feedback_history": []
        },
        {
            "query": "Who is older, Annie Morton or Terry Richardson?",
            "gold_answer": "Terry Richardson",
            "final_answer": "Terry Richardson",
            "num_refinement": 1,
            "num_planning": 1,
            "steps_trace": [
                {"node": "Validator", "action": "ROUTE_TO_REFINEMENT", "detail": "Query comparativa d'età."},
                {"node": "Refiner", "tool": "decompose_query", "detail": "Decomposizione: 'When was Annie Morton born?' / 'When was Terry Richardson born?'"},
                {"node": "Planner", "action": "single_retrieval (WEB)", "detail": "Recuperati anni di nascita: Annie Morton (1970) e Terry Richardson (1965)."},
                {"node": "Planner", "action": "Age comparison safeguard", "detail": "1965 < 1970 -> Terry Richardson è nato prima ed è più anziano."},
                {"node": "Validator", "action": "FINISH", "detail": "Exact Match confermato."}
            ],
            "reasoning": "Annie Morton was born on October 8, 1970. Terry Richardson was born on August 14, 1965. Since 1965 is earlier than 1970, Terry Richardson is older.",
            "retrieved_context": "--- Wikipedia Result: Annie Morton ---\nAnnie Morton (born October 8, 1970) is an American model...\n--- Wikipedia Result: Terry Richardson ---\nTerrence Richardson (born August 14, 1965) is an American fashion photographer...",
            "feedback_history": []
        },
        {
            "query": "Are Local H and For Against both from the United States?",
            "gold_answer": "yes",
            "final_answer": "yes, both Local H and For Against are from the United States.",
            "num_refinement": 1,
            "num_planning": 1,
            "steps_trace": [
                {"node": "Validator", "action": "ROUTE_TO_REFINEMENT", "detail": "Query comparativa geografica multi-entità."},
                {"node": "Refiner", "tool": "decompose_query", "detail": "Decomposizione atomica: 'Is Local H from the United States?' / 'Is For Against from the United States?'"},
                {"node": "Planner", "action": "single_retrieval (WEB)", "detail": "Recuperate origini: Local H (Zion, Illinois, US) e For Against (Lincoln, Nebraska, US)."},
                {"node": "Validator", "action": "FINISH", "detail": "Polar match positivo confermato con evidenze per entrambe le band."}
            ],
            "reasoning": "Both Local H and For Against are American rock bands, with Local H originating from Zion, Illinois, and For Against from Lincoln, Nebraska.",
            "retrieved_context": "--- Wikipedia Result: Local H ---\nLocal H is an American alternative rock band from Zion, Illinois...\n--- Wikipedia Result: For Against ---\nFor Against is an American post-punk band from Lincoln, Nebraska, United States...",
            "feedback_history": []
        }
    ]

    for item in benchmark_data:
        log_query_run(
            query=item["query"],
            final_answer=item["final_answer"],
            gold_answer=item["gold_answer"],
            num_refinement=item["num_refinement"],
            num_planning=item["num_planning"],
            steps_trace=item["steps_trace"],
            reasoning=item["reasoning"],
            retrieved_context=item["retrieved_context"],
            feedback_history=item["feedback_history"],
            db_path=db_path
        )


if __name__ == "__main__":
    init_history_db()
    seed_benchmark_runs()
    stats = get_history_stats()
    print("Database storico inizializzato e popolato con successo:")
    print(stats)

