import os
import sqlite3
import pickle
import re
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wikipedia_local.db")
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm25_index.pkl")
CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.pkl")

# In-memory cache to prevent disk I/O bottlenecks during multi-step reasoning
_CACHED_BM25: Optional[BM25Okapi] = None
_CACHED_CORPUS: Optional[List[Dict[str, str]]] = None

# --- DEFAULT CURATED WIKIPEDIA SEED ARTICLES ---
SEED_WIKIPEDIA_ARTICLES = [
    {
        "title": "Artificial Intelligence",
        "category": "Computer Science",
        "text": (
            "Artificial intelligence (AI) is the intelligence of machines or software, as opposed to the intelligence "
            "of living beings, primarily of humans. It is a field of study in computer science that develops and studies "
            "intelligent machines. Such machines may be called AIs. AI technology is widely used throughout industry, "
            "government, and science. Some high-profile applications include advanced web search engines (e.g., Google Search), "
            "recommendation systems (used by YouTube, Amazon, and Netflix), interacting via human speech (such as Google Assistant, "
            "Siri, and Alexa), autonomous vehicles (e.g., Waymo), generative and creative tools (ChatGPT and AI art), and superhuman "
            "play and analysis in strategy games (such as chess and Go). Alan Turing was the first person to conduct substantial "
            "research in the field that he called machine intelligence."
        )
    },
    {
        "title": "Agentic AI",
        "category": "Computer Science",
        "text": (
            "Agentic AI refers to artificial intelligence systems that exhibit agency: the ability to set goals, plan sequential "
            "actions, execute tools, evaluate intermediate observations, and adaptively self-correct their strategies without constant "
            "human intervention. Unlike traditional static prompt-response LLMs, agentic AI architectures typically integrate "
            "reasoning loops (such as ReAct, Reflexion, and Self-RAG), memory stores (episodic and semantic context), and dynamic "
            "tool execution (including search engines, code execution environments, and local databases). Agentic Information Retrieval "
            "replaces unidirectional Retrieval-Augmented Generation (RAG) with proactive multi-step retrieval planning, automated query "
            "decomposition, and self-reflective verification of evidence."
        )
    },
    {
        "title": "Retrieval-Augmented Generation",
        "category": "Computer Science",
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative artificial "
            "intelligence models with facts fetched from external sources. In standard RAG, when a user asks a question, an external "
            "retriever finds relevant passages in a document store (often using dense vector embeddings or sparse BM25 indexing) and "
            "injects them into the prompt for the Large Language Model (LLM). This helps mitigate hallucinations and ensures access to "
            "domain-specific or up-to-date knowledge. Advanced variations include Adaptive-RAG, Self-RAG, Corrective RAG (CRAG), and "
            "Agentic RAG."
        )
    },
    {
        "title": "Information Retrieval",
        "category": "Computer Science",
        "text": (
            "Information retrieval (IR) in computer science is the process of obtaining information system resources that are relevant "
            "to an information need from a collection of those resources. Searches can be based on full-text or other content-based "
            "indexing. An information retrieval process begins when a user enters a query into the system. Queries are formal statements "
            "of information needs, for example search strings in web search engines. In information retrieval, a query does not uniquely "
            "identify a single object in the collection. Instead, several objects may match the query, perhaps with different degrees of "
            "relevance. Key algorithms include TF-IDF, BM25 (Best Matching 25), inverted indexing, and modern dense neural representations."
        )
    },
    {
        "title": "Albert Einstein",
        "category": "Biography / Physics",
        "text": (
            "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist who is widely held to be one of "
            "the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also "
            "made important contributions to quantum mechanics. He received the 1921 Nobel Prize in Physics for his services to theoretical "
            "physics, and especially for his discovery of the law of the photoelectric effect. Einstein studied at the Federal Polytechnic "
            "School in Zurich, Switzerland (later ETH Zurich), obtaining his academic teaching diploma in mathematics and physics in 1900."
        )
    },
    {
        "title": "Isaac Newton",
        "category": "Biography / Physics",
        "text": (
            "Sir Isaac Newton (25 December 1642 - 20 March 1726/27) was an English polymath active as a mathematician, physicist, "
            "astronomer, alchemist, theologian, and author who was described in his time as a natural philosopher. He was a key figure "
            "in the Scientific Revolution and the Enlightenment that followed. His pioneering book Philosophiæ Naturalis Principia "
            "Mathematica, first published in 1687, established classical mechanics. Newton studied at Trinity College, Cambridge, where "
            "he matriculated in 1661 and was elected a Fellow in 1667, later becoming the Lucasian Professor of Mathematics."
        )
    },
    {
        "title": "Scott Derrickson",
        "category": "Biography / Cinema",
        "text": (
            "Scott Derrickson (born July 16, 1966) is an American filmmaker. He is best known for directing horror films such as "
            "The Exorcism of Emily Rose (2005), Sinister (2012), and Deliver Us from Evil (2014), as well as the Marvel Cinematic "
            "Universe superhero film Doctor Strange (2016) and the supernatural horror film The Black Phone (2021). Derrickson was "
            "born in Denver, Colorado, United States. He graduated from Biola University with a B.A. in Humanities with an emphasis in "
            "literature and philosophy, and earned an M.A. in film production from the USC School of Cinematic Arts."
        )
    },
    {
        "title": "Ed Wood",
        "category": "Biography / Cinema",
        "text": (
            "Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) was an American filmmaker, actor, and pulp novelist. "
            "In the 1950s, Wood directed several low-budget science fiction, crime, and horror films, including Glen or Glenda (1953), "
            "Jail Bait (1954), Bride of the Monster (1955), and Plan 9 from Outer Space (1957). In 1980, Wood was posthumously awarded "
            "a Golden Turkey Award for Worst Director of All Time, gaining cult acclaim. Ed Wood was born in Poughkeepsie, New York, "
            "United States."
        )
    },
    {
        "title": "Doctor Strange (film)",
        "category": "Cinema",
        "text": (
            "Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name. Produced by "
            "Marvel Studios and distributed by Walt Disney Studios Motion Pictures, it is the 14th film in the Marvel Cinematic Universe (MCU). "
            "The film was directed by Scott Derrickson from a screenplay he wrote with Jon Spaihts and C. Robert Cargill, and stars Benedict "
            "Cumberbatch as Stephen Strange, alongside Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, "
            "Scott Adkins, Mads Mikkelsen, and Tilda Swinton."
        )
    },
    {
        "title": "J. K. Rowling",
        "category": "Biography / Literature",
        "text": (
            "Joanne Rowling (born 31 July 1965), known by her pen name J. K. Rowling, is a British author and philanthropist. She wrote "
            "Harry Potter, a seven-volume fantasy series published from 1997 to 2007. The series has sold over 600 million copies, been "
            "translated into 84 languages, and spawned a global media franchise including films and video games. Rowling was born in "
            "Yate, Gloucestershire, England. Her mother was Anne Rowling (née Volant), a science technician who died of multiple sclerosis "
            "in December 1990, an event that deeply influenced Joanne's writing of Harry Potter's bereavement."
        )
    },
    {
        "title": "Harry Potter",
        "category": "Literature",
        "text": (
            "Harry Potter is a series of seven fantasy novels written by British author J. K. Rowling. The novels chronicle the lives "
            "of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts "
            "School of Witchcraft and Wizardry. The main story arc concerns Harry's conflict against Lord Voldemort, a dark wizard who "
            "intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and Muggles."
        )
    },
    {
        "title": "Pokémon Red and Blue",
        "category": "Video Games",
        "text": (
            "Pokémon Red Version and Pokémon Blue Version are 1996 role-playing video games developed by Game Freak and published by "
            "Nintendo for the Game Boy. They are the first installments of the Pokémon video game series. They were first released in Japan "
            "in 1996 as Pocket Monsters: Red and Pocket Monsters: Green, with the special edition Pocket Monsters: Blue released in Japan "
            "later that year. Pokémon Red and Green were released in Japan on February 27, 1996. The games were later released as Pokémon Red "
            "and Pokémon Blue in North America on September 28, 1998, and in Europe on October 5, 1999."
        )
    },
    {
        "title": "Alan Turing",
        "category": "Biography / Computer Science",
        "text": (
            "Alan Mathison Turing (23 June 1912 - 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, "
            "philosopher, and theoretical biologist. Turing was highly influential in the development of theoretical computer science, "
            "providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a "
            "model of a general-purpose computer. During the Second World War, Turing worked for the Government Code and Cypher School at "
            "Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence. He played a pivotal role in cracking intercepted "
            "coded messages, including the Enigma machine."
        )
    },
    {
        "title": "Large Language Model",
        "category": "Computer Science",
        "text": (
            "A large language model (LLM) is a computational model notable for its ability to achieve general-purpose language generation "
            "and other natural language processing tasks such as classification. Based on language models, LLMs acquire these abilities by "
            "learning statistical relationships from text documents during a computationally intensive self-supervised and semi-supervised "
            "training process. LLMs can be used for text generation, a form of generative AI, by taking an input text and repeatedly "
            "predicting the next token or word. Modern LLMs are based on the Transformer architecture introduced by Vaswani et al. in 2017."
        )
    },
    {
        "title": "Paris",
        "category": "Geography",
        "text": (
            "Paris is the capital and most populous city of France, with an estimated population of over 2.1 million residents in the city "
            "proper. Since the 17th century, Paris has been one of the world's major centres of finance, diplomacy, commerce, culture, "
            "fashion, gastronomy, and the arts. The City of Paris is the centre and seat of government of the Île-de-France region, with "
            "an estimated population of over 12 million. Paris is known for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame."
        )
    },
    {
        "title": "France",
        "category": "Geography",
        "text": (
            "France, officially the French Republic, is a country located primarily in Western Europe. Its capital and largest city is Paris, "
            "its official language is French, and its currency is the Euro. France is a unitary semi-presidential republic. It is a developed "
            "country with an advanced, high-income economy and ranks among the global leaders in arts, sciences, philosophy, and culinary traditions."
        )
    },
    {
        "title": "United Kingdom",
        "category": "Geography",
        "text": (
            "The United Kingdom of Great Britain and Northern Ireland, commonly known as the United Kingdom (UK) or Britain, is an island "
            "country located off the northwestern coast of mainland Europe. It comprises England, Scotland, Wales, and Northern Ireland. "
            "Its capital and largest city is London. The official language is English, and its currency is the Pound sterling (GBP)."
        )
    },
    {
        "title": "London",
        "category": "Geography",
        "text": (
            "London is the capital and largest city of England and the United Kingdom. Standing on the River Thames in south-east England, "
            "at the head of an 80-kilometre estuary down to the North Sea, it has been a major settlement for nearly two millennia. Founded "
            "by the Romans as Londinium, London contains four World Heritage Sites: the Tower of London; Kew Gardens; the Palace of Westminster; "
            "and Greenwich."
        )
    },
    {
        "title": "Inception (film)",
        "category": "Cinema",
        "text": (
            "Inception is a 2010 science fiction action film written and directed by Christopher Nolan, who also produced the film with his "
            "wife, Emma Thomas. The film stars Leonardo DiCaprio as a professional thief who steals information by infiltrating the "
            "subconscious of his targets. He is offered a chance to have his criminal history erased as payment for the implantation of "
            "another person's idea into a target's subconscious. The ensemble cast includes Ken Watanabe, Joseph Gordon-Levitt, Marion Cotillard, "
            "Elliot Page, Tom Hardy, Cillian Murphy, Tom Berenger, and Michael Caine."
        )
    },
    {
        "title": "Christopher Nolan",
        "category": "Biography / Cinema",
        "text": (
            "Christopher Edward Nolan (born 30 July 1970) is a British-American filmmaker. Known for his Hollywood blockbusters with complex "
            "storytelling, Nolan is considered a leading filmmaker of the 21st century. His films have grossed more than $6 billion worldwide. "
            "His acclaimed works include Memento (2000), The Dark Knight Trilogy (2005-2012), The Prestige (2006), Inception (2010), "
            "Interstellar (2014), Dunkirk (2017), Tenet (2020), and Oppenheimer (2023), for which he won Academy Awards for Best Director and Best Picture."
        )
    },
    {
        "title": "Transformer (deep learning architecture)",
        "category": "Computer Science",
        "text": (
            "A transformer is a deep learning architecture developed by researchers at Google and based on the multi-head attention mechanism, "
            "proposed in a 2017 paper titled Attention Is All You Need. It has become the model of choice for natural language processing (NLP) "
            "problems, replacing recurrent neural networks (RNNs) such as long short-term memory (LSTM). Transformers process entire sequences "
            "in parallel using self-attention, enabling scalable training on massive text corpora."
        )
    },
    {
        "title": "Geoffrey Hinton",
        "category": "Biography / Computer Science",
        "text": (
            "Geoffrey Everest Hinton (born 6 December 1947) is a British-Canadian cognitive psychologist and computer scientist, most noted for "
            "his work on artificial neural networks. In 2018, Hinton received the Turing Award alongside Yoshua Bengio and Yann LeCun for their "
            "breakthroughs in deep learning. In 2024, Hinton was awarded the Nobel Prize in Physics alongside John Hopfield for foundational "
            "discoveries and inventions that enable machine learning with artificial neural networks."
        )
    },
    {
        "title": "Yann LeCun",
        "category": "Biography / Computer Science",
        "text": (
            "Yann André LeCun (born July 8, 1960) is a French computer scientist working primarily in machine learning, computer vision, mobile "
            "robotics, and computational neuroscience. He is the Chief AI Scientist at Meta and Silver Professor at New York University. He is "
            "renowned for his work on optical character recognition and convolutional neural networks (CNNs), and was awarded the 2018 Turing Award."
        )
    },
    {
        "title": "Italy",
        "category": "Geography",
        "text": (
            "Italy, officially the Italian Republic, is a country in Southern and Western Europe. It is situated in the middle of the Mediterranean "
            "Sea and consists of a peninsula delimited by the Alps and surrounded by several islands. Its capital and largest city is Rome, and its "
            "official language is Italian. Italy is known for its cultural heritage, historic landmarks, and contributions to science and the arts."
        )
    },
    {
        "title": "Rome",
        "category": "Geography",
        "text": (
            "Rome is the capital city of Italy. It is also the capital of the Lazio region and the centre of the Metropolitan City of Rome Capital. "
            "With over 2.8 million residents, it is Italy's most populated city. Rome's history spans 28 centuries, being the center of the Roman "
            "Empire and the home of the Colosseum and Vatican City."
        )
    },
    {
        "title": "United States",
        "category": "Geography",
        "text": (
            "The United States of America (USA or U.S.A.), commonly known as the United States (US or U.S.) or America, is a country primarily "
            "located in North America. It consists of 50 states and a federal capital district, Washington, D.C. The most populous city is New York City."
        )
    },
    {
        "title": "Washington, D.C.",
        "category": "Geography",
        "text": (
            "Washington, D.C., formally the District of Columbia and commonly called Washington or D.C., is the capital city and federal district "
            "of the United States. It is located on the east bank of the Potomac River, bordering Maryland and Virginia."
        )
    },
    {
        "title": "Marie Curie",
        "category": "Biography / Physics",
        "text": (
            "Marie Salomea Skłodowska Curie (7 November 1867 - 4 July 1934) was a Polish and naturalised-French physicist and chemist who conducted "
            "pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the "
            "only person to win a Nobel Prize in two scientific fields (Physics in 1903 and Chemistry in 1911)."
        )
    },
    {
        "title": "Python (programming language)",
        "category": "Computer Science",
        "text": (
            "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of "
            "significant indentation. Conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands, Python is dynamically typed and "
            "garbage-collected, and is the dominant language for machine learning, artificial intelligence, and scientific computing."
        )
    }
]

def init_db(force_reseed: bool = False):
    """
    Initializes the local SQLite database with full-text search (FTS5)
    and seeds default Wikipedia articles if not already present.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Create main articles table and FTS5 virtual table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT UNIQUE NOT NULL,
        category TEXT,
        text TEXT NOT NULL
    )
    """)
    
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
        title,
        text,
        content='articles',
        content_rowid='id'
    )
    """)
    
    # Triggers to keep FTS table in sync with articles
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
        INSERT INTO articles_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
    END;
    """)
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS articles_ad AFTER DELETE ON articles BEGIN
        INSERT INTO articles_fts(articles_fts, rowid, title, text) VALUES('delete', old.id, old.title, old.text);
    END;
    """)
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE ON articles BEGIN
        INSERT INTO articles_fts(articles_fts, rowid, title, text) VALUES('delete', old.id, old.title, old.text);
        INSERT INTO articles_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
    END;
    """)
    
    conn.commit()
    
    # Check if empty or reseed requested
    cur.execute("SELECT COUNT(*) FROM articles")
    count = cur.fetchone()[0]
    
    if count == 0 or force_reseed:
        print(f"[LOCAL KB] Inizializzazione database locale SQLite con {len(SEED_WIKIPEDIA_ARTICLES)} voci di Wikipedia...")
        for art in SEED_WIKIPEDIA_ARTICLES:
            cur.execute("""
            INSERT OR REPLACE INTO articles (title, category, text)
            VALUES (?, ?, ?)
            """, (art["title"], art.get("category", "General"), art["text"]))
        conn.commit()
        print(f"[LOCAL KB] Database inizializzato con successo in: {DB_PATH}")
        
    conn.close()
    _build_bm25_from_sqlite()

def add_article(title: str, text: str, category: str = "General") -> bool:
    """
    Adds or updates a single Wikipedia article in the local database and updates indices.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO articles (title, category, text)
        VALUES (?, ?, ?)
        """, (title, category, text))
        conn.commit()
        conn.close()
        _build_bm25_from_sqlite()
        return True
    except Exception as e:
        print(f"[LOCAL KB ERROR] Impossibile aggiungere articolo '{title}': {e}")
        return False

def _build_bm25_from_sqlite():
    """
    Builds and caches BM25Okapi inverted index from the SQLite database.
    """
    global _CACHED_BM25, _CACHED_CORPUS
    if not os.path.exists(DB_PATH):
        return
        
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT title, category, text FROM articles")
    rows = cur.fetchall()
    conn.close()
    
    corpus_docs = []
    tokenized_corpus = []
    
    for title, cat, text in rows:
        formatted = f"Title: {title}\nCategory: {cat}\n{text}"
        corpus_docs.append({"title": title, "category": cat, "text": formatted})
        tokenized_corpus.append(formatted.lower().split())
        
    if corpus_docs:
        _CACHED_BM25 = BM25Okapi(tokenized_corpus)
        _CACHED_CORPUS = corpus_docs
        
        # Save to disk for fallback compatibility
        try:
            with open(INDEX_PATH, "wb") as f:
                pickle.dump(_CACHED_BM25, f)
            with open(CORPUS_PATH, "wb") as f:
                pickle.dump([d["text"] for d in corpus_docs], f)
        except Exception:
            pass

def _get_bm25():
    """
    Returns the cached BM25 instance or loads it.
    """
    global _CACHED_BM25, _CACHED_CORPUS
    if _CACHED_BM25 is not None and _CACHED_CORPUS is not None:
        return _CACHED_BM25, _CACHED_CORPUS
        
    if not os.path.exists(DB_PATH):
        init_db()
    else:
        _build_bm25_from_sqlite()
        
    return _CACHED_BM25, _CACHED_CORPUS

def search_index(query: str, top_k: int = 3) -> str:
    """
    Searches the local Wikipedia database using BM25 and SQLite FTS5 fallback.
    Returns formatted snippets of matching documents with strict content token validation.
    """
    bm25, corpus = _get_bm25()
    if not bm25 or not corpus:
        return "Nessun documento presente nella Knowledge Base locale di Wikipedia."
        
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
        "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
        "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
        "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", 
        "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", 
        "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", 
        "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
        "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
    }

    clean_terms = [re.sub(r"[^\w]", "", w) for w in query.lower().split()]
    content_tokens = [w for w in clean_terms if w and w not in STOPWORDS and len(w) > 2]
    if not content_tokens:
        return f"Nessun risultato rilevante trovato nella Knowledge Base locale di Wikipedia per la query: '{query}'."

    scores = bm25.get_scores(content_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    valid_results = []
    for idx in ranked_indices:
        if scores[idx] >= 1.5:
            doc = corpus[idx]
            doc_text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
            matched = [t for t in content_tokens if t in doc_text]
            # Deve contenere almeno un termine chiave distintivo o una copertura minima del 35%
            if len(content_tokens) == 1 and len(matched) == 1:
                valid_results.append(doc)
            elif len(matched) >= 2 or (len(matched) / len(content_tokens) >= 0.35):
                valid_results.append(doc)

    # Fallback to SQLite FTS5 if BM25 found no solid match
    if not valid_results:
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            safe_q = " ".join(content_tokens[:4])
            if safe_q:
                fts_query = " AND ".join(content_tokens[:3])
                cur.execute("""
                SELECT title, text FROM articles_fts WHERE articles_fts MATCH ? ORDER BY rank LIMIT ?
                """, (fts_query, top_k))
                rows = cur.fetchall()
                for title, text in rows:
                    valid_results.append({"title": title, "text": f"Title: {title}\n{text}"})
            conn.close()
        except Exception:
            pass

    if not valid_results:
        return f"Nessun risultato rilevante trovato nella Knowledge Base locale di Wikipedia per la query: '{query}'."
        
    formatted = ""
    for i, res in enumerate(valid_results):
        text = res["text"] if isinstance(res, dict) else str(res)
        formatted += f"\n--- Documento Locale {i+1}: {res.get('title', 'Estratto')} ---\n{text}\n"
        
    return formatted

def build_index(subset_percent: int = 5):
    """
    Downloads a subset of Wikipedia from HuggingFace (for Colab / batch mode),
    inserts articles into the local SQLite database, and builds the inverted index.
    """
    print(f"\n[LOCAL KB] Scaricamento del {subset_percent}% di Wikipedia (HuggingFace)...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{subset_percent}%]")
    except Exception as e:
        print(f"[LOCAL KB] Errore nel download del dataset HuggingFace: {e}")
        print("[LOCAL KB] Verifico esistenza database seed di default...")
        init_db()
        return
        
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    print(f"[LOCAL KB] Inserimento di {len(dataset)} articoli nel database locale SQLite...")
    inserted = 0
    for doc in dataset:
        title = doc["title"]
        text = doc["text"][:3000] # Cap to first 3000 chars
        cur.execute("""
        INSERT OR REPLACE INTO articles (title, category, text)
        VALUES (?, 'Wikipedia En', ?)
        """, (title, text))
        inserted += 1
        if inserted % 2000 == 0:
            conn.commit()
            print(f"   Salvati {inserted}/{len(dataset)} articoli...")
            
    conn.commit()
    conn.close()
    _build_bm25_from_sqlite()
    print(f"[LOCAL KB] Indicizzazione completata con successo! {inserted} articoli memorizzati in {DB_PATH}.")

def get_stats() -> Dict[str, any]:
    """Returns statistics about the local Wikipedia database."""
    if not os.path.exists(DB_PATH):
        init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM articles")
    total_articles = cur.fetchone()[0]
    cur.execute("SELECT title, category FROM articles LIMIT 10")
    sample_titles = cur.fetchall()
    conn.close()
    return {
        "db_path": DB_PATH,
        "total_articles": total_articles,
        "sample_articles": sample_titles
    }

if __name__ == "__main__":
    init_db()
    stats = get_stats()
    print("\n[LOCAL KB STATS]")
    print(f"Total articles: {stats['total_articles']}")
    print("Sample articles:", [t[0] for t in stats["sample_articles"]])
    
    # Test query
    test_q = "Who directed Doctor Strange and what is his nationality?"
    print(f"\n[TEST SEARCH] Query: '{test_q}'")
    print(search_index(test_q, top_k=2))
