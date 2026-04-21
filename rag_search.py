import chromadb
import os
import re
from collections import Counter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from extract_pdf import extract_text_from_pdf, chunk_text
from logger import get_logger

load_dotenv()

log = get_logger("rag_search")

# Local embedding model — no API key needed, runs on device
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
log.info("[RAG] SentenceTransformer embedding model loaded (all-MiniLM-L6-v2)")

client_chroma = chromadb.PersistentClient(path="data/vectordb")
log.info("[RAG] ChromaDB client initialised at 'data/vectordb'")

# ================= EMBEDDING =================

def embed_text(text):
    """
    Embed text via local SentenceTransformer model.
    No API key, no rate limits, no token limits.
    Returns a 384-dimensional vector.
    """
    log.debug(f"[EMBED] Embedding text ({len(text.split())} words)...")
    try:
        vector = embedding_model.encode(text, convert_to_numpy=True).tolist()
        log.debug(f"[EMBED] ✅ Embedding successful ({len(vector)} dims)")
        return vector
    except Exception as e:
        log.error(f"[EMBED] ❌ Embedding failed: {e}")
        raise


# ================= DOMAIN KEYWORDS =================

DOMAIN_KEYWORDS = {
    "dishwasher_manual": [
        "dishwasher", "dishes", "cutlery", "rinse aid", "spray arm",
        "basket", "rack", "tableware", "plates", "glasses",
        "utensils", "silverware", "wash cycle", "dishwasher salt",
        "upper basket", "lower basket", "filter mesh"
    ],
    "washing_machine_manual": [
        "washing machine", "laundry", "clothes", "fabric", "garment",
        "spin", "drum", "detergent drawer", "wash programme",
        "wool", "cotton", "synthetics", "fabric softener",
        "stain", "spin speed", "load", "lint filter"
    ]
}

DYNAMIC_KEYWORDS = {}

_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "your",
    "have", "will", "when", "which", "they", "been", "also", "into",
    "more", "than", "then", "their", "there", "about", "using", "used",
    "make", "each", "after"
}


def extract_dynamic_keywords(text, top_n=20):
    words = re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split()
    filtered = [w for w in words if len(w) >= 4 and w not in _STOPWORDS]
    return [word for word, _ in Counter(filtered).most_common(top_n)]


# ================= INTENT CLASSIFICATION =================

INTENT_PATTERNS = {
    "location":      r'\b(where|location|find|place|position|located)\b',
    "howto":         r'\b(how|steps|instructions|procedure|guide|do i|can i)\b',
    "troubleshoot":  r'\b(error|problem|not working|broken|failed|fault|issue|fix|repair)\b',
    "definition":    r'\b(what is|what are|explain|define|meaning|means)\b',
    "specification": r'\b(capacity|temperature|speed|setting|programme|mode|type)\b',
}


def detect_intent(query):
    q = query.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, q):
            log.debug(f"[INTENT] Detected intent: '{intent}'")
            return intent
    log.debug("[INTENT] No specific intent detected → 'general'")
    return "general"


# ================= ENTITY EXTRACTION =================

def extract_entities(query, manual_name):
    domain_words = DOMAIN_KEYWORDS.get(manual_name, [])
    q_lower = query.lower()
    found = [word for word in domain_words if word in q_lower]
    if found:
        log.debug(f"[ENTITIES] Found domain entities: {found}")
    else:
        log.debug("[ENTITIES] No domain entities found in query")
    return found


# ================= SCORING =================

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 ]', ' ', text.lower())


def keyword_score(query, doc):
    q_words = {w for w in clean_text(query).split() if len(w) > 2}
    d_text = clean_text(doc)
    return sum(1 for w in q_words if w in d_text)


def domain_relevance_score(doc, manual_name):
    doc_lower = doc.lower()
    own_keywords = DOMAIN_KEYWORDS.get(manual_name, [])
    if not own_keywords:
        own_keywords = DYNAMIC_KEYWORDS.get(manual_name, [])
    own_score = sum(1 for kw in own_keywords if kw in doc_lower)

    penalty = 0
    for other_manual, other_keywords in DOMAIN_KEYWORDS.items():
        if other_manual != manual_name:
            penalty += sum(1 for kw in other_keywords if kw in doc_lower)

    return own_score - (penalty * 2)


def total_chunk_score(query, doc, manual_name):
    return (keyword_score(query, doc) * 2) + domain_relevance_score(doc, manual_name)


# ================= RELEVANCE THRESHOLDS =================

MIN_RELEVANCE_SCORE = 1
CONFIDENCE_HIGH_THRESHOLD = 6
CONFIDENCE_MED_THRESHOLD = 2


def assess_confidence(chunks_with_scores):
    if not chunks_with_scores:
        return "none"
    best = max(s for _, s in chunks_with_scores)
    if best >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if best >= CONFIDENCE_MED_THRESHOLD:
        return "medium"
    return "low"


# ================= LOAD MANUAL =================

def load_manual(manual_name, file_path):
    log.info(f"[LOAD] ── Starting load for '{manual_name}' ──")

    # Step 1 — Get or create ChromaDB collection
    log.debug(f"[LOAD] [1/4] Accessing ChromaDB collection '{manual_name}'...")
    try:
        collection = client_chroma.get_or_create_collection(name=manual_name)
        log.debug(f"[LOAD] ✅ [1/4] Collection ready")
    except Exception as e:
        log.error(f"[LOAD] ❌ [1/4] Failed to create collection '{manual_name}': {e}")
        return

    # Already loaded check
    if collection.count() > 0:
        log.info(f"[LOAD] ✅ '{manual_name}' already has {collection.count()} chunks — skipping ingestion")
        return

    # Step 2 — Check file exists
    log.debug(f"[LOAD] [2/4] Checking file path '{file_path}'...")
    if not os.path.exists(file_path):
        log.error(f"[LOAD] ❌ [2/4] File not found: '{file_path}'")
        return
    log.debug(f"[LOAD] ✅ [2/4] File found")

    # Step 3 — Extract and chunk
    log.debug(f"[LOAD] [3/4] Extracting text from PDF...")
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        log.error(f"[LOAD] ❌ [3/4] No text extracted from '{file_path}'")
        return
    log.info(f"[LOAD] ✅ [3/4] Extracted {len(text)} characters")

    chunks = chunk_text(text)

    if chunks and manual_name not in DOMAIN_KEYWORDS:
        keywords = extract_dynamic_keywords(text)
        DYNAMIC_KEYWORDS[manual_name] = keywords
        log.info(f"[LOAD] ✅ Dynamic keywords extracted for '{manual_name}': {keywords[:5]}...")

    log.info(f"[LOAD] [4/4] Embedding {len(chunks)} chunks into ChromaDB...")

    # Step 4 — Embed and store each chunk
    success = 0
    failed = 0

    for i, chunk in enumerate(chunks):
        try:
            embedding = embed_text(chunk)
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{manual_name}_chunk_{i}"]
            )
            success += 1

            if i > 0 and i % 50 == 0:
                log.debug(f"[LOAD] Progress: {i}/{len(chunks)} chunks embedded...")

        except Exception as e:
            log.warning(f"[LOAD] ⚠️ Skipping chunk {i}: {e}")
            failed += 1
            continue

    log.info(f"[LOAD] ✅ [4/4] '{manual_name}' done — {success} embedded, {failed} skipped")


# ================= SEARCH =================

def search_manual(query, manual_name, top_k=8):
    log.info(f"[SEARCH] ── Query for '{manual_name}' ──")
    log.debug(f"[SEARCH] Query: '{query[:60]}{'...' if len(query) > 60 else ''}'")

    # Step 1 — Access collection
    log.debug(f"[SEARCH] [1/5] Accessing collection '{manual_name}'...")
    try:
        collection = client_chroma.get_or_create_collection(name=manual_name)
    except Exception as e:
        log.error(f"[SEARCH] ❌ [1/5] Cannot access collection: {e}")
        return [], "none"

    if collection.count() == 0:
        log.error(f"[SEARCH] ❌ [1/5] Collection '{manual_name}' is empty — was it loaded?")
        return [], "none"
    log.debug(f"[SEARCH] ✅ [1/5] Collection has {collection.count()} chunks")

    # Step 2 — Intent detection
    intent = detect_intent(query)
    log.debug(f"[SEARCH] [2/5] Intent: '{intent}'")

    # Step 3 — Semantic search
    log.debug(f"[SEARCH] [3/5] Embedding query and running semantic search (top_k={top_k})...")
    try:
        query_embedding = embed_text(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count())
        )
        docs = results.get("documents", [[]])[0]
        log.info(f"[SEARCH] ✅ [3/5] Semantic search returned {len(docs)} candidate chunk(s)")
    except Exception as e:
        log.error(f"[SEARCH] ❌ [3/5] Query embedding or search FAILED: {e}")
        return [], "none"

    if not docs:
        log.warning(f"[SEARCH] ⚠️ No documents returned from ChromaDB")
        return [], "none"

    # Step 4 — Score, rank, filter
    log.debug(f"[SEARCH] [4/5] Scoring and filtering chunks...")
    scored = [(doc, total_chunk_score(query, doc, manual_name)) for doc in docs]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_scores = [s for _, s in scored[:3]]
    log.debug(f"[SEARCH] Top 3 chunk scores: {top_scores}")

    filtered = [(doc, s) for doc, s in scored if s >= MIN_RELEVANCE_SCORE]
    confidence = assess_confidence(filtered if filtered else scored)
    log.info(f"[SEARCH] ✅ [4/5] After filtering: {len(filtered)} chunk(s) pass threshold | confidence='{confidence}'")

    # Step 5 — Fallback if weak
    log.debug(f"[SEARCH] [5/5] Checking if fallback retrieval needed...")
    if confidence in ("low", "none") or not filtered:
        entities = extract_entities(query, manual_name)
        if entities:
            expanded_query = query + " " + " ".join(entities[:3])
            log.info(f"[SEARCH] ⚠️ [5/5] Low confidence — trying fallback with expanded query: '{expanded_query[:60]}'")
            try:
                expanded_embedding = embed_text(expanded_query)
                retry_results = collection.query(
                    query_embeddings=[expanded_embedding],
                    n_results=min(top_k, collection.count())
                )
                retry_docs = retry_results.get("documents", [[]])[0]
                retry_scored = [(doc, total_chunk_score(query, doc, manual_name)) for doc in retry_docs]
                retry_scored.sort(key=lambda x: x[1], reverse=True)
                retry_filtered = [(doc, s) for doc, s in retry_scored if s >= MIN_RELEVANCE_SCORE]

                if retry_filtered and len(retry_filtered) >= len(filtered):
                    filtered = retry_filtered
                    confidence = assess_confidence(filtered)
                    log.info(f"[SEARCH] ✅ [5/5] Fallback succeeded — {len(filtered)} chunk(s) | confidence='{confidence}'")
                else:
                    log.warning(f"[SEARCH] ⚠️ [5/5] Fallback did not improve results")
            except Exception as e:
                log.warning(f"[SEARCH] ⚠️ [5/5] Fallback embedding failed: {e}")
        else:
            log.debug("[SEARCH] [5/5] No domain entities to expand with — skipping fallback")

    # Last resort
    if not filtered:
        filtered = scored[:2]
        confidence = "low"
        log.warning(f"[SEARCH] ⚠️ No relevant chunks found — using top {len(filtered)} semantic result(s) as last resort")

    final_chunks = [doc for doc, _ in filtered[:3]]
    log.info(f"[SEARCH] ✅ Final result: {len(final_chunks)} chunk(s) | intent='{intent}' | confidence='{confidence}'")

    return final_chunks, confidence