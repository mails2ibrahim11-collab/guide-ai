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

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
log.info("[RAG] SentenceTransformer embedding model loaded (all-MiniLM-L6-v2)")

client_chroma = chromadb.PersistentClient(path="data/vectordb")
log.info("[RAG] ChromaDB client initialised at 'data/vectordb'")


# ================= EMBEDDING =================

def embed_text(text):
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
    "list_all":      r'\b(all|every|list|complete|entire|full list|everything)\b',
}


def detect_intent(query):
    q = query.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, q):
            log.debug(f"[INTENT] Detected intent: '{intent}'")
            return intent
    log.debug("[INTENT] No specific intent detected → 'general'")
    return "general"


def is_list_all_query(query):
    """Returns True if the query is asking for an exhaustive list of everything."""
    patterns = [
        r'\b(all|every|list all|list everything|complete list|full list)\b',
        r'\b(tell me all|show me all|give me all|what are all)\b',
        r'\b(everything about|all the|all his|all her|all their)\b',
    ]
    q = query.lower()
    return any(re.search(p, q) for p in patterns)


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

MIN_RELEVANCE_SCORE      = 1
CONFIDENCE_HIGH_THRESHOLD = 6
CONFIDENCE_MED_THRESHOLD  = 2

# If a collection has this many chunks or fewer, treat it as a small doc
# and return everything rather than filtering
SMALL_DOC_CHUNK_THRESHOLD = 20


def assess_confidence(chunks_with_scores):
    if not chunks_with_scores:
        return "none"
    best = max(s for _, s in chunks_with_scores)
    if best >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if best >= CONFIDENCE_MED_THRESHOLD:
        return "medium"
    return "low"


def is_uploaded_manual(manual_name):
    return manual_name not in DOMAIN_KEYWORDS


def is_small_document(collection):
    """
    Returns True if the document is small enough that we should return
    all chunks rather than filtering — avoids missing content in short docs
    like resumes, one-pagers, or brief product sheets.
    """
    return collection.count() <= SMALL_DOC_CHUNK_THRESHOLD


# ================= LOAD MANUAL =================

def collection_has_page_markers(collection):
    try:
        sample = collection.peek(limit=min(collection.count(), 5))
        docs = sample.get("documents", []) or []
        return any(re.search(r"\[Page\s+\d+\]", doc or "") for doc in docs)
    except Exception as e:
        log.warning(f"[LOAD] Could not inspect collection page markers: {e}")
        return True


def load_manual(manual_name, file_path):
    log.info(f"[LOAD] ── Starting load for '{manual_name}' ──")

    log.debug(f"[LOAD] [1/4] Accessing ChromaDB collection '{manual_name}'...")
    try:
        collection = client_chroma.get_or_create_collection(name=manual_name)
        log.debug(f"[LOAD] ✅ [1/4] Collection ready")
    except Exception as e:
        log.error(f"[LOAD] ❌ [1/4] Failed to create collection '{manual_name}': {e}")
        return

    if collection.count() > 0 and not collection_has_page_markers(collection):
        log.warning(f"[LOAD] Rebuilding '{manual_name}' so source links can target PDF pages")
        try:
            client_chroma.delete_collection(manual_name)
            collection = client_chroma.get_or_create_collection(name=manual_name)
        except Exception as e:
            log.error(f"[LOAD] Failed to rebuild collection '{manual_name}': {e}")
            return

    if collection.count() > 0:
        log.info(f"[LOAD] ✅ '{manual_name}' already has {collection.count()} chunks — skipping ingestion")
        return

    log.debug(f"[LOAD] [2/4] Checking file path '{file_path}'...")
    if not os.path.exists(file_path):
        log.error(f"[LOAD] ❌ [2/4] File not found: '{file_path}'")
        return
    log.debug(f"[LOAD] ✅ [2/4] File found")

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

    success = 0
    failed  = 0

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

SYNONYM_MAP = {
    "oily":    ["greasy", "fatty", "oil residue"],
    "greasy":  ["oily", "fatty", "grease"],
    "dirty":   ["soiled", "stained", "contaminated"],
    "broken":  ["faulty", "not working", "failed", "malfunction"],
    "stopped": ["not working", "failed", "ceased"],
    "noise":   ["sound", "rattling", "grinding", "vibration"],
    "noisy":   ["loud", "rattling", "vibrating"],
    "smell":   ["odour", "odor", "stench"],
    "hot":     ["overheating", "temperature", "heat"],
    "leak":    ["leaking", "water leak", "dripping"],
    "clog":    ["blocked", "obstruction", "jammed"],
    "worn":    ["degraded", "damaged", "deteriorated"],
    "fix":     ["repair", "resolve", "troubleshoot"],
    "error":   ["fault", "failure", "problem", "issue"],
    "plate":   ["dish", "crockery", "tableware"],
    "fork":    ["cutlery", "utensil", "silverware"],
    "knife":   ["cutlery", "utensil"],
    "spoon":   ["cutlery", "utensil"],
    "clothes": ["laundry", "garments", "fabric"],
    "shirt":   ["garment", "clothing", "fabric"],
    "load":    ["capacity", "weight", "drum"],
    "door":    ["hatch", "lid", "porthole"],
    "spray":   ["water jet", "nozzle", "arm"],
    "blocked": ["clogged", "obstructed", "jammed"],
    "strange": ["unusual", "abnormal", "unexpected"],
    "sound":   ["noise", "rattling", "grinding"],
}


def expand_query_with_synonyms(query):
    words  = re.sub(r"[^a-z0-9 ]", " ", query.lower()).split()
    extras = []
    seen   = set(words)
    for word in words:
        for syn in SYNONYM_MAP.get(word, []):
            if syn not in seen:
                extras.append(syn)
                seen.add(syn)
    if extras:
        expanded = query + " " + " ".join(extras[:8])
        log.debug(f"[SEARCH] Synonym expansion: +{extras[:8]}")
        return expanded
    return query


def search_manual(query, manual_name, top_k=8):
    query = expand_query_with_synonyms(query)
    log.info(f"[SEARCH] ── Query for '{manual_name}' ──")
    log.debug(f"[SEARCH] Query: '{query[:80]}{'...' if len(query) > 80 else ''}'" )

    # Step 1 — Access collection
    log.debug(f"[SEARCH] [1/5] Accessing collection '{manual_name}'...")
    try:
        collection = client_chroma.get_or_create_collection(name=manual_name)
    except Exception as e:
        log.error(f"[SEARCH] ❌ [1/5] Cannot access collection: {e}")
        return [], "none"

    total_chunks = collection.count()
    if total_chunks == 0:
        log.error(f"[SEARCH] ❌ [1/5] Collection '{manual_name}' is empty — was it loaded?")
        return [], "none"

    log.debug(f"[SEARCH] ✅ [1/5] Collection has {total_chunks} chunks")

    # Step 2 — Intent detection
    intent      = detect_intent(query)
    list_query  = is_list_all_query(query)
    log.debug(f"[SEARCH] [2/5] Intent: '{intent}' | list_all={list_query}")

    # ── SMALL DOCUMENT FAST PATH ─────────────────────────────────────────────
    # If the document is short (resume, one-pager, brief spec sheet) OR the
    # query is asking for an exhaustive list, return ALL chunks so nothing
    # gets filtered out by hybrid scoring.
    small_doc = total_chunks <= SMALL_DOC_CHUNK_THRESHOLD
    if small_doc or list_query:
        try:
            all_results = collection.query(
                query_embeddings=[embed_text(query)],
                n_results=total_chunks
            )
            all_docs = all_results.get("documents", [[]])[0]
            log.info(
                f"[SEARCH] ✅ {'Small doc' if small_doc else 'List-all query'} "
                f"— returning all {len(all_docs)} chunks, no filtering"
            )
            return all_docs, "high"
        except Exception as e:
            log.error(f"[SEARCH] ❌ Small-doc/list-all fetch failed: {e}")
            # fall through to normal path

    # Step 3 — Semantic search
    # For larger docs bump top_k to catch more candidates before scoring
    effective_top_k = min(max(top_k, 12), total_chunks)
    log.debug(f"[SEARCH] [3/5] Semantic search (top_k={effective_top_k})...")
    try:
        query_embedding = embed_text(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_top_k
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

    # For uploaded manuals with all-zero hybrid scores, trust semantic search
    all_zero = all(s == 0 for _, s in scored)
    if all_zero and is_uploaded_manual(manual_name):
        final_chunks = [doc for doc, _ in scored[:5]]
        log.info(f"[SEARCH] ✅ Uploaded manual — all hybrid scores zero, using top 5 semantic results")
        return final_chunks, "medium"

    filtered   = [(doc, s) for doc, s in scored if s >= MIN_RELEVANCE_SCORE]
    confidence = assess_confidence(filtered if filtered else scored)
    log.info(f"[SEARCH] ✅ [4/5] After filtering: {len(filtered)} chunk(s) | confidence='{confidence}'")

    # Step 5 — Fallback if weak
    log.debug(f"[SEARCH] [5/5] Checking if fallback retrieval needed...")
    if confidence in ("low", "none") or not filtered:
        entities = extract_entities(query, manual_name)
        if entities:
            expanded_query = query + " " + " ".join(entities[:3])
            log.info(f"[SEARCH] ⚠️ [5/5] Low confidence — fallback: '{expanded_query[:60]}'")
            try:
                expanded_embedding = embed_text(expanded_query)
                retry_results = collection.query(
                    query_embeddings=[expanded_embedding],
                    n_results=min(effective_top_k, total_chunks)
                )
                retry_docs    = retry_results.get("documents", [[]])[0]
                retry_scored  = [(doc, total_chunk_score(query, doc, manual_name)) for doc in retry_docs]
                retry_scored.sort(key=lambda x: x[1], reverse=True)
                retry_filtered = [(doc, s) for doc, s in retry_scored if s >= MIN_RELEVANCE_SCORE]

                if retry_filtered and len(retry_filtered) >= len(filtered):
                    filtered   = retry_filtered
                    confidence = assess_confidence(filtered)
                    log.info(f"[SEARCH] ✅ [5/5] Fallback succeeded — {len(filtered)} chunk(s)")
                else:
                    log.warning(f"[SEARCH] ⚠️ [5/5] Fallback did not improve results")
            except Exception as e:
                log.warning(f"[SEARCH] ⚠️ [5/5] Fallback embedding failed: {e}")
        else:
            log.debug("[SEARCH] [5/5] No domain entities — skipping fallback")

    # Last resort
    if not filtered:
        filtered   = scored[:3]
        confidence = "low"
        log.warning(f"[SEARCH] ⚠️ No relevant chunks — using top {len(filtered)} semantic results as last resort")

    # Return up to 5 chunks for larger docs (was 3 — more context = better answers)
    final_chunks = [doc for doc, _ in filtered[:5]]
    log.info(f"[SEARCH] ✅ Final result: {len(final_chunks)} chunk(s) | intent='{intent}' | confidence='{confidence}'")

    return final_chunks, confidence