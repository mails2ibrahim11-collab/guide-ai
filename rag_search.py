import chromadb
import os
import re
import time
from dotenv import load_dotenv
from google import genai
from extract_pdf import extract_text_from_pdf, chunk_text

load_dotenv()

client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
client_chroma = chromadb.PersistentClient(path="data/vectordb")

# ================= EMBEDDING TOKEN SAFETY =================
# Gemini embedding-001 limit is ~2048 tokens (~1500 words).
# We hard-cap input to 800 words before embedding to be safe.
MAX_EMBED_WORDS = 800


def safe_truncate(text):
    """Truncate text to MAX_EMBED_WORDS words before sending to embedding API."""
    words = text.split()
    if len(words) > MAX_EMBED_WORDS:
        return " ".join(words[:MAX_EMBED_WORDS])
    return text


# ================= DOMAIN KEYWORDS (STRICT ISOLATION) =================
# Used to boost domain-relevant chunks and penalize cross-manual chunks.

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
            return intent
    return "general"


# ================= ENTITY EXTRACTION =================

def extract_entities(query, manual_name):
    """Extract appliance-specific terms mentioned in the query."""
    domain_words = DOMAIN_KEYWORDS.get(manual_name, [])
    q_lower = query.lower()
    return [word for word in domain_words if word in q_lower]


# ================= EMBEDDING =================

def embed_text(text):
    """
    Embed text using Gemini embedding-001.
    Always truncates first to avoid token limit errors.
    Separate retry logic for:
    - Token/size errors: progressively halve the text (up to 2 halvings)
    - Transient API errors: exponential backoff (up to 3 retries)
    """
    text = safe_truncate(text)

    # Handle token size errors with up to 2 halvings
    for halving in range(3):
        try:
            result = client_genai.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return result.embeddings[0].values

        except Exception as e:
            error_str = str(e).lower()

            if any(kw in error_str for kw in ["token", "size", "too large", "exceed", "limit"]):
                words = text.split()
                if len(words) <= 10:
                    print(f"❌ Text too short to halve further. Giving up.")
                    raise
                text = " ".join(words[:max(len(words) // 2, 10)])
                print(f"⚠️ Token limit hit. Halving to {len(text.split())} words (attempt {halving+1}/3)...")
                continue

            # Transient error — exponential backoff
            if halving < 2:
                wait = 2 ** halving
                print(f"⚠️ Embedding failed (attempt {halving+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"❌ Embedding failed after all retries: {e}")
                raise

    raise RuntimeError("Embedding failed after all retries")


# ================= SCORING =================

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 ]', ' ', text.lower())


def keyword_score(query, doc):
    q_words = {w for w in clean_text(query).split() if len(w) > 2}
    d_text = clean_text(doc)
    return sum(1 for w in q_words if w in d_text)


def domain_relevance_score(doc, manual_name):
    """
    +1 for each own-domain keyword found in chunk.
    -2 for each other-domain keyword found (strict isolation penalty).
    """
    doc_lower = doc.lower()
    own_keywords = DOMAIN_KEYWORDS.get(manual_name, [])
    own_score = sum(1 for kw in own_keywords if kw in doc_lower)

    penalty = 0
    for other_manual, other_keywords in DOMAIN_KEYWORDS.items():
        if other_manual != manual_name:
            penalty += sum(1 for kw in other_keywords if kw in doc_lower)

    return own_score - (penalty * 2)


def total_chunk_score(query, doc, manual_name):
    return (keyword_score(query, doc) * 2) + domain_relevance_score(doc, manual_name)


# ================= RELEVANCE THRESHOLDS =================

MIN_RELEVANCE_SCORE = 1    # chunks below this score are rejected
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
    """
    Load a PDF into its own isolated ChromaDB collection.
    Each manual is a completely separate collection — zero cross-contamination.
    Skips loading if collection already has chunks.
    """
    try:
        collection = client_chroma.get_or_create_collection(name=manual_name)
    except Exception as e:
        print(f"❌ Failed to create collection {manual_name}: {e}")
        return

    if collection.count() > 0:
        print(f"✅ {manual_name} already loaded ({collection.count()} chunks) — skipping")
        return

    print(f"📄 Loading {manual_name} from {file_path}...")

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    text = extract_text_from_pdf(file_path)
    if not text.strip():
        print(f"❌ No text extracted from {file_path}")
        return

    chunks = chunk_text(text)
    print(f"   Embedding {len(chunks)} chunks into ChromaDB...")

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

            # Small delay every 10 chunks to avoid rate limiting
            if i > 0 and i % 10 == 0:
                time.sleep(0.5)
                print(f"   Progress: {i}/{len(chunks)} chunks embedded...")

        except Exception as e:
            print(f"   ⚠️ Skipping chunk {i}: {e}")
            failed += 1
            continue

    print(f"✅ {manual_name} done — {success} embedded, {failed} skipped")


# ================= SEARCH =================

def search_manual(query, manual_name, top_k=8):
    """
    Multi-step retrieval pipeline:
    Step 1 — Semantic search in the correct manual's collection only.
    Step 2 — Score and rank by keyword + domain relevance.
    Step 3 — Filter by relevance threshold.
    Step 4 — Fallback: retry with domain-expanded query if results are weak.
    Step 5 — Last resort: return top semantic results even if low scoring.

    Returns: (list_of_chunks, confidence_level)
    confidence_level: 'high' | 'medium' | 'low' | 'none'
    """
    try:
        collection = client_chroma.get_or_create_collection(name=manual_name)
    except Exception as e:
        print(f"❌ Cannot access collection {manual_name}: {e}")
        return [], "none"

    if collection.count() == 0:
        print(f"⚠️ Collection '{manual_name}' is empty — was it loaded?")
        return [], "none"

    intent = detect_intent(query)

    # --- STEP 1: Primary semantic search ---
    try:
        query_embedding = embed_text(query)
    except Exception as e:
        print(f"❌ Query embedding failed: {e}")
        return [], "none"

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )
    docs = results.get("documents", [[]])[0]

    if not docs:
        return [], "none"

    # --- STEP 2: Score and rank ---
    scored = [(doc, total_chunk_score(query, doc, manual_name)) for doc in docs]
    scored.sort(key=lambda x: x[1], reverse=True)

    # --- STEP 3: Filter by threshold ---
    filtered = [(doc, s) for doc, s in scored if s >= MIN_RELEVANCE_SCORE]
    confidence = assess_confidence(filtered if filtered else scored)

    # --- STEP 4: Fallback with expanded query ---
    if confidence in ("low", "none") or not filtered:
        entities = extract_entities(query, manual_name)
        if entities:
            expanded_query = query + " " + " ".join(entities[:3])
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
                    print(f"🔄 Fallback retrieval: {len(filtered)} chunks, confidence={confidence}")
            except Exception as e:
                print(f"⚠️ Fallback embedding failed: {e}")

    # --- STEP 5: Last resort — top semantic results ---
    if not filtered:
        filtered = scored[:2]
        confidence = "low"
        print(f"⚠️ Using top semantic results (low confidence)")

    final_chunks = [doc for doc, _ in filtered[:3]]
    print(f"🔍 [{manual_name}] intent={intent} | chunks={len(final_chunks)} | confidence={confidence}")

    return final_chunks, confidence