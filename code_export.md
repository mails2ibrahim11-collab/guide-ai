# GuideAI — Full Code Export
> Generated: 2026-05-05  
> Branch: main | Last commit: 2372875 — refactor: remove dead code, fix efficiency issues, clean unused files

---

## Project Overview

Flask-SocketIO web app (GuideAI) that acts as an AI-powered support assistant.  
Customers chat or call in; a RAG pipeline searches PDF manuals; Groq LLaMA 3.3 70B generates answers.  
An agent monitors live calls and receives real-time AI suggestions.

**Stack:**
- Flask + Flask-SocketIO (eventlet) — web server
- ChromaDB + SentenceTransformer (all-MiniLM-L6-v2) — vector search / RAG
- Groq API — LLaMA 3.3 70B for answers, Whisper large-v3 for transcription
- PyMuPDF (fitz) + Tesseract — PDF extraction / OCR
- SQLite (via database.py) — users, sessions, chats, calls, reports
- LiveKit — WebRTC audio for live calls

**Key files:**
- `main.py` — Flask routes + SocketIO handlers
- `database.py` — all SQLite operations
- `llm_suggestions.py` — Groq LLM calls (answer gen, scoring, reports)
- `rag_search.py` — ChromaDB vector search pipeline
- `extract_pdf.py` — PDF text extraction + chunking
- `logger.py` — shared logger

---

## logger.py

```python
import logging
import sys
from datetime import datetime

LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)-20s  %(message)s"
DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

# Log level guide:
#   log.debug(...)    → internal step details
#   log.info(...)     → normal checkpoints
#   log.warning(...)  → recoverable issues
#   log.error(...)    → failures that returned an error response
#   log.critical(...) → startup failures
```

---

## extract_pdf.py

```python
import fitz  # PyMuPDF
import re
import io
import pytesseract
from PIL import Image
from logger import get_logger

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

log = get_logger("extract_pdf")


# ================= OCR FALLBACK =================

def extract_text_from_page(page, page_num):
    text = page.get_text()
    if len(text.strip()) > 20:
        return text

    log.debug(f"[OCR] Page {page_num} is image-based — running OCR")
    try:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img)
    except Exception:
        log.warning("[OCR] ⚠️ pytesseract not available — skipping OCR for this page")
        return ""


# ================= EXTRACT =================

def extract_text_from_pdf(file_path):
    log.info(f"[PDF] Opening '{file_path}'...")
    text = ""
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        log.debug(f"[PDF] Document has {total_pages} page(s)")

        for i, page in enumerate(doc):
            page_text = extract_text_from_page(page, i + 1)
            text += f"\n[Page {i + 1}]\n{page_text}\n"
            log.debug(f"[PDF] Page {i+1}/{total_pages} — {len(page_text)} chars extracted")

        log.info(f"[PDF] ✅ Extraction complete — {len(text)} total characters from {total_pages} page(s)")

    except Exception as e:
        log.error(f"[PDF] ❌ Failed to read '{file_path}': {e}")

    return text


# ================= SECTION DETECTION =================

SECTION_PATTERNS = [
    r'^\d+[\.\s]+[A-Z][^\n]{3,60}$',
    r'^[A-Z][A-Z\s]{4,40}$',
    r'^\s*(Chapter|Section|Part)\s+\d+',
]


def is_heading(line):
    line = line.strip()
    if not line or len(line) > 80:
        return False
    for pattern in SECTION_PATTERNS:
        if re.match(pattern, line):
            return True
    if line.isupper() and 4 < len(line) < 60:
        return True
    return False


# ================= CHUNK SETTINGS =================

CHUNK_SIZE = 200
OVERLAP = 40


def split_into_word_chunks(text, prefix=""):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + CHUNK_SIZE]
        chunk = (prefix + " ".join(chunk_words)).strip()
        if chunk:
            chunks.append(chunk)
        i += (CHUNK_SIZE - OVERLAP)
    return chunks


# ================= SECTION-AWARE CHUNKING =================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    log.info(f"[CHUNK] Starting chunking ({len(text)} chars, chunk_size={chunk_size}, overlap={overlap})...")

    page_blocks = re.findall(r"\[Page\s+(\d+)\]\s*(.*?)(?=\n\[Page\s+\d+\]|\Z)", text, flags=re.S)
    if page_blocks:
        all_chunks = []
        for page_num, page_body in page_blocks:
            page_body = page_body.strip()
            if not page_body:
                continue
            page_chunks = split_into_word_chunks(page_body, prefix=f"[Page {page_num}] ")
            all_chunks.extend(page_chunks)
            log.debug(f"[CHUNK] Page {page_num} → {len(page_chunks)} chunk(s)")

        if all_chunks:
            log.info(f"[CHUNK] ✅ Page-aware complete — {len(page_blocks)} page(s) → {len(all_chunks)} chunks")
            return all_chunks

    lines = text.split('\n')
    log.debug(f"[CHUNK] Scanning {len(lines)} lines for section headings...")

    sections = []
    current_heading = "General"
    current_body = []
    headings_found = 0

    for line in lines:
        if is_heading(line):
            if current_body:
                sections.append((current_heading, " ".join(current_body)))
            current_heading = line.strip()
            current_body = []
            headings_found += 1
        else:
            stripped = line.strip()
            if stripped:
                current_body.append(stripped)

    if current_body:
        sections.append((current_heading, " ".join(current_body)))

    log.debug(f"[CHUNK] Detected {headings_found} heading(s) → {len(sections)} section(s)")

    if len(sections) > 3:
        log.info(f"[CHUNK] Using section-aware chunking ({len(sections)} sections)")
        all_chunks = []
        for heading, body in sections:
            prefix = f"[{heading}] "
            section_chunks = split_into_word_chunks(body, prefix=prefix)
            all_chunks.extend(section_chunks)
            log.debug(f"[CHUNK] Section '{heading[:40]}' → {len(section_chunks)} chunk(s)")

        log.info(f"[CHUNK] ✅ Section-aware complete — {len(sections)} sections → {len(all_chunks)} chunks")
        return all_chunks

    log.warning(f"[CHUNK] ⚠️ Only {len(sections)} section(s) detected — falling back to sliding window")
    all_chunks = split_into_word_chunks(text)
    log.info(f"[CHUNK] ✅ Sliding window complete — {len(all_chunks)} chunks")
    return all_chunks
```

---

## rag_search.py

```python
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

MIN_RELEVANCE_SCORE       = 1
CONFIDENCE_HIGH_THRESHOLD = 6
CONFIDENCE_MED_THRESHOLD  = 2
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

    try:
        collection = client_chroma.get_or_create_collection(name=manual_name)
    except Exception as e:
        log.error(f"[LOAD] ❌ Failed to create collection '{manual_name}': {e}")
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

    if not os.path.exists(file_path):
        log.error(f"[LOAD] ❌ File not found: '{file_path}'")
        return

    text = extract_text_from_pdf(file_path)
    if not text.strip():
        log.error(f"[LOAD] ❌ No text extracted from '{file_path}'")
        return
    log.info(f"[LOAD] ✅ Extracted {len(text)} characters")

    chunks = chunk_text(text)

    if chunks and manual_name not in DOMAIN_KEYWORDS:
        keywords = extract_dynamic_keywords(text)
        DYNAMIC_KEYWORDS[manual_name] = keywords
        log.info(f"[LOAD] ✅ Dynamic keywords extracted for '{manual_name}': {keywords[:5]}...")

    log.info(f"[LOAD] Embedding {len(chunks)} chunks into ChromaDB...")

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

    log.info(f"[LOAD] ✅ '{manual_name}' done — {success} embedded, {failed} skipped")


# ================= SEARCH =================

def search_manual(query, manual_name, top_k=8):
    log.info(f"[SEARCH] ── Query for '{manual_name}' ──")
    log.debug(f"[SEARCH] Query: '{query[:60]}{'...' if len(query) > 60 else ''}'")

    try:
        collection = client_chroma.get_or_create_collection(name=manual_name)
    except Exception as e:
        log.error(f"[SEARCH] ❌ Cannot access collection: {e}")
        return [], "none"

    total_chunks = collection.count()
    if total_chunks == 0:
        log.error(f"[SEARCH] ❌ Collection '{manual_name}' is empty — was it loaded?")
        return [], "none"

    log.debug(f"[SEARCH] ✅ Collection has {total_chunks} chunks")

    intent     = detect_intent(query)
    list_query = is_list_all_query(query)
    log.debug(f"[SEARCH] Intent: '{intent}' | list_all={list_query}")

    # Small doc or exhaustive list — return all chunks, skip filtering
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

    effective_top_k = min(max(top_k, 12), total_chunks)
    log.debug(f"[SEARCH] Semantic search (top_k={effective_top_k})...")
    try:
        query_embedding = embed_text(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_top_k
        )
        docs = results.get("documents", [[]])[0]
        log.info(f"[SEARCH] ✅ Semantic search returned {len(docs)} candidate chunk(s)")
    except Exception as e:
        log.error(f"[SEARCH] ❌ Query embedding or search FAILED: {e}")
        return [], "none"

    if not docs:
        log.warning(f"[SEARCH] ⚠️ No documents returned from ChromaDB")
        return [], "none"

    log.debug(f"[SEARCH] Scoring and filtering chunks...")
    scored = [(doc, total_chunk_score(query, doc, manual_name)) for doc in docs]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_scores = [s for _, s in scored[:3]]
    log.debug(f"[SEARCH] Top 3 chunk scores: {top_scores}")

    all_zero = all(s == 0 for _, s in scored)
    if all_zero and is_uploaded_manual(manual_name):
        final_chunks = [doc for doc, _ in scored[:5]]
        log.info(f"[SEARCH] ✅ Uploaded manual — all hybrid scores zero, using top 5 semantic results")
        return final_chunks, "medium"

    filtered   = [(doc, s) for doc, s in scored if s >= MIN_RELEVANCE_SCORE]
    confidence = assess_confidence(filtered if filtered else scored)
    log.info(f"[SEARCH] ✅ After filtering: {len(filtered)} chunk(s) | confidence='{confidence}'")

    # Fallback with entity expansion if confidence is low
    log.debug(f"[SEARCH] Checking if fallback retrieval needed...")
    if confidence in ("low", "none") or not filtered:
        entities = extract_entities(query, manual_name)
        if entities:
            expanded_query = query + " " + " ".join(entities[:3])
            log.info(f"[SEARCH] ⚠️ Low confidence — fallback: '{expanded_query[:60]}'")
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
                    log.info(f"[SEARCH] ✅ Fallback succeeded — {len(filtered)} chunk(s)")
                else:
                    log.warning(f"[SEARCH] ⚠️ Fallback did not improve results")
            except Exception as e:
                log.warning(f"[SEARCH] ⚠️ Fallback embedding failed: {e}")
        else:
            log.debug("[SEARCH] No domain entities — skipping fallback")

    if not filtered:
        filtered   = scored[:3]
        confidence = "low"
        log.warning(f"[SEARCH] ⚠️ No relevant chunks — using top {len(filtered)} semantic results as last resort")

    final_chunks = [doc for doc, _ in filtered[:5]]
    log.info(f"[SEARCH] ✅ Final result: {len(final_chunks)} chunk(s) | intent='{intent}' | confidence='{confidence}'")

    return final_chunks, confidence
```

---

## llm_suggestions.py

```python
import os
import re
import json
from dotenv import load_dotenv
from groq import Groq
from logger import get_logger

load_dotenv(override=True)

log = get_logger("llm_suggestions")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

log.info(f"[LLM] Groq client initialised — model: {GROQ_MODEL}")

HARDCODED_MANUAL_KEYS = {"dishwasher_manual", "washing_machine_manual"}


# ================= ADAPTIVE TONE =================

def get_adaptive_instructions(session_score):
    if session_score is None:
        return "Be clear, thorough and helpful."
    if session_score >= 8:
        return (
            "Previous answers were rated excellent. "
            "Be concise and confident. Keep answers direct and focused."
        )
    elif session_score >= 6:
        return (
            "Previous answers were rated good. "
            "Stay clear and structured. If anything is ambiguous, ask for clarification."
        )
    elif session_score >= 4:
        return (
            "Previous answers were rated average. Be extra careful and thorough. "
            "Use numbered steps wherever possible. Double-check your answer against the context."
        )
    else:
        return (
            "Previous answers were rated poorly. Take extra care. "
            "Use step-by-step formatting. "
            "If the context does not clearly answer the question, say so and ask the user to clarify. "
            "Do NOT guess."
        )


# ================= ANSWER GENERATION =================

def generate_answer(query, context, history=None, manual_name=None,
                    confidence="high", session_score=None):

    manual_readable = manual_name.replace("_", " ").title() if manual_name else "this document"
    is_hardcoded    = manual_name in HARDCODED_MANUAL_KEYS

    if not context or confidence == "none":
        return (
            f"I couldn't find relevant information in the {manual_readable} document "
            "for your question. Could you rephrase it or be more specific?"
        )

    adaptive_instruction = get_adaptive_instructions(session_score)
    combined_context     = "\n\n---\n\n".join(context)

    if confidence == "low":
        confidence_note = (
            "⚠️ WARNING: Retrieved context has LOW relevance. "
            "Be honest — if the context doesn't clearly answer the question, say so."
        )
    elif confidence == "medium":
        confidence_note = (
            "The retrieved context is partially relevant. "
            "Answer what you can. Clearly flag anything you are inferring."
        )
    else:
        confidence_note = "The retrieved context is highly relevant. Answer confidently."

    history_str = ""
    if history:
        recent      = history[-6:]
        history_str = "\n".join(
            f"{sender.upper()}: {message}"
            for sender, message, _ in recent
        )

    output_format = """
OUTPUT FORMAT — MANDATORY:
- Use bullet points (•) for lists of facts, options, or features.
- Use numbered steps (1. 2. 3.) for any procedure, sequence, or troubleshooting.
- Use **bold** for component names, settings, warnings, error codes, or key values.
- Keep each bullet/step to 1–2 clear sentences.
- Use sub-bullets (  –) only when genuinely needed for clarity.
- NO filler phrases like "Sure!", "Great question!", "Of course!" — start directly with the answer.
- NO vague statements like "it depends" without immediately explaining what it depends on.
- Be specific: include exact names, values, temperatures, durations, settings from the manual.
- If the manual gives a specific error code, cycle name, or part number — use it.
- End with one short follow-up offer only if the answer might need clarification.
"""

    list_all_patterns = [
        r'\b(all|every|list all|list everything|complete list|full list)\b',
        r'\b(tell me all|show me all|give me all|what are all)\b',
        r'\b(everything about|all the|all his|all her|all their)\b',
    ]
    is_list_all = any(re.search(p, query.lower()) for p in list_all_patterns)
    exhaustive_note = (
        "\nIMPORTANT: The user is asking for a COMPLETE, EXHAUSTIVE list. "
        "Scan every part of the provided context carefully. "
        "Do NOT stop at the first item you find. "
        "Include ALL instances mentioned anywhere in the context, even if they appear in different sections.\n"
    ) if is_list_all else ""

    if is_hardcoded:
        prompt = f"""You are an expert support assistant for a {manual_readable}. You have deep knowledge of this specific appliance.

ADAPTIVE BEHAVIOR:
{adaptive_instruction}

RETRIEVAL CONFIDENCE:
{confidence_note}
{exhaustive_note}
STRICT RULES:
- Answer ONLY using the manual context provided. Do not invent or assume.
- Reference specific parts, settings, cycle names, error codes exactly as they appear in the manual.
- If a step has a specific duration, temperature, or setting value — include it.
- Do NOT mix information from other appliances.
- If the answer is not in the context, say clearly: "The manual doesn't cover this specifically" and suggest what the user could try or check.

INTENT HANDLING:
- "where is X" → describe exact physical location (e.g. "bottom of the tub, behind the lower spray arm")
- "how do I" → give precise numbered steps with exact settings/values
- "error / not working / fault" → identify the specific fault first, then give step-by-step resolution
- "what is / why" → give a concise, accurate explanation using manual terminology
- Unclear query → ask ONE focused clarifying question before answering
{output_format}
MANUAL CONTEXT:
{combined_context}

RECENT CONVERSATION:
{history_str if history_str else "No prior conversation."}

USER QUESTION:
{query}

Answer:"""

    else:
        prompt = f"""You are a knowledgeable assistant answering questions based on the document "{manual_readable}".

ADAPTIVE BEHAVIOR:
{adaptive_instruction}

RETRIEVAL CONFIDENCE:
{confidence_note}
{exhaustive_note}
STRICT RULES:
- Answer ONLY using the document context below. Do not invent or assume.
- Quote specific values, names, steps, or terms exactly as they appear in the document.
- If the answer is not clearly in the context, say so honestly and specifically.
- Never give a vague or generic answer when the document has specific information.
{output_format}
DOCUMENT CONTEXT:
{combined_context}

RECENT CONVERSATION:
{history_str if history_str else "No prior conversation."}

USER QUESTION:
{query}

Answer:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.4
        )
        answer = response.choices[0].message.content.strip()
        log.info(f"[GENERATE] ✅ Groq responded ({len(answer)} chars)")
        return answer
    except Exception as e:
        log.error(f"[GENERATE] ❌ Groq API call FAILED: {e}")
        return "The AI assistant is temporarily unavailable. Please try again in a moment."


# ================= SELF-EVALUATION =================

def analyze_satisfaction(answer, query=None, context_confidence=None, is_uploaded=False):
    if context_confidence == "none":
        return 2

    cant_find_phrases = [
        "couldn't find", "not in the manual", "not in the document",
        "unable to find", "no relevant", "could you clarify", "could you rephrase"
    ]
    if any(p in answer.lower() for p in cant_find_phrases):
        return 4

    confidence_note = "Note: low-confidence context.\n" if context_confidence == "low" else ""

    rubric = """Scoring guide:
10 = Perfect: precise, complete, directly answers the question in clear bullets
8-9 = Excellent: very helpful, minor gaps
6-7 = Good: helpful but missing some detail
4-5 = Adequate: partially answers
2-3 = Weak: vague, off-topic, or incomplete
1   = Poor: wrong or made things up"""

    prompt = f"""Rate the quality of this AI-generated answer from 1 to 10.

{confidence_note}{rubric}

Answer to rate:
{answer}

Reply with ONLY a single integer from 1 to 10. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        match = re.search(r'\d+', response.choices[0].message.content.strip())
        if match:
            score = max(1, min(int(match.group()), 10))
            log.info(f"[SCORE] ✅ Self-eval score = {score}/10")
            return score
        return 5
    except Exception as e:
        log.error(f"[SCORE] ❌ Scoring FAILED: {e}")
        return 5


# ================= CONVERSATION SENTIMENT =================

def analyze_conversation_sentiment(chat_history):
    if not chat_history or len(chat_history) < 4:
        return None

    history_text = "\n".join(
        f"{sender}: {message}"
        for sender, message, _ in chat_history[-10:]
    )

    prompt = f"""Analyze this support conversation and rate the user's overall satisfaction from 1 to 10.

1  = Very frustrated, problem not solved
5  = Neutral — partially helped
10 = Clearly satisfied, problem resolved

Conversation:
{history_text}

Reply with ONLY a single integer from 1 to 10. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        match = re.search(r'\d+', response.choices[0].message.content.strip())
        if match:
            score = max(1, min(int(match.group()), 10))
            log.info(f"[SENTIMENT] ✅ Sentiment score = {score}/10")
            return score
        return None
    except Exception as e:
        log.error(f"[SENTIMENT] ❌ Sentiment analysis FAILED: {e}")
        return None


# ================= CALL GRADING =================

def grade_agent_turn(customer_query, ai_suggestion, agent_actual_response, manual_name):
    if not agent_actual_response or not agent_actual_response.strip():
        return 3

    manual_readable = manual_name.replace("_", " ").title() if manual_name else "the document"

    prompt = f"""You are grading a support agent's response during a live call.

The agent had access to an AI suggestion based on the {manual_readable}.

CUSTOMER QUERY:
{customer_query}

AI SUGGESTED ANSWER:
{ai_suggestion if ai_suggestion else "No suggestion available."}

AGENT'S ACTUAL RESPONSE:
{agent_actual_response}

Grade 1-10 based on:
- Accuracy — correct information?
- Completeness — fully addressed the question?
- Clarity — easy to understand?
- Use of AI suggestion — appropriately used or improved on it?

Scoring:
10 = Perfect  |  8-9 = Excellent  |  6-7 = Good
4-5 = Adequate  |  2-3 = Weak  |  1 = Poor/wrong

Reply with ONLY a single integer from 1 to 10. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        match = re.search(r'\d+', response.choices[0].message.content.strip())
        if match:
            score = max(1, min(int(match.group()), 10))
            log.info(f"[CALL_GRADE] ✅ Turn score = {score}/10")
            return score
        return 5
    except Exception as e:
        log.error(f"[CALL_GRADE] ❌ Grading FAILED: {e}")
        return 5


# ================= CALL SUMMARY (for customer) =================

def generate_call_summary(turns, manual_name):
    if not turns:
        return "Your support session has ended. Thank you for reaching out."

    manual_readable = manual_name.replace("_", " ").title() if manual_name else "the document"

    transcript_lines = []
    for t in turns:
        speaker = "Customer" if t["speaker"] == "customer" else "Agent"
        text    = t["edited_text"] or t["original_text"]
        transcript_lines.append(f"{speaker}: {text}")
    transcript_text = "\n".join(transcript_lines[-10:])

    prompt = f"""You are summarising a support call for the customer.

Manual / product discussed: {manual_readable}

TRANSCRIPT (last portion):
{transcript_text}

Write a friendly 3-4 sentence summary for the customer covering:
- What issue or question they raised
- What was resolved or explained
- Any action they should take next (if applicable)

Keep it warm, concise, and in plain language. No bullet points. No technical jargon.
Do NOT mention scores, ratings, or the agent's name."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4
        )
        summary = response.choices[0].message.content.strip()
        log.info(f"[SUMMARY] ✅ Summary generated ({len(summary)} chars)")
        return summary
    except Exception as e:
        log.error(f"[SUMMARY] ❌ Summary generation FAILED: {e}")
        return "Your support session has ended. Thank you for contacting us."


# ================= CALL REPORT GENERATION =================

def generate_call_report(call_id, agent, manual_name, turns, overall_score, customer_rating):
    if not turns:
        return "No conversation data available for this call."

    manual_readable = manual_name.replace("_", " ").title() if manual_name else "Unknown"

    transcript_lines = []
    for t in turns:
        speaker = t["speaker"].upper()
        text    = t["edited_text"] or t["original_text"]
        transcript_lines.append(f"{speaker}: {text}")
    transcript_text = "\n".join(transcript_lines)

    total_turns = len([t for t in turns if t["speaker"] == "customer"])
    used_as_is  = len([t for t in turns if t.get("agent_used_ai") == 1])
    edited      = len([t for t in turns if t.get("agent_used_ai") == 2])
    ignored     = len([t for t in turns if t.get("agent_used_ai") == 0])

    turn_scores    = [t["turn_score"] for t in turns if t.get("turn_score") is not None]
    avg_turn_score = round(sum(turn_scores) / len(turn_scores), 2) if turn_scores else 0

    rating_line = (
        f"Customer rating: {customer_rating}/10"
        if customer_rating and customer_rating > 0
        else "Customer rating: Not provided"
    )

    prompt = f"""You are generating a professional end-of-call report for a support agent.

CALL DETAILS:
- Call ID: {call_id}
- Agent: {agent}
- Manual used: {manual_readable}
- Overall AI score: {overall_score}/10
- {rating_line}
- Average turn score: {avg_turn_score}/10
- Total customer turns: {total_turns}
- Used AI suggestion as-is: {used_as_is}
- Edited AI suggestion: {edited}
- Ignored AI suggestion: {ignored}

FULL TRANSCRIPT:
{transcript_text}

Write a professional call report with these sections:
1. CALL SUMMARY — 2-3 sentences on what the call was about and overall outcome
2. AGENT PERFORMANCE — how well the agent handled the call and used AI assistance
3. KEY MOMENTS — 2-3 specific moments from the transcript (good or needs improvement)
4. AREAS FOR IMPROVEMENT — specific, actionable suggestions
5. OVERALL VERDICT — one sentence final assessment

Tone: professional but constructive. Reference actual transcript moments.
Use clear section headers. Write in paragraphs, not bullet points."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.4
        )
        report = response.choices[0].message.content.strip()
        log.info(f"[REPORT] ✅ Report generated ({len(report)} chars)")
        return report
    except Exception as e:
        log.error(f"[REPORT] ❌ Report generation FAILED: {e}")
        return f"Report generation failed. Call score: {overall_score}/10. {rating_line}."
```

---

## database.py

```python
import sqlite3
import hashlib
import uuid
from datetime import datetime

DB_PATH = "data/database.db"

def connect():
    return sqlite3.connect(DB_PATH)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_now():
    return datetime.now().isoformat()

# ── Schema / init ────────────────────────────────────────────────────────────
# Tables: users, sessions, chats, manuals, calls, call_turns, call_reports,
#         citation_feedback
# See full init_db() for CREATE TABLE statements and migration ALTER TABLEs.

def init_db(): ...          # creates all tables + migrations
def register_user(...): ... # INSERT user
def login_user(...): ...    # returns role string or None
def get_user_role(...): ... # returns role or None

def create_session(...): ...
def update_session(...): ...  # EWA score: 0.7*old + 0.3*new
def get_session_score(...): ...
def get_all_sessions(user): ...
def get_session_manual(user, session_name): ...
def rename_session(...): ...
def delete_session(...): ...

def save_uploaded_manual(key, label, file_path, owner='system'): ...
def delete_uploaded_manual(key): ...
def get_uploaded_manuals(): ...         # all rows for server restore
def get_manuals_by_owner(owner): ...    # customer's own + system manuals
def get_manual_owner(key): ...

def get_manual_session_counts(): ...
def get_active_manual_session_counts(): ...  # sessions active in last 24h

def save_message(...): ...
def get_chat_history(user, session_name): ...
def get_recent_chat_history(user, session_name, limit=3): ...

def create_call(agent, manual_name, customer_id=None, session_id=None): ...  # returns call_id
def get_call(call_id): ...              # returns dict
def update_call_manual(call_id, manual_name): ...
def update_call_customer(call_id, customer_id, session_id=None): ...
def end_call(call_id, final_score): ...
def save_customer_rating(call_id, rating): ...

def save_call_turn(call_id, speaker, original_text, edited_text,
                   ai_suggestion, rag_confidence, agent_used_ai, turn_score): ...
def get_call_turns(call_id): ...        # returns list of dicts

def update_call_report_rating(call_id, rating): ...  # syncs rating after customer submits
def save_call_report(...): ...
def get_agent_reports(agent): ...
def get_call_report(call_id): ...

def save_citation_feedback(call_id, agent, manual_key, page,
                           query_text, source_excerpt, feedback): ...
```

> **Note:** database.py is shown as a summary above to save space. The full implementation is in the repo — every function listed exists with complete SQLite logic.

---

## main.py — Key sections

### Imports & setup

```python
from flask import Flask, request, jsonify, session, redirect, url_for, render_template, send_file, send_from_directory
from flask_socketio import SocketIO, join_room, emit
from groq import Groq as GroqClient
import os, re, json, io

from database import (init_db, login_user, register_user,
    save_message, get_chat_history, get_recent_chat_history, get_all_sessions,
    create_session, update_session, get_session_manual,
    get_session_score, rename_session, delete_session,
    get_manual_session_counts, get_active_manual_session_counts,
    save_uploaded_manual, delete_uploaded_manual, get_uploaded_manuals,
    get_manuals_by_owner, get_manual_owner,
    create_call, get_call, update_call_manual, update_call_customer,
    end_call, save_customer_rating, save_call_turn, get_call_turns,
    save_call_report, get_agent_reports,
    get_call_report, update_call_report_rating, save_citation_feedback)

from rag_search import load_manual, search_manual
from llm_suggestions import (generate_answer, analyze_satisfaction,
    analyze_conversation_sentiment, grade_agent_turn,
    generate_call_report, generate_call_summary)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "guideai-secret-key-change-in-prod")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

AGENT_ID       = os.getenv("AGENT_ID",      "agent")
AGENT_PASSWORD = os.getenv("AGENT_PASSWORD", "agent123")

AVAILABLE_MANUALS = {
    "dishwasher_manual":      "Dishwasher",
    "washing_machine_manual": "Washing Machine"
}
MANUAL_FILES = {
    "dishwasher_manual":      "data/manual.pdf",
    "washing_machine_manual": "data/washing_machine.pdf"
}
HARDCODED_MANUAL_KEYS = {"dishwasher_manual", "washing_machine_manual"}

# LiveKit config — resolved once at startup
def _build_livekit_url(raw: str) -> str:
    url = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', raw).strip()
    url = re.sub(r'[\[\]()]', '', url).strip()
    if url and not url.startswith('wss://'):
        url = 'wss://' + url.lstrip('/')
    if not url or 'livekit.cloud' not in url or '[' in url or '(' in url:
        return 'wss://guideai-sr6dd6z9.livekit.cloud'
    return url

LIVEKIT_API_KEY    = os.getenv("LIVEKIT_API_KEY",    "").strip()
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "").strip()
LIVEKIT_URL        = _build_livekit_url(os.getenv("LIVEKIT_URL", ""))

_groq_client = GroqClient(api_key=os.getenv("GROQ_API_KEY"))  # singleton

active_calls  = {}   # in-memory call state
agent_in_call = set()
```

### Source linking helpers

```python
SOURCE_STOPWORDS = { "this","that","with","from","your","have","will","when",
    "which","there","their","about","using","manual","page","into","then","than",
    "also","what","where","how","why","does","should","could","would","please","tell","help" }

def source_keywords(*texts, limit=12):
    keywords = []
    for text in texts:
        for word in re.findall(r"[a-zA-Z0-9]{4,}", strip_page_markers(text).lower()):
            if word in SOURCE_STOPWORDS or word in keywords:
                continue
            keywords.append(word)
            if len(keywords) >= limit:
                return keywords
    return keywords

def source_match_score(query, answer, chunk):
    # Answer terms 2×, query terms 1× — answer reveals what was actually sourced
    answer_terms = source_keywords(answer, limit=10)
    query_terms  = [t for t in source_keywords(query, limit=8) if t not in answer_terms]
    chunk_text   = strip_page_markers(chunk).lower()
    return (sum(2 for t in answer_terms if t in chunk_text) +
            sum(1 for t in query_terms  if t in chunk_text))

def source_links_for_manual(manual_name, chunks, query="", answer="", call_id="", max_sources=3):
    sources    = []
    seen_pages = set()
    # Pre-sort chunks by answer-relevance, not RAG order
    ranked = sorted(chunks, key=lambda c: source_match_score(query, answer, c), reverse=True)
    for chunk in ranked:
        page = page_number_from_chunk(chunk)
        if not page or page in seen_pages:
            continue
        seen_pages.add(page)
        highlight = " ".join(source_keywords(answer, query, chunk))
        excerpt   = source_excerpt(chunk)
        score     = source_match_score(query, answer, chunk)
        sources.append({
            "manual_key": manual_name, "page": page, "label": f"Page {page}",
            "excerpt": excerpt, "highlight": highlight,
            "match_score": score, "confidence": source_confidence_label(score),
            "url": url_for("manual_source", manual_key=manual_name, page=page,
                           call_id=call_id, query=query[:300],
                           highlight=highlight, excerpt=excerpt)
        })
        if len(sources) >= max_sources:
            return sources
    # Fallback: infer pages from PDF text if no [Page N] markers in chunks
    for page in infer_source_pages_from_pdf(manual_name, chunks, answer=answer, max_pages=max_sources):
        if page in seen_pages:
            continue
        highlight = " ".join(source_keywords(answer, query, *chunks))
        excerpt   = source_excerpt(ranked[0] if ranked else "")
        score     = max(source_match_score(query, answer, c) for c in chunks) if chunks else 0
        sources.append({...})
        if len(sources) >= max_sources:
            break
    return sources
```

### /ask route (chat)

```python
@app.route("/ask", methods=["POST"])
def ask():
    # Auth, validate, get manual_name from session
    ...
    relevant_docs, confidence = search_manual(query, manual_name)
    generation_confidence = confidence if not is_uploaded else "medium"

    answer = generate_answer(query=query, context=relevant_docs,
                             history=chat_history, manual_name=manual_name,
                             confidence=generation_confidence, session_score=session_score)

    answer_score = analyze_satisfaction(answer, query=query,
                                        context_confidence=generation_confidence,
                                        is_uploaded=is_uploaded)

    # EWA session score update; every 4 messages blend with sentiment
    if len(chat_history) % 4 == 0:
        sentiment_score = analyze_conversation_sentiment(chat_history)
        blended = round((answer_score * 0.6) + (sentiment_score * 0.4), 2) if sentiment_score else answer_score
        update_session(user, session_name, blended)
    else:
        update_session(user, session_name, answer_score)

    return jsonify({"answer": answer, "confidence": confidence, "satisfaction_score": answer_score})
```

### SocketIO — customer_message (live call RAG)

```python
@socketio.on("customer_message")
def on_customer_message(data):
    # Skip filler / non-actionable text
    if not is_actionable_query(text):
        emit("rag_suggestion", {"suggestion": "Filler detected...", "confidence": "none", ...},
             room=f"{call_id}_agent")
        return

    relevant_docs, confidence = search_manual(text, manual_name)
    suggestion = generate_answer(query=text, context=relevant_docs, manual_name=manual_name,
                                 confidence=generation_confidence, session_score=call_state["running_score"])

    sources    = source_links_for_manual(manual_name, relevant_docs, query=text,
                                         answer=suggestion, call_id=call_id)
    flashcards = flashcards_from_answer(suggestion)

    emit("rag_suggestion", {"suggestion": suggestion, "confidence": confidence,
                             "flashcards": flashcards, "sources": sources},
         room=f"{call_id}_agent")
    emit("manual_sources", {"query": text, "sources": sources},
         room=f"{call_id}_customer")
```

### Call finalisation

```python
def _finalise_call(call_id):
    # end_call() in DB, ask customer to rate, generate report + summary
    end_call(call_id, final_score)
    emit("rating_request", {}, room=f"{call_id}_customer")
    turns       = get_call_turns(call_id)
    summary     = generate_call_summary(turns, manual_name)
    report_text = generate_call_report(call_id, agent, manual_name, turns,
                                        overall_score=final_score,
                                        customer_rating=call.get("customer_rating") or 0)
    save_call_report(call_id, agent, report_text, transcript_json,
                     overall_score=final_score, customer_rating=0)
    # Write call record back into customer's chat session
    save_message(customer_id, session_name, manual_name, call_marker, "system")
    save_message(customer_id, session_name, manual_name, summary, "ai")
    emit("call_ended",  {"report_url": ..., "final_score": ...}, room=f"{call_id}_agent")
    emit("call_summary", {"summary": summary},                   room=f"{call_id}_customer")
    active_calls.pop(call_id, None)
    agent_in_call.discard(call_id)
```

---

## Recent changes (this session)

| File | Change |
|------|--------|
| `llm_suggestions.py` | Changed `GROQ_MODEL` from `llama-3.1-8b-instant` → `llama-3.3-70b-versatile` |
| `main.py` | `source_match_score` — answer terms now weighted 2×, query terms 1× |
| `main.py` | `source_links_for_manual` — chunks pre-sorted by answer-relevance before page selection |
| `main.py` | `infer_source_pages_from_pdf` — added `answer` param, same 2×/1× weighting |
| `main.py` | Removed dead code: `get_customer_manuals()`, 3 empty WebRTC stubs, duplicate comment block |
| `main.py` | Groq singleton `_groq_client` at module level; LiveKit config resolved once at startup |
| `main.py` | `on_customer_rating` now calls `update_call_report_rating()` instead of inline SQL |
| `database.py` | Added `update_call_report_rating(call_id, rating)` |
| `database.py` | Removed `get_pending_call()` and `get_call_score_history()` (unreachable) |
| `rag_search.py` | Merged double `collection.count()` calls into cached `total_chunks` variable |
| `extract_pdf.py` | Fixed mojibake characters (corrupted UTF-8) in two log strings on lines 108 and 111 |
| `.gitignore` | Added `*.log` entry |
