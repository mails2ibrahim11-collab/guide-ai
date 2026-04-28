<div align="center">

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    ██████╗ ██╗   ██╗██╗██████╗ ███████╗ █████╗ ██╗           ║
║   ██╔════╝ ██║   ██║██║██╔══██╗██╔════╝██╔══██╗██║           ║
║   ██║  ███╗██║   ██║██║██║  ██║█████╗  ███████║██║           ║
║   ██║   ██║██║   ██║██║██║  ██║██╔══╝  ██╔══██║██║           ║
║   ╚██████╔╝╚██████╔╝██║██████╔╝███████╗██║  ██║██║           ║
║    ╚═════╝  ╚═════╝ ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝           ║
║                                                               ║
║          Intelligent Document Assistant · RAG · AI           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6B35?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-Persistent-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

> **Ask anything. From any document. Get answers that actually make sense.**

</div>

---

## ◈ What is GuideAI?

GuideAI is a production-grade **Retrieval-Augmented Generation (RAG)** system that turns any PDF into a conversational AI assistant. Instead of reading through pages of documentation, users simply ask — and the system retrieves the exact relevant sections and generates precise, grounded answers.

Built from scratch. No LangChain. No magic abstractions. Every component intentionally designed.

---

## ◈ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                       │
│              Text Input  ·  Voice Input  ·  Chat            │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│                      FLASK BACKEND                           │
│         Auth  ·  Sessions  ·  Upload  ·  API Routes          │
└──────┬───────────────────────┬───────────────────────────────┘
       │                       │
┌──────▼──────┐      ┌─────────▼─────────────────────────────┐
│   SQLite    │      │            RAG PIPELINE               │
│─────────────│      │───────────────────────────────────────│
│  · Users    │      │  1. Intent Detection                  │
│  · Sessions │      │  2. Query Embedding (local)           │
│  · Chats    │      │  3. Semantic Search → ChromaDB        │
│  · Manuals  │      │  4. Hybrid Scoring                    │
└─────────────┘      │     (keyword + domain + semantic)     │
                     │  5. Confidence Assessment             │
                     └─────────────────┬─────────────────────┘
                                       │
                     ┌─────────────────▼─────────────────────┐
                     │           GROQ LLM ENGINE             │
                     │───────────────────────────────────────│
                     │  · Adaptive prompt (session score)    │
                     │  · Domain-aware generation            │
                     │  · Self-evaluation scoring            │
                     │  · Conversation sentiment analysis    │
                     └───────────────────────────────────────┘
```

---

## ◈ Feature Breakdown

### 🔍 Hybrid Retrieval Pipeline
Not just vector search. A three-layer scoring system:

| Layer | What it does |
|-------|-------------|
| **Semantic Search** | ChromaDB cosine similarity — finds conceptually similar chunks |
| **Keyword Score** | Exact word matching weighted ×2 — anchors results to query terms |
| **Domain Relevance** | Own-manual keywords boosted, cross-manual keywords penalized ×2 |

For uploaded manuals where keyword scoring returns zero — the system automatically falls back to pure semantic results, ensuring retrieval never silently fails.

---

### 🔄 Self-Improving Session Loop

Every answer the AI gives is automatically scored. That score shapes the next response.

```
Answer generated
      ↓
LLM-as-judge evaluates (1–10)
      ↓
Every 4 messages: blended with conversation sentiment
  [answer × 0.6] + [sentiment × 0.4]
      ↓
Rolling average stored: (old × 0.7) + (new × 0.3)
      ↓
Next prompt adapts behavior:

  Score ≥ 8  →  Concise, confident
  Score ≥ 6  →  Balanced, ask for clarification
  Score ≥ 4  →  Careful, numbered steps
  Score < 4  →  Maximum caution, admit uncertainty
```

No user ratings. No manual feedback. Fully automated quality signal.

---

### 📤 Dynamic Manual Upload

Upload any PDF. It becomes queryable in seconds.

```
PDF uploaded
    ↓
Text extracted (PyMuPDF)
    ↓  [if page is image-based]
OCR fallback (Tesseract via pytesseract + Pillow)
    ↓
Section-aware chunking (200 words, 40-word overlap)
    ↓
Dynamic keywords extracted (top 20 frequent terms)
    ↓
Embeddings generated locally (sentence-transformers)
    ↓
Stored in ChromaDB + metadata saved to SQLite
    ↓
Available immediately — persists across server restarts
```

---

### 🎯 Confidence System

Two separate confidence pipelines depending on manual type:

**Hardcoded manuals** (Dishwasher, Washing Machine)
- Confidence derived from hybrid keyword/domain scoring
- Tuned thresholds: high ≥ 6, medium ≥ 2, low < 2

**Uploaded manuals** (any PDF)
- Hybrid scoring not calibrated for arbitrary domains
- Confidence derived from LLM self-evaluation score instead
- Score ≥ 8 → High · Score ≥ 5 → Medium · Score < 5 → Low

---

### 🧩 Section-Aware Chunking

```python
SECTION_PATTERNS = [
    r'^\d+[\.\s]+[A-Z][^\n]{3,60}$',    # 1. Installation
    r'^[A-Z][A-Z\s]{4,40}$',             # TROUBLESHOOTING
    r'^\s*(Chapter|Section|Part)\s+\d+', # Chapter 3
]
```

Each chunk is prefixed with its section heading before embedding. A chunk about filter cleaning becomes `[MAINTENANCE] remove the filter by turning...` — giving semantic context that boosts retrieval precision.

---

## ◈ Tech Stack

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER              TECHNOLOGY              PURPOSE          │
├──────────────────────────────────────────────────────────────┤
│  Generation    Groq · LLaMA 3.3 70B        Fast inference    │
│  Embeddings    sentence-transformers        Local, free      │
│                all-MiniLM-L6-v2            384 dimensions    │
│  Vector DB     ChromaDB (persistent)        Semantic search  │
│  Backend       Python · Flask               API + routing    │
│  Database      SQLite                       Sessions + meta  │
│  PDF Extract   PyMuPDF (fitz)               Text extraction  │
│  OCR           Tesseract + pytesseract      Scanned pages    │
│  Image         Pillow                       Format bridge    │
│  Frontend      HTML · CSS · Vanilla JS      Chat UI          │
│  Voice         Web Speech API               Hands-free input │
│  Security      hashlib SHA-256              Password hashing │
│                werkzeug secure_filename     Safe uploads     │
└──────────────────────────────────────────────────────────────┘
```

---

## ◈ Project Structure

```
guide-ai/
│
├── 📁 data/
│   ├── database.db           ← SQLite (users, sessions, chats, manuals)
│   ├── vectordb/             ← ChromaDB persistent storage
│   ├── manual.pdf            ← Dishwasher manual
│   ├── washing_machine.pdf   ← Washing machine manual
│   └── [uploaded].pdf        ← Dynamically uploaded manuals
│
├── 📁 templates/
│   ├── login.html            ← Auth page
│   ├── dashboard.html        ← Main chat interface
│   └── manage_manuals.html   ← Manual management page
│
├── main.py                   ← Flask app + all routes
├── database.py               ← All DB operations
├── rag_search.py             ← Retrieval pipeline
├── llm_suggestions.py        ← Generation + scoring + sentiment
├── extract_pdf.py            ← PDF extraction + OCR + chunking
├── logger.py                 ← Centralised logging
├── requirements.txt          ← Dependencies
└── .env                      ← API keys (never commit this)
```

---

## ◈ Key Design Decisions

**Why RAG over fine-tuning?**
Zero training infrastructure. Knowledge base updates by re-ingesting a PDF, not retraining a model. Answers are auditable — traceable to exact source chunks.

**Why local embeddings over an embedding API?**
No rate limits during bulk ingestion. No API cost. Works offline. `all-MiniLM-L6-v2` at 384 dimensions is sufficient for document-level semantic search.

**Why Groq over OpenAI?**
Same API structure, faster inference, generous free tier. LLaMA 3.3 70B performs comparably to GPT-4o for structured factual Q&A at a fraction of the cost.

**Why automated scoring over user ratings?**
Users don't rate messages. LLM-as-judge gives a continuous quality signal with zero friction. Sentiment adds a cross-check — a satisfied-sounding user confirms what the score predicts.

**Why SQLite over PostgreSQL?**
Zero configuration, file-based, appropriate for single-server deployment. PostgreSQL is the upgrade path for multi-user production scale — the schema is fully compatible.

---

## ◈ Roadmap

- [ ] 📷 Gemini Vision — diagram and image understanding in PDFs
- [ ] 📊 Session score history charts and analytics UI
- [ ] 🔁 Multi-step retrieval refinement for complex queries
- [ ] 🔊 Text-to-speech response output
- [ ] 🌍 Multi-language document support
- [ ] ☁️ Production deployment with PostgreSQL + persistent file storage

---

## ◈ Important Notes

- `.env` must never be committed — contains API keys
- `data/vectordb/` and `data/database.db` should be in `.gitignore`
- Default manuals (Dishwasher, Washing Machine) are protected from deletion
- Uploaded manuals survive server restarts via SQLite metadata + disk storage
- Deleting a manual must be done through the UI — never manually from disk

---

<div align="center">

---

**Built by Mohammed Ibrahim Faheem**
BTech CSE · M.S. Ramaiah University of Applied Sciences · Bengaluru

*If this helped you — give it a ⭐*

</div>
