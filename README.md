<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/Google_Gemini-2.0_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
<img src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>

<br/>
<br/>

```
  ██████╗ ██╗   ██╗██╗██████╗ ███████╗ █████╗ ██╗
 ██╔════╝ ██║   ██║██║██╔══██╗██╔════╝██╔══██╗██║
 ██║  ███╗██║   ██║██║██║  ██║█████╗  ███████║██║
 ██║   ██║██║   ██║██║██║  ██║██╔══╝  ██╔══██║██║
 ╚██████╔╝╚██████╔╝██║██████╔╝███████╗██║  ██║██║
  ╚═════╝  ╚═════╝ ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝
```

### Ask anything about any appliance manual — and get a real answer.

**GuideAI** is a RAG-powered AI assistant that reads your technical manuals,
understands your questions, and improves its answers over time.

<br/>

[**Live Demo**](#) · [**Report Bug**](../../issues) · [**Request Feature**](../../issues)

<br/>

</div>

---

## What is GuideAI?

Most appliance manuals are 60-page PDFs nobody reads. GuideAI turns them into a conversational assistant — ask in plain English, get a precise answer from the actual manual. No hallucination, no guessing. Everything it says comes directly from your documents.

It supports multiple manuals simultaneously, keeps every chat in a separate session, scores its own answers, and adapts its behavior based on how well it has been performing. The longer you use it, the better it gets.

---

## Features at a Glance

| Feature | Description |
|---|---|
| 🔍 **Multi-manual RAG** | Each chat is locked to one manual — zero cross-contamination |
| 🧠 **Self-improving loop** | AI scores its own answers and changes behavior based on past performance |
| 🎤 **Voice input** | Speak your question directly using the Web Speech API |
| 💬 **Session-based chats** | Named conversations with full history, rename, and delete |
| 📊 **Confidence scoring** | Every answer shows retrieval confidence — high, medium, or low |
| 🔒 **Secure auth** | SHA-256 password hashing, signed session cookies |
| 🪵 **Full logging** | Checkpoint-level terminal logs across every pipeline stage |
| 📄 **Section-aware chunking** | Preserves document structure for better retrieval accuracy |

---

## How It Works

```
User asks a question
        │
        ▼
Flask resolves which manual this session is locked to (from DB)
        │
        ▼
RAG Pipeline ─── embed query ──► ChromaDB vector search
        │                              │
        │         top 8 candidates ◄──┘
        │
        ▼
Hybrid Scorer
  keyword score × 2
  + domain relevance score
  − cross-manual penalty
        │
        ▼
Top 3 chunks + confidence level
        │
        ▼
Gemini 2.0 Flash generates answer
  shaped by: context · history · session score · confidence
        │
        ▼
Gemini scores its own answer (1–10)
        │
        ▼
Session score updated (weighted rolling average)
        │
        ▼
Score feeds back into next answer's prompt ── ↻ self-improving loop
```

---

## Tech Stack

**Backend**
- [Python 3.10+](https://python.org) — core language
- [Flask](https://flask.palletsprojects.com) — web framework and REST API
- [SQLite](https://sqlite.org) — persistent storage for users, sessions, chats

**AI / ML**
- [Google Gemini 2.0 Flash](https://ai.google.dev) — answer generation and self-evaluation
- [Gemini Embedding 001](https://ai.google.dev) — semantic vector embeddings

**Vector Database**
- [ChromaDB](https://trychroma.com) — persistent local vector store, one collection per manual

**Document Processing**
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io) — PDF text extraction
- Custom section-aware chunker — heading detection + sliding window fallback

**Frontend**
- Vanilla HTML, CSS, JavaScript — no frameworks
- Web Speech API — browser-native voice input (Chrome / Edge)

---

## Project Structure

```
dishwasher-assistant/
│
├── data/
│   ├── vectordb/           ← ChromaDB persistent store (auto-generated)
│   ├── database.db         ← SQLite database (auto-generated)
│   ├── manual.pdf          ← Dishwasher manual
│   └── washing_machine.pdf ← Washing machine manual
│
├── templates/
│   ├── login.html          ← Login + register UI
│   └── dashboard.html      ← Main chat interface
│
├── main.py                 ← Flask app + /ask pipeline (7 checkpoints)
├── database.py             ← All DB operations (users, sessions, chats)
├── rag_search.py           ← Embedding, retrieval, hybrid scoring
├── extract_pdf.py          ← PDF extraction + section-aware chunking
├── llm_suggestions.py      ← Answer generation + self-evaluation + sentiment
├── logger.py               ← Centralised logging factory
├── sentiment.py            ← Conversation-level sentiment analysis
├── .env                    ← API keys (never commit this)
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)
- Chrome or Edge browser (for voice input)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/guideai.git
cd guideai
```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up your environment file**

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=any_random_string_for_flask_sessions
```

**5. Add your PDF manuals**

Place your PDF files in the `data/` folder:

```
data/manual.pdf             ← Dishwasher manual
data/washing_machine.pdf    ← Washing machine manual
```

**6. Run the app**

```bash
python main.py
```

Open your browser at `http://localhost:5000`

> **First run note:** The server will embed all PDF chunks into ChromaDB on startup. This takes 2–5 minutes depending on manual size. Subsequent starts are instant.

---

## Adding a New Manual

1. Drop your PDF into `data/`
2. Add two lines in `main.py`:

```python
AVAILABLE_MANUALS = {
    "dishwasher_manual":      "Dishwasher",
    "washing_machine_manual": "Washing Machine",
    "your_new_manual":        "Your Label Here",   # ← add this
}

MANUAL_FILES = {
    "dishwasher_manual":      "data/manual.pdf",
    "washing_machine_manual": "data/washing_machine.pdf",
    "your_new_manual":        "data/your_file.pdf",  # ← and this
}
```

3. Delete `data/vectordb/` and restart — the new manual will be ingested automatically.

---

## The Self-Improving Loop

GuideAI's most distinctive feature is that it gets better within a session based on how well it is performing.

```
Answer generated
      │
      ▼
Gemini evaluates the answer quality  →  score 1–10
      │
      ▼
Every 4 messages: conversation sentiment also scored
      │
      ▼
Blended score = (answer quality × 0.6) + (sentiment × 0.4)
      │
      ▼
Rolling average = (old score × 0.7) + (new score × 0.3)
      │
      ▼
Score stored in SQLite
      │
      ▼
On the next question, score is read back and shapes the prompt:

  Score ≥ 8  →  concise and confident
  Score 6–8  →  clear and structured
  Score 4–6  →  careful, use numbered steps
  Score < 4  →  maximum caution, admit uncertainty, do not guess
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Redirect to login or dashboard |
| `GET/POST` | `/login` | Authenticate user |
| `POST` | `/register` | Create new account |
| `GET` | `/dashboard` | Main chat interface |
| `GET` | `/manuals` | List available manuals |
| `POST` | `/create_session` | Create a new chat session bound to a manual |
| `POST` | `/ask` | Submit a question — runs the full 7-step pipeline |
| `GET` | `/history` | Fetch chat history for a session |
| `GET` | `/sessions` | List all sessions for current user |
| `POST` | `/rename_session` | Rename a session |
| `POST` | `/delete_session` | Delete a session and its messages |
| `GET` | `/logout` | Clear session and redirect to login |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Your Google Gemini API key |
| `SECRET_KEY` | Yes | Flask session signing key — set any random string |

---

## Known Limitations

- Voice input requires Chrome or Edge — Firefox does not support the Web Speech API
- Manuals must be text-based PDFs — scanned image PDFs will not extract correctly
- Currently supports static manuals only — upload from UI is a planned feature
- SQLite is suitable for single-user or small deployments — use PostgreSQL for production at scale

---

## Roadmap

- [ ] Upload manuals directly from the UI
- [ ] Text-to-speech responses
- [ ] Multi-language support
- [ ] Cloud deployment (Docker + PostgreSQL)
- [ ] Analytics dashboard for session quality trends
- [ ] Smarter query rewriting layer

---

## Author

**Mohammed Ibrahim Faheem**
BTech CSE — M.S. Ramaiah University of Applied Sciences

---

## License

This project is for educational and demonstration purposes.

---

<div align="center">

If this project helped you, consider giving it a ⭐ on GitHub.

</div>
