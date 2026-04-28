# GuideAI

> **RAG-powered support platform** — turn any PDF manual into a live support agent with real-time AI assistance, call monitoring, and intelligent conversation scoring.

---

## What It Does

GuideAI lets customers get instant answers from product manuals via a chat interface, and escalate to a live agent when needed. The agent sees AI-generated suggestions in real time, powered by retrieval-augmented generation (RAG) over the uploaded manual.

```
Customer types a question
        ↓
RAG searches the manual (ChromaDB + SentenceTransformers)
        ↓
LLM generates a structured answer (Groq / LLaMA 3.3 70B)
        ↓
Agent sees the suggestion → edits or sends → Customer sees it
        ↓
Every turn is scored, graded, and compiled into a call report
```

---

## Features

### Customer
- 💬 Chat with an AI assistant trained on any uploaded PDF
- 📞 Escalate to a live agent directly from the chat
- 📋 Call summary saved back into the chat history after the session
- ⭐ Rate the support session after the call ends
- 📤 Upload and manage your own manuals

### Agent
- 🔔 Real-time incoming call notifications with chat context preview
- 🤖 AI-generated answer suggestions per customer message
- ✏️ Edit suggestions before sending, or write your own
- 📖 Override the active manual mid-call
- 📊 Live turn-by-turn score chart
- 📋 Auto-generated call report after every session
- 📁 Full call history with AI scores and customer ratings

### Platform
- 🔐 Role-based authentication — hardcoded agent, self-registered customers
- 📄 Upload any PDF manual — OCR fallback for scanned documents
- 🧠 Self-improving scoring loop — answers adapt based on session score
- 🔍 Hybrid RAG — semantic search + keyword scoring + domain relevance
- ⚡ Real-time via Flask-SocketIO + eventlet

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-SocketIO |
| AI Generation | Groq API — `llama-3.3-70b-versatile` |
| Embeddings | SentenceTransformers — `all-MiniLM-L6-v2` (local, 384 dims) |
| Vector Store | ChromaDB (persistent) |
| PDF Processing | PyMuPDF, Tesseract OCR, Pillow |
| Database | SQLite |
| Frontend | HTML, CSS, Vanilla JS, Web Speech API |
| Real-time | Flask-SocketIO + eventlet |

---

## Project Structure

```
guideai/
├── main.py                  ← Flask app, all routes, Socket.IO handlers
├── database.py              ← All DB operations
├── rag_search.py            ← Retrieval pipeline, hybrid scoring
├── llm_suggestions.py       ← Generation, scoring, sentiment, call grading
├── extract_pdf.py           ← PDF extraction + OCR + chunking
├── logger.py                ← Centralised logging
├── .env                     ← GROQ_API_KEY, SECRET_KEY, AGENT_ID, AGENT_PASSWORD
├── data/
│   ├── database.db
│   ├── vectordb/            ← ChromaDB persistent storage
│   └── *.pdf                ← Uploaded manuals
└── templates/
    ├── login.html
    ├── dashboard.html           ← Customer dashboard
    ├── agent_dashboard.html     ← Agent dashboard
    ├── manage_manuals.html
    ├── call_customer.html
    ├── call_agent.html
    ├── call_report.html
    └── call_ended.html
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/mails2ibrahim11-collab/guide-ai.git
cd guide-ai
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_random_secret_key_here
AGENT_ID=agent
AGENT_PASSWORD=your_agent_password_here
```

Generate a secure secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Install Tesseract OCR (for scanned PDFs)

Download from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki) and install to the default path, or update the path in `extract_pdf.py`.

### 4. Add your manuals

Place your PDF files in the `data/` folder and register them in `main.py`:

```python
AVAILABLE_MANUALS = {
    "your_manual_key": "Display Name"
}
MANUAL_FILES = {
    "your_manual_key": "data/your_file.pdf"
}
```

### 5. Run

```bash
python main.py
```

Open `http://localhost:5000` in your browser.

---

## Testing with Two Roles

Since the agent and customer share the same server, use **two different browsers** to avoid session conflicts:

| Browser | Role | URL |
|---|---|---|
| Chrome | Agent | `http://localhost:5000` → login as agent |
| Firefox | Customer | `http://localhost:5000` → register + login as customer |

To test on two devices on the same network, change the run command in `main.py`:

```python
socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
```

Then open `http://<your-local-ip>:5000` on the second device.

---

## How the RAG Pipeline Works

```
PDF → text extraction (PyMuPDF + Tesseract OCR fallback)
    → section-aware chunking (200 words, 40-word overlap)
    → embedding (all-MiniLM-L6-v2, 384 dims)
    → stored in ChromaDB

Query → embed query
      → semantic search (ChromaDB cosine similarity)
      → hybrid scoring (keyword × 2 + domain relevance)
      → top 3 chunks selected
      → passed to LLaMA 3.3 70B via Groq
      → structured bullet-point answer returned
```

**Confidence levels:**
- `high` — score ≥ 6, answer is reliable
- `medium` — score ≥ 2, some relevant context found
- `low` — score < 2, answer may be incomplete

**Adaptive scoring:** Every answer is self-evaluated 1–10. The session score is a rolling average that adjusts the prompt style — low scores trigger more cautious, step-by-step responses.

---

## Database Schema

```sql
users         (unique_id, password, role)
sessions      (id, user, session_name, manual_name, last_used, score)
chats         (id, user, session_name, manual_name, sender, message, timestamp)
manuals       (key, label, file_path, created_at, owner)
calls         (id, agent, customer_id, session_id, manual_name, status, created_at, ended_at, final_score, customer_rating)
call_turns    (id, call_id, speaker, original_text, edited_text, ai_suggestion, rag_confidence, agent_used_ai, turn_score, timestamp)
call_reports  (id, call_id, agent, report_text, transcript, overall_score, customer_rating, created_at)
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key — get one free at [console.groq.com](https://console.groq.com) |
| `SECRET_KEY` | Flask session secret — use a long random string |
| `AGENT_ID` | Username for the agent account (default: `agent`) |
| `AGENT_PASSWORD` | Password for the agent account (default: `agent123`) |

---

## Roadmap

- [ ] WebRTC audio — real voice calls between customer and agent
- [ ] Multi-agent support — multiple agents handling concurrent calls
- [ ] Gemini Vision — image and diagram understanding in PDFs
- [ ] Analytics dashboard — session score trends over time
- [ ] Email notifications — alert agent when customer initiates a call

---

## License

MIT