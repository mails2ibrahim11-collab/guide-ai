# GuideAI

> **RAG-powered customer support platform** — turn any PDF manual into a live support agent with real-time AI assistance, voice calls, live transcription, and intelligent conversation scoring.

---

## What It Does

GuideAI lets customers get instant answers from product manuals via a chat interface, and escalate to a live voice + text agent session when needed. The agent sees AI-generated suggestions in real time as the customer speaks or types, powered by retrieval-augmented generation (RAG) over the uploaded manual. Every turn is scored, graded, and compiled into a full call report.

```
Customer speaks or types a question
        ↓
Web Speech API transcribes voice → text
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
- 📞 Escalate to a live agent with one click
- 🎙️ Full voice call with the agent via LiveKit (production-grade audio)
- 📝 Voice transcribed in real time via Web Speech API
- 📋 Call summary saved back into the chat history after the session
- ⭐ Rate the support session after the call ends
- 📤 Upload and manage your own PDF manuals

### Agent
- 🔔 Real-time incoming call notifications with chat context preview
- 🤖 AI-generated answer suggestions per customer message (text and voice)
- ✏️ Edit suggestions before sending, or write your own
- 📖 Override the active manual mid-call
- 📊 Live turn-by-turn score chart
- 📋 Auto-generated call report after every session
- 📁 Full call history with AI scores and customer ratings
- ⭐ Customer rating reflected on dashboard live without refresh

### Platform
- 🔐 Role-based authentication — hardcoded agent, self-registered customers
- 📄 Upload any PDF manual — OCR fallback for scanned documents
- 🧠 Self-improving scoring loop — answers adapt based on session score
- 🔍 Hybrid RAG — semantic search + keyword scoring + domain relevance
- ⚡ Real-time via Flask-SocketIO + eventlet
- 🎙️ Production-grade P2P audio via LiveKit Cloud

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-SocketIO |
| Async Engine | Eventlet |
| AI Generation | Groq API — `llama-3.3-70b-versatile` |
| Embeddings | SentenceTransformers — `all-MiniLM-L6-v2` (local, 384 dims) |
| Vector Store | ChromaDB (persistent, local) |
| PDF Processing | PyMuPDF, Tesseract OCR, Pillow |
| Database | SQLite |
| Voice Calls | LiveKit Cloud (P2P audio, no credit card) |
| Speech-to-Text | Web Speech API (Chrome built-in) |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Real-time | Flask-SocketIO + eventlet |
| HTTPS (dev) | Cloudflare Tunnel |

---

## Architecture

Two separate real-time channels run simultaneously during a call:

```
Browser (Customer)              Flask Server              Browser (Agent)
──────────────────              ────────────              ───────────────
SocketIO ──────────────────────────────────────────────── SocketIO
(text, events, suggestions)                               (suggestions, scores)

LiveKit SDK ──────────────────────────────────────────── LiveKit SDK
            ↕ via LiveKit Cloud (P2P audio)
            Flask never touches audio
```

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
├── .env                     ← Secrets (never committed)
├── requirements.txt
├── data/
│   ├── database.db
│   ├── vectordb/            ← ChromaDB persistent storage
│   └── *.pdf                ← Uploaded manuals
└── templates/
    ├── login.html
    ├── dashboard.html           ← Customer chat interface
    ├── agent_dashboard.html     ← Agent home with call history
    ├── manage_manuals.html
    ├── call_customer.html       ← Customer call page (voice + text)
    ├── call_agent.html          ← Agent call monitor
    ├── call_report.html
    ├── call_ended.html
    └── call_new.html
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
AGENT_ID=your_agent_username
AGENT_PASSWORD=your_agent_password

LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxxx
LIVEKIT_API_SECRET=your_livekit_secret
```

Generate a secure secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Get free API keys:
- **Groq** — [console.groq.com](https://console.groq.com) (LLM, free tier)
- **LiveKit** — [cloud.livekit.io](https://cloud.livekit.io) (voice calls, no credit card)

### 3. Install Tesseract OCR (for scanned PDFs)

Download from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki) and install to the default path.

### 4. Run

```bash
python main.py
```

Open `http://localhost:5000`.

---

## Running with HTTPS (required for voice calls on other devices)

Voice calls and microphone access require HTTPS on non-localhost URLs. Use Cloudflare Tunnel:

```bash
# Terminal 1 — Flask server
python main.py

# Terminal 2 — Cloudflare tunnel
.\cloudflared.exe tunnel --url http://localhost:5000
```

Open the `https://xxx.trycloudflare.com` URL on both devices. The tunnel URL changes every restart — keep Terminal 2 running during your session.

---

## Testing with Two Roles

Use two different browsers to avoid session conflicts:

| Browser | Role | Login |
|---|---|---|
| Chrome | Customer | Register a new account |
| Chrome Incognito | Agent | Use AGENT_ID / AGENT_PASSWORD from .env |

---

## How the RAG Pipeline Works

```
PDF → text extraction (PyMuPDF + Tesseract OCR fallback)
    → chunking (200 words, 40-word overlap)
    → embedding (all-MiniLM-L6-v2, 384 dims)
    → stored in ChromaDB

Query → embed query (384 dims)
      → ChromaDB cosine similarity search (top 12 candidates)
      → hybrid scoring:
            keyword_score = query words in chunk × 2
            domain_score  = domain keywords − (other domain keywords × 2)
      → top 5 chunks selected
      → passed to LLaMA 3.3 70B via Groq
      → structured bullet-point answer returned
```

**Confidence levels:**
- `high` — best chunk score ≥ 6
- `medium` — best chunk score ≥ 2
- `low` — below that, answer may be incomplete

**Adaptive scoring:** Every answer is self-evaluated 1-10. The session score is a rolling weighted average (70% old, 30% new) that adjusts the LLM tone — poor scores trigger more careful responses, high scores allow more confident concise answers.

---

## Voice Call Flow

```
Customer clicks "Start Voice Call"
        ↓
POST /livekit-token → JWT with room_create permission
        ↓
LiveKit SDK connects → room "guideai-{call_id}" auto-created
        ↓
Agent receives popup → accepts → joins same LiveKit room
        ↓
P2P audio established via LiveKit Cloud
        ↓
Web Speech API runs continuously on customer device
Customer speaks → sentence detected → sent to RAG pipeline
        ↓
Agent sees AI suggestion while hearing customer voice
```

---

## Database Schema

```sql
users         (unique_id, password, role)
sessions      (id, user, session_name, manual_name, last_used, score)
chats         (id, user, session_name, manual_name, sender, message, timestamp)
manuals       (key, label, file_path, created_at, owner)
calls         (id, agent, customer_id, session_id, manual_name, status,
               created_at, ended_at, final_score, customer_rating)
call_turns    (id, call_id, speaker, original_text, edited_text,
               ai_suggestion, rag_confidence, agent_used_ai, turn_score, timestamp)
call_reports  (id, call_id, agent, report_text, transcript,
               overall_score, customer_rating, created_at)
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key — [console.groq.com](https://console.groq.com) |
| `SECRET_KEY` | Flask session secret — long random string |
| `AGENT_ID` | Agent login username |
| `AGENT_PASSWORD` | Agent login password |
| `LIVEKIT_URL` | LiveKit Cloud WebSocket URL (`wss://...`) |
| `LIVEKIT_API_KEY` | LiveKit API key |
| `LIVEKIT_API_SECRET` | LiveKit API secret |

---

## Groq Rate Limits (Free Tier)

| Limit | Value |
|---|---|
| Requests per minute | 30 |
| Tokens per day | 100,000 |
| Requests per day | 6,000 |

During heavy testing these limits can be reached. Switch temporarily to `llama-3.1-8b-instant` in `llm_suggestions.py` while waiting for the limit to reset.

---

## Roadmap

- [ ] Multi-agent support
- [ ] Analytics dashboard — score trends, coverage gaps
- [ ] Gemini Vision — image understanding in PDFs
- [ ] Production deployment — Railway/Render + proper domain + SSL
- [ ] Fine-tuned embeddings for domain-specific retrieval

---

## License

MIT
