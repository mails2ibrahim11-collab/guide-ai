<div align="center">

<img src="https://img.shields.io/badge/GuideAI-1.0.0-7c6af7?style=for-the-badge&labelColor=0d0d0f" alt="version"/>
<img src="https://img.shields.io/badge/Python-3.12-7c6af7?style=for-the-badge&logo=python&logoColor=white&labelColor=0d0d0f" alt="python"/>
<img src="https://img.shields.io/badge/Flask-3.0-7c6af7?style=for-the-badge&logo=flask&logoColor=white&labelColor=0d0d0f" alt="flask"/>
<img src="https://img.shields.io/badge/LLaMA_3.3_70B-Groq-7c6af7?style=for-the-badge&labelColor=0d0d0f" alt="llm"/>

<br/><br/>

```
   ██████╗ ██╗   ██╗██╗██████╗ ███████╗ █████╗ ██╗
  ██╔════╝ ██║   ██║██║██╔══██╗██╔════╝██╔══██╗██║
  ██║  ███╗██║   ██║██║██║  ██║█████╗  ███████║██║
  ██║   ██║██║   ██║██║██║  ██║██╔══╝  ██╔══██║██║
  ╚██████╔╝╚██████╔╝██║██████╔╝███████╗██║  ██║██║
   ╚═════╝  ╚═════╝ ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝
```

### **RAG-Powered AI Customer Support Platform**
*Turn any PDF manual into an intelligent, voice-enabled support system*

<br/>

[**Live Demo**](https://trunks-scouring-hardwired.ngrok-free.dev) · [**Documentation**](#-architecture) · [**Quick Start**](#-quick-start) · [**API Reference**](#-api-reference)

<br/>

---

</div>

## What is GuideAI?

GuideAI converts PDF product manuals into a fully intelligent support system. Customers chat with an AI trained on those manuals, and can escalate to a live human agent. During live calls, the agent receives **real-time AI suggestions** based on what the customer is saying — powered by the same RAG pipeline.

> **The core idea:** Agents don't need to memorise every product. The AI retrieves the right information on demand, the agent verifies and delivers. Every turn is automatically scored, graded, and reported.

<br/>

## ✦ Features

| Feature | Description |
|---------|-------------|
| 🤖 **RAG Pipeline** | ChromaDB + MiniLM semantic search over PDF manuals |
| 🎙️ **Live Voice Calls** | P2P audio via LiveKit WebRTC |
| 📝 **Real-time STT** | AssemblyAI Universal Streaming transcription |
| ⚡ **Instant Suggestions** | LLaMA 3.3 70B generates answers in 200-800ms via Groq |
| 📊 **Auto Scoring** | LLM self-evaluates every answer 1-10, EWA session tracking |
| 🔍 **Source Viewer** | Highlights exact PDF page where answer came from |
| 💡 **Flashcard Popup** | Key points extracted per answer for agent quick-scan |
| 🧠 **Synonym Expansion** | oily→greasy, broken→faulty — handles informal language |
| 📋 **Call Reports** | Auto-generated professional post-call analysis |
| 📁 **Multi-manual** | Upload any PDF, system adapts dynamically |

<br/>

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         BROWSER                             │
│                                                             │
│  Customer Chat    Customer Call    Agent Monitor            │
│  (dashboard)      (call_customer)  (call_agent)             │
│                                                             │
└──────┬────────────────────┬──────────────────┬──────────────┘
       │ HTTP + SocketIO    │ LiveKit SDK       │ HTTP + SocketIO
       │                    │ + AssemblyAI STT  │
       ▼                    ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK SERVER (main.py)                   │
│                                                             │
│   /ask route          SocketIO events      Token routes     │
│   Text chat RAG       customer_message     /livekit-token   │
│                       agent_response       /assemblyai-token│
│                                                             │
│   rag_search.py    llm_suggestions.py    database.py        │
│   ChromaDB+MiniLM  Groq LLaMA 3.3 70B   SQLite 7 tables     │
└──────┬────────────────────┬───────────────────┬─────────────┘
       │                    │                   │
       ▼                    ▼                   ▼
  ChromaDB             Groq API            LiveKit Cloud
  (vectordb/)          LLaMA 3.3 70B       WebRTC P2P Audio
  384-dim embeddings   200-800ms latency   STUN/TURN/ICE
```

**Two parallel real-time channels:**
- **SocketIO** — text, events, suggestions, scores, call control
- **LiveKit** — audio only, P2P, Flask never touches audio

<br/>

## 🔬 RAG Pipeline

```
PDF File
   ↓
PyMuPDF extracts text  →  Tesseract OCR fallback (scanned pages)
   ↓
chunk_text()  →  Page-aware → Section-aware → Sliding window
   200 words, 40 overlap
   ↓
embed_text()  →  SentenceTransformers all-MiniLM-L6-v2
   384-dimensional vectors
   ↓
ChromaDB.add()  →  Persisted to data/vectordb/

━━━━━━━━━━━━━━━━━━  QUERY TIME  ━━━━━━━━━━━━━━━━━━

Customer query
   ↓
expand_query_with_synonyms()  →  oily→greasy, noise→rattling...
   ↓
embed_text(query)  →  384-dim vector
   ↓
ChromaDB cosine search  →  Top 12 candidates
   ↓
Hybrid scoring  →  keyword×2 + phrase×4 + domain score
   ↓
Top 5 chunks + confidence  →  generate_answer()
   ↓
LLaMA 3.3 70B  →  Structured answer + self-score
```

<br/>

## ⚙️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Web Server** | Flask + Flask-SocketIO + Eventlet | Lightweight, SocketIO integration, green threads |
| **LLM** | Groq → LLaMA 3.3 70B | 200-800ms on LPU chips — fastest available |
| **Embeddings** | SentenceTransformers MiniLM-L6-v2 | Local, free, 10ms per embed, 384-dim |
| **Vector Store** | ChromaDB | Zero setup, persistent, free |
| **STT** | AssemblyAI Universal Streaming | No mic conflict with LiveKit, accurate |
| **Voice** | LiveKit Cloud (WebRTC) | Managed STUN/TURN/ICE, Flask never touches audio |
| **PDF** | PyMuPDF + Tesseract + Pillow | Text layer + OCR fallback for scanned pages |
| **Database** | SQLite | Zero setup, parameterised queries, easy migration |
| **Auth** | bcrypt + Flask sessions | One-way hashing, HTTPONLY cookies |

<br/>

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.12+
Tesseract OCR  →  https://github.com/UB-Mannheim/tesseract/wiki
```

### Installation

```bash
# 1. Clone
git clone https://github.com/mails2ibrahim11-collab/guide-ai.git
cd guide-ai

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Environment variables
cp .env.example .env
# Fill in your API keys (see below)

# 5. Add your PDF manuals to data/
# Update MANUAL_FILES in main.py if needed

# 6. Run
python main.py
```

Open `http://localhost:5000`

<br/>

## 🔑 Environment Variables

```env
# Required
GROQ_API_KEY=your_groq_key               # console.groq.com
SECRET_KEY=your_long_random_secret       # any random string
LIVEKIT_API_KEY=your_livekit_api_key     # cloud.livekit.io
LIVEKIT_API_SECRET=your_livekit_secret
LIVEKIT_URL=wss://your-project.livekit.cloud
ASSEMBLYAI_API_KEY=your_assemblyai_key   # assemblyai.com
AGENT_ID=agent
AGENT_PASSWORD=agent123
```

<br/>

## 📁 Project Structure

```
guide-ai/
├── main.py                  ← Flask routes + SocketIO handlers (1861 lines)
├── database.py              ← All SQLite operations — 7 tables
├── rag_search.py            ← ChromaDB pipeline + synonym expansion
├── llm_suggestions.py       ← Groq LLM — answers, scoring, reports
├── extract_pdf.py           ← PDF extraction + smart chunking
├── logger.py                ← Centralised logging
├── requirements.txt
├── .env                     ← Secrets (never committed)
├── data/
│   ├── database.db          ← SQLite
│   ├── vectordb/            ← ChromaDB embeddings
│   └── *.pdf                ← Manuals
└── templates/
    ├── login.html
    ├── dashboard.html        ← Customer chat
    ├── agent_dashboard.html  ← Agent home
    ├── call_customer.html    ← Customer call (5-state machine)
    ├── call_agent.html       ← Agent monitor + AI suggestions
    ├── call_report.html      ← Post-call report
    ├── source_viewer.html    ← PDF page viewer
    └── manage_manuals.html   ← Upload/delete manuals
```

<br/>

## 🗄 Database Schema

```sql
users          → unique_id, password (bcrypt), role
sessions       → user, session_name, manual_name, score (EWA)
chats          → user, session_name, sender, message, timestamp
manuals        → key, label, file_path, owner
calls          → call_id, agent, customer_id, manual_name, status, final_score
call_turns     → call_id, speaker, original_text, ai_suggestion, turn_score
call_reports   → call_id, report_text, transcript (JSON), overall_score
citation_feedback → call_id, manual_key, page, feedback (useful/wrong)
```

<br/>

## 📞 Live Call Flow

```
Customer clicks "Call Agent"
   ↓  POST /call/request
Flask creates call_id, emits incoming_call → agent dashboard
   ↓
Customer selects manual → customer_join socket event
Agent joins → agent_join socket event
   ↓
Customer clicks "Start Voice Call"
   ↓  GET /livekit-token  (2hr TTL JWT)
LiveKit P2P audio established  ←→  AssemblyAI STT starts streaming
   ↓
Customer speaks → AssemblyAI transcript → customer_message event
Flask: search_manual() → generate_answer(is_voice=True)
   ↓
emit rag_suggestion → agent screen
Agent sees: suggestion + flashcards + source links + score
   ↓
Agent responds → grade_agent_turn() → live score chart updates
   ↓
End call → generate_call_summary() + generate_call_report()
Customer rates 1-10 → stored in call_reports
```

<br/>

## 📊 Scoring System

| Score Type | Source | Effect |
|------------|--------|--------|
| Turn score | LLM self-rates answer 1-10 | Updates session EWA |
| Session score | `0.7×old + 0.3×new` | Controls LLM adaptive tone |
| Sentiment | Every 4 msgs, blended `0.6×answer + 0.4×sentiment` | Feeds session score |
| Agent turn | `grade_agent_turn()` vs AI suggestion | Live chart on agent screen |
| Customer rating | Post-call 1-10 | Stored in report |
| Citation feedback | Useful +3 / Wrong -5 | Future source ranking |

**Adaptive LLM behaviour based on session score:**
- `< 4` → Maximum caution, no guessing, step-by-step only
- `4-6` → Extra careful, numbered steps
- `6-8` → Balanced, structured
- `≥ 8` → Confident, concise

<br/>

## 🌐 Deployment (AWS + ngrok)

```bash
# On EC2 (Ubuntu 24.04, t3.medium)

# Install dependencies
sudo apt update && sudo apt install -y python3 python3-pip python3-venv nginx git tesseract-ocr

# Clone and setup
git clone https://github.com/mails2ibrahim11-collab/guide-ai.git
cd guide-ai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure environment
nano .env  # add your keys

# Setup systemd services (auto-start on boot)
sudo systemctl enable guideai ngrok
sudo systemctl start guideai ngrok
```

HTTPS via ngrok static domain — microphone access works in all browsers.

<br/>

## 🔒 Security

- **SQL injection** — all queries use parameterised `?` placeholders
- **Passwords** — bcrypt one-way hash, never stored in plaintext
- **Session cookies** — `HTTPONLY=True`, `SAMESITE=Lax`
- **API keys** — `.env` only, never in source code, in `.gitignore`
- **LiveKit tokens** — JWT with 2-hour TTL, role-specific grants
- **AssemblyAI tokens** — temporary tokens generated server-side, API key never sent to browser
- **Manual ownership** — customers can only delete their own uploaded manuals

<br/>

## 📖 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Text chat — runs full RAG pipeline |
| `/livekit-token` | POST | Generate LiveKit JWT for voice call |
| `/assemblyai-token` | GET | Generate temporary AssemblyAI token |
| `/call/request` | POST | Customer requests live agent call |
| `/call/<id>` | GET | Customer call page |
| `/call/<id>/agent` | GET | Agent monitor page |
| `/manual_source/<key>` | GET | Source viewer page |
| `/manual_source_image/<key>` | GET | Highlighted PDF page as PNG |
| `/upload_manual` | POST | Upload new PDF manual |
| `/citation_feedback` | POST | Agent marks source useful/wrong |

**SocketIO Events:**

| Event | Direction | Description |
|-------|-----------|-------------|
| `customer_message` | Client→Server | Customer text/voice message |
| `rag_suggestion` | Server→Agent | AI suggestion + flashcards + sources |
| `agent_response` | Client→Server | Agent sends response |
| `live_score_update` | Server→Agent | Turn score + running score |
| `incoming_call` | Server→Agent | New call notification |
| `voice_end` | Bidirectional | End voice call |

<br/>

## 🛠 Known Limitations

| Limitation | Upgrade Path |
|------------|-------------|
| Groq 12K TPM (free tier) | Upgrade to Dev tier ($5/month) → 200K TPM |
| SQLite single-writer | PostgreSQL — swap `connect()`, queries stay identical |
| ChromaDB local | Pinecone/Weaviate for millions of vectors |
| ScriptProcessorNode deprecated | Migrate to AudioWorkletNode |
| Single agent account | Add agent management UI + role routing |

<br/>

## 👥 Team

| Role | Responsibility |
|------|---------------|
| **ML + Integration** | RAG pipeline, LLM, model training, full project integration |
| **Backend** | Flask routes, database, API design |
| **Blockchain** | Smart contract, chain integration |
| **Frontend** | UI/UX, templates, client-side logic |

<br/>

---

</div>
