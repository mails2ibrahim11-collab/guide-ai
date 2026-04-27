# 🚀 GuideAI — Multi-Manual Intelligent Assistant

GuideAI is a **Retrieval-Augmented Generation (RAG)-based AI assistant** that provides accurate, context-aware answers from any uploaded document or technical manual.

It supports **multiple documents**, **dynamic uploads**, **voice input**, and **adaptive AI behavior**, making it a scalable foundation for intelligent support systems across any domain.

---

## 🔥 Features

* 🧠 **Multi-Manual Support**
  Ask questions across different manuals and documents (Dishwasher, Washing Machine, or any uploaded PDF)

* 📤 **Dynamic Manual Upload**
  Upload any PDF from the frontend — the system ingests, embeds, and makes it available for chat instantly. Uploads persist across server restarts.

* 🔍 **Advanced RAG Pipeline**
  Combines:
  * Semantic search (local embeddings via sentence-transformers)
  * Keyword scoring
  * Domain-aware filtering
  * Confidence-based ranking

* 🎯 **Confidence-Aware Answers**
  * Hardcoded manuals — confidence based on hybrid keyword/domain scoring
  * Uploaded manuals — confidence based on LLM self-evaluation score for accuracy

* 🔄 **Self-Improving AI**
  * Session-based rolling score (weighted 70/30 average)
  * Adaptive response behavior based on session performance
  * Sentiment-based feedback loop every 4 messages

* 🌐 **Domain-Neutral for Uploaded Documents**
  Uploaded PDFs use a generic prompt and scoring rubric — works for resumes, legal documents, textbooks, product specs, or any domain

* 💬 **Session-Based Conversations**
  * Full chat history tracking
  * Rename / delete sessions
  * Manual-specific chat isolation

* ⚙️ **Manual Management Page**
  * View all manuals with total and active chat counts
  * Upload and delete manuals from a dedicated page
  * Default manuals are protected from deletion

* 🎤 **Voice Input Support**
  * Speech-to-text query handling
  * Hands-free interaction

* 🧩 **Section-Aware Chunking**
  * Detects document headings using regex patterns
  * Prefixes chunks with section name for better retrieval
  * Falls back to sliding window for unstructured PDFs

* 🧠 **Intent Detection**
  * Classifies queries as location / howto / troubleshoot / definition / specification
  * Improves answer relevance and formatting

* 📄 **OCR Support**
  * Detects image-based PDF pages
  * Falls back to Tesseract OCR for scanned documents
  * Text-based pages are never sent to OCR (performance optimised)

---

## 🏗️ Architecture

```
User (Text / Voice)
↓
Frontend (HTML, JS)
↓
Flask Backend API
↓
RAG Pipeline (ChromaDB + Hybrid Scoring)
↓
Groq LLM (LLaMA 3.3 70B) — Generation
↓
Response + Confidence + Score
```

---

## ⚙️ Tech Stack

### 🔧 Backend
* Python
* Flask
* SQLite

### 🤖 AI / ML
* **Groq API** — `llama-3.3-70b-versatile` (generation, self-evaluation, sentiment)
* **sentence-transformers** — `all-MiniLM-L6-v2` (local embeddings, no API key needed)

### 📦 Vector Database
* ChromaDB (Persistent storage)

### 📄 Document Processing
* PyMuPDF (PDF text extraction)
* Tesseract + pytesseract (OCR for image-based pages)
* Pillow (image format conversion for OCR)
* Custom section-aware chunking

### 🌐 Frontend
* HTML, CSS, JavaScript (Vanilla)
* Web Speech API (voice input)

### 🔐 Security
* dotenv (.env for API keys)
* hashlib SHA-256 (password hashing)
* werkzeug secure_filename (safe file uploads)

---

## 🧠 How It Works

1. User selects a manual and asks a question (text or voice)
2. Query is analyzed for intent and embedded locally
3. Relevant chunks retrieved using hybrid scoring (semantic + keyword + domain)
4. For uploaded manuals — if hybrid scores are zero, top semantic results used directly
5. Context passed to Groq LLM with appropriate prompt (appliance or generic)
6. AI generates a grounded response
7. Answer scored using domain-appropriate rubric
8. Session score updated, future responses adapt based on performance

---

## 📂 Project Structure

```
guide-ai/
│
├── data/                 # Manuals + vector database + uploaded PDFs
├── templates/            # HTML frontend
│   ├── login.html
│   ├── dashboard.html
│   └── manage_manuals.html
├── database.py           # DB logic + manual persistence
├── rag_search.py         # Retrieval system + hybrid scoring
├── llm_suggestions.py    # AI response + scoring + sentiment
├── extract_pdf.py        # PDF processing + OCR
├── main.py               # Flask app + all routes
├── logger.py             # Centralised logging
├── sentiment.py          # Legacy (unused)
└── .env                  # API keys (should NOT be public)
```

---

## ⚠️ Important Notes

* `.env` should NOT be committed (contains API keys)
* `vectordb/` and `database.db` should be ignored in production
* Uploaded manuals persist across restarts via SQLite + disk storage
* Default manuals (Dishwasher, Washing Machine) cannot be deleted from the UI

---

## 🚀 Future Improvements

* 📷 Gemini Vision for diagram and image understanding in PDFs
* 📊 UI analytics — session score history charts
* 🔁 Multi-step retrieval refinement
* 🔊 Text-to-speech responses
* 🌍 Multi-language support

---

## 🎯 Vision

To build a **universal AI assistant** capable of understanding and answering queries from **any document or manual**, across any domain — from appliance manuals to resumes, legal documents to technical specifications.

---

## 👤 Author

**Mohammed Ibrahim Faheem**
BTech CSE — M.S. Ramaiah University of Applied Sciences

---

## ⭐ If you found this useful

Give it a star ⭐ on GitHub
