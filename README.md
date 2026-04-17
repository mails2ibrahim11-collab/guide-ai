# 🚀 GuideAI – Multi-Manual Intelligent Assistant

GuideAI is a **Retrieval-Augmented Generation (RAG)-based AI assistant** that provides accurate, context-aware answers from technical manuals.

It supports **multiple devices**, **voice input**, and **adaptive AI behavior**, making it a scalable foundation for intelligent support systems.

---

## 🔥 Features

* 🧠 **Multi-Manual Support**
  Ask questions across different manuals (Dishwasher, Washing Machine, etc.)

* 🔍 **Advanced RAG Pipeline**
  Combines:

  * Semantic search (embeddings)
  * Keyword scoring
  * Domain-aware filtering
  * Confidence-based ranking

* 🎯 **Confidence-Aware Answers**
  AI evaluates reliability of retrieved context before answering

* 🔄 **Self-Improving AI**

  * Session-based scoring
  * Adaptive response behavior
  * Sentiment-based feedback loop

* 💬 **Session-Based Conversations**

  * Chat history tracking
  * Rename / delete sessions
  * Manual-specific chat isolation

* 🎤 **Voice Input Support**

  * Speech-to-text query handling
  * Hands-free interaction

* 🧩 **Section-Aware Chunking**

  * Preserves document structure
  * Improves retrieval accuracy

* 🧠 **Intent Detection**

  * Understands query type (where, how, what, why)
  * Improves answer relevance

---

## 🏗️ Architecture

User (Text / Voice)
↓
Frontend (HTML, JS)
↓
Flask Backend API
↓
RAG Pipeline (ChromaDB + Retrieval Logic)
↓
Gemini LLM (Generation)
↓
Response + Score + Confidence

---

## ⚙️ Tech Stack

### 🔧 Backend

* Python
* Flask
* SQLite

### 🤖 AI / ML

* Google Gemini API

  * `gemini-2.0-flash` (generation)
  * `gemini-embedding-001` (embeddings)

### 📦 Vector Database

* ChromaDB (Persistent storage)

### 📄 Document Processing

* PyMuPDF (PDF extraction)
* Custom section-aware chunking

### 🌐 Frontend

* HTML
* CSS
* JavaScript (Vanilla)

### 🔐 Security

* dotenv (.env for API keys)
* hashlib (password hashing)

---

## 🧠 How It Works

1. User selects a manual and asks a question (text or voice)
2. Query is analyzed for intent and keywords
3. Relevant chunks are retrieved using hybrid scoring
4. Context is passed to Gemini LLM
5. AI generates a grounded response
6. Answer is scored and session is updated
7. Future responses adapt based on performance

---

## 📂 Project Structure

```
guide-ai/
│
├── data/                 # Manuals + vector database
├── templates/            # HTML frontend
├── database.py           # DB logic
├── rag_search.py         # Retrieval system
├── llm_suggestions.py    # AI response logic
├── extract_pdf.py        # PDF processing
├── sentiment.py          # Feedback system
├── main.py               # Flask app
└── .env                  # API keys (should NOT be public)
```

---

## ⚠️ Important Notes

* `.env` should NOT be committed (contains API keys)
* `vectordb/` and `database.db` should be ignored in production
* Currently supports **static manuals only**

---

## 🚀 Future Improvements

* 📤 Upload manuals directly from UI
* 🧠 Smarter query understanding layer
* 🔁 Multi-step retrieval refinement
* 📊 Better UI analytics & visualization
* 🔊 Text-to-speech responses

---

## 🎯 Vision

To build a **universal AI assistant** capable of understanding and answering queries from **any technical manual**, across domains.

---

## 👤 Author

**Mohammed Ibrahim Faheem**
BTech CSE – M.S. Ramaiah University of Applied Sciences

---

## ⭐ If you found this useful

Give it a star ⭐ on GitHub
