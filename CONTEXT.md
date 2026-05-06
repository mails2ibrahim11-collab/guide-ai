# GuideAI — Project Context for Claude Code

## Stack
Flask + SocketIO, ChromaDB, SentenceTransformers, Groq LLaMA 3.3 70B, LiveKit, SQLite

## Active improvement plan
1. Source viewer accuracy — main.py + source_viewer.html
2. Answer quality — llm_suggestions.py  
3. Flashcard redesign — call_agent.html + main.py
4. Voice call retry logic — call_customer.html + call_agent.html
5. Agent STT scoring — call_agent.html + main.py

## Key decisions made
- Agent voice scoring: use Web Speech API on agent tab, pipe transcript to grade_agent_turn()
- Flashcards: keyword + 1 sentence max, cycle one at a time
- Answers: add one-line summary at top, no markdown during voice