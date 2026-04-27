import os
import re
from dotenv import load_dotenv
from groq import Groq
from logger import get_logger

load_dotenv(override=True)

log = get_logger("llm_suggestions")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

log.info(f"[LLM] Groq client initialised — model: {GROQ_MODEL}")

# Hardcoded manual keys — use appliance-specific prompts
HARDCODED_MANUAL_KEYS = {"dishwasher_manual", "washing_machine_manual"}


# ================= ADAPTIVE TONE =================

def get_adaptive_instructions(session_score):
    if session_score is None:
        log.debug("[ADAPTIVE] No prior score — using neutral instructions")
        return "Be clear, thorough and helpful."

    if session_score >= 8:
        log.debug(f"[ADAPTIVE] Score={session_score} → HIGH — using concise/confident mode")
        return (
            "Previous answers were rated excellent. "
            "Be concise and confident. The user is satisfied — keep answers direct and focused."
        )
    elif session_score >= 6:
        log.debug(f"[ADAPTIVE] Score={session_score} → GOOD — using balanced mode")
        return (
            "Previous answers were rated good. "
            "Stay clear and structured. If anything is ambiguous, ask for clarification."
        )
    elif session_score >= 4:
        log.debug(f"[ADAPTIVE] Score={session_score} → AVERAGE — using careful/step-by-step mode")
        return (
            "Previous answers were rated average. Be extra careful and thorough. "
            "Use numbered steps wherever possible. Double-check your answer against the context."
        )
    else:
        log.debug(f"[ADAPTIVE] Score={session_score} → POOR — using maximum-caution mode")
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
    is_hardcoded = manual_name in HARDCODED_MANUAL_KEYS

    log.info(f"[GENERATE] ── Generating answer ──")
    log.info(f"[GENERATE] Manual='{manual_readable}' | confidence='{confidence}' | session_score={session_score}")

    # Step 1 — Check for empty context
    log.debug("[GENERATE] [1/4] Checking context availability...")
    if not context or confidence == "none":
        log.warning(f"[GENERATE] ⚠️ [1/4] No context available — returning fallback message")
        return (
            f"I couldn't find relevant information in the {manual_readable} document for your question. "
            "Could you rephrase it or be more specific about what you need?"
        )
    log.debug(f"[GENERATE] ✅ [1/4] Context available — {len(context)} chunk(s)")

    # Step 2 — Build prompt
    log.debug("[GENERATE] [2/4] Building prompt...")
    adaptive_instruction = get_adaptive_instructions(session_score)
    combined_context = "\n\n---\n\n".join(context)

    if confidence == "low":
        confidence_note = (
            f"⚠️ WARNING: Retrieved context has LOW relevance to this query. "
            f"Be honest — if the context doesn't clearly answer the question, say so and ask the user to clarify."
        )
    elif confidence == "medium":
        confidence_note = (
            "The retrieved context is partially relevant. "
            "Answer what you can from the context. Clearly flag anything you are inferring."
        )
    else:
        confidence_note = "The retrieved context is highly relevant. Answer confidently."

    history_str = ""
    if history:
        recent = history[-6:]
        history_str = "\n".join(
            f"{sender.upper()}: {message}"
            for sender, message, _ in recent
        )
        log.debug(f"[GENERATE] Including {len(recent)} recent message(s) from history")
    else:
        log.debug("[GENERATE] No prior history to include")

    # Appliance-specific prompt for hardcoded manuals
    # Generic document prompt for uploaded manuals
    if is_hardcoded:
        prompt = f"""You are a professional support assistant for a {manual_readable}.

ADAPTIVE BEHAVIOR:
{adaptive_instruction}

RETRIEVAL CONFIDENCE:
{confidence_note}

STRICT RULES:
- Answer ONLY using the manual context provided below
- DO NOT use outside knowledge
- DO NOT mention or mix information from other appliances
- If the answer is not in the context, admit it clearly — do not make things up

INTENT HANDLING:
- "where" → describe the physical location clearly
- "how to" → give numbered step-by-step instructions
- "error / not working" → diagnose + give solution steps
- "what / why" → explain simply and clearly
- Vague or unclear query → ask the user to clarify before guessing

OUTPUT STYLE:
- Natural, helpful language
- Use numbered lists for multi-step answers
- Use **bold** for important terms or warnings
- Be concise — no padding

MANUAL CONTEXT:
{combined_context}

RECENT CONVERSATION:
{history_str if history_str else "No prior conversation."}

USER QUESTION:
{query}

Answer:"""
    else:
        prompt = f"""You are a helpful assistant answering questions based on a document called "{manual_readable}".

ADAPTIVE BEHAVIOR:
{adaptive_instruction}

RETRIEVAL CONFIDENCE:
{confidence_note}

STRICT RULES:
- Answer ONLY using the document context provided below
- DO NOT use outside knowledge
- If the answer is not clearly in the context, say so honestly — do not make things up
- Be direct and helpful

OUTPUT STYLE:
- Natural, clear language
- Use numbered lists for multi-part answers
- Use **bold** for key terms
- Be concise — no padding

DOCUMENT CONTEXT:
{combined_context}

RECENT CONVERSATION:
{history_str if history_str else "No prior conversation."}

USER QUESTION:
{query}

Answer:"""

    log.debug(f"[GENERATE] ✅ [2/4] Prompt built ({len(prompt)} chars)")

    # Step 3 — Call Groq
    log.debug(f"[GENERATE] [3/4] Calling Groq API ({GROQ_MODEL})...")
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
        log.info(f"[GENERATE] ✅ [3/4] Groq responded ({len(answer)} chars)")

    except Exception as e:
        log.error(f"[GENERATE] ❌ [3/4] Groq API call FAILED: {e}")
        return "The AI assistant is temporarily unavailable. Please try again in a moment."

    log.debug("[GENERATE] [4/4] Returning answer to caller")
    log.info(f"[GENERATE] ✅ Generation complete")

    return answer


# ================= SELF-EVALUATION =================

def analyze_satisfaction(answer, query=None, context_confidence=None, is_uploaded=False):
    log.debug(f"[SCORE] Evaluating answer quality (confidence='{context_confidence}', uploaded={is_uploaded})...")

    if context_confidence == "none":
        log.debug("[SCORE] context_confidence='none' → returning score=2")
        return 2

    cant_find_phrases = [
        "couldn't find", "not in the manual", "not in the document",
        "unable to find", "no relevant", "could you clarify", "could you rephrase"
    ]
    if any(p in answer.lower() for p in cant_find_phrases):
        log.debug("[SCORE] Answer contains 'couldn't find' phrase → returning score=4")
        return 4

    confidence_note = ""
    if context_confidence == "low":
        confidence_note = "Note: this answer was generated from low-confidence retrieved context.\n"

    # Generic rubric for uploaded manuals, appliance rubric for hardcoded ones
    if is_uploaded:
        rubric = """Scoring guide:
10 = Perfect: precise, complete, directly and fully answers the question
8-9 = Excellent: very helpful, minor gaps or could be slightly clearer
6-7 = Good: helpful but missing some detail or clarity
4-5 = Adequate: partially answers the question
2-3 = Weak: vague, off-topic, or largely incomplete
1   = Poor: wrong, refused without good reason, or made things up"""
    else:
        rubric = """Scoring guide:
10 = Perfect: precise, complete, step-by-step where needed, directly answers the appliance question
8-9 = Excellent: very helpful, minor gaps
6-7 = Good: helpful but could be more complete or clearer
4-5 = Adequate: partially answers the question
2-3 = Weak: vague, off-topic, or incomplete
1   = Poor: wrong, refused without good reason, or made things up"""

    prompt = f"""Rate the quality of this AI-generated answer from 1 to 10.

{confidence_note}
{rubric}

Answer to rate:
{answer}

Reply with ONLY a single integer from 1 to 10. Nothing else."""

    try:
        log.debug("[SCORE] Calling Groq for self-evaluation...")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        match = re.search(r'\d+', response.choices[0].message.content.strip())
        if match:
            score = max(1, min(int(match.group()), 10))
            log.info(f"[SCORE] ✅ Self-evaluation score = {score}/10")
            return score
        log.warning("[SCORE] ⚠️ Could not parse score from Groq response → defaulting to 5")
        return 5
    except Exception as e:
        log.error(f"[SCORE] ❌ Satisfaction scoring FAILED: {e} → defaulting to 5")
        return 5


# ================= CONVERSATION SENTIMENT =================

def analyze_conversation_sentiment(chat_history):
    log.debug(f"[SENTIMENT] Analysing conversation sentiment ({len(chat_history)} messages)...")

    if not chat_history or len(chat_history) < 4:
        log.debug("[SENTIMENT] Not enough messages for sentiment analysis — skipping")
        return None

    history_text = "\n".join(
        f"{sender}: {message}"
        for sender, message, _ in chat_history[-10:]
    )

    prompt = f"""Analyze this support conversation and rate the user's overall satisfaction from 1 to 10.

1  = Very frustrated, repeatedly complaining, problem not solved
5  = Neutral — partially helped
10 = Clearly satisfied, thanking, problem resolved

Conversation:
{history_text}

Reply with ONLY a single integer from 1 to 10. Nothing else."""

    try:
        log.debug("[SENTIMENT] Calling Groq for sentiment analysis...")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        match = re.search(r'\d+', response.choices[0].message.content.strip())
        if match:
            score = max(1, min(int(match.group()), 10))
            log.info(f"[SENTIMENT] ✅ Conversation sentiment score = {score}/10")
            return score
        log.warning("[SENTIMENT] ⚠️ Could not parse sentiment score — returning None")
        return None
    except Exception as e:
        log.error(f"[SENTIMENT] ❌ Sentiment analysis FAILED: {e}")
        return None