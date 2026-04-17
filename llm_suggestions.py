import os
import re
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ================= ADAPTIVE TONE =================

def get_adaptive_instructions(session_score):
    """
    This is the self-improving feedback loop.
    The session's rolling score directly changes how the AI responds next time.
    Low score → more careful, structured, willing to admit uncertainty.
    High score → concise and confident.
    """
    if session_score is None:
        return "Be clear, thorough and helpful."

    if session_score >= 8:
        return (
            "Previous answers were rated excellent. "
            "Be concise and confident. The user is satisfied — keep answers direct and focused."
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
    """
    Generate a grounded, adaptive answer.

    Adapts based on:
    - confidence: retrieval quality (high/medium/low/none)
    - session_score: rolling quality score → self-improving loop
    - history: last 6 messages for conversational context
    """
    manual_readable = manual_name.replace("_", " ").title() if manual_name else "this appliance"
    adaptive_instruction = get_adaptive_instructions(session_score)

    # No context at all → don't hallucinate
    if not context or confidence == "none":
        return (
            f"I couldn't find relevant information in the {manual_readable} manual for your question. "
            "Could you rephrase it or be more specific about what you need?"
        )

    combined_context = "\n\n---\n\n".join(context)

    # Confidence-aware instruction
    if confidence == "low":
        confidence_note = (
            f"⚠️ WARNING: Retrieved context has LOW relevance to this query. "
            f"Be honest — if the context doesn't clearly answer the question, say: "
            f"'I couldn't find a clear answer in the {manual_readable} manual. Could you clarify?'"
        )
    elif confidence == "medium":
        confidence_note = (
            "The retrieved context is partially relevant. "
            "Answer what you can from the context. Clearly flag anything you are inferring."
        )
    else:
        confidence_note = "The retrieved context is highly relevant. Answer confidently."

    # Recent conversation (last 6 messages)
    history_str = ""
    if history:
        recent = history[-6:]
        history_str = "\n".join(
            f"{sender.upper()}: {message}"
            for sender, message, _ in recent
        )

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
- Do not copy raw manual text verbatim
- Be concise — no padding

MANUAL CONTEXT:
{combined_context}

RECENT CONVERSATION:
{history_str if history_str else "No prior conversation."}

USER QUESTION:
{query}

Answer:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"❌ Gemini generation error: {e}")
        return "The AI assistant is temporarily unavailable. Please try again in a moment."


# ================= SELF-EVALUATION =================

def analyze_satisfaction(answer, query=None, context_confidence=None):
    """
    Evaluate the quality of a generated answer.
    This score feeds back into the session's rolling score,
    which in turn changes how future answers are generated.
    That is the self-improving loop.
    """

    # Pre-checks before calling Gemini
    if context_confidence == "none":
        return 2  # No context → poor answer by definition

    cant_find_phrases = [
        "couldn't find", "not in the manual", "unable to find",
        "no relevant", "could you clarify", "could you rephrase"
    ]
    if any(p in answer.lower() for p in cant_find_phrases):
        return 4  # Honest fallback — not wrong, but not a full answer

    confidence_note = ""
    if context_confidence == "low":
        confidence_note = "Note: this answer was generated from low-confidence retrieved context.\n"

    prompt = f"""Rate the quality of this AI-generated appliance support answer from 1 to 10.

{confidence_note}
Scoring guide:
10 = Perfect: precise, complete, step-by-step where needed, directly answers the question
8-9 = Excellent: very helpful, minor gaps
6-7 = Good: helpful but could be more complete or clearer
4-5 = Adequate: partially answers the question
2-3 = Weak: vague, off-topic, or incomplete
1   = Poor: wrong, refused without good reason, or made things up

Answer to rate:
{answer}

Reply with ONLY a single integer from 1 to 10. Nothing else."""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        match = re.search(r'\d+', response.text.strip())
        if match:
            return max(1, min(int(match.group()), 10))
        return 5
    except Exception as e:
        print(f"⚠️ Satisfaction scoring failed: {e}")
        return 5


# ================= CONVERSATION SENTIMENT =================

def analyze_conversation_sentiment(chat_history):
    """
    Analyzes the full conversation to detect the user's overall satisfaction trend.
    This is blended with the per-answer score every 4 messages to produce
    a more accurate session quality score.
    """
    if not chat_history or len(chat_history) < 4:
        return None  # Not enough data yet

    history_text = "\n".join(
        f"{sender}: {message}"
        for sender, message, _ in chat_history[-10:]
    )

    prompt = f"""Analyze this appliance support conversation and rate the user's overall satisfaction from 1 to 10.

1  = Very frustrated, repeatedly complaining, problem not solved
5  = Neutral — partially helped
10 = Clearly satisfied, thanking, problem resolved

Conversation:
{history_text}

Reply with ONLY a single integer from 1 to 10. Nothing else."""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        match = re.search(r'\d+', response.text.strip())
        if match:
            return max(1, min(int(match.group()), 10))
        return None
    except Exception as e:
        print(f"⚠️ Sentiment analysis failed: {e}")
        return None