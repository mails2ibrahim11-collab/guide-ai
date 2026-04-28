import os
import re
import json
from dotenv import load_dotenv
from groq import Groq
from logger import get_logger

load_dotenv(override=True)

log = get_logger("llm_suggestions")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

log.info(f"[LLM] Groq client initialised — model: {GROQ_MODEL}")

HARDCODED_MANUAL_KEYS = {"dishwasher_manual", "washing_machine_manual"}


# ================= ADAPTIVE TONE =================

def get_adaptive_instructions(session_score):
    if session_score is None:
        log.debug("[ADAPTIVE] No prior score — using neutral instructions")
        return "Be clear, thorough and helpful."

    if session_score >= 8:
        log.debug(f"[ADAPTIVE] Score={session_score} → HIGH — concise/confident mode")
        return (
            "Previous answers were rated excellent. "
            "Be concise and confident. Keep answers direct and focused."
        )
    elif session_score >= 6:
        log.debug(f"[ADAPTIVE] Score={session_score} → GOOD — balanced mode")
        return (
            "Previous answers were rated good. "
            "Stay clear and structured. If anything is ambiguous, ask for clarification."
        )
    elif session_score >= 4:
        log.debug(f"[ADAPTIVE] Score={session_score} → AVERAGE — careful/step-by-step mode")
        return (
            "Previous answers were rated average. Be extra careful and thorough. "
            "Use numbered steps wherever possible. Double-check your answer against the context."
        )
    else:
        log.debug(f"[ADAPTIVE] Score={session_score} → POOR — maximum-caution mode")
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
    is_hardcoded    = manual_name in HARDCODED_MANUAL_KEYS

    log.info(f"[GENERATE] ── Generating answer ──")
    log.info(f"[GENERATE] Manual='{manual_readable}' | confidence='{confidence}' | score={session_score}")

    if not context or confidence == "none":
        log.warning("[GENERATE] ⚠️ No context — returning fallback message")
        return (
            f"I couldn't find relevant information in the {manual_readable} document "
            "for your question. Could you rephrase it or be more specific?"
        )

    adaptive_instruction = get_adaptive_instructions(session_score)
    combined_context     = "\n\n---\n\n".join(context)

    if confidence == "low":
        confidence_note = (
            "⚠️ WARNING: Retrieved context has LOW relevance. "
            "Be honest — if the context doesn't clearly answer the question, say so."
        )
    elif confidence == "medium":
        confidence_note = (
            "The retrieved context is partially relevant. "
            "Answer what you can. Clearly flag anything you are inferring."
        )
    else:
        confidence_note = "The retrieved context is highly relevant. Answer confidently."

    history_str = ""
    if history:
        recent      = history[-6:]
        history_str = "\n".join(
            f"{sender.upper()}: {message}"
            for sender, message, _ in recent
        )

    # ── OUTPUT FORMAT RULES (shared by both prompt types) ──────────
    output_format = """
OUTPUT FORMAT — MANDATORY:
- Use bullet points (•) for lists of facts, options, or features.
- Use numbered steps (1. 2. 3.) for any procedure, sequence, or troubleshooting.
- Use **bold** for component names, settings, warnings, error codes, or key values.
- Keep each bullet/step to 1–2 clear sentences.
- Use sub-bullets (  –) only when genuinely needed for clarity.
- NO filler phrases like "Sure!", "Great question!", "Of course!" — start directly with the answer.
- NO vague statements like "it depends" without immediately explaining what it depends on.
- Be specific: include exact names, values, temperatures, durations, settings from the manual.
- If the manual gives a specific error code, cycle name, or part number — use it.
- End with one short follow-up offer only if the answer might need clarification.
"""

    if is_hardcoded:
        prompt = f"""You are an expert support assistant for a {manual_readable}. You have deep knowledge of this specific appliance.

ADAPTIVE BEHAVIOR:
{adaptive_instruction}

RETRIEVAL CONFIDENCE:
{confidence_note}

STRICT RULES:
- Answer ONLY using the manual context provided. Do not invent or assume.
- Reference specific parts, settings, cycle names, error codes exactly as they appear in the manual.
- If a step has a specific duration, temperature, or setting value — include it.
- Do NOT mix information from other appliances.
- If the answer is not in the context, say clearly: "The manual doesn't cover this specifically" and suggest what the user could try or check.

INTENT HANDLING:
- "where is X" → describe exact physical location (e.g. "bottom of the tub, behind the lower spray arm")
- "how do I" → give precise numbered steps with exact settings/values
- "error / not working / fault" → identify the specific fault first, then give step-by-step resolution
- "what is / why" → give a concise, accurate explanation using manual terminology
- Unclear query → ask ONE focused clarifying question before answering
{output_format}
MANUAL CONTEXT:
{combined_context}

RECENT CONVERSATION:
{history_str if history_str else "No prior conversation."}

USER QUESTION:
{query}

Answer:"""

    else:
        prompt = f"""You are a knowledgeable assistant answering questions based on the document "{manual_readable}".

ADAPTIVE BEHAVIOR:
{adaptive_instruction}

RETRIEVAL CONFIDENCE:
{confidence_note}

STRICT RULES:
- Answer ONLY using the document context below. Do not invent or assume.
- Quote specific values, names, steps, or terms exactly as they appear in the document.
- If the answer is not clearly in the context, say so honestly and specifically.
- Never give a vague or generic answer when the document has specific information.
{output_format}
DOCUMENT CONTEXT:
{combined_context}

RECENT CONVERSATION:
{history_str if history_str else "No prior conversation."}

USER QUESTION:
{query}

Answer:"""

    log.debug(f"[GENERATE] Prompt built ({len(prompt)} chars) — calling Groq...")

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.4
        )
        answer = response.choices[0].message.content.strip()
        log.info(f"[GENERATE] ✅ Groq responded ({len(answer)} chars)")
        return answer

    except Exception as e:
        log.error(f"[GENERATE] ❌ Groq API call FAILED: {e}")
        return "The AI assistant is temporarily unavailable. Please try again in a moment."


# ================= SELF-EVALUATION =================

def analyze_satisfaction(answer, query=None, context_confidence=None, is_uploaded=False):
    log.debug(f"[SCORE] Evaluating answer quality (confidence='{context_confidence}')...")

    if context_confidence == "none":
        return 2

    cant_find_phrases = [
        "couldn't find", "not in the manual", "not in the document",
        "unable to find", "no relevant", "could you clarify", "could you rephrase"
    ]
    if any(p in answer.lower() for p in cant_find_phrases):
        return 4

    confidence_note = "Note: low-confidence context.\n" if context_confidence == "low" else ""

    rubric = """Scoring guide:
10 = Perfect: precise, complete, directly answers the question in clear bullets
8-9 = Excellent: very helpful, minor gaps
6-7 = Good: helpful but missing some detail
4-5 = Adequate: partially answers
2-3 = Weak: vague, off-topic, or incomplete
1   = Poor: wrong or made things up"""

    prompt = f"""Rate the quality of this AI-generated answer from 1 to 10.

{confidence_note}{rubric}

Answer to rate:
{answer}

Reply with ONLY a single integer from 1 to 10. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        match = re.search(r'\d+', response.choices[0].message.content.strip())
        if match:
            score = max(1, min(int(match.group()), 10))
            log.info(f"[SCORE] ✅ Self-eval score = {score}/10")
            return score
        return 5
    except Exception as e:
        log.error(f"[SCORE] ❌ Scoring FAILED: {e}")
        return 5


# ================= CONVERSATION SENTIMENT =================

def analyze_conversation_sentiment(chat_history):
    log.debug(f"[SENTIMENT] Analysing sentiment ({len(chat_history)} messages)...")

    if not chat_history or len(chat_history) < 4:
        return None

    history_text = "\n".join(
        f"{sender}: {message}"
        for sender, message, _ in chat_history[-10:]
    )

    prompt = f"""Analyze this support conversation and rate the user's overall satisfaction from 1 to 10.

1  = Very frustrated, problem not solved
5  = Neutral — partially helped
10 = Clearly satisfied, problem resolved

Conversation:
{history_text}

Reply with ONLY a single integer from 1 to 10. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        match = re.search(r'\d+', response.choices[0].message.content.strip())
        if match:
            score = max(1, min(int(match.group()), 10))
            log.info(f"[SENTIMENT] ✅ Sentiment score = {score}/10")
            return score
        return None
    except Exception as e:
        log.error(f"[SENTIMENT] ❌ Sentiment analysis FAILED: {e}")
        return None


# ================= CALL GRADING =================

def grade_agent_turn(customer_query, ai_suggestion, agent_actual_response, manual_name):
    log.debug("[CALL_GRADE] Grading agent turn...")

    if not agent_actual_response or not agent_actual_response.strip():
        return 3

    manual_readable = manual_name.replace("_", " ").title() if manual_name else "the document"

    prompt = f"""You are grading a support agent's response during a live call.

The agent had access to an AI suggestion based on the {manual_readable}.

CUSTOMER QUERY:
{customer_query}

AI SUGGESTED ANSWER:
{ai_suggestion if ai_suggestion else "No suggestion available."}

AGENT'S ACTUAL RESPONSE:
{agent_actual_response}

Grade 1-10 based on:
- Accuracy — correct information?
- Completeness — fully addressed the question?
- Clarity — easy to understand?
- Use of AI suggestion — appropriately used or improved on it?

Scoring:
10 = Perfect  |  8-9 = Excellent  |  6-7 = Good
4-5 = Adequate  |  2-3 = Weak  |  1 = Poor/wrong

Reply with ONLY a single integer from 1 to 10. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        match = re.search(r'\d+', response.choices[0].message.content.strip())
        if match:
            score = max(1, min(int(match.group()), 10))
            log.info(f"[CALL_GRADE] ✅ Turn score = {score}/10")
            return score
        return 5
    except Exception as e:
        log.error(f"[CALL_GRADE] ❌ Grading FAILED: {e}")
        return 5


# ================= CALL SUMMARY (for customer) =================

def generate_call_summary(turns, manual_name):
    """
    Generates a short 3-4 line plain-language summary of the call
    shown to the customer before the rating screen.
    """
    log.info("[SUMMARY] Generating customer-facing call summary...")

    if not turns:
        return "Your support session has ended. Thank you for reaching out."

    manual_readable = manual_name.replace("_", " ").title() if manual_name else "the document"

    transcript_lines = []
    for t in turns:
        speaker = "Customer" if t["speaker"] == "customer" else "Agent"
        text    = t["edited_text"] or t["original_text"]
        transcript_lines.append(f"{speaker}: {text}")
    transcript_text = "\n".join(transcript_lines[-10:])  # last 10 turns max

    prompt = f"""You are summarising a support call for the customer.

Manual / product discussed: {manual_readable}

TRANSCRIPT (last portion):
{transcript_text}

Write a friendly 3-4 sentence summary for the customer covering:
- What issue or question they raised
- What was resolved or explained
- Any action they should take next (if applicable)

Keep it warm, concise, and in plain language. No bullet points. No technical jargon.
Do NOT mention scores, ratings, or the agent's name."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4
        )
        summary = response.choices[0].message.content.strip()
        log.info(f"[SUMMARY] ✅ Summary generated ({len(summary)} chars)")
        return summary
    except Exception as e:
        log.error(f"[SUMMARY] ❌ Summary generation FAILED: {e}")
        return "Your support session has ended. Thank you for contacting us."


# ================= CALL REPORT GENERATION =================

def generate_call_report(call_id, agent, manual_name, turns, overall_score, customer_rating):
    """
    Generates a comprehensive end-of-call report for the agent.
    customer_rating is optional — if 0 it means the customer didn't rate.
    """
    log.info(f"[REPORT] Generating call report for '{call_id}'...")

    if not turns:
        return "No conversation data available for this call."

    manual_readable = manual_name.replace("_", " ").title() if manual_name else "Unknown"

    transcript_lines = []
    for t in turns:
        speaker = t["speaker"].upper()
        text    = t["edited_text"] or t["original_text"]
        transcript_lines.append(f"{speaker}: {text}")
    transcript_text = "\n".join(transcript_lines)

    total_turns = len([t for t in turns if t["speaker"] == "customer"])
    used_as_is  = len([t for t in turns if t.get("agent_used_ai") == 1])
    edited      = len([t for t in turns if t.get("agent_used_ai") == 2])
    ignored     = len([t for t in turns if t.get("agent_used_ai") == 0])

    turn_scores    = [t["turn_score"] for t in turns if t.get("turn_score") is not None]
    avg_turn_score = round(sum(turn_scores) / len(turn_scores), 2) if turn_scores else 0

    rating_line = (
        f"Customer rating: {customer_rating}/10"
        if customer_rating and customer_rating > 0
        else "Customer rating: Not provided"
    )

    prompt = f"""You are generating a professional end-of-call report for a support agent.

CALL DETAILS:
- Call ID: {call_id}
- Agent: {agent}
- Manual used: {manual_readable}
- Overall AI score: {overall_score}/10
- {rating_line}
- Average turn score: {avg_turn_score}/10
- Total customer turns: {total_turns}
- Used AI suggestion as-is: {used_as_is}
- Edited AI suggestion: {edited}
- Ignored AI suggestion: {ignored}

FULL TRANSCRIPT:
{transcript_text}

Write a professional call report with these sections:
1. CALL SUMMARY — 2-3 sentences on what the call was about and overall outcome
2. AGENT PERFORMANCE — how well the agent handled the call and used AI assistance
3. KEY MOMENTS — 2-3 specific moments from the transcript (good or needs improvement)
4. AREAS FOR IMPROVEMENT — specific, actionable suggestions
5. OVERALL VERDICT — one sentence final assessment

Tone: professional but constructive. Reference actual transcript moments.
Use clear section headers. Write in paragraphs, not bullet points."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.4
        )
        report = response.choices[0].message.content.strip()
        log.info(f"[REPORT] ✅ Report generated ({len(report)} chars)")
        return report
    except Exception as e:
        log.error(f"[REPORT] ❌ Report generation FAILED: {e}")
        return f"Report generation failed. Call score: {overall_score}/10. {rating_line}."