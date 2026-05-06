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
                    confidence="high", session_score=None, is_voice=False):

    manual_readable = manual_name.replace("_", " ").title() if manual_name else "this document"
    is_hardcoded    = manual_name in HARDCODED_MANUAL_KEYS
    no_context      = not context or confidence == "none"

    log.info(f"[GENERATE] ── Generating answer ──")
    log.info(f"[GENERATE] Manual='{manual_readable}' | confidence='{confidence}' | score={session_score} | voice={is_voice}")

    adaptive_instruction = get_adaptive_instructions(session_score)
    # Truncate context to stay within token limits
    raw_context      = "\n\n---\n\n".join(context) if context else ""
    combined_context = raw_context[:6000] + ("..." if len(raw_context) > 6000 else "")

    history_str = ""
    if history:
        recent      = history[-3:]
        history_str = "\n".join(
            f"{sender.upper()}: {message}"
            for sender, message, _ in recent
        )

    list_all_patterns = [
        r'\b(all|every|list all|list everything|complete list|full list)\b',
        r'\b(tell me all|show me all|give me all|what are all)\b',
        r'\b(everything about|all the)\b',
    ]
    is_list_all    = any(re.search(p, query.lower()) for p in list_all_patterns)
    exhaustive_note = (
        "\nIMPORTANT: The user wants a COMPLETE list. Scan ALL context sections. Include every instance.\n"
    ) if is_list_all else ""

    if no_context:
        confidence_note = (
            "No relevant context was retrieved from the manual. "
            "Use your general knowledge about this type of product/document to answer. "
            "Clearly state when you are reasoning from general knowledge rather than the manual."
        )
        context_block = "No relevant manual content was retrieved for this query."
    elif confidence == "low":
        confidence_note = (
            "Retrieved context has LOW relevance to the query. "
            "Use what is available plus your general knowledge about this product type. "
            "Clearly distinguish manual content from general knowledge."
        )
        context_block = combined_context
    elif confidence == "medium":
        confidence_note = (
            "Context is partially relevant. Answer what you can from it. "
            "Supplement with general knowledge where the manual is silent, flagging clearly."
        )
        context_block = combined_context
    else:
        confidence_note = "Context is highly relevant. Prioritise it in your answer."
        context_block = combined_context

    if is_voice:
        output_format = """
OUTPUT FORMAT — VOICE (agent will relay this verbally):
- Plain sentences only. No bullets, no asterisks, no numbered lists with symbols.
- Start with ONE sentence that directly answers the question.
- Follow with 2-3 supporting sentences if needed.
- For procedures: "First do X. Then do Y. Finally do Z."
- Max 80 words unless a procedure requires more.
- No markdown, no bold markers.
"""
    else:
        output_format = """
OUTPUT FORMAT:
- FIRST LINE: one plain sentence directly answering the question (the "headline").
- Then a blank line.
- Then detail: bullets (•) for lists, numbered steps (1. 2. 3.) for procedures.
- **Bold** for component names, error codes, settings, critical values.
- Each bullet/step: 1-2 sentences max.
- No filler openers. No "it depends" without explaining what it depends on.
- Include exact names, values, temperatures, durations from the manual where available.
"""

    synonym_note = """
SYNONYM & PARAPHRASE AWARENESS:
- The user may use everyday words that differ from technical manual terms.
- Examples: "oily" = "greasy" = "fatty residue", "stopped working" = "not functioning" = "fault",
  "strange sound" = "unusual noise" = "rattling/grinding", "wear and tear" = "component degradation".
- Always reason about what the user MEANS, not just the exact words they used.
- If the query contains multiple questions or sub-problems, address each one separately.
"""

    if is_hardcoded:
        prompt = f"""You are an expert support assistant for a {manual_readable}.

ADAPTIVE BEHAVIOR: {adaptive_instruction}

RETRIEVAL CONFIDENCE: {confidence_note}
{exhaustive_note}
{synonym_note}
ANSWERING RULES:
- Prioritise the manual context below for specific values, part names, error codes, and procedures.
- When the manual does not cover a specific question, use your expert knowledge about this type of appliance — but clearly label it as general guidance.
- For fault/noise/malfunction queries: identify the likely cause first, then give resolution steps.
- For wear & tear queries: explain what degrades, typical lifespan, and what to check/replace.
- Never refuse to answer a reasonable appliance question just because it is not in the context.
- If genuinely uncertain, say what you know and what the user should check with a technician.
- If the user asks for a complete list of something, be sure to provide a comprehensive answer covering all relevant context sections. Do not miss any instances.   
- Address ALL parts of a multi-part question.
- If the context does not clearly answer the question, do NOT guess. Instead, say you don't have enough information and suggest what the user should check or clarify.
- NEVER mention specific brands, manufacturer names, model numbers, or company names unless they appear explicitly in the manual context. Do not invent brand attributions.
{output_format}
MANUAL CONTEXT:
{context_block}

RECENT CONVERSATION:
{history_str if history_str else "No prior conversation."}

USER QUESTION:
{query}

Answer:"""

    else:
        prompt = f"""You are a knowledgeable support assistant for the document "{manual_readable}".

ADAPTIVE BEHAVIOR: {adaptive_instruction}

RETRIEVAL CONFIDENCE: {confidence_note}
{exhaustive_note}
{synonym_note}
ANSWERING RULES:
- Use the document context as your primary source.
- When the document does not cover something, use your general knowledge relevant to this domain — label it as general guidance.
- Reason about what the user means, not just exact keyword matches.
- Address ALL parts of a multi-part question.
- Never return a generic "I can't find this" if you can provide useful guidance from general knowledge.
- NEVER mention specific brands, manufacturer names, model numbers, or company names unless they appear explicitly in the document context. Do not invent brand attributions.
{output_format}
DOCUMENT CONTEXT:
{context_block}

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
            max_tokens=800,
            temperature=0.35
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