from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from dotenv import load_dotenv
import os

from logger import get_logger

from database import (
    init_db, login_user, register_user,
    save_message, get_chat_history, get_all_sessions,
    create_session, update_session, get_session_manual,
    get_session_score, rename_session, delete_session
)

from rag_search import load_manual, search_manual
from llm_suggestions import generate_answer, analyze_satisfaction, analyze_conversation_sentiment

load_dotenv()

log = get_logger("main")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "guideai-secret-key-change-in-prod")

# ================= AVAILABLE MANUALS =================

AVAILABLE_MANUALS = {
    "dishwasher_manual": "Dishwasher",
    "washing_machine_manual": "Washing Machine"
}

MANUAL_FILES = {
    "dishwasher_manual": "data/manual.pdf",
    "washing_machine_manual": "data/washing_machine.pdf"
}


# ================= STARTUP =================

def startup():
    log.info("=" * 60)
    log.info("GuideAI starting up...")
    log.info("=" * 60)

    # Step 1 — Database
    log.info("[STARTUP] Initialising database...")
    try:
        init_db()
        log.info("[STARTUP] ✅ Database ready")
    except Exception as e:
        log.critical(f"[STARTUP] ❌ Database init FAILED: {e}")
        raise

    # Step 2 — Manuals
    log.info(f"[STARTUP] Loading {len(MANUAL_FILES)} manual(s)...")
    for manual_name, file_path in MANUAL_FILES.items():
        log.info(f"[STARTUP] Loading '{manual_name}' from '{file_path}'...")
        try:
            load_manual(manual_name, file_path)
            log.info(f"[STARTUP] ✅ '{manual_name}' ready")
        except Exception as e:
            log.error(f"[STARTUP] ❌ Could not load '{manual_name}': {e}")

    log.info("[STARTUP] ✅ Startup complete — Flask is ready")
    log.info("=" * 60)


startup()


# ================= HOME =================

@app.route("/")
def index():
    if "user" in session:
        log.debug(f"[/] User '{session['user']}' already logged in → redirect to dashboard")
        return redirect(url_for("dashboard"))
    log.debug("[/] No session → redirect to login")
    return redirect(url_for("login"))


# ================= AUTH =================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        unique_id = data.get("unique_id", "").strip()
        password = data.get("password", "").strip()

        log.info(f"[LOGIN] Attempt for user '{unique_id}'")

        if not unique_id or not password:
            log.warning("[LOGIN] ❌ Missing credentials in request")
            return jsonify({"success": False})

        if login_user(unique_id, password):
            session["user"] = unique_id
            log.info(f"[LOGIN] ✅ '{unique_id}' authenticated successfully")
            return jsonify({"success": True})
        else:
            log.warning(f"[LOGIN] ❌ Authentication failed for '{unique_id}'")
            return jsonify({"success": False})

    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    unique_id = data.get("unique_id", "").strip()
    password = data.get("password", "").strip()

    log.info(f"[REGISTER] Registration attempt for '{unique_id}'")

    if not unique_id or not password:
        log.warning("[REGISTER] ❌ Missing fields")
        return jsonify({"success": False})

    if register_user(unique_id, password):
        log.info(f"[REGISTER] ✅ User '{unique_id}' created successfully")
        return jsonify({"success": True})
    else:
        log.warning(f"[REGISTER] ❌ User '{unique_id}' already exists")
        return jsonify({"success": False})


# ================= DASHBOARD =================

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        log.warning("[DASHBOARD] No session → redirect to login")
        return redirect(url_for("login"))
    log.debug(f"[DASHBOARD] Serving dashboard for '{session['user']}'")
    return render_template("dashboard.html", user=session["user"])


# ================= MANUALS =================

@app.route("/manuals", methods=["GET"])
def manuals():
    log.debug(f"[MANUALS] Returning {len(AVAILABLE_MANUALS)} manuals")
    return jsonify({
        "manuals": [
            {"key": k, "label": v}
            for k, v in AVAILABLE_MANUALS.items()
        ]
    })


# ================= CREATE SESSION =================

@app.route("/create_session", methods=["POST"])
def create_new_session():
    if "user" not in session:
        log.warning("[CREATE_SESSION] ❌ Unauthenticated request")
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    session_name = data.get("session_name", "").strip()
    manual_name = data.get("manual_name", "").strip()
    user = session["user"]

    log.info(f"[CREATE_SESSION] User='{user}' | Name='{session_name}' | Manual='{manual_name}'")

    if not session_name or manual_name not in AVAILABLE_MANUALS:
        log.warning(f"[CREATE_SESSION] ❌ Invalid data — session_name='{session_name}' manual='{manual_name}'")
        return jsonify({"error": "Invalid data"}), 400

    create_session(user, session_name, manual_name)
    log.info(f"[CREATE_SESSION] ✅ Session '{session_name}' created for '{user}'")

    return jsonify({"success": True})


# ================= ASK — full pipeline =================

@app.route("/ask", methods=["POST"])
def ask():
    if "user" not in session:
        log.warning("[ASK] ❌ Unauthenticated request")
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    query = data.get("query", "").strip()
    session_name = data.get("session_name", "").strip()

    if not query or not session_name:
        log.warning(f"[ASK] ❌ Invalid input — query='{query[:30]}' session='{session_name}'")
        return jsonify({"error": "Invalid input"}), 400

    user = session["user"]
    log.info(f"[ASK] ─────────────────────────────────────────────")
    log.info(f"[ASK] User='{user}' | Session='{session_name}'")
    log.info(f"[ASK] Query: '{query[:80]}{'...' if len(query) > 80 else ''}'")

    # Checkpoint 1 — Resolve manual from DB
    log.debug("[ASK] [1/7] Resolving manual from session...")
    manual_name = get_session_manual(user, session_name)
    if not manual_name:
        log.error(f"[ASK] ❌ [1/7] Session '{session_name}' not found in DB")
        return jsonify({"error": "Session not found"}), 404
    log.info(f"[ASK] ✅ [1/7] Manual resolved → '{manual_name}'")

    # Checkpoint 2 — Get session score (drives adaptive behavior)
    log.debug("[ASK] [2/7] Fetching session score for adaptive generation...")
    session_score = get_session_score(user, session_name)
    log.info(f"[ASK] ✅ [2/7] Session score = {session_score}")

    # Checkpoint 3 — Save user message
    log.debug("[ASK] [3/7] Saving user message to DB...")
    try:
        save_message(user, session_name, manual_name, query, "user")
        log.info("[ASK] ✅ [3/7] User message saved")
    except Exception as e:
        log.error(f"[ASK] ❌ [3/7] Failed to save user message: {e}")

    # Checkpoint 4 — RAG retrieval
    log.debug("[ASK] [4/7] Starting RAG retrieval...")
    try:
        chat_history = get_chat_history(user, session_name)
        relevant_docs, confidence = search_manual(query, manual_name)
        log.info(f"[ASK] ✅ [4/7] RAG done — {len(relevant_docs)} chunk(s) | confidence='{confidence}'")
    except Exception as e:
        log.error(f"[ASK] ❌ [4/7] RAG retrieval FAILED: {e}")
        return jsonify({"error": "Retrieval failed. Please try again."}), 500

    # Checkpoint 5 — LLM answer generation
    log.debug("[ASK] [5/7] Calling Gemini for answer generation...")
    try:
        answer = generate_answer(
            query=query,
            context=relevant_docs,
            history=chat_history,
            manual_name=manual_name,
            confidence=confidence,
            session_score=session_score  # feeds into adaptive prompt
        )
        log.info(f"[ASK] ✅ [5/7] Answer generated ({len(answer)} chars)")
    except Exception as e:
        log.error(f"[ASK] ❌ [5/7] LLM generation FAILED: {e}")
        return jsonify({"error": "AI generation failed. Please try again."}), 500

    # Checkpoint 6 — Save AI response
    log.debug("[ASK] [6/7] Saving AI response to DB...")
    try:
        save_message(user, session_name, manual_name, answer, "ai")
        log.info("[ASK] ✅ [6/7] AI response saved")
    except Exception as e:
        log.error(f"[ASK] ❌ [6/7] Failed to save AI response: {e}")

    # Checkpoint 7 — Dynamic scoring + self-improving loop
    log.debug("[ASK] [7/7] Running dynamic scoring pipeline...")

    # Per-answer quality score — Gemini evaluates its own answer
    answer_score = analyze_satisfaction(
        answer=answer,
        query=query,
        context_confidence=confidence
    )
    log.info(f"[ASK] ✅ [7/7] Per-answer score = {answer_score}/10")

    # Every 4 messages, blend with conversation-level sentiment
    if len(chat_history) % 4 == 0:
        log.debug("[ASK] [7/7] Running conversation sentiment analysis (every 4 messages)...")
        sentiment_score = analyze_conversation_sentiment(chat_history)
        if sentiment_score is not None:
            # 60% answer quality, 40% user satisfaction trend
            blended = round((answer_score * 0.6) + (sentiment_score * 0.4), 2)
            log.info(f"[ASK] [7/7] Blended score: {answer_score} (answer) × 0.6 + {sentiment_score} (sentiment) × 0.4 = {blended}")
            update_session(user, session_name, blended)
        else:
            log.debug("[ASK] [7/7] Sentiment skipped — using answer score only")
            update_session(user, session_name, answer_score)
    else:
        update_session(user, session_name, answer_score)

    log.info(f"[ASK] ✅ Full pipeline complete | confidence='{confidence}' | score={answer_score}")
    log.info(f"[ASK] ─────────────────────────────────────────────")

    return jsonify({
        "answer": answer,
        "confidence": confidence,
        "satisfaction_score": answer_score
    })


# ================= HISTORY =================

@app.route("/history", methods=["GET"])
def history():
    if "user" not in session:
        log.warning("[HISTORY] ❌ Unauthenticated request")
        return jsonify({"error": "Not logged in"}), 401

    user = session["user"]
    session_name = request.args.get("session_name", "").strip()

    log.debug(f"[HISTORY] Fetching history for '{user}' | session='{session_name}'")
    chats = get_chat_history(user, session_name)
    log.debug(f"[HISTORY] ✅ Returned {len(chats)} message(s)")

    return jsonify({
        "chats": [
            {"sender": c[0], "message": c[1], "timestamp": c[2]}
            for c in chats
        ]
    })


# ================= SESSIONS =================

@app.route("/sessions", methods=["GET"])
def get_sessions():
    if "user" not in session:
        log.warning("[SESSIONS] ❌ Unauthenticated request")
        return jsonify({"error": "Not logged in"}), 401

    user = session["user"]
    log.debug(f"[SESSIONS] Fetching all sessions for '{user}'")
    all_sessions = get_all_sessions(user)
    log.debug(f"[SESSIONS] ✅ Found {len(all_sessions)} session(s)")

    return jsonify({
        "sessions": [
            {
                "session_name": s[0],
                "manual_name": s[1],
                "manual_label": AVAILABLE_MANUALS.get(s[1], s[1]),
                "last_used": s[2],
                "score": s[3]
            }
            for s in all_sessions
        ]
    })


# ================= RENAME =================

@app.route("/rename_session", methods=["POST"])
def rename():
    data = request.get_json()
    user = session["user"]
    old_name = data.get("old_name", "")
    new_name = data.get("new_name", "")

    log.info(f"[RENAME] User='{user}' | '{old_name}' → '{new_name}'")
    try:
        rename_session(user, old_name, new_name)
        log.info("[RENAME] ✅ Renamed successfully")
    except Exception as e:
        log.error(f"[RENAME] ❌ Failed: {e}")
        return jsonify({"success": False}), 500

    return jsonify({"success": True})


# ================= DELETE =================

@app.route("/delete_session", methods=["POST"])
def delete():
    data = request.get_json()
    user = session["user"]
    session_name = data.get("session_name", "")

    log.info(f"[DELETE] User='{user}' | Session='{session_name}'")
    try:
        delete_session(user, session_name)
        log.info(f"[DELETE] ✅ Session '{session_name}' deleted")
    except Exception as e:
        log.error(f"[DELETE] ❌ Failed: {e}")
        return jsonify({"success": False}), 500

    return jsonify({"success": True})


# ================= LOGOUT =================

@app.route("/logout")
def logout():
    user = session.get("user", "unknown")
    session.pop("user", None)
    log.info(f"[LOGOUT] ✅ User '{user}' logged out")
    return redirect(url_for("login"))


# ================= RUN =================

if __name__ == "__main__":
    log.info("[RUN] Starting Flask development server...")
    app.run(debug=True)