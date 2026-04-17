from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from dotenv import load_dotenv
import os

from database import (
    init_db, login_user, register_user,
    save_message, get_chat_history, get_all_sessions,
    create_session, update_session, get_session_manual,
    get_session_score, rename_session, delete_session
)

from rag_search import load_manual, search_manual
from llm_suggestions import generate_answer, analyze_satisfaction, analyze_conversation_sentiment

load_dotenv()

app = Flask(__name__)

# Static secret key — sessions survive server restarts
app.secret_key = os.getenv("SECRET_KEY", "guideai-secret-key-change-in-prod")

# ================= AVAILABLE MANUALS =================
# Add new manuals here. Key must match ChromaDB collection name.

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
    init_db()

    # Load ALL manuals on startup
    for manual_name, file_path in MANUAL_FILES.items():
        try:
            load_manual(manual_name, file_path)
        except Exception as e:
            print(f"⚠️ Could not load {manual_name}: {e}")


startup()


# ================= HOME =================

@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


# ================= AUTH =================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        unique_id = data.get("unique_id", "").strip()
        password = data.get("password", "").strip()

        if not unique_id or not password:
            return jsonify({"success": False, "error": "Missing credentials"})

        if login_user(unique_id, password):
            session["user"] = unique_id
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Invalid credentials"})

    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    unique_id = data.get("unique_id", "").strip()
    password = data.get("password", "").strip()

    if not unique_id or not password:
        return jsonify({"success": False, "error": "Missing fields"})

    if register_user(unique_id, password):
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "User already exists"})


# ================= DASHBOARD =================

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])


# ================= MANUALS LIST =================

@app.route("/manuals", methods=["GET"])
def manuals():
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
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    session_name = data.get("session_name", "").strip()
    manual_name = data.get("manual_name", "").strip()

    if not session_name:
        return jsonify({"error": "Session name is required"}), 400

    if manual_name not in AVAILABLE_MANUALS:
        return jsonify({"error": f"Unknown manual: '{manual_name}'"}), 400

    user = session["user"]
    create_session(user, session_name, manual_name)

    return jsonify({
        "success": True,
        "session_name": session_name,
        "manual_name": manual_name,
        "manual_label": AVAILABLE_MANUALS[manual_name]
    })


# ================= ASK =================

@app.route("/ask", methods=["POST"])
def ask():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    query = data.get("query", "").strip()
    session_name = data.get("session_name", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    if not session_name:
        return jsonify({"error": "Session name is required"}), 400

    user = session["user"]

    # Always get manual from DB — never assume or default
    manual_name = get_session_manual(user, session_name)
    if not manual_name:
        return jsonify({"error": "Session not found. Please create a session first."}), 404
    if manual_name not in AVAILABLE_MANUALS:
        return jsonify({"error": f"Manual '{manual_name}' is not available"}), 400

    # Get rolling score for adaptive generation
    session_score = get_session_score(user, session_name)

    # Save user message
    save_message(user, session_name, manual_name, query, "user")

    # Get chat history for context window
    chat_history = get_chat_history(user, session_name)

    # === MULTI-STEP RAG ===
    relevant_docs, confidence = search_manual(query, manual_name)

    # === ADAPTIVE ANSWER GENERATION ===
    # session_score drives adaptive behavior (self-improving loop)
    answer = generate_answer(
        query=query,
        context=relevant_docs,
        history=chat_history,
        manual_name=manual_name,
        confidence=confidence,
        session_score=session_score
    )

    # Save AI response
    save_message(user, session_name, manual_name, answer, "ai")

    # === PER-ANSWER SELF-EVALUATION ===
    answer_score = analyze_satisfaction(
        answer=answer,
        query=query,
        context_confidence=confidence
    )

    # === CONVERSATION SENTIMENT BLEND (every 4 messages) ===
    # Blends per-answer score with conversation-level satisfaction
    if len(chat_history) % 4 == 0:
        sentiment_score = analyze_conversation_sentiment(chat_history)
        if sentiment_score is not None:
            blended = round((answer_score * 0.6) + (sentiment_score * 0.4), 2)
            update_session(user, session_name, blended)
        else:
            update_session(user, session_name, answer_score)
    else:
        update_session(user, session_name, answer_score)

    return jsonify({
        "answer": answer,
        "satisfaction_score": answer_score,
        "confidence": confidence,
        "manual_used": AVAILABLE_MANUALS.get(manual_name, manual_name)
    })


# ================= HISTORY =================

@app.route("/history", methods=["GET"])
def history():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    user = session["user"]
    session_name = request.args.get("session_name", "").strip()

    if not session_name:
        return jsonify({"error": "Session name is required"}), 400

    chats = get_chat_history(user, session_name)

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
        return jsonify({"error": "Not logged in"}), 401

    user = session["user"]
    all_sessions = get_all_sessions(user)

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


# ================= RENAME SESSION =================

@app.route("/rename_session", methods=["POST"])
def rename():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    old_name = data.get("old_name", "").strip()
    new_name = data.get("new_name", "").strip()

    if not old_name or not new_name:
        return jsonify({"error": "Both old and new names are required"}), 400

    rename_session(session["user"], old_name, new_name)
    return jsonify({"success": True})


# ================= DELETE SESSION =================

@app.route("/delete_session", methods=["POST"])
def delete():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    session_name = data.get("session_name", "").strip()

    if not session_name:
        return jsonify({"error": "Session name is required"}), 400

    delete_session(session["user"], session_name)
    return jsonify({"success": True})


# ================= LOGOUT =================

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)