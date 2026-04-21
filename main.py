from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import os
import re

from logger import get_logger

from database import (
    init_db, login_user, register_user,
    save_message, get_chat_history, get_all_sessions,
    create_session, update_session, get_session_manual,
    get_session_score, rename_session, delete_session,
    get_manual_session_counts, get_active_manual_session_counts,
    save_uploaded_manual, delete_uploaded_manual, get_uploaded_manuals
)

from rag_search import load_manual, search_manual
from llm_suggestions import generate_answer, analyze_satisfaction, analyze_conversation_sentiment

load_dotenv()

log = get_logger("main")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "guideai-secret-key-change-in-prod")

AVAILABLE_MANUALS = {
    "dishwasher_manual": "Dishwasher",
    "washing_machine_manual": "Washing Machine"
}

MANUAL_FILES = {
    "dishwasher_manual": "data/manual.pdf",
    "washing_machine_manual": "data/washing_machine.pdf"
}

UPLOAD_FOLDER = "data"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_UPLOAD_MB = 50


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_manual_key(display_name):
    key = re.sub(r"[^a-z0-9]+", "_", display_name.lower().strip())
    key = key.strip("_")
    return f"{key}_manual"


def startup():
    log.info("=" * 60)
    log.info("GuideAI starting up...")
    log.info("=" * 60)

    log.info("[STARTUP] Initialising database...")
    try:
        init_db()
        log.info("[STARTUP] ✅ Database ready")
    except Exception as e:
        log.critical(f"[STARTUP] ❌ Database init FAILED: {e}")
        raise

    log.info(f"[STARTUP] Loading {len(MANUAL_FILES)} hardcoded manual(s)...")
    for manual_name, file_path in MANUAL_FILES.items():
        log.info(f"[STARTUP] Loading '{manual_name}' from '{file_path}'...")
        try:
            load_manual(manual_name, file_path)
            log.info(f"[STARTUP] ✅ '{manual_name}' ready")
        except Exception as e:
            log.error(f"[STARTUP] ❌ Could not load '{manual_name}': {e}")

    uploaded = get_uploaded_manuals()
    log.info(f"[STARTUP] Restoring {len(uploaded)} uploaded manual(s) from database...")
    for key, label, file_path in uploaded:
        if not os.path.exists(file_path):
            log.warning(f"[STARTUP] ⚠️ Uploaded manual '{key}' file missing at '{file_path}' — skipping")
            continue
        try:
            load_manual(key, file_path)
            AVAILABLE_MANUALS[key] = label
            MANUAL_FILES[key] = file_path
            log.info(f"[STARTUP] ✅ Restored uploaded manual '{key}' — '{label}'")
        except Exception as e:
            log.error(f"[STARTUP] ❌ Could not restore '{key}': {e}")

    log.info("[STARTUP] ✅ Startup complete — Flask is ready")
    log.info("=" * 60)


startup()


@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        unique_id = data.get("unique_id", "").strip()
        password = data.get("password", "").strip()
        log.info(f"[LOGIN] Attempt for user '{unique_id}'")
        if not unique_id or not password:
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
        return jsonify({"success": False})
    if register_user(unique_id, password):
        log.info(f"[REGISTER] ✅ User '{unique_id}' created successfully")
        return jsonify({"success": True})
    else:
        log.warning(f"[REGISTER] ❌ User '{unique_id}' already exists")
        return jsonify({"success": False})


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])


@app.route("/manuals", methods=["GET"])
def manuals():
    return jsonify({
        "manuals": [{"key": k, "label": v} for k, v in AVAILABLE_MANUALS.items()]
    })


@app.route("/manage_manuals")
def manage_manuals():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("manage_manuals.html", user=session["user"])


@app.route("/manual_stats", methods=["GET"])
def manual_stats():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    session_counts = get_manual_session_counts()
    active_counts  = get_active_manual_session_counts()
    manuals_list = []
    for key, label in AVAILABLE_MANUALS.items():
        manuals_list.append({
            "key":            key,
            "label":          label,
            "file_path":      MANUAL_FILES.get(key, ""),
            "session_count":  session_counts.get(key, 0),
            "active_count":   active_counts.get(key, 0),
            "is_default":     key in {"dishwasher_manual", "washing_machine_manual"}
        })
    return jsonify({"manuals": manuals_list})


@app.route("/upload_manual", methods=["POST"])
def upload_manual():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    display_name = request.form.get("display_name", "").strip()
    if not display_name:
        return jsonify({"error": "Display name is required"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > MAX_UPLOAD_MB:
        return jsonify({"error": f"File too large. Max size is {MAX_UPLOAD_MB}MB"}), 400

    manual_key = make_manual_key(display_name)
    if manual_key in AVAILABLE_MANUALS:
        return jsonify({"error": f"A manual named '{display_name}' already exists"}), 409

    file_path = os.path.join(UPLOAD_FOLDER, f"{manual_key}.pdf")
    try:
        file.save(file_path)
        log.info(f"[UPLOAD] ✅ File saved → '{file_path}' ({size_mb:.1f}MB)")
    except Exception as e:
        log.error(f"[UPLOAD] ❌ Failed to save file: {e}")
        return jsonify({"error": "Failed to save file"}), 500

    try:
        load_manual(manual_key, file_path)
        log.info(f"[UPLOAD] ✅ '{manual_key}' ingested successfully")
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        log.error(f"[UPLOAD] ❌ Ingestion failed: {e}")
        return jsonify({"error": "Failed to process PDF. Please try again."}), 500

    AVAILABLE_MANUALS[manual_key] = display_name
    MANUAL_FILES[manual_key] = file_path
    save_uploaded_manual(manual_key, display_name, file_path)
    log.info(f"[UPLOAD] ✅ Manual '{display_name}' registered and ready")

    return jsonify({"success": True, "key": manual_key, "label": display_name})


@app.route("/delete_manual", methods=["POST"])
def delete_manual():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    manual_key = data.get("manual_key", "").strip()

    hardcoded = {"dishwasher_manual", "washing_machine_manual"}
    if manual_key in hardcoded:
        return jsonify({"error": "Default manuals cannot be deleted"}), 403

    if manual_key not in AVAILABLE_MANUALS:
        return jsonify({"error": "Manual not found"}), 404

    try:
        from rag_search import client_chroma
        client_chroma.delete_collection(manual_key)
        log.info(f"[DELETE_MANUAL] ✅ ChromaDB collection '{manual_key}' deleted")
    except Exception as e:
        log.warning(f"[DELETE_MANUAL] ⚠️ Could not delete ChromaDB collection: {e}")

    file_path = MANUAL_FILES.get(manual_key)
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            log.warning(f"[DELETE_MANUAL] ⚠️ Could not delete file: {e}")

    AVAILABLE_MANUALS.pop(manual_key, None)
    MANUAL_FILES.pop(manual_key, None)
    delete_uploaded_manual(manual_key)

    log.info(f"[DELETE_MANUAL] ✅ Manual '{manual_key}' fully removed")
    return jsonify({"success": True})


@app.route("/create_session", methods=["POST"])
def create_new_session():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    session_name = data.get("session_name", "").strip()
    manual_name = data.get("manual_name", "").strip()
    user = session["user"]

    log.info(f"[CREATE_SESSION] User='{user}' | Name='{session_name}' | Manual='{manual_name}'")

    if not session_name or manual_name not in AVAILABLE_MANUALS:
        return jsonify({"error": "Invalid data"}), 400

    create_session(user, session_name, manual_name)
    log.info(f"[CREATE_SESSION] ✅ Session '{session_name}' created for '{user}'")
    return jsonify({"success": True})


@app.route("/ask", methods=["POST"])
def ask():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    query = data.get("query", "").strip()
    session_name = data.get("session_name", "").strip()

    if not query or not session_name:
        return jsonify({"error": "Invalid input"}), 400

    user = session["user"]
    log.info(f"[ASK] ─────────────────────────────────────────────")
    log.info(f"[ASK] User='{user}' | Session='{session_name}'")
    log.info(f"[ASK] Query: '{query[:80]}{'...' if len(query) > 80 else ''}'")

    manual_name = get_session_manual(user, session_name)
    if not manual_name:
        return jsonify({"error": "Session not found"}), 404
    log.info(f"[ASK] ✅ [1/7] Manual resolved → '{manual_name}'")

    session_score = get_session_score(user, session_name)
    log.info(f"[ASK] ✅ [2/7] Session score = {session_score}")

    try:
        save_message(user, session_name, manual_name, query, "user")
    except Exception as e:
        log.error(f"[ASK] ❌ [3/7] Failed to save user message: {e}")

    try:
        chat_history = get_chat_history(user, session_name)
        relevant_docs, confidence = search_manual(query, manual_name)
        log.info(f"[ASK] ✅ [4/7] RAG done — {len(relevant_docs)} chunk(s) | confidence='{confidence}'")
    except Exception as e:
        log.error(f"[ASK] ❌ [4/7] RAG retrieval FAILED: {e}")
        return jsonify({"error": "Retrieval failed. Please try again."}), 500

    try:
        answer = generate_answer(
            query=query,
            context=relevant_docs,
            history=chat_history,
            manual_name=manual_name,
            confidence=confidence,
            session_score=session_score
        )
        log.info(f"[ASK] ✅ [5/7] Answer generated ({len(answer)} chars)")
    except Exception as e:
        log.error(f"[ASK] ❌ [5/7] LLM generation FAILED: {e}")
        return jsonify({"error": "AI generation failed. Please try again."}), 500

    try:
        save_message(user, session_name, manual_name, answer, "ai")
    except Exception as e:
        log.error(f"[ASK] ❌ [6/7] Failed to save AI response: {e}")

    answer_score = analyze_satisfaction(answer=answer, query=query, context_confidence=confidence)
    log.info(f"[ASK] ✅ [7/7] Per-answer score = {answer_score}/10")

    if len(chat_history) % 4 == 0:
        sentiment_score = analyze_conversation_sentiment(chat_history)
        if sentiment_score is not None:
            blended = round((answer_score * 0.6) + (sentiment_score * 0.4), 2)
            update_session(user, session_name, blended)
        else:
            update_session(user, session_name, answer_score)
    else:
        update_session(user, session_name, answer_score)

    log.info(f"[ASK] ✅ Full pipeline complete | confidence='{confidence}' | score={answer_score}")
    log.info(f"[ASK] ─────────────────────────────────────────────")

    return jsonify({"answer": answer, "confidence": confidence, "satisfaction_score": answer_score})


@app.route("/history", methods=["GET"])
def history():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    user = session["user"]
    session_name = request.args.get("session_name", "").strip()
    chats = get_chat_history(user, session_name)
    return jsonify({
        "chats": [{"sender": c[0], "message": c[1], "timestamp": c[2]} for c in chats]
    })


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


@app.route("/rename_session", methods=["POST"])
def rename():
    if "user" not in session:
        return jsonify({"success": False}), 401
    data = request.get_json()
    user = session["user"]
    old_name = data.get("old_name", "")
    new_name = data.get("new_name", "")
    log.info(f"[RENAME] User='{user}' | '{old_name}' → '{new_name}'")
    try:
        rename_session(user, old_name, new_name)
        return jsonify({"success": True})
    except Exception as e:
        log.error(f"[RENAME] ❌ Failed: {e}")
        return jsonify({"success": False}), 500


@app.route("/delete_session", methods=["POST"])
def delete():
    if "user" not in session:
        return jsonify({"success": False}), 401
    data = request.get_json()
    user = session["user"]
    session_name = data.get("session_name", "")
    log.info(f"[DELETE] User='{user}' | Session='{session_name}'")
    try:
        delete_session(user, session_name)
        return jsonify({"success": True})
    except Exception as e:
        log.error(f"[DELETE] ❌ Failed: {e}")
        return jsonify({"success": False}), 500


@app.route("/logout")
def logout():
    user = session.get("user", "unknown")
    session.pop("user", None)
    log.info(f"[LOGOUT] ✅ User '{user}' logged out")
    return redirect(url_for("login"))


if __name__ == "__main__":
    log.info("[RUN] Starting Flask development server...")
    app.run(debug=True, use_reloader=False)