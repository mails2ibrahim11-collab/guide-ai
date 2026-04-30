from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from flask_socketio import SocketIO, join_room, emit
from dotenv import load_dotenv
import os
import re
import json

from logger import get_logger

from database import (
    init_db, login_user, register_user, get_user_role,
    save_message, get_chat_history, get_all_sessions,
    create_session, update_session, get_session_manual,
    get_session_score, rename_session, delete_session,
    get_manual_session_counts, get_active_manual_session_counts,
    save_uploaded_manual, delete_uploaded_manual, get_uploaded_manuals,
    get_manuals_by_owner, get_manual_owner,
    create_call, get_call, update_call_manual, update_call_customer,
    end_call, save_customer_rating, save_call_turn, get_call_turns,
    get_call_score_history, save_call_report, get_agent_reports,
    get_call_report
)

from rag_search import load_manual, search_manual
from llm_suggestions import (
    generate_answer, analyze_satisfaction, analyze_conversation_sentiment,
    grade_agent_turn, generate_call_report, generate_call_summary
)

load_dotenv()

log = get_logger("main")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "guideai-secret-key-change-in-prod")
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_HTTPONLY=True,
)
app.jinja_env.filters['fromjson'] = json.loads
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ── Hardcoded agent credentials (set in .env) ──────────────────
AGENT_ID       = os.getenv("AGENT_ID",       "agent")
AGENT_PASSWORD = os.getenv("AGENT_PASSWORD",  "agent123")

AVAILABLE_MANUALS = {
    "dishwasher_manual":      "Dishwasher",
    "washing_machine_manual": "Washing Machine"
}

MANUAL_FILES = {
    "dishwasher_manual":      "data/manual.pdf",
    "washing_machine_manual": "data/washing_machine.pdf"
}

HARDCODED_MANUAL_KEYS = {"dishwasher_manual", "washing_machine_manual"}

UPLOAD_FOLDER      = "data"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_UPLOAD_MB      = 50

# In-memory state for active calls
active_calls = {}

# In-memory set tracking which call_ids have had the agent join their socket room
# Used so the agent dashboard knows whether to show "waiting" or "in call"
agent_in_call = set()


# ================= HELPERS =================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_manual_key(display_name):
    key = re.sub(r"[^a-z0-9]+", "_", display_name.lower().strip())
    key = key.strip("_")
    return f"{key}_manual"


def score_to_confidence(score):
    if score >= 8:
        return "high"
    elif score >= 5:
        return "medium"
    else:
        return "low"


def require_login(role=None):
    """
    Returns the logged-in user if valid, else None.
    Optionally checks that the user has the required role.
    """
    if "user" not in session:
        return None
    if role and session.get("role") != role:
        return None
    return session["user"]


def get_customer_manuals(customer_id):
    """
    Returns the AVAILABLE_MANUALS dict filtered to what a customer can see:
    all hardcoded manuals + their own uploaded manuals.
    """
    rows = get_manuals_by_owner(customer_id)
    result = {}
    for key, label, file_path, owner in rows:
        if key in AVAILABLE_MANUALS:
            result[key] = AVAILABLE_MANUALS[key]
    return result


# ================= STARTUP =================

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

    # Ensure agent account exists in DB
    from database import connect, hash_password
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT unique_id FROM users WHERE unique_id=?", (AGENT_ID,))
    if not c.fetchone():
        c.execute(
            "INSERT INTO users (unique_id, password, role) VALUES (?, ?, ?)",
            (AGENT_ID, hash_password(AGENT_PASSWORD), "agent")
        )
        conn.commit()
        log.info(f"[STARTUP] ✅ Agent account '{AGENT_ID}' created")
    else:
        # Always sync password and role in case .env changed
        c.execute(
            "UPDATE users SET password=?, role='agent' WHERE unique_id=?",
            (hash_password(AGENT_PASSWORD), AGENT_ID)
        )
        conn.commit()
        log.info(f"[STARTUP] ✅ Agent account '{AGENT_ID}' verified")
    conn.close()

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
            log.warning(f"[STARTUP] ⚠️ Uploaded manual '{key}' file missing — skipping")
            continue
        try:
            load_manual(key, file_path)
            AVAILABLE_MANUALS[key] = label
            MANUAL_FILES[key]      = file_path
            log.info(f"[STARTUP] ✅ Restored '{key}'")
        except Exception as e:
            log.error(f"[STARTUP] ❌ Could not restore '{key}': {e}")

    log.info("[STARTUP] ✅ Startup complete — Flask is ready")
    log.info("=" * 60)


startup()


# ================= AUTH ROUTES =================

@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data      = request.get_json()
        unique_id = data.get("unique_id", "").strip()
        password  = data.get("password",  "").strip()
        role_hint = data.get("role", "customer").strip()   # 'agent' or 'customer' from UI toggle

        log.info(f"[LOGIN] Attempt — id='{unique_id}' role_hint='{role_hint}'")

        if not unique_id or not password:
            return jsonify({"success": False, "error": "Please fill in all fields."})

        role = login_user(unique_id, password)   # returns role string or None

        if role is None:
            log.warning(f"[LOGIN] ❌ Bad credentials for '{unique_id}'")
            return jsonify({"success": False, "error": "Invalid ID or password."})

        # Prevent a customer account logging in via the agent toggle and vice-versa
        if role != role_hint:
            log.warning(f"[LOGIN] ❌ Role mismatch — account is '{role}', tried as '{role_hint}'")
            return jsonify({"success": False,
                            "error": f"This account is registered as a {role}. Please select the correct role."})

        session["user"] = unique_id
        session["role"] = role
        log.info(f"[LOGIN] ✅ '{unique_id}' authenticated as '{role}'")
        return jsonify({"success": True, "role": role})

    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    """Customer-only registration. Agents are hardcoded."""
    data      = request.get_json()
    unique_id = data.get("unique_id", "").strip()
    password  = data.get("password",  "").strip()

    log.info(f"[REGISTER] Attempt for '{unique_id}'")

    if not unique_id or not password:
        return jsonify({"success": False, "error": "Please fill in all fields."})

    if unique_id == AGENT_ID:
        return jsonify({"success": False, "error": "That ID is reserved."})

    if register_user(unique_id, password, role='customer'):
        log.info(f"[REGISTER] ✅ Customer '{unique_id}' created")
        return jsonify({"success": True})
    else:
        log.warning(f"[REGISTER] ❌ '{unique_id}' already exists")
        return jsonify({"success": False, "error": "Username already taken."})


@app.route("/logout")
def logout():
    user = session.get("user", "unknown")
    session.clear()
    log.info(f"[LOGOUT] ✅ '{user}' logged out")
    return redirect(url_for("login"))


# ================= DASHBOARD =================

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    role = session.get("role", "customer")
    if role == "agent":
        # Agent dashboard — shows call history + all manuals
        reports = get_agent_reports(session["user"])
        return render_template(
            "agent_dashboard.html",
            user=session["user"],
            reports=reports,
            manuals=list(AVAILABLE_MANUALS.items())
        )
    else:
        # Customer dashboard — chat interface
        return render_template("dashboard.html", user=session["user"])


# ================= MANUAL ROUTES =================

@app.route("/manuals", methods=["GET"])
def manuals():
    """
    Returns manuals available to the caller.
    - Customers: all hardcoded manuals + their own uploaded manuals ONLY
    - Agents / unauthenticated: all manuals
    Hardcoded manuals are always included regardless of DB state.
    """
    if "user" in session and session.get("role") == "customer":
        customer_id = session["user"]

        # Always start with hardcoded manuals (these live in memory, not DB)
        result = [
            {"key": key, "label": AVAILABLE_MANUALS[key], "is_own": False}
            for key in HARDCODED_MANUAL_KEYS
            if key in AVAILABLE_MANUALS
        ]

        # Then add this customer's own uploaded manuals from DB
        rows = get_manuals_by_owner(customer_id)
        for key, label, file_path, owner in rows:
            if key in HARDCODED_MANUAL_KEYS:
                continue   # already added above
            if key not in AVAILABLE_MANUALS:
                continue   # not loaded in memory (file missing etc.)
            result.append({
                "key":    key,
                "label":  AVAILABLE_MANUALS[key],
                "is_own": True
            })

        return jsonify({"manuals": result})

    # Agent or unauthenticated — return everything
    return jsonify({
        "manuals": [{"key": k, "label": v, "is_own": False}
                    for k, v in AVAILABLE_MANUALS.items()]
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
    current_user   = session["user"]
    is_customer    = session.get("role") == "customer"

    # Build visible key set
    if is_customer:
        # Hardcoded always visible + their own uploads
        visible_keys = set(HARDCODED_MANUAL_KEYS)
        rows = get_manuals_by_owner(current_user)
        for row in rows:
            visible_keys.add(row[0])
    else:
        visible_keys = set(AVAILABLE_MANUALS.keys())

    manuals_list = []
    for key, label in AVAILABLE_MANUALS.items():
        if key not in visible_keys:
            continue
        owner      = get_manual_owner(key)  # None means key not in DB (hardcoded)
        is_default = key in HARDCODED_MANUAL_KEYS
        is_own     = (not is_default) and (owner == current_user or owner is None)
        manuals_list.append({
            "key":           key,
            "label":         label,
            "file_path":     MANUAL_FILES.get(key, ""),
            "session_count": session_counts.get(key, 0),
            "active_count":  active_counts.get(key, 0),
            "is_default":    is_default,
            "is_own":        is_own
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

    owner = session["user"]
    AVAILABLE_MANUALS[manual_key] = display_name
    MANUAL_FILES[manual_key]      = file_path
    save_uploaded_manual(manual_key, display_name, file_path, owner=owner)
    log.info(f"[UPLOAD] ✅ Manual '{display_name}' registered by '{owner}'")

    return jsonify({"success": True, "key": manual_key, "label": display_name})


@app.route("/delete_manual", methods=["POST"])
def delete_manual():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data       = request.get_json()
    manual_key = data.get("manual_key", "").strip()

    if manual_key in HARDCODED_MANUAL_KEYS:
        return jsonify({"error": "Default manuals cannot be deleted"}), 403

    if manual_key not in AVAILABLE_MANUALS:
        return jsonify({"error": "Manual not found"}), 404

    # Customers can only delete their own uploaded manuals.
    # owner == None means it was uploaded before the owner migration — allow deletion
    # since only the uploader would know it exists.
    if session.get("role") == "customer":
        owner = get_manual_owner(manual_key)
        # owner is None  → key not in DB at all (shouldn't happen but allow)
        # owner is the NULL sentinel (row exists, owner column is NULL) → allow
        # owner == session user → allow
        # owner == someone else → deny
        if owner is not None and owner != "system" and owner != session["user"]:
            return jsonify({"error": "You can only delete your own manuals"}), 403

    try:
        from rag_search import client_chroma
        client_chroma.delete_collection(manual_key)
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

    log.info(f"[DELETE_MANUAL] ✅ '{manual_key}' removed")
    return jsonify({"success": True})


# ================= CHAT SESSION ROUTES =================

@app.route("/create_session", methods=["POST"])
def create_new_session():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data         = request.get_json()
    session_name = data.get("session_name", "").strip()
    manual_name  = data.get("manual_name",  "").strip()
    user         = session["user"]

    if not session_name or manual_name not in AVAILABLE_MANUALS:
        return jsonify({"error": "Invalid data"}), 400

    # Check for duplicate name before inserting
    existing = get_session_manual(user, session_name)
    if existing:
        return jsonify({
            "error": f'A chat named "{session_name}" already exists. Please choose a different name.'
        }), 409

    create_session(user, session_name, manual_name)
    log.info(f"[CREATE_SESSION] ✅ Session '{session_name}' for '{user}'")
    return jsonify({"success": True})


@app.route("/ask", methods=["POST"])
def ask():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data         = request.get_json()
    query        = data.get("query",        "").strip()
    session_name = data.get("session_name", "").strip()

    if not query or not session_name:
        return jsonify({"error": "Invalid input"}), 400

    user = session["user"]
    log.info(f"[ASK] User='{user}' | Session='{session_name}'")

    manual_name = get_session_manual(user, session_name)
    if not manual_name:
        return jsonify({"error": "Session not found"}), 404

    # Check if the manual still exists — it may have been deleted after the
    # session was created. Only applies to uploaded manuals; hardcoded ones
    # are always present.
    if manual_name not in HARDCODED_MANUAL_KEYS and manual_name not in AVAILABLE_MANUALS:
        removed_answer = (
            "⚠️ The manual for this chat has been removed. "
            "Please go to **Upload Manual** and re-upload the PDF to continue "
            "getting answers in this session."
        )
        # Save the notice as an AI message so it appears in history too
        try:
            save_message(user, session_name, manual_name, removed_answer, "ai")
        except Exception:
            pass
        return jsonify({
            "answer":             removed_answer,
            "confidence":         "none",
            "satisfaction_score": 1
        })

    is_uploaded   = manual_name not in HARDCODED_MANUAL_KEYS
    session_score = get_session_score(user, session_name)

    try:
        save_message(user, session_name, manual_name, query, "user")
    except Exception as e:
        log.error(f"[ASK] ❌ Failed to save user message: {e}")

    try:
        chat_history              = get_chat_history(user, session_name)
        relevant_docs, confidence = search_manual(query, manual_name)
    except Exception as e:
        log.error(f"[ASK] ❌ RAG retrieval FAILED: {e}")
        return jsonify({"error": "Retrieval failed. Please try again."}), 500

    generation_confidence = confidence if not is_uploaded else "medium"

    try:
        answer = generate_answer(
            query=query,
            context=relevant_docs,
            history=chat_history,
            manual_name=manual_name,
            confidence=generation_confidence,
            session_score=session_score
        )
    except Exception as e:
        log.error(f"[ASK] ❌ LLM generation FAILED: {e}")
        return jsonify({"error": "AI generation failed. Please try again."}), 500

    try:
        save_message(user, session_name, manual_name, answer, "ai")
    except Exception as e:
        log.error(f"[ASK] ❌ Failed to save AI response: {e}")

    answer_score = analyze_satisfaction(
        answer=answer,
        query=query,
        context_confidence=generation_confidence,
        is_uploaded=is_uploaded
    )

    if is_uploaded:
        confidence = score_to_confidence(answer_score)

    if len(chat_history) % 4 == 0:
        sentiment_score = analyze_conversation_sentiment(chat_history)
        if sentiment_score is not None:
            blended = round((answer_score * 0.6) + (sentiment_score * 0.4), 2)
            update_session(user, session_name, blended)
        else:
            update_session(user, session_name, answer_score)
    else:
        update_session(user, session_name, answer_score)

    return jsonify({"answer": answer, "confidence": confidence,
                    "satisfaction_score": answer_score})


@app.route("/history", methods=["GET"])
def history():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    user         = session["user"]
    session_name = request.args.get("session_name", "").strip()
    chats        = get_chat_history(user, session_name)
    return jsonify({
        "chats": [{"sender": c[0], "message": c[1], "timestamp": c[2]} for c in chats]
    })


@app.route("/sessions", methods=["GET"])
def get_sessions():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    user         = session["user"]
    all_sessions = get_all_sessions(user)
    return jsonify({
        "sessions": [
            {
                "session_name": s[0],
                "manual_name":  s[1],
                "manual_label": AVAILABLE_MANUALS.get(s[1], s[1]),
                "last_used":    s[2],
                "score":        s[3]
            }
            for s in all_sessions
        ]
    })


@app.route("/rename_session", methods=["POST"])
def rename():
    if "user" not in session:
        return jsonify({"success": False}), 401
    data     = request.get_json()
    user     = session["user"]
    old_name = data.get("old_name", "")
    new_name = data.get("new_name", "")
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
    data         = request.get_json()
    user         = session["user"]
    session_name = data.get("session_name", "")
    try:
        delete_session(user, session_name)
        return jsonify({"success": True})
    except Exception as e:
        log.error(f"[DELETE] ❌ Failed: {e}")
        return jsonify({"success": False}), 500


# ================= CALL HTTP ROUTES =================

@app.route("/call/new", methods=["GET", "POST"])
def call_new():
    """
    Agent-only route — kept for direct agent-initiated calls if needed.
    Primary flow is now customer-initiated via /call/request.
    """
    if not require_login(role="agent"):
        return redirect(url_for("login"))

    if request.method == "POST":
        data        = request.get_json()
        manual_name = data.get("manual_name", "").strip()

        if manual_name not in AVAILABLE_MANUALS:
            return jsonify({"error": "Invalid manual"}), 400

        agent   = session["user"]
        call_id = create_call(agent, manual_name)

        active_calls[call_id] = {
            "manual":          manual_name,
            "agent":           agent,
            "last_suggestion": None,
            "last_query":      None,
            "score_history":   [],
            "running_score":   5.0,
            "chat_history":    []   # text chat before call escalation
        }

        log.info(f"[CALL_NEW] ✅ Call '{call_id}' by '{agent}' | manual='{manual_name}'")
        return jsonify({"success": True, "call_id": call_id})

    return render_template("call_new.html", user=session["user"],
                           manuals=list(AVAILABLE_MANUALS.items()))


@app.route("/call/request", methods=["POST"])
def call_request():
    """
    Customer initiates a call request from the chat UI.
    Creates the call record and notifies the agent via SocketIO.
    Returns the call_id so the customer can join the call room.
    """
    if not require_login(role="customer"):
        return jsonify({"error": "Not logged in"}), 401

    data         = request.get_json()
    manual_name  = data.get("manual_name",  "").strip()
    session_name = data.get("session_name", "").strip()

    if manual_name not in AVAILABLE_MANUALS:
        return jsonify({"error": "Invalid manual"}), 400

    customer_id = session["user"]
    agent       = AGENT_ID   # single agent for now

    # Fetch text chat history to pass as context
    chat_history = []
    if session_name:
        raw = get_chat_history(customer_id, session_name)
        chat_history = [{"sender": r[0], "message": r[1]} for r in raw]

    call_id = create_call(agent, manual_name, customer_id=customer_id)

    active_calls[call_id] = {
        "manual":          manual_name,
        "agent":           agent,
        "customer_id":     customer_id,
        "session_name":    session_name,   # ← needed to write back to chat history
        "last_suggestion": None,
        "last_query":      None,
        "score_history":   [],
        "running_score":   5.0,
        "chat_history":    chat_history
    }

    # Notify agent dashboard via socket
    socketio.emit("incoming_call", {
        "call_id":      call_id,
        "customer_id":  customer_id,
        "manual_label": AVAILABLE_MANUALS.get(manual_name, manual_name),
        "manual_name":  manual_name,
        "chat_preview": chat_history[-3:] if chat_history else []
    }, room=f"agent_{agent}")

    log.info(f"[CALL_REQUEST] ✅ Call '{call_id}' requested by customer '{customer_id}'")
    return jsonify({"success": True, "call_id": call_id})


@app.route("/call/<call_id>")
def call_customer(call_id):
    """Customer call view — no login required."""
    call = get_call(call_id)
    if not call or call["status"] == "ended":
        return render_template("call_ended.html"), 404

    manual_label = AVAILABLE_MANUALS.get(call["manual_name"], call["manual_name"])
    return render_template(
        "call_customer.html",
        call_id=call_id,
        manual_label=manual_label,
        manuals=list(AVAILABLE_MANUALS.items())
    )


@app.route("/call/<call_id>/agent")
def call_agent(call_id):
    """Agent call monitor view."""
    if not require_login(role="agent"):
        return redirect(url_for("login"))

    call = get_call(call_id)
    if not call:
        return "Call not found", 404

    # Load text chat history for context panel
    chat_history = active_calls.get(call_id, {}).get("chat_history", [])

    manual_label = AVAILABLE_MANUALS.get(call["manual_name"], call["manual_name"])
    return render_template(
        "call_agent.html",
        call_id=call_id,
        user=session["user"],
        manual_label=manual_label,
        manuals=list(AVAILABLE_MANUALS.items()),
        chat_history=chat_history
    )


@app.route("/call/<call_id>/report")
def call_report_view(call_id):
    if not require_login(role="agent"):
        return redirect(url_for("login"))

    report = get_call_report(call_id)
    if not report:
        return "Report not found", 404

    return render_template("call_report.html",
                           report=report,
                           user=session["user"])


@app.route("/call/reports", methods=["GET"])
def agent_reports():
    if not require_login(role="agent"):
        return jsonify({"error": "Not logged in"}), 401
    reports = get_agent_reports(session["user"])
    return jsonify({"reports": reports})


# ================= SOCKET.IO EVENTS =================

@socketio.on("connect")
def on_connect():
    log.debug(f"[SOCKET] Client connected: {request.sid}")


# ── Agent room: agent joins a persistent room keyed to their ID
#    so incoming_call notifications work even on the dashboard ──
@socketio.on("agent_online")
def on_agent_online(data):
    """Agent connects to dashboard — joins their personal notification room."""
    try:
        agent_id = data.get("agent_id", "").strip()
        if agent_id:
            join_room(f"agent_{agent_id}")
            log.info(f"[SOCKET] Agent '{agent_id}' online — joined room 'agent_{agent_id}'")
    except Exception as e:
        log.error(f"[SOCKET] ❌ agent_online error: {e}", exc_info=True)


@socketio.on("customer_join")
def on_customer_join(data):
    try:
        call_id     = data.get("call_id", "").strip().upper()
        manual_key  = data.get("manual_name", "").strip()
        customer_id = data.get("customer_id", "").strip()

        call = get_call(call_id)
        if not call or call["status"] == "ended":
            emit("error", {"message": "Call not found or already ended"})
            return

        join_room(call_id)
        join_room(f"{call_id}_customer")

        if manual_key and manual_key in AVAILABLE_MANUALS:
            update_call_manual(call_id, manual_key)
            if call_id in active_calls:
                active_calls[call_id]["manual"] = manual_key

        if customer_id:
            update_call_customer(call_id, customer_id)

        manual_name  = active_calls.get(call_id, {}).get("manual", call["manual_name"])
        manual_label = AVAILABLE_MANUALS.get(manual_name, "Unknown")

        log.info(f"[SOCKET] Customer joined call '{call_id}' | manual='{manual_label}'")

        emit("customer_joined", {
            "manual_name":  manual_name,
            "manual_label": manual_label,
            "customer_id":  customer_id
        }, room=f"{call_id}_agent")

        emit("manual_confirmed", {"manual_label": manual_label})

    except Exception as e:
        log.error(f"[SOCKET] ❌ customer_join error: {e}", exc_info=True)
        emit("error", {"message": "Failed to join call"})


@socketio.on("agent_join")
def on_agent_join(data):
    try:
        call_id = data.get("call_id", "").strip().upper()
        call    = get_call(call_id)
        if not call:
            emit("error", {"message": "Call not found"})
            return

        join_room(call_id)
        join_room(f"{call_id}_agent")

        if call_id not in active_calls:
            active_calls[call_id] = {
                "manual":          call["manual_name"],
                "agent":           call["agent"],
                "last_suggestion": None,
                "last_query":      None,
                "score_history":   [],
                "running_score":   5.0,
                "chat_history":    []
            }

        agent_in_call.add(call_id)
        log.info(f"[SOCKET] Agent joined call '{call_id}'")

        emit("agent_joined", {"call_id": call_id})
        emit("agent_joined_call", {
            "message": "Agent has joined. You can now start chatting."
        }, room=f"{call_id}_customer")

    except Exception as e:
        log.error(f"[SOCKET] ❌ agent_join error: {e}", exc_info=True)
        emit("error", {"message": "Failed to join call"})


@socketio.on("customer_message")
def on_customer_message(data):
    call_id  = data.get("call_id", "").strip().upper()
    text     = data.get("text",    "").strip()
    is_voice = data.get("is_voice", False)

    if not text or call_id not in active_calls:
        return

    call_state  = active_calls[call_id]
    manual_name = call_state["manual"]

    log.info(f"[CALL_MSG] Call='{call_id}' | Customer: '{text[:60]}'")

    # Forward raw transcription to agent
    emit("live_transcription", {"text": text, "is_voice": is_voice},
         room=f"{call_id}_agent")

    # RAG + suggestion
    try:
        is_uploaded           = manual_name not in HARDCODED_MANUAL_KEYS
        relevant_docs, confidence = search_manual(text, manual_name)
        generation_confidence = confidence if not is_uploaded else "medium"

        suggestion = generate_answer(
            query=text,
            context=relevant_docs,
            manual_name=manual_name,
            confidence=generation_confidence,
            session_score=call_state["running_score"]
        )

        call_state["last_suggestion"] = suggestion
        call_state["last_query"]      = text

        emit("rag_suggestion", {
            "suggestion":     suggestion,
            "confidence":     confidence,
            "original_query": text
        }, room=f"{call_id}_agent")

    except Exception as e:
        log.error(f"[CALL_RAG] ❌ RAG failed for call '{call_id}': {e}")
        emit("rag_suggestion", {
            "suggestion":     "Could not retrieve a suggestion. Please answer manually.",
            "confidence":     "none",
            "original_query": text
        }, room=f"{call_id}_agent")


@socketio.on("agent_response")
def on_agent_response(data):
    try:
        call_id        = data.get("call_id",       "").strip().upper()
        agent_response = data.get("response",      "").strip()
        edited_query   = data.get("edited_query",  "").strip()
        agent_used_ai  = data.get("agent_used_ai", 0)

        if call_id not in active_calls:
            return

        call_state      = active_calls[call_id]
        manual_name     = call_state["manual"]
        last_query      = call_state["last_query"]      or ""
        last_suggestion = call_state["last_suggestion"] or ""
        query_for_grade = edited_query if edited_query else last_query

        turn_score = grade_agent_turn(
            customer_query=query_for_grade,
            ai_suggestion=last_suggestion,
            agent_actual_response=agent_response,
            manual_name=manual_name
        )

        old_score   = call_state["running_score"]
        new_running = round((0.7 * old_score) + (0.3 * turn_score), 2)
        call_state["running_score"] = new_running
        call_state["score_history"].append(turn_score)

        save_call_turn(
            call_id=call_id,
            speaker="customer",
            original_text=last_query,
            edited_text=edited_query or last_query,
            ai_suggestion=last_suggestion,
            rag_confidence="",
            agent_used_ai=agent_used_ai,
            turn_score=turn_score
        )

        log.info(f"[CALL_GRADE] Call='{call_id}' | Turn={turn_score} | Running={new_running}")

        emit("agent_message", {"text": agent_response}, room=f"{call_id}_customer")

        emit("live_score_update", {
            "turn_score":    turn_score,
            "running_score": new_running,
            "score_history": call_state["score_history"]
        }, room=f"{call_id}_agent")

    except Exception as e:
        log.error(f"[SOCKET] ❌ agent_response error: {e}", exc_info=True)


@socketio.on("manual_override")
def on_manual_override(data):
    try:
        call_id    = data.get("call_id",    "").strip().upper()
        manual_key = data.get("manual_name", "").strip()

        if manual_key not in AVAILABLE_MANUALS:
            emit("error", {"message": "Manual not found"})
            return

        update_call_manual(call_id, manual_key)
        if call_id in active_calls:
            active_calls[call_id]["manual"] = manual_key

        manual_label = AVAILABLE_MANUALS[manual_key]
        log.info(f"[CALL_MANUAL] Call='{call_id}' | Manual → '{manual_key}'")
        emit("manual_changed", {"manual_label": manual_label}, room=call_id)

    except Exception as e:
        log.error(f"[SOCKET] ❌ manual_override error: {e}", exc_info=True)


def _finalise_call(call_id):
    """
    Shared logic for ending a call — called by both end_call and
    customer_end_call so behaviour is identical regardless of who ends it.
    """
    call = get_call(call_id)
    if not call:
        return

    call_state   = active_calls.get(call_id, {})
    manual_name  = call_state.get("manual",       call["manual_name"])
    agent        = call["agent"]
    final_score  = call_state.get("running_score", 5.0)

    # customer_id — prefer active_calls, fall back to DB
    customer_id  = call_state.get("customer_id") or call.get("customer_id")

    # session_name — prefer active_calls
    session_name = call_state.get("session_name", "").strip()

    # Fallback: if session_name is missing, try to find the most recent session
    # for this customer that uses this manual
    if not session_name and customer_id:
        try:
            all_s = get_all_sessions(customer_id)
            # Most recent session using the same manual
            match = next((s for s in all_s if s[1] == manual_name), None)
            if match:
                session_name = match[0]
                log.info(f"[CALL_END] session_name fallback → '{session_name}'")
        except Exception as e:
            log.warning(f"[CALL_END] session_name fallback failed: {e}")

    log.info(f"[CALL_END] Finalising call='{call_id}' | customer='{customer_id}' | session='{session_name}'")

    # Mark ended in DB
    end_call(call_id, final_score)

    # Ask customer to rate
    emit("rating_request", {}, room=f"{call_id}_customer")

    # Generate report
    turns = get_call_turns(call_id)

    # 3-4 line summary for customer
    summary = generate_call_summary(turns, manual_name)

    report_text = generate_call_report(
        call_id=call_id,
        agent=agent,
        manual_name=manual_name,
        turns=turns,
        overall_score=final_score,
        customer_rating=call.get("customer_rating") or 0
    )

    save_call_report(
        call_id=call_id,
        agent=agent,
        report_text=report_text,
        transcript=json.dumps([{
            "speaker":   t["speaker"],
            "text":      t["edited_text"] or t["original_text"],
            "timestamp": t["timestamp"]
        } for t in turns]),
        overall_score=final_score,
        customer_rating=call.get("customer_rating") or 0
    )

    # ── Write call record into the customer's chat session ───────────
    if customer_id and session_name:
        try:
            manual_label = AVAILABLE_MANUALS.get(manual_name, manual_name)
            call_marker  = (
                f"📞 Live support call — Call ID: {call_id} | "
                f"Manual: {manual_label} | "
                f"Score: {round(final_score, 1)}/10"
            )
            save_message(customer_id, session_name, manual_name, call_marker, "system")
            if summary:
                save_message(customer_id, session_name, manual_name, summary, "ai")
            log.info(f"[CALL_END] ✅ Call record saved to session '{session_name}'")
        except Exception as e:
            log.error(f"[CALL_END] ❌ Failed to write call record to chat: {e}")
    else:
        log.warning(f"[CALL_END] ⚠️ Skipping chat write — customer_id='{customer_id}' session='{session_name}'")

    log.info(f"[CALL_END] Call='{call_id}' finalised | Score={final_score}")

    # Notify agent with report link
    emit("call_ended", {
        "report_url":  f"/call/{call_id}/report",
        "final_score": final_score,
        "report":      report_text
    }, room=f"{call_id}_agent")

    # Notify customer with summary
    emit("call_summary", {
        "summary": summary
    }, room=f"{call_id}_customer")

    active_calls.pop(call_id, None)
    agent_in_call.discard(call_id)


@socketio.on("end_call")
def on_end_call(data):
    """Agent ends the call."""
    call_id = data.get("call_id", "").strip().upper()
    log.info(f"[CALL_END] Agent ended call '{call_id}'")
    try:
        _finalise_call(call_id)
    except Exception as e:
        log.error(f"[CALL_END] ❌ _finalise_call crashed for '{call_id}': {e}", exc_info=True)


@socketio.on("customer_end_call")
def on_customer_end_call(data):
    """Customer ends the call — triggers same finalisation as agent ending."""
    call_id = data.get("call_id", "").strip().upper()
    log.info(f"[CALL_END] Customer ended call '{call_id}'")
    emit("customer_left", {"message": "Customer ended the call."}, room=f"{call_id}_agent")
    try:
        _finalise_call(call_id)
    except Exception as e:
        log.error(f"[CALL_END] ❌ _finalise_call crashed for '{call_id}': {e}", exc_info=True)


@socketio.on("customer_rating")
def on_customer_rating(data):
    call_id = data.get("call_id", "").strip().upper()
    rating  = data.get("rating", 5)

    try:
        rating = max(0, min(int(rating), 10))
    except (ValueError, TypeError):
        rating = 5

    # Save to calls table
    save_customer_rating(call_id, rating)
    log.info(f"[CALL_RATING] Call='{call_id}' | Rating={rating}/10")

    # Patch call_reports so the dashboard shows the correct rating on refresh
    from database import connect as db_connect
    try:
        conn = db_connect()
        c    = conn.cursor()
        c.execute(
            "UPDATE call_reports SET customer_rating=? WHERE call_id=?",
            (rating, call_id)
        )
        conn.commit()
        conn.close()
        log.info(f"[CALL_RATING] ✅ call_reports patched for '{call_id}'")
    except Exception as e:
        log.error(f"[CALL_RATING] ❌ Failed to patch call_reports: {e}")

    # Look up agent for this call so we can notify their dashboard room
    call = get_call(call_id)
    agent_id = call["agent"] if call else None

    # Emit to both the call room AND the agent's persistent dashboard room
    emit("customer_rated", {"rating": rating, "call_id": call_id},
         room=f"{call_id}_agent")
    if agent_id:
        socketio.emit("customer_rated", {"rating": rating, "call_id": call_id},
                      room=f"agent_{agent_id}")

    # Redirect customer to dashboard
    emit("go_dashboard", {}, room=f"{call_id}_customer")


# ================= WEBRTC SIGNALLING =================
# These are pure relay events — the server never reads the content,
# just forwards between customer and agent rooms.
# WebRTC handles the actual peer-to-peer audio negotiation.

@socketio.on("webrtc_offer")
def on_webrtc_offer(data):
    try:
        call_id = data.get("call_id", "").strip().upper()
        log.info(f"[WEBRTC] Offer received for call '{call_id}' — relaying to agent")
        emit("webrtc_offer", data, room=f"{call_id}_agent")
    except Exception as e:
        log.error(f"[WEBRTC] ❌ webrtc_offer error: {e}", exc_info=True)


@socketio.on("webrtc_answer")
def on_webrtc_answer(data):
    try:
        call_id = data.get("call_id", "").strip().upper()
        log.info(f"[WEBRTC] Answer received for call '{call_id}' — relaying to customer")
        emit("webrtc_answer", data, room=f"{call_id}_customer")
    except Exception as e:
        log.error(f"[WEBRTC] ❌ webrtc_answer error: {e}", exc_info=True)


@socketio.on("ice_candidate")
def on_ice_candidate(data):
    try:
        call_id = data.get("call_id", "").strip().upper()
        sender  = data.get("sender", "customer")
        if sender == "customer":
            emit("ice_candidate", data, room=f"{call_id}_agent")
        else:
            emit("ice_candidate", data, room=f"{call_id}_customer")
        log.debug(f"[WEBRTC] ICE candidate relayed for call '{call_id}' from '{sender}'")
    except Exception as e:
        log.error(f"[WEBRTC] ❌ ice_candidate error: {e}", exc_info=True)


@socketio.on("webrtc_end")
def on_webrtc_end(data):
    try:
        call_id = data.get("call_id", "").strip().upper()
        sender  = data.get("sender", "customer")
        log.info(f"[WEBRTC] Voice call ended by '{sender}' for call '{call_id}'")
        if sender == "customer":
            emit("webrtc_end", data, room=f"{call_id}_agent")
        else:
            emit("webrtc_end", data, room=f"{call_id}_customer")
    except Exception as e:
        log.error(f"[WEBRTC] ❌ webrtc_end error: {e}", exc_info=True)


# ================= RUN =================

if __name__ == "__main__":
    log.info("[RUN] Starting Flask-SocketIO development server...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)