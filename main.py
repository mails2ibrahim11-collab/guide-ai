from flask import Flask, request, jsonify, session, redirect, url_for, render_template, send_file, send_from_directory
from flask_socketio import SocketIO, join_room, emit
from dotenv import load_dotenv
from groq import Groq as GroqClient
import os
import re
import json
import io

from logger import get_logger

from database import (
    init_db, login_user, register_user,
    save_message, get_chat_history, get_recent_chat_history, get_all_sessions,
    create_session, update_session, get_session_manual,
    get_session_score, rename_session, delete_session,
    get_manual_session_counts, get_active_manual_session_counts,
    save_uploaded_manual, delete_uploaded_manual, get_uploaded_manuals,
    get_manuals_by_owner, get_manual_owner,
    create_call, get_call, update_call_manual, update_call_customer,
    end_call, save_customer_rating, save_call_turn, get_call_turns,
    save_call_report, get_agent_reports,
    get_call_report, update_call_report_rating, save_citation_feedback,
    get_citation_feedback
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

# ── LiveKit config (read once at startup) ──────────────────────
def _build_livekit_url(raw: str) -> str:
    url = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', raw).strip()
    url = re.sub(r'[\[\]()]', '', url).strip()
    if url and not url.startswith('wss://'):
        url = 'wss://' + url.lstrip('/')
    if not url or 'livekit.cloud' not in url or '[' in url or '(' in url:
        return 'wss://guideai-sr6dd6z9.livekit.cloud'
    return url

LIVEKIT_API_KEY    = os.getenv("LIVEKIT_API_KEY",    "").strip()
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "").strip()
LIVEKIT_URL        = _build_livekit_url(os.getenv("LIVEKIT_URL", ""))

# ── Groq client singleton ───────────────────────────────────────
_groq_client = GroqClient(api_key=os.getenv("GROQ_API_KEY"))

UPLOAD_FOLDER      = "data"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_UPLOAD_MB      = 50

# In-memory state for active calls
active_calls = {}

# In-memory set tracking which call_ids have had the agent join their socket room
# Used so the agent dashboard knows whether to show "waiting" or "in call"
agent_in_call = set()

FILLER_ONLY_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you|okay|ok|hmm|uh|um|yep|yeah|bye|goodbye)[\s!.?,]*$",
    r"^(can you hear me|one sec|just a second|hold on|yes)[\s!.?,]*$"
]


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


def is_actionable_query(text):
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    if len(cleaned) < 3:
        return False
    if any(re.match(p, cleaned) for p in FILLER_ONLY_PATTERNS):
        return False
    return ("?" in cleaned) or any(
        kw in cleaned for kw in ["how", "what", "where", "why", "issue", "problem", "error", "not working", "manual"]
    )


def extract_page_numbers(chunks):
    pages = set()
    for chunk in chunks:
        for m in re.findall(r"\[Page\s+(\d+)\]", chunk):
            pages.add(int(m))
    return sorted(pages)


def page_number_from_chunk(chunk):
    match = re.search(r"\[Page\s+(\d+)\]", chunk or "")
    return int(match.group(1)) if match else None


def strip_page_markers(text):
    return re.sub(r"\[Page\s+\d+\]", " ", text or "").strip()


SOURCE_STOPWORDS = {
    "this", "that", "with", "from", "your", "have", "will", "when",
    "which", "there", "their", "about", "using", "manual", "page",
    "into", "then", "than", "also", "what", "where", "how", "why",
    "does", "should", "could", "would", "please", "tell", "help",
    "just", "very", "some", "each", "been", "they", "them"
}


def source_keywords(*texts, limit=12):
    keywords = []
    for text in texts:
        cleaned = strip_page_markers(text).lower()
        for word in re.findall(r"[a-zA-Z0-9]{4,}", cleaned):
            if word in SOURCE_STOPWORDS or word in keywords:
                continue
            keywords.append(word)
            if len(keywords) >= limit:
                return keywords
    return keywords


def source_phrases(text, max_phrases=6):
    cleaned = strip_page_markers(text).lower()
    words   = re.findall(r"[a-zA-Z]{3,}", cleaned)
    phrases = []
    for i in range(len(words) - 1):
        if words[i] not in SOURCE_STOPWORDS and words[i+1] not in SOURCE_STOPWORDS:
            phrase = f"{words[i]} {words[i+1]}"
            if phrase not in phrases:
                phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return phrases


def _stem(word):
    for suffix in ("ing", "tion", "ness", "ment", "ers", "ed", "es", "er", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def source_excerpt(chunk, max_words=40):
    cleaned   = strip_page_markers(chunk)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    best      = ""
    best_len  = 0
    for s in sentences:
        s = s.strip()
        wc = len(s.split())
        if wc > best_len and wc >= 6:
            best     = s
            best_len = wc
    if not best:
        best = cleaned
    words = best.split()
    return " ".join(words[:max_words])


def source_match_score(query, answer, chunk):
    chunk_text     = strip_page_markers(chunk).lower()
    answer_terms   = source_keywords(answer, limit=12)
    answer_phrases = source_phrases(answer, max_phrases=6)
    query_terms    = [t for t in source_keywords(query, limit=8) if t not in answer_terms]

    score  = sum(3   for t in answer_terms   if t in chunk_text)
    score += sum(4   for p in answer_phrases  if p in chunk_text)
    score += sum(1   for t in query_terms     if t in chunk_text)
    for t in answer_terms:
        if t not in chunk_text:
            stem = _stem(t)
            if stem != t and stem in chunk_text:
                score += 0.5
    return score


def source_confidence_label(score):
    if score >= 10:
        return "high"
    if score >= 5:
        return "medium"
    if score > 0:
        return "low"
    return "unknown"


def infer_source_pages_from_pdf(manual_name, chunks, answer="", max_pages=5):
    file_path = MANUAL_FILES.get(manual_name)
    if not file_path or not os.path.exists(file_path):
        return []

    try:
        import fitz
    except Exception:
        return []

    # Answer terms weighted 2×, chunk terms weighted 1×
    answer_terms = source_keywords(answer, limit=12) if answer else []
    chunk_terms  = []
    for chunk in chunks[:3]:
        for word in re.findall(r"[a-zA-Z0-9]{4,}", strip_page_markers(chunk).lower()):
            if word not in SOURCE_STOPWORDS and word not in answer_terms and word not in chunk_terms:
                chunk_terms.append(word)
            if len(chunk_terms) >= 12:
                break
        if len(chunk_terms) >= 12:
            break

    if not answer_terms and not chunk_terms:
        return []

    scores = []
    try:
        doc = fitz.open(file_path)
        for index, page in enumerate(doc, start=1):
            page_text = page.get_text().lower()
            score = (sum(2 for w in answer_terms if w in page_text) +
                     sum(1 for w in chunk_terms  if w in page_text))
            if score:
                scores.append((index, score))
    except Exception as e:
        log.warning(f"[SOURCE] Could not infer PDF source page for '{manual_name}': {e}")
        return []

    scores.sort(key=lambda item: item[1], reverse=True)
    return sorted(page for page, _ in scores[:max_pages])


def source_links_for_manual(manual_name, chunks, query="", answer="", call_id="", max_sources=3):
    sources    = []
    seen_pages = set()

    # Load past agent feedback for this manual + query
    # {page: net_score} — positive = useful, negative = wrong
    feedback_scores = get_citation_feedback(manual_name, query_text=query)

    # Score each chunk and apply feedback adjustment
    def adjusted_score(chunk):
        base  = source_match_score(query, answer, chunk)
        page  = page_number_from_chunk(chunk)
        delta = feedback_scores.get(page, 0) if page else 0
        return base + delta

    # Rank chunks by adjusted score — feedback now directly affects page selection
    ranked = sorted(chunks, key=adjusted_score, reverse=True)

    for chunk in ranked:
        page = page_number_from_chunk(chunk)
        if not page or page in seen_pages:
            continue
        seen_pages.add(page)

        base_score  = source_match_score(query, answer, chunk)
        fb_delta    = feedback_scores.get(page, 0)
        final_score = base_score + fb_delta

        # Skip pages the agent has consistently marked wrong
        # (net score <= -5 means at least one strong "wrong" with no offsetting "useful")
        if fb_delta <= -5 and base_score < 5:
            log.debug(f"[SOURCE] Skipping page {page} — feedback penalised (delta={fb_delta}, base={base_score})")
            continue

        highlight = " ".join(source_keywords(answer, query, chunk))
        excerpt   = source_excerpt(chunk)
        sources.append({
            "manual_key": manual_name,
            "page": page,
            "label": f"Page {page}",
            "excerpt": excerpt,
            "highlight": highlight,
            "match_score": round(final_score, 1),
            "confidence": source_confidence_label(final_score),
            "url": url_for(
                "manual_source",
                manual_key=manual_name,
                page=page,
                call_id=call_id,
                query=query[:300],
                highlight=highlight,
                excerpt=excerpt
            )
        })
        if len(sources) >= max_sources:
            return sources

    for page in infer_source_pages_from_pdf(manual_name, chunks, answer=answer, max_pages=max_sources):
        if page in seen_pages:
            continue
        highlight = " ".join(source_keywords(answer, query, *chunks))
        excerpt = source_excerpt(ranked[0] if ranked else "")
        score = max(source_match_score(query, answer, chunk) for chunk in chunks) if chunks else 0
        sources.append({
            "manual_key": manual_name,
            "page": page,
            "label": f"Page {page}",
            "excerpt": excerpt,
            "highlight": highlight,
            "match_score": score,
            "confidence": source_confidence_label(score),
            "url": url_for(
                "manual_source",
                manual_key=manual_name,
                page=page,
                call_id=call_id,
                query=query[:300],
                highlight=highlight,
                excerpt=excerpt
            )
        })
        if len(sources) >= max_sources:
            break

    if sources:
        return sources
    return []


def flashcards_from_answer(answer):
    """
    Converts an AI answer into flashcards using the LLM.
    Each card: {"keyword": "topic label", "body": "one concise sentence"}.
    The LLM extracts meaningful topic labels — not just first words.
    Falls back to regex parsing if LLM call fails.
    """
    prompt = """Convert the following support answer into 2-4 flashcards.
Each flashcard must have:
- keyword: a short 2-4 word TOPIC LABEL that names what this card is about (e.g. "Detergent Drawer", "Spin Speed", "Error Code E3", "Filter Location"). NOT the first words of the sentence.
- body: one clear, concise sentence (max 20 words) summarising the key fact.

Rules:
- keyword must be a meaningful topic, not a sentence fragment
- body must be self-contained and actionable
- Skip filler lines, focus on facts and steps
- Return ONLY valid JSON array, no markdown, no explanation

Answer to convert:
{answer}

Return format:
[{{"keyword": "Topic Label", "body": "One clear sentence."}}, ...]"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt.format(answer=answer)}],
            max_tokens=400,
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
        cards = json.loads(raw)
        # Validate and clean
        result = []
        for c in cards:
            if isinstance(c, dict) and c.get("keyword") and c.get("body"):
                kw   = str(c["keyword"]).strip()[:40]
                body = str(c["body"]).strip()[:150]
                if kw and body:
                    result.append({"keyword": kw, "body": body})
        if result:
            log.debug(f"[FLASHCARD] LLM generated {len(result)} card(s)")
            return result[:4]
    except Exception as e:
        log.warning(f"[FLASHCARD] LLM extraction failed: {e} — using regex fallback")

    # ── Regex fallback ────────────────────────────────────────────────
    def clean(text):
        return re.sub(r'\*\*(.*?)\*\*', r'\1', text).strip(" -\u2022\t")

    raw_parts = [p.strip() for p in re.split(r'\n+', answer) if p.strip()]
    cards = []
    step_labels = {
        "1": "Step One", "2": "Step Two", "3": "Step Three",
        "4": "Step Four", "5": "Step Five"
    }
    for part in raw_parts:
        part = part.strip(" -\u2022\t*#")
        if not part or len(part) < 15:
            continue

        # Numbered step — use step label as keyword
        step_match = re.match(r'^(\d+)[.)]\.?\s+(.*)', part)
        bold_match  = re.search(r'\*\*(.*?)\*\*', part)

        if step_match:
            num     = step_match.group(1)
            keyword = step_labels.get(num, f"Step {num}")
            body    = clean(step_match.group(2))
        elif bold_match:
            keyword = bold_match.group(1).strip()[:40]
            after   = part[bold_match.end():].strip(" :-")
            body    = clean(after) if after else clean(part)
        else:
            # Use subject of sentence — first noun-like word after stopwords
            words = re.sub(r'[^a-zA-Z0-9 ]', " ", clean(part)).split()
            skip  = {"the","a","an","to","is","in","it","of","and","or",
                     "for","be","use","if","you","your","this","that"}
            nouns = [w for w in words if w.lower() not in skip and len(w) > 3][:3]
            keyword = " ".join(nouns).title() if nouns else "Key Point"
            sentence_match = re.search(r'[^.!?]+[.!?]', clean(part))
            body = sentence_match.group(0).strip() if sentence_match else clean(part)

        sentence_match = re.search(r'[^.!?]+[.!?]', body)
        body = sentence_match.group(0).strip() if sentence_match else body
        if len(body) > 120:
            body = body[:117].rstrip() + "..."

        if keyword and body and len(body) > 5:
            cards.append({"keyword": keyword, "body": body})
        if len(cards) >= 4:
            break

    if not cards:
        first = re.search(r'[^.!?]+[.!?]', clean(answer))
        body  = first.group(0).strip() if first else clean(answer)[:120]
        cards.append({"keyword": "Key Point", "body": body})

    return cards


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


# ================= LIVEKIT TOKEN =================

@app.route("/livekit-token", methods=["POST"])
def livekit_token():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data      = request.get_json()
    call_id   = data.get("call_id",  "").strip().upper()
    role      = data.get("role",     "customer")

    if not call_id:
        return jsonify({"error": "call_id required"}), 400

    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET or not LIVEKIT_URL:
        return jsonify({"error": "LiveKit not configured"}), 500

    room_name = f"guideai-{call_id}"

    try:
        from livekit.api import AccessToken, VideoGrants

        identity = session["user"]
        from datetime import timedelta
        token = (
            AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(f"{role}-{identity}")
            .with_name(role.capitalize())
            .with_ttl(timedelta(hours=2))   # 2 hours — prevents mid-call expiry
            .with_grants(VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                room_create=True,
            ))
            .to_jwt()
        )

        log.info(f"[LIVEKIT] ✅ Token for '{role}-{identity}' in room '{room_name}'")
        return jsonify({"token": token, "url": LIVEKIT_URL, "room": room_name})

    except Exception as e:
        log.error(f"[LIVEKIT] ❌ Failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ================= WHISPER TRANSCRIPTION =================

# Known Whisper hallucinations when audio is silence/noise
WHISPER_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks", "thanks.",
    "you", "you.", "bye", "bye.", "goodbye",
    "b, a,", "b,a,", "a,", "b,",
    ".", ",", "...", " ", "",
    "subtitles by", "transcribed by", "www.",
    "i don't know", "i don't know.",
}

@app.route("/assemblyai-token", methods=["GET"])
def assemblyai_token():
    """Creates a temporary AssemblyAI session token for browser streaming."""
    if not session.get("user"):
        return jsonify({"error": "Not logged in"}), 401
    api_key = os.getenv("ASSEMBLYAI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "AssemblyAI not configured"}), 500
    try:
        import urllib.request, json as _json
        req = urllib.request.Request(
            "https://streaming.assemblyai.com/v3/token?expires_in_seconds=600&max_session_duration_seconds=10800",
            headers={"Authorization": api_key},
            method="GET"
        )
        with urllib.request.urlopen(req) as resp:
            data = _json.loads(resp.read())
        return jsonify({"token": data["token"]})
    except Exception as e:
        log.error(f"[ASSEMBLYAI] Token generation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Receives an audio blob from the customer's browser,
    sends to Groq Whisper, filters hallucinations, returns text.
    """
    if "file" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["file"]
    if not audio_file:
        return jsonify({"error": "Empty audio file"}), 400

    try:
        audio_bytes = audio_file.read()
        if len(audio_bytes) < 3000:
            return jsonify({"text": "", "skipped": True})

        transcription = _groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=("audio.webm", audio_bytes, "audio/webm"),
            response_format="verbose_json",  # gives us more metadata
            language="en"
        )

        text = transcription.text.strip() if hasattr(transcription, 'text') else str(transcription).strip()

        # Filter known hallucinations
        if text.lower() in WHISPER_HALLUCINATIONS:
            log.debug(f"[WHISPER] Filtered hallucination: '{text}'")
            return jsonify({"text": "", "skipped": True})

        # Filter very short phrases that are likely noise
        word_count = len(text.split())
        if word_count < 2 and text.lower() not in {"hello", "yes", "no", "help", "okay", "ok"}:
            log.debug(f"[WHISPER] Filtered short noise: '{text}'")
            return jsonify({"text": "", "skipped": True})

        log.info(f"[WHISPER] ✅ Transcribed: '{text[:80]}{'...' if len(text) > 80 else ''}'")
        return jsonify({"text": text})

    except Exception as e:
        log.error(f"[WHISPER] ❌ Transcription failed: {e}")
        return jsonify({"error": str(e)}), 500


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

    # Fetch only the recent preview so the customer can enter the call quickly.
    chat_preview = []
    if session_name:
        raw = get_recent_chat_history(customer_id, session_name, limit=3)
        chat_preview = [{"sender": r[0], "message": r[1]} for r in raw]

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
        "chat_history":    chat_preview
    }

    # Notify agent dashboard via socket
    socketio.emit("incoming_call", {
        "call_id":      call_id,
        "customer_id":  customer_id,
        "manual_label": AVAILABLE_MANUALS.get(manual_name, manual_name),
        "manual_name":  manual_name,
        "chat_preview": chat_preview
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
        manual_name=call["manual_name"],
        manual_label=manual_label,
        auto_join=bool(call.get("customer_id")),
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


@app.route("/manual_pdf/<manual_key>")
def manual_pdf(manual_key):
    if manual_key not in MANUAL_FILES:
        return "Manual not found", 404
    file_path = MANUAL_FILES[manual_key]
    folder = os.path.dirname(file_path) or "."
    filename = os.path.basename(file_path)
    return send_from_directory(folder, filename)


@app.route("/manual_source/<manual_key>")
def manual_source(manual_key):
    if manual_key not in MANUAL_FILES:
        return "Manual not found", 404
    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1
    return render_template(
        "source_viewer.html",
        manual_key=manual_key,
        manual_label=AVAILABLE_MANUALS.get(manual_key, manual_key),
        page=page,
        call_id=request.args.get("call_id", ""),
        query_text=request.args.get("query", ""),
        highlight=request.args.get("highlight", ""),
        excerpt=request.args.get("excerpt", ""),
        is_agent=session.get("role") == "agent"
    )


@app.route("/manual_source_image/<manual_key>")
def manual_source_image(manual_key):
    if manual_key not in MANUAL_FILES:
        return "Manual not found", 404

    file_path = MANUAL_FILES[manual_key]
    try:
        page_num = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page_num = 1

    # Build highlight terms: individual words + 2-word phrases from highlight param
    raw_highlight = request.args.get("highlight", "")
    single_terms  = source_keywords(raw_highlight, limit=18)
    multi_phrases = source_phrases(raw_highlight, max_phrases=8)
    all_terms     = multi_phrases + single_terms  # phrases first — longer matches preferred

    try:
        import fitz
        doc = fitz.open(file_path)
        page_num = min(page_num, len(doc))
        page = doc[page_num - 1]

        highlighted = set()
        for term in all_terms:
            # quads=True gives tighter rects; flags=fitz.TEXT_DEHYPHENATE handles hyphenated words
            rects = page.search_for(term, quads=False)
            # Also try title-cased and upper variants for scanned docs
            if not rects:
                rects = page.search_for(term.title(), quads=False)
            if not rects:
                rects = page.search_for(term.upper(), quads=False)
            for rect in rects:
                key = (round(rect.x0), round(rect.y0))
                if key in highlighted:
                    continue
                highlighted.add(key)
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1, 0.85, 0.1))
                annot.update()

        # Render at 2.0x for sharper text — previous 1.7x was slightly blurry on retina
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        return send_file(io.BytesIO(pix.tobytes("png")), mimetype="image/png")
    except Exception as e:
        log.error(f"[SOURCE] Failed to render source page '{manual_key}' page={page_num}: {e}", exc_info=True)
        return "Could not render source page", 500


@app.route("/citation_feedback", methods=["POST"])
def citation_feedback():
    if not require_login(role="agent"):
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    feedback = data.get("feedback", "").strip().lower()
    if feedback not in {"useful", "wrong"}:
        return jsonify({"error": "Invalid feedback"}), 400

    manual_key = data.get("manual_key", "").strip()
    if manual_key not in MANUAL_FILES:
        return jsonify({"error": "Manual not found"}), 404

    try:
        page = max(1, int(data.get("page", 1)))
    except (TypeError, ValueError):
        page = 1

    save_citation_feedback(
        call_id=data.get("call_id", "").strip().upper(),
        agent=session["user"],
        manual_key=manual_key,
        page=page,
        query_text=data.get("query_text", "").strip(),
        source_excerpt=data.get("source_excerpt", "").strip(),
        feedback=feedback
    )

    return jsonify({"success": True})


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
        if call_id in agent_in_call:
            emit("agent_joined_call", {
                "message": "Agent has joined. You can now start chatting."
            })

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

    if not is_actionable_query(text):
        emit("rag_suggestion", {
            "suggestion": "Filler/small talk detected — waiting for a manual-related question.",
            "confidence": "none",
            "original_query": text,
            "flashcards": [],
            "sources": []
        }, room=f"{call_id}_agent")
        emit("manual_sources", {
            "query": text,
            "sources": [],
            "note": "No manual source for this message."
        }, room=f"{call_id}_customer")
        return

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
            session_score=call_state["running_score"],
            is_voice=True
        )

        call_state["last_suggestion"] = suggestion
        call_state["last_query"]      = text
        sources = source_links_for_manual(manual_name, relevant_docs, query=text, answer=suggestion, call_id=call_id)
        flashcards = flashcards_from_answer(suggestion)

        emit("rag_suggestion", {
            "suggestion":     suggestion,
            "confidence":     confidence,
            "original_query": text,
            "flashcards": flashcards,
            "sources": sources
        }, room=f"{call_id}_agent")
        emit("manual_sources", {
            "query": text,
            "sources": sources
        }, room=f"{call_id}_customer")
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

    # Sync rating into call_reports so dashboard shows correct value on refresh
    try:
        update_call_report_rating(call_id, rating)
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


# ================= SIGNALLING =================
# LiveKit handles all audio P2P — these are lightweight relay events only.

@socketio.on("livekit_room")
def on_livekit_room(data):
    """Customer started a LiveKit room — relay room name to agent."""
    try:
        call_id   = data.get("call_id", "").strip().upper()
        room_name = data.get("room",    "").strip()
        log.info(f"[LIVEKIT] Room '{room_name}' for call '{call_id}' — notifying agent")
        emit("livekit_room", {"room": room_name, "call_id": call_id},
             room=f"{call_id}_agent")
    except Exception as e:
        log.error(f"[LIVEKIT] ❌ livekit_room relay error: {e}", exc_info=True)

@socketio.on("voice_end")
def on_voice_end(data):
    try:
        call_id = data.get("call_id", "").strip().upper()
        sender  = data.get("sender", "customer")
        if sender == "customer":
            emit("voice_end", data, room=f"{call_id}_agent")
        else:
            emit("voice_end", data, room=f"{call_id}_customer")
    except Exception as e:
        log.error(f"[VOICE] ❌ voice_end error: {e}", exc_info=True)


# ================= RUN =================

if __name__ == "__main__":
    log.info("[RUN] Starting Flask-SocketIO development server...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)