import sqlite3
import hashlib
import uuid
from datetime import datetime

DB_PATH = "data/database.db"

# ================= CORE =================

def connect():
    return sqlite3.connect(DB_PATH)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def get_now():
    return datetime.now().isoformat()


# ================= INIT =================

def init_db():
    conn = connect()
    c = conn.cursor()

    # users — role is 'agent' or 'customer'
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            unique_id TEXT PRIMARY KEY,
            password  TEXT,
            role      TEXT DEFAULT 'customer'
        )
    """)

    # Migrate existing users table if role column missing
    try:
        c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'customer'")
        conn.commit()
    except Exception:
        pass  # Column already exists

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user         TEXT,
            session_name TEXT,
            manual_name  TEXT,
            last_used    TEXT,
            score        REAL DEFAULT 5,
            UNIQUE(user, session_name)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user         TEXT,
            session_name TEXT,
            manual_name  TEXT,
            sender       TEXT,
            message      TEXT,
            timestamp    TEXT
        )
    """)

    # manuals — owner is customer unique_id or 'system' for hardcoded
    c.execute("""
        CREATE TABLE IF NOT EXISTS manuals (
            key        TEXT PRIMARY KEY,
            label      TEXT,
            file_path  TEXT,
            created_at TEXT,
            owner      TEXT DEFAULT 'system'
        )
    """)

    # Migrate existing manuals table if owner column missing
    try:
        c.execute("ALTER TABLE manuals ADD COLUMN owner TEXT DEFAULT 'system'")
        conn.commit()
    except Exception:
        pass  # Column already exists

    # ── CALL TABLES ──────────────────────────────────────────────

    c.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id              TEXT PRIMARY KEY,
            agent           TEXT,
            customer_id     TEXT,
            session_id      INTEGER,
            manual_name     TEXT,
            status          TEXT DEFAULT 'active',
            created_at      TEXT,
            ended_at        TEXT,
            final_score     REAL,
            customer_rating INTEGER
        )
    """)

    # Migrate existing calls table — add customer_id and session_id if missing
    for col, typedef in [("customer_id", "TEXT"), ("session_id", "INTEGER")]:
        try:
            c.execute(f"ALTER TABLE calls ADD COLUMN {col} {typedef}")
            conn.commit()
        except Exception:
            pass

    c.execute("""
        CREATE TABLE IF NOT EXISTS call_turns (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id         TEXT,
            speaker         TEXT,
            original_text   TEXT,
            edited_text     TEXT,
            ai_suggestion   TEXT,
            rag_confidence  TEXT,
            agent_used_ai   INTEGER DEFAULT 0,
            turn_score      REAL,
            timestamp       TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS call_reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id         TEXT,
            agent           TEXT,
            report_text     TEXT,
            transcript      TEXT,
            overall_score   REAL,
            customer_rating INTEGER,
            created_at      TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS citation_feedback (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id       TEXT,
            agent         TEXT,
            manual_key    TEXT,
            page          INTEGER,
            query_text    TEXT,
            source_excerpt TEXT,
            feedback      TEXT,
            created_at    TEXT
        )
    """)

    conn.commit()
    conn.close()


# ================= USERS =================

def register_user(unique_id, password, role='customer'):
    """Register a new user. Role is 'customer' by default; agent is hardcoded."""
    conn = connect()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users VALUES (?, ?, ?)",
            (unique_id, hash_password(password), role)
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def login_user(unique_id, password):
    """
    Validates credentials.
    Returns the role string ('agent' or 'customer') on success, None on failure.
    """
    conn = connect()
    c = conn.cursor()
    c.execute(
        "SELECT role FROM users WHERE unique_id=? AND password=?",
        (unique_id, hash_password(password))
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def get_user_role(unique_id):
    """Returns the role of a user, or None if not found."""
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE unique_id=?", (unique_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


# ================= SESSIONS =================

def create_session(user, session_name, manual_name):
    conn = connect()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO sessions (user, session_name, manual_name, last_used, score)
            VALUES (?, ?, ?, ?, ?)
        """, (user, session_name, manual_name, get_now(), 5))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def update_session(user, session_name, new_score):
    conn = connect()
    c = conn.cursor()
    c.execute(
        "SELECT score FROM sessions WHERE user=? AND session_name=?",
        (user, session_name)
    )
    result = c.fetchone()
    updated = round((0.7 * result[0]) + (0.3 * new_score), 2) if result else new_score
    c.execute("""
        UPDATE sessions SET last_used=?, score=?
        WHERE user=? AND session_name=?
    """, (get_now(), updated, user, session_name))
    conn.commit()
    conn.close()


def get_session_score(user, session_name):
    conn = connect()
    c = conn.cursor()
    c.execute(
        "SELECT score FROM sessions WHERE user=? AND session_name=?",
        (user, session_name)
    )
    result = c.fetchone()
    conn.close()
    return result[0] if result else None


def get_all_sessions(user):
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT session_name, manual_name, last_used, score
        FROM sessions WHERE user=?
        ORDER BY last_used DESC
    """, (user,))
    sessions = c.fetchall()
    conn.close()
    return sessions


def get_session_manual(user, session_name):
    conn = connect()
    c = conn.cursor()
    c.execute(
        "SELECT manual_name FROM sessions WHERE user=? AND session_name=?",
        (user, session_name)
    )
    result = c.fetchone()
    conn.close()
    return result[0] if result else None


def rename_session(user, old_name, new_name):
    conn = connect()
    c = conn.cursor()
    c.execute(
        "UPDATE sessions SET session_name=? WHERE user=? AND session_name=?",
        (new_name, user, old_name)
    )
    c.execute(
        "UPDATE chats SET session_name=? WHERE user=? AND session_name=?",
        (new_name, user, old_name)
    )
    conn.commit()
    conn.close()


def delete_session(user, session_name):
    conn = connect()
    c = conn.cursor()
    c.execute(
        "DELETE FROM sessions WHERE user=? AND session_name=?",
        (user, session_name)
    )
    c.execute(
        "DELETE FROM chats WHERE user=? AND session_name=?",
        (user, session_name)
    )
    conn.commit()
    conn.close()


# ================= MANUAL PERSISTENCE =================

def save_uploaded_manual(key, label, file_path, owner='system'):
    """
    Persists an uploaded manual to the DB.
    owner = customer's unique_id, or 'system' for hardcoded manuals.
    """
    conn = connect()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT OR REPLACE INTO manuals (key, label, file_path, created_at, owner)
            VALUES (?, ?, ?, ?, ?)
        """, (key, label, file_path, get_now(), owner))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def delete_uploaded_manual(key):
    conn = connect()
    c = conn.cursor()
    c.execute("DELETE FROM manuals WHERE key=?", (key,))
    conn.commit()
    conn.close()


def get_uploaded_manuals():
    """Returns ALL uploaded (non-system) manuals for server restore on startup."""
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT key, label, file_path FROM manuals ORDER BY created_at ASC")
    rows = c.fetchall()
    conn.close()
    return rows


def get_manuals_by_owner(owner):
    """
    Returns manuals visible to a specific customer:
    their own uploads + all system-owned manuals.
    Also catches rows where owner is NULL (older uploads before migration).
    Returns list of (key, label, file_path, owner) tuples.
    """
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT key, label, file_path, COALESCE(owner, ?) as owner FROM manuals
        WHERE owner = 'system' OR owner = ? OR owner IS NULL
        ORDER BY created_at ASC
    """, (owner, owner))
    rows = c.fetchall()
    conn.close()
    return rows


def get_manual_owner(key):
    """
    Returns the owner of a manual key.
    Returns 'system' for hardcoded, the user id for uploads,
    or None if not found in DB at all.
    NULL in DB means uploaded before owner migration — treated as deletable.
    """
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT owner FROM manuals WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return None          # key not in DB at all
    return row[0]            # may be None (NULL) for pre-migration rows


# ================= MANUAL STATS =================

def get_manual_session_counts():
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT manual_name, COUNT(*) as count
        FROM sessions GROUP BY manual_name
    """)
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def get_active_manual_session_counts():
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT s.manual_name, COUNT(DISTINCT s.id) as count
        FROM sessions s
        INNER JOIN chats c ON s.user = c.user AND s.session_name = c.session_name
        WHERE c.timestamp >= datetime('now', '-24 hours')
        GROUP BY s.manual_name
    """)
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


# ================= CHATS =================

def save_message(user, session_name, manual_name, message, sender):
    conn = connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO chats (user, session_name, manual_name, sender, message, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user, session_name, manual_name, sender, message, get_now()))
    conn.commit()
    conn.close()


def get_chat_history(user, session_name):
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT sender, message, timestamp
        FROM chats WHERE user=? AND session_name=?
        ORDER BY timestamp ASC
    """, (user, session_name))
    chats = c.fetchall()
    conn.close()
    return chats


def get_recent_chat_history(user, session_name, limit=3):
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT sender, message, timestamp
        FROM chats WHERE user=? AND session_name=?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user, session_name, limit))
    chats = list(reversed(c.fetchall()))
    conn.close()
    return chats


# ================= CALLS =================

def create_call(agent, manual_name, customer_id=None, session_id=None):
    """
    Creates a new call session. Returns the unique call_id.
    customer_id and session_id are set when customer initiates the call.
    """
    call_id = str(uuid.uuid4())[:8].upper()
    conn = connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO calls (id, agent, customer_id, session_id, manual_name, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'active', ?)
    """, (call_id, agent, customer_id, session_id, manual_name, get_now()))
    conn.commit()
    conn.close()
    return call_id


def get_call(call_id):
    """Returns full call row as a dict."""
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT * FROM calls WHERE id=?", (call_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    keys = ["id", "agent", "customer_id", "session_id", "manual_name",
            "status", "created_at", "ended_at", "final_score", "customer_rating"]
    return dict(zip(keys, row))


def update_call_manual(call_id, manual_name):
    """Agent overrides the manual mid-call."""
    conn = connect()
    c = conn.cursor()
    c.execute(
        "UPDATE calls SET manual_name=? WHERE id=?",
        (manual_name, call_id)
    )
    conn.commit()
    conn.close()


def update_call_customer(call_id, customer_id, session_id=None):
    """Sets the customer_id (and optionally session_id) once customer joins."""
    conn = connect()
    c = conn.cursor()
    c.execute(
        "UPDATE calls SET customer_id=?, session_id=? WHERE id=?",
        (customer_id, session_id, call_id)
    )
    conn.commit()
    conn.close()


def end_call(call_id, final_score):
    """Marks call as ended and saves the final AI score."""
    conn = connect()
    c = conn.cursor()
    c.execute("""
        UPDATE calls SET status='ended', ended_at=?, final_score=?
        WHERE id=?
    """, (get_now(), final_score, call_id))
    conn.commit()
    conn.close()


def save_customer_rating(call_id, rating):
    """Saves the 0-10 rating submitted by the customer at end of call."""
    conn = connect()
    c = conn.cursor()
    c.execute(
        "UPDATE calls SET customer_rating=? WHERE id=?",
        (rating, call_id)
    )
    conn.commit()
    conn.close()


# ================= CALL TURNS =================

def save_call_turn(call_id, speaker, original_text, edited_text,
                   ai_suggestion, rag_confidence, agent_used_ai, turn_score):
    """
    Saves one turn of the conversation.
    agent_used_ai: 0=ignored, 1=used as-is, 2=edited then used
    """
    conn = connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO call_turns (
            call_id, speaker, original_text, edited_text,
            ai_suggestion, rag_confidence, agent_used_ai, turn_score, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (call_id, speaker, original_text, edited_text,
          ai_suggestion, rag_confidence, agent_used_ai, turn_score, get_now()))
    conn.commit()
    conn.close()


def get_call_turns(call_id):
    """Returns all turns for a call ordered by time."""
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT speaker, original_text, edited_text, ai_suggestion,
               rag_confidence, agent_used_ai, turn_score, timestamp
        FROM call_turns WHERE call_id=?
        ORDER BY timestamp ASC
    """, (call_id,))
    rows = c.fetchall()
    conn.close()
    keys = ["speaker", "original_text", "edited_text", "ai_suggestion",
            "rag_confidence", "agent_used_ai", "turn_score", "timestamp"]
    return [dict(zip(keys, row)) for row in rows]


# ================= CALL REPORTS =================

def update_call_report_rating(call_id, rating):
    """Syncs customer rating into call_reports after it is submitted."""
    conn = connect()
    c = conn.cursor()
    c.execute(
        "UPDATE call_reports SET customer_rating=? WHERE call_id=?",
        (rating, call_id)
    )
    conn.commit()
    conn.close()


def save_call_report(call_id, agent, report_text, transcript,
                     overall_score, customer_rating):
    """Saves the final AI-generated call report to DB."""
    conn = connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO call_reports (
            call_id, agent, report_text, transcript,
            overall_score, customer_rating, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (call_id, agent, report_text, transcript,
          overall_score, customer_rating, get_now()))
    conn.commit()
    conn.close()


def get_agent_reports(agent):
    """Returns all call reports for an agent, newest first."""
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT call_id, overall_score, customer_rating, created_at
        FROM call_reports WHERE agent=?
        ORDER BY created_at DESC
    """, (agent,))
    rows = c.fetchall()
    conn.close()
    keys = ["call_id", "overall_score", "customer_rating", "created_at"]
    return [dict(zip(keys, row)) for row in rows]


def get_call_report(call_id):
    """Returns full report for a specific call."""
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT * FROM call_reports WHERE call_id=?", (call_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    keys = ["id", "call_id", "agent", "report_text", "transcript",
            "overall_score", "customer_rating", "created_at"]
    return dict(zip(keys, row))


# ================= CITATION FEEDBACK =================

def save_citation_feedback(call_id, agent, manual_key, page, query_text,
                           source_excerpt, feedback):
    conn = connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO citation_feedback (
            call_id, agent, manual_key, page, query_text,
            source_excerpt, feedback, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        call_id, agent, manual_key, page, query_text,
        source_excerpt, feedback, get_now()
    ))
    conn.commit()
    conn.close()


def get_citation_feedback(manual_key, query_text="", limit=200):
    """
    Returns aggregated feedback for a manual as:
        {page: net_score}
    where net_score = (useful_count * 3) - (wrong_count * 5)

    If query_text is provided, feedback from similar queries is weighted
    more heavily (simple substring match on stored query_text).
    Pages with net_score < 0 should be penalised in source ranking.
    Pages with net_score > 0 should be boosted.
    """
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT page, feedback, query_text
        FROM citation_feedback
        WHERE manual_key = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (manual_key, limit))
    rows = c.fetchall()
    conn.close()

    scores = {}
    q_lower = query_text.lower() if query_text else ""

    for page, feedback, stored_query in rows:
        if page is None:
            continue
        # Weight: 2x if query is similar, 1x otherwise
        similar = q_lower and stored_query and (
            q_lower[:30] in stored_query.lower() or
            stored_query.lower()[:30] in q_lower
        )
        weight = 2 if similar else 1

        delta = (3 * weight) if feedback == "useful" else (-5 * weight)
        scores[page] = scores.get(page, 0) + delta

    return scores