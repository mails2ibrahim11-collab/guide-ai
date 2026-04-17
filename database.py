import sqlite3
import hashlib
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

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            unique_id TEXT PRIMARY KEY,
            password TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            session_name TEXT,
            manual_name TEXT,
            last_used TEXT,
            score REAL DEFAULT 5,
            UNIQUE(user, session_name)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            session_name TEXT,
            manual_name TEXT,
            sender TEXT,
            message TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


# ================= USERS =================

def register_user(unique_id, password):
    conn = connect()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (unique_id, hash_password(password)))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def login_user(unique_id, password):
    conn = connect()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE unique_id=? AND password=?",
        (unique_id, hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    return user is not None


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
        pass  # Already exists — silent ignore
    finally:
        conn.close()


def update_session(user, session_name, new_score):
    """
    Weighted rolling average: 70% old score, 30% new score.
    This smooths out single bad/good answers and tracks trend.
    """
    conn = connect()
    c = conn.cursor()

    c.execute(
        "SELECT score FROM sessions WHERE user=? AND session_name=?",
        (user, session_name)
    )
    result = c.fetchone()

    if result:
        updated = round((0.7 * result[0]) + (0.3 * new_score), 2)
    else:
        updated = new_score

    c.execute("""
        UPDATE sessions SET last_used=?, score=?
        WHERE user=? AND session_name=?
    """, (get_now(), updated, user, session_name))

    conn.commit()
    conn.close()


def get_session_score(user, session_name):
    """
    Fetch the current rolling score for a session.
    Used by generate_answer() to adapt response behavior.
    This is the key to the self-improving loop.
    """
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
        FROM sessions
        WHERE user=?
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
        FROM chats
        WHERE user=? AND session_name=?
        ORDER BY timestamp ASC
    """, (user, session_name))
    chats = c.fetchall()
    conn.close()
    return chats