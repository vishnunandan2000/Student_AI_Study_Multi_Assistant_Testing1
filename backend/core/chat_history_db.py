import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_DIR = Path(__file__).resolve().parent.parent.parent / "temp"
DB_PATH = DB_DIR / "chat_history.db"


def get_db_connection() -> sqlite3.Connection:
    """Creates directory if needed and returns SQLite connection with foreign keys enabled."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Initializes the database schema if tables do not exist."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            );
        """)
        conn.commit()


def create_session(title: str = "New Chat") -> str:
    """Creates a new session record and returns its session_id."""
    init_db()
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now)
        )
        conn.commit()
    return session_id


def get_all_sessions() -> List[Dict[str, Any]]:
    """Returns all chat sessions ordered by last update time descending."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_session_messages(session_id: str) -> List[Dict[str, str]]:
    """Retrieves all messages for a specific session formatted for Gradio (list of dicts)."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]


def add_message(session_id: str, role: str, content: str):
    """Adds a message to a session and updates session's updated_at timestamp."""
    init_db()
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now)
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id)
        )
        conn.commit()


def update_session_title(session_id: str, title: str):
    """Updates the title of a session."""
    init_db()
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?",
            (title, session_id)
        )
        conn.commit()


def delete_session(session_id: str):
    """Deletes a single session and all associated messages."""
    init_db()
    with get_db_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()


def delete_all_sessions():
    """Permanently deletes all sessions and messages."""
    init_db()
    with get_db_connection() as conn:
        conn.execute("DELETE FROM sessions")
        conn.commit()


# Initialize database schema on module import
init_db()
