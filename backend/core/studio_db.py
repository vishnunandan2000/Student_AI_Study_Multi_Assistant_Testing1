"""
Database and Job Management for GenAI Video Intelligence Studio.
Manages persistent state for:
- Video Editing Jobs (status, progress, stages, logs, artifact paths)
- YouTube Video Intelligence Analyses (metadata, sentiment, recommendations)
Stored in SQLite via standard sqlite3 for reliable zero-dependency local operation.
"""

import os
import json
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DB_DIR / "app.db"


def get_db_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_studio_db():
    """Initializes tables for jobs and YouTube analyses."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS studio_jobs (
            job_id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            current_stage TEXT DEFAULT 'Queued',
            progress REAL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            duration_seconds REAL DEFAULT 0.0,
            error_message TEXT DEFAULT '',
            input_params TEXT DEFAULT '{}',
            output_files TEXT DEFAULT '{}',
            logs TEXT DEFAULT '[]'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS youtube_analyses (
            video_id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT,
            channel TEXT,
            metadata_json TEXT DEFAULT '{}',
            sentiment_json TEXT DEFAULT '{}',
            recommendations_json TEXT DEFAULT '{}',
            analyzed_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_records (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            current_version INTEGER DEFAULT 1,
            char_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            embedding_model TEXT DEFAULT 'nomic-embed-text / llama3.2',
            indexed_at TEXT,
            metadata_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_versions (
            version_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            version_number INTEGER NOT NULL,
            filename TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            content_text TEXT NOT NULL,
            chunks_json TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            change_summary TEXT DEFAULT '',
            FOREIGN KEY(document_id) REFERENCES document_records(document_id)
        )
    """)

    conn.commit()
    conn.close()


# Ensure DB is initialized on module load
init_studio_db()


def create_job(job_id: str, job_type: str, input_params: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a new job in queued state."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO studio_jobs (job_id, job_type, status, current_stage, progress, created_at, updated_at, input_params)
        VALUES (?, ?, 'queued', 'Queued', 0.0, ?, ?, ?)
    """, (job_id, job_type, now, now, json.dumps(input_params)))

    conn.commit()
    conn.close()
    return get_job(job_id)


def update_job_progress(job_id: str, stage: str, progress: float, log_entry: Optional[str] = None):
    """Updates job processing stage, progress percentage, and appends a log entry."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT logs FROM studio_jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    logs = json.loads(row["logs"]) if row and row["logs"] else []

    if log_entry:
        logs.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {log_entry}")

    cursor.execute("""
        UPDATE studio_jobs
        SET status = 'running', current_stage = ?, progress = ?, updated_at = ?, logs = ?
        WHERE job_id = ?
    """, (stage, min(100.0, max(0.0, progress)), now, json.dumps(logs), job_id))

    conn.commit()
    conn.close()


def complete_job(job_id: str, output_files: Dict[str, Any], duration_seconds: float = 0.0):
    """Marks job as completed with validated output artifacts."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE studio_jobs
        SET status = 'completed', current_stage = 'Completed', progress = 100.0,
            updated_at = ?, duration_seconds = ?, output_files = ?
        WHERE job_id = ?
    """, (now, duration_seconds, json.dumps(output_files), job_id))

    conn.commit()
    conn.close()


def fail_job(job_id: str, error_message: str):
    """Marks job as failed with error details."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT logs FROM studio_jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    logs = json.loads(row["logs"]) if row and row["logs"] else []
    logs.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ERROR: {error_message}")

    cursor.execute("""
        UPDATE studio_jobs
        SET status = 'failed', current_stage = 'Failed', updated_at = ?,
            error_message = ?, logs = ?
        WHERE job_id = ?
    """, (now, error_message, json.dumps(logs), job_id))

    conn.commit()
    conn.close()


_ACTIVE_CANCEL_SET = set()


def request_job_cancellation(job_id: str):
    """Requests immediate cancellation for an active job in memory and in the database."""
    _ACTIVE_CANCEL_SET.add(job_id)
    cancel_job(job_id)


def is_job_cancelled(job_id: str) -> bool:
    """Checks whether cancellation has been requested for a given job."""
    if job_id in _ACTIVE_CANCEL_SET:
        return True
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM studio_jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()
        return bool(row and row["status"] == "cancelled")
    except Exception:
        return False


def clear_job_cancellation(job_id: str):
    """Clears cancellation flag from active tracking."""
    _ACTIVE_CANCEL_SET.discard(job_id)


def cancel_job(job_id: str):
    """Marks job as cancelled by user."""
    _ACTIVE_CANCEL_SET.add(job_id)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE studio_jobs
        SET status = 'cancelled', current_stage = 'Cancelled by User', updated_at = ?
        WHERE job_id = ?
    """, (now, job_id))

    conn.commit()
    conn.close()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full details for a given job."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM studio_jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "job_id": row["job_id"],
        "job_type": row["job_type"],
        "status": row["status"],
        "current_stage": row["current_stage"],
        "progress": row["progress"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "duration_seconds": row["duration_seconds"],
        "error_message": row["error_message"],
        "input_params": json.loads(row["input_params"] or "{}"),
        "output_files": json.loads(row["output_files"] or "{}"),
        "logs": json.loads(row["logs"] or "[]"),
    }


def list_recent_jobs(limit: int = 20) -> List[Dict[str, Any]]:
    """Lists recent jobs sorted by creation timestamp descending."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM studio_jobs ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "job_id": r["job_id"],
            "job_type": r["job_type"],
            "status": r["status"],
            "current_stage": r["current_stage"],
            "progress": r["progress"],
            "created_at": r["created_at"],
            "duration_seconds": r["duration_seconds"],
            "output_files": json.loads(r["output_files"] or "{}")
        }
        for r in rows
    ]


def save_youtube_analysis(
    video_id: str,
    url: str,
    title: str,
    channel: str,
    metadata: Dict[str, Any],
    sentiment: Dict[str, Any],
    recommendations: Dict[str, Any]
):
    """Saves or updates a YouTube video analysis record."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO youtube_analyses (video_id, url, title, channel, metadata_json, sentiment_json, recommendations_json, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        video_id,
        url,
        title,
        channel,
        json.dumps(metadata),
        json.dumps(sentiment),
        json.dumps(recommendations),
        now
    ))

    conn.commit()
    conn.close()


def get_youtube_analysis(video_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves cached YouTube analysis for a video ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM youtube_analyses WHERE video_id = ?", (video_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "video_id": row["video_id"],
        "url": row["url"],
        "title": row["title"],
        "channel": row["channel"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "sentiment": json.loads(row["sentiment_json"] or "{}"),
        "recommendations": json.loads(row["recommendations_json"] or "{}"),
        "analyzed_at": row["analyzed_at"],
    }


def save_document_record(
    document_id: str,
    filename: str,
    file_type: str,
    char_count: int = 0,
    chunk_count: int = 0,
    embedding_model: str = "nomic-embed-text / llama3.2",
    metadata: Optional[Dict[str, Any]] = None,
    current_version: int = 1
) -> Dict[str, Any]:
    """Inserts or updates a tracked document record."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO document_records (
            document_id, filename, file_type, current_version, char_count,
            chunk_count, embedding_model, indexed_at, metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id) DO UPDATE SET
            filename = excluded.filename,
            file_type = excluded.file_type,
            current_version = excluded.current_version,
            char_count = excluded.char_count,
            chunk_count = excluded.chunk_count,
            embedding_model = excluded.embedding_model,
            indexed_at = excluded.indexed_at,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at
    """, (
        document_id,
        filename,
        file_type,
        current_version,
        char_count,
        chunk_count,
        embedding_model,
        now,
        json.dumps(metadata or {}),
        now,
        now
    ))

    conn.commit()
    conn.close()
    return get_document_record(document_id) or {}


def get_document_record(document_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a document record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM document_records WHERE document_id = ?", (document_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "document_id": row["document_id"],
        "filename": row["filename"],
        "file_type": row["file_type"],
        "current_version": row["current_version"],
        "char_count": row["char_count"],
        "chunk_count": row["chunk_count"],
        "embedding_model": row["embedding_model"],
        "indexed_at": row["indexed_at"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_document_records(limit: int = 50) -> List[Dict[str, Any]]:
    """Lists tracked documents ordered by update timestamp descending."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM document_records ORDER BY updated_at DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "document_id": r["document_id"],
            "filename": r["filename"],
            "file_type": r["file_type"],
            "current_version": r["current_version"],
            "char_count": r["char_count"],
            "chunk_count": r["chunk_count"],
            "embedding_model": r["embedding_model"],
            "indexed_at": r["indexed_at"],
            "metadata": json.loads(r["metadata_json"] or "{}"),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


def save_document_version(
    version_id: str,
    document_id: str,
    version_number: int,
    filename: str,
    content_hash: str,
    content_text: str,
    chunks: Optional[List[Dict[str, Any]]] = None,
    change_summary: str = ""
):
    """Saves a historical version snapshot for a document."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO document_versions (
            version_id, document_id, version_number, filename, content_hash,
            content_text, chunks_json, created_at, change_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        version_id,
        document_id,
        version_number,
        filename,
        content_hash,
        content_text,
        json.dumps(chunks or []),
        now,
        change_summary
    ))

    conn.commit()
    conn.close()


def get_document_versions(document_id: str) -> List[Dict[str, Any]]:
    """Retrieves all version snapshots for a document."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM document_versions WHERE document_id = ? ORDER BY version_number ASC", (document_id,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "version_id": r["version_id"],
            "document_id": r["document_id"],
            "version_number": r["version_number"],
            "filename": r["filename"],
            "content_hash": r["content_hash"],
            "content_text": r["content_text"],
            "chunks": json.loads(r["chunks_json"] or "[]"),
            "created_at": r["created_at"],
            "change_summary": r["change_summary"],
        }
        for r in rows
    ]


def list_document_jobs(limit: int = 30) -> List[Dict[str, Any]]:
    """Lists document processing jobs ordered by creation timestamp descending."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM studio_jobs WHERE job_type LIKE 'document_%' ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "job_id": r["job_id"],
            "job_type": r["job_type"],
            "status": r["status"],
            "current_stage": r["current_stage"],
            "progress": r["progress"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "duration_seconds": r["duration_seconds"],
            "error_message": r["error_message"],
            "input_params": json.loads(r["input_params"] or "{}"),
            "output_files": json.loads(r["output_files"] or "{}"),
            "logs": json.loads(r["logs"] or "[]"),
        }
        for r in rows
    ]

