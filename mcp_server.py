import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError


mcp = FastMCP("secure-notes")

DB_PATH = Path(os.getenv("NOTES_DB", "data/notes.sqlite")).resolve()
MAX_NOTE_LENGTH = int(os.getenv("MAX_NOTE_LENGTH", "4000"))


def ensure_storage():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.chmod(DB_PATH.parent, 0o700) #linux/macos
    except OSError:
        pass

    with sqlite3.connect(DB_PATH, timeout=5) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_notes_owner_id
            ON notes(owner_id)
            """
        )

    try:
        os.chmod(DB_PATH, 0o600)
    except OSError:
        pass


def connect_db():
    ensure_storage()
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def current_identity():
    user_id = os.getenv("MCP_USER_ID", "").strip()

    if not user_id:
        raise ToolError("Требуется аутентификация: MCP_USER_ID не задан")

    return user_id


def row_to_note(row):
    return {
        "id": row["id"],
        "owner_id": row["owner_id"],
        "text": row["text"],
        "created_at": row["created_at"],
    }


def add_note(text):
    user_id = current_identity()

    clean_text = (text or "").strip()
    if not clean_text:
        raise ToolError("Текст заметки не должен быть пустым")

    if len(clean_text) > MAX_NOTE_LENGTH:
        raise ToolError(f"Заметка слишком длинная. Максимальная длина: {MAX_NOTE_LENGTH} символов")

    created_at = datetime.now(timezone.utc).isoformat()

    with closing(connect_db()) as conn:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO notes(owner_id, text, created_at)
                VALUES (?, ?, ?)
                """,
                (user_id, clean_text, created_at),
            )
            note_id = int(cursor.lastrowid)

    return {
        "id": note_id,
        "owner_id": user_id,
        "text": clean_text,
        "created_at": created_at,
    }


def list_notes():
    user_id = current_identity()

    with closing(connect_db()) as conn:
        rows = conn.execute(
            """
            SELECT id, owner_id, text, created_at
            FROM notes
            WHERE owner_id = ?
            ORDER BY id ASC
            """,
            (user_id,),
        ).fetchall()

    return [row_to_note(row) for row in rows]


def delete_note(id):
    user_id = current_identity()

    try:
        note_id = int(id)
    except (TypeError, ValueError):
        raise ToolError("ID заметки должен быть положительным целым числом")

    if note_id <= 0:
        raise ToolError("ID заметки должен быть положительным целым числом")

    with closing(connect_db()) as conn:
        with conn:
            cursor = conn.execute(
                "DELETE FROM notes WHERE id = ? AND owner_id = ?",
                (note_id, user_id),
            )
            deleted_count = cursor.rowcount

    if deleted_count == 0:
        raise ToolError("Заметка не найдена или доступ запрещен")

    return {"deleted": True, "id": note_id}


def update_note(id, text):
    user_id = current_identity()

    try:
        note_id = int(id)
    except (TypeError, ValueError):
        raise ToolError("ID заметки должен быть положительным целым числом")

    if note_id <= 0:
        raise ToolError("ID заметки должен быть положительным целым числом")

    clean_text = (text or "").strip()
    if not clean_text:
        raise ToolError("Текст заметки не должен быть пустым")

    if len(clean_text) > MAX_NOTE_LENGTH:
        raise ToolError(f"Заметка слишком длинная. Максимальная длина: {MAX_NOTE_LENGTH} символов")

    with closing(connect_db()) as conn:
        with conn:
            cursor = conn.execute(
                "UPDATE notes SET text = ? WHERE id = ? AND owner_id = ?",
                (clean_text, note_id, user_id),
            )
            updated_count = cursor.rowcount

    if updated_count == 0:
        raise ToolError("Заметка не найдена или доступ запрещен")

    with closing(connect_db()) as conn:
        row = conn.execute(
            """
            SELECT id, owner_id, text, created_at
            FROM notes
            WHERE id = ? AND owner_id = ?
            """,
            (note_id, user_id),
        ).fetchone()

    return row_to_note(row)


mcp.add_tool(add_note)
mcp.add_tool(list_notes)
mcp.add_tool(delete_note)
mcp.add_tool(update_note)


if __name__ == "__main__":
    ensure_storage()
    mcp.run(transport="stdio")
