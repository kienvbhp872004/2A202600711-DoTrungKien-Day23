"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    Supports in-memory persistence by default and SQLite for local durable checkpoints.
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        sqlite_target = database_url or "outputs/langgraph_lab.sqlite"
        if sqlite_target.startswith("sqlite:///"):
            sqlite_target = sqlite_target.removeprefix("sqlite:///")

        db_path = Path(sqlite_target)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        saver = SqliteSaver(conn=conn)
        saver.setup()
        return saver

    if kind == "postgres":
        raise ValueError(
            "Postgres checkpointer is not configured in this project. "
            "Use kind='memory' or kind='sqlite'."
        )

    raise ValueError(f"Unknown checkpointer kind: {kind}")
