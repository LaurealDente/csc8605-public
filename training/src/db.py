# worker_2d/app/db.py

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import Optional

from .config import Settings


# ==========================================================
# Engine management (singleton optional)
# ==========================================================

_engine: Optional[Engine] = None


def get_engine(settings: Settings) -> Engine:
    """
    Create (or reuse) a SQLAlchemy engine based on Settings.
    """
    global _engine

    if _engine is None:
        _engine = create_engine(
            settings.build_db_url(),
            pool_pre_ping=True,
        )

    return _engine


# ==========================================================
# Generic helpers
# ==========================================================

def update_task(
    engine: Engine,
    table: str,
    task_id: int,
    **fields,
) -> None:
    """
    Update a row in a task table.

    Example:
        update_task(engine, "tasks_2d", 5, status="done", result_path="...")
    """

    if not fields:
        return

    # Build SET clause safely
    set_clauses = [f"{k} = :{k}" for k in fields.keys()]
    set_clauses.append("updated_at = NOW()")

    sql = text(
        f"""
        UPDATE {table}
        SET {", ".join(set_clauses)}
        WHERE id = :task_id
        """
    )

    params = dict(fields)
    params["task_id"] = task_id

    with engine.begin() as conn:
        conn.execute(sql, params)


def fetch_task(
    engine: Engine,
    table: str,
    task_id: int,
) -> dict:
    """
    Fetch a task row by id.
    Returns empty dict if not found.
    """

    sql = text(
        f"""
        SELECT *
        FROM {table}
        WHERE id = :task_id
        """
    )

    with engine.connect() as conn:
        row = conn.execute(sql, {"task_id": task_id}).mappings().fetchone()

    return dict(row) if row else {}