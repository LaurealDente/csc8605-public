# training_3d/src/db.py
"""DB helpers — identique à la version 2D."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import Optional

from .config import Settings

_engine: Optional[Engine] = None


def get_engine(settings: Settings) -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.build_db_url(),
            pool_pre_ping=True,
        )
    return _engine


def update_task(engine: Engine, table: str, task_id: int, **fields) -> None:
    if not fields:
        return
    set_clauses = [f"{k} = :{k}" for k in fields.keys()]
    set_clauses.append("updated_at = NOW()")
    sql = text(
        f"UPDATE {table} SET {', '.join(set_clauses)} WHERE id = :task_id"
    )
    params = dict(fields)
    params["task_id"] = task_id
    with engine.begin() as conn:
        conn.execute(sql, params)


def fetch_task(engine: Engine, table: str, task_id: int) -> dict:
    sql = text(f"SELECT * FROM {table} WHERE id = :task_id")
    with engine.connect() as conn:
        row = conn.execute(sql, {"task_id": task_id}).mappings().fetchone()
    return dict(row) if row else {}
