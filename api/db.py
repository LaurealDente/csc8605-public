from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config import DATABASE_URL, Pipeline, get_pipeline_config

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    future=True,
)

def get_distinct_categories_from_table(table_name: str) -> list[str]:
    query = text(f"""
        SELECT DISTINCT category
        FROM {table_name}
        WHERE category IS NOT NULL
          AND category <> ''
        ORDER BY category ASC
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    return [row[0] for row in rows]

def check_db_connection() -> str:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:
        return f"error: {exc}"


def insert_task(
    *,
    pipeline: Pipeline,
    image_path: str,
    image_url: str,
    category: str,
    model_name: str,
    model_version: str,
) -> int:
    cfg = get_pipeline_config(pipeline)
    task_table = cfg["task_table"]
    task_type = cfg["task_type"]

    sql = text(f"""
        INSERT INTO {task_table}
            (status, task_type, image_path, image_url, category, model_name, model_version)
        VALUES
            ('pending', :task_type, :image_path, :image_url, :category, :model_name, :model_version)
        RETURNING id;
    """)

    try:
        with engine.begin() as conn:
            task_id = conn.execute(
                sql,
                {
                    "task_type": task_type,
                    "image_path": image_path,
                    "image_url": image_url,
                    "category": category,
                    "model_name": model_name,
                    "model_version": model_version,
                },
            ).scalar_one()
        return int(task_id)
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Erreur insertion tâche en base: {exc}") from exc


# ⭐ Nouvelle fonction pour les tâches multimodales
def insert_task_mm(
    *,
    pipeline: Pipeline,
    image_path: str,
    image_url: str,
    depth_path: str,
    depth_url: str,
    category: str,
    model_name: str,
    model_version: str,
) -> int:
    """Insert une tâche 3D multimodale (RGB + Depth) dans tasks_3d."""
    cfg = get_pipeline_config(pipeline)
    task_table = cfg["task_table"]
    task_type = cfg.get("task_type_mm", "3d_anomaly_mm")

    sql = text(f"""
        INSERT INTO {task_table}
            (status, task_type, image_path, image_url, depth_path, depth_url,
             category, model_name, model_version)
        VALUES
            ('pending', :task_type, :image_path, :image_url, :depth_path, :depth_url,
             :category, :model_name, :model_version)
        RETURNING id;
    """)

    try:
        with engine.begin() as conn:
            task_id = conn.execute(
                sql,
                {
                    "task_type": task_type,
                    "image_path": image_path,
                    "image_url": image_url,
                    "depth_path": depth_path,
                    "depth_url": depth_url,
                    "category": category,
                    "model_name": model_name,
                    "model_version": model_version,
                },
            ).scalar_one()
        return int(task_id)
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Erreur insertion tâche MM en base: {exc}") from exc


def get_task(task_id: int, pipeline: Pipeline) -> Optional[Dict[str, Any]]:
    cfg = get_pipeline_config(pipeline)
    task_table = cfg["task_table"]

    sql = text(f"SELECT * FROM {task_table} WHERE id = :id;")

    try:
        with engine.begin() as conn:
            row = conn.execute(sql, {"id": task_id}).mappings().fetchone()
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Erreur lecture tâche: {exc}") from exc

    if not row:
        return None

    data = dict(row)
    data["pipeline"] = pipeline.value
    return data


def update_task_status(
    *,
    pipeline: Pipeline,
    task_id: int,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    cfg = get_pipeline_config(pipeline)
    task_table = cfg["task_table"]

    sql = text(f"""
        UPDATE {task_table}
        SET
            status = :status,
            error_message = :error_message
        WHERE id = :task_id;
    """)

    try:
        with engine.begin() as conn:
            conn.execute(
                sql,
                {
                    "status": status,
                    "error_message": error_message,
                    "task_id": task_id,
                },
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Erreur mise à jour tâche: {exc}") from exc


def get_task_counts_by_status(pipeline: Pipeline) -> Dict[str, int]:
    cfg = get_pipeline_config(pipeline)
    task_table = cfg["task_table"]

    sql = text(f"SELECT status, COUNT(*) AS cnt FROM {task_table} GROUP BY status;")

    try:
        with engine.begin() as conn:
            rows = conn.execute(sql).fetchall()
    except SQLAlchemyError:
        return {}

    return {str(row[0]): int(row[1]) for row in rows}


def ensure_tables_exist() -> None:
    ddl_2d = """
    CREATE TABLE IF NOT EXISTS tasks_2d (
        id BIGSERIAL PRIMARY KEY,
        status VARCHAR(32) NOT NULL,
        task_type VARCHAR(64) NOT NULL,
        image_path TEXT NOT NULL,
        image_url TEXT NOT NULL,
        category VARCHAR(128),
        model_name VARCHAR(128),
        model_version VARCHAR(64),
        anomaly_score DOUBLE PRECISION,
        pred_label VARCHAR(64),
        result_json JSONB,
        error_message TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        started_at TIMESTAMP NULL,
        finished_at TIMESTAMP NULL
    );
    """

    # ⭐ tasks_3d avec colonnes depth_path / depth_url
    ddl_3d = """
    CREATE TABLE IF NOT EXISTS tasks_3d (
        id BIGSERIAL PRIMARY KEY,
        status VARCHAR(32) NOT NULL,
        task_type VARCHAR(64) NOT NULL,
        image_path TEXT NOT NULL,
        image_url TEXT NOT NULL,
        depth_path TEXT,
        depth_url TEXT,
        category VARCHAR(128),
        model_name VARCHAR(128),
        model_version VARCHAR(64),
        anomaly_score DOUBLE PRECISION,
        pred_label VARCHAR(64),
        result_json JSONB,
        error_message TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        started_at TIMESTAMP NULL,
        finished_at TIMESTAMP NULL
    );
    """

    # ⭐ Migration : ajouter les colonnes depth si la table existe déjà
    migrate_depth = [
        "ALTER TABLE tasks_3d ADD COLUMN IF NOT EXISTS depth_path TEXT;",
        "ALTER TABLE tasks_3d ADD COLUMN IF NOT EXISTS depth_url TEXT;",
    ]

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_tasks_2d_status ON tasks_2d(status);",
        "CREATE INDEX IF NOT EXISTS idx_tasks_2d_created_at ON tasks_2d(created_at);",
        "CREATE INDEX IF NOT EXISTS idx_tasks_3d_status ON tasks_3d(status);",
        "CREATE INDEX IF NOT EXISTS idx_tasks_3d_created_at ON tasks_3d(created_at);",
    ]

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl_2d))
            conn.execute(text(ddl_3d))
            for stmt in migrate_depth:
                conn.execute(text(stmt))
            for idx in indexes:
                conn.execute(text(idx))
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Erreur création tables/index: {exc}") from exc
