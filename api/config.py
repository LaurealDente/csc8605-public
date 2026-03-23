from __future__ import annotations

import os
from enum import Enum
from typing import Dict


class Pipeline(str, Enum):
    two_d = "2d"
    three_d = "3d"


APP_TITLE = os.getenv("APP_TITLE", "PFE API — Détection d'anomalies 2D / 3D")
APP_VERSION = os.getenv("APP_VERSION", "3.0.0")

# ---------------------------------------------------------------------------
# Public URLs (Ingress)
# ---------------------------------------------------------------------------

PUBLIC_API_URL: str = os.getenv(
    "PUBLIC_API_URL",
    "https://api.exemple.com",
)

ADMINER_PUBLIC_URL: str = os.getenv(
    "ADMINER_PUBLIC_URL",
    "https://adminer.exemple.com",
)

GRAFANA_PUBLIC_URL: str = os.getenv(
    "GRAFANA_PUBLIC_URL",
    "https://grafana.exemple.com",
)

IMAGES_PUBLIC_UI_URL: str = os.getenv(
    "IMAGES_PUBLIC_UI_URL",
    "https://images.exemple.com",
)

MLFLOW_PUBLIC_URL: str = os.getenv(
    "MLFLOW_PUBLIC_URL",
    "https://mlflow.exemple.com",
)

PREFECT_PUBLIC_URL: str = os.getenv(
    "PREFECT_PUBLIC_URL",
    "https://prefect.exemple.com",
)

PROMETHEUS_PUBLIC_URL: str = os.getenv(
    "PROMETHEUS_PUBLIC_URL",
    "https://prometheus.exemple.com",
)

# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

DB_HOST: str = os.getenv("DB_HOST", "postgres")
DB_PORT: str = os.getenv("DB_PORT", "5432").split(":")[-1].rstrip("/")
DB_NAME: str = os.getenv("DB_NAME", "anomaly_detection")
DB_USER: str = os.getenv("DB_USER", "admin")
DB_PASS: str = os.getenv("DB_PASS", "")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ---------------------------------------------------------------------------
# Stockage images
# ---------------------------------------------------------------------------

IMAGES_STORAGE_ROOT: str = os.getenv("IMAGES_STORAGE_ROOT", "/images_storage")
IMAGES_PUBLIC_BASE: str = os.getenv(
    "IMAGES_PUBLIC_BASE",
    "https://images.exemple.com",
)

# ---------------------------------------------------------------------------
# RabbitMQ
# ---------------------------------------------------------------------------

RABBIT_HOST: str = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBIT_PORT: int = int(os.getenv("RABBITMQ_PORT_NUMBER", "5672"))
RABBIT_USER: str = os.getenv("RABBITMQ_USER", "guest")
RABBIT_PASS: str = os.getenv("RABBITMQ_PASSWORD", "")

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-service:5000",
)

# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------

MAX_UPLOAD_SIZE_BYTES: int = int(
    os.getenv("MAX_UPLOAD_SIZE_BYTES", str(50 * 1024 * 1024))
)

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
    "application/octet-stream",
}

# ---------------------------------------------------------------------------
# Config par pipeline
# ---------------------------------------------------------------------------

PIPELINE_CONFIG: Dict[str, Dict[str, str]] = {
    "2d": {
        "task_table": os.getenv("TASK_TABLE_2D", "tasks_2d"),
        "rabbit_queue": os.getenv("RABBITMQ_QUEUE_2D", "tasks_2d"),
        "worker_admin_url": os.getenv("WORKER_ADMIN_URL_2D", "http://worker-2d-service:8080"),
        "worker_health_url": os.getenv("WORKER_HEALTH_URL_2D", "http://worker-2d-service:8080/health"),
        "category_default": os.getenv("CATEGORY_DEFAULT_2D", "pill"),
        "model_name_default": os.getenv("MODEL_NAME_DEFAULT_2D", "resnet_knn_2d"),
        "model_version_default": os.getenv("MODEL_VERSION_DEFAULT_2D", "v1"),
        "mlflow_model_name": os.getenv("MLFLOW_MODEL_NAME_2D", "resnet_knn_2d"),
        "task_type": "2d_anomaly",
        "output_prefix": "results/tasks_2d",
        "display_name": "Pipeline 2D",
    },
    "3d": {
        "task_table": os.getenv("TASK_TABLE_3D", "tasks_3d"),
        "rabbit_queue": os.getenv("RABBITMQ_QUEUE_3D", "tasks_3d"),
        "worker_admin_url": os.getenv("WORKER_ADMIN_URL_3D", "http://worker-3d-service:8080"),
        "worker_health_url": os.getenv("WORKER_HEALTH_URL_3D", "http://worker-3d-service:8080/health"),
        "category_default": os.getenv("CATEGORY_DEFAULT_3D", "bagel"),
        "model_name_default": os.getenv("MODEL_NAME_DEFAULT_3D", "mm_patchcore_3d"),
        "model_version_default": os.getenv("MODEL_VERSION_DEFAULT_3D", "v1"),
        "mlflow_model_name": os.getenv("MLFLOW_MODEL_NAME_3D", "mm_patchcore_3d"),
        "task_type": "3d_anomaly",
        "output_prefix": "results/tasks_3d",
        "display_name": "Pipeline 3D",
    },
}


def get_pipeline_config(pipeline: Pipeline) -> Dict[str, str]:
    return PIPELINE_CONFIG[pipeline.value]
