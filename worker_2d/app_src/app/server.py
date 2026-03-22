from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import (
    Counter, Gauge, Histogram,
    CONTENT_TYPE_LATEST, REGISTRY, generate_latest,
)

from .queue_consumer import main as start_consumer

tasks_processed_total = Counter(
    "worker_tasks_processed_total",
    "Nombre total de taches traitees",
    ["status"],
)
task_duration_seconds = Histogram(
    "worker_task_duration_seconds",
    "Duree de traitement d une tache",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
anomaly_score_histogram = Histogram(
    "worker_anomaly_score",
    "Distribution des scores anomalie",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
model_cache_version = Gauge(
    "worker_model_cache_version",
    "Version du modele en cache",
)

app = FastAPI(title="Worker 2D Admin API", version="1.0.0")

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/mlflow_model_cache")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "resnet_knn_2d")


@app.get("/health")
def health():
    result = {
        "status": "ok",
        "mlflow": "ok",
        "model_production_version": None,
        "model_cached": False,
    }
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        latest = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
        version_str = str(latest.version)
        result["model_production_version"] = version_str
        cache_path = Path(MODEL_CACHE_DIR) / f"{MLFLOW_MODEL_NAME}_v{version_str}"
        result["model_cached"] = (cache_path / "embeddings.npy").exists()
        model_cache_version.set(int(version_str))
    except Exception as e:
        result["mlflow"] = f"error: {e}"
        result["status"] = "degraded"
    return result


@app.post("/reload-model")
def reload_model(force: bool = False):
    cache_dir = Path(MODEL_CACHE_DIR)
    if not cache_dir.exists():
        return {"status": "ok", "message": "Cache deja vide.", "deleted": []}
    deleted = []
    if force:
        for item in cache_dir.iterdir():
            shutil.rmtree(str(item), ignore_errors=True)
            deleted.append(item.name)
    else:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()
            latest = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
            production_cache = f"{MLFLOW_MODEL_NAME}_v{latest.version}"
            for item in cache_dir.iterdir():
                if item.name != production_cache:
                    shutil.rmtree(str(item), ignore_errors=True)
                    deleted.append(item.name)
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "ok", "message": "Cache vide. Rechargement a la prochaine tache.", "deleted": deleted}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def _start_http_server():
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")


def main():
    http_thread = threading.Thread(target=_start_http_server, daemon=True)
    http_thread.start()
    print("✅ Serveur HTTP demarre sur :8080")
    start_consumer()


if __name__ == "__main__":
    main()
