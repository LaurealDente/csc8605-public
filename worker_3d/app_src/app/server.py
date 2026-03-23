# worker_3d/app_src/app/server.py
"""
Serveur HTTP d'administration du worker 3D.
Expose /health, /reload-model, /metrics sur le port 8080.

V2 : vérifie aussi le modèle MM-PatchCore.
"""

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
    "worker3d_tasks_processed_total",
    "Nombre total de taches 3D traitees",
    ["status"],
)
task_duration_seconds = Histogram(
    "worker3d_task_duration_seconds",
    "Duree de traitement d une tache 3D",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
anomaly_score_histogram = Histogram(
    "worker3d_anomaly_score",
    "Distribution des scores anomalie 3D",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
model_cache_version = Gauge(
    "worker3d_model_cache_version",
    "Version du modele 3D en cache",
)
model_mm_cache_version = Gauge(
    "worker3d_model_mm_cache_version",
    "Version du modele MM-PatchCore 3D en cache",
)

app = FastAPI(title="Worker 3D Admin API", version="2.0.0")

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/mlflow_model_cache")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "resnet_knn_3d")
MLFLOW_MODEL_NAME_MM = os.getenv("MLFLOW_MODEL_NAME_MM", "mm_patchcore_3d")


def _check_model_version(model_name: str, marker_file: str) -> dict:
    """Vérifie la version production d'un modèle MLflow."""
    result = {
        "model_name": model_name,
        "production_version": None,
        "cached": False,
        "status": "ok",
    }
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        latest = client.get_model_version_by_alias(model_name, "production")
        version_str = str(latest.version)
        result["production_version"] = version_str

        cache_path = Path(MODEL_CACHE_DIR) / f"{model_name}_v{version_str}"
        for candidate in [cache_path, cache_path / "model_artifacts"]:
            if (candidate / marker_file).exists():
                result["cached"] = True
                break
    except Exception as e:
        result["status"] = f"error: {e}"

    return result


@app.get("/health")
def health():
    # V1 model check
    v1 = _check_model_version(MLFLOW_MODEL_NAME, "embeddings.npy")
    if v1["production_version"]:
        try:
            model_cache_version.set(int(v1["production_version"]))
        except (ValueError, TypeError):
            pass

    # V2 MM-PatchCore model check
    v2 = _check_model_version(MLFLOW_MODEL_NAME_MM, "mm_patchcore_meta.json")
    if v2["production_version"]:
        try:
            model_mm_cache_version.set(int(v2["production_version"]))
        except (ValueError, TypeError):
            pass

    global_status = "ok"
    if v1["status"] != "ok" and v2["status"] != "ok":
        global_status = "degraded"

    return {
        "status": global_status,
        "pipeline": "3d",
        "models": {
            "v1_global_knn": v1,
            "v2_mm_patchcore": v2,
        },
    }


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
        # Garder les versions production de V1 et V2
        keep_names = set()
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()

            for model_name in [MLFLOW_MODEL_NAME, MLFLOW_MODEL_NAME_MM]:
                try:
                    latest = client.get_model_version_by_alias(model_name, "production")
                    keep_names.add(f"{model_name}_v{latest.version}")
                except Exception:
                    pass

            for item in cache_dir.iterdir():
                if item.name not in keep_names:
                    shutil.rmtree(str(item), ignore_errors=True)
                    deleted.append(item.name)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {
        "status": "ok",
        "message": "Cache vidé. Rechargement à la prochaine tâche.",
        "deleted": deleted,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def _start_http_server():
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")


def main():
    http_thread = threading.Thread(target=_start_http_server, daemon=True)
    http_thread.start()
    print("✅ [Worker 3D] Serveur HTTP démarré sur :8080")
    start_consumer()


if __name__ == "__main__":
    main()
