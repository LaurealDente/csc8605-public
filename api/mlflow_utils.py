from __future__ import annotations

from typing import Any, Dict, Optional

from config import MLFLOW_TRACKING_URI, Pipeline, get_pipeline_config


def get_mlflow_production_version(model_name: str) -> Optional[str]:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        try:
            mv = client.get_model_version_by_alias(model_name, "production")
            if mv and getattr(mv, "version", None) is not None:
                return str(mv.version)
        except Exception:
            pass

        return None

    except Exception:
        return None


def get_pipeline_production_version(pipeline: Pipeline) -> Optional[str]:
    cfg = get_pipeline_config(pipeline)
    return get_mlflow_production_version(cfg["mlflow_model_name"])


def get_pipeline_mlflow_health(pipeline: Pipeline) -> Dict[str, Any]:
    cfg = get_pipeline_config(pipeline)

    model_name = cfg["mlflow_model_name"]
    version = get_mlflow_production_version(model_name)

    return {
        "status": "ok" if version is not None else "degraded",
        "models": {
            "main": {
                "model_name": model_name,
                "production_version": version,
                "status": "ok" if version is not None else "degraded",
            }
        },
    }