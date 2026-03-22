# training_3d/src/mlflow_loader.py
"""
MLflow helpers pour le pipeline 3D.
V2 : support complet Multimodal PatchCore avec toutes les métriques.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "resnet_knn_3d")
MLFLOW_MODEL_NAME_MM: str = os.getenv("MLFLOW_MODEL_NAME_MM", "mm_patchcore_3d")
MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/mlflow_model_cache")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "worker_3d_fit")
MLFLOW_EXPERIMENT_NAME_MM: str = os.getenv("MLFLOW_EXPERIMENT_NAME_MM", "worker_3d_fit_mm")


def _get_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


# ---------------------------------------------------------------------------
# V1 — global embedding (inchangé)
# ---------------------------------------------------------------------------

def start_fit_run(
    backbone_name: str, batch_size: int, num_workers: int,
    table_name: str, model_dir: str,
    extra_params: Optional[Dict[str, Any]] = None,
) -> mlflow.ActiveRun:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_name = f"fit_3d_{backbone_name}_{int(time.time())}"
    params = {
        "pipeline": "3d", "backbone": backbone_name,
        "batch_size": batch_size, "num_workers": num_workers,
        "table_name": table_name, "model_dir": model_dir,
    }
    if extra_params:
        params.update(extra_params)
    active_run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(params)
    return active_run


def log_fit_metrics(model_dir: str, duration_seconds: float) -> None:
    metrics: Dict[str, float] = {"fit_duration_seconds": duration_seconds}
    embeddings_path = Path(model_dir) / "embeddings.npy"
    if embeddings_path.exists():
        bank = np.load(str(embeddings_path))
        metrics["bank_size"] = float(bank.shape[0])
        if bank.ndim > 1:
            metrics["embedding_dim"] = float(bank.shape[1])
    threshold_path = Path(model_dir) / "threshold.json"
    if threshold_path.exists():
        with threshold_path.open("r", encoding="utf-8") as f:
            th_data = json.load(f)
        if isinstance(th_data, dict) and "threshold" in th_data:
            metrics["threshold"] = float(th_data["threshold"])
    mlflow.log_metrics(metrics)


def log_fit_artifacts(model_dir: str) -> None:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"model_dir introuvable : {model_dir}")
    mlflow.log_artifacts(str(model_path), artifact_path="model_artifacts")


def register_run_to_registry(
    run_id: str, model_name: str = MLFLOW_MODEL_NAME,
    artifact_subpath: str = "model_artifacts",
) -> str:
    client = _get_client()
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass
    source = f"runs:/{run_id}/{artifact_subpath}"
    version = client.create_model_version(name=model_name, source=source, run_id=run_id)
    print(f"[MLflow Registry] {model_name} version {version.version}")
    return str(version.version)


def get_production_model_dir(
    model_name: str = MLFLOW_MODEL_NAME,
    local_cache_dir: str = MODEL_CACHE_DIR,
    force_download: bool = False,
) -> str:
    client = _get_client()
    try:
        latest = client.get_model_version_by_alias(model_name, "production")
    except Exception:
        raise RuntimeError(f"[MLflow] Aucune version de '{model_name}' avec l'alias 'production'.")
    version_str = str(latest.version)
    run_id = latest.run_id
    cache_path = Path(local_cache_dir) / f"{model_name}_v{version_str}"
    if (cache_path / "embeddings.npy").exists() and not force_download:
        return str(cache_path.resolve())
    if cache_path.exists():
        shutil.rmtree(str(cache_path))
    cache_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model_artifacts", dst_path=str(cache_path))
    if not (cache_path / "embeddings.npy").exists():
        raise RuntimeError("[MLflow] embeddings.npy absent après téléchargement.")
    return str(cache_path.resolve())


def get_current_production_version(model_name: str = MLFLOW_MODEL_NAME) -> Optional[str]:
    try:
        client = _get_client()
        version = client.get_model_version_by_alias(model_name, "production")
        return str(version.version)
    except Exception:
        return None


def log_eval_metrics(
    model_dir: str, config_path: str, table_name: str,
    backbone_name: str = "resnet18", threshold: float = 0.35,
    k: int = 5, num_workers: int = 0, batch_size: int = 8,
) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from .config import Settings
    from .data import PFEDataManager3D
    settings = Settings.from_yaml(config_path)
    dm = PFEDataManager3D(settings=settings)
    df = dm.get_dataset(table_name)
    df_test = df[df["split"] == "test"].copy() if "split" in df.columns else df.copy()
    if len(df_test) == 0:
        return {}
    normal_values = {"0", "normal", "good", "false"}
    y_true = [0 if str(row["label"]).lower() in normal_values else 1 for _, row in df_test.iterrows()]
    from .inference import predict_anomaly
    y_pred, y_scores = [], []
    for _, row in df_test.iterrows():
        try:
            img = dm.load_image(row["filepath"])
            score, label = predict_anomaly(pil_img=img, model_dir=model_dir, backbone_name=backbone_name, threshold=threshold, k=k)
            y_scores.append(score)
            y_pred.append(1 if label == "anomaly" else 0)
        except Exception:
            y_scores.append(0.0)
            y_pred.append(0)
    metrics = {
        "eval_n_test": float(len(df_test)),
        "eval_accuracy": accuracy_score(y_true, y_pred),
        "eval_precision": precision_score(y_true, y_pred, zero_division=0),
        "eval_recall": recall_score(y_true, y_pred, zero_division=0),
        "eval_f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true)) == 2:
        metrics["eval_auc_roc"] = roc_auc_score(y_true, y_scores)
    mlflow.log_metrics(metrics)
    return metrics


# ---------------------------------------------------------------------------
# V2 — Multimodal PatchCore
# ---------------------------------------------------------------------------

def start_fit_mm_run(
    table_name: str,
    model_dir: str,
    category: Optional[str] = None,
    alpha_rgb: float = 0.5,
    alpha_depth: float = 0.5,
    n_neighbors: int = 1,
    max_patches: int = 200_000,
    use_coreset: bool = True,
    use_multiscale: bool = True,
    extra_params: Optional[Dict[str, Any]] = None,
) -> mlflow.ActiveRun:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME_MM)

    run_name = f"fit_mm_3d_{category or 'all'}_{int(time.time())}"
    params: Dict[str, Any] = {
        "pipeline": "3d_mm",
        "model_type": "mm_patchcore",
        "table_name": table_name,
        "model_dir": model_dir,
        "category": category or "all",
        "alpha_rgb": alpha_rgb,
        "alpha_depth": alpha_depth,
        "n_neighbors": n_neighbors,
        "max_patches": max_patches,
        "use_coreset": use_coreset,
        "use_multiscale": use_multiscale,
    }
    if extra_params:
        params.update(extra_params)

    active_run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(params)
    return active_run


def log_fit_mm_metrics(model_dir: str, duration_seconds: float) -> None:
    """Logue les métriques du fit (taille banques, seuils calibrés)."""
    metrics: Dict[str, float] = {"fit_duration_seconds": duration_seconds}

    meta_path = Path(model_dir) / "mm_patchcore_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        metrics["rgb_bank_size"] = float(meta.get("rgb_bank_size", 0))
        metrics["depth_bank_size"] = float(meta.get("depth_bank_size", 0))
        metrics["image_size"] = float(meta.get("image_size", 224))

    thresholds_path = Path(model_dir) / "mm_patchcore_thresholds.json"
    if thresholds_path.exists():
        with thresholds_path.open("r", encoding="utf-8") as f:
            thresholds = json.load(f)
        for k, v in thresholds.items():
            try:
                metrics[f"calibrated_{k}"] = float(v)
            except (ValueError, TypeError):
                pass

    mlflow.log_metrics(metrics)


def log_eval_mm_full(
    eval_results: Dict[str, Any],
    split: str = "test",
) -> None:
    """
    Logue TOUTES les métriques d'évaluation dans MLflow.
    Attend le dict retourné par eval_mm_patchcore.run_full_evaluation().
    """
    prefix = f"eval_{split}"
    metrics: Dict[str, float] = {}

    # Image-level metrics
    im = eval_results.get("image_metrics", {})
    for k in [
        "n", "n_normal", "n_anomaly", "accuracy", "precision", "recall",
        "f1", "auroc", "ap", "fpr", "fnr", "tp", "fp", "tn", "fn",
        "best_f1", "best_f1_threshold", "best_precision", "best_recall",
        "threshold_at_fpr_01", "threshold_at_fpr_05", "threshold_at_fpr_10",
        "threshold", "duration_s", "errors",
    ]:
        v = im.get(k)
        if v is not None:
            try:
                metrics[f"{prefix}_image_{k}"] = float(v)
            except (ValueError, TypeError):
                pass

    # Score stats
    for cls in ["normal", "anomaly", "all"]:
        stats = im.get(f"score_stats_{cls}", {})
        for stat_k in ["mean", "std", "min", "max", "median"]:
            v = stats.get(stat_k)
            if v is not None:
                metrics[f"{prefix}_score_{cls}_{stat_k}"] = float(v)

    # Pixel-level metrics
    px = eval_results.get("pixel_metrics", {})
    for k in [
        "n_images", "n_pixels", "pixel_auroc", "pixel_ap",
        "pixel_f1_mean", "pixel_f1_std", "pixel_threshold", "duration_s",
    ]:
        v = px.get(k)
        if v is not None:
            try:
                metrics[f"{prefix}_pixel_{k}"] = float(v)
            except (ValueError, TypeError):
                pass

    # Per-category metrics (flat)
    cat_metrics = eval_results.get("category_metrics", {})
    for cat, cat_data in cat_metrics.items():
        cat_clean = cat.replace(" ", "_").replace("-", "_")
        for k in ["n", "f1", "auroc", "ap", "precision", "recall", "fpr"]:
            v = cat_data.get(k)
            if v is not None:
                try:
                    metrics[f"{prefix}_cat_{cat_clean}_{k}"] = float(v)
                except (ValueError, TypeError):
                    pass

    # Log tout d'un coup
    if metrics:
        mlflow.log_metrics(metrics)
        print(f"[MLflow] {len(metrics)} métriques loggées pour split='{split}'")

    # Log le JSON complet comme artifact
    eval_json = eval_results.copy()
    if "image_metrics" in eval_json:
        im_copy = dict(eval_json["image_metrics"])
        im_copy.pop("per_sample", None)
        eval_json["image_metrics"] = im_copy

    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_eval_{split}.json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(eval_json, f, indent=2, ensure_ascii=False)
        tmp_path = f.name
    mlflow.log_artifact(tmp_path, artifact_path="eval_results")
    os.unlink(tmp_path)


def log_fit_mm_artifacts(model_dir: str) -> None:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"model_dir introuvable : {model_dir}")
    mlflow.log_artifacts(str(model_path), artifact_path="model_artifacts")


def get_production_mm_model_dir(
    model_name: str = MLFLOW_MODEL_NAME_MM,
    local_cache_dir: str = MODEL_CACHE_DIR,
    force_download: bool = False,
) -> str:
    client = _get_client()
    try:
        latest = client.get_model_version_by_alias(model_name, "production")
    except Exception:
        raise RuntimeError(f"[MLflow] Aucune version de '{model_name}' avec l'alias 'production'.")
    version_str = str(latest.version)
    run_id = latest.run_id
    cache_path = Path(local_cache_dir) / f"{model_name}_v{version_str}"
    for candidate in [cache_path, cache_path / "model_artifacts"]:
        if (candidate / "mm_patchcore_meta.json").exists() and not force_download:
            return str(candidate.resolve())
    if cache_path.exists():
        shutil.rmtree(str(cache_path))
    cache_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model_artifacts", dst_path=str(cache_path))
    for candidate in [cache_path, cache_path / "model_artifacts"]:
        if (candidate / "mm_patchcore_meta.json").exists():
            return str(candidate.resolve())
    raise RuntimeError("[MLflow MM] mm_patchcore_meta.json absent après téléchargement.")
