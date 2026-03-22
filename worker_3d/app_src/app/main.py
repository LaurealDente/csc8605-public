# worker_3d/app_src/app/main.py
"""
Point d'entrée CLI du worker 3D de détection d'anomalies.

Commandes :
  predict      — Inférence V1 (RGB seul, global k-NN)
  fit          — Construit la bank V1
  predict-mm   — ⭐ Inférence Multimodal PatchCore (RGB + Depth) avec heatmaps
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import requests
from PIL import Image
from io import BytesIO

from .config import Settings
from .db import get_engine, update_task
from .data import PFEDataManager3D
from .inference import load_reference_bank, predict_anomaly, fit_reference_bank
from .io_utils import write_result

# Multimodal PatchCore imports
from .multimodal_patchcore import (
    MultimodalPatchCore,
    save_heatmap,
    save_overlay,
)


# ---------------------------
# Helpers
# ---------------------------

def _task_image_ref(task: Dict[str, Any]) -> str:
    return (
        task.get("image_url")
        or task.get("filepath")
        or task.get("image_path")
        or task.get("image")
        or ""
    )


def _task_depth_ref(task: Dict[str, Any]) -> str:
    """Résout la référence depth/xyz depuis le payload de la tâche."""
    return (
        task.get("xyz_filepath")
        or task.get("depth_filepath")
        or task.get("xyz_path")
        or task.get("depth_path")
        or ""
    )


def _get_model_dir(task: Dict[str, Any]) -> str:
    """
    Charge le modèle 3D en production depuis MLflow Registry.
    Fallback sur le chemin local si MLflow est injoignable.
    """
    import mlflow
    import shutil
    from mlflow.tracking import MlflowClient

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "resnet_knn_3d")
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/tmp/mlflow_model_cache")

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        latest = client.get_model_version_by_alias(model_name, "production")
        version_str = str(latest.version)
        run_id = latest.run_id

        cache_path = Path(cache_dir) / f"{model_name}_v{version_str}"

        for candidate in [cache_path, cache_path / "model_artifacts"]:
            if (candidate / "embeddings.npy").exists():
                print(
                    f"[MLflow] Modèle en cache : {candidate} "
                    f"(version {version_str})"
                )
                return str(candidate.resolve())

        if cache_path.exists():
            shutil.rmtree(str(cache_path))
        cache_path.mkdir(parents=True, exist_ok=True)

        print(
            f"[MLflow] Téléchargement {model_name} version {version_str}..."
        )
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/model_artifacts",
            dst_path=str(cache_path),
        )

        for candidate in [cache_path, cache_path / "model_artifacts"]:
            if (candidate / "embeddings.npy").exists():
                print(f"[MLflow] Modèle téléchargé : {candidate}")
                return str(candidate.resolve())

        raise RuntimeError("embeddings.npy absent après téléchargement.")

    except Exception as e:
        print(f"[MLflow] Erreur, fallback sur modèle local : {e}")
        task_model_name = task.get("model_name", "resnet_knn_3d")
        task_model_version = task.get("model_version", "v1")
        worker_root = Path(__file__).resolve().parents[1]
        model_dir = (
            worker_root / "models"
            / f"{task_model_name}_{task_model_version}"
        )
        return str(model_dir.resolve())


def _get_mm_model_dir(task: Dict[str, Any]) -> str:
    """
    Résout le répertoire du modèle MM-PatchCore.
    Essaie MLflow d'abord, puis fallback local.
    """
    import mlflow
    import shutil
    from mlflow.tracking import MlflowClient

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name = os.getenv("MLFLOW_MODEL_NAME_MM", "mm_patchcore_3d")
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/tmp/mlflow_model_cache")

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        latest = client.get_model_version_by_alias(model_name, "production")
        version_str = str(latest.version)
        run_id = latest.run_id

        cache_path = Path(cache_dir) / f"{model_name}_v{version_str}"

        for candidate in [cache_path, cache_path / "model_artifacts"]:
            if (candidate / "mm_patchcore_meta.json").exists():
                print(
                    f"[MLflow] Modèle MM en cache : {candidate} "
                    f"(version {version_str})"
                )
                return str(candidate.resolve())

        if cache_path.exists():
            shutil.rmtree(str(cache_path))
        cache_path.mkdir(parents=True, exist_ok=True)

        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/model_artifacts",
            dst_path=str(cache_path),
        )

        for candidate in [cache_path, cache_path / "model_artifacts"]:
            if (candidate / "mm_patchcore_meta.json").exists():
                return str(candidate.resolve())

        raise RuntimeError("mm_patchcore_meta.json absent après téléchargement.")

    except Exception as e:
        print(f"[MLflow] MM fallback local : {e}")
        task_model_name = task.get("model_name", "mm_patchcore_3d")
        task_model_version = task.get("model_version", "v1")
        worker_root = Path(__file__).resolve().parents[1]
        model_dir = (
            worker_root / "models"
            / f"{task_model_name}_{task_model_version}"
        )
        return str(model_dir.resolve())


def _load_task_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("task_json must contain a JSON object.")
    return data


def _load_task_image(dm: PFEDataManager3D, img_ref: str) -> Image.Image:
    if isinstance(img_ref, str) and (
        img_ref.startswith("http://") or img_ref.startswith("https://")
    ):
        r = requests.get(img_ref, timeout=(10, 120))
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    return dm.load_image(img_ref, strict=True)


def _safe_update_task(engine, task_table, task_id, **fields):
    try:
        update_task(engine, task_table, task_id, **fields)
    except Exception:
        pass


# ---------------------------
# Commands V1 (inchangées)
# ---------------------------

def cmd_predict(args) -> None:
    settings = Settings.from_yaml(args.config)
    engine = get_engine(settings)

    task = _load_task_json(args.task_json)

    if "task_id" not in task:
        raise KeyError("task.json missing required field: 'task_id'")

    task_id = int(task["task_id"])
    task_table = args.task_table

    output_dir = Path(settings.outputs_dir) / str(task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    _safe_update_task(
        engine, task_table, task_id,
        status="running", output_dir=str(output_dir),
    )

    try:
        dm = PFEDataManager3D(settings=settings)

        img_ref = _task_image_ref(task)
        if not img_ref:
            raise ValueError("Task must contain an image reference.")

        img = _load_task_image(dm, img_ref)
        model_dir = _get_model_dir(task)
        bank = load_reference_bank(model_dir)

        # Load threshold
        threshold_path = Path(model_dir) / "threshold.json"
        if threshold_path.exists():
            with threshold_path.open() as f:
                threshold = json.load(f)["threshold"]
        else:
            threshold = settings.threshold

        score, label = predict_anomaly(
            img, bank, k=settings.knn_k,
            threshold=threshold, model_dir=model_dir,
        )

        result = {
            "task_id": task_id,
            "status": "done",
            "model": (
                f"{task.get('model_name', 'resnet_knn_3d')}"
                f":{task.get('model_version', 'v1')}"
            ),
            "image_ref": img_ref,
            "anomaly_score": score,
            "pred_label": label,
        }

        result_json_path = write_result(output_dir, result)

        _safe_update_task(
            engine, task_table, task_id,
            status="done", anomaly_score=score,
            pred_label=label, result_json=result_json_path,
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        _safe_update_task(
            engine, task_table, task_id,
            status="failed", error_message=str(e),
        )
        raise


def cmd_fit(args) -> None:
    model_dir = str(Path(args.output_model_dir).resolve())
    fit_reference_bank(
        model_dir=model_dir,
        config_path=args.config,
        table_name=args.table_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"✅ Reference bank 3D created: {model_dir}/embeddings.npy")


# ---------------------------
# ⭐ Command V2 : predict-mm
# ---------------------------

def cmd_predict_mm(args) -> None:
    """
    Inférence Multimodal PatchCore sur une tâche JSON.
    Génère heatmaps fusionnées, RGB, depth + overlay.
    """
    settings = Settings.from_yaml(args.config)
    engine = get_engine(settings)

    task = _load_task_json(args.task_json)

    if "task_id" not in task:
        raise KeyError("task.json missing required field: 'task_id'")

    task_id = int(task["task_id"])
    task_table = args.task_table

    output_dir = Path(settings.outputs_dir) / str(task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    _safe_update_task(
        engine, task_table, task_id,
        status="running", output_dir=str(output_dir),
    )

    try:
        dm = PFEDataManager3D(settings=settings)

        # RGB
        img_ref = _task_image_ref(task)
        if not img_ref:
            raise ValueError("Task must contain an image reference.")
        rgb_img = _load_task_image(dm, img_ref)

        # Depth
        depth_ref = _task_depth_ref(task)
        if not depth_ref:
            raise ValueError(
                "Task must contain a depth/xyz reference "
                "(xyz_filepath, depth_filepath, etc.)."
            )
        depth_arr = dm.load_depth_map(depth_ref, strict=True)
        if depth_arr is not None and depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 2]

        # Charger le modèle
        model_dir = args.model_dir or _get_mm_model_dir(task)
        model = MultimodalPatchCore.load(model_dir)

        # Prédiction
        pred = model.predict(rgb_img, depth_arr, upsample_to_input=True)

        # Seuil — per-category > best_f1 > mean+3std > 0.5
        category = task.get("category", "").strip().lower()
        threshold = None

        # 1. Try per-category thresholds
        th_cat_path = Path(model_dir) / "thresholds_per_category.json"
        if th_cat_path.exists() and category:
            with th_cat_path.open() as f:
                per_cat = json.load(f)
            if category in per_cat:
                threshold = float(per_cat[category])
                print(f"[threshold] Using per-category threshold for '{category}': {threshold}")

        # 2. Fallback to model thresholds
        if threshold is None:
            threshold = float(
                task.get(
                    "threshold",
                    model.thresholds.get(
                        "image_best_f1",
                        model.thresholds.get("image_mean_plus_3std", 0.5),
                    ),
                )
            )

        fused_score = pred["fused_score"]
        pred_label = "anomaly" if fused_score >= threshold else "normal"

        # Heatmaps
        heatmap_path = save_heatmap(
            pred["fused_map"], output_dir / "heatmap_fused.png",
        )
        overlay_path = save_overlay(
            rgb_img, pred["fused_map"],
            output_dir / "overlay_fused.png",
            alpha=args.overlay_alpha,
        )
        heatmap_rgb_path = save_heatmap(
            pred["rgb_map"], output_dir / "heatmap_rgb.png",
        )
        heatmap_depth_path = save_heatmap(
            pred["depth_map"], output_dir / "heatmap_depth.png",
        )

        result = {
            "task_id": task_id,
            "status": "done",
            "model_type": "mm_patchcore",
            "model_dir": str(Path(model_dir).resolve()),
            "image_ref": img_ref,
            "depth_ref": depth_ref,
            "fused_score": float(fused_score),
            "rgb_score": float(pred["rgb_score"]),
            "depth_score": float(pred["depth_score"]),
            "anomaly_score": float(fused_score),
            "pred_label": pred_label,
            "threshold": threshold,
            "artifacts": {
                "heatmap_fused": heatmap_path,
                "overlay_fused": overlay_path,
                "heatmap_rgb": heatmap_rgb_path,
                "heatmap_depth": heatmap_depth_path,
            },
        }

        result_json_path = write_result(output_dir, result)
        result["artifacts"]["result_json"] = result_json_path

        _safe_update_task(
            engine, task_table, task_id, status="done",
            anomaly_score=float(fused_score), pred_label=pred_label,
            result_json=result_json_path,
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        _safe_update_task(
            engine, task_table, task_id,
            status="failed", error_message=str(e),
        )
        raise


# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(prog="worker_3d")
    sub = p.add_subparsers(dest="cmd", required=True)

    # predict (V1)
    p_pred = sub.add_parser("predict", help="Inférence V1")
    p_pred.add_argument("--task-json", required=True)
    p_pred.add_argument("--config", default="conf/config.yaml")
    p_pred.add_argument(
        "--task-table", default=os.getenv("TASK_TABLE", "tasks_3d")
    )
    p_pred.set_defaults(func=cmd_predict)

    # fit (V1)
    p_fit = sub.add_parser("fit", help="Construit la bank V1")
    p_fit.add_argument("--config", default="conf/config.yaml")
    p_fit.add_argument("--table-name", default="mvtec_3d_anomaly_detection")
    p_fit.add_argument("--output-model-dir", required=True)
    p_fit.add_argument("--batch-size", type=int, default=64)
    p_fit.add_argument("--num-workers", type=int, default=4)
    p_fit.set_defaults(func=cmd_fit)

    # predict-mm (V2)
    p_pmm = sub.add_parser(
        "predict-mm",
        help="⭐ Inférence Multimodal PatchCore avec heatmaps",
    )
    p_pmm.add_argument("--task-json", required=True)
    p_pmm.add_argument("--config", default="conf/config.yaml")
    p_pmm.add_argument(
        "--task-table", default=os.getenv("TASK_TABLE", "tasks_3d")
    )
    p_pmm.add_argument("--model-dir", default=None)
    p_pmm.add_argument("--overlay-alpha", type=float, default=0.45)
    p_pmm.set_defaults(func=cmd_predict_mm)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
