# worker_2d/app/main.py

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict

import requests
from PIL import Image
from io import BytesIO

from .config import Settings
from .db import get_engine, update_task
from .data import PFEDataManager
from .inference import (
    load_reference_bank,
    load_bank_meta,
    predict_anomaly,
    predict_patch_anomaly,
    fit_reference_bank,
    save_patch_heatmap,
    save_patch_overlay,
)
from .io_utils import write_result

# optional finetune command (only if you added train_finetune.py)
try:
    from .train_finetune import finetune_backbone  # type: ignore
except Exception:
    finetune_backbone = None  # noqa: N816


# ---------------------------
# Helpers
# ---------------------------

def _task_image_ref(task: Dict[str, Any]) -> str:
    return (
        task.get("image_path")
        or task.get("filepath")
        or task.get("image_url")
        or task.get("image")
        or ""
    )


def _get_model_dir(task: Dict[str, Any]) -> str:
    """
    Charge le modèle en production depuis MLflow Registry.
    Fallback sur le chemin local si MLflow est injoignable.
    """
    import os, mlflow, shutil
    from mlflow.tracking import MlflowClient
    from pathlib import Path

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "resnet_knn_2d")
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/tmp/mlflow_model_cache")

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        latest = client.get_model_version_by_alias(model_name, "production")
        version_str = str(latest.version)
        run_id = latest.run_id

        cache_path = Path(cache_dir) / f"{model_name}_v{version_str}"

        # Vérifier cache : embeddings.npy ou patch_bank.npy, dans / ou /model_artifacts
        for sub in ["", "model_artifacts"]:
            for fname in ["embeddings.npy", "patch_bank.npy"]:
                if (cache_path / sub / fname).exists():
                    model_path = cache_path if not sub else cache_path / sub
                    print(f"[MLflow] Modèle en cache : {model_path} (version {version_str})")
                    return str(model_path.resolve())

        if cache_path.exists():
            shutil.rmtree(str(cache_path))
        cache_path.mkdir(parents=True, exist_ok=True)

        print(f"[MLflow] Téléchargement {model_name} version {version_str}...")
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/model_artifacts",
            dst_path=str(cache_path),
        )

        # Après téléchargement, les fichiers sont dans model_artifacts/
        model_path = cache_path / "model_artifacts"
        if not model_path.exists():
            model_path = cache_path
        has_bank = any(
            (model_path / f).exists()
            for f in ["embeddings.npy", "patch_bank.npy"]
        )
        if not has_bank:
            raise RuntimeError(
                f"embeddings.npy ou patch_bank.npy absent après téléchargement dans {model_path}."
            )

        print(f"[MLflow] Modèle téléchargé : {model_path}")
        return str(model_path.resolve())

    except Exception as e:
        print(f"[MLflow] Erreur, fallback sur modèle local : {e}")
        task_model_name = task.get("model_name", "resnet_knn")
        task_model_version = task.get("model_version", "v1")
        worker_root = Path(__file__).resolve().parents[1]
        model_dir = worker_root / "models" / f"{task_model_name}_{task_model_version}"
        return str(model_dir.resolve())


def _load_task_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("task_json must contain a JSON object.")
    return data


def _load_task_image(dm: PFEDataManager, img_ref: str) -> Image.Image:
    """
    Priority: local file > HTTP download > dm.load_image fallback.
    """
    img_ref = str(img_ref).strip()

    # 1) Local file (e.g. /images_storage/uploads/...)
    if not img_ref.startswith("http") and os.path.isfile(img_ref):
        return Image.open(img_ref).convert("RGB")

    # 2) HTTP download
    if img_ref.startswith("http://") or img_ref.startswith("https://"):
        try:
            r = requests.get(img_ref, timeout=(10, 120))
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            print(f"[warn] HTTP load failed ({e}), trying dm.load_image fallback")

    # 3) Fallback via PFEDataManager (cache + image server + local)
    return dm.load_image(img_ref, strict=True)


# ---------------------------
# Commands
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

    # status running (best effort)
    try:
        update_task(engine, task_table, task_id, status="running", output_dir=str(output_dir))
    except Exception:
        pass

    try:
        dm = PFEDataManager(settings=settings)

        img_ref = _task_image_ref(task)
        if not img_ref:
            raise ValueError("Task must contain an image reference: image_url or filepath (etc.)")

        img = _load_task_image(dm, img_ref)

        model_dir = _get_model_dir(task)
        bank, bank_mode = load_reference_bank(model_dir)

        # ----------------------------------------
        # Load threshold (per-category or global)
        # ----------------------------------------

        category = task.get("category", "").strip().lower()

        # 1. Try per-category thresholds
        threshold = None
        for sub in ["", "model_artifacts"]:
            th_cat_path = Path(model_dir) / sub / "thresholds_per_category.json" if sub else Path(model_dir) / "thresholds_per_category.json"
            if th_cat_path.exists() and category:
                with th_cat_path.open() as f:
                    per_cat = json.load(f)
                if category in per_cat:
                    threshold = float(per_cat[category])
                    print(f"[threshold] Using per-category threshold for '{category}': {threshold}")
                    break

        # 2. Fallback to global threshold.json
        if threshold is None:
            threshold_path = Path(model_dir) / "threshold.json"
            if threshold_path.exists():
                with threshold_path.open() as f:
                    threshold = float(json.load(f)["threshold"])
                print(f"[threshold] Using global threshold: {threshold}")

        # 3. Fallback to config
        if threshold is None:
            threshold = settings.threshold
            print(f"[threshold] Using config default: {threshold}")

        # ----------------------------------------
        # Predict — auto-route based on bank mode
        # ----------------------------------------

        heatmap_path = None
        overlay_path = None

        if bank_mode == "patch":
            # Read feature_layer from meta if available, fallback to layer3
            meta = load_bank_meta(model_dir)
            feature_layer = meta.get("feature_layer", "layer3")

            print(f"[predict] mode=patch feature_layer={feature_layer} bank={bank.shape}")

            patch_result = predict_patch_anomaly(
                img,
                patch_bank=bank,
                model_dir=model_dir,
                feature_layer=feature_layer,
                patch_neighbors=1,
                image_score_mode="topk_mean",
                topk=settings.knn_k,
                threshold=threshold,
            )

            score = patch_result["image_score"]
            label = patch_result["pred_label"]

            # Save heatmap + overlay artifacts
            try:
                heatmap_path = save_patch_heatmap(
                    patch_result["patch_map"],
                    out_path=output_dir / "heatmap.png",
                    out_size=img.size,  # (width, height)
                )
                overlay_path = save_patch_overlay(
                    img,
                    patch_result["patch_map"],
                    out_path=output_dir / "overlay.png",
                    out_size=img.size,
                )
            except Exception as e:
                print(f"[warn] Heatmap/overlay save failed: {e}")

        else:
            # Global mode (dim=512)
            print(f"[predict] mode=global bank={bank.shape}")

            score, label = predict_anomaly(
                img,
                bank,
                k=settings.knn_k,
                threshold=threshold,
                model_dir=model_dir,
            )

        result = {
            "task_id": task_id,
            "status": "done",
            "model": f"{task.get('model_name','resnet_knn')}:{task.get('model_version','v1')}",
            "image_ref": img_ref,
            "anomaly_score": score,
            "pred_label": label,
            "inference_mode": bank_mode,
            "artifacts": {
                "result_json": str((output_dir / "result.json").resolve()).replace("\\", "/"),
                "heatmap_png": heatmap_path,
                "overlay_png": overlay_path,
            },
        }

        result_json_path = write_result(output_dir, result)

        # status done (best effort)
        try:
            update_task(
                engine,
                task_table,
                task_id,
                status="done",
                anomaly_score=score,
                pred_label=label,
                result_json=result_json_path,
            )
        except Exception:
            pass

        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        err = str(e)
        try:
            update_task(engine, task_table, task_id, status="failed", error_message=err)
        except Exception:
            pass
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

    print(f"✅ Reference bank created: {model_dir}/embeddings.npy")


def cmd_finetune(args) -> None:
    """
    Optional command: finetune backbone then you should run 'fit' to rebuild embeddings.npy.
    Requires worker_2d/app/train_finetune.py to exist.
    """
    if finetune_backbone is None:
        raise RuntimeError(
            "finetune_backbone is not available. "
            "Add worker_2d/app/train_finetune.py (and its dependencies) to enable this command."
        )

    out_model_dir = str(Path(args.output_model_dir).resolve())

    finetune_backbone(
        config_path=args.config,
        table_name=args.table_name,
        output_model_dir=out_model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
    )

    print(f"✅ Finetuned backbone saved in: {out_model_dir}/backbone_finetuned.pt")
    print("ℹ️ Now run: worker_2d fit --output-model-dir <same_dir> to rebuild embeddings.npy")


# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(prog="worker_2d")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- predict ----
    p_pred = sub.add_parser("predict", help="Run inference on one task JSON")
    p_pred.add_argument("--task-json", required=True)
    p_pred.add_argument("--config", default="conf/config.yaml")
    p_pred.add_argument("--task-table", default=os.getenv("TASK_TABLE", "tasks_2d"))
    p_pred.set_defaults(func=cmd_predict)

    # ---- fit ----
    p_fit = sub.add_parser("fit", help="Build reference bank (embeddings.npy) from DB")
    p_fit.add_argument("--config", default="conf/config.yaml")
    p_fit.add_argument("--table-name", default="mvtec_anomaly_detection")
    p_fit.add_argument("--output-model-dir", required=True)
    p_fit.add_argument("--batch-size", type=int, default=64)
    p_fit.add_argument("--num-workers", type=int, default=4)
    # (deprecated) kept for backward compat
    p_fit.add_argument("--image-list", required=False, help="(deprecated) not used anymore")
    p_fit.set_defaults(func=cmd_fit)

    # ---- finetune (optional) ----
    p_ft = sub.add_parser("finetune", help="Finetune ResNet18 backbone on (train, good) proxy task")
    p_ft.add_argument("--config", default="conf/config.yaml")
    p_ft.add_argument("--table-name", default="mvtec_anomaly_detection")
    p_ft.add_argument("--output-model-dir", required=True)
    p_ft.add_argument("--epochs", type=int, default=5)
    p_ft.add_argument("--batch-size", type=int, default=32)
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--num-workers", type=int, default=4)
    p_ft.set_defaults(func=cmd_finetune)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
