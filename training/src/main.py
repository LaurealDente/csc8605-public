# training/src/main.py

from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from PIL import Image

from .config import Settings
from .data import PFEDataManager
from .db import get_engine, update_task
from .patch_inference import (
    predict_patch_anomaly,
    save_patch_heatmap,
    save_patch_overlay,
)

# optional patch fit command
try:
    from .patch_inference import fit_patch_reference_bank  # type: ignore
except Exception:
    fit_patch_reference_bank = None  # noqa: N816

# optional finetune command
try:
    from .train_finetune import finetune_backbone  # type: ignore
except Exception:
    finetune_backbone = None  # noqa: N816

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


# ---------------------------
# Helpers
# ---------------------------

def _resolve_model_dir(task: Dict[str, Any]) -> str:
    use_mlflow = os.getenv("USE_MLFLOW_REGISTRY", "false").lower() == "true"
    if not use_mlflow:
        return _get_model_dir(task) # Fallback local

    from .mlflow_loader import MODEL_CACHE_DIR, MLFLOW_MODEL_NAME, get_production_model_dir
    model_name = task.get("model_name", os.getenv("MLFLOW_MODEL_NAME", MLFLOW_MODEL_NAME))
    cache_dir = os.getenv("MODEL_CACHE_DIR", MODEL_CACHE_DIR)

    try:
        return get_production_model_dir(model_name=model_name, local_cache_dir=cache_dir)
    except Exception as exc:
        print(f"[MLflow] Échec, fallback local : {exc}")
        return _get_model_dir(task)

def _task_image_ref(task: Dict[str, Any]) -> str:
    """
    Resolve image reference from task payload.
    Accepted keys:
      - image_url
      - filepath
      - image_path
      - image
    """
    return (
        task.get("image_url")
        or task.get("filepath")
        or task.get("image_path")
        or task.get("image")
        or ""
    )


def _get_model_dir(task: Dict[str, Any]) -> str:
    """
    Resolve:
      worker_2d/models/<model_name>_<model_version>
    """
    model_name = task.get("model_name", "resnet_knn")
    model_version = task.get("model_version", "v1")

    worker_root = Path(__file__).resolve().parents[1]
    model_dir = worker_root / "models" / f"{model_name}_{model_version}"
    return str(model_dir.resolve())


def _load_task_json(path: str) -> Dict[str, Any]:
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"Task JSON not found: {task_path}")

    with task_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("task_json must contain a JSON object.")

    return data


def _load_task_image(dm: PFEDataManager, img_ref: str) -> Image.Image:
    """
    Load task image:
    - if http(s): download strictly
    - else: use dm.load_image(..., strict=True)
    """
    img_ref = str(img_ref).strip()
    if not img_ref:
        raise ValueError("Empty image reference.")

    if img_ref.startswith("http://") or img_ref.startswith("https://"):
        resp = requests.get(img_ref, timeout=(10, 120))
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    return dm.load_image(img_ref, strict=True)


def _load_threshold_from_json(path: Path) -> Optional[float]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        return None

    if "threshold" in payload:
        return float(payload["threshold"])

    tuning = payload.get("tuning")
    if isinstance(tuning, dict) and "threshold" in tuning:
        return float(tuning["threshold"])

    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and "threshold" in metrics:
        return float(metrics["threshold"])

    return None


def _resolve_threshold(model_dir: str, fallback_threshold: float) -> float:
    """
    Resolution order:
      1) <model_dir>/threshold.json
      2) <model_dir>/exp_patch_validation.json
      3) <model_dir>/exp_patch_test.json
      4) fallback_threshold from config
    """
    model_path = Path(model_dir)

    candidates = [
        model_path / "threshold.json",
        model_path / "exp_patch_validation.json",
        model_path / "exp_patch_test.json",
    ]

    for path in candidates:
        value = _load_threshold_from_json(path)
        if value is not None:
            return float(value)

    return float(fallback_threshold)


def _load_patch_selection(
    model_dir: str,
    fallback_threshold: float,
) -> Dict[str, Any]:
    """
    Load optional patch selection config from <model_dir>/selection.json.

    Supported keys:
      - threshold
      - backbone
      - feature_layer
      - patch_neighbors
      - image_score_mode
      - topk
    """
    selection_path = Path(model_dir) / "selection.json"

    defaults = {
        "threshold": float(fallback_threshold),
        "backbone": "resnet18",
        "feature_layer": "layer3",
        "patch_neighbors": 1,
        "image_score_mode": "topk_mean",
        "topk": 5,
    }

    if not selection_path.exists():
        return defaults

    with selection_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid selection file format: {selection_path}")

    return {
        "threshold": float(payload.get("threshold", defaults["threshold"])),
        "backbone": str(payload.get("backbone", defaults["backbone"])).lower(),
        "feature_layer": str(payload.get("feature_layer", defaults["feature_layer"])),
        "patch_neighbors": int(payload.get("patch_neighbors", defaults["patch_neighbors"])),
        "image_score_mode": str(payload.get("image_score_mode", defaults["image_score_mode"])).lower(),
        "topk": int(payload.get("topk", defaults["topk"])),
    }


def _safe_update_task(
    engine,
    task_table: str,
    task_id: int,
    **fields: Any,
) -> None:
    """
    Best-effort DB update.
    """
    try:
        update_task(engine, task_table, task_id, **fields)
    except Exception:
        pass


def write_result(output_dir: str | Path, result: Dict[str, Any]) -> str:
    """
    Write result.json inside output_dir.

    Adds:
        - written_at (UTC ISO 8601)
    Returns:
        Absolute normalized path to result.json
    """

    if not isinstance(result, dict):
        raise ValueError("Result must be a dictionary.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "result.json"

    # Copy result to avoid mutating caller dict
    result_data = dict(result)
    result_data["written_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    # Normalize path (important for Windows Docker cases)
    return str(result_path.resolve()).replace("\\", "/")

# ---------------------------
# Commands
# ---------------------------

def cmd_predict(args) -> None:
    settings = Settings.from_yaml(args.config)
    engine = get_engine(settings)
    task = _load_task_json(args.task_json)
    task_id = int(task["task_id"])
    task_table = args.task_table

    output_dir = Path(settings.outputs_dir) / str(task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    _safe_update_task(engine, task_table, task_id, status="running", output_dir=str(output_dir.resolve()))

    try:
        dm = PFEDataManager(settings=settings)
        img_ref = _task_image_ref(task)
        img = _load_task_image(dm, img_ref)

        # 1. RÉSOLUTION VIA MLFLOW (Le grand changement ici)
        model_dir = _resolve_model_dir(task)
        base_threshold = _resolve_threshold(model_dir, settings.threshold)

        selection = _load_patch_selection(model_dir=model_dir, fallback_threshold=base_threshold)

        # 2. PRÉDICTION PATCH
        pred = predict_patch_anomaly(
            pil_img=img,
            patch_bank=None,
            model_dir=model_dir,
            backbone_name=str(task.get("backbone", selection["backbone"])).lower(),
            feature_layer=str(task.get("feature_layer", selection["feature_layer"])),
            patch_neighbors=int(task.get("patch_neighbors", selection["patch_neighbors"])),
            image_score_mode=str(task.get("image_score_mode", selection["image_score_mode"])).lower(),
            topk=int(task.get("topk", selection["topk"])),
            threshold=float(task.get("threshold", selection["threshold"])),
        )

        # Sauvegarde des images
        heatmap_path = save_patch_heatmap(pred["patch_map"], output_dir / "heatmap.png", img.size).replace("\\", "/")
        overlay_path = save_patch_overlay(img, pred["patch_map"], output_dir / "overlay.png", float(args.overlay_alpha), img.size).replace("\\", "/")

        # 3. ÉCRITURE DU RÉSULTAT ET SYNCHRO DB
        result = {
            "task_id": task_id,
            "status": "done",
            "image_ref": img_ref,
            "anomaly_score": float(pred["image_score"]), 
            "pred_label": pred["pred_label"],
            "artifacts": {
                "heatmap_png": heatmap_path,
                "overlay_png": overlay_path,
            },
        }
        
        result_json_path = write_result(output_dir, result)
        result["artifacts"]["result_json"] = result_json_path

        _safe_update_task(
            engine, task_table, task_id,
            status="done",
            anomaly_score=float(pred["image_score"]),
            pred_label=pred["pred_label"],
            result_json=result_json_path,
        )

    except Exception as e:
        _safe_update_task(engine, task_table, task_id, status="failed", error_message=str(e))
        raise


def cmd_fit(args) -> None:
    if fit_patch_reference_bank is None:
        raise RuntimeError("fit_patch_reference_bank est introuvable.")

    import time
    import mlflow
    from .mlflow_loader import (
        MLFLOW_MODEL_NAME, start_fit_run, log_fit_metrics, log_eval_metrics, 
        log_fit_artifacts, register_run_to_registry
    )

    model_dir = str(Path(args.output_model_dir).resolve())
    backbone_name = str(args.backbone).lower()
    model_name = os.getenv("MLFLOW_MODEL_NAME", MLFLOW_MODEL_NAME)

    extra_params = {
        "feature_layer": args.feature_layer,
        "fit_split": args.fit_split,
        "normal_only": args.normal_only,
        "category": getattr(args, "category", None),
    }

    # 1. OUVERTURE DU RUN MLFLOW
    with start_fit_run(
        backbone_name=backbone_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        table_name=args.table_name,
        model_dir=model_dir,
        extra_params=extra_params
    ) as run:
        
        start_time = time.time()

        # 2. ENTRAÎNEMENT PATCH (Coreset, etc.)
        fit_patch_reference_bank(
            model_dir=model_dir,
            config_path=args.config,
            table_name=args.table_name,
            backbone_name=backbone_name,
            feature_layer=args.feature_layer,
            fit_split=args.fit_split,
            normal_only=bool(args.normal_only),
            category=getattr(args, "category", None),
        )

        duration = time.time() - start_time

        # 3. LOGGING MLFLOW (Métriques et Artefacts)
        log_fit_metrics(model_dir=model_dir, duration_seconds=duration)
        log_fit_artifacts(model_dir=model_dir) # Va uploader le patch_bank.npy !


        # 3b. EVALUATION SUR LE SPLIT TEST
        print("[Eval] Lancement de l'evaluation sur le split test...")
        try:
            eval_metrics = log_eval_metrics(
                model_dir=model_dir,
                config_path=args.config,
                table_name=args.table_name,
                backbone_name=backbone_name,
                feature_layer=args.feature_layer,
            )
            auroc = eval_metrics.get('eval_auroc', 'N/A')
            f1 = eval_metrics.get('optimal_f1', 'N/A')
            print(f"[Eval] AUROC={auroc}, F1={f1}")
        except Exception as eval_err:
            print(f"[Eval] Evaluation echouee (non bloquant): {eval_err}")

        run_id = run.info.run_id
        
        # 4. REGISTRY
        version = register_run_to_registry(run_id=run_id, model_name=model_name)
        print(f"[MLflow] Version {version} créée pour le modèle Patch '{model_name}'.")

    print(f"✅ Patch reference bank créée et envoyée sur MLflow depuis : {model_dir}")


def cmd_finetune(args) -> None:
    """
    Finetune backbone then rebuild patch_bank.npy with 'fit'.
    Requires worker_2d/app/train_finetune.py to exist.
    """
    if finetune_backbone is None:
        raise RuntimeError(
            "finetune_backbone is not available. "
            "Add worker_2d/app/train_finetune.py (and its dependencies) "
            "to enable this command."
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
    print(
        "ℹ️ Now run: worker_2d fit "
        "--output-model-dir <same_dir> "
        "--feature-layer layer3 "
        "--fit-split train "
        "--normal-only"
    )


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="worker_2d")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- predict ----
    p_pred = sub.add_parser("predict", help="Run patch inference on one task JSON")
    p_pred.add_argument("--task-json", required=True)
    p_pred.add_argument("--config", default="conf/config.yaml")
    p_pred.add_argument("--task-table", default=os.getenv("TASK_TABLE", "tasks_2d"))
    p_pred.add_argument("--overlay-alpha", type=float, default=0.45)
    p_pred.set_defaults(func=cmd_predict)

    # ---- fit ----
    p_fit = sub.add_parser("fit", help="Build patch_bank.npy from DB")
    p_fit.add_argument("--config", default="conf/config.yaml")
    p_fit.add_argument("--table-name", default="mvtec_anomaly_detection")
    p_fit.add_argument("--output-model-dir", required=True)
    p_fit.add_argument("--batch-size", type=int, default=64)
    p_fit.add_argument("--num-workers", type=int, default=4)
    p_fit.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    p_fit.add_argument("--feature-layer", default="layer3")
    p_fit.add_argument("--fit-split", default="train")
    p_fit.add_argument("--normal-only", action="store_true")
    p_fit.add_argument("--category", default=None, help="Filter by category (e.g. pill, bottle)")
    # deprecated, kept for backward compatibility
    p_fit.add_argument("--image-list", required=False, help="(deprecated) not used anymore")
    p_fit.set_defaults(func=cmd_fit)

    # ---- finetune ----
    p_ft = sub.add_parser(
        "finetune",
        help="Finetune ResNet backbone on proxy task",
    )
    p_ft.add_argument("--config", default="conf/config.yaml")
    p_ft.add_argument("--table-name", default="mvtec_ad_2")
    p_ft.add_argument("--output-model-dir", required=True)
    p_ft.add_argument("--epochs", type=int, default=5)
    p_ft.add_argument("--batch-size", type=int, default=32)
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--num-workers", type=int, default=4)
    p_ft.set_defaults(func=cmd_finetune)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
