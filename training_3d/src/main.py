# training_3d/src/main.py
"""
Point d'entrée CLI du pipeline d'entraînement 3D.

Commandes :
  fit          — Construit la bank d'embeddings globaux (V1, RGB seul)
  predict      — Inférence V1 image-level sur une tâche JSON

  fit-mm       — ⭐ Entraîne le modèle Multimodal PatchCore (RGB + Depth)
  eval-mm      — ⭐ Évalue le modèle MM-PatchCore sur un split
  predict-mm   — ⭐ Inférence multimodale avec heatmaps de localisation

Usage :
    # V1 (legacy) — RGB seul, global k-NN
    python -m training_3d.src fit --output-model-dir ./models/resnet_knn_3d_v1
    python -m training_3d.src predict --task-json /tmp/task_42.json

    # V2 — Multimodal PatchCore (RGB + Depth)
    python -m training_3d.src fit-mm \\
        --table-name mvtec_3d_anomaly_detection \\
        --model-dir ./models/mm_patchcore_v1 \\
        --category bagel

    python -m training_3d.src predict-mm \\
        --task-json /tmp/task_42.json \\
        --model-dir ./models/mm_patchcore_v1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import requests
from PIL import Image

from .config import Settings
from .data import PFEDataManager3D
from .db import get_engine, update_task
from .inference import (
    clear_all_model_caches,
    fit_reference_bank,
    predict_anomaly,
)
from .io_utils import write_result

# Multimodal PatchCore imports
from .multimodal_patchcore import (
    MultimodalPatchCore,
    SamplePaths,
    build_samples_from_dataframe,
    save_heatmap,
    save_overlay,
)
from .eval_mm_patchcore import run_full_evaluation


# ---------------------------------------------------------------------------
# Helpers (inchangés)
# ---------------------------------------------------------------------------

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


def _resolve_model_dir(task: Dict[str, Any]) -> str:
    use_mlflow = os.getenv("USE_MLFLOW_REGISTRY", "false").lower() == "true"

    if not use_mlflow:
        return _get_local_model_dir(task)

    from .mlflow_loader import (
        MODEL_CACHE_DIR,
        MLFLOW_MODEL_NAME,
        get_production_model_dir,
    )

    model_name = task.get(
        "model_name", os.getenv("MLFLOW_MODEL_NAME", MLFLOW_MODEL_NAME)
    )
    cache_dir = os.getenv("MODEL_CACHE_DIR", MODEL_CACHE_DIR)

    try:
        return get_production_model_dir(
            model_name=model_name, local_cache_dir=cache_dir
        )
    except Exception as exc:
        print(f"[MLflow] Fallback sur le répertoire local : {exc}")
        return _get_local_model_dir(task)


def _get_local_model_dir(task: Dict[str, Any]) -> str:
    model_name = task.get("model_name", "resnet_knn_3d")
    model_version = task.get("model_version", "v1")
    worker_root = Path(__file__).resolve().parents[1]
    model_dir = worker_root / "models" / f"{model_name}_{model_version}"
    return str(model_dir.resolve())


def _load_task_json(path: str) -> Dict[str, Any]:
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"Task JSON introuvable : {task_path}")
    with task_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("task_json doit contenir un objet JSON (dict).")
    return data


def _load_task_image(dm: PFEDataManager3D, img_ref: str) -> Image.Image:
    if isinstance(img_ref, str) and (
        img_ref.startswith("http://") or img_ref.startswith("https://")
    ):
        resp = requests.get(img_ref, timeout=(10, 120))
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return dm.load_image(img_ref, strict=True)


def _load_threshold(model_dir: str, fallback_threshold: float) -> float:
    threshold_path = Path(model_dir) / "threshold.json"
    if not threshold_path.exists():
        return float(fallback_threshold)
    with threshold_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload.get("threshold", fallback_threshold))


def _load_model_selection(
    model_dir: str, fallback_k: int, fallback_threshold: float
) -> Dict[str, Any]:
    selection_path = Path(model_dir) / "selection.json"
    if not selection_path.exists():
        return {
            "k": int(fallback_k),
            "threshold": float(fallback_threshold),
            "score_mode": "mean",
            "bank_mode": "global",
            "backbone": "resnet18",
        }
    with selection_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "k": int(payload.get("k", fallback_k)),
        "threshold": float(payload.get("threshold", fallback_threshold)),
        "score_mode": str(payload.get("score_mode", "mean")),
        "bank_mode": str(payload.get("bank_mode", "global")),
        "backbone": str(payload.get("backbone", "resnet18")).lower(),
    }


def _safe_update_task(engine, task_table, task_id, **fields):
    try:
        update_task(engine, task_table, task_id, **fields)
    except Exception as exc:
        print(
            f"[DB] Avertissement : impossible de mettre à jour "
            f"task {task_id} : {exc}"
        )


# ---------------------------------------------------------------------------
# Commande V1 : predict (inchangée)
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> None:
    settings = Settings.from_yaml(args.config)
    engine = get_engine(settings)
    task = _load_task_json(args.task_json)

    if "task_id" not in task:
        raise KeyError("task.json missing 'task_id'")

    task_id = int(task["task_id"])
    task_table = args.task_table

    output_dir = Path(settings.outputs_dir) / str(task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    _safe_update_task(
        engine, task_table, task_id,
        status="running", output_dir=str(output_dir.resolve()),
    )

    try:
        dm = PFEDataManager3D(settings=settings)
        img_ref = _task_image_ref(task)
        if not img_ref:
            raise ValueError("La tâche ne contient aucune référence d'image.")

        img = _load_task_image(dm, img_ref)
        model_dir = _resolve_model_dir(task)
        fallback_threshold = _load_threshold(model_dir, settings.threshold)
        selection = _load_model_selection(
            model_dir, settings.knn_k, fallback_threshold
        )

        backbone_name = str(
            task.get("backbone", selection["backbone"])
        ).lower()
        score_mode = str(
            task.get("score_mode", selection["score_mode"])
        ).lower()
        k_value = int(task.get("k", selection["k"]))
        threshold = float(task.get("threshold", selection["threshold"]))

        score, label = predict_anomaly(
            img, bank=None, k=k_value, threshold=threshold,
            model_dir=model_dir, score_mode=score_mode,
            backbone_name=backbone_name,
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
            "threshold": threshold,
            "k": k_value,
            "score_mode": score_mode,
            "backbone": backbone_name,
        }

        result_json_path = write_result(output_dir, result)
        _safe_update_task(
            engine, task_table, task_id, status="done",
            anomaly_score=score, pred_label=label,
            result_json=result_json_path,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as exc:
        _safe_update_task(
            engine, task_table, task_id,
            status="failed", error_message=str(exc),
        )
        raise


# ---------------------------------------------------------------------------
# Commande V1 : fit (inchangée)
# ---------------------------------------------------------------------------

def cmd_fit(args: argparse.Namespace) -> None:
    import mlflow
    from .mlflow_loader import (
        MLFLOW_MODEL_NAME,
        log_fit_artifacts,
        log_fit_metrics,
        log_eval_metrics,
        register_run_to_registry,
        start_fit_run,
    )

    model_dir = str(Path(args.output_model_dir).resolve())
    backbone_name = str(args.backbone).lower()
    model_name = os.getenv("MLFLOW_MODEL_NAME", MLFLOW_MODEL_NAME)

    extra_params: Dict[str, Any] = {}
    if hasattr(args, "k") and args.k is not None:
        extra_params["k"] = args.k
    if hasattr(args, "threshold") and args.threshold is not None:
        extra_params["threshold"] = args.threshold

    with start_fit_run(
        backbone_name=backbone_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        table_name=args.table_name,
        model_dir=model_dir,
        extra_params=extra_params or None,
    ) as run:

        start_time = time.time()

        fit_reference_bank(
            model_dir=model_dir,
            config_path=args.config,
            table_name=args.table_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            backbone_name=backbone_name,
        )

        duration = time.time() - start_time
        log_fit_metrics(model_dir=model_dir, duration_seconds=duration)

        log_eval_metrics(
            model_dir=model_dir,
            config_path=args.config,
            table_name=args.table_name,
            backbone_name=backbone_name,
            threshold=getattr(args, "threshold", 0.35),
            k=getattr(args, "k", 5),
            num_workers=0,
        )

        log_fit_artifacts(model_dir=model_dir)

        run_id = run.info.run_id
        print(f"[MLflow] Run ID : {run_id}")

        version = register_run_to_registry(
            run_id=run_id, model_name=model_name
        )
        print(
            f"[MLflow] Version {version} créée pour '{model_name}'.\n"
            f"         → Aller sur l'UI MLflow pour promouvoir en Production."
        )

    clear_all_model_caches(model_dir=model_dir, backbone_name=backbone_name)
    print(f"✅ Bank de référence 3D créée : {model_dir}/embeddings.npy")


# ---------------------------------------------------------------------------
# ⭐ Commande V2 : fit-mm — Entraînement Multimodal PatchCore
# ---------------------------------------------------------------------------

def cmd_fit_mm(args: argparse.Namespace) -> None:
    """
    Entraîne un modèle Multimodal PatchCore (RGB + Depth).

    Pipeline complet :
      1. Charge les samples normaux depuis la DB
      2. Extrait les patches multiscale RGB et Depth
      3. Construit les banques de patches avec réduction coreset
      4. Calibre les seuils sur le split de validation
      5. Sauvegarde le modèle
      6. Évaluation complète (image + pixel + par catégorie) sur validation ET test
      7. Logue TOUTES les métriques dans MLflow
    """
    from .mlflow_loader import (
        MLFLOW_MODEL_NAME_MM,
        start_fit_mm_run,
        log_fit_mm_metrics,
        log_fit_mm_artifacts,
        log_eval_mm_full,
        register_run_to_registry,
    )

    model_dir = str(Path(args.model_dir).resolve())
    model_name_mm = os.getenv("MLFLOW_MODEL_NAME_MM", MLFLOW_MODEL_NAME_MM)

    settings = Settings.from_yaml(args.config)
    dm = PFEDataManager3D(settings=settings)

    # Charger le DataFrame complet (avec xyz_filepath)
    df = dm.get_dataset(table=args.table_name, verbose=True)

    # Construire les samples d'entraînement
    train_samples = build_samples_from_dataframe(
        df,
        split=args.fit_split,
        normal_only=args.normal_only,
        category=args.category,
    )
    print(f"[fit-mm] Samples d'entraînement : {len(train_samples)}")

    # Créer le modèle
    model = MultimodalPatchCore(
        image_size=args.image_size,
        alpha_rgb=args.alpha_rgb,
        alpha_depth=args.alpha_depth,
        n_neighbors=args.k,
        use_multiscale=not args.disable_multiscale,
        use_late_fusion=True,
    )

    # MLflow run
    with start_fit_mm_run(
        table_name=args.table_name,
        model_dir=model_dir,
        category=args.category,
        alpha_rgb=args.alpha_rgb,
        alpha_depth=args.alpha_depth,
        n_neighbors=args.k,
        max_patches=args.max_patches,
        use_coreset=not args.no_coreset,
        use_multiscale=not args.disable_multiscale,
    ) as run:

        start_time = time.time()

        model.fit(
            train_samples,
            dm,
            max_patches_per_modality=args.max_patches,
            coreset=not args.no_coreset,
            pre_sample_size=args.coreset_pre_sample_size,
            proj_dim=args.coreset_proj_dim,
        )

        duration = time.time() - start_time
        print(f"[fit-mm] Durée du fit : {duration:.1f}s")

        # Calibration des seuils sur le split validation
        try:
            val_samples = build_samples_from_dataframe(
                df,
                split=args.val_split,
                normal_only=False,
                category=args.category,
            )
            thresholds = model.calibrate_thresholds(
                val_samples, dm, use_best_f1_if_labels_exist=True,
            )
            print(f"[fit-mm] Seuils calibrés : {thresholds}")
        except Exception as e:
            print(f"[fit-mm] Calibration ignorée : {e}")

        # Sauvegarder
        model.save(Path(model_dir))

        # MLflow : métriques du fit
        log_fit_mm_metrics(model_dir=model_dir, duration_seconds=duration)

        # ===============================================
        # Évaluation complète sur VALIDATION
        # ===============================================
        try:
            val_samples_eval = build_samples_from_dataframe(
                df, split=args.val_split, normal_only=False,
                category=args.category,
            )
            print(f"\n[fit-mm] Évaluation validation ({len(val_samples_eval)} samples)...")
            val_results = run_full_evaluation(
                model, val_samples_eval, dm,
                threshold_key="image_mean_plus_3std",
                pixel_threshold_key="pixel_mean_plus_3std",
            )
            log_eval_mm_full(val_results, split="val")
        except Exception as e:
            print(f"[fit-mm] Évaluation validation ignorée : {e}")

        # ===============================================
        # Évaluation complète sur TEST
        # ===============================================
        try:
            test_samples = build_samples_from_dataframe(
                df, split="test", normal_only=False,
                category=args.category,
            )
            print(f"\n[fit-mm] Évaluation test ({len(test_samples)} samples)...")
            test_results = run_full_evaluation(
                model, test_samples, dm,
                threshold_key="image_mean_plus_3std",
                pixel_threshold_key="pixel_mean_plus_3std",
            )
            log_eval_mm_full(test_results, split="test")

            # Save per-category thresholds
            cat_metrics = test_results.get("category_metrics", {})
            per_cat_thresholds = {}
            for cat, m in cat_metrics.items():
                if "threshold" in m:
                    per_cat_thresholds[cat] = m["threshold"]
            if per_cat_thresholds:
                import json as _json2
                th_cat_path = Path(model_dir) / "thresholds_per_category.json"
                with th_cat_path.open("w", encoding="utf-8") as f:
                    _json2.dump(per_cat_thresholds, f, indent=2)
                print(f"[fit-mm] thresholds_per_category.json sauvegardé ({len(per_cat_thresholds)} catégories)")

            # Sauvegarder le JSON d'éval dans le model_dir
            eval_path = Path(model_dir) / "eval_test.json"
            eval_save = dict(test_results)
            if "image_metrics" in eval_save:
                im = dict(eval_save["image_metrics"])
                im.pop("per_sample", None)
                eval_save["image_metrics"] = im
            import json as _json
            with eval_path.open("w", encoding="utf-8") as f:
                _json.dump(eval_save, f, indent=2, ensure_ascii=False)
            print(f"[fit-mm] Éval test sauvegardée : {eval_path}")
        except Exception as e:
            print(f"[fit-mm] Évaluation test ignorée : {e}")

        # Artifacts (inclut eval_test.json)
        log_fit_mm_artifacts(model_dir=model_dir)

        run_id = run.info.run_id
        print(f"\n[MLflow MM] Run ID : {run_id}")

        version = register_run_to_registry(
            run_id=run_id, model_name=model_name_mm,
        )
        print(
            f"[MLflow MM] Version {version} créée pour '{model_name_mm}'.\n"
            f"         → Aller sur l'UI MLflow pour promouvoir en Production."
        )

    print(
        f"\n✅ Modèle MM-PatchCore entraîné et sauvé dans {model_dir}\n"
        f"   Fichiers créés :\n"
        f"     - rgb_patch_bank.npy\n"
        f"     - depth_patch_bank.npy\n"
        f"     - mm_patchcore_meta.json\n"
        f"     - mm_patchcore_thresholds.json\n"
        f"     - eval_test.json"
    )


# ---------------------------------------------------------------------------
# ⭐ Commande V2 : eval-mm — Évaluation du modèle
# ---------------------------------------------------------------------------

def cmd_eval_mm(args: argparse.Namespace) -> None:
    """
    Évalue le modèle MM-PatchCore avec toutes les métriques :
    image-level, pixel-level, par catégorie.
    """
    settings = Settings.from_yaml(args.config)
    dm = PFEDataManager3D(settings=settings)

    model = MultimodalPatchCore.load(args.model_dir)

    df = dm.get_dataset(table=args.table_name, verbose=True)
    eval_samples = build_samples_from_dataframe(
        df, split=args.split, normal_only=False, category=args.category,
    )
    print(f"[eval-mm] Samples d'évaluation : {len(eval_samples)}")

    results = run_full_evaluation(
        model, eval_samples, dm,
        threshold_key=args.threshold_key,
        pixel_threshold_key=args.pixel_threshold_key,
    )

    # Sauvegarder le JSON
    output_path = Path(args.model_dir) / f"eval_{args.split}.json"
    save_data = dict(results)
    if "image_metrics" in save_data:
        im = dict(save_data["image_metrics"])
        im.pop("per_sample", None)
        save_data["image_metrics"] = im

    save_data["table_name"] = args.table_name
    save_data["split"] = args.split
    save_data["category"] = args.category

    import json as _json
    with output_path.open("w", encoding="utf-8") as f:
        _json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Évaluation sauvegardée : {output_path}")


# ---------------------------------------------------------------------------
# ⭐ Commande V2 : predict-mm — Inférence multimodale
# ---------------------------------------------------------------------------

def cmd_predict_mm(args: argparse.Namespace) -> None:
    """
    Inférence multimodale sur une tâche JSON.
    Génère les heatmaps et overlays en plus du score.
    """
    settings = Settings.from_yaml(args.config)
    engine = get_engine(settings)
    task = _load_task_json(args.task_json)

    if "task_id" not in task:
        raise KeyError("task.json missing 'task_id'")

    task_id = int(task["task_id"])
    task_table = args.task_table

    output_dir = Path(settings.outputs_dir) / str(task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    _safe_update_task(
        engine, task_table, task_id,
        status="running", output_dir=str(output_dir.resolve()),
    )

    try:
        dm = PFEDataManager3D(settings=settings)

        # Charger image RGB
        img_ref = _task_image_ref(task)
        if not img_ref:
            raise ValueError("La tâche ne contient aucune référence d'image RGB.")
        rgb_img = _load_task_image(dm, img_ref)

        # Charger depth map
        depth_ref = _task_depth_ref(task)
        if not depth_ref:
            raise ValueError(
                "La tâche ne contient aucune référence depth/xyz. "
                "Clés attendues : xyz_filepath, depth_filepath, xyz_path, depth_path"
            )
        depth_arr = dm.load_depth_map(depth_ref, strict=True)
        if depth_arr is not None and depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 2]

        # Charger le modèle MM-PatchCore
        model_dir = args.model_dir or _resolve_model_dir(task)
        model = MultimodalPatchCore.load(model_dir)

        # Prédiction
        pred = model.predict(rgb_img, depth_arr, upsample_to_input=True)

        # Seuil
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

        # Sauvegarder heatmaps
        heatmap_path = save_heatmap(
            pred["fused_map"],
            output_dir / "heatmap_fused.png",
        )
        overlay_path = save_overlay(
            rgb_img,
            pred["fused_map"],
            output_dir / "overlay_fused.png",
            alpha=args.overlay_alpha,
        )

        # Heatmaps par modalité
        heatmap_rgb_path = save_heatmap(
            pred["rgb_map"],
            output_dir / "heatmap_rgb.png",
        )
        heatmap_depth_path = save_heatmap(
            pred["depth_map"],
            output_dir / "heatmap_depth.png",
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
            "alpha_rgb": model.alpha_rgb,
            "alpha_depth": model.alpha_depth,
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

    except Exception as exc:
        _safe_update_task(
            engine, task_table, task_id,
            status="failed", error_message=str(exc),
        )
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="training_3d",
        description=(
            "Pipeline d'entraînement 3D — "
            "V1 (ResNet + k-NN) et V2 (Multimodal PatchCore)"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- predict (V1 legacy) ----
    p_pred = sub.add_parser("predict", help="Inférence V1 sur une tâche JSON")
    p_pred.add_argument("--task-json", required=True)
    p_pred.add_argument("--config", default="conf/config.yaml")
    p_pred.add_argument(
        "--task-table", default=os.getenv("TASK_TABLE", "tasks_3d")
    )
    p_pred.set_defaults(func=cmd_predict)

    # ---- fit (V1 legacy) ----
    p_fit = sub.add_parser("fit", help="Construit la bank d'embeddings V1")
    p_fit.add_argument("--config", default="conf/config.yaml")
    p_fit.add_argument("--table-name", default="mvtec_3d_anomaly_detection")
    p_fit.add_argument("--output-model-dir", required=True)
    p_fit.add_argument("--batch-size", type=int, default=64)
    p_fit.add_argument("--num-workers", type=int, default=4)
    p_fit.add_argument(
        "--backbone", default="resnet18", choices=["resnet18", "resnet50"]
    )
    p_fit.add_argument("--image-list", required=False, help="(deprecated)")
    p_fit.set_defaults(func=cmd_fit)

    # ---- fit-mm (V2 Multimodal PatchCore) ----
    p_fmm = sub.add_parser(
        "fit-mm",
        help="⭐ Entraîne le modèle Multimodal PatchCore (RGB + Depth)",
    )
    p_fmm.add_argument("--config", default="conf/config.yaml")
    p_fmm.add_argument("--table-name", default="mvtec_3d_anomaly_detection")
    p_fmm.add_argument("--model-dir", required=True)
    p_fmm.add_argument("--fit-split", default="train")
    p_fmm.add_argument("--val-split", default="validation")
    p_fmm.add_argument("--normal-only", action="store_true", default=True)
    p_fmm.add_argument("--category", default=None)
    p_fmm.add_argument("--image-size", type=int, default=224)
    p_fmm.add_argument("--max-patches", type=int, default=200_000)
    p_fmm.add_argument("--no-coreset", action="store_true")
    p_fmm.add_argument("--coreset-pre-sample-size", type=int, default=60_000)
    p_fmm.add_argument("--coreset-proj-dim", type=int, default=128)
    p_fmm.add_argument("--alpha-rgb", type=float, default=0.5)
    p_fmm.add_argument("--alpha-depth", type=float, default=0.5)
    p_fmm.add_argument("--k", type=int, default=1)
    p_fmm.add_argument("--disable-multiscale", action="store_true")
    p_fmm.set_defaults(func=cmd_fit_mm)

    # ---- eval-mm ----
    p_emm = sub.add_parser(
        "eval-mm",
        help="⭐ Évalue le modèle MM-PatchCore (image + pixel + catégorie)",
    )
    p_emm.add_argument("--config", default="conf/config.yaml")
    p_emm.add_argument("--table-name", default="mvtec_3d_anomaly_detection")
    p_emm.add_argument("--model-dir", required=True)
    p_emm.add_argument("--split", default="test")
    p_emm.add_argument("--category", default=None)
    p_emm.add_argument("--threshold-key", default="image_mean_plus_3std")
    p_emm.add_argument("--pixel-threshold-key", default="pixel_mean_plus_3std")
    p_emm.set_defaults(func=cmd_eval_mm)

    # ---- predict-mm (V2) ----
    p_pmm = sub.add_parser(
        "predict-mm",
        help="⭐ Inférence multimodale avec heatmaps",
    )
    p_pmm.add_argument("--task-json", required=True)
    p_pmm.add_argument("--config", default="conf/config.yaml")
    p_pmm.add_argument(
        "--task-table", default=os.getenv("TASK_TABLE", "tasks_3d")
    )
    p_pmm.add_argument("--model-dir", default=None)
    p_pmm.add_argument("--overlay-alpha", type=float, default=0.45)
    p_pmm.set_defaults(func=cmd_predict_mm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
