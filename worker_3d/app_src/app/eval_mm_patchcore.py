# training_3d/src/eval_mm_patchcore.py
"""
Évaluation complète du modèle Multimodal PatchCore.

Métriques calculées :
  Image-level : AUROC, AP, F1, best F1 threshold, precision, recall,
                mean/std/min/max scores par classe, per-sample details
  Pixel-level : pixel AUROC, pixel AP, pixel F1 (via ground truth masks)
  Par catégorie : F1, AUROC, AP ventilés par catégorie produit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import Settings
from .data import PFEDataManager3D
from .multimodal_patchcore import (
    NORMAL_LABELS,
    MultimodalPatchCore,
    SamplePaths,
    build_samples_from_dataframe,
)


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


# ------------------------------------------------------------------
# Best F1 threshold search
# ------------------------------------------------------------------

def compute_best_f1_threshold(
    y_true: List[int], y_score: List[float]
) -> Dict[str, float]:
    """Cherche le seuil qui maximise le F1 sur la courbe precision-recall."""
    if not y_true or len(set(y_true)) < 2:
        return {}

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return {}

    f1s = (
        2 * precisions[:-1] * recalls[:-1]
        / np.clip(precisions[:-1] + recalls[:-1], 1e-12, None)
    )
    best_idx = int(np.nanargmax(f1s))

    return {
        "best_f1_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1s[best_idx]),
        "best_precision": float(precisions[:-1][best_idx]),
        "best_recall": float(recalls[:-1][best_idx]),
    }


def compute_threshold_at_target_fpr(
    y_true: List[int], y_score: List[float], target_fpr: float = 0.05
) -> Optional[float]:
    """Calcule le seuil pour un FPR cible donné."""
    normal_scores = [s for s, t in zip(y_score, y_true) if t == 0]
    if not normal_scores:
        return None
    q = 1.0 - target_fpr
    return float(np.quantile(normal_scores, q))


# ------------------------------------------------------------------
# Score statistics
# ------------------------------------------------------------------

def compute_score_stats(scores: List[float]) -> Dict[str, float]:
    """Statistiques détaillées d'un vecteur de scores."""
    if not scores:
        return {}
    arr = np.array(scores, dtype=np.float32)
    return {
        "n": int(len(arr)),
        "mean": _safe_float(arr.mean()),
        "std": _safe_float(arr.std()),
        "min": _safe_float(arr.min()),
        "max": _safe_float(arr.max()),
        "median": _safe_float(np.median(arr)),
        "q25": _safe_float(np.percentile(arr, 25)),
        "q75": _safe_float(np.percentile(arr, 75)),
    }


# ------------------------------------------------------------------
# Image-level evaluation
# ------------------------------------------------------------------

def evaluate_image_metrics(
    model: MultimodalPatchCore,
    samples: List[SamplePaths],
    dm: PFEDataManager3D,
    threshold: float,
) -> Dict[str, Any]:
    """
    Évaluation image-level complète.
    Retourne F1, AUROC, AP, confusion matrix, score stats, per-sample details.
    """
    y_true: List[int] = []
    y_score: List[float] = []
    y_pred: List[int] = []
    per_sample: List[Dict] = []
    errors = 0

    start = time.time()

    for i, s in enumerate(samples, 1):
        try:
            rgb = dm.load_image(s.rgb_ref, strict=True)
            depth = dm.load_depth_map(s.depth_ref, strict=True)
            if depth is not None and depth.ndim == 3:
                depth = depth[..., 2]

            pred = model.predict(rgb, depth, upsample_to_input=True)
            score = float(pred["fused_score"])
            rgb_score = float(pred["rgb_score"])
            depth_score = float(pred["depth_score"])
        except Exception as e:
            errors += 1
            per_sample.append({
                "index": i - 1,
                "category": s.category,
                "label": s.label,
                "rgb_ref": s.rgb_ref,
                "error": str(e),
            })
            continue

        if s.label is None:
            continue

        gt = 0 if str(s.label).lower() in NORMAL_LABELS else 1
        yp = int(score >= threshold)

        y_true.append(gt)
        y_score.append(score)
        y_pred.append(yp)

        per_sample.append({
            "index": i - 1,
            "category": s.category,
            "split": s.split,
            "label": s.label,
            "rgb_ref": s.rgb_ref,
            "depth_ref": s.depth_ref,
            "fused_score": score,
            "rgb_score": rgb_score,
            "depth_score": depth_score,
            "pred": yp,
            "gt": gt,
            "correct": int(yp == gt),
        })

        if i % 50 == 0:
            print(f"  [eval image] {i}/{len(samples)} samples...")

    duration = time.time() - start

    if not y_true:
        return {"n": 0, "errors": errors, "duration_s": duration, "per_sample": per_sample}

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    # Score stats par classe
    normal_scores = [s for s, t in zip(y_score, y_true) if t == 0]
    anomaly_scores = [s for s, t in zip(y_score, y_true) if t == 1]

    out: Dict[str, Any] = {
        "n": len(y_true),
        "n_normal": int(sum(1 for t in y_true if t == 0)),
        "n_anomaly": int(sum(1 for t in y_true if t == 1)),
        "errors": errors,
        "threshold": float(threshold),
        "duration_s": round(duration, 2),
        # Classification metrics
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "precision": _safe_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _safe_float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": _safe_float(fpr),
        "fnr": _safe_float(fnr),
        # Confusion matrix
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        # Score distributions
        "score_stats_normal": compute_score_stats(normal_scores),
        "score_stats_anomaly": compute_score_stats(anomaly_scores),
        "score_stats_all": compute_score_stats(y_score),
    }

    # Ranking metrics (need both classes)
    if len(set(y_true)) > 1:
        out["auroc"] = _safe_float(roc_auc_score(y_true, y_score))
        out["ap"] = _safe_float(average_precision_score(y_true, y_score))
        out.update(compute_best_f1_threshold(y_true, y_score))

        # Threshold at target FPR
        for target_fpr in [0.01, 0.05, 0.10]:
            thr_fpr = compute_threshold_at_target_fpr(y_true, y_score, target_fpr)
            if thr_fpr is not None:
                out[f"threshold_at_fpr_{int(target_fpr*100):02d}"] = _safe_float(thr_fpr)

    out["per_sample"] = per_sample
    return out


# ------------------------------------------------------------------
# Pixel-level evaluation
# ------------------------------------------------------------------

def evaluate_pixel_metrics(
    model: MultimodalPatchCore,
    samples: List[SamplePaths],
    dm: PFEDataManager3D,
    pixel_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Évaluation pixel-level avec les masques ground truth.
    Calcule pixel AUROC, pixel AP, pixel F1.
    """
    all_gt: List[np.ndarray] = []
    all_score: List[np.ndarray] = []
    f1_values: List[float] = []
    used = 0
    skipped = 0

    start = time.time()

    for i, s in enumerate(samples, 1):
        try:
            rgb = dm.load_image(s.rgb_ref, strict=True)
            depth = dm.load_depth_map(s.depth_ref, strict=True)
            if depth is not None and depth.ndim == 3:
                depth = depth[..., 2]

            pred = model.predict(rgb, depth, upsample_to_input=True)
            fmap = np.asarray(pred["fused_map"], dtype=np.float32)

            # Charger le masque ground truth
            mask = _load_mask(dm, s.mask_ref, fallback_shape=fmap.shape)
            if mask is None or mask.shape != fmap.shape:
                skipped += 1
                continue

            all_gt.append(mask.reshape(-1).astype(np.uint8))
            all_score.append(fmap.reshape(-1).astype(np.float32))
            used += 1

            if pixel_threshold is not None:
                y_true_px = mask.reshape(-1).astype(np.uint8)
                y_pred_px = (fmap.reshape(-1) >= pixel_threshold).astype(np.uint8)
                if len(np.unique(y_true_px)) > 1:
                    f1_values.append(float(f1_score(y_true_px, y_pred_px, zero_division=0)))

        except Exception:
            skipped += 1
            continue

        if i % 50 == 0:
            print(f"  [eval pixel] {i}/{len(samples)} samples...")

    duration = time.time() - start

    if not all_gt:
        return {"n_images": 0, "skipped": skipped, "duration_s": round(duration, 2)}

    y_true_all = np.concatenate(all_gt, axis=0)
    y_score_all = np.concatenate(all_score, axis=0)

    out: Dict[str, Any] = {
        "n_images": used,
        "skipped": skipped,
        "n_pixels": int(len(y_true_all)),
        "duration_s": round(duration, 2),
    }

    if len(np.unique(y_true_all)) > 1:
        out["pixel_auroc"] = _safe_float(roc_auc_score(y_true_all, y_score_all))
        out["pixel_ap"] = _safe_float(average_precision_score(y_true_all, y_score_all))

    if pixel_threshold is not None:
        out["pixel_threshold"] = float(pixel_threshold)
        if f1_values:
            out["pixel_f1_mean"] = _safe_float(np.mean(f1_values))
            out["pixel_f1_std"] = _safe_float(np.std(f1_values))

    return out


def _load_mask(
    dm: PFEDataManager3D,
    mask_ref: Optional[str],
    fallback_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Charge un masque ground truth. Retourne None si absent."""
    if not mask_ref or str(mask_ref).strip().lower() in {"", "nan", "none", "null"}:
        return np.zeros(fallback_shape, dtype=np.uint8)

    try:
        from PIL import Image
        img = dm.load_image(mask_ref, strict=False)
        arr = np.array(img.convert("L"))
        if arr.shape != fallback_shape:
            arr = np.array(
                Image.fromarray(arr).resize(
                    (fallback_shape[1], fallback_shape[0]),
                    resample=Image.NEAREST,
                )
            )
        return (arr > 0).astype(np.uint8)
    except Exception:
        return np.zeros(fallback_shape, dtype=np.uint8)


# ------------------------------------------------------------------
# Per-category evaluation
# ------------------------------------------------------------------

def evaluate_by_category(
    model: MultimodalPatchCore,
    samples: List[SamplePaths],
    dm: PFEDataManager3D,
    threshold: float,
) -> Dict[str, Dict]:
    """Ventile les métriques image-level par catégorie produit."""
    by_cat: Dict[str, List[SamplePaths]] = {}
    for s in samples:
        cat = s.category or "unknown"
        by_cat.setdefault(cat, []).append(s)

    summary: Dict[str, Dict] = {}
    for cat in sorted(by_cat):
        cat_samples = by_cat[cat]
        metrics = evaluate_image_metrics(model, cat_samples, dm, threshold)
        summary[cat] = {
            "n": metrics.get("n", 0),
            "n_normal": metrics.get("n_normal", 0),
            "n_anomaly": metrics.get("n_anomaly", 0),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "auroc": metrics.get("auroc"),
            "ap": metrics.get("ap"),
            "fpr": metrics.get("fpr"),
            "mean_score_normal": metrics.get("score_stats_normal", {}).get("mean"),
            "mean_score_anomaly": metrics.get("score_stats_anomaly", {}).get("mean"),
        }
        print(
            f"  [{cat}] n={summary[cat]['n']} "
            f"F1={summary[cat]['f1']} "
            f"AUROC={summary[cat]['auroc']} "
            f"AP={summary[cat]['ap']}"
        )

    return summary


# ------------------------------------------------------------------
# Full evaluation pipeline
# ------------------------------------------------------------------

def run_full_evaluation(
    model: MultimodalPatchCore,
    samples: List[SamplePaths],
    dm: PFEDataManager3D,
    threshold_key: str = "image_mean_plus_3std",
    pixel_threshold_key: str = "pixel_mean_plus_3std",
) -> Dict[str, Any]:
    """
    Lance l'évaluation complète (image + pixel + par catégorie).
    Retourne un dict avec toutes les métriques.
    """
    # Résoudre les seuils
    if threshold_key not in model.thresholds:
        available = list(model.thresholds.keys())
        raise KeyError(
            f"Seuil '{threshold_key}' introuvable. Disponibles : {available}"
        )
    image_threshold = float(model.thresholds[threshold_key])

    pixel_threshold = None
    if pixel_threshold_key in model.thresholds:
        pixel_threshold = float(model.thresholds[pixel_threshold_key])

    print(f"\n=== IMAGE-LEVEL EVALUATION (threshold={image_threshold:.6f}) ===")
    image_metrics = evaluate_image_metrics(model, samples, dm, image_threshold)

    # Affichage
    for k in ["n", "accuracy", "precision", "recall", "f1", "auroc", "ap",
              "fpr", "fnr", "best_f1", "best_f1_threshold"]:
        if k in image_metrics:
            print(f"  {k}: {image_metrics[k]}")

    print(f"\n=== PIXEL-LEVEL EVALUATION ===")
    pixel_metrics = evaluate_pixel_metrics(model, samples, dm, pixel_threshold)
    for k, v in pixel_metrics.items():
        if k not in ("per_sample",):
            print(f"  {k}: {v}")

    print(f"\n=== PER-CATEGORY EVALUATION ===")
    category_metrics = evaluate_by_category(model, samples, dm, image_threshold)

    return {
        "image_metrics": image_metrics,
        "pixel_metrics": pixel_metrics,
        "category_metrics": category_metrics,
        "model_meta": model.meta,
        "thresholds": model.thresholds,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation complète Multimodal PatchCore"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--table-name", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--category", default=None)
    parser.add_argument("--threshold-key", default="image_mean_plus_3std")
    parser.add_argument("--pixel-threshold-key", default="pixel_mean_plus_3std")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    settings = Settings.from_yaml(args.config)
    dm = PFEDataManager3D(settings=settings)
    model = MultimodalPatchCore.load(args.model_dir)

    df = dm.get_dataset(table=args.table_name, verbose=True)
    samples = build_samples_from_dataframe(
        df, split=args.split, normal_only=False, category=args.category,
    )
    print(f"Eval samples: {len(samples)}")

    results = run_full_evaluation(
        model, samples, dm,
        threshold_key=args.threshold_key,
        pixel_threshold_key=args.pixel_threshold_key,
    )

    results["table_name"] = args.table_name
    results["split"] = args.split
    results["category"] = args.category
    results["model_dir"] = str(Path(args.model_dir).resolve())

    # Save
    output_json = args.output_json or str(
        Path(args.model_dir) / f"eval_{args.split}.json"
    )
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove per_sample from saved JSON to keep it small
    save_data = dict(results)
    if "image_metrics" in save_data:
        im = dict(save_data["image_metrics"])
        im.pop("per_sample", None)
        save_data["image_metrics"] = im

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Évaluation sauvegardée : {out_path}")


if __name__ == "__main__":
    main()
