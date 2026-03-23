from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "resnet_knn_2d")
MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/mlflow_model_cache")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "worker_2d_fit_v2")


def _get_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def start_fit_run(
    backbone_name: str,
    batch_size: int,
    num_workers: int,
    table_name: str,
    model_dir: str,
    extra_params: Optional[Dict[str, Any]] = None,
) -> mlflow.ActiveRun:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    run_name = f"fit_{backbone_name}_{int(time.time())}"

    params: Dict[str, Any] = {
        "backbone": backbone_name,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "table_name": table_name,
        "model_dir": model_dir,
    }
    if extra_params:
        params.update(extra_params)

    active_run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(params)
    return active_run


def log_fit_metrics(model_dir: str, duration_seconds: float) -> None:
    metrics: Dict[str, float] = {"fit_duration_seconds": duration_seconds}

    # 1. NOUVELLE MÉTHODE (Patch-based)
    meta_path = Path(model_dir) / "patch_bank_meta.json"
    if meta_path.exists():
        import json
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        # On loggue le nombre de patchs finaux et la dimension
        if "n_patches_after_reduction" in meta:
            metrics["bank_size"] = float(meta["n_patches_after_reduction"])
        if "embedding_dim" in meta:
            metrics["embedding_dim"] = float(meta["embedding_dim"])

    # 2. ANCIENNE MÉTHODE (Global-based - Fallback)
    elif (Path(model_dir) / "embeddings.npy").exists():
        bank = np.load(str(Path(model_dir) / "embeddings.npy"))
        metrics["bank_size"] = float(bank.shape[0])
        if bank.ndim > 1:
            metrics["embedding_dim"] = float(bank.shape[1])

    # 3. Seuil (Commun aux deux méthodes)
    threshold_path = Path(model_dir) / "threshold.json"
    if threshold_path.exists():
        import json
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
    run_id: str,
    model_name: str = MLFLOW_MODEL_NAME,
    artifact_subpath: str = "model_artifacts",
) -> str:
    client = _get_client()

    # Créer le modèle dans le Registry s'il n'existe pas
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass  # existe déjà

    # Créer une version pointant vers les artefacts du run
    source = f"runs:/{run_id}/{artifact_subpath}"
    version = client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
    )
    print(f"[MLflow Registry] Modèle enregistré : {model_name} version {version.version}")
    print(f"[MLflow Registry] Promouvoir la version {version.version} en Production via l'UI MLflow.")
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
        raise RuntimeError(
            f"[MLflow] Aucune version de '{model_name}' avec l'alias 'production'."
        )

    version_str = str(latest.version)
    run_id = latest.run_id
    cache_path = Path(local_cache_dir) / f"{model_name}_v{version_str}"
    
    # Vérification de présence du modèle (Patch ou Global)
    has_patch = (cache_path / "patch_bank.npy").exists()
    has_global = (cache_path / "embeddings.npy").exists()
    model_cached = has_patch or has_global

    if model_cached and not force_download:
        print(f"[MLflow] Modèle en cache : {cache_path} (version {version_str})")
        return str(cache_path.resolve())

    if cache_path.exists():
        shutil.rmtree(str(cache_path))
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"[MLflow] Téléchargement {model_name} version {version_str}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model_artifacts",
        dst_path=str(cache_path),
    )

    # Vérification post-téléchargement
    if not ((cache_path / "patch_bank.npy").exists() or (cache_path / "embeddings.npy").exists()):
        raise RuntimeError(f"[MLflow] patch_bank.npy ou embeddings.npy absent après téléchargement.")

    print(f"[MLflow] Modèle prêt : {cache_path}")
    return str(cache_path.resolve())


def get_current_production_version(
    model_name: str = MLFLOW_MODEL_NAME,
) -> Optional[str]:
    try:
        client = _get_client()
        version = client.get_model_version_by_alias(model_name, "production")
        return str(version.version)
    except Exception as exc:
        print(f"[MLflow] Impossible de récupérer la version production : {exc}")
        return None


def log_eval_metrics(
    model_dir: str,
    config_path: str,
    table_name: str,
    backbone_name: str = "resnet18",
    feature_layer: str = "layer3",
    num_workers: int = 0,
    batch_size: int = 8,
) -> dict:
    """
    Evalue le modele sur le split test et logue les metriques dans MLflow.

    Metriques loguees :
      Image-level  : eval_auroc, eval_ap
      Optimal      : optimal_threshold, optimal_f1, optimal_precision,
                     optimal_recall, optimal_accuracy, optimal_specificity,
                     optimal_tp/fp/tn/fn
      Fixed 0.5    : fixed05_accuracy/precision/recall/f1
      Scores       : score_normal_mean/std/median/max,
                     score_anomaly_mean/std/median/min,
                     score_gap, score_separability (Cohen d)
      Per-category : cat_auroc_<name>, cat_f1_<name>,
                     cat_auroc_mean/std/min, cat_f1_mean
      Dataset      : data_n_total/train/test/categories/test_normal/test_anomaly
      Meta         : eval_errors, eval_duration_seconds
    """
    import json
    import time
    from pathlib import Path

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
    from .data import PFEDataManager
    from .patch_inference import predict_patch_anomaly

    print("[Eval] Chargement du dataset...")
    settings = Settings.from_yaml(config_path)
    dm = PFEDataManager(settings=settings)
    df = dm.get_dataset(table_name, verbose=True)

    categories = sorted(df["category"].unique().tolist()) if "category" in df.columns else []
    df_train = df[df["split"] == "train"] if "split" in df.columns else df
    df_test = df[df["split"] == "test"] if "split" in df.columns else df

    if len(df_test) == 0:
        print("[Eval] Aucune image test trouvee.")
        return {}

    normal_values = {"0", "normal", "good", "false"}
    y_true, y_scores, cat_list = [], [], []
    errors = 0
    eval_start = time.time()

    print(f"[Eval] {len(df_test)} images test, {len(categories)} categories")

    for i, (_, row) in enumerate(df_test.iterrows()):
        gt = 0 if str(row["label"]).lower().strip() in normal_values else 1
        y_true.append(gt)
        cat_list.append(str(row.get("category", "unknown")))

        try:
            img = dm.load_image(str(row["filepath"]), strict=False)
            pred = predict_patch_anomaly(
                pil_img=img,
                model_dir=model_dir,
                backbone_name=backbone_name,
                feature_layer=feature_layer,
                patch_neighbors=1,
                threshold=0.5,
            )
            y_scores.append(float(pred["image_score"]))
        except Exception:
            errors += 1
            y_scores.append(0.0)

        if (i + 1) % 100 == 0:
            print(f"[Eval] {i+1}/{len(df_test)} images...")

    eval_duration = time.time() - eval_start
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    cat_array = np.array(cat_list)
    metrics = {}

    # --- Dataset stats ---
    metrics["data_n_total"] = float(len(df))
    metrics["data_n_train"] = float(len(df_train))
    metrics["data_n_test"] = float(len(df_test))
    metrics["data_n_categories"] = float(len(categories))
    metrics["data_n_test_normal"] = float(np.sum(y_true == 0))
    metrics["data_n_test_anomaly"] = float(np.sum(y_true == 1))
    metrics["eval_errors"] = float(errors)
    metrics["eval_duration_seconds"] = eval_duration

    # --- Score distributions ---
    normal_scores = y_scores[y_true == 0]
    anomaly_scores = y_scores[y_true == 1]

    if len(normal_scores) > 0:
        metrics["score_normal_mean"] = float(np.mean(normal_scores))
        metrics["score_normal_std"] = float(np.std(normal_scores))
        metrics["score_normal_median"] = float(np.median(normal_scores))
        metrics["score_normal_max"] = float(np.max(normal_scores))

    if len(anomaly_scores) > 0:
        metrics["score_anomaly_mean"] = float(np.mean(anomaly_scores))
        metrics["score_anomaly_std"] = float(np.std(anomaly_scores))
        metrics["score_anomaly_median"] = float(np.median(anomaly_scores))
        metrics["score_anomaly_min"] = float(np.min(anomaly_scores))

    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        gap = float(np.mean(anomaly_scores) - np.mean(normal_scores))
        pooled_std = float(np.sqrt((np.std(normal_scores)**2 + np.std(anomaly_scores)**2) / 2))
        metrics["score_gap"] = gap
        if pooled_std > 0:
            metrics["score_separability"] = gap / pooled_std

    # --- AUROC et AP (threshold-independent) ---
    if len(set(y_true)) == 2:
        metrics["eval_auroc"] = float(roc_auc_score(y_true, y_scores))
        metrics["eval_ap"] = float(average_precision_score(y_true, y_scores))

    # --- Optimal threshold (maximize F1) ---
    if len(set(y_true)) == 2:
        precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_scores)
        f1s = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0,
        )
        best_idx = np.argmax(f1s)
        optimal_threshold = float(thresholds_pr[min(best_idx, len(thresholds_pr) - 1)])
        metrics["optimal_threshold"] = optimal_threshold
        metrics["optimal_f1"] = float(f1s[best_idx])
        metrics["optimal_precision"] = float(precisions[best_idx])
        metrics["optimal_recall"] = float(recalls[best_idx])

        y_pred_opt = (y_scores >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_opt, labels=[0, 1]).ravel()
        metrics["optimal_tp"] = float(tp)
        metrics["optimal_fp"] = float(fp)
        metrics["optimal_tn"] = float(tn)
        metrics["optimal_fn"] = float(fn)
        metrics["optimal_accuracy"] = float(accuracy_score(y_true, y_pred_opt))
        metrics["optimal_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # --- Fixed threshold 0.5 ---
    y_pred_fixed = (y_scores >= 0.5).astype(int)
    metrics["fixed05_accuracy"] = float(accuracy_score(y_true, y_pred_fixed))
    metrics["fixed05_precision"] = float(precision_score(y_true, y_pred_fixed, zero_division=0))
    metrics["fixed05_recall"] = float(recall_score(y_true, y_pred_fixed, zero_division=0))
    metrics["fixed05_f1"] = float(f1_score(y_true, y_pred_fixed, zero_division=0))

    # --- Per-category ---
    per_cat_thresholds = {}
    if len(categories) > 1 and len(set(y_true)) == 2:
        cat_aurocs, cat_f1s = [], []
        for cat in categories:
            mask = cat_array == cat
            yt_cat, ys_cat = y_true[mask], y_scores[mask]
            if len(set(yt_cat)) < 2 or len(yt_cat) < 2:
                continue
            cat_auroc = float(roc_auc_score(yt_cat, ys_cat))
            cat_aurocs.append(cat_auroc)
            metrics[f"cat_auroc_{cat}"] = cat_auroc

            # --- Per-category threshold (normal-only: mean + 2*std) ---
            normal_scores_cat = ys_cat[yt_cat == 0]
            if len(normal_scores_cat) > 0:
                cat_th = float(np.mean(normal_scores_cat) + 2 * np.std(normal_scores_cat))
            else:
                cat_th = 0.5
            per_cat_thresholds[cat] = round(cat_th, 6)
            metrics[f"cat_threshold_{cat}"] = cat_th

            yp_cat = (ys_cat >= cat_th).astype(int)
            cat_f1 = float(f1_score(yt_cat, yp_cat, zero_division=0))
            cat_f1s.append(cat_f1)
            metrics[f"cat_f1_{cat}"] = cat_f1

        if cat_aurocs:
            metrics["cat_auroc_mean"] = float(np.mean(cat_aurocs))
            metrics["cat_auroc_std"] = float(np.std(cat_aurocs))
            metrics["cat_auroc_min"] = float(np.min(cat_aurocs))
        if cat_f1s:
            metrics["cat_f1_mean"] = float(np.mean(cat_f1s))

    # --- Save per-category thresholds ---
    if per_cat_thresholds:
        th_cat_path = Path(model_dir) / "thresholds_per_category.json"
        with th_cat_path.open("w", encoding="utf-8") as f:
            json.dump(per_cat_thresholds, f, indent=2)
        print(f"[Eval] thresholds_per_category.json sauvegardé ({len(per_cat_thresholds)} catégories)")

    # --- Save global threshold (normal-only: mean + 2*std) ---
    if len(normal_scores) > 0:
        global_threshold = float(np.mean(normal_scores) + 2 * np.std(normal_scores))
    else:
        global_threshold = 0.5
    th_path = Path(model_dir) / "threshold.json"
    with th_path.open("w", encoding="utf-8") as f:
        json.dump({"threshold": round(global_threshold, 6)}, f, indent=2)
    print(f"[Eval] threshold.json sauvegardé: {global_threshold:.6f}")

    # --- Log to MLflow ---
    float_metrics = {k: float(v) for k, v in metrics.items() if not isinstance(v, str)}
    str_params = {k: v for k, v in metrics.items() if isinstance(v, str)}
    mlflow.log_metrics(float_metrics)
    if str_params:
        mlflow.log_params(str_params)

    eval_path = Path(model_dir) / "eval_results.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump({k: v if isinstance(v, str) else round(v, 6) for k, v in metrics.items()}, f, indent=2)
    mlflow.log_artifact(str(eval_path))

    print("[Eval] Metriques loguees :")
    for k, v in sorted(float_metrics.items()):
        print(f"  {k}: {v:.4f}")

    return metrics
