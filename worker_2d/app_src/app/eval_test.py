# worker_2d/app/eval_test.py

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

from .config import Settings
from .data import PFEDataManager
from .inference import load_reference_bank, predict_anomaly


def eval_on_test(
    config_path: str,
    model_dir: str,
    table_name: str,
    split: str = "test",
    k: int = 5,
    limit: int | None = None,
):
    settings = Settings.from_yaml(config_path)
    dm = PFEDataManager(settings=settings)

    df = dm.get_dataset(table=table_name, verbose=True)
    df = df[df["split"].astype(str) == split].copy()

    if limit is not None:
        df = df.head(int(limit))

    if df.empty:
        raise RuntimeError(f"No rows found for split={split} in table={table_name}")

    model_dir = str(Path(model_dir).resolve())

    # load bank
    bank = load_reference_bank(model_dir)

    # load threshold from threshold.json
    threshold_path = Path(model_dir) / "threshold.json"
    if threshold_path.exists():
        threshold = float(json.loads(threshold_path.read_text())["threshold"])
    else:
        threshold = float(settings.threshold)

    print(f"Using threshold = {threshold:.6f}")

    # y_true: 0 good, 1 anomaly
    y_true = (df["label"].astype(str) != "good").astype(int).to_numpy()

    scores = []
    y_pred = []

    for fp in df["filepath"].astype(str).tolist():
        img = dm.load_image(fp, strict=True)
        score, pred_label = predict_anomaly(
            img,
            bank,
            k=k,
            threshold=threshold,
            model_dir=model_dir,   # IMPORTANT: use finetuned backbone if present
        )
        scores.append(score)
        y_pred.append(1 if pred_label == "anomaly" else 0)

    scores = np.array(scores, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.int32)

    # metrics
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4, zero_division=0)

    # ROC-AUC and AP need both classes present
    auroc = None
    ap = None
    if len(np.unique(y_true)) == 2:
        auroc = float(roc_auc_score(y_true, scores))
        ap = float(average_precision_score(y_true, scores))

    print("\nConfusion matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    print("\nClassification report:")
    print(rep)
    if auroc is not None:
        print(f"\nAUROC: {auroc:.4f}")
        print(f"Average Precision (AP): {ap:.4f}")

    # save artifacts
    out = {
        "table": table_name,
        "split": split,
        "k": k,
        "threshold": threshold,
        "n": int(len(y_true)),
        "n_anomaly": int(y_true.sum()),
        "confusion_matrix": cm.tolist(),
        "auroc": auroc,
        "average_precision": ap,
        "report": rep,
    }

    out_path = Path(model_dir) / f"eval_{table_name}_{split}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n✅ Saved eval report: {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="conf/config.yaml")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--table-name", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    eval_on_test(
        config_path=args.config,
        model_dir=args.model_dir,
        table_name=args.table_name,
        split=args.split,
        k=args.k,
        limit=args.limit,
    )