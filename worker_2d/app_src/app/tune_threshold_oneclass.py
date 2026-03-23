# worker_2d/app/tune_threshold_oneclass.py

from __future__ import annotations
import numpy as np
import json
from pathlib import Path

from .config import Settings
from .data import PFEDataManager
from .inference import load_reference_bank, predict_anomaly


def tune_threshold_oneclass(
    config_path: str,
    model_dir: str,
    table_name: str = "mvtec_ad_2",
    split: str = "validation",
    target_fpr: float = 0.01,
    k: int = 5,
):
    """
    Calibrate threshold using GOOD-only validation split.
    """

    settings = Settings.from_yaml(config_path)
    dm = PFEDataManager(settings=settings)

    df = dm.get_dataset(table=table_name, verbose=True)
    df = df[df["split"].astype(str) == split]
    df = df[df["label"].astype(str) == "good"]

    if df.empty:
        raise RuntimeError("Validation split contains no good samples.")

    print(f"Validation good samples: {len(df)}")

    model_dir = str(Path(model_dir).resolve())
    bank = load_reference_bank(model_dir)

    scores = []

    for fp in df["filepath"].astype(str).tolist():
        img = dm.load_image(fp, strict=True)
        score, _ = predict_anomaly(
            img,
            bank,
            k=k,
            threshold=0.0,
            model_dir=model_dir,  # important for finetuned backbone
        )
        scores.append(score)

    scores = np.array(scores, dtype=np.float32)

    # threshold = quantile for target FPR
    q = 1.0 - target_fpr
    threshold = float(np.quantile(scores, q))

    print(f"Chosen threshold (FPR={target_fpr*100:.2f}%): {threshold:.6f}")

    result = {
        "method": "one_class_quantile",
        "split": split,
        "target_fpr": target_fpr,
        "threshold": threshold,
        "num_samples": int(len(scores)),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
    }

    out_path = Path(model_dir) / "threshold.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved threshold to: {out_path}")

    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="conf/config.yaml")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--table-name", default="mvtec_ad_2")
    p.add_argument("--split", default="validation")
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    tune_threshold_oneclass(
        config_path=args.config,
        model_dir=args.model_dir,
        table_name=args.table_name,
        split=args.split,
        target_fpr=args.target_fpr,
        k=args.k,
    )