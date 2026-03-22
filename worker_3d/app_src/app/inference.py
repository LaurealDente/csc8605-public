# training_3d/src/inference.py
"""
Inference pour le pipeline 3D.

V1 simple : même approche ResNet18 + k-NN que la 2D.
Les images RGB du dataset MVTec 3D-AD sont utilisées directement.
Les depth maps seront intégrées dans une version future.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

from .config import Settings
from .data import MVTec3DDataset, PFEDataManager3D

import mlflow
from mlflow.tracking import MlflowClient


# ---------------------------
# Device + preprocess
# ---------------------------

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ---------------------------
# Helpers
# ---------------------------

def _resolve_model_key(model_dir: Optional[str] = None) -> str:
    if model_dir is None:
        return "__default__"
    return str(Path(model_dir).resolve())


def _resolve_embedder_cache_key(
    model_dir: Optional[str] = None,
    backbone_name: str = "resnet18",
) -> str:
    return f"{_resolve_model_key(model_dir)}::{str(backbone_name).lower()}"


def _resolve_bank_cache_key(model_dir: str, backbone_name: str = "resnet18") -> str:
    return f"{_resolve_model_key(model_dir)}::{str(backbone_name).lower()}"


def _resolve_knn_cache_key(model_dir: str, backbone_name: str, k: int) -> Tuple[str, str, int]:
    return (_resolve_model_key(model_dir), str(backbone_name).lower(), int(k))


def _l2_normalize_numpy(x: np.ndarray, axis: int = 1) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12
    return x / norms


# ---------------------------
# Dataframe filtering helpers
# ---------------------------

def _parse_normal_values(raw: str) -> set[str]:
    return {v.strip().lower() for v in str(raw).split(",") if v.strip()}


def _find_label_column(df: pd.DataFrame, explicit_label_col: str | None = None) -> str | None:
    if explicit_label_col is not None:
        if explicit_label_col not in df.columns:
            raise ValueError(f"Label column '{explicit_label_col}' not found.")
        return explicit_label_col
    candidates = ["label", "target", "y", "is_anomaly", "anomaly", "status", "class", "ground_truth"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _filter_fit_dataframe(
    df: pd.DataFrame,
    fit_split: str = "train",
    normal_only: bool = False,
    label_col: str | None = None,
    normal_values: str = "0,normal,good,false",
) -> pd.DataFrame:
    out = df.copy()

    if "split" in out.columns and fit_split:
        before = len(out)
        out = out[out["split"].astype(str).str.lower() == str(fit_split).lower()].copy()
        print(f"ℹ️ Fit split filter: split='{fit_split}' -> {len(out)}/{before} rows kept")

    if normal_only:
        detected = _find_label_column(out, explicit_label_col=label_col)
        if detected is None:
            raise ValueError("normal-only filtering requested but no label column found.")
        allowed = _parse_normal_values(normal_values)
        before = len(out)
        out = out[out[detected].astype(str).str.strip().str.lower().isin(allowed)].copy()
        print(f"ℹ️ Normal-only filter: {len(out)}/{before} rows kept")

    if len(out) == 0:
        raise ValueError("No samples left after filtering.")
    return out


def _print_fit_dataframe_summary(df: pd.DataFrame, explicit_label_col: str | None = None) -> None:
    print("\n=== FIT DATA SUMMARY ===")
    print(f"Rows used: {len(df)}")
    if "split" in df.columns:
        print("Split counts:", df["split"].astype(str).value_counts(dropna=False).to_dict())
    detected = _find_label_column(df, explicit_label_col=explicit_label_col)
    if detected is not None:
        print(f"Label column: {detected}")
        print("Label counts:", df[detected].astype(str).value_counts(dropna=False).to_dict())
    print("========================\n")


def _find_image_column(df: pd.DataFrame) -> str:
    candidates = ["image_url", "image_path", "path", "url", "filepath", "file_path", "filename"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find image column. Columns: {list(df.columns)}")


class FilteredImageDataset(Dataset):
    def __init__(self, dm: PFEDataManager3D, df: pd.DataFrame, transform=None) -> None:
        self.dm = dm
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.image_col = _find_image_column(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self.dm.load_image(row[self.image_col])
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def build_filtered_dataset(
    dm: PFEDataManager3D,
    table_name: str,
    fit_split: str = "train",
    normal_only: bool = False,
    label_col: str | None = None,
    normal_values: str = "0,normal,good,false",
    transform=None,
) -> Dataset:
    df = dm.get_dataset(table_name)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = _filter_fit_dataframe(
        df, fit_split=fit_split, normal_only=normal_only,
        label_col=label_col, normal_values=normal_values,
    )
    _print_fit_dataframe_summary(df, explicit_label_col=label_col)
    return FilteredImageDataset(dm=dm, df=df, transform=transform)


# ---------------------------
# Backbone / embedder cache
# ---------------------------

def _build_backbone(
    model_dir: Optional[str] = None,
    backbone_name: str = "resnet18",
) -> nn.Module:
    backbone_name = str(backbone_name).lower()

    if backbone_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
    elif backbone_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone_name={backbone_name}")

    if model_dir:
        ft_path = Path(model_dir) / "backbone_finetuned.pt"
        if ft_path.exists():
            state = torch.load(str(ft_path), map_location="cpu")
            model.load_state_dict(state, strict=False)
            print(f"✅ Loaded finetuned backbone: {ft_path}")

    return model.eval().to(_DEVICE)


_EMBEDDER_CACHE: Dict[str, nn.Module] = {}


def clear_embedder_cache(model_dir=None, backbone_name=None):
    if model_dir is None and backbone_name is None:
        _EMBEDDER_CACHE.clear()
        return
    keys_to_delete = [k for k in _EMBEDDER_CACHE
                      if (model_dir is None or _resolve_model_key(model_dir) in k)
                      and (backbone_name is None or str(backbone_name).lower() in k)]
    for k in keys_to_delete:
        _EMBEDDER_CACHE.pop(k, None)


def get_embedder(model_dir=None, backbone_name="resnet18") -> nn.Module:
    key = _resolve_embedder_cache_key(model_dir=model_dir, backbone_name=backbone_name)
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = _build_backbone(model_dir=model_dir, backbone_name=backbone_name)
    return _EMBEDDER_CACHE[key]


# ---------------------------
# Embedding extraction
# ---------------------------

def _pil_to_tensor(pil_img) -> torch.Tensor:
    return _PREPROCESS(pil_img).unsqueeze(0)


def image_to_embedding(pil_img, model_dir=None, backbone_name="resnet18") -> np.ndarray:
    x = _pil_to_tensor(pil_img).to(_DEVICE)
    embedder = get_embedder(model_dir=model_dir, backbone_name=backbone_name)
    with torch.no_grad():
        z = embedder(x).squeeze(0)
        z = z / (torch.norm(z) + 1e-12)
    return z.detach().cpu().numpy().astype(np.float32)


# ---------------------------
# Reference bank cache
# ---------------------------

_REFERENCE_BANK_CACHE: Dict[str, np.ndarray] = {}


def clear_reference_bank_cache(model_dir=None, backbone_name=None):
    if model_dir is None and backbone_name is None:
        _REFERENCE_BANK_CACHE.clear()
        return
    keys_to_delete = [k for k in _REFERENCE_BANK_CACHE
                      if (model_dir is None or _resolve_model_key(model_dir) in k)
                      and (backbone_name is None or str(backbone_name).lower() in k)]
    for k in keys_to_delete:
        _REFERENCE_BANK_CACHE.pop(k, None)


def load_reference_bank(model_dir: str, use_cache=True, backbone_name="resnet18") -> np.ndarray:
    cache_key = _resolve_bank_cache_key(model_dir=model_dir, backbone_name=backbone_name)
    if use_cache and cache_key in _REFERENCE_BANK_CACHE:
        return _REFERENCE_BANK_CACHE[cache_key]

    emb_path = Path(model_dir) / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing reference bank: {emb_path}")

    bank = np.load(str(emb_path)).astype(np.float32)
    if bank.ndim != 2:
        raise ValueError(f"Invalid bank shape {bank.shape}")

    bank = _l2_normalize_numpy(bank, axis=1)
    if use_cache:
        _REFERENCE_BANK_CACHE[cache_key] = bank
    return bank


# ---------------------------
# k-NN cache
# ---------------------------

@dataclass
class CachedKNN:
    bank: np.ndarray
    nn: NearestNeighbors
    k: int
    model_dir: str
    backbone_name: str


_KNN_CACHE: Dict[Tuple[str, str, int], CachedKNN] = {}


def clear_knn_cache(model_dir=None, backbone_name=None, k=None):
    if model_dir is None and backbone_name is None and k is None:
        _KNN_CACHE.clear()
        return
    keys_to_delete = []
    for key in _KNN_CACHE:
        m, b, kk = key
        if ((model_dir is None or _resolve_model_key(model_dir) == m) and
            (backbone_name is None or str(backbone_name).lower() == b) and
                (k is None or int(k) == kk)):
            keys_to_delete.append(key)
    for k_ in keys_to_delete:
        _KNN_CACHE.pop(k_, None)


def get_cached_knn(model_dir: str, k: int, backbone_name="resnet18") -> CachedKNN:
    model_key = _resolve_model_key(model_dir)
    backbone_key = str(backbone_name).lower()
    k_eff = max(1, int(k))
    cache_key = (model_key, backbone_key, k_eff)

    cached = _KNN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    bank = load_reference_bank(model_dir=model_key, use_cache=True, backbone_name=backbone_key)
    if len(bank) == 0:
        raise ValueError("Reference bank is empty.")

    if k_eff > len(bank):
        k_eff = len(bank)

    nn_model = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn_model.fit(bank)

    cached = CachedKNN(bank=bank, nn=nn_model, k=k_eff, model_dir=model_key, backbone_name=backbone_key)
    _KNN_CACHE[cache_key] = cached
    return cached


def clear_all_model_caches(model_dir=None, backbone_name=None):
    clear_embedder_cache(model_dir=model_dir, backbone_name=backbone_name)
    clear_reference_bank_cache(model_dir=model_dir, backbone_name=backbone_name)
    clear_knn_cache(model_dir=model_dir, backbone_name=backbone_name)


# ---------------------------
# Fit reference bank
# ---------------------------

def fit_reference_bank(
    model_dir: str,
    config_path: str = "conf/config.yaml",
    table_name: str = "mvtec_3d_anomaly_detection",
    batch_size: int = 64,
    num_workers: int = 4,
    backbone_name: str = "resnet18",
    fit_split: str = "train",
    normal_only: bool = False,
    label_col: str | None = None,
    normal_values: str = "0,normal,good,false",
) -> None:
    settings = Settings.from_yaml(config_path)
    dm = PFEDataManager3D(settings=settings)

    dataset = build_filtered_dataset(
        dm=dm, table_name=table_name, fit_split=fit_split,
        normal_only=normal_only, label_col=label_col,
        normal_values=normal_values, transform=_PREPROCESS,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No images found after filtering (table={table_name}).")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    model_dir = str(Path(model_dir).resolve())
    backbone_name = str(backbone_name).lower()
    embedder = get_embedder(model_dir=model_dir, backbone_name=backbone_name)

    embs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(_DEVICE, non_blocking=True)
            feats = embedder(imgs)
            feats = feats / (torch.norm(feats, dim=1, keepdim=True) + 1e-12)
            embs.append(feats.cpu())

    embeddings = torch.cat(embs, dim=0).numpy().astype(np.float32)

    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "embeddings.npy")
    np.save(out_path, embeddings)

    clear_reference_bank_cache(model_dir=model_dir, backbone_name=backbone_name)
    clear_knn_cache(model_dir=model_dir, backbone_name=backbone_name)

    print(f"✅ embeddings saved: {embeddings.shape} -> {out_path}")


# ---------------------------
# Distance aggregation
# ---------------------------

def aggregate_knn_distances(dists: np.ndarray, score_mode: str = "mean") -> float:
    dists = np.asarray(dists, dtype=np.float32).reshape(-1)
    if len(dists) == 0:
        raise ValueError("dists is empty")

    score_mode = str(score_mode).lower()
    if score_mode == "min":
        return float(np.min(dists))
    if score_mode == "mean":
        return float(np.mean(dists))
    if score_mode == "median":
        return float(np.median(dists))
    if score_mode == "max":
        return float(np.max(dists))
    if score_mode == "weighted_mean":
        weights = 1.0 / (np.arange(len(dists), dtype=np.float32) + 1.0)
        weights = weights / weights.sum()
        return float(np.sum(dists * weights))
    raise ValueError(f"Unknown score_mode={score_mode}")


# ---------------------------
# Predict anomaly
# ---------------------------

def predict_anomaly(
    pil_img,
    bank: Optional[np.ndarray] = None,
    k: int = 5,
    threshold: float = 0.35,
    model_dir: Optional[str] = None,
    score_mode: str = "mean",
    backbone_name: str = "resnet18",
) -> Tuple[float, str]:
    emb = image_to_embedding(pil_img, model_dir=model_dir, backbone_name=backbone_name)[None, :]

    if model_dir is not None:
        cached = get_cached_knn(model_dir=model_dir, k=k, backbone_name=backbone_name)
        dists, _ = cached.nn.kneighbors(emb, return_distance=True)
        score = aggregate_knn_distances(dists[0], score_mode=score_mode)
        label = "anomaly" if score > float(threshold) else "normal"
        return score, label

    if bank is None or len(bank) == 0:
        raise ValueError("Reference bank is empty.")

    bank = _l2_normalize_numpy(np.asarray(bank, dtype=np.float32), axis=1)
    k_eff = max(1, min(int(k), len(bank)))
    nn_model = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn_model.fit(bank)
    dists, _ = nn_model.kneighbors(emb, return_distance=True)

    score = aggregate_knn_distances(dists[0], score_mode=score_mode)
    label = "anomaly" if score > float(threshold) else "normal"
    return score, label
