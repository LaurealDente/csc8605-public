# worker_2d/app/inference.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from PIL import Image

from .config import Settings
from .data import PFEDataManager, MVTecDataset


# ---------------------------
# Device + preprocess
# ---------------------------

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------
# Embedder (supports finetuned backbone)
# ---------------------------

def _build_backbone(model_dir: Optional[str] = None) -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Identity()

    if model_dir:
        ft_path = Path(model_dir) / "backbone_finetuned.pt"
        if ft_path.exists():
            state = torch.load(str(ft_path), map_location="cpu")
            m.load_state_dict(state, strict=False)
            print(f"✅ Loaded finetuned backbone: {ft_path}")

    return m.eval().to(_DEVICE)


_EMBEDDER_CACHE: Dict[str, nn.Module] = {}


def get_embedder(model_dir: Optional[str] = None) -> nn.Module:
    key = str(Path(model_dir).resolve()) if model_dir else "__default__"
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = _build_backbone(model_dir)
    return _EMBEDDER_CACHE[key]


def _pil_to_tensor(pil_img) -> torch.Tensor:
    return _PREPROCESS(pil_img).unsqueeze(0)


# ---------------------------
# L2 normalization
# ---------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (N, D), got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


# ---------------------------
# Global embedding (mode=global, dim=512)
# ---------------------------

def image_to_embedding(pil_img, model_dir: Optional[str] = None) -> np.ndarray:
    x = _pil_to_tensor(pil_img).to(_DEVICE)
    embedder = get_embedder(model_dir)

    with torch.no_grad():
        z = embedder(x).squeeze(0)
        z = z / (torch.norm(z) + 1e-12)

    return z.detach().cpu().numpy().astype(np.float32)


# ---------------------------
# Patch-level feature extraction (mode=patch)
# ---------------------------

def _forward_to_layer(
    model: nn.Module,
    x: torch.Tensor,
    feature_layer: str = "layer3",
) -> torch.Tensor:
    feature_layer = str(feature_layer).lower()

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x1 = model.layer1(x)
    if feature_layer == "layer1":
        return x1

    x2 = model.layer2(x1)
    if feature_layer == "layer2":
        return x2

    x3 = model.layer3(x2)
    if feature_layer == "layer3":
        return x3

    if feature_layer == "layer23":
        x2_resized = torch.nn.functional.interpolate(
            x2, size=x3.shape[-2:], mode="bilinear", align_corners=False,
        )
        return torch.cat([x2_resized, x3], dim=1)

    x4 = model.layer4(x3)
    if feature_layer == "layer4":
        return x4

    raise ValueError(f"Unsupported feature_layer={feature_layer}")


def image_to_patch_embeddings(
    pil_img,
    model_dir: Optional[str] = None,
    feature_layer: str = "layer3",
) -> np.ndarray:
    x = _pil_to_tensor(pil_img).to(_DEVICE)
    model = get_embedder(model_dir)

    with torch.no_grad():
        fmap = _forward_to_layer(model, x, feature_layer=feature_layer)
        fmap = fmap.squeeze(0)
        c, h, w = fmap.shape
        patches = fmap.permute(1, 2, 0).reshape(h * w, c)
        patches = patches / (torch.norm(patches, dim=1, keepdim=True) + 1e-12)

    return patches.cpu().numpy().astype(np.float32)


def image_to_patch_grid(
    pil_img,
    model_dir: Optional[str] = None,
    feature_layer: str = "layer3",
) -> Tuple[np.ndarray, Tuple[int, int]]:
    x = _pil_to_tensor(pil_img).to(_DEVICE)
    model = get_embedder(model_dir)

    with torch.no_grad():
        fmap = _forward_to_layer(model, x, feature_layer=feature_layer)
        fmap = fmap.squeeze(0)
        c, h, w = fmap.shape
        patches = fmap.permute(1, 2, 0).reshape(h * w, c)
        patches = patches / (torch.norm(patches, dim=1, keepdim=True) + 1e-12)

    return patches.cpu().numpy().astype(np.float32), (h, w)


# ---------------------------
# Reference bank I/O (auto-detect mode)
# ---------------------------

def load_reference_bank(model_dir: str) -> Tuple[np.ndarray, str]:
    """
    Load reference bank and detect its type.
    Returns: (bank, mode) where mode is "patch" or "global"
    """
    candidates = [
        (Path(model_dir) / "patch_bank.npy", "patch"),
        (Path(model_dir) / "model_artifacts" / "patch_bank.npy", "patch"),
        (Path(model_dir) / "embeddings.npy", "global"),
        (Path(model_dir) / "model_artifacts" / "embeddings.npy", "global"),
    ]

    emb_path = None
    mode = None
    for c, m in candidates:
        if c.exists():
            emb_path = c
            mode = m
            break

    if emb_path is None:
        raise FileNotFoundError(
            f"Missing reference bank in {model_dir}. "
            "Run 'fit' or provide embeddings.npy or patch_bank.npy."
        )

    bank = np.load(str(emb_path)).astype(np.float32)

    if bank.ndim != 2:
        raise ValueError(f"Invalid bank shape {bank.shape}, expected (N, D).")

    bank = l2_normalize(bank)

    print(f"✅ Bank loaded: {emb_path.name} shape={bank.shape} mode={mode}")
    return bank, mode


def load_bank_meta(model_dir: str) -> Dict:
    """Load patch_bank_meta.json if present."""
    for sub in ["", "model_artifacts"]:
        path = Path(model_dir) / sub / "patch_bank_meta.json" if sub else Path(model_dir) / "patch_bank_meta.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    return {}


# ---------------------------
# Fit reference bank (from DB) — global mode
# ---------------------------

def fit_reference_bank(
    model_dir: str,
    config_path: str = "conf/config.yaml",
    table_name: str = "mvtec_anomaly_detection",
    batch_size: int = 64,
    num_workers: int = 4,
):
    settings = Settings.from_yaml(config_path)
    dm = PFEDataManager(settings=settings)

    dataset = MVTecDataset(dm, table_name=table_name, split="train", only_good=True)
    dataset.transform = _PREPROCESS

    if len(dataset) == 0:
        raise RuntimeError(
            f"No training 'good' images found (table={table_name}, split=train, label=good)."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_dir = str(Path(model_dir).resolve())
    embedder = get_embedder(model_dir)

    embs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(_DEVICE, non_blocking=True)
            feats = embedder(imgs)
            feats = feats / (torch.norm(feats, dim=1, keepdim=True) + 1e-12)
            embs.append(feats.cpu())

    embeddings = torch.cat(embs, dim=0).numpy().astype(np.float32)

    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(model_dir, "embeddings.npy"), embeddings)
    print("✅ embeddings saved:", embeddings.shape, "->", os.path.join(model_dir, "embeddings.npy"))


# ---------------------------
# Predict anomaly — global mode (dim=512)
# ---------------------------

def predict_anomaly(
    pil_img,
    bank: np.ndarray,
    k: int = 5,
    threshold: float = 0.35,
    model_dir: Optional[str] = None,
) -> Tuple[float, str]:
    if bank is None or len(bank) == 0:
        raise ValueError("Reference bank is empty. Run fit_reference_bank() first.")

    emb = image_to_embedding(pil_img, model_dir=model_dir)[None, :]

    k_eff = max(1, min(int(k), len(bank)))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn.fit(bank)
    dists, _ = nn.kneighbors(emb, return_distance=True)

    score = float(np.mean(dists))
    label = "anomaly" if score > float(threshold) else "normal"
    return score, label


# ---------------------------
# Patch-level scoring helpers
# ---------------------------

def aggregate_patch_scores(
    patch_scores: np.ndarray,
    image_score_mode: str = "topk_mean",
    topk: int = 5,
) -> float:
    patch_scores = np.asarray(patch_scores, dtype=np.float32).reshape(-1)
    if len(patch_scores) == 0:
        raise ValueError("patch_scores is empty")

    mode = str(image_score_mode).lower()

    if mode == "max":
        return float(np.max(patch_scores))
    if mode == "mean":
        return float(np.mean(patch_scores))
    if mode == "median":
        return float(np.median(patch_scores))
    if mode == "topk_mean":
        k_eff = max(1, min(int(topk), len(patch_scores)))
        top_vals = np.sort(patch_scores)[-k_eff:]
        return float(np.mean(top_vals))

    raise ValueError(f"Unknown image_score_mode={mode}")


# ---------------------------
# Predict anomaly — patch mode
# ---------------------------

def predict_patch_anomaly(
    pil_img,
    patch_bank: np.ndarray,
    model_dir: Optional[str] = None,
    feature_layer: str = "layer3",
    patch_neighbors: int = 1,
    image_score_mode: str = "topk_mean",
    topk: int = 5,
    threshold: Optional[float] = None,
) -> Dict:
    patch_bank = np.asarray(patch_bank, dtype=np.float32)
    patch_bank = l2_normalize(patch_bank)
    n_eff = max(1, min(int(patch_neighbors), len(patch_bank)))
    nn_model = NearestNeighbors(n_neighbors=n_eff, metric="cosine")
    nn_model.fit(patch_bank)

    patches, (h, w) = image_to_patch_grid(
        pil_img,
        model_dir=model_dir,
        feature_layer=feature_layer,
    )

    patches = np.asarray(patches, dtype=np.float32)
    patches = l2_normalize(patches)

    dists, _ = nn_model.kneighbors(patches, return_distance=True)

    if dists.ndim == 2 and dists.shape[1] > 1:
        patch_scores = dists.mean(axis=1).astype(np.float32)
    else:
        patch_scores = dists.reshape(-1).astype(np.float32)

    image_score = aggregate_patch_scores(
        patch_scores,
        image_score_mode=image_score_mode,
        topk=topk,
    )

    patch_map = patch_scores.reshape(h, w)

    pred_label = None
    if threshold is not None:
        pred_label = "anomaly" if float(image_score) > float(threshold) else "normal"

    return {
        "image_score": float(image_score),
        "pred_label": pred_label,
        "patch_scores": patch_scores,
        "patch_map": patch_map,
        "grid_size": (int(h), int(w)),
    }


# ---------------------------
# Heatmap utilities
# ---------------------------

def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_min, x_max = float(x.min()), float(x.max())
    if abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def upsample_patch_map(
    patch_map: np.ndarray,
    out_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    patch_map = normalize_map(patch_map)
    arr_u8 = (patch_map * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr_u8, mode="L")
    pil = pil.resize(out_size, resample=Image.BICUBIC)
    return np.asarray(pil).astype(np.float32) / 255.0


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    heatmap = normalize_map(heatmap)
    colored = cm.get_cmap("jet")(heatmap)[..., :3]
    return (colored * 255.0).clip(0, 255).astype(np.uint8)


def overlay_heatmap_on_image(
    pil_img,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    out_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    if out_size is None:
        out_size = pil_img.size

    base = pil_img.resize(out_size, resample=Image.BICUBIC)
    base_np = np.asarray(base).astype(np.float32)

    heatmap_resized = upsample_patch_map(heatmap, out_size=out_size)
    heat_rgb = heatmap_to_rgb(heatmap_resized).astype(np.float32)

    alpha = min(max(float(alpha), 0.0), 1.0)
    overlay = (1.0 - alpha) * base_np + alpha * heat_rgb
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_patch_heatmap(
    patch_map: np.ndarray,
    out_path: str | Path,
    out_size: Tuple[int, int] = (224, 224),
) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = upsample_patch_map(patch_map, out_size=out_size)
    heat_rgb = heatmap_to_rgb(heatmap)
    Image.fromarray(heat_rgb).save(str(out_path))
    return str(out_path.resolve())


def save_patch_overlay(
    pil_img,
    patch_map: np.ndarray,
    out_path: str | Path,
    alpha: float = 0.45,
    out_size: Optional[Tuple[int, int]] = None,
) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap_on_image(pil_img, heatmap=patch_map, alpha=alpha, out_size=out_size)
    Image.fromarray(overlay).save(str(out_path))
    return str(out_path.resolve())
