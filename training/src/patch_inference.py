# worker_2d/app/patch_inference.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import Settings
from .data import MVTecDataset, PFEDataManager
from .inference import image_to_patch_grid
import pandas as pd
from torch.utils.data import Dataset
from .inference import image_to_patch_embeddings
from typing import Dict, Optional, Tuple, List
from PIL import Image, ImageFilter

# ---------------------------
# Cache
# ---------------------------

_PATCH_BANK_CACHE: Dict[str, np.ndarray] = {}
_PATCH_NN_CACHE: Dict[Tuple[str, int], NearestNeighbors] = {}


def _resolve_patch_bank_key(model_dir: str, feature_layer: str, backbone_name: str) -> str:
    model_dir = str(Path(model_dir).resolve())
    return f"{model_dir}::{feature_layer.lower()}::{backbone_name.lower()}"


def clear_patch_bank_cache(
    model_dir: Optional[str] = None,
    feature_layer: Optional[str] = None,
    backbone_name: Optional[str] = None,
) -> None:
    if model_dir is None and feature_layer is None and backbone_name is None:
        _PATCH_BANK_CACHE.clear()
        return

    model_key = str(Path(model_dir).resolve()) if model_dir is not None else None
    feat_key = feature_layer.lower() if feature_layer is not None else None
    back_key = backbone_name.lower() if backbone_name is not None else None

    keys_to_delete = []
    for key in _PATCH_BANK_CACHE:
        cache_model, cache_feat, cache_back = key.split("::")
        model_match = model_key is None or cache_model == model_key
        feat_match = feat_key is None or cache_feat == feat_key
        back_match = back_key is None or cache_back == back_key
        if model_match and feat_match and back_match:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        _PATCH_BANK_CACHE.pop(key, None)


def clear_patch_nn_cache(
    model_dir: Optional[str] = None,
    feature_layer: Optional[str] = None,
    backbone_name: Optional[str] = None,
    n_neighbors: Optional[int] = None,
) -> None:
    if model_dir is None and feature_layer is None and backbone_name is None and n_neighbors is None:
        _PATCH_NN_CACHE.clear()
        return

    model_key = str(Path(model_dir).resolve()) if model_dir is not None else None
    feat_key = feature_layer.lower() if feature_layer is not None else None
    back_key = backbone_name.lower() if backbone_name is not None else None
    nn_key = int(n_neighbors) if n_neighbors is not None else None

    keys_to_delete = []
    for key in _PATCH_NN_CACHE:
        cache_bank_key, cache_n = key
        cache_model, cache_feat, cache_back = cache_bank_key.split("::")

        model_match = model_key is None or cache_model == model_key
        feat_match = feat_key is None or cache_feat == feat_key
        back_match = back_key is None or cache_back == back_key
        n_match = nn_key is None or cache_n == nn_key

        if model_match and feat_match and back_match and n_match:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        _PATCH_NN_CACHE.pop(key, None)


def clear_all_patch_caches(
    model_dir: Optional[str] = None,
    feature_layer: Optional[str] = None,
    backbone_name: Optional[str] = None,
) -> None:
    clear_patch_bank_cache(
        model_dir=model_dir,
        feature_layer=feature_layer,
    )
    clear_patch_nn_cache(
        model_dir=model_dir,
        feature_layer=feature_layer,
    )


# ---------------------------
# Small helpers
# ---------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Apply row-wise L2 normalization to feature vectors.

    Args:
        x: np.ndarray of shape (N, D)

    Returns:
        np.ndarray of shape (N, D), L2-normalized
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array of shape (N, D), got {x.shape}")

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _ensure_uint8_rgb(pil_img: Image.Image) -> Image.Image:
    return pil_img.convert("RGB")


def _ensure_out_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def _parse_normal_values(raw: str) -> set[str]:
    return {v.strip().lower() for v in str(raw).split(",") if v.strip()}


def _find_label_column(
    df: pd.DataFrame,
    explicit_label_col: str | None = None,
) -> str | None:
    if explicit_label_col is not None:
        if explicit_label_col not in df.columns:
            raise ValueError(
                f"Requested label column '{explicit_label_col}' not found in dataframe columns: {list(df.columns)}"
            )
        return explicit_label_col

    candidates = [
        "label",
        "target",
        "y",
        "is_anomaly",
        "anomaly",
        "status",
        "class",
        "ground_truth",
    ]
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
        detected_label_col = _find_label_column(out, explicit_label_col=label_col)
        if detected_label_col is None:
            raise ValueError(
                "normal-only filtering requested, but no label column was found. "
                "Use --label-col to specify it explicitly."
            )

        allowed_normals = _parse_normal_values(normal_values)
        before = len(out)

        normalized_series = out[detected_label_col].astype(str).str.strip().str.lower()
        out = out[normalized_series.isin(allowed_normals)].copy()

        print(
            f"ℹ️ Normal-only filter: column='{detected_label_col}', "
            f"normal_values={sorted(allowed_normals)} -> {len(out)}/{before} rows kept"
        )

    if len(out) == 0:
        raise ValueError("No samples left after filtering for patch bank construction.")

    return out


def _print_fit_dataframe_summary(
    df: pd.DataFrame,
    explicit_label_col: str | None = None,
) -> None:
    print("\n=== PATCH BANK FIT DATA SUMMARY ===")
    print(f"Rows used: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if "split" in df.columns:
        print("Split counts:")
        print(df["split"].astype(str).value_counts(dropna=False).to_dict())

    detected_label_col = _find_label_column(df, explicit_label_col=explicit_label_col)
    if detected_label_col is not None:
        print(f"Label column: {detected_label_col}")
        print("Label counts:")
        print(df[detected_label_col].astype(str).value_counts(dropna=False).to_dict())

    print("==================================\n")


class FilteredImageDataset(Dataset):
    """
    Dataset wrapper built from a filtered dataframe returned by PFEDataManager.get_dataset().
    Expected columns:
      - filepath
      - optionally split / label / category
    """

    def __init__(self, dm: PFEDataManager, df: pd.DataFrame) -> None:
        self.dm = dm
        self.df = df.reset_index(drop=True).copy()

        if "filepath" not in self.df.columns:
            raise ValueError(
                f"Filtered dataframe must contain a 'filepath' column. Columns={list(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filepath = str(row["filepath"])
        img = self.dm.load_image(filepath, strict=True)
        return img, 0

# ---------------------------
# Patch bank I/O
# ---------------------------

def patch_bank_path(model_dir: str) -> Path:
    return Path(model_dir) / "patch_bank.npy"


def patch_bank_meta_path(model_dir: str) -> Path:
    return Path(model_dir) / "patch_bank_meta.json"


def load_patch_reference_bank(
    model_dir: str,
    feature_layer: str = "layer3",
    backbone_name: str = "resnet18",
    use_cache: bool = True,
) -> np.ndarray:
    """
    Load patch_bank.npy and return normalized bank [N, D].
    """
    key = _resolve_patch_bank_key(model_dir, feature_layer, backbone_name)

    if use_cache and key in _PATCH_BANK_CACHE:
        return _PATCH_BANK_CACHE[key]

    path = patch_bank_path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing patch bank: {path}")

    bank = np.load(str(path)).astype(np.float32)
    if bank.ndim != 2:
        raise ValueError(f"Invalid patch bank shape: {bank.shape}, expected (N, D)")

    # Safety normalization at load time
    bank = l2_normalize(bank)

    if use_cache:
        _PATCH_BANK_CACHE[key] = bank

    return bank


def load_patch_bank_meta(model_dir: str) -> Dict:
    path = patch_bank_meta_path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing patch bank meta: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid patch bank meta format: {path}")

    return payload


def get_cached_patch_nn(
    model_dir: str,
    feature_layer: str = "layer3",
    backbone_name: str = "resnet18",
    n_neighbors: int = 1,
) -> Tuple[np.ndarray, NearestNeighbors]:
    bank_key = _resolve_patch_bank_key(model_dir, feature_layer, backbone_name)
    n_neighbors = max(1, int(n_neighbors))
    cache_key = (bank_key, n_neighbors)

    cached = _PATCH_NN_CACHE.get(cache_key)
    bank = load_patch_reference_bank(
        model_dir=model_dir,
        feature_layer=feature_layer,
        use_cache=True,
    )

    if len(bank) == 0:
        raise ValueError("Patch bank is empty.")

    n_eff = min(n_neighbors, len(bank))

    if cached is not None:
        return bank, cached

    # With L2-normalized features, cosine distance is a strong default
    nn_model = NearestNeighbors(n_neighbors=n_eff, metric="cosine")
    nn_model.fit(bank)
    _PATCH_NN_CACHE[(bank_key, n_eff)] = nn_model

    return bank, nn_model

def _random_projection(
    x: np.ndarray,
    out_dim: int = 64,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Simple Gaussian random projection.
    Input:
      x: [N, D]
    Output:
      z: [N, out_dim]
    """
    x = np.asarray(x, dtype=np.float32)
    n, d = x.shape

    if out_dim is None or out_dim <= 0 or out_dim >= d:
        return x

    rng = np.random.default_rng(int(random_seed))
    proj = rng.standard_normal(size=(d, out_dim)).astype(np.float32)
    proj = proj / np.sqrt(out_dim)

    z = x @ proj
    return z.astype(np.float32)


def _greedy_coreset_indices(
    x: np.ndarray,
    coreset_size: int,
    random_seed: int = 42,
    start_idx: int | None = None,
) -> np.ndarray:
    """
    Greedy k-center style coreset selection.
    x must be [N, D].
    Returns indices of selected samples.
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)

    if coreset_size >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(random_seed))

    if start_idx is None:
        start_idx = int(rng.integers(0, n))

    selected = np.empty(coreset_size, dtype=np.int64)
    selected[0] = start_idx

    # squared euclidean distances to the first center
    diff = x - x[start_idx]
    min_dist = np.sum(diff * diff, axis=1)

    for i in range(1, coreset_size):
        next_idx = int(np.argmax(min_dist))
        selected[i] = next_idx

        diff = x - x[next_idx]
        dist = np.sum(diff * diff, axis=1)
        min_dist = np.minimum(min_dist, dist)

    return selected


def build_simple_coreset(
    patch_bank: np.ndarray,
    coreset_size: int,
    random_seed: int = 42,
    pre_sample_size: int | None = 50000,
    proj_dim: int | None = 64,
) -> np.ndarray:
    """
    Build a simple coreset from patch_bank.

    Steps:
      1) L2 normalize
      2) optional random pre-sampling
      3) optional projection
      4) greedy coreset selection
    """
    x = np.asarray(patch_bank, dtype=np.float32)

    if len(x) == 0:
        raise ValueError("patch_bank is empty")

    x = l2_normalize(x)

    rng = np.random.default_rng(int(random_seed))

    # optional pre-sampling to keep greedy tractable
    if pre_sample_size is not None and len(x) > int(pre_sample_size):
        pre_idx = rng.choice(len(x), size=int(pre_sample_size), replace=False)
        x_work = x[pre_idx]
    else:
        pre_idx = None
        x_work = x

    # optional projection
    z = _random_projection(
        x_work,
        out_dim=proj_dim if proj_dim is not None else 0,
        random_seed=random_seed,
    )

    k = min(int(coreset_size), len(z))
    sel_local = _greedy_coreset_indices(
        z,
        coreset_size=k,
        random_seed=random_seed,
    )

    if pre_idx is not None:
        sel_global = pre_idx[sel_local]
    else:
        sel_global = sel_local

    return x[sel_global].astype(np.float32)
# ---------------------------
# Fit patch bank
# ---------------------------

def fit_patch_reference_bank(
    model_dir: str,
    config_path: str = "conf/config.yaml",
    table_name: str = "mvtec_anomaly_detection",
    backbone_name: str = "resnet18",
    feature_layer: str = "layer3",
    max_patches: int = 200000,
    random_seed: int = 42,
    fit_split: str = "train",
    normal_only: bool = False,
    label_col: str | None = None,
    normal_values: str = "0,normal,good,false",
    bank_selection: str = "random",
    coreset_pre_sample_size: int | None = 50000,
    coreset_proj_dim: int | None = 64,
    category: str | None = None,
) -> None:
    """
    Build a patch-level reference bank from filtered dataset images.

    Structured version:
      - optional split filtering
      - optional normal-only filtering
      - random subsampling to max_patches
    """
    settings = Settings.from_yaml(config_path)
    dm = PFEDataManager(settings=settings)

    df = dm.get_dataset(
        table=table_name,
        load_images=False,
        verbose=True,
        raise_on_error=True,
    )

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = _filter_fit_dataframe(
        df,
        fit_split=fit_split,
        normal_only=normal_only,
        label_col=label_col,
        normal_values=normal_values,
    )

    _print_fit_dataframe_summary(df, explicit_label_col=label_col)

    dataset = FilteredImageDataset(dm=dm, df=df)

    if len(dataset) == 0:
        raise RuntimeError(
            f"No images found after filtering "
            f"(table={table_name}, split={fit_split}, normal_only={normal_only})."
        )

    model_dir = str(Path(model_dir).resolve())
    backbone_name = str(backbone_name).lower()

    all_patches = []

    if category is not None:
        before = len(df)
        df = df[df["category"].astype(str) == str(category)].copy()
        print(f"ℹ️ Category filter: category='{category}' -> {len(df)}/{before} rows kept")

    for idx in range(len(dataset)):
        pil_img, _ = dataset[idx]

        patches = image_to_patch_embeddings(
            pil_img=pil_img,
            model_dir=model_dir,

            feature_layer=feature_layer,
        )

        if patches is None or len(patches) == 0:
            continue

        all_patches.append(patches)

    if not all_patches:
        raise RuntimeError("No patch embeddings were extracted. Patch bank cannot be built.")

    patch_bank = np.concatenate(all_patches, axis=0).astype(np.float32)
    patch_bank = l2_normalize(patch_bank)

    initial_n = len(patch_bank)
    selection_mode = str(bank_selection).lower()

    if max_patches is not None and len(patch_bank) > int(max_patches):
        if selection_mode == "random":
            rng = np.random.default_rng(int(random_seed))
            keep_idx = rng.choice(len(patch_bank), size=int(max_patches), replace=False)
            patch_bank = patch_bank[keep_idx]
            print(f"ℹ️ Random subsampling: {initial_n} -> {len(patch_bank)} patches")

        elif selection_mode == "coreset":
            patch_bank = build_simple_coreset(
                patch_bank=patch_bank,
                coreset_size=int(max_patches),
                random_seed=random_seed,
                pre_sample_size=coreset_pre_sample_size,
                proj_dim=coreset_proj_dim,
            )
            print(f"ℹ️ Coreset selection: {initial_n} -> {len(patch_bank)} patches")

        else:
            raise ValueError(
                f"Unknown bank_selection='{bank_selection}'. Expected 'random' or 'coreset'."
            )
    else:
        print(f"ℹ️ No reduction applied: {initial_n} patches kept")

    os.makedirs(model_dir, exist_ok=True)

    patch_bank_path = Path(model_dir) / "patch_bank.npy"
    np.save(str(patch_bank_path), patch_bank)

    # si tu as déjà ces fonctions dans ton fichier, garde-les
    if "clear_patch_bank_cache" in globals():
        clear_patch_bank_cache(model_dir=model_dir, backbone_name=backbone_name)

    if "clear_patch_knn_cache" in globals():
        clear_patch_knn_cache(model_dir=model_dir, backbone_name=backbone_name)
    meta = {
        "table_name": table_name,
        "backbone": backbone_name,
        "feature_layer": feature_layer,
        "fit_split": fit_split,
        "normal_only": bool(normal_only),
        "label_col": label_col,
        "normal_values": normal_values,
        "bank_selection": selection_mode,
        "random_seed": int(random_seed),
        "max_patches": int(max_patches) if max_patches is not None else None,
        "coreset_pre_sample_size": (
            int(coreset_pre_sample_size) if coreset_pre_sample_size is not None else None
        ),
        "coreset_proj_dim": (
            int(coreset_proj_dim) if coreset_proj_dim is not None else None
        ),
        "n_patches_before_reduction": int(initial_n),
        "n_patches_after_reduction": int(len(patch_bank)),
        "embedding_dim": int(patch_bank.shape[1]),
        "category": category,
    }

    meta_path = patch_bank_meta_path(model_dir)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✅ patch_bank saved: {patch_bank.shape} -> {patch_bank_path}")


# ---------------------------
# Patch-level scoring
# ---------------------------

def aggregate_patch_scores(
    patch_scores: np.ndarray,
    image_score_mode: str = "topk_mean",
    topk: int = 5,
) -> float:
    """
    Aggregate per-patch anomaly scores into one image-level score.

    Supported:
      - max
      - mean
      - topk_mean
      - median
    """
    patch_scores = np.asarray(patch_scores, dtype=np.float32).reshape(-1)

    if len(patch_scores) == 0:
        raise ValueError("patch_scores is empty")

    image_score_mode = str(image_score_mode).lower()

    if image_score_mode == "max":
        return float(np.max(patch_scores))

    if image_score_mode == "mean":
        return float(np.mean(patch_scores))

    if image_score_mode == "median":
        return float(np.median(patch_scores))

    if image_score_mode == "topk_mean":
        k_eff = max(1, min(int(topk), len(patch_scores)))
        top_vals = np.sort(patch_scores)[-k_eff:]
        return float(np.mean(top_vals))

    raise ValueError(f"Unknown image_score_mode={image_score_mode}")


def predict_patch_anomaly(
    pil_img,
    patch_bank: Optional[np.ndarray] = None,
    model_dir: Optional[str] = None,
    backbone_name: str = "resnet18",
    feature_layer: str = "layer3",
    patch_neighbors: int = 1,
    image_score_mode: str = "topk_mean",
    topk: int = 5,
    threshold: Optional[float] = None,
) -> Dict:
    """
    Patch-level anomaly prediction.

    Returns:
      {
        "image_score": float,
        "pred_label": Optional[str],
        "patch_scores": np.ndarray [P],
        "patch_map": np.ndarray [H, W],
        "grid_size": (H, W),
      }
    """
    if patch_bank is None:
        if model_dir is None:
            raise ValueError("Provide patch_bank or model_dir.")
        _, nn_model = get_cached_patch_nn(
            model_dir=model_dir,
            feature_layer=feature_layer,

            n_neighbors=patch_neighbors,
        )
    else:
        patch_bank = np.asarray(patch_bank, dtype=np.float32)
        if patch_bank.ndim != 2:
            raise ValueError(f"Invalid patch bank shape: {patch_bank.shape}")

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
    if patches.ndim != 2:
        raise ValueError(f"Invalid query patch shape: {patches.shape}, expected (N, D)")

    # Critical addition: normalize query patches before nearest-neighbor search
    patches = l2_normalize(patches)

    dists, _ = nn_model.kneighbors(patches, return_distance=True)

    # Patch anomaly score = mean distance to patch_neighbors nearest normal patches
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
    x_min = float(x.min())
    x_max = float(x.max())
    if abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def upsample_patch_map(
    patch_map: np.ndarray,
    out_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Upsample a patch map [H, W] to [out_h, out_w].

    Uses PIL resize to avoid extra dependency on cv2.
    out_size = (width, height)
    """
    patch_map = np.asarray(patch_map, dtype=np.float32)
    patch_map = normalize_map(patch_map)

    arr_u8 = (patch_map * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr_u8, mode="L")
    pil = pil.resize(out_size, resample=Image.BICUBIC)

    out = np.asarray(pil).astype(np.float32) / 255.0
    return out


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    """
    Convert [H, W] normalized heatmap to RGB uint8 using matplotlib colormap.
    """
    import matplotlib.cm as cm

    heatmap = normalize_map(heatmap)
    colored = cm.get_cmap("jet")(heatmap)[..., :3]
    colored = (colored * 255.0).clip(0, 255).astype(np.uint8)
    return colored


def overlay_heatmap_on_image(
    pil_img,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    out_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Returns overlay as RGB uint8 array.
    """
    pil_img = _ensure_uint8_rgb(pil_img)

    if out_size is None:
        out_size = pil_img.size  # (width, height)

    base = pil_img.resize(out_size, resample=Image.BICUBIC)
    base_np = np.asarray(base).astype(np.float32)

    heatmap_resized = upsample_patch_map(heatmap, out_size=out_size)
    heat_rgb = heatmap_to_rgb(heatmap_resized).astype(np.float32)

    alpha = float(alpha)
    alpha = min(max(alpha, 0.0), 1.0)

    overlay = (1.0 - alpha) * base_np + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def save_patch_heatmap(
    patch_map: np.ndarray,
    out_path: str | Path,
    out_size: Tuple[int, int] = (224, 224),
) -> str:
    """
    Save heatmap as RGB PNG and return absolute path.
    """
    _ensure_out_dir(out_path)
    heatmap = upsample_patch_map(patch_map, out_size=out_size)
    heat_rgb = heatmap_to_rgb(heatmap)
    Image.fromarray(heat_rgb).save(str(out_path))
    return str(Path(out_path).resolve())


def save_patch_overlay(
    pil_img,
    patch_map: np.ndarray,
    out_path: str | Path,
    alpha: float = 0.45,
    out_size: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Save overlay PNG and return absolute path.
    """
    _ensure_out_dir(out_path)
    overlay = overlay_heatmap_on_image(
        pil_img=pil_img,
        heatmap=patch_map,
        alpha=alpha,
        out_size=out_size,
    )
    Image.fromarray(overlay).save(str(out_path))
    return str(Path(out_path).resolve())

# ---------------------------
# Cache
# ---------------------------

_PATCH_BANK_CACHE: Dict[str, np.ndarray] = {}
_PATCH_NN_CACHE: Dict[Tuple[str, int], NearestNeighbors] = {}


def _resolve_patch_bank_key(model_dir: str, feature_layer: str, backbone_name: str) -> str:
    model_dir = str(Path(model_dir).resolve())
    return f"{model_dir}::{feature_layer.lower()}::{backbone_name.lower()}"


def clear_patch_bank_cache(
    model_dir: Optional[str] = None,
    feature_layer: Optional[str] = None,
    backbone_name: Optional[str] = None,
) -> None:
    if model_dir is None and feature_layer is None and backbone_name is None:
        _PATCH_BANK_CACHE.clear()
        return

    model_key = str(Path(model_dir).resolve()) if model_dir is not None else None
    feat_key = feature_layer.lower() if feature_layer is not None else None
    back_key = backbone_name.lower() if backbone_name is not None else None

    keys_to_delete = []
    for key in _PATCH_BANK_CACHE:
        cache_model, cache_feat, cache_back = key.split("::")
        model_match = model_key is None or cache_model == model_key
        feat_match = feat_key is None or cache_feat == feat_key
        back_match = back_key is None or cache_back == back_key
        if model_match and feat_match and back_match:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        _PATCH_BANK_CACHE.pop(key, None)


def clear_patch_nn_cache(
    model_dir: Optional[str] = None,
    feature_layer: Optional[str] = None,
    backbone_name: Optional[str] = None,
    n_neighbors: Optional[int] = None,
) -> None:
    if model_dir is None and feature_layer is None and backbone_name is None and n_neighbors is None:
        _PATCH_NN_CACHE.clear()
        return

    model_key = str(Path(model_dir).resolve()) if model_dir is not None else None
    feat_key = feature_layer.lower() if feature_layer is not None else None
    back_key = backbone_name.lower() if backbone_name is not None else None
    nn_key = int(n_neighbors) if n_neighbors is not None else None

    keys_to_delete = []
    for key in _PATCH_NN_CACHE:
        cache_bank_key, cache_n = key
        cache_model, cache_feat, cache_back = cache_bank_key.split("::")

        model_match = model_key is None or cache_model == model_key
        feat_match = feat_key is None or cache_feat == feat_key
        back_match = back_key is None or cache_back == back_key
        n_match = nn_key is None or cache_n == nn_key

        if model_match and feat_match and back_match and n_match:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        _PATCH_NN_CACHE.pop(key, None)


def clear_all_patch_caches(
    model_dir: Optional[str] = None,
    feature_layer: Optional[str] = None,
    backbone_name: Optional[str] = None,
) -> None:
    clear_patch_bank_cache(
        model_dir=model_dir,
        feature_layer=feature_layer,
    )
    clear_patch_nn_cache(
        model_dir=model_dir,
        feature_layer=feature_layer,
    )


# ---------------------------
# Small helpers
# ---------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array of shape (N, D), got {x.shape}")

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _ensure_uint8_rgb(pil_img: Image.Image) -> Image.Image:
    return pil_img.convert("RGB")


def _ensure_out_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _parse_normal_values(raw: str) -> set[str]:
    return {v.strip().lower() for v in str(raw).split(",") if v.strip()}


def _find_label_column(
    df: pd.DataFrame,
    explicit_label_col: str | None = None,
) -> str | None:
    if explicit_label_col is not None:
        if explicit_label_col not in df.columns:
            raise ValueError(
                f"Requested label column '{explicit_label_col}' not found in dataframe columns: {list(df.columns)}"
            )
        return explicit_label_col

    candidates = [
        "label",
        "target",
        "y",
        "is_anomaly",
        "anomaly",
        "status",
        "class",
        "ground_truth",
    ]
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
        detected_label_col = _find_label_column(out, explicit_label_col=label_col)
        if detected_label_col is None:
            raise ValueError(
                "normal-only filtering requested, but no label column was found. "
                "Use --label-col to specify it explicitly."
            )

        allowed_normals = _parse_normal_values(normal_values)
        before = len(out)

        normalized_series = out[detected_label_col].astype(str).str.strip().str.lower()
        out = out[normalized_series.isin(allowed_normals)].copy()

        print(
            f"ℹ️ Normal-only filter: column='{detected_label_col}', "
            f"normal_values={sorted(allowed_normals)} -> {len(out)}/{before} rows kept"
        )

    if len(out) == 0:
        raise ValueError("No samples left after filtering for patch bank construction.")

    return out


def _print_fit_dataframe_summary(
    df: pd.DataFrame,
    explicit_label_col: str | None = None,
) -> None:
    print("\n=== PATCH BANK FIT DATA SUMMARY ===")
    print(f"Rows used: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if "split" in df.columns:
        print("Split counts:")
        print(df["split"].astype(str).value_counts(dropna=False).to_dict())

    detected_label_col = _find_label_column(df, explicit_label_col=explicit_label_col)
    if detected_label_col is not None:
        print(f"Label column: {detected_label_col}")
        print("Label counts:")
        print(df[detected_label_col].astype(str).value_counts(dropna=False).to_dict())

    print("==================================\n")


class FilteredImageDataset(Dataset):
    def __init__(self, dm: PFEDataManager, df: pd.DataFrame) -> None:
        self.dm = dm
        self.df = df.reset_index(drop=True).copy()

        if "filepath" not in self.df.columns:
            raise ValueError(
                f"Filtered dataframe must contain a 'filepath' column. Columns={list(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filepath = str(row["filepath"])
        img = self.dm.load_image(filepath, strict=True)
        return img, 0


# ---------------------------
# Patch bank I/O
# ---------------------------

def patch_bank_path(model_dir: str) -> Path:
    return Path(model_dir) / "patch_bank.npy"


def patch_bank_meta_path(model_dir: str) -> Path:
    return Path(model_dir) / "patch_bank_meta.json"


def load_patch_reference_bank(
    model_dir: str,
    feature_layer: str = "layer3",
    backbone_name: str = "resnet18",
    use_cache: bool = True,
) -> np.ndarray:
    key = _resolve_patch_bank_key(model_dir, feature_layer, backbone_name)

    if use_cache and key in _PATCH_BANK_CACHE:
        return _PATCH_BANK_CACHE[key]

    path = patch_bank_path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing patch bank: {path}")

    bank = np.load(str(path)).astype(np.float32)
    if bank.ndim != 2:
        raise ValueError(f"Invalid patch bank shape: {bank.shape}, expected (N, D)")

    bank = l2_normalize(bank)

    if use_cache:
        _PATCH_BANK_CACHE[key] = bank

    return bank


def load_patch_bank_meta(model_dir: str) -> Dict:
    path = patch_bank_meta_path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing patch bank meta: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid patch bank meta format: {path}")

    return payload


def get_cached_patch_nn(
    model_dir: str,
    feature_layer: str = "layer3",
    backbone_name: str = "resnet18",
    n_neighbors: int = 1,
) -> Tuple[np.ndarray, NearestNeighbors]:
    bank_key = _resolve_patch_bank_key(model_dir, feature_layer, backbone_name)
    n_neighbors = max(1, int(n_neighbors))
    cache_key = (bank_key, n_neighbors)

    cached = _PATCH_NN_CACHE.get(cache_key)
    bank = load_patch_reference_bank(
        model_dir=model_dir,
        feature_layer=feature_layer,
        use_cache=True,
    )

    if len(bank) == 0:
        raise ValueError("Patch bank is empty.")

    n_eff = min(n_neighbors, len(bank))

    if cached is not None:
        return bank, cached

    nn_model = NearestNeighbors(n_neighbors=n_eff, metric="cosine")
    nn_model.fit(bank)
    _PATCH_NN_CACHE[(bank_key, n_eff)] = nn_model

    return bank, nn_model


def _random_projection(
    x: np.ndarray,
    out_dim: int = 64,
    random_seed: int = 42,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n, d = x.shape

    if out_dim is None or out_dim <= 0 or out_dim >= d:
        return x

    rng = np.random.default_rng(int(random_seed))
    proj = rng.standard_normal(size=(d, out_dim)).astype(np.float32)
    proj = proj / np.sqrt(out_dim)

    z = x @ proj
    return z.astype(np.float32)


def _greedy_coreset_indices(
    x: np.ndarray,
    coreset_size: int,
    random_seed: int = 42,
    start_idx: int | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)

    if coreset_size >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(random_seed))

    if start_idx is None:
        start_idx = int(rng.integers(0, n))

    selected = np.empty(coreset_size, dtype=np.int64)
    selected[0] = start_idx

    diff = x - x[start_idx]
    min_dist = np.sum(diff * diff, axis=1)

    for i in range(1, coreset_size):
        next_idx = int(np.argmax(min_dist))
        selected[i] = next_idx

        diff = x - x[next_idx]
        dist = np.sum(diff * diff, axis=1)
        min_dist = np.minimum(min_dist, dist)

    return selected


def build_simple_coreset(
    patch_bank: np.ndarray,
    coreset_size: int,
    random_seed: int = 42,
    pre_sample_size: int | None = 50000,
    proj_dim: int | None = 64,
) -> np.ndarray:
    x = np.asarray(patch_bank, dtype=np.float32)

    if len(x) == 0:
        raise ValueError("patch_bank is empty")

    x = l2_normalize(x)

    rng = np.random.default_rng(int(random_seed))

    if pre_sample_size is not None and len(x) > int(pre_sample_size):
        pre_idx = rng.choice(len(x), size=int(pre_sample_size), replace=False)
        x_work = x[pre_idx]
    else:
        pre_idx = None
        x_work = x

    z = _random_projection(
        x_work,
        out_dim=proj_dim if proj_dim is not None else 0,
        random_seed=random_seed,
    )

    k = min(int(coreset_size), len(z))
    sel_local = _greedy_coreset_indices(
        z,
        coreset_size=k,
        random_seed=random_seed,
    )

    if pre_idx is not None:
        sel_global = pre_idx[sel_local]
    else:
        sel_global = sel_local

    return x[sel_global].astype(np.float32)


# ---------------------------
# Fit patch bank
# ---------------------------

def fit_patch_reference_bank(
    model_dir: str,
    config_path: str = "conf/config.yaml",
    table_name: str = "mvtec_anomaly_detection",
    backbone_name: str = "resnet18",
    feature_layer: str = "layer3",
    max_patches: int = 200000,
    random_seed: int = 42,
    fit_split: str = "train",
    normal_only: bool = False,
    label_col: str | None = None,
    normal_values: str = "0,normal,good,false",
    bank_selection: str = "random",
    coreset_pre_sample_size: int | None = 50000,
    coreset_proj_dim: int | None = 64,
    category: str | None = None,
) -> None:
    settings = Settings.from_yaml(config_path)
    dm = PFEDataManager(settings=settings)

    df = dm.get_dataset(
        table=table_name,
        load_images=False,
        verbose=True,
        raise_on_error=True,
    )

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = _filter_fit_dataframe(
        df,
        fit_split=fit_split,
        normal_only=normal_only,
        label_col=label_col,
        normal_values=normal_values,
    )

    _print_fit_dataframe_summary(df, explicit_label_col=label_col)

    dataset = FilteredImageDataset(dm=dm, df=df)

    if len(dataset) == 0:
        raise RuntimeError(
            f"No images found after filtering "
            f"(table={table_name}, split={fit_split}, normal_only={normal_only})."
        )

    model_dir = str(Path(model_dir).resolve())
    backbone_name = str(backbone_name).lower()

    all_patches = []

    if category is not None:
        before = len(df)
        df = df[df["category"].astype(str) == str(category)].copy()
        print(f"ℹ️ Category filter: category='{category}' -> {len(df)}/{before} rows kept")

    for idx in range(len(dataset)):
        pil_img, _ = dataset[idx]

        patches = image_to_patch_embeddings(
            pil_img=pil_img,
            model_dir=model_dir,

            feature_layer=feature_layer,
        )

        if patches is None or len(patches) == 0:
            continue

        all_patches.append(patches)

    if not all_patches:
        raise RuntimeError("No patch embeddings were extracted. Patch bank cannot be built.")

    patch_bank = np.concatenate(all_patches, axis=0).astype(np.float32)
    patch_bank = l2_normalize(patch_bank)

    initial_n = len(patch_bank)
    selection_mode = str(bank_selection).lower()

    if max_patches is not None and len(patch_bank) > int(max_patches):
        if selection_mode == "random":
            rng = np.random.default_rng(int(random_seed))
            keep_idx = rng.choice(len(patch_bank), size=int(max_patches), replace=False)
            patch_bank = patch_bank[keep_idx]
            print(f"ℹ️ Random subsampling: {initial_n} -> {len(patch_bank)} patches")

        elif selection_mode == "coreset":
            patch_bank = build_simple_coreset(
                patch_bank=patch_bank,
                coreset_size=int(max_patches),
                random_seed=random_seed,
                pre_sample_size=coreset_pre_sample_size,
                proj_dim=coreset_proj_dim,
            )
            print(f"ℹ️ Coreset selection: {initial_n} -> {len(patch_bank)} patches")

        else:
            raise ValueError(
                f"Unknown bank_selection='{bank_selection}'. Expected 'random' or 'coreset'."
            )
    else:
        print(f"ℹ️ No reduction applied: {initial_n} patches kept")

    os.makedirs(model_dir, exist_ok=True)

    patch_bank_path_obj = Path(model_dir) / "patch_bank.npy"
    np.save(str(patch_bank_path_obj), patch_bank)

    if "clear_patch_bank_cache" in globals():
        clear_patch_bank_cache(model_dir=model_dir, backbone_name=backbone_name)

    meta = {
        "table_name": table_name,
        "backbone": backbone_name,
        "feature_layer": feature_layer,
        "fit_split": fit_split,
        "normal_only": bool(normal_only),
        "label_col": label_col,
        "normal_values": normal_values,
        "bank_selection": selection_mode,
        "random_seed": int(random_seed),
        "max_patches": int(max_patches) if max_patches is not None else None,
        "coreset_pre_sample_size": (
            int(coreset_pre_sample_size) if coreset_pre_sample_size is not None else None
        ),
        "coreset_proj_dim": (
            int(coreset_proj_dim) if coreset_proj_dim is not None else None
        ),
        "n_patches_before_reduction": int(initial_n),
        "n_patches_after_reduction": int(len(patch_bank)),
        "embedding_dim": int(patch_bank.shape[1]),
        "category": category,
    }

    meta_path = patch_bank_meta_path(model_dir)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✅ patch_bank saved: {patch_bank.shape} -> {patch_bank_path_obj}")


# ---------------------------
# Patch-level scoring
# ---------------------------

def aggregate_patch_scores(
    patch_scores: np.ndarray,
    image_score_mode: str = "topk_mean",
    topk: int = 5,
) -> float:
    patch_scores = np.asarray(patch_scores, dtype=np.float32).reshape(-1)

    if len(patch_scores) == 0:
        raise ValueError("patch_scores is empty")

    image_score_mode = str(image_score_mode).lower()

    if image_score_mode == "max":
        return float(np.max(patch_scores))

    if image_score_mode == "mean":
        return float(np.mean(patch_scores))

    if image_score_mode == "median":
        return float(np.median(patch_scores))

    if image_score_mode == "topk_mean":
        k_eff = max(1, min(int(topk), len(patch_scores)))
        top_vals = np.sort(patch_scores)[-k_eff:]
        return float(np.mean(top_vals))

    raise ValueError(f"Unknown image_score_mode={image_score_mode}")


def predict_patch_anomaly(
    pil_img,
    patch_bank: Optional[np.ndarray] = None,
    model_dir: Optional[str] = None,
    backbone_name: str = "resnet18",
    feature_layer: str = "layer3",
    patch_neighbors: int = 1,
    image_score_mode: str = "topk_mean",
    topk: int = 5,
    threshold: Optional[float] = None,
) -> Dict:
    if patch_bank is None:
        if model_dir is None:
            raise ValueError("Provide patch_bank or model_dir.")
        _, nn_model = get_cached_patch_nn(
            model_dir=model_dir,
            feature_layer=feature_layer,

            n_neighbors=patch_neighbors,
        )
    else:
        patch_bank = np.asarray(patch_bank, dtype=np.float32)
        if patch_bank.ndim != 2:
            raise ValueError(f"Invalid patch bank shape: {patch_bank.shape}")

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
    if patches.ndim != 2:
        raise ValueError(f"Invalid query patch shape: {patches.shape}, expected (N, D)")

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
    x_min = float(x.min())
    x_max = float(x.max())
    if abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def upsample_patch_map(
    patch_map: np.ndarray,
    out_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    patch_map = np.asarray(patch_map, dtype=np.float32)
    patch_map = normalize_map(patch_map)

    arr_u8 = (patch_map * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr_u8, mode="L")
    pil = pil.resize(out_size, resample=Image.BICUBIC)

    out = np.asarray(pil).astype(np.float32) / 255.0
    return out


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm

    heatmap = normalize_map(heatmap)
    colored = cm.get_cmap("jet")(heatmap)[..., :3]
    colored = (colored * 255.0).clip(0, 255).astype(np.uint8)
    return colored


def overlay_heatmap_on_image(
    pil_img,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    out_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    pil_img = _ensure_uint8_rgb(pil_img)

    if out_size is None:
        out_size = pil_img.size

    base = pil_img.resize(out_size, resample=Image.BICUBIC)
    base_np = np.asarray(base).astype(np.float32)

    heatmap_resized = upsample_patch_map(heatmap, out_size=out_size)
    heat_rgb = heatmap_to_rgb(heatmap_resized).astype(np.float32)

    alpha = float(alpha)
    alpha = min(max(alpha, 0.0), 1.0)

    overlay = (1.0 - alpha) * base_np + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def save_patch_heatmap(
    patch_map: np.ndarray,
    out_path: str | Path,
    out_size: Tuple[int, int] = (224, 224),
) -> str:
    _ensure_out_dir(out_path)
    heatmap = upsample_patch_map(patch_map, out_size=out_size)
    heat_rgb = heatmap_to_rgb(heatmap)
    Image.fromarray(heat_rgb).save(str(out_path))
    return str(Path(out_path).resolve())


def save_patch_overlay(
    pil_img,
    patch_map: np.ndarray,
    out_path: str | Path,
    alpha: float = 0.45,
    out_size: Optional[Tuple[int, int]] = None,
) -> str:
    _ensure_out_dir(out_path)
    overlay = overlay_heatmap_on_image(
        pil_img=pil_img,
        heatmap=patch_map,
        alpha=alpha,
        out_size=out_size,
    )
    Image.fromarray(overlay).save(str(out_path))
    return str(Path(out_path).resolve())


# =========================================================
# NEW: anomaly mask + contour / outlined image
# =========================================================

def _neighbors8(y: int, x: int, h: int, w: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def _remove_small_components(mask: np.ndarray, min_area: int = 80) -> np.ndarray:
    """
    Keep only connected components with at least min_area pixels.
    8-connectivity, no external dependency.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    kept = np.zeros((h, w), dtype=bool)

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            component = []
            visited[y, x] = True

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in _neighbors8(cy, cx, h, w):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(component) >= int(min_area):
                for cy, cx in component:
                    kept[cy, cx] = True

    return kept


def _binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    h, w = out.shape

    for _ in range(max(1, int(iterations))):
        expanded = out.copy()
        ys, xs = np.where(out)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 1), min(h, y + 2)
            x0, x1 = max(0, x - 1), min(w, x + 2)
            expanded[y0:y1, x0:x1] = True
        out = expanded

    return out


def _binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    h, w = out.shape

    for _ in range(max(1, int(iterations))):
        eroded = np.zeros_like(out, dtype=bool)
        for y in range(h):
            for x in range(w):
                y0, y1 = max(0, y - 1), min(h, y + 2)
                x0, x1 = max(0, x - 1), min(w, x + 2)
                if np.all(out[y0:y1, x0:x1]):
                    eroded[y, x] = True
        out = eroded

    return out


def patch_map_to_anomaly_mask(
    patch_map: np.ndarray,
    out_size: Tuple[int, int],
    threshold_rel: float = 0.65,
    blur_radius: float = 2.0,
    min_area: int = 80,
) -> np.ndarray:
    """
    Convert patch anomaly map to cleaned binary mask in image resolution.
    threshold_rel is applied on normalized map in [0,1].
    """
    up = upsample_patch_map(patch_map, out_size=out_size)
    up_u8 = (normalize_map(up) * 255.0).clip(0, 255).astype(np.uint8)

    pil = Image.fromarray(up_u8, mode="L")
    if blur_radius and float(blur_radius) > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    score = np.asarray(pil).astype(np.float32) / 255.0
    mask = score >= float(threshold_rel)

    if min_area and int(min_area) > 1:
        mask = _remove_small_components(mask, min_area=int(min_area))

    return mask.astype(bool)


def build_outline_from_mask(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """
    Build border pixels from binary mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    dil = _binary_dilation(mask, iterations=max(1, int(thickness)))
    ero = _binary_erosion(mask, iterations=1)
    outline = np.logical_and(dil, np.logical_not(ero))
    return outline


def overlay_mask_outline_on_image(
    pil_img,
    mask: np.ndarray,
    outline_color: Tuple[int, int, int] = (255, 0, 0),
    outline_thickness: int = 2,
    fill_alpha: float = 0.12,
) -> np.ndarray:
    """
    Overlay anomaly region with light fill + colored contour.
    """
    pil_img = _ensure_uint8_rgb(pil_img)
    base = np.asarray(pil_img).astype(np.float32)
    mask = np.asarray(mask, dtype=bool)

    if mask.shape != base.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {base.shape[:2]}"
        )

    out = base.copy()

    fill_alpha = float(fill_alpha)
    fill_alpha = min(max(fill_alpha, 0.0), 1.0)

    if fill_alpha > 0 and mask.any():
        tint = np.array(outline_color, dtype=np.float32).reshape(1, 1, 3)
        out[mask] = (1.0 - fill_alpha) * out[mask] + fill_alpha * tint.reshape(3)

    outline = build_outline_from_mask(mask, thickness=outline_thickness)
    if outline.any():
        out[outline] = np.array(outline_color, dtype=np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def save_patch_outline(
    pil_img,
    patch_map: np.ndarray,
    out_path: str | Path,
    threshold_rel: float = 0.65,
    blur_radius: float = 2.0,
    min_area: int = 80,
    outline_thickness: int = 2,
    fill_alpha: float = 0.12,
    outline_color: Tuple[int, int, int] = (255, 0, 0),
    out_size: Optional[Tuple[int, int]] = None,
) -> Tuple[str, np.ndarray]:
    """
    Save image with anomaly outlined.
    Returns:
      (saved_path, mask)
    """
    _ensure_out_dir(out_path)
    pil_img = _ensure_uint8_rgb(pil_img)

    if out_size is None:
        out_size = pil_img.size

    base = pil_img.resize(out_size, resample=Image.BICUBIC)

    mask = patch_map_to_anomaly_mask(
        patch_map=patch_map,
        out_size=out_size,
        threshold_rel=threshold_rel,
        blur_radius=blur_radius,
        min_area=min_area,
    )

    outlined = overlay_mask_outline_on_image(
        pil_img=base,
        mask=mask,
        outline_color=outline_color,
        outline_thickness=outline_thickness,
        fill_alpha=fill_alpha,
    )

    Image.fromarray(outlined).save(str(out_path))
    return str(Path(out_path).resolve()), mask


def save_patch_mask(
    patch_map: np.ndarray,
    out_path: str | Path,
    out_size: Tuple[int, int],
    threshold_rel: float = 0.65,
    blur_radius: float = 2.0,
    min_area: int = 80,
) -> str:
    _ensure_out_dir(out_path)

    mask = patch_map_to_anomaly_mask(
        patch_map=patch_map,
        out_size=out_size,
        threshold_rel=threshold_rel,
        blur_radius=blur_radius,
        min_area=min_area,
    )

    mask_u8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(str(out_path))
    return str(Path(out_path).resolve())