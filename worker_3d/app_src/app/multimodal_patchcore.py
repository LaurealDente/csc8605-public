# training_3d/src/multimodal_patchcore.py
"""
Multimodal PatchCore (RGB + Depth) pour la détection d'anomalies 3D.

Adapté de la méthodologie de référence :
  - Extraction de patches multiscale (layer2 + layer3) via ResNet18
  - Banques mémoire séparées RGB / Depth
  - Fusion tardive : score = alpha_rgb * rgb + alpha_depth * depth
  - Réduction par coreset (greedy furthest-point sampling)
  - Calibration de seuils (mean+3std ou best F1)
  - Génération de heatmaps de localisation

Usage :
    from .multimodal_patchcore import MultimodalPatchCore, SamplePaths
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from .config import Settings
from .data import PFEDataManager3D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]
NORMAL_LABELS = {"good", "normal", "0", "false"}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _parse_values(raw: str) -> set[str]:
    return {v.strip().lower() for v in str(raw).split(",") if v.strip()}


def _safe_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _random_projection(
    x: np.ndarray, out_dim: int = 128, seed: int = 42
) -> np.ndarray:
    if out_dim <= 0 or out_dim >= x.shape[1]:
        return x
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal(
        (x.shape[1], out_dim), dtype=np.float32
    ) / math.sqrt(out_dim)
    return (x @ proj).astype(np.float32)


def _greedy_coreset_indices(
    x: np.ndarray, coreset_size: int, seed: int = 42
) -> np.ndarray:
    """Greedy furthest-point sampling pour la sélection de coreset."""
    n = len(x)
    if coreset_size >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
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


def build_coreset(
    bank: np.ndarray,
    target_size: int,
    seed: int = 42,
    pre_sample_size: int = 60000,
    proj_dim: int = 128,
) -> np.ndarray:
    """Réduit une banque de patches via coreset greedy."""
    bank = l2_normalize(bank)
    rng = np.random.default_rng(seed)

    if len(bank) > pre_sample_size:
        pre_idx = rng.choice(len(bank), size=pre_sample_size, replace=False)
        work = bank[pre_idx]
    else:
        pre_idx = None
        work = bank

    work_proj = _random_projection(work, out_dim=proj_dim, seed=seed)
    sel_local = _greedy_coreset_indices(
        work_proj, min(target_size, len(work_proj)), seed=seed
    )

    if pre_idx is not None:
        sel = pre_idx[sel_local]
    else:
        sel = sel_local

    return bank[sel]


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class SamplePaths:
    """Référence vers un échantillon multimodal dans la DB."""
    rgb_ref: str
    depth_ref: str
    label: Optional[str] = None
    split: Optional[str] = None
    category: Optional[str] = None
    mask_ref: Optional[str] = None


# ---------------------------------------------------------------------
# Construction de samples depuis la DB
# ---------------------------------------------------------------------

RGB_CANDIDATES = [
    "filepath", "rgb_filepath", "rgb_path",
    "image_path", "image_filepath", "path",
]
DEPTH_CANDIDATES = [
    "xyz_filepath", "xyz_path", "depth_filepath",
    "depth_path", "z_path", "z_filepath",
    "pointcloud_filepath", "pointcloud_path",
]
MASK_CANDIDATES = [
    "gt_filepath", "mask_filepath", "mask_path",
    "gt_path", "ground_truth_path",
]


def _find_first_existing(row: pd.Series, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c])
    return None


def _clean_ref(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    if s == "" or s.lower() in {"null", "none", "nan"}:
        return None
    return s


def build_samples_from_dataframe(
    df: pd.DataFrame,
    split: Optional[str] = None,
    normal_only: bool = False,
    category: Optional[str] = None,
    normal_values: str = "good,normal,0,false",
) -> List[SamplePaths]:
    """
    Construit une liste de SamplePaths à partir d'un DataFrame contenant
    les colonnes de la table mvtec_3d_anomaly_detection.
    """
    work = df.copy()

    if split and "split" in work.columns:
        work = work[work["split"].astype(str).str.lower() == split.lower()].copy()

    if category and "category" in work.columns:
        work = work[work["category"].astype(str) == str(category)].copy()

    if normal_only and "label" in work.columns:
        allowed = _parse_values(normal_values)
        work = work[work["label"].astype(str).str.lower().isin(allowed)].copy()

    out: List[SamplePaths] = []
    for _, row in work.iterrows():
        rgb_ref = _clean_ref(_find_first_existing(row, RGB_CANDIDATES))
        if rgb_ref is None:
            raise ValueError(
                f"Pas de colonne RGB trouvée. Colonnes disponibles : {list(row.index)}"
            )

        depth_ref = _clean_ref(_find_first_existing(row, DEPTH_CANDIDATES))
        if depth_ref is None:
            raise ValueError(
                f"Pas de colonne depth/xyz trouvée. Colonnes disponibles : {list(row.index)}"
            )

        mask_ref = _clean_ref(_find_first_existing(row, MASK_CANDIDATES))
        label = _clean_ref(row["label"]) if "label" in row.index else None
        split_value = _clean_ref(row["split"]) if "split" in row.index else None
        cat_value = _clean_ref(row["category"]) if "category" in row.index else None

        out.append(SamplePaths(
            rgb_ref=rgb_ref,
            depth_ref=depth_ref,
            label=label,
            split=split_value,
            category=cat_value,
            mask_ref=mask_ref,
        ))

    if not out:
        raise ValueError(
            f"Aucun sample après filtrage (split={split}, "
            f"category={category}, normal_only={normal_only})"
        )
    return out


# ---------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------

class ResNetPatchExtractor(nn.Module):
    """
    ResNet18 tronqué : extrait les feature maps de layer2 et layer3
    pour obtenir des patches multiscale.
    """
    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.eval().to(DEVICE)

    @torch.no_grad()
    def forward_multiscale(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x1 = m.layer1(x)
        x2 = m.layer2(x1)
        x3 = m.layer3(x2)
        return x2, x3  # (B, 128, 14, 14) et (B, 256, 7, 7) pour input 224


# ---------------------------------------------------------------------
# Multimodal PatchCore
# ---------------------------------------------------------------------

class MultimodalPatchCore:
    """
    PatchCore multimodal RGB + Depth.

    Pipeline :
      1. Extraction de patches multiscale (layer2+layer3 concaténés)
      2. Banques mémoire séparées pour RGB et Depth
      3. Réduction par coreset greedy
      4. Scoring par k-NN cosinus sur chaque modalité
      5. Fusion tardive : score = alpha_rgb * score_rgb + alpha_depth * score_depth
    """

    def __init__(
        self,
        image_size: int = 224,
        alpha_rgb: float = 0.5,
        alpha_depth: float = 0.5,
        n_neighbors: int = 1,
        use_late_fusion: bool = True,
        use_multiscale: bool = True,
    ) -> None:
        self.image_size = int(image_size)
        self.alpha_rgb = float(alpha_rgb)
        self.alpha_depth = float(alpha_depth)
        self.n_neighbors = int(n_neighbors)
        self.use_late_fusion = bool(use_late_fusion)
        self.use_multiscale = bool(use_multiscale)

        self.extractor_rgb = ResNetPatchExtractor()
        self.extractor_depth = ResNetPatchExtractor()

        self.rgb_bank: Optional[np.ndarray] = None
        self.depth_bank: Optional[np.ndarray] = None
        self.rgb_nn: Optional[NearestNeighbors] = None
        self.depth_nn: Optional[NearestNeighbors] = None
        self.thresholds: Dict[str, float] = {}
        self.meta: Dict[str, Any] = {}

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])

    # -----------------------------------------------------------------
    # Depth → pseudo-RGB
    # -----------------------------------------------------------------

    def _depth_to_pil(self, depth: np.ndarray) -> Image.Image:
        """Convertit un array depth (H, W) en image pseudo-RGB pour ResNet."""
        depth = np.asarray(depth, dtype=np.float32).copy()
        depth[~np.isfinite(depth)] = 0.0

        valid = np.isfinite(depth) & (depth != 0)
        if valid.any():
            vmin = float(np.quantile(depth[valid], 0.01))
            vmax = float(np.quantile(depth[valid], 0.99))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            depth = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            depth = np.zeros_like(depth, dtype=np.float32)

        depth_u8 = (depth * 255.0).astype(np.uint8)
        return Image.fromarray(depth_u8, mode="L").convert("RGB")

    # -----------------------------------------------------------------
    # Patch extraction
    # -----------------------------------------------------------------

    def _patchify(
        self,
        feat_l2: torch.Tensor,
        feat_l3: torch.Tensor,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Concatène les feature maps layer2+layer3 (multiscale),
        reshape en (H*W, C) patches L2-normalisés.
        """
        if self.use_multiscale:
            feat_l2 = F.interpolate(
                feat_l2, size=feat_l3.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            feat = torch.cat([feat_l2, feat_l3], dim=1)
        else:
            feat = feat_l3

        feat = feat.squeeze(0)
        c, h, w = feat.shape
        patches = feat.permute(1, 2, 0).reshape(h * w, c)
        patches = F.normalize(patches, dim=1)
        return patches.detach().cpu().numpy().astype(np.float32), (h, w)

    def extract_rgb_patches(
        self, rgb_img: Image.Image
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        x = self.rgb_transform(rgb_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            l2, l3 = self.extractor_rgb.forward_multiscale(x)
        return self._patchify(l2, l3)

    def extract_depth_patches(
        self, depth_arr: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        x = self.rgb_transform(self._depth_to_pil(depth_arr)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            l2, l3 = self.extractor_depth.forward_multiscale(x)
        return self._patchify(l2, l3)

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(
        self,
        samples: Sequence[SamplePaths],
        dm: PFEDataManager3D,
        max_patches_per_modality: int = 200_000,
        coreset: bool = True,
        pre_sample_size: int = 60_000,
        proj_dim: int = 128,
        seed: int = 42,
    ) -> None:
        """
        Construit les banques de patches RGB et Depth à partir
        des échantillons normaux.
        """
        rgb_all: List[np.ndarray] = []
        depth_all: List[np.ndarray] = []

        for i, s in enumerate(samples, 1):
            rgb = dm.load_image(s.rgb_ref, strict=True)
            depth = dm.load_depth_map(s.depth_ref, strict=True)

            # Extraire le canal Z si (H, W, 3)
            if depth is not None and depth.ndim == 3:
                depth = depth[..., 2]
            if depth is None:
                raise ValueError(f"Impossible de charger le depth map : {s.depth_ref}")

            rgb_p, _ = self.extract_rgb_patches(rgb)
            depth_p, _ = self.extract_depth_patches(depth)

            rgb_all.append(rgb_p)
            depth_all.append(depth_p)

            if i % 50 == 0:
                print(f"[fit] {i}/{len(samples)} samples traités")

        rgb_bank = l2_normalize(np.concatenate(rgb_all, axis=0))
        depth_bank = l2_normalize(np.concatenate(depth_all, axis=0))

        print(
            f"[fit] Banques brutes : RGB={rgb_bank.shape}, "
            f"Depth={depth_bank.shape}"
        )

        # Réduction par coreset
        if len(rgb_bank) > max_patches_per_modality:
            rgb_bank = (
                build_coreset(
                    rgb_bank, max_patches_per_modality,
                    seed, pre_sample_size, proj_dim,
                )
                if coreset
                else rgb_bank[:max_patches_per_modality]
            )

        if len(depth_bank) > max_patches_per_modality:
            depth_bank = (
                build_coreset(
                    depth_bank, max_patches_per_modality,
                    seed + 1, pre_sample_size, proj_dim,
                )
                if coreset
                else depth_bank[:max_patches_per_modality]
            )

        self.rgb_bank = l2_normalize(rgb_bank)
        self.depth_bank = l2_normalize(depth_bank)

        print(
            f"[fit] Banques finales : RGB={self.rgb_bank.shape}, "
            f"Depth={self.depth_bank.shape}"
        )

        # Construire les modèles k-NN
        self.rgb_nn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.rgb_bank)),
            metric="cosine",
        )
        self.depth_nn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.depth_bank)),
            metric="cosine",
        )

        self.rgb_nn.fit(self.rgb_bank)
        self.depth_nn.fit(self.depth_bank)

        self.meta = {
            "image_size": self.image_size,
            "alpha_rgb": self.alpha_rgb,
            "alpha_depth": self.alpha_depth,
            "n_neighbors": self.n_neighbors,
            "use_late_fusion": self.use_late_fusion,
            "use_multiscale": self.use_multiscale,
            "rgb_bank_size": int(len(self.rgb_bank)),
            "depth_bank_size": int(len(self.depth_bank)),
        }

    # -----------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------

    def _patch_scores(
        self, patches: np.ndarray, nn_model: NearestNeighbors
    ) -> np.ndarray:
        dists, _ = nn_model.kneighbors(l2_normalize(patches))
        return dists.mean(axis=1).astype(np.float32)

    def _resize_score_map(
        self, score_map: np.ndarray, target_hw: Tuple[int, int]
    ) -> np.ndarray:
        target_h, target_w = target_hw
        sm = np.asarray(score_map, dtype=np.float32)
        sm_min, sm_max = float(sm.min()), float(sm.max())

        if sm_max <= sm_min:
            return np.full((target_h, target_w), sm_min, dtype=np.float32)

        sm_norm = (sm - sm_min) / (sm_max - sm_min + 1e-12)
        sm_u8 = (sm_norm * 255.0).astype(np.uint8)
        resized_u8 = np.array(
            Image.fromarray(sm_u8).resize(
                (target_w, target_h), resample=Image.BILINEAR
            )
        )
        resized = resized_u8.astype(np.float32) / 255.0
        resized = resized * (sm_max - sm_min) + sm_min
        return resized.astype(np.float32)

    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------

    def predict(
        self,
        rgb_img: Image.Image,
        depth_arr: np.ndarray,
        upsample_to_input: bool = True,
    ) -> Dict[str, Any]:
        """
        Prédiction multimodale sur une paire (RGB, Depth).

        Retourne :
          - rgb_map, depth_map, fused_map : score maps (H, W)
          - rgb_score, depth_score, fused_score : scores image-level
        """
        if self.rgb_nn is None or self.depth_nn is None:
            raise RuntimeError(
                "Le modèle doit être entraîné (fit) ou chargé (load) avant predict()."
            )

        rgb_p, rgb_grid = self.extract_rgb_patches(rgb_img)
        depth_p, depth_grid = self.extract_depth_patches(depth_arr)

        rgb_scores = self._patch_scores(rgb_p, self.rgb_nn).reshape(rgb_grid)
        depth_scores = self._patch_scores(depth_p, self.depth_nn).reshape(depth_grid)

        fused_grid = self.alpha_rgb * rgb_scores + self.alpha_depth * depth_scores

        if upsample_to_input:
            target_hw = (rgb_img.size[1], rgb_img.size[0])  # (H, W)
            fused_map = self._resize_score_map(fused_grid, target_hw)
            rgb_map = self._resize_score_map(rgb_scores, target_hw)
            depth_map = self._resize_score_map(depth_scores, target_hw)
        else:
            fused_map = fused_grid.astype(np.float32)
            rgb_map = rgb_scores.astype(np.float32)
            depth_map = depth_scores.astype(np.float32)

        return {
            "rgb_map": rgb_map,
            "depth_map": depth_map,
            "fused_map": fused_map,
            "rgb_score": float(rgb_map.max()),
            "depth_score": float(depth_map.max()),
            "fused_score": float(fused_map.max()),
        }

    # -----------------------------------------------------------------
    # Calibration de seuils
    # -----------------------------------------------------------------

    def calibrate_thresholds(
        self,
        samples: Sequence[SamplePaths],
        dm: PFEDataManager3D,
        use_best_f1_if_labels_exist: bool = True,
    ) -> Dict[str, float]:
        """
        Calibre les seuils sur un split de validation.
        Stratégies :
          - mean + 3*std (non supervisé)
          - best F1 (si labels disponibles)
        """
        pixel_values: List[np.ndarray] = []
        image_scores: List[float] = []
        y_true_img: List[int] = []
        y_score_img: List[float] = []

        for s in samples:
            rgb = dm.load_image(s.rgb_ref, strict=True)
            depth = dm.load_depth_map(s.depth_ref, strict=True)
            if depth is not None and depth.ndim == 3:
                depth = depth[..., 2]

            pred = self.predict(rgb, depth, upsample_to_input=True)

            fmap = np.asarray(pred["fused_map"], dtype=np.float32)
            pixel_values.append(fmap.reshape(-1))
            image_scores.append(float(pred["fused_score"]))

            if s.label is not None:
                y_true_img.append(
                    0 if str(s.label).lower() in NORMAL_LABELS else 1
                )
                y_score_img.append(float(pred["fused_score"]))

        pixel_concat = np.concatenate(pixel_values)
        pixel_thr = float(pixel_concat.mean() + 3.0 * pixel_concat.std())
        image_thr = float(np.mean(image_scores) + 3.0 * np.std(image_scores))

        self.thresholds = {
            "pixel_mean_plus_3std": pixel_thr,
            "image_mean_plus_3std": image_thr,
        }

        if (
            use_best_f1_if_labels_exist
            and y_true_img
            and len(set(y_true_img)) > 1
        ):
            precisions, recalls, thrs = precision_recall_curve(
                y_true_img, y_score_img
            )
            f1s = (
                2 * precisions * recalls
                / np.clip(precisions + recalls, 1e-12, None)
            )
            if len(thrs) > 0:
                best_idx = int(np.nanargmax(f1s[:-1]))
                self.thresholds["image_best_f1"] = float(thrs[best_idx])
                self.thresholds["image_best_f1_value"] = float(f1s[best_idx])

        return self.thresholds

    # -----------------------------------------------------------------
    # Évaluation image-level
    # -----------------------------------------------------------------

    def evaluate_image_level(
        self,
        samples: Sequence[SamplePaths],
        dm: PFEDataManager3D,
        threshold_key: str = "image_mean_plus_3std",
    ) -> Dict[str, float]:
        if threshold_key not in self.thresholds:
            raise KeyError(
                f"Clé de seuil inconnue : {threshold_key}. "
                f"Disponibles : {list(self.thresholds)}"
            )

        thr = float(self.thresholds[threshold_key])
        y_true: List[int] = []
        y_pred: List[int] = []
        y_score: List[float] = []

        for s in samples:
            if s.label is None:
                continue

            rgb = dm.load_image(s.rgb_ref, strict=True)
            depth = dm.load_depth_map(s.depth_ref, strict=True)
            if depth is not None and depth.ndim == 3:
                depth = depth[..., 2]

            pred = self.predict(rgb, depth)
            score = float(pred["fused_score"])
            gt = 0 if str(s.label).lower() in NORMAL_LABELS else 1

            y_true.append(gt)
            y_pred.append(int(score >= thr))
            y_score.append(score)

        if not y_true:
            return {"n": 0}

        good_scores = [s for s, t in zip(y_score, y_true) if t == 0]
        bad_scores = [s for s, t in zip(y_score, y_true) if t == 1]

        return {
            "n": len(y_true),
            "threshold": thr,
            "f1": float(f1_score(y_true, y_pred)),
            "mean_score_good": float(np.mean(good_scores)) if good_scores else 0.0,
            "mean_score_bad": float(np.mean(bad_scores)) if bad_scores else 0.0,
        }

    # -----------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------

    def save(self, model_dir: str | Path) -> None:
        """Sauvegarde les banques de patches et les métadonnées."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.rgb_bank is None or self.depth_bank is None:
            raise RuntimeError("Rien à sauvegarder : les banques sont vides.")

        np.save(
            model_dir / "rgb_patch_bank.npy",
            self.rgb_bank.astype(np.float32),
        )
        np.save(
            model_dir / "depth_patch_bank.npy",
            self.depth_bank.astype(np.float32),
        )

        _safe_json_dump(model_dir / "mm_patchcore_meta.json", self.meta)
        _safe_json_dump(
            model_dir / "mm_patchcore_thresholds.json", self.thresholds
        )

        print(
            f"✅ Modèle MM-PatchCore sauvegardé dans {model_dir}\n"
            f"   RGB bank : {self.rgb_bank.shape}\n"
            f"   Depth bank : {self.depth_bank.shape}"
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> "MultimodalPatchCore":
        """Charge un modèle MM-PatchCore depuis le disque."""
        model_dir = Path(model_dir)

        with (model_dir / "mm_patchcore_meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        model = cls(
            image_size=int(meta["image_size"]),
            alpha_rgb=float(meta["alpha_rgb"]),
            alpha_depth=float(meta["alpha_depth"]),
            n_neighbors=int(meta["n_neighbors"]),
            use_late_fusion=bool(meta["use_late_fusion"]),
            use_multiscale=bool(meta["use_multiscale"]),
        )

        model.rgb_bank = l2_normalize(
            np.load(model_dir / "rgb_patch_bank.npy").astype(np.float32)
        )
        model.depth_bank = l2_normalize(
            np.load(model_dir / "depth_patch_bank.npy").astype(np.float32)
        )

        model.rgb_nn = NearestNeighbors(
            n_neighbors=min(model.n_neighbors, len(model.rgb_bank)),
            metric="cosine",
        )
        model.depth_nn = NearestNeighbors(
            n_neighbors=min(model.n_neighbors, len(model.depth_bank)),
            metric="cosine",
        )

        model.rgb_nn.fit(model.rgb_bank)
        model.depth_nn.fit(model.depth_bank)

        thresholds_path = model_dir / "mm_patchcore_thresholds.json"
        if thresholds_path.exists():
            with thresholds_path.open("r", encoding="utf-8") as f:
                model.thresholds = json.load(f)

        model.meta = meta

        print(
            f"✅ Modèle MM-PatchCore chargé depuis {model_dir}\n"
            f"   RGB bank : {model.rgb_bank.shape}\n"
            f"   Depth bank : {model.depth_bank.shape}"
        )
        return model


# ---------------------------------------------------------------------
# Utilitaires de heatmap
# ---------------------------------------------------------------------

def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_min, x_max = float(x.min()), float(x.max())
    if abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    """Convertit une heatmap (H, W) normalisée en RGB uint8 (colormap jet)."""
    import matplotlib.cm as cm

    heatmap = normalize_map(heatmap)
    colored = cm.get_cmap("jet")(heatmap)[..., :3]
    return (colored * 255.0).clip(0, 255).astype(np.uint8)


def overlay_heatmap_on_image(
    pil_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Superpose une heatmap sur l'image RGB. Retourne un array RGB uint8."""
    pil_img = pil_img.convert("RGB")
    out_size = pil_img.size  # (W, H)

    base = pil_img.resize(out_size, resample=Image.BICUBIC)
    base_np = np.asarray(base).astype(np.float32)

    hm_norm = normalize_map(heatmap)
    hm_u8 = (hm_norm * 255.0).clip(0, 255).astype(np.uint8)
    hm_resized = np.array(
        Image.fromarray(hm_u8, mode="L").resize(out_size, resample=Image.BICUBIC)
    ).astype(np.float32) / 255.0

    heat_rgb = heatmap_to_rgb(hm_resized).astype(np.float32)

    alpha = min(max(alpha, 0.0), 1.0)
    overlay = (1.0 - alpha) * base_np + alpha * heat_rgb
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_heatmap(
    score_map: np.ndarray,
    out_path: str | Path,
) -> str:
    """Sauvegarde une heatmap en PNG coloré."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heat_rgb = heatmap_to_rgb(score_map)
    Image.fromarray(heat_rgb).save(str(out_path))
    return str(out_path.resolve())


def save_overlay(
    pil_img: Image.Image,
    score_map: np.ndarray,
    out_path: str | Path,
    alpha: float = 0.45,
) -> str:
    """Sauvegarde un overlay heatmap+image en PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap_on_image(pil_img, score_map, alpha=alpha)
    Image.fromarray(overlay).save(str(out_path))
    return str(out_path.resolve())
