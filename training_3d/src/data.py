# training_3d/src/data.py
"""
Data management pour le pipeline 3D.

Différences avec la version 2D :
  - La table mvtec_3d_anomaly_detection a 3 colonnes de fichiers :
    filepath (RGB), xyz_filepath (depth .tiff), gt_filepath (masque)
  - PFEDataManager3D charge les depth maps en plus des images RGB
  - MVTec3DDataset expose (rgb_tensor, depth_tensor, label) ou juste (rgb_tensor, label)

Changements V2 :
  - load_depth_map supporte HTTP + cache (comme load_image)
  - get_dataset retourne aussi xyz_filepath et gt_filepath
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .config import Settings
from .db import get_engine


@dataclass
class PFEDataManager3D:
    """Gestionnaire de données pour le pipeline 3D."""

    settings: Settings
    engine: Optional[Engine] = None

    def __post_init__(self) -> None:
        self.cache_dir = Path(os.getenv("PFE_IMG_CACHE", "/tmp/pfe_img_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

        if self.engine is None:
            self.engine = get_engine(self.settings)

    def _qualify_table(self, table: str) -> str:
        t = str(table).strip()
        if "." in t:
            return t
        schema = getattr(self.settings, "db_schema", None)
        if schema:
            return f"{schema}.{t}"
        return t

    def _normalize_db_path(self, filepath: str) -> str:
        fp = str(filepath).replace("\\", "/").strip()
        if not fp:
            return fp
        if fp.startswith("/"):
            fp = fp.lstrip("/")
        if fp.startswith("images_storage/"):
            fp = fp[len("images_storage/"):]
        root_base = os.path.basename(str(self.settings.images_root).rstrip("/"))
        if root_base and fp.startswith(root_base + "/"):
            fp = fp[len(root_base) + 1:]
        return fp.lstrip("/")

    def _url_for(self, filepath: str) -> str:
        rel = self._normalize_db_path(filepath)
        return f"{self.settings.images_url.rstrip('/')}/{rel.lstrip('/')}"

    def _local_for(self, filepath: str) -> str:
        rel = self._normalize_db_path(filepath)
        return str(Path(self.settings.images_root) / rel)

    def load_image(
        self,
        filepath: str,
        strict: bool = True,
        timeout_s: float = 10.0,
        retries: int = 2,
        backoff_s: float = 0.4,
    ) -> Image.Image:
        """Load an RGB image (.png) via cache → HTTP → local fallback."""
        rel = self._normalize_db_path(filepath).lstrip("/")
        cache_path = self.cache_dir / rel

        try:
            if cache_path.exists():
                return Image.open(cache_path).convert("RGB")
        except Exception:
            pass

        url = self._url_for(filepath)
        last_exc: Optional[Exception] = None

        for i in range(max(0, int(retries)) + 1):
            try:
                r = self._session.get(url, timeout=timeout_s)
                r.raise_for_status()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(r.content)
                return Image.open(BytesIO(r.content)).convert("RGB")
            except Exception as e:
                last_exc = e
                if i < retries:
                    time.sleep(backoff_s * (2 ** i))

        local_path = self._local_for(filepath)
        try:
            if os.path.exists(local_path):
                return Image.open(local_path).convert("RGB")
        except Exception as e:
            last_exc = e

        if strict:
            raise RuntimeError(
                f"Cannot load image. url={url} filepath={filepath} "
                f"local_path={local_path} last_error={repr(last_exc)}"
            )
        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    def load_depth_map(
        self,
        xyz_filepath: str,
        strict: bool = True,
        timeout_s: float = 30.0,
        retries: int = 2,
        backoff_s: float = 0.4,
    ) -> Optional[np.ndarray]:
        """
        Charge un depth map .tiff et retourne un array numpy (H, W, 3).
        Le fichier .tiff MVTec 3D-AD contient 3 canaux (x, y, z).

        Résolution : cache → HTTP → local fallback (comme load_image).
        """
        if not xyz_filepath:
            return None

        import tifffile

        rel = self._normalize_db_path(xyz_filepath).lstrip("/")
        cache_path = self.cache_dir / rel

        # 0) Cache
        try:
            if cache_path.exists():
                data = tifffile.imread(str(cache_path))
                return data.astype(np.float32)
        except Exception:
            pass

        # 1) HTTP
        url = self._url_for(xyz_filepath)
        last_exc: Optional[Exception] = None

        for i in range(max(0, int(retries)) + 1):
            try:
                r = self._session.get(url, timeout=timeout_s)
                r.raise_for_status()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(r.content)
                data = tifffile.imread(str(cache_path))
                return data.astype(np.float32)
            except Exception as e:
                last_exc = e
                if i < retries:
                    time.sleep(backoff_s * (2 ** i))

        # 2) Local fallback
        local_path = self._local_for(xyz_filepath)
        try:
            if os.path.exists(local_path):
                data = tifffile.imread(local_path)
                return data.astype(np.float32)
        except Exception as e:
            last_exc = e

        if strict:
            raise RuntimeError(
                f"Cannot load depth map. url={url} "
                f"xyz_filepath={xyz_filepath} "
                f"local_path={local_path} last_error={repr(last_exc)}"
            )
        return None

    def depth_to_pseudo_rgb(self, depth: np.ndarray) -> Image.Image:
        """
        Convertit un depth map (H, W, 3) en image pseudo-RGB pour ResNet.
        Utilise le canal Z normalisé et répliqué sur 3 canaux.
        """
        if depth is None:
            return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        # Extraire le canal Z (profondeur)
        z_channel = depth[:, :, 2]

        # Masquer les NaN/Inf
        valid_mask = np.isfinite(z_channel)
        if valid_mask.any():
            z_min = z_channel[valid_mask].min()
            z_max = z_channel[valid_mask].max()
            if z_max > z_min:
                z_norm = (z_channel - z_min) / (z_max - z_min)
            else:
                z_norm = np.zeros_like(z_channel)
        else:
            z_norm = np.zeros_like(z_channel)

        z_norm = np.nan_to_num(z_norm, nan=0.0, posinf=1.0, neginf=0.0)

        # Convertir en uint8 et répliquer en 3 canaux
        z_uint8 = (z_norm * 255).clip(0, 255).astype(np.uint8)
        rgb = np.stack([z_uint8, z_uint8, z_uint8], axis=-1)

        return Image.fromarray(rgb, mode="RGB")

    def get_dataset(
        self,
        table: str = "mvtec_3d_anomaly_detection",
        limit: Optional[int] = None,
        verbose: bool = True,
        raise_on_error: bool = True,
    ) -> pd.DataFrame:
        """
        Charge les métadonnées du dataset 3D depuis la DB.
        Retourne : category, split, label, filepath, xyz_filepath, gt_filepath
        """
        if self.engine is None:
            raise RuntimeError("DB engine is None.")

        qt = self._qualify_table(table)
        base_sql = f"""
            SELECT category, split, label, filepath, xyz_filepath, gt_filepath
            FROM {qt}
            ORDER BY filepath ASC
        """
        if limit is not None:
            base_sql += " LIMIT :limit"

        try:
            if limit is None:
                df = pd.read_sql_query(text(base_sql), self.engine)
            else:
                df = pd.read_sql_query(
                    text(base_sql), self.engine,
                    params={"limit": int(limit)},
                )

            for col in ["category", "split", "label"]:
                if col in df.columns:
                    df[col] = df[col].astype("category")

            if verbose:
                print(f"✅ {len(df)} rows loaded from '{qt}'")
            return df

        except Exception as e:
            msg = f"❌ SQL error in get_dataset(table={qt}): {e}"
            if verbose:
                print(msg)
            if raise_on_error:
                raise
            return pd.DataFrame()


class MVTec3DDataset(Dataset):
    """
    Dataset PyTorch pour MVTec 3D-AD.

    V1 simple : charge uniquement les images RGB.
    Le depth map sera utilisé dans une version future.
    """

    def __init__(
        self,
        dm: PFEDataManager3D,
        table_name: str = "mvtec_3d_anomaly_detection",
        split: str = "train",
        only_good: bool = False,
        image_size: int = 256,
        limit: Optional[int] = None,
        transform=None,
        use_depth: bool = False,
    ):
        self.dm = dm
        self.table_name = table_name
        self.split = split
        self.only_good = only_good
        self.use_depth = use_depth

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        df = dm.get_dataset(
            table=table_name,
            verbose=False,
            raise_on_error=True,
        )

        if df.empty:
            raise RuntimeError(
                f"Empty dataset for table={table_name}."
            )

        if "split" in df.columns:
            df = df[
                df["split"].astype(str).str.strip().str.lower()
                == str(split).strip().lower()
            ]

        if only_good and "label" in df.columns:
            df = df[
                df["label"].astype(str).str.strip().str.lower() == "good"
            ]

        if limit is not None:
            df = df.head(int(limit))

        df = df.reset_index(drop=True)

        if df.empty:
            print(
                f"⚠️ MVTec3DDataset empty "
                f"(table={table_name}, split={split}, only_good={only_good})"
            )

        self.data = df

    def __len__(self) -> int:
        return len(self.data)

    def get_meta(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        return {
            "category": str(row["category"]),
            "split": str(row["split"]),
            "label": str(row["label"]),
            "filepath": str(row["filepath"]),
            "xyz_filepath": str(row.get("xyz_filepath", "")),
        }

    def get_raw_item(self, idx: int):
        row = self.data.iloc[idx]
        pil_img = self.dm.load_image(str(row["filepath"]), strict=True)
        y = 0 if str(row["label"]).strip().lower() == "good" else 1
        return pil_img, y

    def __getitem__(self, idx: int):
        pil_img, y = self.get_raw_item(idx)

        if self.transform is not None:
            x = self.transform(pil_img)
        else:
            x = pil_img

        return x, torch.tensor(y, dtype=torch.float32)
