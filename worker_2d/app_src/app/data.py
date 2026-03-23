# worker_2d/app/data.py

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
from typing import Optional

import pandas as pd
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .config import Settings
from .db import get_engine


# ==========================================================
# Data manager (DB + images)
# ==========================================================

@dataclass
class PFEDataManager:
    settings: Settings
    engine: Optional[Engine] = None

    def __post_init__(self) -> None:
        self.cache_dir = Path(os.getenv("PFE_IMG_CACHE", "/tmp/pfe_img_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        if self.engine is None:
            self.engine = get_engine(self.settings)

    # ==========================================================
    # PATH NORMALIZATION
    # ==========================================================

    def _normalize_db_path(self, filepath: str) -> str:
        """
        DB:   images_storage/mvtec_anomaly_detection/...
        WANT: mvtec_anomaly_detection/...
        """
        fp = str(filepath).replace("\\", "/").strip()
        if not fp:
            return fp

        # absolute path -> keep
        if fp.startswith("/"):
            return fp

        # remove explicit images_storage prefix
        if fp.startswith("images_storage/"):
            fp = fp[len("images_storage/"):]

        # remove basename(images_root) if present
        root_base = os.path.basename(self.settings.images_root.rstrip("/"))
        if fp.startswith(root_base + "/"):
            fp = fp[len(root_base) + 1:]

        return fp.lstrip("/")

    def _url_for(self, filepath: str) -> str:
        rel = self._normalize_db_path(filepath)
        return f"{self.settings.images_url.rstrip('/')}/{rel}"

    def _local_for(self, filepath: str) -> str:
        """
        Local path fallback (rarely used on Arcadia).
        """
        fp = self._normalize_db_path(filepath)
        return str(Path(self.settings.images_root) / fp)

    # ==========================================================
    # IMAGE LOADING
    # ==========================================================

    def load_image(self, filepath: str, strict: bool = True, timeout_s: int = 10) -> Image.Image:
        rel = self._normalize_db_path(filepath).lstrip("/")
        cache_path = self.cache_dir / rel

        # 0) Cache disque
        try:
            if cache_path.exists():
                return Image.open(cache_path).convert("RGB")
        except Exception:
            pass

        url = self._url_for(filepath)

        # 1) Réseau -> puis save cache
        try:
            r = self._session.get(url, timeout=timeout_s)
            r.raise_for_status()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(r.content)
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            pass

        # 2) Local fallback
        try:
            local_path = self._local_for(filepath)
            if os.path.exists(local_path):
                return Image.open(local_path).convert("RGB")
        except Exception:
            pass

        if strict:
            raise RuntimeError(f"Cannot load image. url={url} filepath={filepath}")

        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    # ==========================================================
    # DB ACCESS
    # ==========================================================

    def get_dataset(
        self,
        table: str = "mvtec_anomaly_detection",
        limit: Optional[int] = None,
        load_images: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:

        assert self.engine is not None

        base_sql = f"""
            SELECT category, split, label, filepath
            FROM {table}
            ORDER BY filepath ASC
        """

        if limit is not None:
            base_sql += " LIMIT :limit"

        try:
            if limit is None:
                df = pd.read_sql_query(text(base_sql), self.engine)
            else:
                df = pd.read_sql_query(
                    text(base_sql),
                    self.engine,
                    params={"limit": int(limit)},
                )

            for col in ["category", "split", "label"]:
                if col in df.columns:
                    df[col] = df[col].astype("category")

            if verbose:
                print(f"✅ {len(df)} rows loaded from '{table}'")

            if load_images and not df.empty:
                df["image"] = df["filepath"].apply(
                    lambda p: self.load_image(p, strict=True)
                )

            return df

        except Exception as e:
            if verbose:
                print(f"❌ SQL error: {e}")
            return pd.DataFrame()

    # ==========================================================
    # DEBUG / VISUALIZATION
    # ==========================================================

    def get_summary(self, df: pd.DataFrame) -> None:
        if df.empty:
            print("⚠️ Empty DataFrame.")
            return

        print(f"Total images: {len(df)}")

        if "split" in df.columns:
            print(df["split"].value_counts())

        if "category" in df.columns:
            print(df["category"].value_counts().head(10))

    def show_gallery(
        self,
        df: pd.DataFrame,
        n: int = 5,
        category: Optional[str] = None,
        label: Optional[str] = None,
        seed: int = 42,
    ) -> None:

        if df.empty:
            print("⚠️ Empty DataFrame.")
            return

        filtered = df.copy()

        if category:
            filtered = filtered[filtered["category"] == category]

        if label:
            filtered = filtered[filtered["label"] == label]

        if filtered.empty:
            print("⚠️ No images found.")
            return

        sample = filtered.sample(
            n=min(n, len(filtered)),
            random_state=seed
        )

        plt.figure(figsize=(15, 4))
        for i, (_, row) in enumerate(sample.iterrows()):
            img = self.load_image(str(row["filepath"]), strict=False)

            plt.subplot(1, len(sample), i + 1)
            plt.imshow(img)

            lab = row.get("label", "?")
            color = "green" if str(lab) == "good" else "red"
            plt.title(f"{row.get('category','?')}\n{lab}",
                      fontsize=9,
                      color=color)
            plt.axis("off")

        plt.tight_layout()
        plt.show()


# ==========================================================
# PyTorch Dataset
# ==========================================================

class MVTecDataset(Dataset):

    def __init__(
        self,
        dm: PFEDataManager,
        table_name: str = "mvtec_anomaly_detection",
        split: str = "train",
        only_good: bool = False,
        image_size: int = 256,
    ):
        self.dm = dm

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        df = dm.get_dataset(table=table_name, verbose=False)

        if "split" in df.columns:
            df = df[df["split"] == split]

        if only_good and "label" in df.columns:
            df = df[df["label"] == "good"]

        self.data = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        img = self.dm.load_image(str(row["filepath"]), strict=True)
        img_tensor = self.transform(img)

        label_val = 0.0 if str(row["label"]) == "good" else 1.0
        return img_tensor, torch.tensor(label_val, dtype=torch.float32)
