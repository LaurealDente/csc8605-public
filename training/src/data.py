# worker_2d/app/data.py

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
class PFEDataManager:
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
        """
        DB filepath -> relative path

        Example:
          images_storage/mvtec_ad_2/can/test/good/001.png
          -> mvtec_ad_2/can/test/good/001.png
        """
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
        """
        Load one image using:
        0) local cache
        1) HTTP
        2) local filesystem fallback

        If strict=False and everything fails, returns a black image.
        """
        rel = self._normalize_db_path(filepath).lstrip("/")
        cache_path = self.cache_dir / rel

        # 0) cache
        try:
            if cache_path.exists():
                return Image.open(cache_path).convert("RGB")
        except Exception:
            pass

        # 1) HTTP
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

        # 2) local fallback
        local_path = self._local_for(filepath)
        try:
            if os.path.exists(local_path):
                return Image.open(local_path).convert("RGB")
        except Exception as e:
            last_exc = e

        if strict:
            raise RuntimeError(
                f"Cannot load image. "
                f"url={url} filepath={filepath} rel={rel} "
                f"local_path={local_path} last_error={repr(last_exc)}"
            )

        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    def get_dataset(
        self,
        table: str = "mvtec_anomaly_detection",
        limit: Optional[int] = None,
        load_images: bool = False,
        verbose: bool = True,
        raise_on_error: bool = True,
    ) -> pd.DataFrame:
        """
        Load dataset metadata from DB.

        Returns columns:
          - category
          - split
          - label
          - filepath
        and optionally:
          - image
        """
        if self.engine is None:
            raise RuntimeError("DB engine is None. Cannot query dataset.")

        qt = self._qualify_table(table)

        base_sql = f"""
            SELECT category, split, label, filepath
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
                    text(base_sql),
                    self.engine,
                    params={"limit": int(limit)},
                )

            for col in ["category", "split", "label"]:
                if col in df.columns:
                    df[col] = df[col].astype("category")

            if verbose:
                print(f"✅ {len(df)} rows loaded from '{qt}'")

            if load_images and not df.empty:
                if verbose:
                    print("📦 Loading images into memory...")
                df["image"] = df["filepath"].astype(str).apply(
                    lambda p: self.load_image(p, strict=True)
                )

            return df

        except Exception as e:
            msg = f"❌ SQL error in get_dataset(table={qt}): {e}"
            if verbose:
                print(msg)
            if raise_on_error:
                raise
            return pd.DataFrame()


class MVTecDataset(Dataset):
    def __init__(
        self,
        dm: PFEDataManager,
        table_name: str = "mvtec_anomaly_detection",
        split: str = "train",
        only_good: bool = False,
        image_size: int = 256,
        limit: Optional[int] = None,
        transform=None,
    ):
        self.dm = dm
        self.table_name = table_name
        self.split = split
        self.only_good = only_good

        # allow external override, otherwise use default transform
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
                f"Empty dataset for table={table_name}. "
                "Check DB connectivity/schema."
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
                f"⚠️ MVTecDataset empty "
                f"(table={table_name}, split={split}, only_good={only_good})"
            )

        # main storage used everywhere
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
        }

    def get_raw_item(self, idx: int):
        """
        Return:
          - PIL image (not transformed)
          - binary label: 0 if good, 1 otherwise
        """
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