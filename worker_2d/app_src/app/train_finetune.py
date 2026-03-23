# worker_2d/app/train_finetune.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

from .config import Settings
from .data import PFEDataManager


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# same normalization as ImageNet
_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class GoodCategoryDataset(Dataset):
    """
    Proxy task: classify MVTec 'category' using ONLY good images (split=train, label=good).
    This adapts the backbone to the domain without needing anomaly labels.
    """
    def __init__(self, dm: PFEDataManager, table: str = "mvtec_anomaly_detection"):
        df = dm.get_dataset(table=table, verbose=False)

        # keep train + good only
        df = df[(df["split"].astype(str) == "train") & (df["label"].astype(str) == "good")].copy()
        if df.empty:
            raise RuntimeError("No samples found for finetune (train & good).")

        # map categories to ids
        cats = sorted(df["category"].astype(str).unique().tolist())
        self.cat2id: Dict[str, int] = {c: i for i, c in enumerate(cats)}
        df["cat_id"] = df["category"].astype(str).map(self.cat2id)

        self.dm = dm
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img: Image.Image = self.dm.load_image(str(row["filepath"]), strict=True)
        x = _PREPROCESS(img)
        y = torch.tensor(int(row["cat_id"]), dtype=torch.long)
        return x, y


def finetune_backbone(
    config_path: str = "conf/config.yaml",
    table_name: str = "mvtec_anomaly_detection",
    output_model_dir: str = "worker_2d/models/resnet_knn_v2",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    freeze_until_layer4: bool = True,
) -> str:
    """
    Train a ResNet18 classifier on (train, good) to predict category.
    Save backbone weights (without classifier head).
    Returns path to saved backbone.
    """
    settings = Settings.from_yaml(config_path)
    dm = PFEDataManager(settings=settings)

    ds = GoodCategoryDataset(dm, table=table_name)
    n_classes = len(ds.cat2id)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # model: backbone + small head
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, n_classes)

    if freeze_until_layer4:
        # freeze everything, then unfreeze layer4 + fc
        for p in backbone.parameters():
            p.requires_grad = False
        for p in backbone.layer4.parameters():
            p.requires_grad = True
        for p in backbone.fc.parameters():
            p.requires_grad = True

    backbone = backbone.to(_DEVICE)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, backbone.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    backbone.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(_DEVICE, non_blocking=True)
            y = y.to(_DEVICE, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            logits = backbone(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

            total_loss += float(loss.item()) * x.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(x.size(0))

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        print(f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | acc={acc:.3f}")

    # save backbone WITHOUT classifier head
    # rebuild embedder-like resnet18 with fc=Identity
    # ----------------------------------------------------------
    # Save ONLY backbone weights (exclude classifier head)
    # ----------------------------------------------------------
    state = backbone.state_dict()

    # Remove classifier parameters to avoid shape mismatch
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}

    out_dir = Path(output_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "backbone_finetuned.pt"
    torch.save(state, out_path)

    # also save cat2id for traceability
    with (out_dir / "cat2id.json").open("w", encoding="utf-8") as f:
        import json
        json.dump(ds.cat2id, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved finetuned backbone (no head): {out_path}")
    return str(out_path.resolve())

    # also save cat2id for traceability
    with (out_dir / "cat2id.json").open("w", encoding="utf-8") as f:
        import json
        json.dump(ds.cat2id, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved finetuned backbone: {out_path}")
    return str(out_path.resolve())
