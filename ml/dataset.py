"""PyTorch Dataset for Dual-Head WEEE Category & Battery Hazard Perception."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset


class DualHeadWEEEDataset(Dataset):
    """Loads RGB WEEE intake photographs with category and battery hazard targets.

    Resolves relative `image_path` entries against the manifest's parent directory
    so the exact same manifest works locally (`data/processed/`) and inside an
    Amazon SageMaker container (`/opt/ml/input/data/train/`).
    """

    def __init__(
        self,
        manifest_csv: Path,
        category_to_idx: Dict[str, int],
        transform=None,
        root_dir: Optional[Path] = None,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.root_dir = Path(root_dir) if root_dir is not None else self.manifest_csv.parent
        self.df = pd.read_csv(self.manifest_csv)
        self.category_to_idx = category_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_absolute() and p.exists():
            return p
        candidate = self.root_dir / p
        if candidate.exists():
            return candidate
        # Fallback: check workspace root relative path
        workspace_candidate = Path.cwd() / p
        if workspace_candidate.exists():
            return workspace_candidate
        return candidate

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(str(row["image_path"]))
        image = Image.open(img_path).convert("RGB")
        cat_target = int(self.category_to_idx[str(row["category"])])
        bat_target = float(row["has_battery"])

        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            torch.tensor(cat_target, dtype=torch.long),
            torch.tensor(bat_target, dtype=torch.float32),
        )
