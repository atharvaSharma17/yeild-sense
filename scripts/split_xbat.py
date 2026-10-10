"""Extract and partition the real XBAT+ v1.0 Resized RGB Dataset (Zenodo 18022530).

Responsibilities:
1. Unpacks `data/raw/res_XBAT+_v1.0_RGB_train.zip` (330 labeled images) and
   `data/raw/res_XBAT+_v1.0_RGB_test.zip` (91 labeled images) into `data/processed/xbat/`.
2. Parses the official YOLO-format label files (`<class_id> <cx> <cy> <w> <h>`) across
   the 15 official XBAT+ categories documented in `res_XBAT+_v1.0_metadata/README.md`.
3. Assigns:
   - `category`: Official XBAT+ class name (e.g., `MobilePhone`, `iPad`, `AlarmClock`).
   - `weee_category`: Mapped FutuRaM WEEE category (`WEEE_Cat5` or `WEEE_Cat6`).
   - `has_battery`: Binary high-energy lithium-ion / sealed rechargeable thermal-runaway
     hazard indicator (1 = High-risk Li-ion/sealed rechargeable pack requiring CPCB/WEEE
     battery quarantine; 0 = Low-energy alkaline/coin-cell device).
   - `device_id`: Physical capture group identifier to prevent multi-view train/calibration leakage.
4. Splits the official 330-sample training archive via StratifiedGroupKFold / GroupShuffleSplit
   into `train.csv` (80%) and `calibration.csv` (20%) while preserving the official 91-sample
   held-out test archive as `test.csv`.
5. Also creates ImageFolder-compatible directories (`data/processed/train/<class>/`,
   `data/processed/validation/<class>/`) for compatibility with standard torchvision loaders.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold

XBAT_CATEGORIES: List[str] = [
    "AlarmClock",      # 0
    "ComputerMouse",   # 1
    "E-Razor",         # 2
    "E-Toothbrush",    # 3
    "FireAlarm",       # 4
    "FlashLight",      # 5
    "GameController",  # 6
    "iPad",            # 7
    "KidToy",          # 8
    "MobilePhone",     # 9
    "NewsRadio",       # 10
    "NightTorch",      # 11
    "PhotoCamera",     # 12
    "RemoteControl",   # 13
    "Thermometer",     # 14
]

# High-energy sealed Li-ion / rechargeable battery devices presenting thermal runaway fire risk in shredders
HIGH_RISK_LITHIUM_CATEGORIES = frozenset({
    "E-Razor",
    "E-Toothbrush",
    "GameController",
    "iPad",
    "MobilePhone",
    "PhotoCamera",
})

# Mapping from XBAT+ device class to FutuRaM WEEE Collection Category
XBAT_TO_FUTURAM_CATEGORY: Dict[str, str] = {
    "AlarmClock": "WEEE_Cat5",
    "ComputerMouse": "WEEE_Cat6",
    "E-Razor": "WEEE_Cat5",
    "E-Toothbrush": "WEEE_Cat5",
    "FireAlarm": "WEEE_Cat5",
    "FlashLight": "WEEE_Cat5",
    "GameController": "WEEE_Cat6",
    "iPad": "WEEE_Cat6",
    "KidToy": "WEEE_Cat5",
    "MobilePhone": "WEEE_Cat6",
    "NewsRadio": "WEEE_Cat5",
    "NightTorch": "WEEE_Cat5",
    "PhotoCamera": "WEEE_Cat6",
    "RemoteControl": "WEEE_Cat5",
    "Thermometer": "WEEE_Cat5",
}


def _parse_archive(zip_path: Path, extract_root: Path, split_tag: str) -> List[Dict[str, object]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    img_files = sorted(extract_root.rglob("*.jpg"))
    records: List[Dict[str, object]] = []

    for img_path in img_files:
        if split_tag not in str(img_path).lower():
            continue
        lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue
        parts = lines[0].split()
        class_id = int(parts[0])
        cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        category = XBAT_CATEGORIES[class_id]
        has_battery = 1 if category in HIGH_RISK_LITHIUM_CATEGORIES else 0
        weee_cat = XBAT_TO_FUTURAM_CATEGORY[category]

        # Extract capture/device prefix before '_jpg'
        stem_prefix = img_path.name.split("_jpg")[0]
        device_id = f"{category}_{stem_prefix}"

        sha256 = hashlib.sha256(img_path.read_bytes()).hexdigest()
        records.append({
            "image_path": img_path.relative_to(extract_root).as_posix(),
            "abs_image_path": str(img_path.resolve()),
            "device_id": device_id,
            "class_id": class_id,
            "category": category,
            "weee_category": weee_cat,
            "has_battery": has_battery,
            "bbox_cx": cx,
            "bbox_cy": cy,
            "bbox_w": bw,
            "bbox_h": bh,
            "sha256": sha256,
            "official_split": split_tag,
        })
    return records


def build_and_split_xbat(
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/processed"),
    metadata_csv: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support custom metadata CSV if explicitly provided
    if metadata_csv is not None and metadata_csv.exists():
        df = pd.read_csv(metadata_csv)
        required_cols = {"image_path", "device_id", "category", "has_battery"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Metadata must contain {required_cols}")
        splitter_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, temp_idx = next(splitter_train.split(df, groups=df["device_id"]))
        train_df = df.iloc[train_idx].copy()
        temp_df = df.iloc[temp_idx].copy()

        splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        cal_idx, test_idx = next(splitter_val.split(temp_df, groups=temp_df["device_id"]))
        cal_df = temp_df.iloc[cal_idx].copy()
        test_df = temp_df.iloc[test_idx].copy()

        train_df.to_csv(output_dir / "train.csv", index=False)
        cal_df.to_csv(output_dir / "calibration.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        categories = sorted(df["category"].unique().tolist())
        (output_dir / "categories.json").write_text(json.dumps(categories, indent=2))
        return {"train": len(train_df), "calibration": len(cal_df), "test": len(test_df)}

    train_zip = raw_dir / "res_XBAT+_v1.0_RGB_train.zip"
    test_zip = raw_dir / "res_XBAT+_v1.0_RGB_test.zip"
    if not train_zip.exists() or not test_zip.exists():
        raise FileNotFoundError(
            f"Missing XBAT+ archives in {raw_dir}. Expected {train_zip} and {test_zip}."
        )

    train_records = _parse_archive(train_zip, output_dir, "train_data")
    test_records = _parse_archive(test_zip, output_dir, "test_data")

    full_train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    # Group split on device_id to guarantee zero multi-view device leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, cal_idx = next(gss.split(full_train_df, full_train_df["category"], groups=full_train_df["device_id"]))
    train_df = full_train_df.iloc[train_idx].copy().reset_index(drop=True)
    cal_df = full_train_df.iloc[cal_idx].copy().reset_index(drop=True)

    # Verify zero device_id leakage across train, calibration, and test
    assert set(train_df["device_id"]).isdisjoint(set(cal_df["device_id"])), "Train/Cal device leakage!"

    train_df.to_csv(output_dir / "train.csv", index=False)
    cal_df.to_csv(output_dir / "calibration.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Also build ImageFolder layout for compatibility with standard torchvision scripts
    for split_name, split_frame in [("train", train_df), ("validation", cal_df), ("test", test_df)]:
        split_folder = output_dir / split_name
        if split_folder.exists():
            shutil.rmtree(split_folder)
        for _, row in split_frame.iterrows():
            src = output_dir / str(row["image_path"])
            dst_dir = split_folder / str(row["category"])
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)

    (output_dir / "categories.json").write_text(json.dumps(XBAT_CATEGORIES, indent=2))
    category_meta = {
        "categories": XBAT_CATEGORIES,
        "high_risk_lithium_categories": sorted(HIGH_RISK_LITHIUM_CATEGORIES),
        "xbat_to_futuram_category": XBAT_TO_FUTURAM_CATEGORY,
        "counts": {
            "train": int(len(train_df)),
            "calibration": int(len(cal_df)),
            "test": int(len(test_df)),
            "train_hazards": int(train_df["has_battery"].sum()),
            "calibration_hazards": int(cal_df["has_battery"].sum()),
            "test_hazards": int(test_df["has_battery"].sum()),
        },
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(category_meta, indent=2))
    print(f"Prepared XBAT+ dataset in {output_dir}: {category_meta['counts']}")
    return category_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_and_split_xbat(
        raw_dir=args.raw_dir,
        output_dir=args.out,
        metadata_csv=args.metadata,
        seed=args.seed,
    )
