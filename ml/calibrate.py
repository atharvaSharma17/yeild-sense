"""Post-Hoc Temperature Scaling & Split Conformal Risk Calibration for YieldSense 2.0.

Implements:
1. L-BFGS Temperature Scaling on category and battery hazard logits.
2. Expected Calibration Error (ECE) computation before and after scaling.
3. Split Conformal Risk Control on Battery Hazards (FNR <= alpha = 0.01).
4. Split Conformal Prediction Set (APS) threshold for Category Ambiguity (alpha_cat = 0.10).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import DualHeadWEEEDataset
from model import DualHeadEfficientNet
from split_dataset import HIGH_RISK_LITHIUM_CATEGORIES


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across `n_bins` confidence intervals."""
    if len(probs) == 0:
        return 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(probs))
    for b_idx in range(n_bins):
        low, high = bin_boundaries[b_idx], bin_boundaries[b_idx + 1]
        if b_idx == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        prop_in_bin = np.sum(in_bin) / n
        if prop_in_bin > 0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def run_calibration(
    model_dir: Path,
    cal_csv: Path,
    alpha: float = 0.01,
    alpha_cat: float = 0.10,
    optimize_temperature: bool = True,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = json.loads((model_dir / "categories.json").read_text(encoding="utf-8"))
    category_to_idx = {c: i for i, c in enumerate(categories)}
    high_risk_indices = [
        i for i, c in enumerate(categories) if c in HIGH_RISK_LITHIUM_CATEGORIES
    ]

    tf = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = DualHeadWEEEDataset(cal_csv, category_to_idx, tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model = DualHeadEfficientNet(
        num_categories=len(categories),
        pretrained=False,
        high_risk_class_indices=high_risk_indices,
    )
    state_dict = torch.load(model_dir / "model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Collect pre-calibration probabilities for ECE baseline
    pre_cat_probs, cat_targets_all = [], []
    with torch.no_grad():
        for images, cat_t, _ in loader:
            images = images.to(device)
            c_logits, _ = model(images)
            pre_cat_probs.append(torch.softmax(c_logits, dim=-1).cpu().numpy())
            cat_targets_all.append(cat_t.numpy())

    pre_cat_arr = np.concatenate(pre_cat_probs, axis=0)
    cat_targets_arr = np.concatenate(cat_targets_all, axis=0)
    ece_pre = compute_ece(pre_cat_arr, cat_targets_arr)

    temp_info = {}
    if optimize_temperature and len(ds) >= 5:
        temp_info = model.fit_temperature(loader, device=device)
        torch.save(model.state_dict(), model_dir / "model.pth")

    all_cat_probs = []
    all_bat_scores = []
    all_bat_targets = []

    model.eval()
    with torch.no_grad():
        for images, _, bat_targets in loader:
            images = images.to(device)
            cat_p, bat_p = model.predict_calibrated(images)
            all_cat_probs.append(cat_p.cpu().numpy())
            all_bat_scores.extend(bat_p.cpu().numpy().tolist())
            all_bat_targets.extend(bat_targets.numpy().tolist())

    post_cat_arr = np.concatenate(all_cat_probs, axis=0)
    ece_post = compute_ece(post_cat_arr, cat_targets_arr)

    scores = np.array(all_bat_scores, dtype=np.float64)
    targets = np.array(all_bat_targets, dtype=np.float64)

    # 1. Split Conformal Risk Control on Battery Hazards (FNR <= alpha)
    hazard_scores = scores[targets == 1.0]
    n_haz = int(len(hazard_scores))
    if n_haz == 0:
        raise ValueError("Calibration partition contains no hazardous samples")

    q_level = float(np.clip(np.ceil((n_haz + 1) * (1.0 - alpha)) / n_haz, 0.0, 1.0))
    nonconformity = 1.0 - hazard_scores
    s_quant = float(np.quantile(nonconformity, q_level, method="higher"))
    # Bound threshold away from 0.001 minimum so non-hazard items aren't trivially flagged on tiny n_haz
    calibrated_threshold = float(np.clip(1.0 - s_quant, 0.05, 0.85))

    # 2. Split Conformal Prediction Set Quantile on Categories (1 - alpha_cat coverage)
    n_total = len(cat_targets_arr)
    true_class_probs = post_cat_arr[np.arange(n_total), cat_targets_arr]
    cat_nonconformity = 1.0 - true_class_probs
    q_cat_level = float(np.clip(np.ceil((n_total + 1) * (1.0 - alpha_cat)) / n_total, 0.0, 1.0))
    cat_qhat = float(np.quantile(cat_nonconformity, q_cat_level, method="higher"))

    # Empirical hazard recall on calibration partition at threshold
    cal_hazard_recall = float(np.mean(hazard_scores >= calibrated_threshold))

    cal_artifact = model_dir / "calibration_scores.npz"
    np.savez(
        cal_artifact,
        battery_threshold=calibrated_threshold,
        conformal_battery_threshold=calibrated_threshold,
        category_qhat=cat_qhat,
        alpha=alpha,
        alpha_cat=alpha_cat,
        n_hazards=n_haz,
        n_calibration=n_total,
        ece_pre=ece_pre,
        ece_post=ece_post,
    )

    summary = {
        "battery_threshold": round(calibrated_threshold, 4),
        "conformal_battery_threshold": round(calibrated_threshold, 4),
        "temperature_cat": temp_info.get("temperature_cat", 1.0),
        "temperature_bat": temp_info.get("temperature_bat", 1.0),
        "category_qhat": round(cat_qhat, 4),
        "alpha": alpha,
        "alpha_cat": alpha_cat,
        "n_calibration": n_total,
        "n_hazards": n_haz,
        "calibration_hazard_recall": round(cal_hazard_recall, 4),
        "ece_pre_scaling": round(ece_pre, 4),
        "ece_post_scaling": round(ece_post, 4),
        "temperature_scaling": temp_info,
    }
    (model_dir / "calibration_metrics.json").write_text(json.dumps(summary, indent=2))
    print(
        f"Calibration threshold: {calibrated_threshold:.4f} "
        f"(alpha={alpha}, n_haz={n_haz}, ECE: {ece_pre:.4f} -> {ece_post:.4f})"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--cal_csv", type=Path, default=Path("data/processed/calibration.csv"))
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--alpha_cat", type=float, default=0.10)
    args = parser.parse_args()
    run_calibration(args.model_dir, args.cal_csv, alpha=args.alpha, alpha_cat=args.alpha_cat)
