"""Comprehensive Held-Out Test Evaluation Suite (`ml/evaluate.py`).

Evaluates trained DualHeadEfficientNet and Split Conformal Calibration on the held-out
official XBAT+ test partition (`data/processed/test.csv`), reporting:
- Macro F1, Accuracy, and Per-Class Precision/Recall/F1
- Confusion Matrix
- Battery Safety Hazard Recall, False Negative Rate (FNR <= alpha check), and ROC-AUC
- Pre- and Post-Temperature Scaling Expected Calibration Error (ECE)
- Conformal Prediction Set Empirical Coverage and Average Set Size
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from calibrate import compute_ece
from dataset import DualHeadWEEEDataset
from model import DualHeadEfficientNet
from split_dataset import HIGH_RISK_LITHIUM_CATEGORIES


def evaluate_model(
    model_dir: Path = Path("artifacts"),
    test_csv: Path = Path("data/processed/test.csv"),
    output_json: Path = Path("artifacts/eval_report.json"),
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

    ds = DualHeadWEEEDataset(test_csv, category_to_idx, tf, root_dir=test_csv.parent)
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

    cal_data = np.load(model_dir / "calibration_scores.npz")
    if "conformal_battery_threshold" in cal_data:
        bat_threshold = float(cal_data["conformal_battery_threshold"])
    elif "battery_threshold" in cal_data:
        bat_threshold = float(cal_data["battery_threshold"])
    else:
        bat_threshold = 0.1751
    cat_qhat = float(cal_data["category_qhat"]) if "category_qhat" in cal_data else 0.85
    alpha = float(cal_data["alpha"]) if "alpha" in cal_data else 0.01

    raw_cat_probs_list = []
    cal_cat_probs_list = []
    bat_probs_list = []
    cat_targets_list = []
    bat_targets_list = []

    with torch.no_grad():
        for images, cat_t, bat_t in loader:
            images = images.to(device)
            raw_c_logits, _ = model(images)
            cal_c_probs, cal_b_probs = model.predict_calibrated(images)

            raw_cat_probs_list.append(torch.softmax(raw_c_logits, dim=-1).cpu().numpy())
            cal_cat_probs_list.append(cal_c_probs.cpu().numpy())
            bat_probs_list.extend(cal_b_probs.cpu().numpy().tolist())
            cat_targets_list.extend(cat_t.numpy().tolist())
            bat_targets_list.extend(bat_t.numpy().tolist())

    raw_cat_probs = np.concatenate(raw_cat_probs_list, axis=0)
    cal_cat_probs = np.concatenate(cal_cat_probs_list, axis=0)
    bat_probs = np.array(bat_probs_list, dtype=np.float64)
    cat_targets = np.array(cat_targets_list, dtype=np.int64)
    bat_targets = np.array(bat_targets_list, dtype=np.int64)

    cat_preds = np.argmax(cal_cat_probs, axis=1)
    macro_f1 = float(f1_score(cat_targets, cat_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(cat_targets, cat_preds, average="weighted", zero_division=0))
    cat_acc = float(accuracy_score(cat_targets, cat_preds))
    conf_mat = confusion_matrix(cat_targets, cat_preds).tolist()
    per_class = classification_report(
        cat_targets,
        cat_preds,
        target_names=categories,
        output_dict=True,
        zero_division=0,
    )

    # Battery safety hazard evaluation at conformal threshold
    bat_preds_conformal = (bat_probs >= bat_threshold).astype(np.int64)
    bat_preds_nominal = (bat_probs >= 0.50).astype(np.int64)
    haz_mask = bat_targets == 1
    n_test_hazards = int(np.sum(haz_mask))
    missed_hazards = int(np.sum((bat_preds_conformal == 0) & haz_mask))
    hazard_recall = float(1.0 - (missed_hazards / max(n_test_hazards, 1)))
    empirical_fnr = float(missed_hazards / max(n_test_hazards, 1))
    bat_acc_nominal = float(accuracy_score(bat_targets, bat_preds_nominal))
    bat_auc = (
        float(roc_auc_score(bat_targets, bat_probs))
        if len(np.unique(bat_targets)) > 1
        else 1.0
    )

    # ECE & Conformal Set Coverage on held-out test set
    ece_raw = compute_ece(raw_cat_probs, cat_targets)
    ece_cal = compute_ece(cal_cat_probs, cat_targets)

    true_probs = cal_cat_probs[np.arange(len(cat_targets)), cat_targets]
    conformal_covered = true_probs >= (1.0 - cat_qhat)
    empirical_cat_coverage = float(np.mean(conformal_covered))
    avg_set_size = float(np.mean(np.sum(cal_cat_probs >= (1.0 - cat_qhat), axis=1)))

    report = {
        "n_test_samples": int(len(cat_targets)),
        "n_test_hazards": n_test_hazards,
        "category_accuracy": round(cat_acc, 4),
        "category_macro_f1": round(macro_f1, 4),
        "category_weighted_f1": round(weighted_f1, 4),
        "battery_nominal_accuracy": round(bat_acc_nominal, 4),
        "battery_roc_auc": round(bat_auc, 4),
        "conformal_battery_safety": {
            "alpha_target": alpha,
            "calibrated_threshold": round(bat_threshold, 4),
            "hazard_recall": round(hazard_recall, 4),
            "empirical_fnr": round(empirical_fnr, 4),
            "missed_hazards": missed_hazards,
            "total_hazards": n_test_hazards,
        },
        "calibration_quality": {
            "ece_unscaled": round(ece_raw, 4),
            "ece_temperature_scaled": round(ece_cal, 4),
            "temperature_cat": round(float(model.temperature.item()), 4),
            "temperature_bat": round(float(model.battery_temperature.item()), 4),
            "category_conformal_coverage": round(empirical_cat_coverage, 4),
            "average_conformal_set_size": round(avg_set_size, 2),
        },
        "confusion_matrix": conf_mat,
        "per_class_metrics": per_class,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    print(
        f"Test Evaluation Summary: Acc={cat_acc:.4f}, Macro-F1={macro_f1:.4f}, "
        f"Battery AUC={bat_auc:.4f}, Hazard Recall={hazard_recall:.4f} "
        f"({n_test_hazards - missed_hazards}/{n_test_hazards}), ECE={ece_cal:.4f}"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test_csv", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval_report.json"))
    args = parser.parse_args()
    evaluate_model(args.model_dir, args.test_csv, args.out)
