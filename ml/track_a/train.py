"""Track A Training & Mixed-Batch Validation Pipeline (`ml/track_a/train.py`).

Trains the ParticleMaterialClassifier and ParticleMassRegressor on 1,976 paired
samples from TU Darmstadt RGB-MFCO DS2, then performs zero-shot mixed-batch
validation across 2,789 physical particles in DS3 (MS1, MS2, MS3).
"""

from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time
import zipfile

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from ml.track_a.composition_eval import bootstrap_batch_scenarios, compute_composition_mae
from ml.track_a.features import extract_particle_classifier_features, extract_particle_features
from ml.track_a.mass_regression import (
    PHYSICAL_CLASSES,
    ParticleMassRegressor,
    ParticleMaterialClassifier,
    reconcile_batch_composition,
)


def load_ds2_data(zip_path: Path):
    """Load and pair 1,976 DS2 particle images with their ground-truth physical masses."""
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_bytes = z.read("MAS/DS2/Masses_DS2_Particles_extracted.csv")
        csv_data = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
        ts_to_mass = {row["timestamp"]: float(row["mass"]) for row in csv_data}

        ds2_imgs = [f for f in z.namelist() if f.startswith("IMG/DS2/") and f.endswith(".png")]
        ts_to_path = {}
        for p in ds2_imgs:
            fname = p.split("/")[-1]
            ts = fname.rsplit("_", 1)[-1].replace(".png", "")
            ts_to_path[ts] = p

        matched_ts = [ts for ts in ts_to_mass if ts in ts_to_path]
        print(f"Loaded {len(matched_ts):,} paired DS2 particles from {zip_path.name}")

        X_clf = []
        X_reg = []
        y_class = []
        y_mass = []
        timestamps = []

        for ts in matched_ts:
            p = ts_to_path[ts]
            mat = p.split("/")[2]  # FEM, NFM, PCB, PLA
            img_bytes = z.read(p)
            with Image.open(io.BytesIO(img_bytes)) as img:
                c_feat = extract_particle_classifier_features(img)
                r_feat = extract_particle_features(img, predicted_class=mat)

            c_vec = [
                c_feat["area_px"],
                c_feat["bbox_area_px"],
                c_feat["aspect_ratio"],
                c_feat["fill_ratio"],
                c_feat["fg_mean_r"],
                c_feat["fg_mean_g"],
                c_feat["fg_mean_b"],
                c_feat["fg_std_r"],
                c_feat["fg_std_g"],
                c_feat["fg_std_b"],
                c_feat["mean_r"],
                c_feat["mean_g"],
                c_feat["mean_b"],
                c_feat["std_intensity"],
                c_feat["green_red_ratio"],
                c_feat["red_blue_ratio"],
            ]

            r_vec = [
                r_feat["area_px"],
                r_feat["bbox_area_px"],
                r_feat["aspect_ratio"],
                r_feat["fill_ratio"],
                r_feat["mean_r"],
                r_feat["mean_g"],
                r_feat["mean_b"],
                r_feat["std_intensity"],
                r_feat["is_fem"],
                r_feat["is_nfm"],
                r_feat["is_pcb"],
                r_feat["is_pla"],
            ]

            X_clf.append(c_vec)
            X_reg.append(r_vec)
            y_class.append(mat)
            y_mass.append(ts_to_mass[ts])
            timestamps.append(ts)

    return (
        np.array(X_clf, dtype=np.float64),
        np.array(X_reg, dtype=np.float64),
        y_class,
        np.array(y_mass, dtype=np.float64),
        timestamps,
    )


def evaluate_ds2_cv(X_clf, X_reg, y_class, y_mass, seed: int = 42):
    """Run 5-fold cross-validation for material classifier and particle mass regressor."""
    print("\n--- Running 5-Fold Cross-Validation on DS2 Paired Samples ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    # 1. Classification CV
    y_encoded = np.array([PHYSICAL_CLASSES.index(c) for c in y_class])
    clf_preds = np.zeros(len(y_class), dtype=int)
    for tr, te in skf.split(X_clf, y_encoded):
        clf = ParticleMaterialClassifier(n_estimators=150, seed=seed)
        clf.fit(X_clf[tr], [PHYSICAL_CLASSES[i] for i in y_encoded[tr]])
        preds = clf.predict(X_clf[te])
        clf_preds[te] = [PHYSICAL_CLASSES.index(p) for p in preds]

    clf_acc = float(accuracy_score(y_encoded, clf_preds))
    clf_f1 = float(f1_score(y_encoded, clf_preds, average="macro"))
    print(f"Particle Classifier CV Accuracy: {clf_acc * 100:.2f}% | Macro-F1: {clf_f1:.4f}")

    # 2. Regression CV
    reg_preds = np.zeros(len(y_mass), dtype=np.float64)
    for tr, te in kf.split(X_reg):
        reg = ParticleMassRegressor(model_type="gbr", seed=seed)
        reg.fit(X_reg[tr], y_mass[tr])
        reg_preds[te] = reg.predict_mass_g(X_reg[te])

    reg_mae = float(mean_absolute_error(y_mass, reg_preds))
    reg_r2 = float(r2_score(y_mass, reg_preds))
    print(f"Particle Mass Regressor CV MAE: {reg_mae:.4f} g | R2: {reg_r2:.4f}")

    return {
        "classification_accuracy": round(clf_acc, 4),
        "classification_macro_f1": round(clf_f1, 4),
        "mass_regression_mae_g": round(reg_mae, 4),
        "mass_regression_r2": round(reg_r2, 4),
    }


def evaluate_ds3_mixed_batches(zip_path: Path, clf: ParticleMaterialClassifier, reg: ParticleMassRegressor):
    """Evaluate trained models on 2,789 unlabelled particles across 3 real mixed batches in DS3."""
    print("\n--- Running Empirical Validation on DS3 Mixed Batches ---")

    true_fractions_ds3 = {
        1: {"FEM": 0.299251, "NFM": 0.300050, "PCB": 0.200000, "PLA": 0.200699},
        2: {"FEM": 0.449703, "NFM": 0.149468, "PCB": 0.101393, "PLA": 0.299436},
        3: {"FEM": 0.151382, "NFM": 0.450897, "PCB": 0.297766, "PLA": 0.099955},
    }

    scale_masses_ds3 = {
        1: 2.0030,  # 2003.0 g = 2.003 kg
        2: 4.0062,  # 4006.2 g = 4.0062 kg
        3: 4.0018,  # 4001.8 g = 4.0018 kg
    }

    batch_results = {}
    all_maes = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for s in [1, 2, 3]:
            sample_prefix = f"IMG/DS3/MS{s}/"
            files = [f for f in z.namelist() if f.startswith(sample_prefix) and f.endswith(".png")]
            predicted_classes = []
            predicted_masses_g = []

            for f in files:
                img_bytes = z.read(f)
                with Image.open(io.BytesIO(img_bytes)) as img:
                    c_feat = extract_particle_classifier_features(img)
                    c_vec = np.array([[
                        c_feat["area_px"],
                        c_feat["bbox_area_px"],
                        c_feat["aspect_ratio"],
                        c_feat["fill_ratio"],
                        c_feat["fg_mean_r"],
                        c_feat["fg_mean_g"],
                        c_feat["fg_mean_b"],
                        c_feat["fg_std_r"],
                        c_feat["fg_std_g"],
                        c_feat["fg_std_b"],
                        c_feat["mean_r"],
                        c_feat["mean_g"],
                        c_feat["mean_b"],
                        c_feat["std_intensity"],
                        c_feat["green_red_ratio"],
                        c_feat["red_blue_ratio"],
                    ]], dtype=np.float64)

                    pred_class = clf.predict(c_vec)[0]
                    r_feat = extract_particle_features(img, predicted_class=pred_class)
                    r_vec = np.array([[
                        r_feat["area_px"],
                        r_feat["bbox_area_px"],
                        r_feat["aspect_ratio"],
                        r_feat["fill_ratio"],
                        r_feat["mean_r"],
                        r_feat["mean_g"],
                        r_feat["mean_b"],
                        r_feat["std_intensity"],
                        r_feat["is_fem"],
                        r_feat["is_nfm"],
                        r_feat["is_pcb"],
                        r_feat["is_pla"],
                    ]], dtype=np.float64)

                    pred_mass = float(reg.predict_mass_g(r_vec)[0])

                predicted_classes.append(pred_class)
                predicted_masses_g.append(pred_mass)

            # Reconcile batch
            scale_mass = scale_masses_ds3[s]
            reconciled = reconcile_batch_composition(
                predicted_classes,
                predicted_masses_g,
                weighbridge_mass_kg=scale_mass,
                include_other=False,
            )

            # Compute composition errors
            true_frac = true_fractions_ds3[s]
            pred_frac = reconciled["fractions"]
            errors = compute_composition_mae(pred_frac, true_frac)
            mae = errors["mean_abs_error"]
            all_maes.append(mae)

            # Generate bootstrap scenarios
            scenarios = bootstrap_batch_scenarios(
                predicted_classes,
                predicted_masses_g,
                n_scenarios=5,
                seed=42,
                base_fractions=pred_frac,
                include_other=False,
            )

            print(f"MS{s} ({len(files)} particles, Scale={scale_mass:.3f} kg):")
            print(f"  Predicted Fractions: {', '.join(f'{k}: {pred_frac[k]*100:.1f}%' for k in PHYSICAL_CLASSES)}")
            print(f"  Ground Truth:        {', '.join(f'{k}: {true_frac[k]*100:.1f}%' for k in PHYSICAL_CLASSES)}")
            print(f"  Batch Composition MAE: {mae * 100:.2f}% (per-class: {', '.join(f'{k}: {errors[k]*100:.2f}%' for k in PHYSICAL_CLASSES)})")

            batch_results[f"MS{s}"] = {
                "num_particles": len(files),
                "weighbridge_mass_kg": scale_mass,
                "raw_visual_mass_kg": reconciled["raw_visual_mass_kg"],
                "visual_to_scale_ratio": reconciled["visual_to_scale_ratio"],
                "predicted_fractions": {k: round(float(v), 4) for k, v in pred_frac.items()},
                "ground_truth_fractions": {k: round(float(v), 4) for k, v in true_frac.items()},
                "composition_mae": round(mae, 4),
                "class_errors": {k: round(float(errors[k]), 4) for k in PHYSICAL_CLASSES},
                "scenarios": [{k: round(float(v), 4) for k, v in sc.items()} for sc in scenarios],
            }

    mean_ds3_mae = float(np.mean(all_maes))
    print(f"\nOverall Mean DS3 Mixed-Batch Composition MAE: {mean_ds3_mae * 100:.2f}%")

    return {
        "batches": batch_results,
        "mean_composition_mae": round(mean_ds3_mae, 4),
    }


def train_and_evaluate(zip_path: Path, output_dir: Path, seed: int = 42):
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Phase 1: Load DS2 & CV
    X_clf, X_reg, y_class, y_mass, timestamps = load_ds2_data(zip_path)
    cv_metrics = evaluate_ds2_cv(X_clf, X_reg, y_class, y_mass, seed=seed)

    # Train final models on all DS2 paired samples
    print("\n--- Training Final Models on Full DS2 Dataset (1,976 particles) ---")
    final_clf = ParticleMaterialClassifier(n_estimators=150, seed=seed)
    final_clf.fit(X_clf, y_class)

    final_reg = ParticleMassRegressor(model_type="gbr", seed=seed)
    final_reg.fit(X_reg, y_mass)

    clf_path = output_dir / "classifier.joblib"
    reg_path = output_dir / "regressor.joblib"
    final_clf.save(clf_path)
    final_reg.save(reg_path)
    print(f"Saved classifier to: {clf_path}")
    print(f"Saved regressor to: {reg_path}")

    # Phase 2: DS3 Validation
    ds3_eval = evaluate_ds3_mixed_batches(zip_path, final_clf, final_reg)

    # Assemble Full Report
    duration = time.time() - t0
    report = {
        "pipeline": "Track A TU Darmstadt RGB-MFCO Particle Composition & Mass Regression",
        "training_dataset": {
            "name": "TU Darmstadt WEEE RGB-MFCO DS2",
            "source_zip": zip_path.name,
            "paired_particles_count": len(timestamps),
            "classes": list(PHYSICAL_CLASSES),
        },
        "evaluation_dataset": {
            "name": "TU Darmstadt WEEE RGB-MFCO DS3 (Mixed Batches MS1, MS2, MS3)",
            "total_mixed_particles": sum(b["num_particles"] for b in ds3_eval["batches"].values()),
        },
        "ds2_cross_validation": cv_metrics,
        "ds3_mixed_batch_validation": ds3_eval,
        "summary_metrics": {
            "particle_mass_mae_g": cv_metrics["mass_regression_mae_g"],
            "particle_mass_r2": cv_metrics["mass_regression_r2"],
            "classification_accuracy": cv_metrics["classification_accuracy"],
            "classification_macro_f1": cv_metrics["classification_macro_f1"],
            "mixed_batch_composition_mae": ds3_eval["mean_composition_mae"],
            "ms1_composition_mae": ds3_eval["batches"]["MS1"]["composition_mae"],
            "ms2_composition_mae": ds3_eval["batches"]["MS2"]["composition_mae"],
            "ms3_composition_mae": ds3_eval["batches"]["MS3"]["composition_mae"],
        },
        "duration_seconds": round(duration, 2),
    }

    report_path = output_dir / "track_a_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nWritten evaluation report to: {report_path}")
    print(f"Total pipeline execution time: {duration:.2f} seconds")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate Track A models")
    parser.add_argument("--zip_path", type=Path, default=Path("RGB-MFCO-dataset-v1.0.0.zip"))
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/track_a"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_and_evaluate(args.zip_path, args.output_dir, seed=args.seed)
