"""Conditional Track A Particle Mass Regression & Weighbridge Reconciliation (`ml/track_a/mass_regression.py`)."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor, RandomForestRegressor

PHYSICAL_CLASSES = ("FEM", "NFM", "PCB", "PLA")
CONTRACT_MATERIALS = ("FEM", "NFM", "PCB", "PLA", "OTHER")


class ParticleMaterialClassifier:
    """Extra-Trees particle material classifier for shredded WEEE crops (FEM, NFM, PCB, PLA)."""

    def __init__(self, n_estimators: int = 150, seed: int = 42):
        self.classes = PHYSICAL_CLASSES
        self.model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=seed)

    def fit(self, X: np.ndarray, y_classes: Union[np.ndarray, List[str]]) -> "ParticleMaterialClassifier":
        y_encoded = np.array([self.classes.index(c) if isinstance(c, str) else int(c) for c in y_classes])
        self.model.fit(X, y_encoded)
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        preds = self.model.predict(X)
        return [self.classes[int(idx)] for idx in preds]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump({"model": self.model, "classes": self.classes}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParticleMaterialClassifier":
        data = joblib.load(path)
        inst = cls()
        inst.model = data["model"]
        inst.classes = tuple(data["classes"])
        return inst


class ParticleMassRegressor:
    """Gradient-boosted / Random-Forest particle mass regressor with weighbridge reconciliation."""

    def __init__(self, model_type: str = "gbr", seed: int = 42):
        self.model_type = model_type
        self.seed = seed
        if model_type == "rf":
            self.model = RandomForestRegressor(n_estimators=100, random_state=seed)
        else:
            self.model = GradientBoostingRegressor(random_state=seed)

    def fit(self, X: np.ndarray, y_mass_g: np.ndarray) -> "ParticleMassRegressor":
        self.model.fit(X, y_mass_g)
        return self

    def predict_mass_g(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.maximum(preds, 1e-3)

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump({"model": self.model, "model_type": self.model_type, "seed": self.seed}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParticleMassRegressor":
        data = joblib.load(path)
        inst = cls(model_type=data.get("model_type", "gbr"), seed=data.get("seed", 42))
        inst.model = data["model"]
        return inst


def reconcile_batch_composition(
    particle_classes: List[str],
    predicted_masses_g: List[float],
    weighbridge_mass_kg: float,
    include_other: bool = False,
    other_fraction: float = 0.015,
) -> Dict[str, object]:
    """Aggregate predicted particle masses into class fractions and reconcile with scale mass."""
    class_totals_g = {c: 0.0 for c in PHYSICAL_CLASSES}
    for cls_name, m_g in zip(particle_classes, predicted_masses_g):
        if cls_name in class_totals_g:
            class_totals_g[cls_name] += max(float(m_g), 0.0)

    total_pred_g = sum(class_totals_g.values())
    if total_pred_g <= 0:
        raise ValueError("Predicted total particle mass must be positive")

    if include_other:
        # Apportion 1.0 - other_fraction across the 4 physical material classes
        remaining_frac = 1.0 - other_fraction
        fractions = {
            c: (class_totals_g[c] / total_pred_g) * remaining_frac for c in PHYSICAL_CLASSES
        }
        fractions["OTHER"] = float(other_fraction)
        # Ensure exact sum to 1.0
        max_c = max(fractions.keys(), key=lambda k: fractions[k])
        fractions[max_c] += 1.0 - sum(fractions.values())

        reconciled_kg = {c: fractions[c] * float(weighbridge_mass_kg) for c in CONTRACT_MATERIALS}
    else:
        fractions = {c: class_totals_g[c] / total_pred_g for c in PHYSICAL_CLASSES}
        reconciled_kg = {c: fractions[c] * float(weighbridge_mass_kg) for c in PHYSICAL_CLASSES}

    discrepancy_ratio = (total_pred_g / 1000.0) / max(float(weighbridge_mass_kg), 1e-6)

    return {
        "fractions": fractions,
        "reconciled_class_mass_kg": reconciled_kg,
        "raw_visual_mass_kg": round(total_pred_g / 1000.0, 4),
        "weighbridge_mass_kg": float(weighbridge_mass_kg),
        "visual_to_scale_ratio": round(discrepancy_ratio, 4),
    }

