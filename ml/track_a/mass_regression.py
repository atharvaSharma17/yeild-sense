"""Conditional Track A Particle Mass Regression & Weighbridge Reconciliation (`ml/track_a/mass_regression.py`)."""

from __future__ import annotations
from typing import Dict, List
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

CLASSES = ("FEM", "NFM", "PCB", "PLA")


class ParticleMassRegressor:
    """Gradient-boosted / Random-Forest particle mass regressor with weighbridge reconciliation."""

    def __init__(self, model_type: str = "gbr", seed: int = 42):
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


def reconcile_batch_composition(
    particle_classes: List[str],
    predicted_masses_g: List[float],
    weighbridge_mass_kg: float,
) -> Dict[str, object]:
    """Aggregate predicted particle masses into class fractions and reconcile with scale mass."""
    class_totals_g = {c: 0.0 for c in CLASSES}
    for cls_name, m_g in zip(particle_classes, predicted_masses_g):
        if cls_name in class_totals_g:
            class_totals_g[cls_name] += max(float(m_g), 0.0)

    total_pred_g = sum(class_totals_g.values())
    if total_pred_g <= 0:
        raise ValueError("Predicted total particle mass must be positive")

    fractions = {c: class_totals_g[c] / total_pred_g for c in CLASSES}
    reconciled_kg = {c: fractions[c] * float(weighbridge_mass_kg) for c in CLASSES}
    discrepancy_ratio = (total_pred_g / 1000.0) / max(float(weighbridge_mass_kg), 1e-6)

    return {
        "fractions": fractions,
        "reconciled_class_mass_kg": reconciled_kg,
        "raw_visual_mass_kg": round(total_pred_g / 1000.0, 4),
        "weighbridge_mass_kg": float(weighbridge_mass_kg),
        "visual_to_scale_ratio": round(discrepancy_ratio, 4),
    }
