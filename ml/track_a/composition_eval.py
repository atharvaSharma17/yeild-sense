"""Conditional Track A Mixed-Batch Composition Evaluation & Group Bootstrap (`ml/track_a/composition_eval.py`)."""

from __future__ import annotations
from typing import Dict, List
import numpy as np

CLASSES = ("FEM", "NFM", "PCB", "PLA")


def compute_composition_mae(
    pred_fractions: Dict[str, float],
    true_fractions: Dict[str, float],
) -> Dict[str, float]:
    errors = {
        c: abs(float(pred_fractions[c]) - float(true_fractions[c])) for c in CLASSES
    }
    errors["mean_abs_error"] = float(np.mean(list(errors.values())))
    return errors


def bootstrap_batch_scenarios(
    particle_classes: List[str],
    predicted_masses_g: List[float],
    n_scenarios: int = 10,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """Group-bootstrap particles within a batch to generate empirical composition scenarios."""
    rng = np.random.default_rng(seed)
    n = len(particle_classes)
    if n == 0:
        raise ValueError("Cannot bootstrap empty particle list")

    scenarios: List[Dict[str, float]] = []
    for _ in range(n_scenarios):
        idx = rng.integers(0, n, size=n)
        totals = {c: 1e-6 for c in CLASSES}
        for i in idx:
            c = particle_classes[int(i)]
            if c in totals:
                totals[c] += max(float(predicted_masses_g[int(i)]), 0.0)
        s = sum(totals.values())
        vec = {c: totals[c] / s for c in CLASSES}
        max_c = max(CLASSES, key=lambda k: vec[k])
        vec[max_c] += 1.0 - sum(vec.values())
        scenarios.append(vec)
    return scenarios
