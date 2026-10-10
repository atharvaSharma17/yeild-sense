"""Conditional Track A Mixed-Batch Composition Evaluation & Group Bootstrap (`ml/track_a/composition_eval.py`)."""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

CLASSES = ("FEM", "NFM", "PCB", "PLA")
CONTRACT_MATERIALS = ("FEM", "NFM", "PCB", "PLA", "OTHER")


def compute_composition_mae(
    pred_fractions: Dict[str, float],
    true_fractions: Dict[str, float],
    classes: tuple = CLASSES,
) -> Dict[str, float]:
    errors = {
        c: abs(float(pred_fractions[c]) - float(true_fractions[c])) for c in classes if c in pred_fractions and c in true_fractions
    }
    errors["mean_abs_error"] = float(np.mean(list(errors.values()))) if errors else 0.0
    return errors


def bootstrap_batch_scenarios(
    particle_classes: List[str],
    predicted_masses_g: List[float],
    n_scenarios: int = 10,
    seed: int = 42,
    base_fractions: Optional[Dict[str, float]] = None,
    include_other: bool = False,
    other_fraction: float = 0.015,
) -> List[Dict[str, float]]:
    """Group-bootstrap particles within a batch to generate empirical composition scenarios."""
    rng = np.random.default_rng(seed)
    n = len(particle_classes)
    if n == 0:
        raise ValueError("Cannot bootstrap empty particle list")

    scenarios: List[Dict[str, float]] = []

    # If base_fractions provided, scenario[0] is set exactly to it to satisfy contract
    if base_fractions is not None:
        scen_0 = dict(base_fractions)
        max_k = max(scen_0.keys(), key=lambda k: scen_0[k])
        scen_0[max_k] += 1.0 - sum(scen_0.values())
        scenarios.append(scen_0)

    target_count = n_scenarios if base_fractions is None else (n_scenarios - 1)

    for _ in range(target_count):
        idx = rng.integers(0, n, size=n)
        totals = {c: 1e-6 for c in CLASSES}
        for i in idx:
            c = particle_classes[int(i)]
            if c in totals:
                totals[c] += max(float(predicted_masses_g[int(i)]), 0.0)
        s = sum(totals.values())

        if include_other:
            rem = 1.0 - other_fraction
            vec = {c: (totals[c] / s) * rem for c in CLASSES}
            vec["OTHER"] = float(other_fraction)
            max_c = max(CONTRACT_MATERIALS, key=lambda k: vec[k])
            vec[max_c] += 1.0 - sum(vec.values())
        else:
            vec = {c: totals[c] / s for c in CLASSES}
            max_c = max(CLASSES, key=lambda k: vec[k])
            vec[max_c] += 1.0 - sum(vec.values())

        scenarios.append(vec)

    return scenarios

