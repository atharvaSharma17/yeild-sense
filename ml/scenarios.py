"""Entropy-Conditioned & Simplex-Normalized Compositional Scenario Generator for YieldSense 2.0.

Bridges:
1. Verified FutuRaM WEEE Category Priors (`data/processed/futuram_priors.json`, Zenodo 17789750)
2. Visual Classification Entropy H(p_cat) -> Dynamic Dirichlet Concentration kappa(H)
3. Exact Unit-Simplex Projection (guarantees sum(s.values()) == 1.0 within < 1e-12)
4. Downside Tail Stress Scenarios (diluting high-value PCB/NFM into PLA)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

CLASSES = ("FEM", "NFM", "PCB", "PLA")

# Fallback literature priors if data/processed/futuram_priors.json has not been generated yet
FALLBACK_PRIORS: Dict[str, Dict[str, float]] = {
    "WEEE_Cat1": {"FEM": 0.5175455, "NFM": 0.0840099, "PCB": 0.0011754, "PLA": 0.3972692},
    "WEEE_Cat2": {"FEM": 0.2338268, "NFM": 0.0633680, "PCB": 0.0734140, "PLA": 0.6293912},
    "WEEE_Cat3": {"FEM": 0.0173097, "NFM": 0.3312082, "PCB": 0.0833633, "PLA": 0.5681188},
    "WEEE_Cat4a": {"FEM": 0.5627563, "NFM": 0.0433719, "PCB": 0.0058649, "PLA": 0.3880069},
    "WEEE_Cat4b": {"FEM": 0.0312695, "NFM": 0.1160180, "PCB": 0.0009849, "PLA": 0.8517276},
    "WEEE_Cat5": {"FEM": 0.3951456, "NFM": 0.1007171, "PCB": 0.0259020, "PLA": 0.4782353},
    "WEEE_Cat6": {"FEM": 0.3169060, "NFM": 0.0220275, "PCB": 0.0809833, "PLA": 0.5800832},
    "Small IT Equipment": {"FEM": 0.3169060, "NFM": 0.0220275, "PCB": 0.0809833, "PLA": 0.5800832},
    "Large Household Appliances": {"FEM": 0.5627563, "NFM": 0.0433719, "PCB": 0.0058649, "PLA": 0.3880069},
    "Consumer Electronics": {"FEM": 0.2338268, "NFM": 0.0633680, "PCB": 0.0734140, "PLA": 0.6293912},
    "Cooling Appliances": {"FEM": 0.5175455, "NFM": 0.0840099, "PCB": 0.0011754, "PLA": 0.3972692},
    "Small Household Appliances": {"FEM": 0.3951456, "NFM": 0.1007171, "PCB": 0.0259020, "PLA": 0.4782353},
    "standard": {"FEM": 0.3951456, "NFM": 0.1007171, "PCB": 0.0259020, "PLA": 0.4782353},
}

# Map XBAT+ 15 intact device classes to FutuRaM WEEE Categories
XBAT_CLASS_TO_WEEE_CAT: Dict[str, str] = {
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

DEFAULT_PRIOR: Dict[str, float] = {"FEM": 0.25, "NFM": 0.25, "PCB": 0.25, "PLA": 0.25}


def normalize_simplex_dict(raw: Dict[str, float]) -> Dict[str, float]:
    """Project a composition dictionary strictly onto the unit simplex (sum == 1.0)."""
    vec = np.array([max(float(raw.get(c, 0.0)), 0.0) for c in CLASSES], dtype=np.float64)
    total = float(vec.sum())
    if total <= 0.0:
        raise ValueError("Composition vector must have positive total mass share")
    vec = vec / total
    # Absorb any floating-point rounding residual into the largest component
    idx_max = int(np.argmax(vec))
    vec[idx_max] += 1.0 - float(np.sum(vec))
    return {c: float(vec[i]) for i, c in enumerate(CLASSES)}


def load_futuram_priors(
    priors_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Load verified FutuRaM priors from JSON, merged with XBAT+ and human aliases."""
    table = dict(FALLBACK_PRIORS)
    candidates = [
        priors_path,
        Path("data/processed/futuram_priors.json"),
        Path(__file__).resolve().parent.parent / "data" / "processed" / "futuram_priors.json",
    ]
    for p in candidates:
        if p is not None and p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for cat_id, info in data.get("categories", {}).items():
                    table[cat_id] = normalize_simplex_dict(info["mean"])
                for alias, mean_vec in data.get("aliases", {}).items():
                    table[alias] = normalize_simplex_dict(mean_vec)
                break
            except Exception:
                continue

    for xbat_cls, weee_cat in XBAT_CLASS_TO_WEEE_CAT.items():
        if weee_cat in table:
            table[xbat_cls] = table[weee_cat]
    return table


FUTURAM_PRIORS: Dict[str, Dict[str, float]] = load_futuram_priors()


def get_category_prior(category: str) -> Dict[str, float]:
    """Retrieve the normalized 4-class composition prior for `category`."""
    priors = load_futuram_priors()
    if category in priors:
        return dict(priors[category])
    if category in XBAT_CLASS_TO_WEEE_CAT:
        mapped = XBAT_CLASS_TO_WEEE_CAT[category]
        if mapped in priors:
            return dict(priors[mapped])
    return dict(DEFAULT_PRIOR)


def compute_entropy_concentration(
    entropy: float,
    kappa_min: float = 5.0,
    kappa_max: float = 120.0,
    beta: float = 1.2,
) -> float:
    """Map prediction Shannon entropy H(p) >= 0 to Dirichlet concentration kappa in [kappa_min, kappa_max]."""
    h = max(float(entropy), 0.0)
    return float(kappa_min + (kappa_max - kappa_min) * np.exp(-beta * h))


def generate_scenarios(
    mean: Dict[str, float],
    n: int = 10,
    concentration: float = 50.0,
    seed: int = 42,
    include_stress: bool = False,
) -> List[Dict[str, float]]:
    """Generate `n` Dirichlet composition scenarios anchored at `mean` (total `n + 1` with mean)."""
    if set(mean.keys()) != set(CLASSES):
        raise ValueError(f"Composition must contain exact keys {CLASSES}")
    vec = np.array([float(mean[c]) for c in CLASSES], dtype=np.float64)
    if np.any(vec < 0) or not np.isclose(vec.sum(), 1.0, atol=1e-5):
        raise ValueError("Composition must be nonnegative and sum to 1")
    if concentration <= 0:
        raise ValueError("Concentration parameter must be positive")

    clean_mean = normalize_simplex_dict(mean)
    alpha = np.maximum(np.array([clean_mean[c] for c in CLASSES]), 1e-6) * float(concentration)
    rng = np.random.default_rng(seed)

    n_random = max(0, n - 1) if (include_stress and n >= 2) else n
    draws = rng.dirichlet(alpha, size=n_random) if n_random > 0 else np.empty((0, len(CLASSES)))

    scenarios: List[Dict[str, float]] = [clean_mean]
    for row in draws:
        scenarios.append(normalize_simplex_dict({c: float(v) for c, v in zip(CLASSES, row)}))

    if include_stress and n >= 2:
        # Deterministic downside dilution stress scenario: depress high-value PCB & NFM, inflate PLA
        dilution_factor = float(np.clip(1.0 - 2.5 / np.sqrt(max(concentration, 4.0)), 0.45, 0.88))
        stressed = dict(clean_mean)
        lost_pcb = stressed["PCB"] * (1.0 - dilution_factor)
        lost_nfm = stressed["NFM"] * (1.0 - dilution_factor)
        stressed["PCB"] -= lost_pcb
        stressed["NFM"] -= lost_nfm
        stressed["PLA"] += lost_pcb + lost_nfm
        scenarios.append(normalize_simplex_dict(stressed))

    return scenarios


def generate_adaptive_scenarios(
    category: str,
    entropy: float,
    n_scenarios: int = 10,
    kappa_min: float = 5.0,
    kappa_max: float = 120.0,
    beta: float = 1.2,
    seed: int = 42,
    include_stress: bool = True,
) -> List[Dict[str, float]]:
    """Generate entropy-conditioned Dirichlet composition scenarios centered on FutuRaM prior."""
    prior = get_category_prior(category)
    kappa = compute_entropy_concentration(
        entropy=entropy,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        beta=beta,
    )
    return generate_scenarios(
        mean=prior,
        n=n_scenarios,
        concentration=kappa,
        seed=seed,
        include_stress=include_stress,
    )
