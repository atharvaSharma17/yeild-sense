"""Transform verified FutuRaM WEEE composition CSV (Zenodo 17789750) into canonical priors.

Mapping Methodology (explicitly documented per Implementation Guide §2):
- Source unit in FutuRaM v14.2 is `ppm` (parts per million, 1,000,000 ppm = 1.0 mass share).
- For each WEEE Category (`Layer 1` in {WEEE_Cat1..WEEE_Cat6}) and Scenario/Year:
  1. `PCB` fraction is extracted directly from `parameterCode == 'c-p'` where `Layer 2 == 'PCB'`.
  2. For non-PCB components (`Layer 2 != 'PCB'`, `parameterCode == 'm-c'`):
     - `FEM` (Ferrous Metals): `Layer 3 == 'ferrousMetals'`
     - `NFM` (Non-Ferrous Metals): `Layer 3` in `{'AlAndAlAlloys', 'CuAndCuAlloys', 'nonCuAndAlAlloys'}`
     - `PLA` (Plastics & non-metallic organic/inorganic matrix in 4-class projection):
       `Layer 3` in `{'WEEEMaterialRest', 'glassAndCeramics', 'SiSemiconductor'}`
       (We also record the exact split between `WEEEMaterialRest` [plastics/polymers] and
       `glassAndCeramics` as `OTHER_INORGANIC` in the 5-class audit metadata so zero mass is hidden.)
  3. All 4-class vectors (`FEM`, `NFM`, `PCB`, `PLA`) are strictly normalized to sum to 1.0
     on the unit simplex with IEEE-754 precision.
  4. Cross-year/scenario dispersion is computed to ground category uncertainty spreads.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

CLASSES = ("FEM", "NFM", "PCB", "PLA")

CATEGORY_DESCRIPTIONS = {
    "WEEE_Cat1": "Temperature Exchange Equipment (Cooling & Freezing Appliances)",
    "WEEE_Cat2": "Screens, Monitors, and Equipment Containing Screens >100 cm2",
    "WEEE_Cat3": "Lamps and Lighting Equipment",
    "WEEE_Cat4a": "Large Household & Industrial Equipment",
    "WEEE_Cat4b": "Photovoltaic Panels",
    "WEEE_Cat5": "Small Household & Consumer Equipment",
    "WEEE_Cat6": "Small IT and Telecommunication Equipment",
}

# Friendly aliases used by UI / human operators mapped to canonical FutuRaM Layer 1 keys
HUMAN_CATEGORY_ALIASES = {
    "Small IT Equipment": "WEEE_Cat6",
    "Large Household Appliances": "WEEE_Cat4a",
    "Consumer Electronics": "WEEE_Cat2",
    "Cooling Appliances": "WEEE_Cat1",
    "Small Household Appliances": "WEEE_Cat5",
    "Lamps": "WEEE_Cat3",
    "Photovoltaic Panels": "WEEE_Cat4b",
    "standard": "WEEE_Cat5",
}


def _extract_single_slice(grp: pd.DataFrame) -> Dict[str, float]:
    """Extract raw ppm totals for FEM, NFM, PCB, PLA (and glass/other audit info) from one slice."""
    cp = grp[grp["parameterCode"] == "c-p"]
    pcb_ppm = float(cp[cp["Layer 2"] == "PCB"]["Value"].sum())

    # Material-in-component rows excluding PCB (since PCB is already counted as a whole rich stream)
    mc_non_pcb = grp[(grp["parameterCode"] == "m-c") & (grp["Layer 2"] != "PCB")]
    by_mat = mc_non_pcb.groupby("Layer 3")["Value"].sum().to_dict()

    fem_ppm = float(by_mat.get("ferrousMetals", 0.0))
    nfm_ppm = float(
        by_mat.get("AlAndAlAlloys", 0.0)
        + by_mat.get("CuAndCuAlloys", 0.0)
        + by_mat.get("nonCuAndAlAlloys", 0.0)
    )
    polymer_rest_ppm = float(by_mat.get("WEEEMaterialRest", 0.0))
    glass_si_ppm = float(
        by_mat.get("glassAndCeramics", 0.0) + by_mat.get("SiSemiconductor", 0.0)
    )
    pla_matrix_ppm = polymer_rest_ppm + glass_si_ppm

    total_ppm = fem_ppm + nfm_ppm + pcb_ppm + pla_matrix_ppm
    if total_ppm <= 0:
        raise ValueError("Slice has non-positive total ppm")

    raw_vec = np.array([fem_ppm, nfm_ppm, pcb_ppm, pla_matrix_ppm], dtype=np.float64)
    norm_vec = raw_vec / raw_vec.sum()
    # Adjust largest component for exact sum == 1.0
    max_idx = int(np.argmax(norm_vec))
    norm_vec[max_idx] += 1.0 - float(np.sum(norm_vec))

    return {
        "fractions": {c: float(norm_vec[i]) for i, c in enumerate(CLASSES)},
        "audit_ppm": {
            "FEM_ppm": round(fem_ppm, 2),
            "NFM_ppm": round(nfm_ppm, 2),
            "PCB_ppm": round(pcb_ppm, 2),
            "PLA_polymer_rest_ppm": round(polymer_rest_ppm, 2),
            "OTHER_glass_ceramics_si_ppm": round(glass_si_ppm, 2),
            "total_accounted_ppm": round(total_ppm, 2),
        },
    }


def transform_futuram(
    csv_path: Path = Path("data/raw/futuram_composition.csv"),
    output_path: Path = Path("data/processed/futuram_priors.json"),
) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing FutuRaM CSV at {csv_path}")

    df = pd.read_csv(csv_path)
    dedicated = df[df["Stock/Flow ID"] == "WEEE_generatedDedicated"].copy()

    categories_out: Dict[str, Any] = {}
    for cat_id, cat_grp in dedicated.groupby("Layer 1"):
        # Baseline reference: BAU 2023
        bau_2023 = cat_grp[(cat_grp["Scenario"] == "BAU") & (cat_grp["Year"] == 2023)]
        if bau_2023.empty:
            bau_2023 = cat_grp.iloc[:0]

        ref_slice = _extract_single_slice(bau_2023 if not bau_2023.empty else cat_grp)

        # Compute empirical dispersion across all available (Scenario, Year) combinations
        slice_vectors = []
        for (_, _), sub_grp in cat_grp.groupby(["Scenario", "Year"]):
            try:
                s_info = _extract_single_slice(sub_grp)
                slice_vectors.append([s_info["fractions"][c] for c in CLASSES])
            except ValueError:
                continue

        arr = np.array(slice_vectors, dtype=np.float64)
        std_vec = np.std(arr, axis=0) if len(arr) > 1 else np.zeros(4)
        min_vec = np.min(arr, axis=0) if len(arr) > 0 else np.zeros(4)
        max_vec = np.max(arr, axis=0) if len(arr) > 0 else np.ones(4)

        categories_out[str(cat_id)] = {
            "category_id": str(cat_id),
            "description": CATEGORY_DESCRIPTIONS.get(str(cat_id), str(cat_id)),
            "mean": ref_slice["fractions"],
            "audit_ppm_2023_bau": ref_slice["audit_ppm"],
            "cross_scenario_year_std": {c: float(std_vec[i]) for i, c in enumerate(CLASSES)},
            "cross_scenario_year_min": {c: float(min_vec[i]) for i, c in enumerate(CLASSES)},
            "cross_scenario_year_max": {c: float(max_vec[i]) for i, c in enumerate(CLASSES)},
            "n_trajectories_evaluated": int(len(arr)),
        }

    # Add human-readable alias mappings
    aliases_out: Dict[str, Dict[str, float]] = {}
    for alias, target_cat in HUMAN_CATEGORY_ALIASES.items():
        if target_cat in categories_out:
            aliases_out[alias] = categories_out[target_cat]["mean"]

    payload = {
        "dataset": "FutuRaM Consolidated WEEE Composition v14.2",
        "zenodo_record": "17789750",
        "reference_slice": "Scenario=BAU, Year=2023, Stock/Flow=WEEE_generatedDedicated",
        "classes": list(CLASSES),
        "categories": categories_out,
        "aliases": aliases_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote verified FutuRaM priors for {len(categories_out)} categories to {output_path}")
    return payload


if __name__ == "__main__":
    transform_futuram()
