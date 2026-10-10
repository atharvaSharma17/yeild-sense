"""Generate Deterministic Synthetic Benchmarks & 30-Replication Evaluation Suite (`scripts/make_synthetic.py`).

Implements the experimental protocol from Implementation Guide §11 & HLD §12.1:
1. Generates reproducible batch fixtures (20, 50, 100 lots) grounded in verified FutuRaM priors.
2. Models joint supplier/collection quality scenarios where specialized hydrometallurgical
   routes have high upside on nominal PCB/NFM grades but steep downside exposure under
   plastic/ferrous compositional dilution.
3. Separates `calibration_scenarios` (used by optimizer to choose route assignments) from
   `held_out_evaluation_scenarios` to prevent in-sample optimization bias.
4. Evaluates FIXED_ROUTE, MEAN_GREEDY, EXPECTED, ROBUST, and CVAR policies across 30 seeded
   replications and writes `data/synthetic/scenarios.json` and `data/processed/benchmark_summary.json`.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = ROOT / "backend"
for p in (ROOT, BACKEND_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.optimizer import Lot, Route, compare_policies
from app.scenarios import CLASSES, get_category_prior, normalize_simplex_dict

BENCHMARK_CATEGORIES = [
    "WEEE_Cat6",
    "WEEE_Cat5",
    "WEEE_Cat2",
    "WEEE_Cat1",
    "WEEE_Cat4a",
]


def build_benchmark_routes(scale_factor: float = 1.0) -> List[Route]:
    """Create 4 recovery routes with realistic threshold economics:
    - ROUTE-PCB-HYDROMET: High chemical treatment fee (46 INR/kg) + very high PCB/precious payout.
      Highly profitable when PCB >= 6.5%, but loses money if batch is diluted with plastics.
    - ROUTE-NFM-SMELT: Medium fee (22 INR/kg), strong NFM + moderate PCB recovery.
    - ROUTE-MECH-BULK: Low fee (7.5 INR/kg), resilient mechanical separation across FEM/NFM/PLA.
    - ROUTE-BAT-QUARANTINE: CPCB-authorized battery disassembly & safe quarantine line.
    """
    return [
        Route(
            route_id="ROUTE-PCB-HYDROMET",
            capacity_kg=1400.0 * scale_factor,
            cost_per_kg=46.0,
            transport_per_lot=950.0,
            recovery={"FEM": 0.15, "NFM": 0.65, "PCB": 0.95, "PLA": 0.02},
            price={"FEM": 20.0, "NFM": 160.0, "PCB": 740.0, "PLA": 5.0},
            accepted_categories=frozenset(BENCHMARK_CATEGORIES),
            allowed_battery_states=frozenset({"confirmed_absent"}),
            authorized=True,
        ),
        Route(
            route_id="ROUTE-NFM-SMELT",
            capacity_kg=1300.0 * scale_factor,
            cost_per_kg=22.0,
            transport_per_lot=550.0,
            recovery={"FEM": 0.70, "NFM": 0.92, "PCB": 0.48, "PLA": 0.25},
            price={"FEM": 24.0, "NFM": 185.0, "PCB": 390.0, "PLA": 12.0},
            accepted_categories=frozenset(BENCHMARK_CATEGORIES),
            allowed_battery_states=frozenset({"confirmed_absent"}),
            authorized=True,
        ),
        Route(
            route_id="ROUTE-MECH-BULK",
            capacity_kg=2200.0 * scale_factor,
            cost_per_kg=7.5,
            transport_per_lot=300.0,
            recovery={"FEM": 0.92, "NFM": 0.80, "PCB": 0.28, "PLA": 0.75},
            price={"FEM": 26.0, "NFM": 155.0, "PCB": 260.0, "PLA": 20.0},
            accepted_categories=frozenset(BENCHMARK_CATEGORIES),
            allowed_battery_states=frozenset({"confirmed_absent"}),
            authorized=True,
        ),
        Route(
            route_id="ROUTE-BAT-QUARANTINE",
            capacity_kg=900.0 * scale_factor,
            cost_per_kg=26.0,
            transport_per_lot=1000.0,
            recovery={"FEM": 0.75, "NFM": 0.75, "PCB": 0.78, "PLA": 0.35},
            price={"FEM": 24.0, "NFM": 165.0, "PCB": 520.0, "PLA": 12.0},
            accepted_categories=frozenset(BENCHMARK_CATEGORIES),
            allowed_battery_states=frozenset(
                {"confirmed_absent", "unknown", "suspect", "confirmed_present"}
            ),
            authorized=True,
        ),
    ]


def _generate_joint_lot_scenarios(
    base_prior: Dict[str, float],
    lot_uncertainty_mult: float,
    regime_pcb_factors: np.ndarray,
    regime_nfm_factors: np.ndarray,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    """Generate simplex-normalized scenarios coupling systemic collection-grade factors with lot noise."""
    scenarios: List[Dict[str, float]] = [normalize_simplex_dict(base_prior)]
    for s_idx in range(len(regime_pcb_factors)):
        pcb_f = float(regime_pcb_factors[s_idx])
        nfm_f = float(regime_nfm_factors[s_idx])
        # Lot-specific idiosyncratic perturbation scaled by lot uncertainty multiplier
        lot_pcb_mult = max(0.10, 1.0 + (pcb_f - 1.0) * lot_uncertainty_mult + rng.normal(0.0, 0.08 * lot_uncertainty_mult))
        lot_nfm_mult = max(0.15, 1.0 + (nfm_f - 1.0) * lot_uncertainty_mult + rng.normal(0.0, 0.08 * lot_uncertainty_mult))
        lot_fem_mult = max(0.20, 1.0 + rng.normal(0.0, 0.06))

        new_pcb = base_prior["PCB"] * lot_pcb_mult
        new_nfm = base_prior["NFM"] * lot_nfm_mult
        new_fem = base_prior["FEM"] * lot_fem_mult
        # Remaining mass goes into PLA matrix
        new_pla = max(0.02, 1.0 - (new_pcb + new_nfm + new_fem))
        scenarios.append(
            normalize_simplex_dict(
                {"FEM": new_fem, "NFM": new_nfm, "PCB": new_pcb, "PLA": new_pla}
            )
        )
    return scenarios


def generate_instance(
    n_lots: int = 20,
    n_cal_scenarios: int = 12,
    n_eval_scenarios: int = 30,
    seed: int = 42,
) -> tuple[List[Lot], List[Lot], List[Route]]:
    rng = np.random.default_rng(seed)
    cal_lots: List[Lot] = []
    eval_lots: List[Lot] = []

    # Calibration systemic collection-grade regimes (includes downside dilution regimes)
    cal_pcb_regimes = np.concatenate([
        np.linspace(0.30, 1.35, n_cal_scenarios - 2),
        np.array([0.22, 0.18]),
    ])
    cal_nfm_regimes = np.concatenate([
        np.linspace(0.45, 1.25, n_cal_scenarios - 2),
        np.array([0.35, 0.30]),
    ])

    # Held-out evaluation regimes (out-of-sample draws from collection quality distribution)
    eval_pcb_regimes = np.clip(rng.normal(0.88, 0.38, size=n_eval_scenarios), 0.15, 1.60)
    eval_nfm_regimes = np.clip(rng.normal(0.92, 0.28, size=n_eval_scenarios), 0.25, 1.50)
    # Ensure realistic tail stress events exist in held-out evaluation
    eval_pcb_regimes[0] = 0.18
    eval_nfm_regimes[0] = 0.32
    eval_pcb_regimes[1] = 0.24
    eval_nfm_regimes[1] = 0.40

    for i in range(n_lots):
        cat = str(rng.choice(BENCHMARK_CATEGORIES, p=[0.40, 0.25, 0.20, 0.08, 0.07]))
        mass_kg = float(np.round(rng.uniform(130.0, 290.0), 1))
        bat_state = "suspect" if (i % 7 == 0) else "confirmed_absent"
        base_prior = dict(get_category_prior(cat))

        # High-entropy / unverified supplier lots have wider compositional spread (1.35x) vs low-entropy lots (0.45x)
        is_high_uncertainty_lot = (i % 2 == 0)
        unc_mult = 1.35 if is_high_uncertainty_lot else 0.45

        cal_scen = _generate_joint_lot_scenarios(
            base_prior, unc_mult, cal_pcb_regimes, cal_nfm_regimes, rng
        )
        eval_scen = _generate_joint_lot_scenarios(
            base_prior, unc_mult, eval_pcb_regimes, eval_nfm_regimes, rng
        )

        cal_lots.append(
            Lot(
                lot_id=f"LOT-{i + 1:03d}",
                mass_kg=mass_kg,
                scenarios=cal_scen,
                category=cat,
                battery_status=bat_state,
            )
        )
        eval_lots.append(
            Lot(
                lot_id=f"LOT-{i + 1:03d}",
                mass_kg=mass_kg,
                scenarios=eval_scen,
                category=cat,
                battery_status=bat_state,
            )
        )

    scale = n_lots / 20.0
    routes = build_benchmark_routes(scale_factor=scale)
    return cal_lots, eval_lots, routes


def run_benchmark_suite(n_replications: int = 30, base_seed: int = 100) -> Dict[str, Any]:
    policies = ("FIXED_ROUTE", "MEAN_GREEDY", "EXPECTED", "ROBUST", "CVAR")
    agg: Dict[str, Dict[str, List[float]]] = {
        p: {
            "expected_profit": [],
            "worst_profit": [],
            "cvar_profit": [],
            "mean_regret": [],
            "worst_regret": [],
            "wall_time_ms": [],
            "capacity_exceedance_kg": [],
        }
        for p in policies
    }

    for rep in range(n_replications):
        seed = base_seed + rep
        cal_lots, eval_lots, routes = generate_instance(
            n_lots=20,
            n_cal_scenarios=12,
            n_eval_scenarios=30,
            seed=seed,
        )
        comp = compare_policies(cal_lots, routes, evaluation_lots=eval_lots, policies=policies)
        for p in policies:
            p_res = comp["policies"][p]
            ev = p_res.get("held_out_evaluation", {})
            if ev:
                agg[p]["expected_profit"].append(float(ev["expected_profit_inr"]))
                agg[p]["worst_profit"].append(float(ev["worst_scenario_profit_inr"]))
                agg[p]["cvar_profit"].append(float(ev["cvar_80_profit_inr"]))
                agg[p]["mean_regret"].append(float(ev.get("mean_regret_inr", 0.0)))
                agg[p]["worst_regret"].append(float(ev.get("worst_case_regret_inr", 0.0)))
                agg[p]["wall_time_ms"].append(float(p_res.get("wall_time_ms", 1.0)))
                agg[p]["capacity_exceedance_kg"].append(
                    float(p_res.get("violations", {}).get("capacity_exceedance_kg", 0.0))
                )

    summary_table: Dict[str, Any] = {}
    for p in policies:
        m = agg[p]
        summary_table[p] = {
            "mean_expected_profit_inr": round(float(np.mean(m["expected_profit"])), 2),
            "ci95_expected_profit_inr": round(
                float(1.96 * np.std(m["expected_profit"]) / np.sqrt(n_replications)), 2
            ),
            "mean_worst_scenario_profit_inr": round(float(np.mean(m["worst_profit"])), 2),
            "mean_cvar80_profit_inr": round(float(np.mean(m["cvar_profit"])), 2),
            "mean_regret_inr": round(float(np.mean(m["mean_regret"])), 2),
            "mean_worst_case_regret_inr": round(float(np.mean(m["worst_regret"])), 2),
            "mean_capacity_exceedance_kg": round(
                float(np.mean(m["capacity_exceedance_kg"])), 2
            ),
            "solver_p50_ms": round(float(np.percentile(m["wall_time_ms"], 50)), 2),
            "solver_p95_ms": round(float(np.percentile(m["wall_time_ms"], 95)), 2),
        }

    greedy_worst_reg = summary_table["MEAN_GREEDY"]["mean_worst_case_regret_inr"]
    robust_worst_reg = summary_table["ROBUST"]["mean_worst_case_regret_inr"]
    cvar_worst_reg = summary_table["CVAR"]["mean_worst_case_regret_inr"]

    regret_reduction_robust_pct = (
        (greedy_worst_reg - robust_worst_reg) / max(greedy_worst_reg, 1e-6) * 100.0
    )
    regret_reduction_cvar_pct = (
        (greedy_worst_reg - cvar_worst_reg) / max(greedy_worst_reg, 1e-6) * 100.0
    )

    demo_cal_lots, _, demo_routes = generate_instance(n_lots=15, seed=42)
    syn_dir = ROOT / "data" / "synthetic"
    syn_dir.mkdir(parents=True, exist_ok=True)
    (syn_dir / "scenarios.json").write_text(
        json.dumps(
            {
                "master_seed": 42,
                "lots": [
                    {
                        "lot_id": l.lot_id,
                        "mass_kg": l.mass_kg,
                        "category": l.category,
                        "battery_status": l.battery_status,
                        "scenarios": l.scenarios,
                    }
                    for l in demo_cal_lots
                ],
                "routes": [
                    {
                        "route_id": r.route_id,
                        "capacity_kg": r.capacity_kg,
                        "cost_per_kg": r.cost_per_kg,
                        "transport_per_lot": r.transport_per_lot,
                        "recovery": r.recovery,
                        "price": r.price,
                        "accepted_categories": sorted(r.accepted_categories),
                        "allowed_battery_states": sorted(r.allowed_battery_states),
                        "authorized": r.authorized,
                    }
                    for r in demo_routes
                ],
            },
            indent=2,
        )
    )

    out = {
        "n_replications": n_replications,
        "lots_per_replication": 20,
        "calibration_scenarios": 13,
        "held_out_evaluation_scenarios": 31,
        "policies": summary_table,
        "robust_worst_case_regret_reduction_vs_greedy_pct": round(
            float(regret_reduction_robust_pct), 2
        ),
        "cvar_worst_case_regret_reduction_vs_greedy_pct": round(
            float(regret_reduction_cvar_pct), 2
        ),
    }
    out_path = ROOT / "data" / "processed" / "benchmark_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("Benchmark Suite Summary:")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run_benchmark_suite()
