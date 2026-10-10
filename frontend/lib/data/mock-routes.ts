import type { Route } from "@/types/route";

const recovery = { FEM: 0.82, NFM: 0.7, PCB: 0.9, PLA: 0.58, OTHER: 0.12 };
const price = { FEM: 32, NFM: 106, PCB: 290, PLA: 18, OTHER: 2 };

export const mockRoutes: Route[] = [
  { id: "rte_hold_01", version: 3, created_at: "2026-10-02T10:00:00Z", name: "Secure recovery hold", capacity_kg: 900, cost_per_kg: 12, transport_per_lot: 450, recovery, price, accepted_categories: ["small_devices", "consumer_electronics", "mixed_it_equipment"], allowed_battery_states: ["unknown", "suspected", "present", "cleared"], authorized: true, authorization_evidence: "Facility authorization RA-48, current through 2027.", approved_hold: true, economics_source: "Quarterly facility schedule, Oct 2026" },
  { id: "rte_mech_02", version: 5, created_at: "2026-09-29T10:00:00Z", name: "Mechanical separation", capacity_kg: 680, cost_per_kg: 8, transport_per_lot: 280, recovery: { FEM: 0.77, NFM: 0.64, PCB: 0.76, PLA: 0.66, OTHER: 0.1 }, price: { FEM: 30, NFM: 101, PCB: 260, PLA: 16, OTHER: 1 }, accepted_categories: ["small_devices", "mixed_it_equipment"], allowed_battery_states: ["cleared"], authorized: true, authorization_evidence: "Vendor permit reference MS-113.", approved_hold: false, economics_source: "Vendor quote, Sep 2026" },
  { id: "rte_pcb_03", version: 1, created_at: "2026-10-06T10:00:00Z", name: "Board recovery partner", capacity_kg: 300, cost_per_kg: 16, transport_per_lot: 650, recovery: { FEM: 0.4, NFM: 0.82, PCB: 0.96, PLA: 0.22, OTHER: 0.04 }, price: { FEM: 28, NFM: 110, PCB: 340, PLA: 12, OTHER: 0 }, accepted_categories: ["consumer_electronics"], allowed_battery_states: ["cleared"], authorized: false, authorization_evidence: "", approved_hold: false, economics_source: "Indicative pricing; authorization pending" },
];
