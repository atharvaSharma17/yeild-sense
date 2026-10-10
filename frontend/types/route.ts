import type { BatteryStatus, Batch, Fractions, Prediction } from "@/types/batch";

export type Route = {
  id: string;
  version: number;
  created_at: string;
  name: string;
  capacity_kg: number;
  cost_per_kg: number;
  transport_per_lot: number;
  recovery: Fractions;
  price: Fractions;
  accepted_categories: string[];
  allowed_battery_states: BatteryStatus[];
  authorized: boolean;
  authorization_evidence: string;
  approved_hold: boolean;
  economics_source: string;
};

export type RouteDraft = Omit<Route, "id" | "version" | "created_at">;
export type RouteListResponse = { items: Route[]; next_cursor: string };
export type OptimizationPolicy = "ROBUST" | "EXPECTED" | "GREEDY" | "FIXED";
export type OptimizationRequest = {
  batch_ids: string[];
  route_ids: string[];
  objective: OptimizationPolicy;
  fixed_route_id?: string;
  assumptions: string[];
};
export type RunInput = {
  batches: Batch[];
  routes: Route[];
  objective: OptimizationPolicy;
  fixed_route_id?: string;
  assumptions: string[];
};
export type SimulationOverride = { route_id: string; capacity_kg?: number; price?: Fractions };
export type SimulationRequest = { base_run_id: string; objective?: OptimizationPolicy; overrides: SimulationOverride[]; assumptions: string[] };
export type Assignment = { batch_id: string; route_id: string; scenario_values?: number[]; expected_value?: number; worst_value?: number };
export type SolverResult = {
  status: "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "TIME_LIMIT" | "NO_SOLUTION_FOUND";
  objective?: number;
  assignments?: Assignment[];
  scenario_values?: number[];
  expected_value?: number;
  worst_value?: number;
  diagnostics?: string[];
  nodes?: number;
  elapsed_ms?: number;
};
export type OptimizationRun = {
  id: string;
  parent_id?: string;
  created_at: string;
  input_sha256: string;
  input: RunInput;
  result: SolverResult;
  currency: string;
  mass_unit: string;
  solver: string;
  limitations: string[];
  model_outputs?: Prediction[];
};
export type RunListRecord = { id: string; created_at: string; status: SolverResult["status"]; report_key: string; input_sha256: string };
export type RunListResponse = { items: RunListRecord[]; next_cursor: string };
