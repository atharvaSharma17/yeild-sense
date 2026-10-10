"use client";

import { batchRepository, RepositoryError } from "@/lib/data/batch-repository";
import { routeRepository } from "@/lib/data/route-repository";
import type { ApiProblem } from "@/types/batch";
import type { OptimizationRequest, OptimizationRun, SimulationRequest } from "@/types/route";

const STORAGE_KEY = "yieldsense.phase-four.runs";
const useMockData = process.env.NEXT_PUBLIC_YIELDSENSE_DATA_MODE !== "api";
const wait = (ms = 760) => new Promise((resolve) => window.setTimeout(resolve, ms));
function stored(): OptimizationRun[] { return JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "[]") as OptimizationRun[]; }
function save(value: OptimizationRun[]) { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value)); }
async function request<T>(path: string, init?: RequestInit): Promise<T> { const response = await fetch(`/api/yieldsense${path}`, { cache: "no-store", ...init }); if (!response.ok) { const body = await response.json().catch(() => ({ error: "The request could not be completed." })) as ApiProblem; throw new RepositoryError(body.error, response.status, body.request_id); } return response.json() as Promise<T>; }
export const optimizationRepository = {
  isMockData: useMockData,
  async optimize(input: OptimizationRequest): Promise<OptimizationRun> {
    if (!useMockData) return request<OptimizationRun>("/v1/optimize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input) });
    await wait();
    const [batchPage, routePage] = await Promise.all([batchRepository.list(), routeRepository.list()]);
    const run: OptimizationRun = { id: `demo_run_${Math.random().toString(36).slice(2, 8)}`, created_at: new Date().toISOString(), input_sha256: "demo-local-not-audit-hash", input: { batches: batchPage.items.filter((batch) => input.batch_ids.includes(batch.id)), routes: routePage.items.filter((route) => input.route_ids.includes(route.id)), objective: input.objective, ...(input.fixed_route_id ? { fixed_route_id: input.fixed_route_id } : {}), assumptions: input.assumptions }, currency: "INR", mass_unit: "kg", solver: "local demo preview", limitations: ["Demo output is illustrative; it was not computed by the backend optimizer.", "Values are estimates, not realized profit or verified savings.", "A live run performs its own server-side eligibility and scenario validation."], result: { status: "FEASIBLE", assignments: input.batch_ids.map((batch_id) => ({ batch_id, route_id: input.objective === "FIXED" ? input.fixed_route_id! : input.route_ids[0] })), scenario_values: [12800], expected_value: 12800, worst_value: 12800, diagnostics: ["Local demo preview only — connect the optimization service for solver diagnostics."], nodes: 0, elapsed_ms: 0 } };
    save([run, ...stored()]); return run;
  },
  async get(id: string): Promise<OptimizationRun> { if (!useMockData) return request<OptimizationRun>(`/v1/runs/${id}`); await wait(220); const run = stored().find((item) => item.id === id); if (!run) throw new RepositoryError("Run report not found in this browser.", 404); return run; },
  async simulate(input: SimulationRequest): Promise<OptimizationRun> {
    if (!useMockData) return request<OptimizationRun>("/v1/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input) });
    await wait(800); const base = stored().find((item) => item.id === input.base_run_id); if (!base) throw new RepositoryError("Select an existing local baseline run before simulating.", 404);
    const routes = base.input.routes.map((route) => { const change = input.overrides.find((item) => item.route_id === route.id); return change ? { ...route, ...(change.capacity_kg !== undefined ? { capacity_kg: change.capacity_kg } : {}), ...(change.price ? { price: change.price } : {}) } : route; });
    const shutdown = input.overrides.some((item) => item.capacity_kg === 0); const expected = shutdown ? undefined : (base.result.expected_value ?? 0) * 0.91; const worst = shutdown ? undefined : (base.result.worst_value ?? 0) * 0.88;
    const derived: OptimizationRun = { ...base, id: `demo_sim_${Math.random().toString(36).slice(2, 8)}`, parent_id: base.id, created_at: new Date().toISOString(), input_sha256: "demo-local-not-audit-hash", input: { ...base.input, routes, objective: input.objective ?? base.input.objective, assumptions: [...base.input.assumptions, ...input.assumptions] }, result: { ...base.result, status: shutdown ? "INFEASIBLE" : "FEASIBLE", expected_value: expected, worst_value: worst, scenario_values: expected === undefined ? [] : [expected], assignments: shutdown ? [] : base.result.assignments, diagnostics: [shutdown ? "A local route shutdown created an illustrative infeasible scenario." : "Local demo preview only — connect the optimization service for solver diagnostics."] }, limitations: ["Demo output is illustrative; it was not computed by the backend simulator.", ...base.limitations] };
    save([derived, ...stored()]); return derived;
  },
};
