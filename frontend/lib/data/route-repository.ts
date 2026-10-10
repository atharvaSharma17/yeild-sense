"use client";

import { RepositoryError } from "@/lib/data/batch-repository";
import { mockRoutes } from "@/lib/data/mock-routes";
import type { ApiProblem } from "@/types/batch";
import type { Route, RouteDraft, RouteListResponse } from "@/types/route";

const STORAGE_KEY = "yieldsense.phase-four.routes";
const useMockData = process.env.NEXT_PUBLIC_YIELDSENSE_DATA_MODE !== "api";
const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
const wait = (ms = 360) => new Promise((resolve) => window.setTimeout(resolve, ms));
function stored(): Route[] { const value = window.localStorage.getItem(STORAGE_KEY); return value ? JSON.parse(value) as Route[] : clone(mockRoutes); }
function save(routes: Route[]) { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(routes)); }
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/yieldsense${path}`, { cache: "no-store", ...init });
  if (!response.ok) { const issue = await response.json().catch(() => ({ error: "The request could not be completed." })) as ApiProblem; throw new RepositoryError(issue.error, response.status, issue.request_id); }
  return response.json() as Promise<T>;
}

export const routeRepository = {
  isMockData: useMockData,
  async list(): Promise<RouteListResponse> { if (!useMockData) return request<RouteListResponse>("/v1/routes?limit=50"); await wait(); return { items: stored(), next_cursor: "" }; },
  async get(id: string): Promise<Route> { if (!useMockData) return request<Route>(`/v1/routes/${id}`); await wait(220); const item = stored().find((route) => route.id === id); if (!item) throw new RepositoryError("Route record not found.", 404); return item; },
  async create(draft: RouteDraft): Promise<Route> { if (!useMockData) return request<Route>("/v1/routes", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(draft) }); await wait(500); const route: Route = { ...draft, id: `rte_${Math.random().toString(36).slice(2, 8)}`, version: 1, created_at: new Date().toISOString() }; save([route, ...stored()]); return route; },
  async update(route: Route): Promise<Route> { if (!useMockData) return request<Route>(`/v1/routes/${route.id}`, { method: "PUT", headers: { "Content-Type": "application/json", "If-Match": String(route.version) }, body: JSON.stringify(route) }); await wait(450); const updated = { ...route, version: route.version + 1 }; save(stored().map((item) => item.id === route.id ? updated : item)); return updated; },
};
