"use client";

import { optimizationRepository } from "@/lib/data/optimization-repository";
import { RepositoryError } from "@/lib/data/batch-repository";
import type { ApiProblem } from "@/types/batch";
import type { OptimizationRun, RunListResponse } from "@/types/route";

const useMockData = process.env.NEXT_PUBLIC_YIELDSENSE_DATA_MODE !== "api";
const STORAGE_KEY = "yieldsense.phase-four.runs";
const wait = (ms = 320) => new Promise((resolve) => window.setTimeout(resolve, ms));
async function request<T>(path: string): Promise<T> { const response = await fetch(`/api/yieldsense${path}`, { cache: "no-store" }); if (!response.ok) { const issue = await response.json().catch(() => ({ error: "The request could not be completed." })) as ApiProblem; throw new RepositoryError(issue.error, response.status, issue.request_id); } return response.json() as Promise<T>; }
export const reportRepository = {
  isMockData: useMockData,
  async list(cursor?: string): Promise<{ items: OptimizationRun[]; nextCursor: string }> {
    if (useMockData) { await wait(); const items = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "[]") as OptimizationRun[]; return { items, nextCursor: "" }; }
    const page = await request<RunListResponse>(`/v1/runs?limit=25${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`);
    const items = await Promise.all(page.items.map((record) => optimizationRepository.get(record.id)));
    return { items, nextCursor: page.next_cursor };
  },
  async downloadImmutableJSON(id: string): Promise<void> {
    if (useMockData) throw new RepositoryError("Demo reports are browser-local previews, not backend-generated immutable JSON artifacts.", 404);
    const response = await fetch(`/api/yieldsense/v1/reports/${id}`, { cache: "no-store", headers: { Accept: "application/json" } });
    if (!response.ok) { const issue = await response.json().catch(() => ({ error: "Unable to download the audit report." })) as ApiProblem; throw new RepositoryError(issue.error, response.status, issue.request_id); }
    const data = await response.blob(); const url = URL.createObjectURL(data); const anchor = document.createElement("a"); anchor.href = url; anchor.download = `yieldsense-audit-${id}.json`; document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url);
  },
};
