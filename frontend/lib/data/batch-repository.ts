"use client";

import { mockBatches, mockImages } from "@/lib/data/mock-batches";
import type { ApiProblem, Batch, BatchDraft, BatchImage, BatchListResponse, Composition, Prediction, PredictionResponse, ModelManifest } from "@/types/batch";

const STORAGE_KEY = "yieldsense.phase-two.batches";
const IMAGE_STORAGE_KEY = "yieldsense.phase-two.images";
const useMockData = process.env.NEXT_PUBLIC_YIELDSENSE_DATA_MODE !== "api";

function clone<T>(value: T): T { return JSON.parse(JSON.stringify(value)) as T; }
function wait(ms = 420) { return new Promise((resolve) => window.setTimeout(resolve, ms)); }
function getBatches(): Batch[] {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return stored ? JSON.parse(stored) as Batch[] : clone(mockBatches);
}
function saveBatches(batches: Batch[]) { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(batches)); }
function getImages(): BatchImage[] {
  const stored = window.localStorage.getItem(IMAGE_STORAGE_KEY);
  return stored ? JSON.parse(stored) as BatchImage[] : clone(mockImages);
}
function saveImages(images: BatchImage[]) { window.localStorage.setItem(IMAGE_STORAGE_KEY, JSON.stringify(images)); }

export class RepositoryError extends Error {
  constructor(message: string, public readonly status: number, public readonly requestId?: string) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/yieldsense${path}`, { cache: "no-store", ...init });
  if (!response.ok) {
    const issue = await response.json().catch(() => ({ error: "The request could not be completed." })) as ApiProblem;
    throw new RepositoryError(issue.error, response.status, issue.request_id);
  }
  return response.json() as Promise<T>;
}

export const batchRepository = {
  isMockData: useMockData,
  async list(): Promise<BatchListResponse> {
    if (!useMockData) return request<BatchListResponse>("/v1/batches?limit=25");
    await wait();
    return { items: getBatches().sort((a, b) => b.created_at.localeCompare(a.created_at)), next_cursor: "" };
  },
  async get(id: string): Promise<Batch> {
    if (!useMockData) return request<Batch>(`/v1/batches/${id}`);
    await wait(260);
    const batch = getBatches().find((item) => item.id === id);
    if (!batch) throw new Error("Batch record not found.");
    return batch;
  },
  async create(draft: BatchDraft): Promise<Batch> {
    if (!useMockData) return request<Batch>("/v1/batches", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(draft) });
    await wait(650);
    const batch: Batch = { ...draft, id: `bch_${Math.random().toString(36).slice(2, 8)}`, version: 1, created_at: new Date().toISOString(), battery_status: "unknown", review_status: "pending", review_evidence: "" };
    const batches = [batch, ...getBatches()];
    saveBatches(batches);
    return batch;
  },
  async update(batch: Batch): Promise<Batch> {
    if (!useMockData) return request<Batch>(`/v1/batches/${batch.id}`, { method: "PUT", headers: { "Content-Type": "application/json", "If-Match": String(batch.version) }, body: JSON.stringify(batch) });
    await wait(520);
    const updated = { ...batch, version: batch.version + 1 };
    saveBatches(getBatches().map((item) => item.id === batch.id ? updated : item));
    return updated;
  },
  async attachComposition(id: string, version: number, composition: Composition): Promise<Batch> {
    if (!useMockData) return request<Batch>("/v1/composition", { method: "POST", headers: { "Content-Type": "application/json", "If-Match": String(version) }, body: JSON.stringify({ batch_id: id, composition }) });
    const current = await this.get(id);
    return this.update({ ...current, composition });
  },
  async images(id: string): Promise<BatchImage[]> {
    if (!useMockData) {
      const images: BatchImage[] = [];
      let cursor = "";
      do {
        const page = await request<{ items: BatchImage[]; next_cursor: string }>(`/v1/batches/${encodeURIComponent(id)}/images?limit=100${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`);
        images.push(...page.items.map((image) => ({ ...image, preview_url: `/api/yieldsense/v1/images/${encodeURIComponent(image.id)}/content` })));
        cursor = page.next_cursor;
      } while (cursor);
      return images.sort((a, b) => b.created_at.localeCompare(a.created_at));
    }
    await wait(180);
    return getImages().filter((image) => image.batch_id === id);
  },
  async uploadImage(batchId: string, file: File): Promise<BatchImage> {
    if (!useMockData) {
      const image = await request<BatchImage>(`/v1/batches/${batchId}/images`, { method: "POST", headers: { "Content-Type": file.type }, body: file });
      return { ...image, preview_url: `/api/yieldsense/v1/images/${encodeURIComponent(image.id)}/content` };
    }
    await wait(700);
    const item: BatchImage = { id: `img_${Math.random().toString(36).slice(2, 8)}`, batch_id: batchId, object_key: `raw/${batchId}/mock-upload.${file.type === "image/png" ? "png" : "jpg"}`, sha256: "local-preview-only", size_bytes: file.size, content_type: file.type as BatchImage["content_type"], created_at: new Date().toISOString(), preview_url: URL.createObjectURL(file) };
    saveImages([...getImages(), item]);
    return item;
  },
  async predictDevice(batchId: string, imageId: string, version: number, modelVersion?: string): Promise<PredictionResponse> {
    if (useMockData) throw new RepositoryError("Demo mode does not fabricate AI predictions. Connect a configured inference worker to run device triage.", 503);
    return request<PredictionResponse>("/v1/predict-device", { method: "POST", headers: { "Content-Type": "application/json", "If-Match": String(version) }, body: JSON.stringify({ batch_id: batchId, image_id: imageId, ...(modelVersion ? { model_version: modelVersion } : {}) }) });
  },
  async getPrediction(id: string): Promise<Prediction> {
    if (useMockData) throw new RepositoryError("No model evidence is available in demo mode.", 404);
    return request<Prediction>(`/v1/predictions/${id}`);
  },
  async models(): Promise<ModelManifest[]> {
    if (useMockData) return [];
    const models: ModelManifest[] = [];
    let cursor = "";
    do {
      const page = await request<{ items: ModelManifest[]; next_cursor: string }>(`/v1/models?limit=100${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`);
      models.push(...page.items);
      cursor = page.next_cursor;
    } while (cursor);
    return models;
  },
  async predictComposition(batchId: string, imageId: string, version: number, modelVersion: string): Promise<PredictionResponse> {
    if (useMockData) throw new RepositoryError("Model inference is unavailable in demo mode.", 503);
    return request<PredictionResponse>("/v1/predict-composition", { method: "POST", headers: { "Content-Type": "application/json", "If-Match": String(version) }, body: JSON.stringify({ batch_id: batchId, image_id: imageId, model_version: modelVersion }) });
  },
};
