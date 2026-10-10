import type { BatteryStatus, Batch, CompositionSource, ReviewStatus } from "@/types/batch";

export function formatMass(value: number) { return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 1 }).format(value); }
export function formatDate(value: string) { return new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value)); }
export function batchShortId(value: string) { return value.replace("bch_", "#").toUpperCase(); }
export function safetyTone(batch: Pick<Batch, "battery_status" | "review_status">): "safe" | "review" | "risk" {
  if (batch.review_status === "approved" && batch.battery_status === "cleared") return "safe";
  if (batch.battery_status === "present" || batch.battery_status === "suspected") return "risk";
  return "review";
}
export function safetyLabel(batch: Pick<Batch, "battery_status" | "review_status">) {
  if (batch.review_status === "approved" && batch.battery_status === "cleared") return "Cleared + approved";
  if (batch.battery_status === "present") return "Approved hold only";
  if (batch.battery_status === "suspected") return "Suspected battery";
  return "Review required";
}
export const sourceLabel: Record<CompositionSource, string> = { measured: "Measured", category_prior: "Category prior", operator_entered: "Operator entered", model: "Model estimate" };
export const batteryLabel: Record<BatteryStatus, string> = { unknown: "Unknown", suspected: "Suspected", present: "Present", cleared: "Cleared" };
export const reviewLabel: Record<ReviewStatus, string> = { pending: "Pending", approved: "Approved" };
