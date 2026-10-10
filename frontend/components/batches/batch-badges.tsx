import { safetyLabel, safetyTone, sourceLabel } from "@/lib/format";
import type { Batch } from "@/types/batch";
import { StatusStamp } from "@/components/ui/status-stamp";

export function SafetyBadge({ batch }: { batch: Pick<Batch, "battery_status" | "review_status"> }) {
  return <StatusStamp tone={safetyTone(batch)}>{safetyLabel(batch)}</StatusStamp>;
}

export function CompositionBadge({ batch }: { batch: Pick<Batch, "composition"> }) {
  return batch.composition ? <StatusStamp tone="neutral">{sourceLabel[batch.composition.source]}</StatusStamp> : <StatusStamp tone="review">Composition needed</StatusStamp>;
}
