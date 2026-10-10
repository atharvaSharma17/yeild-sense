import type { Batch, BatchImage, Fractions } from "@/types/batch";

const standardMix: Fractions = { FEM: 0.33, NFM: 0.17, PCB: 0.12, PLA: 0.28, OTHER: 0.1 };
const plasticsHeavyMix: Fractions = { FEM: 0.17, NFM: 0.08, PCB: 0.06, PLA: 0.56, OTHER: 0.13 };

export const mockBatches: Batch[] = [
  {
    id: "bch_0148",
    version: 3,
    created_at: "2026-10-10T08:24:00Z",
    category: "small_devices",
    mass_kg: 480,
    mass_source: "scale",
    battery_status: "unknown",
    review_status: "pending",
    review_evidence: "",
    composition: { source: "category_prior", evidence: "Verified category prior; demo record", scenarios: [standardMix] },
  },
  {
    id: "bch_0147",
    version: 2,
    created_at: "2026-10-09T15:47:00Z",
    category: "mixed_it_equipment",
    mass_kg: 325,
    mass_source: "scale",
    battery_status: "cleared",
    review_status: "approved",
    review_evidence: "Intake inspection completed by facility reviewer.",
    composition: { source: "operator_entered", evidence: "Sort-floor estimate; demo record", scenarios: [standardMix, plasticsHeavyMix] },
  },
  {
    id: "bch_0146",
    version: 1,
    created_at: "2026-10-09T10:05:00Z",
    category: "consumer_electronics",
    mass_kg: 92,
    mass_source: "scale",
    battery_status: "suspected",
    review_status: "pending",
    review_evidence: "",
  },
  {
    id: "bch_0145",
    version: 4,
    created_at: "2026-10-08T13:12:00Z",
    category: "small_devices",
    mass_kg: 210,
    mass_source: "illustrative",
    battery_status: "present",
    review_status: "approved",
    review_evidence: "Known battery-containing devices isolated for approved hold.",
    composition: { source: "measured", evidence: "Weighed sorting sample; demo record", scenarios: [plasticsHeavyMix] },
  },
];

export const mockImages: BatchImage[] = [
  {
    id: "img_0832",
    batch_id: "bch_0148",
    object_key: "raw/bch_0148/img_0832.jpg",
    sha256: "7b7efaf3ee1df1d1e4e172a2d7a7651f9b1d4450bd9e8935c44c20a479e0af09",
    size_bytes: 824_000,
    content_type: "image/jpeg",
    created_at: "2026-10-10T08:30:00Z",
  },
];
