export const MATERIALS = ["FEM", "NFM", "PCB", "PLA", "OTHER"] as const;
export type Material = (typeof MATERIALS)[number];
export type Fractions = Record<Material, number>;
export type BatteryStatus = "unknown" | "suspected" | "present" | "cleared";
export type ReviewStatus = "pending" | "approved";
export type CompositionSource = "measured" | "category_prior" | "operator_entered" | "model";

export type Composition = {
  track?: "A" | "B";
  model_output_id?: string;
  source: CompositionSource;
  evidence: string;
  model_version?: string;
  scenarios: Fractions[];
};

export type Batch = {
  id: string;
  version: number;
  created_at: string;
  category: string;
  mass_kg: number;
  mass_source: "scale" | "illustrative";
  battery_status: BatteryStatus;
  review_status: ReviewStatus;
  review_evidence: string;
  latest_prediction_id?: string;
  composition?: Composition;
};

export type BatchImage = {
  id: string;
  batch_id: string;
  object_key: string;
  sha256: string;
  size_bytes: number;
  content_type: "image/jpeg" | "image/png";
  created_at: string;
  preview_url?: string;
};

export type BatchListResponse = { items: Batch[]; next_cursor: string };

export type BatchDraft = Pick<Batch, "category" | "mass_kg" | "mass_source"> & {
  composition?: Composition;
};

export type ApiProblem = { error: string; request_id?: string };

export type ModelManifest = {
  model_version: string;
  created_at?: string;
  contract_version: string;
  track: "A" | "B";
  task: "device_classification" | "particle_composition";
  architecture: string;
  artifact_s3_uri: string;
  artifact_sha256: string;
  dataset_manifest_sha256: string;
  data_version: string;
  training_job_arn: string;
  labels: string[];
  battery_review_labels?: string[];
  confidence_threshold: number;
  evaluation: Record<string, number>;
  paired_mass_evidence?: string;
};

export type PredictionResult = {
  contract_version: string;
  request_id: string;
  model_version: string;
  track: "A" | "B";
  task: "device_classification" | "particle_composition";
  probabilities?: Record<string, number>;
  abstained: boolean;
  abstention_reason?: string;
  battery_alert: "unknown" | "suspected" | "none";
  material_mass_kg?: Fractions;
  scenarios?: Fractions[];
};

export type PredictionAssessment = {
  predicted_class?: string;
  confidence: number;
  abstained: boolean;
  reasons: string[];
  composition?: Composition;
  estimated_mass_kg?: number;
  reconciliation_factor?: number;
  mass_discrepancy_kg?: number;
};

export type Prediction = {
  id: string;
  batch_id: string;
  batch_version: number;
  image: BatchImage;
  model: ModelManifest;
  created_at: string;
  result: PredictionResult;
  assessment: PredictionAssessment;
  human_review_required: boolean;
};

export type PredictionResponse = { prediction: Prediction; batch: Batch };
