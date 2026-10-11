"use client";

import { useEffect, useState } from "react";
import { batchRepository, RepositoryError } from "@/lib/data/batch-repository";
import { MATERIALS, type Batch, type BatchImage, type ModelManifest, type Prediction } from "@/types/batch";

export function ModelInferencePanel({ batch, images, prediction, onPredicted, onReload }: {
  batch: Batch; images: BatchImage[]; prediction: Prediction | null;
  onPredicted: (batch: Batch, prediction: Prediction) => void; onReload: () => void;
}) {
  const [models, setModels] = useState<ModelManifest[]>([]);
  const [modelVersion, setModelVersion] = useState("");
  const [imageId, setImageId] = useState("");
  const [running, setRunning] = useState(false);
  const [loadingModels, setLoadingModels] = useState(true);
  const [error, setError] = useState("");
  const [conflict, setConflict] = useState(false);
  const model = models.find((item) => item.model_version === modelVersion);
  const selectedImage = images.some((image) => image.id === imageId) ? imageId : images[0]?.id ?? "";

  useEffect(() => {
    let cancelled = false;
    batchRepository.models().then((items) => {
      if (cancelled) return;
      setModels(items);
      setModelVersion(items.find((item) => item.track === "B")?.model_version ?? items[0]?.model_version ?? "");
    }).catch((issue) => { if (!cancelled) setError(issue instanceof Error ? issue.message : "Unable to load models."); })
      .finally(() => { if (!cancelled) setLoadingModels(false); });
    return () => { cancelled = true; };
  }, []);

  async function run() {
    if (!model || !selectedImage) return;
    setRunning(true); setError(""); setConflict(false);
    try {
      const result = model.task === "particle_composition"
        ? await batchRepository.predictComposition(batch.id, selectedImage, batch.version, model.model_version)
        : await batchRepository.predictDevice(batch.id, selectedImage, batch.version, model.model_version);
      onPredicted(result.batch, result.prediction);
    } catch (issue) {
      const status = issue instanceof RepositoryError ? issue.status : undefined;
      setConflict(status === 409);
      setError(status === 409 ? "This batch changed. Reload it before trying again."
        : status === 503 ? "The ML worker is unavailable. Human review and manual composition remain available."
        : status === 504 ? "Inference timed out. Reload the batch before retrying to check whether a result was saved."
        : issue instanceof Error ? issue.message : "Inference could not be completed.");
    } finally { setRunning(false); }
  }

  return <section className="border-[1.5px] border-ink bg-[#EEEADD] p-5">
    <p className="font-mono text-xs uppercase tracking-wider text-moss">Image analysis</p>
    <h2 className="mt-2 font-display text-2xl font-semibold">Run a registered model</h2>
    <p className="mt-2 text-sm leading-6 text-ink/65">Device triage uses whole-device photos. Particle composition uses compatible shredded-particle images. Every result still requires human review.</p>
    {batchRepository.isMockData ? <p className="mt-4 text-sm">Inference is unavailable in demo mode.</p>
      : loadingModels ? <p className="mt-4 text-sm">Loading available models…</p>
      : models.length === 0 ? <p className="mt-4 text-sm">No models are registered yet. Register a trained model before running image analysis.</p>
      : <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <label className="text-sm font-medium">Model
          <select value={modelVersion} disabled={running} onChange={(event) => setModelVersion(event.target.value)} className="mt-2 w-full border border-ink bg-ivory p-3">
            {models.map((item) => <option key={item.model_version} value={item.model_version}>Track {item.track} · {item.model_version}</option>)}
          </select>
        </label>
        <label className="text-sm font-medium">Image evidence
          <select value={selectedImage} disabled={running || images.length === 0} onChange={(event) => setImageId(event.target.value)} className="mt-2 w-full border border-ink bg-ivory p-3">
            {images.length === 0 && <option value="">Upload an image first</option>}
            {images.map((image) => <option key={image.id} value={image.id}>{image.created_at} · {image.id}</option>)}
          </select>
        </label>
        <button onClick={() => void run()} disabled={running || !selectedImage || !model} className="border border-ink bg-ink px-4 py-3 text-sm font-semibold text-paper disabled:opacity-50">
          {running ? "Analyzing image…" : model?.track === "A" ? "Run particle composition" : "Run device triage"}
        </button>
      </div>}
    {error && <div role="alert" className="mt-4 border-l-4 border-clay bg-[#F4D8CE] p-3 text-sm">
      {error}{conflict && <button onClick={onReload} className="ml-3 underline">Reload batch</button>}
    </div>}
    {prediction && <div className="mt-5 border border-ink bg-ivory p-4">
      <p className="font-mono text-xs">{prediction.model.model_version} · Track {prediction.model.track}</p>
      <p className="mt-2 font-display text-xl font-semibold">{prediction.assessment.abstained ? "Model abstained — review required" : prediction.model.track === "A" ? "Composition evidence saved" : prediction.assessment.predicted_class ?? "No supported class"}</p>
      {prediction.model.track === "B" && !prediction.assessment.abstained && <p className="mt-2 text-sm">Confidence: {(prediction.assessment.confidence * 100).toFixed(1)}%</p>}
      <p className="mt-2 text-sm">Battery alert: {prediction.result.battery_alert}. This does not clear battery status.</p>
      {prediction.model.track === "A" && !prediction.assessment.abstained && batch.composition && <div className="mt-3 grid grid-cols-5 gap-2 text-sm">
        {MATERIALS.map((material) => <div key={material}>{material}<br />{(batch.composition!.scenarios[0][material] * 100).toFixed(1)}%</div>)}
      </div>}
      {prediction.assessment.reasons?.length > 0 && <p className="mt-3 text-sm">{prediction.assessment.reasons.join(" · ")}</p>}
    </div>}
  </section>;
}
