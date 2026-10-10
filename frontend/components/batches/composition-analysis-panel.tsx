"use client";

import { useEffect, useState } from "react";
import { batchRepository, RepositoryError } from "@/lib/data/batch-repository";
import { sourceLabel } from "@/lib/format";
import { MATERIALS, type Batch, type Composition, type CompositionSource, type Fractions, type Material } from "@/types/batch";
import { StatusStamp } from "@/components/ui/status-stamp";

type EditableScenario = Record<Material, string>;
type EditorState = { source: Exclude<CompositionSource, "model">; evidence: string; scenarios: EditableScenario[] };

function toEditor(composition?: Composition): EditorState {
  return {
    source: composition?.source === "model" ? "operator_entered" : composition?.source ?? "operator_entered",
    evidence: composition?.source === "model" ? "" : composition?.evidence ?? "",
    scenarios: (composition?.scenarios ?? [{ FEM: 0, NFM: 0, PCB: 0, PLA: 0, OTHER: 0 }]).map((scenario) => Object.fromEntries(MATERIALS.map((material) => [material, String(Number((scenario[material] * 100).toFixed(2)))])) as EditableScenario),
  };
}

function total(scenario: EditableScenario) { return MATERIALS.reduce((sum, material) => sum + (Number(scenario[material]) || 0), 0); }
function toComposition(editor: EditorState): Composition {
  return { source: editor.source, evidence: editor.evidence.trim(), scenarios: editor.scenarios.map((scenario) => Object.fromEntries(MATERIALS.map((material) => [material, Number(scenario[material]) / 100])) as Fractions) };
}
function sourceDescription(source: CompositionSource) { return { measured: "Observed or measured fractions with supporting evidence.", category_prior: "A category-level prior; useful context, not a batch assay.", operator_entered: "A documented operator estimate; it remains an assumption.", model: "Track A model output stored as immutable evidence." }[source]; }

export function CompositionAnalysisPanel({ batch, onSaved }: { batch: Batch; onSaved: (batch: Batch) => void }) {
  const [editing, setEditing] = useState(!batch.composition);
  const [activeScenario, setActiveScenario] = useState(0);
  const [form, setForm] = useState<EditorState>(() => toEditor(batch.composition));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const isModelDerived = batch.composition?.source === "model";

  useEffect(() => { setForm(toEditor(batch.composition)); setActiveScenario(0); }, [batch.composition]);

  const updateFraction = (material: Material, value: string) => setForm((current) => ({ ...current, scenarios: current.scenarios.map((scenario, index) => index === activeScenario ? { ...scenario, [material]: value } : scenario) }));
  const addScenario = () => { if (form.scenarios.length < 20) { setForm((current) => ({ ...current, scenarios: [...current.scenarios, { ...current.scenarios[activeScenario] }] })); setActiveScenario(form.scenarios.length); } };
  const removeScenario = () => { if (form.scenarios.length > 1) { const next = form.scenarios.filter((_, index) => index !== activeScenario); setForm((current) => ({ ...current, scenarios: next })); setActiveScenario(Math.max(0, activeScenario - 1)); } };
  const save = async () => {
    setError("");
    if (!form.evidence.trim()) { setError("Evidence is required for every composition source."); return; }
    if (form.scenarios.some((scenario) => Math.abs(total(scenario) - 100) > 0.001)) { setError("Each scenario must contain FEM, NFM, PCB, PLA, and OTHER totaling exactly 100%."); return; }
    setSaving(true);
    try { const updated = await batchRepository.attachComposition(batch.id, batch.version, toComposition(form)); onSaved(updated); setEditing(false); } catch (issue) { setError(issue instanceof RepositoryError && issue.status === 409 ? "This batch changed elsewhere. Reload the record before saving again." : issue instanceof Error ? issue.message : "Composition could not be saved."); } finally { setSaving(false); }
  };

  if (!editing && batch.composition) return <section className="border-[1.5px] border-ink bg-ivory"><PanelHeader composition={batch.composition} onEdit={() => setEditing(true)} editable={!isModelDerived} /><div className="p-5"><CompositionReadout composition={batch.composition} activeScenario={activeScenario} onScenario={setActiveScenario} />{isModelDerived && <div className="mt-6 border-l-4 border-clay bg-[#F4D8CE] px-4 py-3 text-sm leading-6">This is model-derived Track A composition evidence. The API requires its stored scenarios to remain exact. To add a human estimate, create a new operator-entered composition rather than modifying this model record.</div>}</div></section>;

  const active = form.scenarios[activeScenario];
  const activeTotal = total(active);
  return <section className="border-[1.5px] border-ink bg-ivory"><div className="flex flex-col justify-between gap-4 border-b border-ink/20 p-5 sm:flex-row sm:items-start"><div><p className="font-mono text-[0.61rem] uppercase tracking-[0.1em] text-moss">Composition analysis</p><h2 className="mt-1 font-display text-2xl font-semibold tracking-[-0.055em]">Keep uncertainty in the record.</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-ink/65">A scenario is one aligned possible material mix. The optimizer compares the same scenario index across every selected batch.</p></div>{batchRepository.isMockData && <StatusStamp tone="review">Demo data mode</StatusStamp>}</div><div className="p-5"><div className="grid gap-5 sm:grid-cols-2"><label className="block text-sm font-medium">Composition source<select value={form.source} onChange={(event) => setForm((current) => ({ ...current, source: event.target.value as EditorState["source"] }))} className="mt-2 w-full border-[1.5px] border-ink bg-paper px-3 py-3 outline-none focus:border-clay"><option value="measured">Measured</option><option value="category_prior">Category prior</option><option value="operator_entered">Operator entered</option></select><span className="mt-2 block text-xs leading-5 text-ink/60">{sourceDescription(form.source)}</span></label><label className="block text-sm font-medium">Evidence / provenance<textarea value={form.evidence} onChange={(event) => setForm((current) => ({ ...current, evidence: event.target.value }))} rows={4} placeholder="State the measurement, data source, or observation supporting these scenarios." className="mt-2 w-full resize-none border-[1.5px] border-ink bg-paper px-3 py-3 text-sm outline-none placeholder:text-ink/35 focus:border-clay" /></label></div><ScenarioTabs count={form.scenarios.length} active={activeScenario} onSelect={setActiveScenario} onAdd={addScenario} onRemove={removeScenario} /><div className="mt-5 flex h-8 overflow-hidden border-[1.5px] border-ink bg-paper" aria-label={`Scenario ${activeScenario + 1} composition`}>{MATERIALS.map((material, index) => <div key={material} className={`${barColours[index]} min-w-[1px] border-r border-ink/50 last:border-r-0`} style={{ width: `${Math.max(0, Number(active[material]) || 0)}%` }} title={`${material}: ${active[material] || 0}%`} />)}</div><div className="mt-2 flex justify-between font-mono text-[0.61rem] uppercase tracking-[0.08em] text-ink/55"><span>Scenario {activeScenario + 1} of {form.scenarios.length}</span><span className={Math.abs(activeTotal - 100) < 0.001 ? "text-moss" : "text-clay"}>{activeTotal.toFixed(2)}% / 100%</span></div><div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-5">{MATERIALS.map((material) => <label key={material} className="border-[1.5px] border-ink bg-paper p-3"><span className="font-mono text-[0.63rem] tracking-[0.08em]">{material}</span><div className="mt-2 flex items-center border-b border-ink/35"><input inputMode="decimal" value={active[material]} onChange={(event) => updateFraction(material, event.target.value)} aria-label={`${material} fraction in scenario ${activeScenario + 1}`} className="min-w-0 w-full bg-transparent py-1 font-display text-xl outline-none" /><span className="text-sm">%</span></div></label>)}</div>{error && <p role="alert" className="mt-4 border-l-4 border-clay bg-[#F4D8CE] px-3 py-2 text-sm text-ink">{error}</p>}<div className="mt-7 flex flex-col-reverse gap-3 border-t border-ink/20 pt-5 sm:flex-row sm:items-center sm:justify-between"><button onClick={() => { setEditing(false); setForm(toEditor(batch.composition)); setError(""); }} className="text-sm font-semibold underline decoration-clay decoration-2 underline-offset-4">Cancel</button><button onClick={() => void save()} disabled={saving || Math.abs(activeTotal - 100) > 0.001 || !form.evidence.trim()} className="border-[1.5px] border-ink bg-moss px-4 py-2.5 font-display text-sm font-semibold text-ivory shadow-sketch-sm disabled:cursor-not-allowed disabled:opacity-50">{saving ? "Saving composition…" : "Save all scenarios"}</button></div></div></section>;
}

function PanelHeader({ composition, onEdit, editable }: { composition: Composition; onEdit: () => void; editable: boolean }) { return <div className="flex flex-col justify-between gap-4 border-b border-ink/20 p-5 sm:flex-row sm:items-start"><div><p className="font-mono text-[0.61rem] uppercase tracking-[0.1em] text-moss">Composition analysis</p><h2 className="mt-1 font-display text-2xl font-semibold tracking-[-0.055em]">Source before certainty.</h2></div><div className="flex items-center gap-3"><StatusStamp tone="neutral">{sourceLabel[composition.source]}</StatusStamp>{editable && <button onClick={onEdit} className="border border-ink bg-paper px-3 py-2 text-sm font-semibold">Edit scenarios</button>}</div></div>; }

const barColours = ["bg-sage", "bg-straw", "bg-[#C99475]", "bg-[#D8D5CA]", "bg-[#B8B2A3]"];

function CompositionReadout({ composition, activeScenario, onScenario }: { composition: Composition; activeScenario: number; onScenario: (index: number) => void }) { const scenario = composition.scenarios[activeScenario] ?? composition.scenarios[0]; return <div><ScenarioTabs count={composition.scenarios.length} active={activeScenario} onSelect={onScenario} /><div className="mt-5 flex h-8 overflow-hidden border-[1.5px] border-ink">{MATERIALS.map((material, index) => <div key={material} className={`${barColours[index]} min-w-[1px] border-r border-ink/50 last:border-r-0`} style={{ width: `${scenario[material] * 100}%` }} />)}</div><div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-5">{MATERIALS.map((material) => <div key={material}><p className="font-mono text-[0.61rem] tracking-[0.08em]">{material}</p><p className="mt-1 font-display text-xl font-semibold">{(scenario[material] * 100).toFixed(1)}%</p></div>)}</div><p className="mt-5 border-t border-ink/20 pt-4 text-sm leading-6 text-ink/65"><strong className="text-ink">Evidence:</strong> {composition.evidence}</p></div>; }

function ScenarioTabs({ count, active, onSelect, onAdd, onRemove }: { count: number; active: number; onSelect: (index: number) => void; onAdd?: () => void; onRemove?: () => void }) { return <div className="mt-6 flex flex-wrap items-center gap-2"><span className="mr-1 font-mono text-[0.61rem] uppercase tracking-[0.09em] text-ink/55">Scenarios</span>{Array.from({ length: count }, (_, index) => <button key={index} onClick={() => onSelect(index)} className={`h-8 min-w-8 border px-2 font-mono text-xs ${active === index ? "border-ink bg-ink text-paper" : "border-ink/40 bg-paper text-ink/65"}`}>{String(index + 1).padStart(2, "0")}</button>)}{onAdd && <button onClick={onAdd} disabled={count >= 20} className="h-8 border border-ink bg-[#DEE6D9] px-2 font-mono text-xs disabled:opacity-40" aria-label="Add a scenario">+</button>}{onRemove && count > 1 && <button onClick={onRemove} className="ml-1 text-xs font-semibold underline decoration-clay decoration-2 underline-offset-4">Remove active</button>}</div>; }
