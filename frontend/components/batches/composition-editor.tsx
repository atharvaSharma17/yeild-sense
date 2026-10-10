"use client";

import type { ChangeEvent } from "react";
import { MATERIALS, type Composition, type CompositionSource, type Fractions, type Material } from "@/types/batch";

export type CompositionFormState = { source: CompositionSource; evidence: string; fractions: Record<Material, string> };
export const blankFractions: Record<Material, string> = { FEM: "", NFM: "", PCB: "", PLA: "", OTHER: "" };

export function compositionToForm(composition?: Composition): CompositionFormState {
  const first = composition?.scenarios[0];
  return { source: composition?.source ?? "operator_entered", evidence: composition?.evidence ?? "", fractions: first ? Object.fromEntries(MATERIALS.map((material) => [material, String(Math.round(first[material] * 100))])) as Record<Material, string> : blankFractions };
}

export function formToComposition(form: CompositionFormState): Composition {
  const scenario = Object.fromEntries(MATERIALS.map((material) => [material, Number(form.fractions[material]) / 100])) as Fractions;
  return { source: form.source, evidence: form.evidence.trim(), scenarios: [scenario] };
}

export function compositionTotal(form: CompositionFormState) { return MATERIALS.reduce((total, material) => total + (Number(form.fractions[material]) || 0), 0); }

export function CompositionEditor({ form, onChange, errors = {} }: { form: CompositionFormState; onChange: (form: CompositionFormState) => void; errors?: Partial<Record<"evidence" | "fractions", string>> }) {
  const total = compositionTotal(form);
  const updateFraction = (event: ChangeEvent<HTMLInputElement>, material: Material) => onChange({ ...form, fractions: { ...form.fractions, [material]: event.target.value } });
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="block text-sm font-medium">Composition source<select value={form.source} onChange={(event) => onChange({ ...form, source: event.target.value as CompositionSource })} className="mt-2 w-full border-[1.5px] border-ink bg-ivory px-3 py-3 text-sm outline-none focus:border-clay"><option value="operator_entered">Operator entered</option><option value="category_prior">Category prior</option><option value="measured">Measured</option></select></label>
        <label className="block text-sm font-medium">Evidence note<textarea value={form.evidence} onChange={(event) => onChange({ ...form, evidence: event.target.value })} rows={3} placeholder="What supports this estimate?" className="mt-2 w-full resize-none border-[1.5px] border-ink bg-ivory px-3 py-3 text-sm outline-none placeholder:text-ink/35 focus:border-clay" /></label>
      </div>
      {errors.evidence && <p className="-mt-3 text-sm text-clay">{errors.evidence}</p>}
      <div>
        <div className="mb-3 flex items-end justify-between gap-4"><div><p className="text-sm font-medium">First scenario mix</p><p className="mt-1 text-xs text-ink/60">All five material classes are required by the API.</p></div><span className={`font-mono text-sm ${Math.abs(total - 100) < 0.001 ? "text-moss" : "text-clay"}`}>{total.toFixed(0)}%</span></div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">{MATERIALS.map((material) => <label key={material} className="border-[1.5px] border-ink bg-ivory p-3 text-xs font-medium"><span className="font-mono tracking-[0.08em]">{material}</span><div className="mt-2 flex items-center border-b border-ink/35"><input inputMode="decimal" value={form.fractions[material]} onChange={(event) => updateFraction(event, material)} aria-label={`${material} percentage`} className="min-w-0 w-full bg-transparent py-1 font-display text-xl outline-none" /><span>%</span></div></label>)}</div>
        {errors.fractions && <p className="mt-3 text-sm text-clay">{errors.fractions}</p>}
      </div>
    </div>
  );
}
