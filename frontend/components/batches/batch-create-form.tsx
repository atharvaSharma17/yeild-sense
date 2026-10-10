"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { CompositionEditor, blankFractions, compositionTotal, formToComposition, type CompositionFormState } from "@/components/batches/composition-editor";
import { batchRepository } from "@/lib/data/batch-repository";
import type { BatchDraft } from "@/types/batch";

type FieldErrors = Partial<Record<"category" | "mass" | "evidence" | "fractions", string>>;

const initialComposition: CompositionFormState = { source: "operator_entered", evidence: "", fractions: blankFractions };

export function BatchCreateForm() {
  const router = useRouter();
  const [category, setCategory] = useState("");
  const [mass, setMass] = useState("");
  const [massSource, setMassSource] = useState<"scale" | "illustrative">("scale");
  const [includeComposition, setIncludeComposition] = useState(true);
  const [composition, setComposition] = useState<CompositionFormState>(initialComposition);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [formError, setFormError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const nextErrors: FieldErrors = {};
    const numericMass = Number(mass);
    if (!category.trim()) nextErrors.category = "Choose or enter the collection category.";
    if (!Number.isFinite(numericMass) || numericMass <= 0) nextErrors.mass = "Enter a measured mass greater than zero.";
    if (includeComposition) {
      if (!composition.evidence.trim()) nextErrors.evidence = "Record the evidence supporting this composition.";
      if (Math.abs(compositionTotal(composition) - 100) > 0.001) nextErrors.fractions = "The five material fractions must total exactly 100%.";
    }
    setErrors(nextErrors); setFormError("");
    if (Object.keys(nextErrors).length > 0) return;
    const draft: BatchDraft = { category: category.trim().toLowerCase().replaceAll(" ", "_"), mass_kg: numericMass, mass_source: massSource, ...(includeComposition ? { composition: formToComposition(composition) } : {}) };
    setSubmitting(true);
    try { const batch = await batchRepository.create(draft); router.push(`/workspace/batches/${batch.id}`); } catch (issue) { setFormError(issue instanceof Error ? issue.message : "The batch could not be created."); setSubmitting(false); }
  };

  return (
    <form onSubmit={submit} noValidate className="pb-28">
      <div className="grid gap-10 xl:grid-cols-[minmax(0,1fr)_330px] xl:gap-14">
        <div className="space-y-10">
          <section className="border-t-[1.5px] border-ink pt-5"><div className="grid gap-7 sm:grid-cols-[150px_1fr]"><div><p className="font-mono text-xs tracking-[0.1em] text-moss">01 / Intake</p><p className="mt-2 text-sm leading-6 text-ink/60">Start with the physical batch and its weight source.</p></div><div className="grid gap-5"><label className="block text-sm font-medium">Collection category<input value={category} onChange={(event) => setCategory(event.target.value)} placeholder="e.g. small_devices" className="mt-2 w-full border-[1.5px] border-ink bg-ivory px-3 py-3 text-base outline-none placeholder:text-ink/35 focus:border-clay" aria-invalid={Boolean(errors.category)} /></label>{errors.category && <p className="-mt-3 text-sm text-clay">{errors.category}</p>}<div className="grid gap-5 sm:grid-cols-2"><label className="block text-sm font-medium">Batch mass<input value={mass} onChange={(event) => setMass(event.target.value)} inputMode="decimal" placeholder="0.0" className="mt-2 w-full border-[1.5px] border-ink bg-ivory px-3 py-3 text-base outline-none placeholder:text-ink/35 focus:border-clay" aria-invalid={Boolean(errors.mass)} /><span className="mt-1 block font-mono text-[0.62rem] uppercase tracking-[0.08em] text-ink/50">Kilograms (kg)</span></label><label className="block text-sm font-medium">Mass source<select value={massSource} onChange={(event) => setMassSource(event.target.value as "scale" | "illustrative")} className="mt-2 w-full border-[1.5px] border-ink bg-ivory px-3 py-3 text-base outline-none focus:border-clay"><option value="scale">Scale measurement</option><option value="illustrative">Illustrative assumption</option></select><span className="mt-1 block text-xs text-ink/50">Use illustrative only for non-operational examples.</span></label></div>{errors.mass && <p className="-mt-3 text-sm text-clay">{errors.mass}</p>}</div></div></section>

          <section className="border-t-[1.5px] border-ink pt-5"><div className="grid gap-7 sm:grid-cols-[150px_1fr]"><div><p className="font-mono text-xs tracking-[0.1em] text-moss">02 / Composition</p><p className="mt-2 text-sm leading-6 text-ink/60">Optional at intake, required later for optimization.</p></div><div><label className="flex cursor-pointer items-start gap-3 border-[1.5px] border-ink bg-[#DEE6D9] p-4"><input checked={includeComposition} onChange={(event) => setIncludeComposition(event.target.checked)} type="checkbox" className="mt-1 h-4 w-4 accent-moss" /><span><span className="block font-display text-lg font-semibold tracking-[-0.04em]">Attach a composition estimate now</span><span className="mt-1 block text-sm leading-6 text-ink/65">All scenario values are attributed evidence, not a claim of physical certainty.</span></span></label>{includeComposition && <div className="mt-6"><CompositionEditor form={composition} onChange={setComposition} errors={{ evidence: errors.evidence, fractions: errors.fractions }} /></div>}</div></div></section>
        </div>
        <aside className="h-fit border-[1.5px] border-ink bg-[#EEEADD] p-5 shadow-sketch-sm xl:sticky xl:top-24"><p className="font-mono text-[0.61rem] uppercase tracking-[0.11em] text-ink/55">A clear starting point</p><h2 className="mt-3 font-display text-2xl font-semibold leading-tight tracking-[-0.06em]">Every new batch begins pending review.</h2><p className="mt-3 text-sm leading-6 text-ink/70">The API deliberately resets battery status to unknown and review status to pending. Image evidence can support triage, but never clears a batch.</p><div className="mt-5 border-t border-ink/25 pt-4 font-mono text-[0.6rem] uppercase leading-5 tracking-[0.08em] text-ink/55">The next step is image evidence and documented human review.</div></aside>
      </div>
      {formError && <div role="alert" className="mt-8 border-[1.5px] border-clay bg-[#F4D8CE] p-4 text-sm">{formError}</div>}
      <div className="mt-10 flex flex-col-reverse items-start gap-4 border-t-[1.5px] border-ink pt-6 sm:flex-row sm:items-center sm:justify-between"><p className="max-w-md text-sm leading-6 text-ink/60">Creating the record does not approve its safety status or create an optimization plan.</p><button type="submit" disabled={submitting} className="border-[1.5px] border-ink bg-moss px-5 py-3 font-display text-sm font-semibold text-ivory shadow-sketch transition hover:-translate-y-0.5 disabled:cursor-wait disabled:opacity-60">{submitting ? "Creating batch…" : "Create batch record"} <span aria-hidden="true">→</span></button></div>
    </form>
  );
}
