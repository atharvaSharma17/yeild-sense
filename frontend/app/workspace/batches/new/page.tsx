import Link from "next/link";
import { BatchCreateForm } from "@/components/batches/batch-create-form";
import { Eyebrow } from "@/components/ui/eyebrow";

export default function NewBatchPage() {
  return <section className="px-5 py-10 sm:px-8 lg:px-12 lg:py-14"><Link href="/workspace/batches" className="font-mono text-[0.63rem] uppercase tracking-[0.1em] text-ink/60 underline decoration-clay decoration-2 underline-offset-4">← Back to inventory</Link><div className="mt-9 border-b-[1.5px] border-ink pb-8"><Eyebrow>New batch</Eyebrow><h1 className="mt-3 font-display text-5xl font-semibold leading-[0.92] tracking-[-0.08em] sm:text-6xl">Record what came<br />through the door.</h1><p className="mt-5 max-w-2xl text-lg leading-8 text-ink/70">Start with mass and category. Attach the best available composition evidence, then continue to the record for images and review.</p></div><div className="mt-10"><BatchCreateForm /></div></section>;
}
