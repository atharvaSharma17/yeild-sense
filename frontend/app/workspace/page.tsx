import Link from "next/link";
import { Eyebrow } from "@/components/ui/eyebrow";

export default function WorkspaceHome() {
  return (
    <section className="px-5 py-12 pb-28 sm:px-8 lg:px-12 lg:py-16">
      <Eyebrow>YieldSense / decision workspace</Eyebrow>
      <h1 className="mt-4 max-w-2xl font-display text-5xl font-semibold leading-[0.92] tracking-[-0.08em] sm:text-6xl">Start with what has actually arrived.</h1>
      <p className="mt-6 max-w-xl text-lg leading-8 text-ink/70">Move from documented intake and human review to constrained planning, simulation, and immutable decision records—without treating estimates as approvals.</p>
      <div className="mt-9 flex flex-wrap gap-3"><Link href="/workspace/batches" className="inline-flex border-[1.5px] border-ink bg-moss px-5 py-3 font-display text-sm font-semibold text-ivory shadow-sketch transition hover:-translate-y-0.5">Open batch inventory <span className="ml-2">→</span></Link><Link href="/workspace/planning" className="inline-flex border-[1.5px] border-ink bg-ivory px-5 py-3 font-display text-sm font-semibold shadow-sketch-sm">Plan a route</Link><Link href="/workspace/reports" className="inline-flex border border-ink px-5 py-3 text-sm font-semibold">Browse reports</Link></div>
    </section>
  );
}
