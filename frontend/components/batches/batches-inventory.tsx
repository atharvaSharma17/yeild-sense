"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { batchRepository } from "@/lib/data/batch-repository";
import { batchShortId, formatDate, formatMass, safetyLabel } from "@/lib/format";
import type { Batch } from "@/types/batch";
import { CompositionBadge, SafetyBadge } from "@/components/batches/batch-badges";
import { Eyebrow } from "@/components/ui/eyebrow";

type SafetyFilter = "all" | "review" | "cleared" | "hold";

export function BatchesInventory() {
  const [batches, setBatches] = useState<Batch[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<SafetyFilter>("all");

  const load = async () => {
    setLoading(true); setError("");
    try { setBatches((await batchRepository.list()).items); } catch (issue) { setError(issue instanceof Error ? issue.message : "Unable to load batch inventory."); } finally { setLoading(false); }
  };
  useEffect(() => { void load(); }, []);

  const filtered = useMemo(() => batches.filter((batch) => {
    const textMatch = `${batch.id} ${batch.category}`.toLowerCase().includes(query.toLowerCase());
    const safetyMatch = filter === "all" || (filter === "review" && batch.review_status === "pending" && batch.battery_status === "unknown") || (filter === "cleared" && batch.review_status === "approved" && batch.battery_status === "cleared") || (filter === "hold" && ["suspected", "present"].includes(batch.battery_status));
    return textMatch && safetyMatch;
  }), [batches, filter, query]);

  const totalMass = batches.reduce((sum, item) => sum + item.mass_kg, 0);
  const reviewCount = batches.filter((item) => item.review_status === "pending").length;

  return (
    <section className="px-5 py-10 pb-28 sm:px-8 lg:px-12 lg:py-14">
      <div className="flex flex-col justify-between gap-6 border-b-[1.5px] border-ink pb-8 sm:flex-row sm:items-end">
        <div><Eyebrow>Batch inventory</Eyebrow><h1 className="mt-3 font-display text-5xl font-semibold leading-[0.92] tracking-[-0.08em] sm:text-6xl">What is waiting<br />to be understood.</h1></div>
        <Link href="/workspace/batches/new" className="inline-flex self-start border-[1.5px] border-ink bg-moss px-5 py-3 font-display text-sm font-semibold text-ivory shadow-sketch transition hover:-translate-y-0.5 sm:self-auto">Register a batch <span className="ml-2">+</span></Link>
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        <Metric label="Recorded mass" value={`${formatMass(totalMass)} kg`} detail="Across this visible inventory" />
        <Metric label="Needs review" value={String(reviewCount).padStart(2, "0")} detail="Human evidence has not been recorded" accent="straw" />
        <Metric label="Data mode" value={batchRepository.isMockData ? "Mock" : "Live"} detail={batchRepository.isMockData ? "Local preview data" : "Connected through secure proxy"} accent="sage" />
      </div>

      <div className="mt-9 flex flex-col gap-3 border-y border-ink/25 py-4 md:flex-row md:items-center md:justify-between">
        <label className="flex max-w-md flex-1 items-center gap-2 border-b-[1.5px] border-ink bg-transparent px-1 py-2 text-sm"><span aria-hidden="true">⌕</span><span className="sr-only">Search batches</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search ID or category" className="w-full bg-transparent outline-none placeholder:text-ink/40" /></label>
        <div className="flex gap-2 overflow-x-auto pb-1 md:pb-0">{(["all", "review", "cleared", "hold"] as SafetyFilter[]).map((item) => <button key={item} onClick={() => setFilter(item)} className={`whitespace-nowrap border px-3 py-2 font-mono text-[0.61rem] uppercase tracking-[0.08em] ${filter === item ? "border-ink bg-ink text-paper" : "border-ink/40 bg-ivory text-ink/65 hover:border-ink"}`}>{item === "all" ? "All records" : item === "review" ? "Review needed" : item === "cleared" ? "Cleared" : "Hold only"}</button>)}</div>
      </div>

      {loading && <InventoryLoading />}
      {error && <div role="alert" className="mt-8 border-[1.5px] border-clay bg-[#F4D8CE] p-5"><p className="font-display text-xl font-semibold">The inventory did not load.</p><p className="mt-1 text-sm text-ink/70">{error}</p><button onClick={() => void load()} className="mt-4 border border-ink bg-ivory px-3 py-2 text-sm font-semibold">Try again</button></div>}
      {!loading && !error && filtered.length === 0 && <div className="mt-8 border-[1.5px] border-ink bg-ivory p-8"><p className="font-display text-2xl font-semibold tracking-[-0.05em]">No batches match this view.</p><button onClick={() => { setFilter("all"); setQuery(""); }} className="mt-3 font-semibold underline decoration-clay decoration-2 underline-offset-4">Clear filters</button></div>}
      {!loading && !error && filtered.length > 0 && <div className="mt-7 overflow-hidden border-[1.5px] border-ink bg-ivory">
        <div className="hidden grid-cols-[.8fr_1.45fr_.7fr_.85fr_.85fr_42px] gap-4 border-b border-ink/25 px-5 py-3 font-mono text-[0.6rem] uppercase tracking-[0.1em] text-ink/55 md:grid"><span>Batch</span><span>Category / received</span><span>Mass</span><span>Composition</span><span>Safety</span><span /></div>
        {filtered.map((batch) => <Link key={batch.id} href={`/workspace/batches/${batch.id}`} className="group block border-b border-ink/15 last:border-b-0 hover:bg-[#F0EEE5]"><div className="grid gap-3 px-5 py-5 md:grid-cols-[.8fr_1.45fr_.7fr_.85fr_.85fr_42px] md:items-center md:gap-4"><div><span className="font-mono text-xs tracking-[0.05em]">{batchShortId(batch.id)}</span><p className="mt-1 text-xs text-ink/50 md:hidden">{formatDate(batch.created_at)}</p></div><div><p className="font-medium capitalize">{batch.category.replaceAll("_", " ")}</p><p className="mt-1 hidden text-xs text-ink/55 md:block">Received {formatDate(batch.created_at)}</p></div><p className="font-display text-xl font-semibold tracking-[-0.045em]">{formatMass(batch.mass_kg)} <span className="text-sm">kg</span></p><div><CompositionBadge batch={batch} /></div><div><SafetyBadge batch={batch} /><p className="mt-2 text-xs text-ink/50 md:hidden">{safetyLabel(batch)}</p></div><span className="hidden text-right text-lg transition-transform group-hover:translate-x-1 md:block" aria-hidden="true">→</span></div></Link>)}
      </div>}
    </section>
  );
}

function Metric({ label, value, detail, accent = "ivory" }: { label: string; value: string; detail: string; accent?: "ivory" | "straw" | "sage" }) {
  const background = { ivory: "bg-ivory", straw: "bg-[#F1E7BE]", sage: "bg-[#DEE6D9]" }[accent];
  return <div className={`border-[1.5px] border-ink p-4 ${background}`}><p className="font-mono text-[0.6rem] uppercase tracking-[0.11em] text-ink/55">{label}</p><p className="mt-2 font-display text-3xl font-semibold tracking-[-0.07em]">{value}</p><p className="mt-2 text-xs text-ink/60">{detail}</p></div>;
}

function InventoryLoading() { return <div className="mt-7 grid gap-px border-[1.5px] border-ink bg-ink">{Array.from({ length: 4 }, (_, index) => <div key={index} className="h-20 animate-pulse bg-ivory" />)}</div>; }
