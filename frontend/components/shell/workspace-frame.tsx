import { StatusStamp } from "@/components/ui/status-stamp";

export function WorkspaceFrame() {
  return (
    <section id="workspace" className="scroll-mt-8 border-y-[1.5px] border-ink bg-ivory">
      <div className="mx-auto max-w-[1440px] px-5 py-5 sm:px-8 lg:px-12">
        <div className="overflow-hidden border-[1.5px] border-ink bg-paper shadow-sketch">
          <div className="flex items-center justify-between border-b-[1.5px] border-ink px-4 py-3 sm:px-5">
            <div className="flex items-center gap-2 font-mono text-[0.62rem] uppercase tracking-[0.12em] text-ink/70">
              <span className="h-2.5 w-2.5 rounded-full border border-ink bg-sage" /> YieldSense workspace
            </div>
            <span className="font-mono text-[0.62rem] uppercase tracking-[0.12em] text-ink/60">Illustrative preview</span>
          </div>
          <div className="grid min-h-[390px] grid-cols-1 lg:grid-cols-[210px_1fr]">
            <aside className="hidden border-r-[1.5px] border-ink bg-[#EEEADD] p-4 lg:block">
              <p className="mb-6 font-mono text-[0.6rem] uppercase tracking-[0.14em] text-ink/55">Decision desk</p>
              {["Overview", "Batches", "Routes", "Planning", "Reports"].map((item, index) => (
                <div key={item} className={`mb-1 flex items-center gap-2 rounded-sm px-3 py-2.5 text-sm ${index === 0 ? "border border-ink bg-ivory font-semibold" : "text-ink/55"}`}>
                  <span className="font-mono text-[0.62rem]">0{index + 1}</span>{item}
                </div>
              ))}
            </aside>
            <div className="p-5 sm:p-8">
              <div className="flex flex-col justify-between gap-5 border-b border-ink/20 pb-6 sm:flex-row sm:items-start">
                <div>
                  <p className="font-mono text-[0.63rem] uppercase tracking-[0.15em] text-moss">Illustrative workspace</p>
                  <h2 className="mt-2 font-display text-3xl font-semibold tracking-[-0.065em]">A clearer path for every batch.</h2>
                </div>
                <StatusStamp tone="review">Example review queue</StatusStamp>
              </div>
              <div className="mt-7 grid gap-4 md:grid-cols-3">
                <div className="border-[1.5px] border-ink bg-ivory p-4">
                  <p className="font-mono text-[0.62rem] uppercase tracking-[0.1em] text-ink/55">Incoming mass</p>
                  <p className="mt-3 font-display text-4xl font-semibold tracking-[-0.07em]">2,460 <span className="text-lg">kg</span></p>
                  <p className="mt-3 text-xs text-ink/60">Example operational snapshot</p>
                </div>
                <div className="border-[1.5px] border-ink bg-[#DEE6D9] p-4">
                  <p className="font-mono text-[0.62rem] uppercase tracking-[0.1em] text-ink/55">Ready to plan</p>
                  <p className="mt-3 font-display text-4xl font-semibold tracking-[-0.07em]">08 <span className="text-lg">batches</span></p>
                  <p className="mt-3 text-xs text-ink/60">Example planning eligibility</p>
                </div>
                <div className="border-[1.5px] border-ink bg-[#F4D8CE] p-4">
                  <p className="font-mono text-[0.62rem] uppercase tracking-[0.1em] text-ink/55">Safety hold</p>
                  <p className="mt-3 font-display text-4xl font-semibold tracking-[-0.07em]">03 <span className="text-lg">batches</span></p>
                  <p className="mt-3 text-xs text-ink/60">Example safety restriction</p>
                </div>
              </div>
              <div className="mt-7 border-[1.5px] border-ink bg-ivory">
                <div className="grid grid-cols-[1.15fr_.7fr_.7fr] gap-3 border-b border-ink/20 px-4 py-3 font-mono text-[0.61rem] uppercase tracking-[0.1em] text-ink/55">
                  <span>Decision queue</span><span>Composition</span><span>Safety</span>
                </div>
                <div className="grid grid-cols-[1.15fr_.7fr_.7fr] gap-3 px-4 py-4 text-sm">
                  <span className="font-medium">Small devices / 480 kg</span><span>Category prior</span><span><StatusStamp tone="review">Review</StatusStamp></span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
