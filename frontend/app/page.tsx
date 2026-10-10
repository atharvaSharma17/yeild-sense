import { HeroSortingIllustration } from "@/components/illustration/hero-sorting";
import { HoldStationIllustration } from "@/components/illustration/hold-station";
import { RouteObjectsIllustration } from "@/components/illustration/route-objects";
import { SiteHeader } from "@/components/shell/site-header";
import { WorkspaceFrame } from "@/components/shell/workspace-frame";
import { ButtonLink } from "@/components/ui/button-link";
import { Eyebrow } from "@/components/ui/eyebrow";
import { StatusStamp } from "@/components/ui/status-stamp";

const principles = [
  ["01", "Safety before economics", "Unknown or suspected battery conditions stay visibly constrained until a human review is documented."],
  ["02", "Uncertainty stays visible", "Composition scenarios are treated as assumptions to examine—not false precision disguised as certainty."],
  ["03", "Every decision leaves a trail", "Plans preserve their route inputs, model evidence, assumptions, solver result, and limitations."],
];

export default function Home() {
  return (
    <main className="overflow-hidden">
      <SiteHeader />

      <section className="relative mx-auto grid max-w-[1440px] grid-cols-1 gap-10 px-5 pb-16 pt-6 sm:px-8 sm:pb-20 lg:grid-cols-[1.03fr_.97fr] lg:items-center lg:gap-12 lg:px-12 lg:pb-28 lg:pt-14">
        <div className="relative z-10 max-w-3xl">
          <Eyebrow>Recovery intelligence, carefully grounded</Eyebrow>
          <h1 className="mt-5 max-w-[760px] font-display text-[clamp(3.35rem,7.2vw,7.2rem)] font-semibold leading-[0.89] tracking-[-0.09em] text-ink">
            Make e-waste recovery decisions that hold up.
          </h1>
          <p className="mt-7 max-w-xl text-lg leading-8 text-ink/75 sm:text-xl">
            YieldSense helps operators weigh uncertainty, capacity, economics, and safety—then choose an eligible recovery path with an auditable record.
          </p>
          <div className="mt-9 flex flex-col gap-4 sm:flex-row sm:items-center">
            <ButtonLink href="/workspace/batches">Explore the decision workspace <span aria-hidden="true">→</span></ButtonLink>
            <ButtonLink href="#method" variant="quiet" className="self-start px-0 py-2">See how it works</ButtonLink>
          </div>
          <div className="mt-12 flex flex-wrap gap-x-7 gap-y-3 border-t-[1.5px] border-ink pt-5 font-mono text-[0.64rem] uppercase tracking-[0.12em] text-ink/65">
            <span>Measured mass</span><span>Human review</span><span>Scenario-aware planning</span>
          </div>
        </div>

        <div className="relative mx-auto w-full max-w-[640px] lg:max-w-none">
          <div className="paper-grain absolute inset-[10%_2%_6%_8%] -rotate-2 border-[1.5px] border-ink bg-sage" />
          <div className="relative border-[1.5px] border-ink bg-ivory p-4 shadow-sketch sm:p-6">
            <HeroSortingIllustration />
          </div>
          <div className="absolute -bottom-5 left-3 border-[1.5px] border-ink bg-straw px-3 py-2 font-mono text-[0.62rem] uppercase tracking-[0.1em] shadow-sketch-sm sm:left-7">
            Sort thoughtfully. Recover responsibly.
          </div>
        </div>
      </section>

      <section id="method" className="scroll-mt-8 border-y-[1.5px] border-ink bg-ink text-paper">
        <div className="mx-auto grid max-w-[1440px] grid-cols-1 lg:grid-cols-[.8fr_1.2fr]">
          <div className="border-b-[1.5px] border-paper/35 px-5 py-12 sm:px-8 lg:border-b-0 lg:border-r-[1.5px] lg:px-12 lg:py-16">
            <Eyebrow className="text-sage">The method</Eyebrow>
            <h2 className="mt-4 max-w-md font-display text-4xl font-semibold leading-[0.95] tracking-[-0.07em] sm:text-5xl">A decision is more than a destination.</h2>
            <p className="mt-6 max-w-sm leading-7 text-paper/70">The work is to make each route choice legible: what is known, what remains uncertain, and what requires a person to intervene.</p>
          </div>
          <div className="grid divide-y-[1.5px] divide-paper/35">
            {[
              ["01", "Register the material", "Record measured batch mass, category, supporting evidence, and its starting safety state."],
              ["02", "Make uncertainty explicit", "Attach attributed composition scenarios. Image triage supports review; it never substitutes for it."],
              ["03", "Choose, test, document", "Compare eligible recovery routes under capacity constraints, then preserve the full audit trail."],
            ].map(([number, title, copy]) => (
              <article key={number} className="grid gap-4 px-5 py-8 sm:grid-cols-[70px_1fr] sm:px-8 lg:px-12">
                <span className="font-mono text-xs tracking-[0.1em] text-sage">{number}</span>
                <div>
                  <h3 className="font-display text-2xl font-semibold tracking-[-0.055em]">{title}</h3>
                  <p className="mt-2 max-w-xl leading-7 text-paper/70">{copy}</p>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-[1440px] grid-cols-1 gap-12 px-5 py-20 sm:px-8 lg:grid-cols-[.85fr_1.15fr] lg:items-center lg:gap-20 lg:px-12 lg:py-28">
        <div className="relative mx-auto w-full max-w-[370px]">
          <div className="absolute inset-2 rotate-3 border-[1.5px] border-ink bg-straw" />
          <div className="relative border-[1.5px] border-ink bg-ivory p-7 shadow-sketch">
            <RouteObjectsIllustration />
          </div>
        </div>
        <div>
          <Eyebrow>Not a black box</Eyebrow>
          <h2 className="mt-4 max-w-2xl font-display text-4xl font-semibold leading-[0.96] tracking-[-0.07em] sm:text-5xl">A calm place to see the consequence of every constraint.</h2>
          <div className="mt-9 grid gap-5 sm:grid-cols-3">
            <div className="border-t-[1.5px] border-ink pt-4"><p className="font-display text-2xl font-semibold tracking-[-0.06em]">Mass</p><p className="mt-2 text-sm leading-6 text-ink/65">Weight stays anchored to its source.</p></div>
            <div className="border-t-[1.5px] border-ink pt-4"><p className="font-display text-2xl font-semibold tracking-[-0.06em]">Risk</p><p className="mt-2 text-sm leading-6 text-ink/65">Worst-case scenarios sit beside expected value.</p></div>
            <div className="border-t-[1.5px] border-ink pt-4"><p className="font-display text-2xl font-semibold tracking-[-0.06em]">Proof</p><p className="mt-2 text-sm leading-6 text-ink/65">Inputs and limitations remain part of the plan.</p></div>
          </div>
        </div>
      </section>

      <WorkspaceFrame />

      <section id="principles" className="scroll-mt-8 mx-auto grid max-w-[1440px] grid-cols-1 lg:grid-cols-[1.15fr_.85fr] lg:gap-12">
        <div className="px-5 py-20 sm:px-8 lg:px-12 lg:py-28">
          <Eyebrow>Operating principles</Eyebrow>
          <h2 className="mt-4 max-w-xl font-display text-4xl font-semibold leading-[0.96] tracking-[-0.07em] sm:text-5xl">Useful only when it remains honest.</h2>
          <div className="mt-11 divide-y-[1.5px] divide-ink">
            {principles.map(([number, title, copy]) => (
              <article key={number} className="grid gap-4 py-6 sm:grid-cols-[64px_1fr]">
                <p className="font-mono text-xs tracking-[0.1em] text-clay">{number}</p>
                <div><h3 className="font-display text-2xl font-semibold tracking-[-0.055em]">{title}</h3><p className="mt-2 max-w-xl leading-7 text-ink/70">{copy}</p></div>
              </article>
            ))}
          </div>
        </div>
        <aside className="relative bg-[#E2E7DC] px-5 py-12 sm:px-8 lg:px-12 lg:py-20">
          <div className="mx-auto max-w-sm">
            <HoldStationIllustration />
            <div className="mt-6 border-[1.5px] border-ink bg-ivory p-5 shadow-sketch-sm">
              <StatusStamp tone="review">Safety gate</StatusStamp>
              <p className="mt-4 font-display text-2xl font-semibold leading-tight tracking-[-0.055em]">A model can raise a flag. A person clears the path.</p>
              <p className="mt-3 text-sm leading-6 text-ink/70">Unknown and suspected battery conditions remain constrained to approved hold routes until human evidence is recorded.</p>
            </div>
          </div>
        </aside>
      </section>

      <footer className="border-t-[1.5px] border-ink bg-ink px-5 py-10 text-paper sm:px-8 lg:px-12">
        <div className="mx-auto flex max-w-[1440px] flex-col justify-between gap-8 sm:flex-row sm:items-end">
          <div><p className="font-display text-3xl font-semibold tracking-[-0.07em]">YieldSense</p><p className="mt-2 max-w-md text-sm leading-6 text-paper/65">Uncertainty-aware recovery route planning for electronic waste.</p></div>
          <p className="font-mono text-[0.61rem] uppercase tracking-[0.11em] text-paper/50">Decision support · not automated safety certification</p>
        </div>
      </footer>
    </main>
  );
}
