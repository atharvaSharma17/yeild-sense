"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Wordmark } from "@/components/ui/wordmark";

const navigation = [
  { href: "/workspace", label: "Overview", number: "01" },
  { href: "/workspace/batches", label: "Batches", number: "02" },
  { href: "/workspace/routes", label: "Routes", number: "03" },
  { href: "/workspace/planning", label: "Planning", number: "04" },
  { href: "/workspace/reports", label: "Reports", number: "05" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="min-h-screen bg-paper">
      <header className="sticky top-0 z-30 border-b-[1.5px] border-ink bg-paper/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1520px] items-center justify-between px-5 py-4 sm:px-8 lg:px-10">
          <Wordmark />
          <div className="flex items-center gap-3">
            <span className="hidden font-mono text-[0.61rem] uppercase tracking-[0.12em] text-ink/50 sm:block">Operations workspace</span>
            <Link href="/workspace/batches/new" className="border-[1.5px] border-ink bg-moss px-3 py-2 font-display text-xs font-semibold tracking-[-0.02em] text-ivory shadow-sketch-sm transition hover:-translate-y-0.5">New batch <span aria-hidden="true">+</span></Link>
          </div>
        </div>
      </header>
      <div className="mx-auto grid max-w-[1520px] lg:grid-cols-[238px_1fr]">
        <aside className="hidden min-h-[calc(100vh-65px)] border-r-[1.5px] border-ink bg-[#EEEADD] p-5 lg:block">
          <p className="mb-6 font-mono text-[0.61rem] uppercase tracking-[0.14em] text-ink/55">Decision desk</p>
          <nav aria-label="Workspace navigation" className="space-y-1">
            {navigation.map((item) => {
          const active = item.href !== "#" && (pathname === item.href || (item.href !== "/workspace" && pathname.startsWith(item.href)));
              return (
                <Link key={item.label} href={item.href} className={`flex items-center gap-2 rounded-sm px-3 py-2.5 text-sm transition ${active ? "border border-ink bg-ivory font-semibold shadow-[2px_2px_0_#18211B]" : "text-ink/65 hover:bg-ivory"}`}>
                  <span className="font-mono text-[0.62rem]">{item.number}</span>{item.label}
                </Link>
              );
            })}
          </nav>
          <div className="mt-10 border-t border-ink/25 pt-5 text-xs leading-5 text-ink/60">This workspace supports decisions; it does not certify safety or authorization.</div>
        </aside>
        <main className="min-w-0">{children}</main>
      </div>
      <nav aria-label="Mobile workspace navigation" className="fixed bottom-0 z-30 flex w-full border-t-[1.5px] border-ink bg-ivory lg:hidden">
        {navigation.map((item) => <Link key={item.label} href={item.href} className={`flex-1 px-1 py-3 text-center font-mono text-[0.52rem] uppercase tracking-[0.04em] ${pathname === item.href || (item.href !== "/workspace" && pathname.startsWith(item.href)) ? "bg-sage text-ink" : "text-ink/55"}`}>{item.label}</Link>)}
      </nav>
    </div>
  );
}
