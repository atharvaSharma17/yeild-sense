import Link from "next/link";
import { ButtonLink } from "@/components/ui/button-link";
import { Wordmark } from "@/components/ui/wordmark";

const links = [
  { href: "#method", label: "Method" },
  { href: "#principles", label: "Principles" },
  { href: "#workspace", label: "Workspace" },
];

export function SiteHeader() {
  return (
    <header className="mx-auto flex w-full max-w-[1440px] items-center justify-between px-5 py-5 sm:px-8 lg:px-12">
      <Wordmark />
      <nav aria-label="Main navigation" className="hidden items-center gap-7 md:flex">
        {links.map((link) => (
          <Link key={link.href} href={link.href} className="text-sm font-medium text-ink/75 transition hover:text-moss">
            {link.label}
          </Link>
        ))}
      </nav>
      <ButtonLink href="/workspace/batches" className="px-3.5 py-2.5 text-xs sm:px-5 sm:text-sm">
        Explore the workspace <span aria-hidden="true">↗</span>
      </ButtonLink>
    </header>
  );
}
