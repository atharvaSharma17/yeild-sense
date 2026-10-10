import Link from "next/link";
import type { ComponentProps } from "react";

type ButtonLinkProps = ComponentProps<typeof Link> & {
  variant?: "primary" | "secondary" | "quiet";
};

export function ButtonLink({ variant = "primary", className = "", children, ...props }: ButtonLinkProps) {
  const styles = {
    primary: "bg-moss text-ivory border-moss shadow-sketch hover:-translate-y-0.5 hover:shadow-[4px_4px_0_#18211B]",
    secondary: "bg-ivory text-ink border-ink shadow-sketch hover:-translate-y-0.5 hover:shadow-[4px_4px_0_#18211B]",
    quiet: "border-transparent bg-transparent text-ink underline decoration-clay decoration-2 underline-offset-4 hover:text-moss",
  };

  return (
    <Link
      {...props}
      className={`inline-flex items-center justify-center gap-2 rounded-sm border-[1.5px] px-5 py-3 font-display text-sm font-semibold tracking-[-0.02em] transition duration-200 ${styles[variant]} ${className}`}
    >
      {children}
    </Link>
  );
}
