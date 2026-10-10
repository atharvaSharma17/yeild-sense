import Link from "next/link";

export function Wordmark() {
  return (
    <Link href="/" className="group inline-flex items-center gap-2.5 rounded-sm" aria-label="YieldSense home">
      <span className="grid h-8 w-8 place-items-center border-[1.5px] border-ink bg-sage shadow-sketch-sm transition-transform group-hover:-translate-y-0.5">
        <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8">
          <path d="M4.5 7.2 12 3l7.5 4.2v9.6L12 21l-7.5-4.2V7.2Z" />
          <path d="M12 3v8.7m7.5-4.5L12 11.7 4.5 7.2" />
          <path d="m9.2 15.5 1.9 1.8 4-4" />
        </svg>
      </span>
      <span className="font-display text-[1.08rem] font-semibold tracking-[-0.06em]">YieldSense</span>
    </Link>
  );
}
