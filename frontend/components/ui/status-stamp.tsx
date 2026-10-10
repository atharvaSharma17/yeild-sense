const tones = {
  safe: "border-moss bg-[#DEE6D9] text-moss",
  review: "border-[#8D6A18] bg-[#F3E7B9] text-[#6E520D]",
  risk: "border-clay bg-[#F4D8CE] text-[#7E402A]",
  neutral: "border-ink bg-transparent text-ink",
};

export function StatusStamp({ children, tone = "neutral" }: { children: React.ReactNode; tone?: keyof typeof tones }) {
  return <span className={`inline-flex rounded-sm border px-2 py-1 font-mono text-[0.61rem] font-medium uppercase tracking-[0.08em] ${tones[tone]}`}>{children}</span>;
}
