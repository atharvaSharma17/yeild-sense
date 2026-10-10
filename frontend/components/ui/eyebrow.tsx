export function Eyebrow({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <p className={`font-mono text-[0.67rem] font-medium uppercase tracking-[0.16em] text-moss ${className}`}>
      {children}
    </p>
  );
}
