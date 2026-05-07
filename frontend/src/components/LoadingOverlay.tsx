export function LoadingOverlay({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-amber-300/20 bg-amber-400/10 p-4 text-sm font-medium text-amber-100 shadow-sm">
      <span className="h-3 w-3 animate-pulse rounded-full bg-amber-500" />
      <span>{text}</span>
    </div>
  )
}
