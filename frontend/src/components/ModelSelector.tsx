type Props = {
  useLora: boolean
  onChange: (value: boolean) => void
}
export function ModelSelector({ useLora, onChange }: Props) {
  return (
    <div>
      <label className="mb-3 block text-sm font-semibold text-stone-200">Image model</label>
      <select
        value={useLora ? 'lora' : 'baseline'}
        onChange={(e) => onChange(e.target.value === 'lora')}
        className="h-14 w-full rounded-xl border border-white/10 bg-stone-950 px-4 text-white shadow-sm outline-none transition focus:border-teal-400 focus:ring-4 focus:ring-teal-400/10"
      >
        <option value="lora">Chinese Painting LoRA</option>
        <option value="baseline">Baseline SD 1.5</option>
      </select>
    </div>
  )
}
