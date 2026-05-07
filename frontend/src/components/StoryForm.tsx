import { useState } from 'react'
import { ModelSelector } from './ModelSelector'

type Props = {
  onSubmit: (payload: {
    idea: string
    n_scenes: number
    num_inference_steps: number
    guidance_scale: number
    use_lora: boolean
  }) => void
  loading: boolean
}
export function StoryForm({ onSubmit, loading }: Props) {
  const promptPresets = [
    'A brave child discovers a moonlit village hidden inside bamboo mountains.',
    'Two Friends Cross a Wooden Bridge in Autumn.',
    'A young painter follows a glowing crane through misty rivers.',
  ]
  const [idea, setIdea] = useState('Two Friends Cross a Wooden Bridge in Autumn.')
  const [nScenes, setNScenes] = useState(3)
  const [steps, setSteps] = useState(30)
  const [guidance, setGuidance] = useState(6.5)
  const [useLora, setUseLora] = useState(true)

  return (
    <div className="w-full overflow-hidden rounded-[2rem] border border-white/10 bg-white/5 shadow-2xl shadow-black/40 backdrop-blur-xl">
      <div className="grid min-h-[560px] gap-0 xl:grid-cols-[420px_minmax(0,1fr)] 2xl:grid-cols-[500px_minmax(0,1fr)]">
        <div className="bg-gradient-to-br from-slate-950 via-stone-950 to-teal-950/80 p-8 text-white md:p-10 xl:p-12">
          <p className="text-xs font-semibold uppercase tracking-[0.35em] text-amber-300">Compose</p>
          <h2 className="mt-5 max-w-xl text-5xl font-bold leading-tight">Create a storybook.</h2>
          <p className="mt-6 max-w-xl text-base leading-7 text-stone-300">
            Turn a single idea into a sequence of illustrated scenes with a refined Chinese landscape style.
          </p>
          <div className="mt-8 rounded-3xl border border-white/10 bg-[linear-gradient(135deg,rgba(20,184,166,0.18),rgba(245,158,11,0.12))] p-5">
            <div className="h-44 rounded-2xl border border-white/10 bg-[radial-gradient(circle_at_30%_20%,rgba(251,191,36,0.45),transparent_18%),linear-gradient(160deg,#0f172a_0%,#134e4a_48%,#431407_100%)] shadow-inner" />
            <div className="mt-4 grid grid-cols-3 gap-3 text-center text-xs text-stone-300">
              <div className="rounded-xl bg-white/10 px-3 py-2">
                <span className="block text-lg font-bold text-white">{nScenes}</span>
                scenes
              </div>
              <div className="rounded-xl bg-white/10 px-3 py-2">
                <span className="block text-lg font-bold text-white">{steps}</span>
                steps
              </div>
              <div className="rounded-xl bg-white/10 px-3 py-2">
                <span className="block text-lg font-bold text-white">{guidance}</span>
                guidance
              </div>
            </div>
          </div>
          <div className="mt-10 grid gap-4 text-sm text-stone-200">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <span className="font-semibold text-white">1. Prompt</span>
              <p className="mt-1 text-stone-400">Describe the moment, mood, and characters.</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <span className="font-semibold text-white">2. Tune</span>
              <p className="mt-1 text-stone-400">Choose scene count and visual generation settings.</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <span className="font-semibold text-white">3. Generate</span>
              <p className="mt-1 text-stone-400">Receive story text, image prompts, and artwork.</p>
            </div>
          </div>
        </div>
        <div className="space-y-8 bg-stone-900/80 p-8 md:p-10 xl:p-12">
        <div>
          <label className="mb-3 block text-sm font-semibold text-stone-200">Story idea</label>
          <textarea
            value={idea}
            onChange={(e) => setIdea(e.target.value)}
            rows={6}
            className="w-full resize-none rounded-2xl border border-white/10 bg-stone-950 px-5 py-4 text-white shadow-inner outline-none transition placeholder:text-stone-500 focus:border-teal-400 focus:ring-4 focus:ring-teal-400/10"
          />
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          {promptPresets.map((preset) => (
            <button
              key={preset}
              type="button"
              onClick={() => setIdea(preset)}
              className="rounded-2xl border border-white/10 bg-white/[0.06] px-4 py-3 text-left text-sm leading-6 text-stone-300 transition hover:border-teal-300/50 hover:bg-teal-400/10 hover:text-white"
            >
              {preset}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-4">
          <div>
            <label className="mb-3 flex items-center justify-between text-sm font-semibold text-stone-200">
              <span>Scenes</span>
              <span className="text-teal-200">{nScenes}</span>
            </label>
            <input type="number" min={3} max={5} value={nScenes} onChange={(e) => setNScenes(Number(e.target.value))} className="h-14 w-full rounded-xl border border-white/10 bg-stone-950 px-4 text-white shadow-sm outline-none transition focus:border-teal-400 focus:ring-4 focus:ring-teal-400/10" />
          </div>
          <div>
            <label className="mb-3 flex items-center justify-between text-sm font-semibold text-stone-200">
              <span>Steps</span>
              <span className="text-teal-200">{steps}</span>
            </label>
            <input type="number" min={15} max={40} value={steps} onChange={(e) => setSteps(Number(e.target.value))} className="h-14 w-full rounded-xl border border-white/10 bg-stone-950 px-4 text-white shadow-sm outline-none transition focus:border-teal-400 focus:ring-4 focus:ring-teal-400/10" />
          </div>
          <div>
            <label className="mb-3 flex items-center justify-between text-sm font-semibold text-stone-200">
              <span>Guidance</span>
              <span className="text-teal-200">{guidance}</span>
            </label>
            <input type="number" step="0.5" min={5} max={12} value={guidance} onChange={(e) => setGuidance(Number(e.target.value))} className="h-14 w-full rounded-xl border border-white/10 bg-stone-950 px-4 text-white shadow-sm outline-none transition focus:border-teal-400 focus:ring-4 focus:ring-teal-400/10" />
          </div>
          <ModelSelector useLora={useLora} onChange={setUseLora} />
        </div>
        <div className="grid gap-5 rounded-3xl border border-white/10 bg-white/[0.04] p-5 xl:grid-cols-3">
          <label className="text-sm font-semibold text-stone-200">
            Scene range
            <input type="range" min={3} max={5} value={nScenes} onChange={(e) => setNScenes(Number(e.target.value))} className="mt-4 w-full accent-teal-400" />
          </label>
          <label className="text-sm font-semibold text-stone-200">
            Render steps
            <input type="range" min={15} max={40} value={steps} onChange={(e) => setSteps(Number(e.target.value))} className="mt-4 w-full accent-teal-400" />
          </label>
          <label className="text-sm font-semibold text-stone-200">
            Style strength
            <input type="range" min={5} max={12} step={0.5} value={guidance} onChange={(e) => setGuidance(Number(e.target.value))} className="mt-4 w-full accent-teal-400" />
          </label>
        </div>
        <button
          disabled={loading}
          onClick={() => onSubmit({ idea, n_scenes: nScenes, num_inference_steps: steps, guidance_scale: guidance, use_lora: useLora })}
          className="inline-flex w-full items-center justify-center rounded-2xl bg-teal-600 px-7 py-4 font-semibold text-white shadow-lg shadow-black/30 transition hover:bg-teal-500 disabled:cursor-not-allowed disabled:opacity-60 md:w-auto"
        >
          {loading ? 'Generating...' : 'Generate Storybook'}
        </button>
        </div>
      </div>
    </div>
  )
}
