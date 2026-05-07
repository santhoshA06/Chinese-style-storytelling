import { useState } from 'react'
import { generateStorybook } from '../api/client'
import { StoryForm } from '../components/StoryForm'
import { StorySceneCard } from '../components/StorySceneCard'
import { UploadQuestionCard } from '../components/UploadQuestionCard'
import { LoadingOverlay } from '../components/LoadingOverlay'
import type { StorybookResponse } from '../types'

export function HomePage() {
  const [storybook, setStorybook] = useState<StorybookResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (payload: {
    idea: string
    n_scenes: number
    num_inference_steps: number
    guidance_scale: number
    use_lora: boolean
  }) => {
    try {
      setLoading(true)
      setError('')
      const data = await generateStorybook(payload)
      setStorybook(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to generate storybook')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="w-full space-y-10 px-8 py-10 xl:px-14">
      <StoryForm onSubmit={handleSubmit} loading={loading} />
      {loading && <LoadingOverlay text="Generating story and images. This can take a while on a laptop GPU." />}
      {error && <div className="rounded-2xl border border-red-400/30 bg-red-950/80 p-4 text-sm font-medium text-red-100 shadow-sm">{error}</div>}
      {storybook && (
        <section className="space-y-6">
          <div className="rounded-[2rem] border border-white/10 bg-stone-950 p-8 shadow-2xl shadow-black/30">
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-amber-300">Generated Storybook</p>
            <h2 className="mt-2 text-3xl font-bold text-white">{storybook.title}</h2>
            <p className="mt-3 max-w-5xl leading-7 text-stone-300">{storybook.summary}</p>
          </div>
          {storybook.scenes.map((scene) => (
            <StorySceneCard key={scene.scene_id} scene={scene} />
          ))}
        </section>
      )}
      <UploadQuestionCard />
    </div>
  )
}
