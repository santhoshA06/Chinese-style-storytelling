import type { Scene } from '../types'
export function StorySceneCard({ scene }: { scene: Scene }) {
  return (
    <div className="grid gap-8 rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-2xl shadow-black/30 backdrop-blur-xl lg:grid-cols-[0.95fr_1.05fr] xl:p-8">
      <div className="flex flex-col justify-between p-2">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.25em] text-amber-300">Scene {scene.scene_id}</p>
          <p className="mt-4 text-lg leading-8 text-stone-200">{scene.story_text}</p>
        </div>
        <details className="mt-6 rounded-2xl border border-white/10 bg-stone-900 p-4">
          <summary className="cursor-pointer text-sm font-semibold text-stone-200">Show image prompt</summary>
          <pre className="mt-3 whitespace-pre-wrap text-sm leading-6 text-stone-400">{scene.image_prompt}</pre>
        </details>
      </div>
      <div>
        {scene.image_url ? (
          <img src={`http://localhost:8000${scene.image_url}`} alt={`Scene ${scene.scene_id}`} className="aspect-[4/3] w-full rounded-3xl border border-white/10 object-cover shadow-lg shadow-black/30" />
        ) : (
          <div className="flex aspect-[4/3] items-center justify-center rounded-3xl border border-dashed border-white/15 bg-stone-900 p-8 text-stone-400">Image not available</div>
        )}
      </div>
    </div>
  )
}
