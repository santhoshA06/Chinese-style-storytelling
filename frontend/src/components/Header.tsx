export function Header() {
  return (
    <header className="border-b border-white/10 bg-stone-950/70 backdrop-blur-xl">
      <div className="flex w-full flex-col gap-3 px-8 py-6 md:flex-row md:items-end md:justify-between xl:px-14">
        <div>
        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-amber-300">Story Studio</p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-white">
          AI Driven Storytelling with Chinese Landscape Art
        </h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-stone-300">
          Professional storybook generation using a local LLM and a Chinese-painting LoRA image model.
        </p>
        </div>
      </div>
    </header>
  )
}
