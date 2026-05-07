import { useState } from 'react'
import { askImageQuestion } from '../api/client'
export function UploadQuestionCard() {
  const [file, setFile] = useState<File | null>(null)
  const [question, setQuestion] = useState('What is shown in this image?')
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const onAsk = async () => {
    if (!file) return
    try {
      setLoading(true)
      setError('')
      const result = await askImageQuestion(file, question)
      setAnswer(result.answer)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }
  return (
    <div className="rounded-[2rem] border border-white/10 bg-white/5 p-8 shadow-2xl shadow-black/30 backdrop-blur-xl xl:p-10">
      <div className="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-amber-300">Image Q&A</p>
          <h2 className="mt-2 text-2xl font-bold text-white">Upload Image and Ask a Question</h2>
        </div>
        <p className="max-w-xl text-sm leading-6 text-stone-300">
          Test image understanding against the artwork or your own references without leaving the studio.
        </p>
      </div>
      <div className="mt-7 grid gap-5 xl:grid-cols-[1fr_1.4fr_auto] xl:items-end">
        <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="block w-full rounded-2xl border border-dashed border-white/15 bg-black p-4 text-sm text-stone-300 file:mr-4 file:rounded-full file:border-0 file:bg-teal-600 file:px-4 file:py-2 file:font-semibold file:text-white" />
        <input value={question} onChange={(e) => setQuestion(e.target.value)} className="h-14 w-full rounded-xl border border-white/10 bg-stone-900 px-4 text-white shadow-sm outline-none transition focus:border-teal-400 focus:ring-4 focus:ring-teal-400/10" />
        <button onClick={onAsk} disabled={!file || loading} className="h-14 rounded-2xl bg-teal-600 px-7 font-semibold text-white shadow-lg shadow-black/30 transition hover:bg-teal-500 disabled:cursor-not-allowed disabled:opacity-50">
          {loading ? 'Asking...' : 'Ask Question'}
        </button>
        {error && <div className="text-sm font-medium text-red-300 xl:col-span-3">{error}</div>}
        {answer && <div className="rounded-2xl border border-white/10 bg-stone-900 p-4 leading-7 text-stone-200 xl:col-span-3">{answer}</div>}
      </div>
    </div>
  )
}
