import type { StorybookResponse } from '../types'

const API_BASE = 'http://localhost:8000/api'

export async function generateStorybook(payload: {
  idea: string
  n_scenes: number
  num_inference_steps: number
  guidance_scale: number
  use_lora: boolean
}): Promise<StorybookResponse> {
  const res = await fetch(`${API_BASE}/storybooks/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function askImageQuestion(file: File, question: string): Promise<{ answer: string }> {
  const form = new FormData()
  form.append('file', file)
  form.append('question', question)
  const res = await fetch(`${API_BASE}/vision/ask`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}