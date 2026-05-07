export type Scene = {
  scene_id: number
  story_text: string
  image_prompt: string
  image_url?: string
}

export type StorybookResponse = {
  storybook_id: string
  title: string
  summary: string
  scenes: Scene[]
}