# System Architecture

```text
USER INPUT
(STORY IDEA)
        ↓
Frontend
(WEB UI)
        ↓
Backend
(ORCHESTRATOR)
        ↓
LLM Story Gen.
(QWEN 2.5 – 3B)
        ↓
Scene Prompts
(REFINED)
        ↓
Diffusion + LoRA
(SD 1.5 FINE-TUNED)
        ↓
Generated Images
(INK-WASH SCENES)
        ↓
STORY BOOK
(JSON + IMAGES)
```

## Workflow Overview

1. User provides a story idea through the frontend web interface  
2. Backend orchestrates the generation pipeline  
3. Qwen 2.5–3B generates scene-wise narrative structure  
4. Scene prompts are refined for visual consistency  
5. Stable Diffusion 1.5 with LoRA generates stylized images  
6. Generated images and story content are combined into a final storybook  
