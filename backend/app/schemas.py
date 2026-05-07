"""
Pydantic schemas for all API request and response models.
Updated to match new config defaults and add lora_scale control.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

class StoryGenerateRequest(BaseModel):
    idea: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="The story idea or theme to generate a storybook from.",
        examples=["A young scholar climbs misty mountains seeking a hermit's wisdom"],
    )
    n_scenes: int = Field(
        default=3,
        ge=3,
        le=5,
        description="Number of illustrated scenes (3-5).",
    )
    num_inference_steps: int = Field(
        default=35,          # Increased from 28 - better quality
        ge=20,
        le=50,
        description="Diffusion inference steps. More = better quality, slower.",
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=5.0,
        le=12.0,
        description=(
            "Classifier-free guidance scale. "
            "SD 1.5: 7.0-7.5 recommended (8.5+ causes artifacts). "
            "SDXL: 7.5-9.0 recommended."
        ),
    )
    use_lora: bool = Field(
        default=True,
        description="Apply the Chinese painting LoRA fine-tuning.",
    )
    lora_scale: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "LoRA adapter strength (0.0 = disabled, 1.0 = full effect). "
            "0.75-0.90 recommended for balanced style without over-stylization."
        ),
    )

class ScenePlan(BaseModel):
    scene_id: int
    story_text: str
    image_prompt: str
    image_url: Optional[str] = None

class StorybookResponse(BaseModel):
    storybook_id: str
    title: str
    summary: str
    scenes: List[ScenePlan]

class ImageQuestionRequest(BaseModel):
    question: str = Field(min_length=2)

class ImageQuestionResponse(BaseModel):
    answer: str

class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    story_model: str
    image_model: str
    use_sdxl: bool
    use_lora: bool
    lora_dir: str
