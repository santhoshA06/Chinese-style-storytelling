from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.schemas import ScenePlan, StoryGenerateRequest, StorybookResponse
from app.services.image_service import ImageService
from app.services.prompt_template import scene_to_image_prompt
from app.services.storage_service import StorageService
from app.services.story_service import StoryService

router = APIRouter(prefix="/storybooks", tags=["storybooks"])

_story_service: StoryService | None = None
_image_service: ImageService | None = None
_storage_service = StorageService()


def get_story_service() -> StoryService:
    global _story_service
    if _story_service is None:
        _story_service = StoryService(settings.model_story_llm)
    return _story_service


def get_image_service() -> ImageService:
    global _image_service
    if _image_service is None:
        _image_service = ImageService(
            settings.model_sd,
            use_lora_default=settings.use_lora,
        )
    return _image_service


@router.post("/generate", response_model=StorybookResponse)
def generate_storybook(payload: StoryGenerateRequest) -> StorybookResponse:
    story_service = get_story_service()
    image_service = get_image_service()

    story = story_service.generate_storybook_plan(payload.idea, payload.n_scenes)

    storybook_id, story_dir = _storage_service.create_storybook_dir()

    scenes: list[ScenePlan] = []
    for scene in story["scenes"]:
        scene_id = int(scene["scene_id"])
        final_prompt = scene_to_image_prompt(scene["image_prompt"])
        image_path = story_dir / f"scene_{scene_id}.png"

        image_service.generate(
            prompt=final_prompt,
            output_path=image_path,
            steps=payload.num_inference_steps,
            guidance=payload.guidance_scale,
            seed=42 + scene_id,
            use_lora=payload.use_lora,
        )

        scenes.append(
            ScenePlan(
                scene_id=scene_id,
                story_text=scene["story_text"],
                image_prompt=final_prompt,
                image_url=f"/storybooks/files/{storybook_id}/scene_{scene_id}.png",
            )
        )

    response = StorybookResponse(
        storybook_id=storybook_id,
        title=story["title"],
        summary=story["summary"],
        scenes=scenes,
    )

    _storage_service.save_json(story_dir / "story.json", response.model_dump())
    return response