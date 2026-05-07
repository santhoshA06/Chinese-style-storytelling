import torch
from fastapi import APIRouter
from app.config import settings
from app.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])

@router.get("", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        cuda_available=torch.cuda.is_available(),
        story_model=settings.model_story_llm,
        image_model=settings.model_sd,
    )
    