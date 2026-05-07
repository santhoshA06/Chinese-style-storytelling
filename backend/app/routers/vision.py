from io import BytesIO
from PIL import Image
from fastapi import APIRouter, File, Form, UploadFile
from app.config import settings
from app.schemas import ImageQuestionResponse
from app.services.vision_service import VisionService

router = APIRouter(prefix="/vision", tags=["vision"])
_vision_service = None

def get_vision_service() -> VisionService:
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service

@router.post("/ask", response_model=ImageQuestionResponse)
async def ask_image_question(file: UploadFile = File(...), question: str = Form(...)):
    if not settings.enable_image_qa:
        return ImageQuestionResponse(answer="Image QA is disabled.")
    content = await file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    answer = get_vision_service().answer(image, question)
    return ImageQuestionResponse(answer=answer)
