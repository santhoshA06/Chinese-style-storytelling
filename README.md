# AI Storybook Generation using Diffusion Models
### Traditional Chinese Ink Painting Style

This capstone project builds an end-to-end AI storybook generation system. A user enters a short story idea, the system generates a structured storyline with multiple scenes, and each scene is converted into an image in a traditional Chinese ink-wash / shan shui inspired style.

The system combines:

- LLM-based story generation
- Scene-aware prompt engineering
- Stable Diffusion image generation
- LoRA fine-tuning for Chinese painting style
- FastAPI backend
- React frontend

---

# Project Overview

Modern text-to-image models often produce photorealistic or generic outputs and may not preserve culturally specific artistic styles well. This project explores how diffusion models can be adapted to generate storybook-style images inspired by traditional Chinese ink painting.

The system generates:
- Story title
- Story summary
- Scene-wise story text
- Scene prompts
- Generated images

---

# Key Features

- Generate a story from a short user idea
- Split story into multiple visual scenes
- Generate scene-specific image prompts
- Generate images using Stable Diffusion 1.5 + LoRA
- Display the story and images in a frontend interface
- Save generated outputs locally
- Includes scripts for:
  - dataset preparation
  - metadata generation
  - LoRA training
  - inference testing

---

# Tech Stack

## Machine Learning / AI
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Diffusers
- Stable Diffusion 1.5
- LoRA / PEFT
- Qwen 2.5–3B Instruct
- BLIP image captioning

## Backend
- FastAPI
- Uvicorn
- Pydantic

## Frontend
- React
- TypeScript / TSX
- Vite

---

# Repository Structure

```text
Capstone_project/
├── backend/
│   │
│   ├── app/
│   │   │
│   │   ├── routers/
│   │   │   ├── health.py
│   │   │   ├── storybook.py
│   │   │   └── vision.py
│   │   │
│   │   ├── services/
│   │   │   ├── image_service.py
│   │   │   ├── prompt_template.py
│   │   │   ├── storage_service.py
│   │   │   ├── story_service.py
│   │   │   └── vision_service.py
│   │   │
│   │   ├── utils/
│   │   │
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── generated/
│   │
│   ├── .env.example
│   └── requirements.txt
│
├── frontend/
│   │
│   ├── src/
│   │   │
│   │   ├── api/
│   │   │   └── client.ts
│   │   │
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   ├── LoadingOverlay.tsx
│   │   │   ├── ModelSelector.tsx
│   │   │   ├── StoryForm.tsx
│   │   │   ├── StorySceneCard.tsx
│   │   │   └── UploadQuestionCard.tsx
│   │   │
│   │   ├── pages/
│   │   │   └── HomePage.tsx
│   │   │
│   │   ├── types/
│   │   │   └── index.ts
│   │   │
│   │   ├── App.tsx
│   │   ├── index.css
│   │   └── main.tsx
│   │
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── scripts/
│   ├── prepare_lora_images.py
│   ├── generate_blip_metadata.py
│   ├── test_lora_inference.py
│   └── train_text_to_image_lora.py
│
├── docs/
│   ├── sample_outputs/
│   └── sample_story.json
│
├── lora_training/
│   └── README.md
│
├── .env.example
├── .gitignore
└── README.md
---

# Important Note

The dataset and LoRA weights are **NOT included** in this repository because of file size limitations.

Expected local structure:

```text
lora_training/
│
├── dataset/
│   ├── images/
│   └── metadata.jsonl
│
└── output/
    └── pytorch_lora_weights.safetensors
```

---

# Environment Setup

## Windows PowerShell

```powershell
cd E:\Capstone_project

python -m venv backend\venv

backend\venv\Scripts\activate
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install PyTorch (CUDA 12.1):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:

```powershell
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file inside the `backend/` folder.

Example:

```env
MODEL_STORY_LLM=Qwen/Qwen2.5-3B-Instruct
MODEL_SD=runwayml/stable-diffusion-v1-5

USE_LORA=true
LORA_DIR=../lora_training/output

DATA_OUTPUT_DIR=./generated

HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here
```

---

# Dataset Preparation

Dataset structure:

```text
lora_training/dataset/images/
```

Example:

```text
lora_training/dataset/images/001.jpg
lora_training/dataset/images/002.jpg
```

# Prepare LoRA Images

Before generating metadata or training LoRA, preprocess the image dataset:

```bash
python scripts/prepare_lora_images.py
```

This script prepares the dataset images for LoRA training by:

- converting images to RGB format
- resizing/cropping images if needed
- removing unsupported or corrupted files
- saving the cleaned images into the expected training folder

Expected output:

```text
lora_training/dataset/images/
```

After this step, run metadata generation:

Metadata format:

```json
{"file_name":"images/001.jpg","text":"misty mountain landscape with river, shan shui style"}
```

---

# Metadata Generation with BLIP

This project uses **BLIP image captioning** to automatically generate captions.

Run:

```bash
python scripts/generate_blip_metadata.py
```

The script:

- captions images
- filters noisy samples
- removes bad captions
- adds style-aware descriptors
- generates `metadata.jsonl`

Example output:

```text
Valid images: 2318
Skipped: 1628
```

---

# If Local Metadata Generation Is Slow

Use **Google Colab**.

## Recommended Colab Workflow

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Extract dataset:

```python
import zipfile
import os

os.makedirs("/content/capstone", exist_ok=True)

with zipfile.ZipFile("/content/drive/MyDrive/Capstone/dataset.zip", "r") as zip_ref:
    zip_ref.extractall("/content/capstone")
```

Expected structure:

```text
/content/capstone/lora_training/dataset/images
```

Run BLIP metadata generation:

```python
%cd /content/capstone
!python scripts/generate_blip_metadata.py
```

Save metadata back to Drive:

```python
!cp /content/capstone/lora_training/dataset/metadata.jsonl /content/drive/MyDrive/Capstone/metadata.jsonl
```

---

# LoRA Training

This project uses the Hugging Face Diffusers LoRA training script.

Download if missing:

```bash
wget -O train_text_to_image_lora.py https://raw.githubusercontent.com/huggingface/diffusers/v0.30.3/examples/text_to_image/train_text_to_image_lora.py
```

---

# Local Training (Windows PowerShell)

```powershell
accelerate launch train_text_to_image_lora.py `
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" `
  --train_data_dir "lora_training/dataset" `
  --caption_column "text" `
  --resolution 512 `
  --random_flip `
  --train_batch_size 1 `
  --gradient_accumulation_steps 4 `
  --learning_rate 8e-5 `
  --lr_scheduler constant `
  --lr_warmup_steps 0 `
  --max_train_steps 1500 `
  --checkpointing_steps 300 `
  --mixed_precision fp16 `
  --gradient_checkpointing `
  --output_dir "lora_training/output"
```

---

# If Local Training Fails

Training diffusion models locally may fail because of:

- limited GPU memory
- CUDA issues
- Windows environment issues

In that case, use Google Colab.

---

# Google Colab Training Setup

## Install dependencies

```python
!pip install -U pip setuptools wheel

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

!pip install diffusers==0.30.3 transformers==4.47.0 accelerate==0.33.0 peft safetensors pillow sentencepiece
```

---

## Download LoRA training script

```python
%cd /content/capstone

!wget -O train_text_to_image_lora.py https://raw.githubusercontent.com/huggingface/diffusers/v0.30.3/examples/text_to_image/train_text_to_image_lora.py
```

---

## Run LoRA Training

```python
!accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/content/capstone/lora_training/dataset" \
  --caption_column="text" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=8e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --checkpointing_steps=300 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --output_dir="/content/capstone/lora_training/output"
```

---

# Save LoRA Weights to Google Drive

```python
!zip -r /content/lora_output.zip /content/capstone/lora_training/output

!cp /content/lora_output.zip /content/drive/MyDrive/Capstone/lora_output.zip

!cp /content/capstone/lora_training/output/pytorch_lora_weights.safetensors /content/drive/MyDrive/Capstone/pytorch_lora_weights.safetensors
```

---

# LoRA Inference Testing

Place weights here:

```text
lora_training/output/pytorch_lora_weights.safetensors
```

Run:

```bash
python scripts/test_lora_inference.py
```

This verifies:

- CUDA availability
- Stable Diffusion loading
- LoRA loading
- image generation

---

# Running the Backend

```bash
cd backend

uvicorn app.main:app --reload --port 8000
```

Backend URL:

```text
http://localhost:8000
```

---

# Running the Frontend

```bash
cd frontend

npm install

npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

# Example Prompt

```text
A young traveler walks across a wooden bridge over a calm river, surrounded by autumn trees with red leaves, misty mountains in the background.
```

---

# Example Story JSON

```json
{
  "title": "Two Friends Cross a Wooden Bridge in Autumn",
  "summary": "Two childhood friends meet again and cross a wooden bridge together.",
  "scenes": [
    {
      "scene_id": 1,
      "story_text": "The friends arrive at the bridge under a clear sky.",
      "image_prompt": "two children crossing a wooden bridge over a stream"
    }
  ]
}
```

---

# Output Location

Generated outputs are saved in:

```text
backend/generated/
```

---

# Known Limitations

- Character consistency across scenes is limited
- Human figures may appear inconsistent
- Works best for:
  - mountains
  - rivers
  - bridges
  - temples
  - landscapes
- Prompt-to-image alignment is not perfect for complex scenes
- Local GPU memory may limit performance

---

# Future Work

- Improve character consistency
- Add ControlNet support
- Deploy web version publicly
- Support multiple art styles
- Improve evaluation methods

---

# License

This project is intended for academic and educational purposes.