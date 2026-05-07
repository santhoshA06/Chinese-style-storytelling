# LoRA Training

This folder contains the workflow and structure used for LoRA fine-tuning of Stable Diffusion 1.5 for traditional Chinese ink painting style generation.

---

# Dataset

The training dataset is hosted on Kaggle:

🔗 Dataset Link:  
https://www.kaggle.com/datasets/santhosh62030/traditional-chinese-ink-painting-images

Download and extract the dataset into:

```text
lora_training/dataset/
```

Expected structure:

```text
lora_training/
│
├── dataset/
│   ├── images/
│   └── metadata.jsonl
│
└── output/
```

---

# Workflow

## 1. Prepare Images

```bash
python scripts/prepare_lora_images.py
```

This script:
- prepares images for training
- removes corrupted files
- standardizes formatting

---

## 2. Generate Metadata

```bash
python scripts/generate_blip_metadata.py
```

This uses BLIP image captioning to:
- generate captions
- filter noisy samples
- create `metadata.jsonl`

---

## 3. Train LoRA

Run:

```bash
accelerate launch train_text_to_image_lora.py
```

The training script fine-tunes Stable Diffusion 1.5 using the prepared dataset and metadata.

---

# Training Output

LoRA weights are saved in:

```text
lora_training/output/
```

Main file:

```text
pytorch_lora_weights.safetensors
```

---

# Inference Testing

Run:

```bash
python scripts/test_lora_inference.py
```

This verifies:
- Stable Diffusion loading
- LoRA loading
- image generation

---

# Models Used

Base diffusion model:

```text
runwayml/stable-diffusion-v1-5
```

Story generation model:

```text
Qwen/Qwen2.5-3B-Instruct
```

Caption generation model:

```text
Salesforce/blip-image-captioning-base
```

---

# Notes

- The dataset and LoRA weights are not stored directly in the repository because of file size limitations.
- Google Colab was used for training when local GPU memory limitations occurred.
- The model performs best on:
  - landscapes
  - mountains
  - rivers
  - bridges
  - temples
  - misty scenery