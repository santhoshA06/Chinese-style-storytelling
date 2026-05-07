import json
import re
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Choose one:
MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
# MODEL_ID = "fancyfeast/llama-joycaption-alpha-two-hf-llava"

IMG_DIR = Path("lora_training/dataset/images")
OUT_FILE = Path("lora_training/dataset/metadata.jsonl")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
USE_IMAGES_PREFIX = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def iter_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def clean_caption(text: str) -> str:
    text = text.strip()

    # Remove obvious chat leftovers
    text = re.sub(r"^assistant\s*[:\-]?\s*", "", text, flags=re.I)
    text = re.sub(r"^caption\s*[:\-]?\s*", "", text, flags=re.I)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove obvious photo words that hurt your painting LoRA
    text = re.sub(r"\b(photo|photograph|photo of|realistic photo)\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ,.")

    return text


def enhance_caption(base_caption: str) -> str:
    """
    Add style language without overwriting image content.
    Keep this mild so the caption still reflects what is actually visible.
    """
    base = clean_caption(base_caption).lower()

    suffix_parts = [
        "traditional Chinese ink wash painting",
        "shan shui style",
        "rice paper texture",
        "elegant brushwork",
    ]

    # Only add atmospheric terms if they're not already there
    if not any(w in base for w in ["mist", "fog", "haze"]):
        suffix_parts.append("soft atmospheric depth")

    if "mountain" in base or "hill" in base or "peak" in base:
        suffix_parts.append("layered mountain distance")

    if "river" in base or "water" in base or "stream" in base:
        suffix_parts.append("water and landscape composition")

    final = f"{base}, " + ", ".join(suffix_parts)
    final = re.sub(r"\s+", " ", final).strip(" ,.")
    return final


def build_prompt_for_model() -> str:
    return (
        "Write one concise training caption for this image. "
        "Describe only what is actually visible. "
        "Focus on landscape, objects, composition, and atmosphere. "
        "Do not mention camera settings. "
        "Do not write a conversation. "
        "Use one sentence."
    )


def caption_image(processor, model, image: Image.Image) -> str:
    prompt = (
        "Write one concise training caption for this image. "
        "Describe only what is actually visible. "
        "Focus on landscape, objects, composition, and atmosphere. "
        "Do not mention camera settings. "
        "Do not write a conversation. "
        "Use one sentence."
    )

    conversation = [
        {
            "role": "user",
            "content": f"<image>\n{prompt}",
        }
    ]

    text_prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text_prompt,
        images=image,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
        )

    decoded = processor.decode(output[0], skip_special_tokens=True)
    if text_prompt in decoded:
        decoded = decoded.split(text_prompt, 1)[-1].strip()
    return decoded.strip()


def main() -> None:
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")

    image_files = list(iter_images(IMG_DIR))
    if not image_files:
        raise ValueError(f"No valid images found in {IMG_DIR}")

    print(f"Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE != "cuda":
        model = model.to(DEVICE)

    rows = []
    for idx, img_path in enumerate(image_files, start=1):
        image = Image.open(img_path).convert("RGB")

        raw_caption = caption_image(processor, model, image)
        final_caption = enhance_caption(raw_caption)

        file_name = f"images/{img_path.name}" if USE_IMAGES_PREFIX else img_path.name
        rows.append({"file_name": file_name, "text": final_caption})

        if idx <= 10:
            print(f"[{idx}] {img_path.name}")
            print(" raw: ", raw_caption)
            print(" final:", final_caption)
            print()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()