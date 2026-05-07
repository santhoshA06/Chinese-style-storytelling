import json
import re
from pathlib import Path
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_ID = "Salesforce/blip-image-captioning-base"

IMG_DIR = Path("/content/capstone/lora_training/dataset/images")
OUT_FILE = Path("/content/capstone/lora_training/dataset/metadata.jsonl")
SKIPPED_FILE = Path("/content/capstone/lora_training/dataset/skipped.json")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Set to 50 for testing, None for full run
LIMIT = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

KEEP_WORDS = [
    "mountain", "hill", "river", "water", "tree", "forest",
    "valley", "bridge", "temple", "pagoda", "lake", "stream"
]

SKIP_WORDS = [
    "flower", "flowers", "rooster", "snake", "horse",
    "street", "people", "building", "city", "car", "vehicle"
]


def clean_caption(text: str) -> str:
    text = text.lower().strip()

    # Remove BLIP garbage/repetition
    if "pe pe" in text or text.count("pe") > 5:
        return ""

    # Remove photography words
    text = re.sub(r"\b(photo|photograph|camera|realistic photo)\b", "", text)

    # Normalize spacing
    text = re.sub(r"\s+", " ", text).strip(" ,. ")

    return text


def is_good_image(caption: str) -> bool:
    cap = caption.lower()

    if any(word in cap for word in SKIP_WORDS):
        return False

    if any(word in cap for word in KEEP_WORDS):
        return True

    return False


def enhance_caption(caption: str) -> str | None:
    base = clean_caption(caption)

    if not base:
        return None

    suffix = []

    if any(w in base for w in ["mountain", "hill", "valley", "cliff"]):
        suffix += ["misty shan shui landscape", "layered mountain depth"]

    if any(w in base for w in ["river", "water", "stream", "lake"]):
        suffix += ["water and ink composition", "soft reflections"]

    if any(w in base for w in ["tree", "forest", "branch"]):
        suffix += ["organic tree forms", "delicate brush strokes"]

    if any(w in base for w in ["bridge", "temple", "pagoda"]):
        suffix += ["traditional Chinese architecture", "harmonious composition"]

    if not suffix:
        suffix += ["traditional Chinese ink painting", "rice paper texture"]

    return f"{base}, " + ", ".join(suffix)


def main() -> None:
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMG_DIR}")

    image_files = [
        p for p in sorted(IMG_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]

    if LIMIT is not None:
        image_files = image_files[:LIMIT]

    print(f"Total images to scan: {len(image_files)}")
    print(f"Device: {DEVICE}")

    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    model.eval()

    rows = []
    skipped = []

    for idx, img_path in enumerate(image_files, start=1):
        print(f"\n[{idx}/{len(image_files)}] {img_path.name}")

        try:
            image = Image.open(img_path).convert("RGB")
            image.load()

            inputs = processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    num_beams=3,
                )

            raw_caption = processor.decode(output[0], skip_special_tokens=True)
            cleaned = clean_caption(raw_caption)

            print("raw:", raw_caption)
            print("cleaned:", cleaned)

            if not cleaned or not is_good_image(cleaned):
                skipped.append({"file_name": img_path.name, "caption": cleaned})
                print("skipped")
                continue

            final_caption = enhance_caption(cleaned)

            if not final_caption:
                skipped.append({"file_name": img_path.name, "caption": cleaned})
                print("skipped")
                continue

            row = {
                "file_name": f"images/{img_path.name}",
                "text": final_caption,
            }

            rows.append(row)
            print("final:", final_caption)

        except Exception as e:
            skipped.append({"file_name": img_path.name, "error": str(e)})
            print("error:", e)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with SKIPPED_FILE.open("w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print("metadata:", OUT_FILE)
    print("valid images:", len(rows))
    print("skipped:", len(skipped))


if __name__ == "__main__":
    main()