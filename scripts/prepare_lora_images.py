from pathlib import Path
from PIL import Image

SRC = Path("lora_training/dataset/images")
SIZE = 512
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def preprocess_image(path: Path):
    img = Image.open(path).convert("RGB")
    img = img.resize((SIZE, SIZE))
    out = path.with_suffix(".jpg")
    img.save(out, quality=95)
    if out != path:
        try:
            path.unlink()
        except Exception:
            pass

def main():
    images = [p for p in SRC.iterdir() if p.suffix.lower() in VALID_EXTS]
    for p in images:
        preprocess_image(p)
    print(f"Processed {len(images)} images")

if __name__ == "__main__":
    main()