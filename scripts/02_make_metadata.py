import json
import random
import re
from pathlib import Path
from typing import List

IMG_DIR = Path("lora_training/dataset/images")
OUT_FILE = Path("lora_training/dataset/metadata.jsonl")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".png"}

# Use this if your training script expects "images/filename.jpg"
USE_IMAGES_PREFIX = True

# Reproducible caption generation
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

STYLE_PHRASES = [
    "traditional Chinese ink wash painting",
    "traditional Chinese landscape painting",
    "classical Chinese shan shui painting",
    "Chinese brush painting",
    "Chinese monochrome landscape painting",
]

TEXTURE_PHRASES = [
    "rice paper texture",
    "delicate brushwork",
    "elegant ink brush strokes",
    "soft monochrome tones",
    "layered ink shading",
]

ATMOSPHERE_PHRASES = [
    "misty atmosphere",
    "soft morning fog",
    "atmospheric depth",
    "quiet poetic mood",
    "subtle empty space composition",
]

LANDSCAPE_PHRASES = [
    "layered mountains and river",
    "distant peaks and valley",
    "rocky hills and sparse trees",
    "river plain with distant mountains",
    "misty hills and water",
    "mountain landscape with trees",
]

PERSON_PHRASES = [
    "small distant human figure",
    "tiny traveler in the landscape",
    "solitary figure within the scenery",
    "small figure crossing the landscape",
]

OBJECT_PHRASES = {
    "bridge": [
        "wooden bridge over a stream",
        "narrow bridge spanning calm water",
        "small bridge in the landscape",
    ],
    "crane": [
        "white crane in the scene",
        "graceful crane near water",
        "crane visible in the landscape",
    ],
    "bird": [
        "birds flying overhead",
        "small birds in the sky",
        "birds visible above the landscape",
    ],
    "boat": [
        "small wooden boat on the water",
        "boat floating on a calm river",
        "fishing boat in the landscape",
    ],
    "temple": [
        "ancient temple in the distance",
        "small temple among the hills",
        "temple architecture in the mountain landscape",
    ],
    "tree": [
        "weathered trees in the foreground",
        "twisted pine trees near the water",
        "old trees framing the scene",
    ],
    "forest": [
        "forest edge with layered foliage",
        "dense trees along the valley",
        "wooded slopes in the distance",
    ],
    "river": [
        "calm river flowing through the landscape",
        "river winding through the valley",
        "still water reflecting the hills",
    ],
    "waterfall": [
        "waterfall descending through rocks",
        "mist rising from a waterfall",
        "cascade within the mountain scene",
    ],
    "village": [
        "small village in the valley",
        "distant houses within the landscape",
        "village nestled among hills",
    ],
    "girl": [
        "young girl in the landscape",
        "small girl figure near the foreground",
        "girl visible within the scenery",
    ],
    "child": [
        "small child in the landscape",
        "child figure within the scenery",
        "young child near the path",
    ],
}

GENERIC_BACKUPS = [
    "traditional Chinese landscape painting, layered mountains, soft mist, elegant brushwork",
    "classical Chinese shan shui painting, river valley, monochrome ink tones, atmospheric depth",
    "Chinese ink wash landscape, distant peaks, trees, quiet empty space composition",
    "traditional brush painting of mountains and water, soft fog, rice paper texture",
]


def clean_name(name: str) -> str:
    return re.sub(r"[_\-]+", " ", name.lower())


def detect_keywords(filename: str) -> List[str]:
    base = clean_name(Path(filename).stem)
    found = []
    for key in OBJECT_PHRASES:
        if re.search(rf"\b{re.escape(key)}\b", base):
            found.append(key)
    return found


def build_caption(filename: str) -> str:
    keywords = detect_keywords(filename)

    style = random.choice(STYLE_PHRASES)
    texture = random.choice(TEXTURE_PHRASES)
    atmosphere = random.choice(ATMOSPHERE_PHRASES)
    landscape = random.choice(LANDSCAPE_PHRASES)

    parts = [style]

    # Preserve detected objects/subjects if filename gives hints
    if keywords:
        for key in random.sample(keywords, k=min(len(keywords), 2)):
            parts.append(random.choice(OBJECT_PHRASES[key]))

        # If people-related keyword appears, reinforce human presence
        if any(k in {"girl", "child"} for k in keywords):
            parts.append(random.choice(PERSON_PHRASES))
    else:
        # Occasionally include a person phrase, but not too often
        if random.random() < 0.18:
            parts.append(random.choice(PERSON_PHRASES))

    parts.extend([landscape, atmosphere, texture])

    caption = ", ".join(parts)

    # Safety fallback if caption ends up too short or too repetitive
    if len(caption.split()) < 8:
        caption = random.choice(GENERIC_BACKUPS)

    return caption


def main() -> None:
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")

    image_files = [
        p for p in sorted(IMG_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]

    if not image_files:
        raise ValueError(f"No valid image files found in: {IMG_DIR}")

    rows = []
    used_captions = {}

    for img_path in image_files:
        file_name = f"images/{img_path.name}" if USE_IMAGES_PREFIX else img_path.name

        # Try a few times to reduce duplicate captions
        caption = None
        for _ in range(5):
            candidate = build_caption(img_path.name)
            if candidate not in used_captions:
                caption = candidate
                break

        if caption is None:
            caption = build_caption(img_path.name)

        used_captions[caption] = used_captions.get(caption, 0) + 1

        rows.append({
            "file_name": file_name,
            "text": caption
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_FILE}")
    print(f"Unique captions: {len(used_captions)}")
    print("Sample captions:")
    for sample in rows[:10]:
        print("-", sample["text"])


if __name__ == "__main__":
    main()