"""
Key principles applied here:

1. SUBJECT FIRST. CLIP weights early tokens far more heavily.
   The previous version started with "masterpiece, best quality, highly detailed"
   which wasted the most-weighted 8 tokens on generic quality words.
   Now the actual scene description goes first.

2. EMPHASIS via parentheses. SD 1.5 (with compel) supports (term) and (term:1.3)
   syntax to boost specific concepts. We boost the main subject and the most
   important visible objects.

3. STYLE LAST. Style tokens come at the end where CLIP weighs them less but
   the LoRA reinforces them anyway.

4. NEGATIVE PROMPT covers all observed failure modes:
   text/calligraphy artifacts, fabric textures, seal stamps, modern objects,
   photo-realism, deformed anatomy, frames/borders.
"""

from __future__ import annotations
import re

STYLE = (
    "traditional Chinese ink wash painting, shan shui art, "
    "rice paper texture, soft mist, sepia tones, "
    "elegant brush strokes"
)

# Quality tokens (kept short, placed at the END to avoid overshadowing the scene description)
QUALITY_TAIL = "highly detailed, masterpiece"

# Negative prompt (comprehensive coverage of all failure modes seen so far)
NEGATIVE = (
    # Wrong art mediums
    "photograph, photorealistic, 3d render, anime, cartoon, oil painting, "
    "watercolor, digital art, "
    # Modern intrusions
    "modern clothing, modern city, cars, electricity poles, signs, "
    # Text artifacts (calligraphy was the biggest visible problem)
    "text, letters, words, calligraphy, writing, inscription, "
    "seal stamp, signature, label, caption, watermark, logo, "
    # Wrong-canvas artifacts (the linen/woven look)
    "canvas texture, fabric texture, linen weave, "
    # Frame/border artifacts (saw these in scene 1 and 3)
    "border, frame, picture frame, matting, "
    # Anatomy / quality issues
    "close-up face, distorted face, deformed, extra limbs, bad anatomy, "
    "low quality, blurry, pixelated, jpeg artifacts"
)

# Subject keyword -> emphasised replacement
# When these terms appear in the scene description, we boost them with
# compel emphasis syntax so SD really pays attention to them.
EMPHASIS_TERMS = [
    "girl", "boy", "child", "woman", "man", "scholar", "monk", "fisherman",
    "bridge", "wooden bridge", "stone bridge", "pavilion",
    "crane", "bird", "owl", "deer", "tiger",
    "boat", "fishing boat",
    "waterfall", "river", "mountain", "pine tree", "bamboo",
    "temple", "pagoda", "village",
    "moon", "sun", "lantern",
]

# Scene type -> atmospheric extras
# These add Chinese-painting-appropriate atmosphere keywords.
SCENE_ATMOSPHERE = {
    "morning":  "soft golden morning light, dawn mist",
    "evening":  "warm amber evening light, dusk haze",
    "night":    "moonlit, soft moonlight, deep shadows",
    "winter":   "snowy peaks, bare trees, cold pale tones",
    "autumn":   "scattered red and gold leaves, crisp air",
    "spring":   "blossoming branches, fresh green tones",
    "rain":     "light rain, wet rocks, drifting mist",
    "fog":      "thick atmospheric fog, layered haze",
    "mist":     "drifting mist, soft atmospheric haze",
    "snow":     "light snow falling, pale winter palette",
}

def _strip_cjk(text: str) -> str:
    """Remove CJK (Chinese, Japanese, Korean) characters and CJK punctuation."""
    cleaned = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]", " ", text)
    cleaned = re.sub(r"[\u3000-\u303f\uff00-\uffef]", " ", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()

def _emphasise_subjects(text: str) -> str:
    """
    Wrap key subject words in compel emphasis syntax: word -> (word:1.3)
    This tells SD to weight these concepts more heavily during generation.
    """
    result = text
    # Sort by length descending so multi-word matches happen before single words
    sorted_terms = sorted(EMPHASIS_TERMS, key=len, reverse=True)
    for term in sorted_terms:
        # Word boundary match, case-insensitive, only emphasise first occurrence
        pattern = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
        match = pattern.search(result)
        if match:
            # Skip if already inside an emphasis block
            start = match.start()
            preceding = result[:start]
            if preceding.count("(") - preceding.count(")") > 0:
                continue
            result = pattern.sub(rf"(\1:1.3)", result, count=1)
    return result

def _detect_atmosphere(text: str) -> str:
    """Detect time-of-day / weather keywords and return matching atmosphere phrase."""
    text_lower = text.lower()
    matched: list[str] = []
    for keyword, phrase in SCENE_ATMOSPHERE.items():
        if re.search(rf"\b{keyword}\b", text_lower):
            matched.append(phrase)
            if len(matched) >= 2:
                break
    return ", ".join(matched)

def scene_to_image_prompt(scene_description: str, for_sdxl: bool = False) -> str:
    """
    Build the final SD prompt.
    Order:
      1. Scene description (with subject emphasis) gets highest CLIP weight
      2. Atmosphere extras (if detected)
      3. Composition hint
      4. Style
      5. Quality
    """
    desc = _strip_cjk(scene_description).strip().rstrip(".,;:")

    if not desc or len(desc.split()) < 3:
        desc = "a small distant figure in a misty Chinese mountain landscape"

    # Boost main subjects with emphasis syntax
    emphasised = _emphasise_subjects(desc)

    # Detect atmosphere
    atmosphere = _detect_atmosphere(desc)

    # Composition explicit but not heavy-handed
    composition = "wide composition, clear depth, balanced foreground and background"

    # Build the final prompt - scene FIRST
    parts = [emphasised]
    if atmosphere:
        parts.append(atmosphere)
    parts.append(composition)
    parts.append(STYLE)
    parts.append(QUALITY_TAIL)

    return ", ".join(p.strip().rstrip(",") for p in parts if p.strip())

def negative_prompt(for_sdxl: bool = False) -> str:
    if for_sdxl:
        return NEGATIVE + ", oversaturated, lens flare, bokeh, depth of field"
    return NEGATIVE
