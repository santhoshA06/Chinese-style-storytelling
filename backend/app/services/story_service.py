"""
Final story_service.py

Improvements:
1. Tightened SYSTEM_PROMPT with concrete examples of strong vs weak image_prompts.
   The 3B Qwen model needs explicit show-don't-tell examples to produce
   visually-specific descriptions.

2. Image prompts are required to mention concrete visible objects:
   the subject, what they are doing, and at least 3 environment details.

3. No more landscape-replacement fallbacks. If the LLM produces a usable
   prompt (even short), we strengthen it inline. Only as a last resort do
   we use a generic safe prompt — and even that preserves the user's idea.
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """You are a story planner for an AI app that generates illustrated storybooks rendered in traditional Chinese landscape painting style.

Given a user idea, return EXACTLY this JSON structure:
{
  "title": "...",
  "summary": "...",
  "scenes": [
    {"scene_id": 1, "story_text": "...", "image_prompt": "..."}
  ]
}

OUTPUT RULES:
- Return JSON only. No markdown. No explanations. No preamble.
- All text fields in English. Zero Chinese characters anywhere.
- Exactly N scenes.

STORY_TEXT RULES:
- 2-3 short sentences
- Child-friendly language
- Describes what happens in this specific scene

IMAGE_PROMPT RULES (CRITICAL — these must be visually concrete):
- 25 to 45 words
- Must include in this order:
  (a) THE SUBJECT (e.g. "a small girl in red robes", "a young scholar")
  (b) WHAT THEY ARE DOING (e.g. "crossing a wooden bridge", "looking at a crane")
  (c) THE SETTING (e.g. "over a still river", "at the edge of a bamboo forest")
  (d) ATMOSPHERE (time of day, weather, light)
  (e) 2-3 VISIBLE OBJECTS in foreground or background (e.g. "willow trees on the bank, distant misty mountains, a pavilion on the hill")
- Keep the subject SMALL in frame unless the user explicitly asks for a portrait
- No close-up faces
- No invented elements that contradict the scene

GOOD EXAMPLE:
"a small girl in flowing red robes standing on an arched wooden bridge over a still river, gazing at a white crane perched on a pine branch, soft morning mist, willow trees on the bank, distant misty mountains visible behind"

BAD EXAMPLES (DO NOT WRITE LIKE THIS):
- "a girl on a bridge" (no detail, no setting, no atmosphere)
- "beautiful Chinese landscape with a girl" (vague, no action)
- "main subject clearly visible" (this is meta-instruction, not visual content)

CONSISTENCY:
- If the user idea mentions a specific object (bridge, crane, temple, boat, deer),
  it MUST appear in every relevant scene.
- Do not introduce unrelated elements (e.g. don't add fishermen if the story is about a girl and a crane).
"""

def _strip_cjk(text: str) -> str:
    cleaned = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef]", " ", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()

def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))

def _word_count(s: str) -> int:
    return len(s.split())

def _strengthen_short_prompt(prompt: str, story_text: str) -> str:
    """
    If the model produced a too-short prompt, add visual specificity drawn from
    the story_text rather than replacing the prompt with an unrelated landscape.
    """
    base = prompt.strip().rstrip(".,;:")
    if not base:
        base = _strip_cjk(story_text).strip().rstrip(".,;:")
    if not base:
        base = "a small figure in a peaceful Chinese landscape"

    # Add atmospheric detail without replacing the core scene
    additions = [
        "soft natural light",
        "atmospheric mist",
        "trees and rocks framing the scene",
        "distant misty mountains in the background",
    ]
    return f"{base}, " + ", ".join(additions)

class StoryService:
    def __init__(self, model_id: str):
        import os
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token:
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
                print("[StoryService] Hugging Face login successful.")
            except Exception as e:
                print(f"[StoryService] Hugging Face login warning: {e}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        print(f"[StoryService] Loading LLM: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.eval()
        print(f"[StoryService] LLM ready on {self.device}.")

    # JSON extraction 

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass
        depth, start = 0, -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        continue
        raise ValueError(f"No valid JSON in output. First 300 chars: {text[:300]}")

    # Generation

    def _generate_once(self, idea: str, n_scenes: int) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User story idea: {idea}\nN: {n_scenes}\n\nReturn JSON only.",
            },
        ]
        chat = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            chat, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1400,
                do_sample=True,
                temperature=0.55,
                top_p=0.9,
                repetition_penalty=1.08,
            )

        n_input = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][n_input:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._extract_json(decoded)

    def generate_storybook_plan(self, idea: str, n_scenes: int) -> Dict[str, Any]:
        result = self._generate_once(idea, n_scenes)

        # Retry if any CJK characters leaked
        if contains_cjk(json.dumps(result, ensure_ascii=False)):
            print("[StoryService] CJK detected — retrying with stricter English-only instruction.")
            result = self._generate_once(
                f"{idea}\n\nCRITICAL: Every field must be English only. Zero Chinese characters anywhere.",
                n_scenes,
            )

        # Validate keys
        for key in ("title", "summary", "scenes"):
            if key not in result:
                raise ValueError(f"Missing key in model output: {key}")
        scenes = result.get("scenes", [])
        if not isinstance(scenes, list):
            raise ValueError("'scenes' is not a list.")

        # Pad/trim to n_scenes
        if len(scenes) > n_scenes:
            scenes = scenes[:n_scenes]
        elif len(scenes) < n_scenes:
            for i in range(len(scenes) + 1, n_scenes + 1):
                scenes.append({
                    "scene_id" : i,
                    "story_text" : "The journey continues through the landscape.",
                    "image_prompt" : "a small distant figure walking through misty mountains, with pine trees and a winding river",
                })

        # Clean each scene
        cleaned = []
        for idx, scene in enumerate(scenes, start=1):
            scene_id   = int(scene.get("scene_id", idx))
            story_text = _strip_cjk(str(scene.get("story_text", "The story continues."))).strip()

            raw_prompt    = str(scene.get("image_prompt", "")).strip()
            image_prompt  = _strip_cjk(raw_prompt).strip()

            if _word_count(image_prompt) < 12:
                print(f"[StoryService] Scene {scene_id} prompt too short ({_word_count(image_prompt)} words) — strengthening with story context.")
                image_prompt = _strengthen_short_prompt(image_prompt, story_text)

            cleaned.append({
                "scene_id":     scene_id,
                "story_text":   story_text,
                "image_prompt": image_prompt,
            })

        result["scenes"]  = cleaned
        result["title"]   = _strip_cjk(str(result.get("title", "")))   .strip() or "Untitled"
        result["summary"] = _strip_cjk(str(result.get("summary", ""))) .strip() or ""
        return result
