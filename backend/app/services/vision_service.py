from __future__ import annotations
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

class VisionService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        self.llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", use_fast=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

    def _caption_image(self, image: Image.Image) -> str:
        inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.caption_model.generate(**inputs, max_new_tokens=40)
        return self.caption_processor.decode(output[0], skip_special_tokens=True).strip()

    def _short_answer(self, image: Image.Image, question: str) -> str:
        inputs = self.vqa_processor(image, question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.vqa_model.generate(**inputs, max_new_tokens=20)
        return self.vqa_processor.decode(output[0], skip_special_tokens=True).strip()

    def _expand_answer(self, question: str, caption: str, short_answer: str) -> str:
        prompt = f"""
You are helping explain an uploaded image to a user.

Question: {question}
Image caption: {caption}
Short answer: {short_answer}

Write a helpful answer in English in 2 to 4 sentences.
Be descriptive but do not invent very specific details that are not supported.
"""

        messages = [
            {"role": "system", "content": "You answer clearly and helpfully in English."},
            {"role": "user", "content": prompt},
        ]

        chat = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.llm_tokenizer(chat, return_tensors="pt").to(self.llm_model.device)
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
            )

        text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.split(prompt)[-1].strip() if prompt in text else text.strip()

    def answer(self, image: Image.Image, question: str) -> str:
        caption = self._caption_image(image)
        short_answer = self._short_answer(image, question)
        return self._expand_answer(question, caption, short_answer)