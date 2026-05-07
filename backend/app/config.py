from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Story LLM ────────────────────────────────────────────────────────────
    # Upgrade from 1.5B → 7B for significantly better image prompt quality.
    # 7B can run on a 16GB GPU (float16). 1.5B produced generic, repetitive prompts.
    # Options:
    #   "Qwen/Qwen2.5-7B-Instruct"   ← recommended (best quality, needs ~14GB VRAM)
    #   "Qwen/Qwen2.5-3B-Instruct"   ← middle ground (needs ~6GB VRAM)
    #   "Qwen/Qwen2.5-1.5B-Instruct" ← original (weakest, use only if GPU < 6GB)
    model_story_llm: str = "Qwen/Qwen2.5-3B-Instruct"
    
    # ── Image Generation ──────────────────────────────────────────────────────
    # SD 1.5 was the original model. SDXL produces much better landscape art
    # but requires retraining LoRA. Set use_sdxl=True to enable the SDXL path.
    # Options:
    #   SD 1.5:  "runwayml/stable-diffusion-v1-5"                (2.0GB, LoRA compatible)
    #   SD 2.1:  "stabilityai/stable-diffusion-2-1"              (3.5GB, better than 1.5)
    #   SDXL:    "stabilityai/stable-diffusion-xl-base-1.0"      (6.5GB, best quality)
    model_sd: str = "runwayml/stable-diffusion-v1-5"

    # Set True to use SDXL pipeline instead of SD 1.5/2.1
    # IMPORTANT: SDXL LoRA weights are NOT compatible with SD 1.5 LoRA weights.
    # If use_sdxl=True, your existing LoRA will not load — retrain on SDXL first.
    use_sdxl: bool = False

    # Optional SDXL refiner — improves fine detail at cost of extra VRAM/time
    # "stabilityai/stable-diffusion-xl-refiner-1.0" or empty string to disable
    model_sdxl_refiner: str = ""

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_dir: str = "../lora_training/output"
    use_lora: bool = True

    # LoRA strength (0.0 = no effect, 1.0 = full effect).
    # For Chinese painting style, 0.75–0.9 is a good balance.
    # Too high (>0.95) makes images look over-stylized.
    lora_scale: float = 0.8

    # ── Image generation defaults ──────────────────────────────────────────────
    # SD 1.5 optimal settings:
    #   guidance_scale: 7.0–7.5 (higher causes artifacts/oversaturation)
    #   num_inference_steps: 30–40 (28 is too few for good landscape detail)
    #   resolution: 768×512 (landscape aspect ratio, max SD1.5 supports well)
    #
    # SDXL optimal settings:
    #   guidance_scale: 7.5–9.0
    #   num_inference_steps: 30–40
    #   resolution: 1024×768 or 1024×1024
    default_guidance_scale: float = 6.5
    default_steps: int = 30
    image_width: int = 768
    image_height: int = 512

    # ── Output ────────────────────────────────────────────────────────────────
    data_output_dir: str = "./generated"

    # ── Vision QA ─────────────────────────────────────────────────────────────
    enable_image_qa: bool = True

    # ── Runtime ───────────────────────────────────────────────────────────────
    # allow_cpu=True lets the app run without a GPU (very slow — dev/testing only)
    allow_cpu: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def output_dir(self) -> Path:
        p = Path(self.data_output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
