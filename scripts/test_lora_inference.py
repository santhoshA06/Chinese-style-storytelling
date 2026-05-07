from pathlib import Path
import traceback
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_DIR = "lora_training/output"

PROMPT = (
    "An ancient temple hidden among tall mountains, pine trees surrounding the structure, light fog drifting through the valley"
)

NEGATIVE = (
    "photograph, realistic camera photo, anime, cartoon, neon colors, "
    "3d render, text, watermark, western oil painting, modern house"
)

def main():
    try:
        print("Checking CUDA...", flush=True)
        print("CUDA available:", torch.cuda.is_available(), flush=True)
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0), flush=True)

        lora_file = Path(LORA_DIR) / "pytorch_lora_weights.safetensors"
        print("Checking LoRA weights...", flush=True)
        print("LoRA path:", lora_file.resolve(), flush=True)
        print("Exists:", lora_file.exists(), flush=True)

        if not lora_file.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_file}")

        print("Loading pipeline...", flush=True)
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            variant="fp16",
        ).to("cuda")

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

        # Do NOT enable xformers here for now
        print("Loading LoRA weights...", flush=True)
        pipe.load_lora_weights(LORA_DIR)
        pipe.fuse_lora()

        print("Generating image...", flush=True)
        image = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            num_inference_steps=20,
            guidance_scale=8.0,
            width=512,
            height=512,
        ).images[0]

        out = Path("backend/generated/lora_test.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)

        print(f"Saved to {out.resolve()}", flush=True)

    except Exception as e:
        print("ERROR:", str(e), flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    main()