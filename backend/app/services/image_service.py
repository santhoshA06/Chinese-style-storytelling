from __future__ import annotations
import os
from pathlib import Path
import torch
from app.config import settings
from app.services.prompt_template import negative_prompt, _strip_cjk

def _is_sdxl_model(model_id: str) -> bool:
    sdxl_markers = ["xl", "sdxl", "stable-diffusion-xl"]
    return any(m in model_id.lower() for m in sdxl_markers)

class ImageService:
    def __init__(self, model_id: str, use_lora_default: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda" and not settings.allow_cpu:
            raise RuntimeError(
                "CUDA GPU required. Set ALLOW_CPU=true in .env to enable CPU mode."
            )
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        self.model_id         = model_id
        self.use_lora_default = use_lora_default
        self._lora_loaded     = False
        self._is_sdxl         = settings.use_sdxl or _is_sdxl_model(model_id)
        self._compel          = None   # initialised after pipe loads

        print(f"[ImageService] Loading {'SDXL' if self._is_sdxl else 'SD'} model: {model_id}")
        print(f"[ImageService] Device: {self.device}")

        self._load_pipeline(model_id)
        self._init_compel()           # set up long-prompt handler
        self._refiner = None
        if self._is_sdxl and settings.model_sdxl_refiner:
            self._load_refiner(settings.model_sdxl_refiner)

    # Pipeline Setup

    def _load_pipeline(self, model_id: str) -> None:
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        common = dict(
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
        )
        if self._is_sdxl:
            from diffusers import StableDiffusionXLPipeline
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                variant="fp16" if self.device == "cuda" else None,
                **common,
            )
        else:
            from diffusers import StableDiffusionPipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id, **common)

        self._set_scheduler()
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        if hasattr(self.pipe, "enable_vae_tiling"):
            self.pipe.enable_vae_tiling()

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[ImageService] xformers enabled.")
        except Exception:
            print("[ImageService] xformers not available - using default attention.")

        if self._is_sdxl and self.device == "cuda":
            try:
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram < 12.0:
                    print(f"[ImageService] Low VRAM ({vram:.1f}GB) - enabling CPU offload.")
                    self.pipe.enable_sequential_cpu_offload()
            except Exception:
                pass

    def _set_scheduler(self) -> None:
        try:
            from diffusers import DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                solver_order=2,
                use_karras_sigmas=True,
            )
            print("[ImageService] Scheduler: DPM++ 2M Karras")
        except Exception as e:
            print(f"[ImageService] Scheduler fallback: {e}")

    # Compel (long prompt handler)

    def _init_compel(self) -> None:
        """
        Initialise compel for long-prompt support.

        Without compel: SD 1.5 CLIP truncates at token 77 — everything after
        is silently ignored. Our style suffix is ~158 tokens, so ALL art style
        keywords were being discarded.

        With compel: the prompt is split into 77-token windows, each encoded
        separately, and the embeddings are concatenated/interpolated so the
        full prompt is seen by the model.

        Install: pip install compel
        """
        try:
            if self._is_sdxl:
                from compel import Compel, ReturnedEmbeddingsType
                self._compel = Compel(
                    tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                    text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False,
                )
            else:
                from compel import Compel
                self._compel = Compel(
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    truncate_long_prompts=False,  # let compel handle chunking
                )
            print("[ImageService] compel long-prompt handler enabled.")
        except ImportError:
            print(
                "[ImageService] WARNING: compel not installed.\n"
                "  Prompts >77 tokens will be truncated (art style keywords lost).\n"
                "  Fix: pip install compel"
            )
            self._compel = None
        except Exception as e:
            print(f"[ImageService] compel init failed ({e}) - falling back to raw strings.")
            self._compel = None

    def _encode_prompt(self, prompt: str, neg_prompt: str):
        """
        Encode prompt using compel (handles >77 tokens) or fall back to raw strings.
        Returns (conditioning, uncond_conditioning) or (prompt_str, neg_str).
        """
        if self._compel is None:
            return prompt, neg_prompt

        try:
            if self._is_sdxl:
                conditioning, pooled = self._compel(prompt)
                neg_conditioning, neg_pooled = self._compel(neg_prompt)
                [conditioning, neg_conditioning] = self._compel.pad_conditioning_tensors_to_same_length(
                    [conditioning, neg_conditioning]
                )
                return (conditioning, pooled), (neg_conditioning, neg_pooled)
            else:
                conditioning     = self._compel(prompt)
                neg_conditioning = self._compel(neg_prompt)
                [conditioning, neg_conditioning] = self._compel.pad_conditioning_tensors_to_same_length(
                    [conditioning, neg_conditioning]
                )
                return conditioning, neg_conditioning
        except Exception as e:
            print(f"[ImageService] compel encode failed ({e}) - using raw string fallback.")
            return prompt, neg_prompt

    # LoRA

    def _ensure_lora(self) -> None:
        if self._lora_loaded:
            return
        lora_path    = Path(settings.lora_dir)
        weights_file = lora_path / "pytorch_lora_weights.safetensors"
        if not weights_file.exists():
            raise FileNotFoundError(
                f"LoRA weights not found: {weights_file}\n"
                "Run LoRA training first, or set USE_LORA=false in .env."
            )
        print(f"[ImageService] Loading LoRA: {lora_path}")
        self.pipe.load_lora_weights(str(lora_path), adapter_name="chinese_painting")
        self._lora_loaded = True
        print("[ImageService] LoRA loaded (unfused).")

    def _set_lora_scale(self, scale: float) -> None:
        if not self._lora_loaded:
            return
        try:
            self.pipe.set_adapters(["chinese_painting"], adapter_weights=[scale])
        except Exception as e:
            print(f"[ImageService] LoRA scale error: {e}")

    def _load_refiner(self, refiner_id: str) -> None:
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            self._refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
            ).to(self.device)
            print(f"[ImageService] SDXL refiner loaded: {refiner_id}")
        except Exception as e:
            print(f"[ImageService] Refiner load failed: {e}")
            self._refiner = None

    # Generation

    def generate(
        self,
        prompt: str,
        output_path: Path,
        steps: int,
        guidance: float,
        seed: int,
        use_lora: bool,
        lora_scale: float | None = None,
    ) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Safety: strip any CJK that might have leaked into the final prompt
        prompt = _strip_cjk(prompt)

        # LoRA
        if use_lora:
            self._ensure_lora()
            self._set_lora_scale(lora_scale if lora_scale is not None else settings.lora_scale)
        elif self._lora_loaded:
            self._set_lora_scale(0.0)

        # Resolution
        width  = settings.image_width
        height = settings.image_height
        if self._is_sdxl and width < 768:
            width, height = 1024, 768

        neg   = negative_prompt(for_sdxl=self._is_sdxl)
        gen   = torch.Generator(device=self.device).manual_seed(seed)

        # Encode with compel (handles >77 tokens correctly)
        cond, uncond = self._encode_prompt(prompt, neg)

        # Compel returns tensors, raw fallback returns strings
        using_compel = not isinstance(cond, str)

        if self._is_sdxl and self._refiner:
            if using_compel:
                (cond_embeds, pooled), (uncond_embeds, neg_pooled) = cond, uncond
                base_out = self.pipe(
                    prompt_embeds=cond_embeds,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=uncond_embeds,
                    negative_pooled_prompt_embeds=neg_pooled,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=gen,
                    width=width, height=height,
                    output_type="latent",
                ).images[0]
            else:
                base_out = self.pipe(
                    prompt=cond, negative_prompt=uncond,
                    num_inference_steps=steps, guidance_scale=guidance,
                    generator=gen, width=width, height=height,
                    output_type="latent",
                ).images[0]
            image = self._refiner(
                prompt=prompt, negative_prompt=neg,
                image=base_out,
                num_inference_steps=max(steps // 3, 10),
                guidance_scale=guidance, generator=gen,
            ).images[0]

        else:
            if using_compel:
                image = self.pipe(
                    prompt_embeds=cond,
                    negative_prompt_embeds=uncond,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=gen,
                    width=width,
                    height=height,
                ).images[0]
            else:
                image = self.pipe(
                    prompt=cond,
                    negative_prompt=uncond,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=gen,
                    width=width,
                    height=height,
                ).images[0]

        image.save(output_path)
        print(f"[ImageService] Saved: {output_path}")
        return str(output_path)
