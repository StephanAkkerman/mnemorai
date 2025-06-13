import os
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKL,
    AutoPipelineForText2Image,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
)
from huggingface_hub import hf_hub_download
from nunchaku import NunchakuT5EncoderModel
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision
from transformers import BitsAndBytesConfig as BitsAndBytesConfig
from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast

from mnemorai.constants.config import config
from mnemorai.logger import logger
from mnemorai.utils.load_models import select_model
from mnemorai.utils.model_mem import manage_memory


class ImageGen:
    def __init__(self, model: str = None):
        self.config = config.get("IMAGE_GEN", {})
        self.offload = self.config.get("OFFLOAD")

        # Select model based on VRAM or provided model
        if model:
            self.model = model
            logger.debug(f"Using provided image model: {self.model}")
        else:
            self.model = select_model(self.config)
            logger.debug(f"Selected image model based on VRAM: {self.model}")

        self.model_name = self.model.split("/")[-1]
        self.output_dir = Path(self.config.get("OUTPUT_DIR", "output")).resolve()
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_gen_params = self.config.get("PARAMS", {})

        # if seed is provided, set it
        if "seed" in self.image_gen_params:
            if not isinstance(self.image_gen_params["seed"], int):
                logger.warning("Seed must be an integer. Using no seed.")
            else:
                self.image_gen_params["generator"] = torch.Generator(
                    device="cuda"
                ).manual_seed(self.image_gen_params["seed"])
            # remove seed from params to avoid passing it to the pipeline
            del self.image_gen_params["seed"]

        # Initialize pipe to None; will be loaded on first use
        self.pipe = None

    def _get_pipe_func(self):
        if "flux" in self.model_name.lower():
            return FluxPipeline
        else:
            return AutoPipelineForText2Image

    def _initialize_pipe(self):
        """Initialize the pipeline."""
        pipe_func = self._get_pipe_func()
        logger.debug(f"Initializing pipeline for model: {self.model}")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if "flux" in self.model_name.lower():
            bfl_repo = "black-forest-labs/FLUX.1-dev"
            device = "cuda"

            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                bfl_repo,
                subfolder="scheduler",
                torch_dtype=dtype,
                cache_dir="models",
            )
            text_encoder = CLIPTextModel.from_pretrained(
                bfl_repo,
                subfolder="text_encoder",
                torch_dtype=dtype,
                cache_dir="models",
            )
            # T5 encoder in int4
            text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
                "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors",
                cache_dir="models",
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                bfl_repo,
                subfolder="tokenizer",
                torch_dtype=dtype,
                clean_up_tokenization_spaces=True,
                cache_dir="models",
            )
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                bfl_repo,
                subfolder="tokenizer_2",
                torch_dtype=dtype,
                clean_up_tokenization_spaces=True,
                cache_dir="models",
            )
            vae = AutoencoderKL.from_pretrained(
                bfl_repo,
                subfolder="vae",
                torch_dtype=dtype,
                cache_dir="models",
            )
            precision = (
                get_precision()
            )  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors",
                offload=self.config.get("OFFLOAD_T5", True),
            )
            # Set attention implementation to fp16
            transformer.set_attention_impl("nunchaku-fp16")

            params = {
                "scheduler": scheduler,
                "vae": vae,
                "tokenizer": tokenizer,
                "tokenizer_2": tokenizer_2,
                "text_encoder": text_encoder,
                "text_encoder_2": text_encoder_2,
                "transformer": transformer,
            }
            self.pipe = FluxPipeline(**params)  # .to(device, dtype=dtype)

            lora_config = self.config.get("FLUX_LORA", {})
            if lora_config.get("USE_LORA", False):
                logger.info("Loading LoRA weights for FLUX model.")
                transformer.update_lora_params(
                    hf_hub_download(
                        lora_config.get("LORA_REPO"),
                        lora_config.get("LORA_FILE"),
                    )
                )

                transformer.set_lora_strength(lora_config.get("LORA_SCALE", 1.0))

            # offload, does not decrease performance
            if self.config.get("SEQUENTIAL_OFFLOAD", True):
                logger.info("Enabling sequential CPU offload for FLUX model.")
                self.pipe.enable_sequential_cpu_offload(device=device)
        else:
            self.pipe = pipe_func.from_pretrained(
                self.model,
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                cache_dir="models",
            )

    @manage_memory(
        targets=["pipe"],
        delete_attrs=["pipe"],
        move_kwargs={"silence_dtype_warnings": True},
    )
    def generate_img(
        self,
        prompt: str = "A flashy bottle that stands out from the other bottles.",
        word1: str = "flashy",
        word2: str = "bottle",
    ):
        """
        Generate an image from a text prompt using a text-to-image model.

        Parameters
        ----------
        prompt : str, optional
            The prompt to give to the model, by default "A flashy bottle that stands out from the other bottles."
        word1 : str, optional
            The first word of mnemonic, by default "flashy"
        word2 : str, optional
            The second word, by default "bottle"
        """
        file_path = self.output_dir / f"{word1}_{word2}_{self.model_name}.png"

        logger.info(f"Generating image for prompt: {prompt}")
        image = self.pipe(prompt=prompt, **self.image_gen_params).images[0]
        logger.info(f"Saving image to: {file_path}")

        image.save(file_path)

        return file_path


if __name__ == "__main__":
    img_gen = ImageGen()
    img_gen.generate_img()
    img_gen.generate_img("Imagine a cat that walks over the moon", "cat", "moon")
