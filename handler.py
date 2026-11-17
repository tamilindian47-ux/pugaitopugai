import os
import io
import base64
import random
import torch
import runpod
from PIL import Image
from diffusers import DiffusionPipeline, AutoPipelineForImage2Image

MODEL_ID = "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized"

# Load pipelines globally so they stay in memory across warm requests
print("Loading Text-to-Image pipeline...")
t2i_pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
t2i_pipe.enable_model_cpu_offload()

print("Loading Image-to-Image pipeline...")
i2i_pipe = AutoPipelineForImage2Image.from_pipe(t2i_pipe)


def decode_base64_image(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def encode_image_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def handler(job):
    job_input = job.get("input", {})

    mode = job_input.get("mode", "t2i")  # "t2i" or "i2i"
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")
    num_images = min(max(int(job_input.get("num_images", 1)), 1), 10)
    steps = int(job_input.get("steps", 28))
    guidance_scale = float(job_input.get("guidance_scale", 5.0))
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    strength = float(job_input.get("strength", 0.75))
    base_seed = job_input.get("seed", None)

    if not prompt:
        return {"error": "Prompt is required."}

    init_image = None
    if mode == "i2i":
        image_b64 = job_input.get("image")
        if not image_b64:
            return {"error": "Input image is required for Image-to-Image mode."}
        init_image = decode_base64_image(image_b64)

    results = []
    for i in range(num_images):
        # Use provided seed for the first image (or increment for batch), otherwise random
        if base_seed is not None and str(base_seed).strip() != "":
            current_seed = int(base_seed) + i
        else:
            current_seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device="cpu").manual_seed(current_seed)

        if mode == "i2i":
            output = i2i_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
        else:
            output = t2i_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            )

        img_b64 = encode_image_base64(output.images[0])
        results.append({
            "image": img_b64,
            "seed": current_seed
        })

    return {"images": results}


runpod.serverless.start({"handler": handler})
