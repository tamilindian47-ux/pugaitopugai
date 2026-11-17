import base64
import gc
import io
import secrets
import threading

import runpod
import torch
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
from diffusers import (
    GGUFQuantizationConfig,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)

MODEL_REPO = "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized"
BASE_REPO = "Qwen/Qwen-Image-2.1"
MODEL_FILE = "qwen-image-2.1-step90000-Q5_K_M.gguf"

pipe = None
lock = threading.Lock()


def load_pipeline():
    global pipe

    if pipe is not None:
        return pipe

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    print("Downloading selected Q5_K_M transformer...", flush=True)
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    print("Loading quantized transformer...", flush=True)
    transformer = QwenImage21Transformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.bfloat16
        ),
        config=BASE_REPO,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    print("Loading base text encoder, processor and VAE...", flush=True)
    loaded_pipe = QwenImage21Pipeline.from_pretrained(
        BASE_REPO,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    loaded_pipe.enable_model_cpu_offload()

    pipe = loaded_pipe
    print("Pipeline ready.", flush=True)
    return pipe


def decode_image(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Image-to-image requires a Base64 image.")

    if value.startswith("data:"):
        value = value.split(",", 1)[1]

    raw = base64.b64decode(value, validate=True)

    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Input image exceeds the 8 MiB application limit.")

    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image)
        image.load()
        return image.convert("RGBA")


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@torch.inference_mode()
def generate(data):
    mode = data.get("mode", "t2i")
    if mode not in ("t2i", "i2i"):
        raise ValueError("Mode must be t2i or i2i.")

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Prompt is required.")

    count = int(data.get("num_images", 1))
    if not 1 <= count <= 10:
        raise ValueError("Number of images must be between 1 and 10.")

    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    if any(v < 256 or v > 1536 or v % 64 for v in (width, height)):
        raise ValueError(
            "Width and height must be multiples of 64, from 256 to 1536."
        )

    steps = int(data.get("steps", 35))
    if not 1 <= steps <= 80:
        raise ValueError("Steps must be between 1 and 80.")

    cfg = float(data.get("guidance_scale", 5.0))
    if not 1.0 <= cfg <= 15.0:
        raise ValueError("CFG must be between 1 and 15.")

    supplied_seed = data.get("seed")
    base_seed = (
        secrets.randbelow(2**32)
        if supplied_seed is None or str(supplied_seed).strip() == ""
        else int(supplied_seed)
    )
    if not 0 <= base_seed < 2**32:
        raise ValueError("Seed must be between 0 and 4294967295.")

    negative = data.get("negative_prompt")
    if negative is None or not str(negative).strip():
        negative = " "

    reference = decode_image(data.get("image")) if mode == "i2i" else None
    pipeline = load_pipeline()
    results = []

    for index in range(count):
        seed = (base_seed + index) % (2**32)
        print(f"Generating image {index + 1}/{count}; seed={seed}", flush=True)

        args = {
            "prompt": prompt,
            "negative_prompt": str(negative),
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "true_cfg_scale": cfg,
            "use_kv_cache": True,
            "generator": torch.Generator("cuda").manual_seed(seed),
        }

        if reference is not None:
            args["image"] = [reference]
            args["output_resolution"] = 1024

        result = pipeline(**args).images[0]
        results.append({
            "image": encode_image(result),
            "seed": seed,
            "width": result.width,
            "height": result.height,
        })
        del result

    return {"images": results}


def handler(job):
    try:
        with lock:
            return generate(job.get("input", {}))
    except Exception as error:
        import traceback
        traceback.print_exc()
        return {"error": f"{type(error).__name__}: {error}"}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})
