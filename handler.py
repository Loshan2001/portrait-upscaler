import fal
import fal.container
from fal.container import ContainerImage
from fal.toolkit import Image, download_model_weights
from fastapi import Response, HTTPException
from pathlib import Path
import json
import uuid
import base64
import requests
import websocket
import traceback
import os
import copy
import random
import tempfile
import time
import subprocess
import sys
import logging
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image as PILImage
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from comfy_models import MODEL_LIST
from workflow import WORKFLOW_JSON

# Suppress urllib and fal toolkit warnings
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("fal.toolkit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="fal.toolkit")

# -------------------------------------------------
# Container setup
# -------------------------------------------------
PWD = Path(__file__).resolve().parent if __file__ else Path(os.getcwd())
dockerfile_path = f"{PWD}/Dockerfile"
custom_image = ContainerImage.from_dockerfile(dockerfile_path)

COMFY_HOST = "127.0.0.1:8188"
DEBUG_LOGS = os.environ.get("FAL_DEBUG") == "1"


def debug_log(message: str) -> None:
    if DEBUG_LOGS:
        print(message)


# -------------------------------------------------
# Presets (now include seed, cfg, denoise)
# -------------------------------------------------
PRESETS = {
    "imperfect_skin": {"cfg": 0.1, "denoise": 0.34, "resolution": 2048, "seed": 12345},
    "high_end_skin": {"cfg": 1.1, "denoise": 0.30, "resolution": 3072, "seed": 67890},
    "smooth_skin": {
        "cfg": 1.1,
        "denoise": 0.30,
        "resolution": 2048,
        "seed": 13579,
        "prompt_override": True,
        "positive_prompt": (
            "ultra realistic portrait of [subject], flawless clear face, "
            "smooth radiant skin texture, fine pores, balanced complexion, "
            "healthy glow, cinematic lighting"
        ),
        "negative_prompt": (
            "freckles, spots, blemishes, acne, pigmentation, redness, "
            "rough skin, waxy skin, plastic texture, airbrushed"
        ),
    },
    "portrait": {"cfg": 0.5, "denoise": 0.35, "resolution": 2048, "seed": 24680},
    "mid_range": {"cfg": 1.4, "denoise": 0.40, "resolution": 2048, "seed": 11223},
    "full_body": {"cfg": 1.5, "denoise": 0.30, "resolution": 2048, "seed": 44556},
}


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def check_server(url, retries=150, delay=0.2):
    """Poll until ComfyUI is ready. Max wait ~30s."""
    for _ in range(retries):
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def image_url_to_base64(image_url: str) -> str:
    """Download image from URL and convert to base64."""
    response = requests.get(image_url)
    response.raise_for_status()
    pil = PILImage.open(BytesIO(response.content))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def upload_images(images):
    for img in images:
        blob = base64.b64decode(img["image"])
        files = {"image": (img["name"], BytesIO(blob), "image/png")}
        r = requests.post(f"http://{COMFY_HOST}/upload/image", files=files)
        r.raise_for_status()


def _download_and_link(model: dict) -> None:
    """Download a single model and symlink it into the ComfyUI models directory."""
    debug_log(f"⬇️  Downloading: {model['url']}")
    cached_path = download_model_weights(model["url"])
    target_path = model["target"]
    ensure_dir(target_path)
    if os.path.exists(target_path) or os.path.islink(target_path):
        os.unlink(target_path)
    os.symlink(cached_path, target_path)
    debug_log(f"✅ Linked: {cached_path} -> {target_path}")


# -------------------------------------------------
# Input Model
# -------------------------------------------------
class SkinFixInput(BaseModel):
    image_url: str = Field(
        ...,
        title="Input Image",
        description="URL of the image to enhance and upscale.",
    )

    mode: Literal["preset", "custom"] = Field(
        default="preset",
        title="Configuration Mode",
        description="Choose 'preset' to use predefined settings or 'custom' for manual control",
    )

    preset_name: Optional[
        Literal[
            "imperfect_skin",
            "high_end_skin",
            "smooth_skin",
            "portrait",
            "mid_range",
            "full_body",
        ]
    ] = Field(
        default="high_end_skin",
        title="Preset",
        description="Select a preset (only active when mode is 'preset')",
    )

    cfg: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        title="Skin Realism",
        description="Adjust skin realism (only active when mode is 'custom')",
    )

    skin_refinement: int = Field(
        default=30,
        ge=0,
        le=100,
        title="Skin Refinement",
        description="Adjust skin refinement level (only active when mode is 'custom')",
    )

    seed: int = Field(default=123456789, title="Random Seed")

    upscale_resolution: Literal[
        1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072
    ] = Field(
        default=2048,
        title="Upscaler Resolution",
        description="Target resolution for upscaling (only active when mode is 'custom')",
    )


# -------------------------------------------------
# Output Model
# -------------------------------------------------
class SkinFixOutput(BaseModel):
    images: list[Image] = Field(
        description="Output images from skin fix processing"
    )


# -------------------------------------------------
# App
# -------------------------------------------------
class PortraitUpscaler(
    fal.App,
    keep_alive=120,
    min_concurrency=0,
    max_concurrency=5,
    name="portrait_upscaler-V2-A",
):
    """Skin Fix - Advanced skin refinement and upscaling."""

    image = custom_image
    machine_type = "GPU-H100"
    requirements = ["websockets", "websocket-client"]

    private_logs = True  # logs may contain sensitive info

    def setup(self):
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            ).strip()
            debug_log(f"🖥️  GPU Type: {gpu_info}")
        except Exception as e:
            debug_log(f"⚠️  Could not detect GPU: {e}")

        debug_log("🚀 Starting ComfyUI...")
        self.comfy = subprocess.Popen(
            [
                "python", "-u", "/comfyui/main.py",
                "--disable-auto-launch",
                "--disable-metadata",
                "--listen", "--port", "8188",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        debug_log(f"⬇️  Downloading {len(MODEL_LIST)} models in parallel...")
        with ThreadPoolExecutor(max_workers=min(8, len(MODEL_LIST))) as executor:
            futures = {
                executor.submit(_download_and_link, model): model
                for model in MODEL_LIST
            }
            for future in as_completed(futures):
                model = futures[future]
                try:
                    future.result()
                except Exception as e:
                    debug_log(f"❌ Failed to download/link {model['url']}: {e}")
                    raise

        debug_log("✅ All models downloaded and linked.")

        debug_log("⏳ Waiting for ComfyUI to become ready...")
        if not check_server(f"http://{COMFY_HOST}/system_stats"):
            raise RuntimeError("ComfyUI failed to start within the timeout window.")
        debug_log("✅ ComfyUI is ready.")

        # Fire-and-forget warmup
        threading.Thread(target=self._run_warmup, daemon=True).start()
        debug_log("🔥 Warmup queued in background — setup complete.")

    def _run_warmup(self):
        debug_log("🔥 Warmup starting...")
        try:
            dummy = PILImage.new("RGB", (512, 512), (0, 0, 0))
            buf = BytesIO()
            dummy.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode()
            image_name = f"warmup_{uuid.uuid4().hex}.png"
            upload_images([{"name": image_name, "image": image_b64}])
        except Exception as e:
            debug_log(f"⚠️  Warmup failed (non-fatal): {e}")

    @fal.endpoint("/")
    async def handler(self, input: SkinFixInput, response: Response) -> SkinFixOutput:
        try:
            job = copy.deepcopy(WORKFLOW_JSON)
            workflow = job["input"]["workflow"]

            image_b64 = image_url_to_base64(input.image_url)
            pil_img = PILImage.open(BytesIO(base64.b64decode(image_b64)))
            input_image_resolution = max(pil_img.size)

            image_name = f"input_{uuid.uuid4().hex}.png"
            upload_images([{"name": image_name, "image": image_b64}])
            workflow["545"]["inputs"]["image"] = image_name

            sampler = workflow["510"]["inputs"]

            # Apply settings
            if input.mode == "preset":
                p = PRESETS[input.preset_name]
                sampler["cfg"] = p["cfg"]
                sampler["denoise"] = p["denoise"]
                sampler["seed"] = p.get("seed", input.seed)
                target_resolution = max(p["resolution"], input_image_resolution)
                if p.get("prompt_override"):
                    workflow["506"]["inputs"]["part1"] = p["positive_prompt"]
                    workflow["507"]["inputs"]["text"] = p["negative_prompt"]
            else:
                sampler["cfg"] = input.cfg
                sampler["denoise"] = 0.30 + (input.skin_refinement / 100.0) * 0.10
                sampler["seed"] = input.seed
                target_resolution = max(input.upscale_resolution, input_image_resolution)

            workflow["548"]["inputs"]["resolution"] = target_resolution
            workflow["548"]["inputs"]["max_resolution"] = 4096
            workflow["549"]["inputs"]["encode_tile_size"] = min(1024, target_resolution)
            workflow["549"]["inputs"]["decode_tile_size"] = min(1024, target_resolution)
            workflow["548"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

            # Run ComfyUI
            client_id = str(uuid.uuid4())
            ws = websocket.WebSocket()
            ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

            resp = requests.post(
                f"http://{COMFY_HOST}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=30,
            )

            if resp.status_code != 200:
                error_detail = resp.text
                debug_log(f"ComfyUI Error Response: {error_detail}")
                raise HTTPException(status_code=500, detail=f"ComfyUI rejected workflow: {error_detail}")

            prompt_id = resp.json()["prompt_id"]

            while True:
                out = ws.recv()
                if not out.strip().startswith("{"):
                    continue
                msg = json.loads(out)
                if msg.get("type") == "executing" and msg["data"]["node"] is None:
                    break

            history = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}").json()

            images = []
            for node in history[prompt_id]["outputs"].values():
                for img in node.get("images", []):
                    params = (
                        f"filename={img['filename']}"
                        f"&subfolder={img.get('subfolder', '')}"
                        f"&type={img['type']}"
                    )
                    r = requests.get(f"http://{COMFY_HOST}/view?{params}")
                    pil_image = PILImage.open(BytesIO(r.content))
                    output_image = Image.from_pil(pil_image, format="png")
                    images.append(output_image)

            ws.close()
            response.headers["x-fal-billable-units"] = str(len(images))
            return SkinFixOutput(images=images)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))