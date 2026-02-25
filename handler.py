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
import time
import subprocess
import logging
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image as PILImage
from pydantic import BaseModel, Field
from typing import Literal
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
# Input / Output Models
# -------------------------------------------------
class SkinFixInput(BaseModel):
    image_url: str = Field(
        ...,
        title="Input Image",
        description="URL of the image to enhance and upscale.",
    )
    cfg: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        title="Skin Realism",
        description="Controls how closely the output follows the prompt. Lower = more natural, higher = more processed.",
    )
    skin_refinement: int = Field(
        default=30,
        ge=0,
        le=100,
        title="Skin Refinement",
        description="Controls the denoise strength. Higher = more refinement, lower = closer to original.",
    )
    upscale_resolution: Literal[1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072] = Field(
        default=2048,
        title="Upscale Resolution",
        description="Target resolution for the longest edge of the output image.",
    )
    seed: int = Field(
        default=123456789,
        title="Seed",
        description="Random seed for reproducibility. Use -1 for a random seed.",
    )


class SkinFixOutput(BaseModel):
    images: list[Image] = Field(description="Output images from skin fix processing")


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
    private_logs = True

    def setup(self):
        # ── 1. Detect GPU ──────────────────────────────────────────────────
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
            ).strip()
            debug_log(f"🖥️  GPU Type: {gpu_info}")
        except Exception as e:
            debug_log(f"⚠️  Could not detect GPU: {e}")

        # ── 2. Start ComfyUI immediately in parallel with model downloads ──
        debug_log("🚀 Starting ComfyUI in background...")
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

        # ── 3. Download all models in parallel ────────────────────────────
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

        # ── 4. Wait for ComfyUI to be ready ───────────────────────────────
        debug_log("⏳ Waiting for ComfyUI to become ready...")
        if not check_server(f"http://{COMFY_HOST}/system_stats"):
            raise RuntimeError("ComfyUI failed to start within the timeout window.")
        debug_log("✅ ComfyUI is ready.")

        # ── 5. Background warmup ───────────────────────────────────────────
        threading.Thread(target=self._run_warmup, daemon=True).start()
        debug_log("🔥 Warmup queued in background — setup complete.")

    def _make_dummy_b64(self, size: int) -> str:
        dummy = PILImage.new("RGB", (size, size), (0, 0, 0))
        buf = BytesIO()
        dummy.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _upload_dummy(self, size: int) -> str:
        image_b64 = self._make_dummy_b64(size)
        image_name = f"warmup_{size}_{uuid.uuid4().hex}.png"
        upload_images([{"name": image_name, "image": image_b64}])
        return image_name

    def _run_warmup(self):
        debug_log("🔥 Warmup starting (512px + 2048px)...")
        for warmup_res in (512, 2048):
            try:
                image_name = self._upload_dummy(warmup_res)

                job = copy.deepcopy(WORKFLOW_JSON)
                workflow = job["input"]["workflow"]

                # Set input image
                workflow["43"]["inputs"]["image"] = image_name

                # Configure upscaler — fixed 2x for warmup
                upscaler = workflow["310:160"]["inputs"]
                upscaler["cfg"]        = 1.0
                upscaler["denoise"]    = 0.30
                upscaler["seed"]       = random.randint(0, 2**32 - 1)
                upscaler["upscale_by"] = 2

                # Keep scheduler denoise in sync
                workflow["310:148"]["inputs"]["denoise"] = 0.30

                client_id = str(uuid.uuid4())
                ws = websocket.WebSocket()
                ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

                resp = requests.post(
                    f"http://{COMFY_HOST}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                    timeout=30,
                )
                if resp.status_code != 200:
                    debug_log(f"⚠️  Warmup {warmup_res}px rejected (non-fatal): {resp.text}")
                    ws.close()
                    continue

                while True:
                    out = ws.recv()
                    if not isinstance(out, str) or not out.strip().startswith("{"):
                        continue
                    msg = json.loads(out)
                    if msg.get("type") == "executing" and msg["data"]["node"] is None:
                        break

                ws.close()
                debug_log(f"✅ Warmup {warmup_res}px complete.")

            except Exception as e:
                debug_log(f"⚠️  Warmup {warmup_res}px failed (non-fatal): {e}")

        debug_log("🔥 Warmup finished.")

    @fal.endpoint("/")
    async def handler(self, input: SkinFixInput, response: Response) -> SkinFixOutput:
        try:
            job = copy.deepcopy(WORKFLOW_JSON)
            workflow = job["input"]["workflow"]

            # ── Fetch and upload input image ───────────────────────────────
            image_b64 = image_url_to_base64(input.image_url)
            pil_img = PILImage.open(BytesIO(base64.b64decode(image_b64)))
            input_image_resolution = max(pil_img.size)

            image_name = f"input_{uuid.uuid4().hex}.png"
            upload_images([{"name": image_name, "image": image_b64}])
            workflow["43"]["inputs"]["image"] = image_name

            # ── Compute denoise and seed ───────────────────────────────────
            denoise_strength = 0.30 + (input.skin_refinement / 100.0) * 0.10
            seed = input.seed if input.seed != -1 else random.randint(0, 2**32 - 1)

            # ── Configure upscaler ─────────────────────────────────────────
            target_resolution = max(input.upscale_resolution, input_image_resolution)
            upscaler = workflow["310:160"]["inputs"]
            upscaler["cfg"]        = input.cfg
            upscaler["denoise"]    = denoise_strength
            upscaler["seed"]       = seed
            upscaler["upscale_by"] = round(target_resolution / input_image_resolution, 2)

            # ── Keep scheduler denoise in sync ─────────────────────────────
            workflow["310:148"]["inputs"]["denoise"] = denoise_strength

            # ── Run ComfyUI ────────────────────────────────────────────────
            client_id = str(uuid.uuid4())
            ws = websocket.WebSocket()
            ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

            resp = requests.post(
                f"http://{COMFY_HOST}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=30,
            )

            if resp.status_code != 200:
                debug_log(f"ComfyUI Error Response: {resp.text}")
                raise HTTPException(
                    status_code=500,
                    detail=f"ComfyUI rejected workflow: {resp.text}",
                )

            prompt_id = resp.json()["prompt_id"]

            # ── Wait for completion ────────────────────────────────────────
            while True:
                out = ws.recv()
                if not isinstance(out, str) or not out.strip().startswith("{"):
                    continue
                msg = json.loads(out)
                if msg.get("type") == "executing" and msg["data"]["node"] is None:
                    break

            # ── Collect output images ──────────────────────────────────────
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