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
# Presets
# -------------------------------------------------
PRESETS = {
    "imperfect_skin": {"cfg": 0.1, "denoise": 0.34, "resolution": 2048, "seed": 12345},
    "high_end_skin":  {"cfg": 1.1, "denoise": 0.30, "resolution": 3072, "seed": 67890},
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
    "portrait":   {"cfg": 0.5, "denoise": 0.35, "resolution": 2048, "seed": 24680},
    "mid_range":  {"cfg": 1.4, "denoise": 0.40, "resolution": 2048, "seed": 11223},
    "full_body":  {"cfg": 1.5, "denoise": 0.30, "resolution": 2048, "seed": 44556},
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
    mode: Literal["preset", "custom"] = Field(
        default="preset",
        title="Configuration Mode",
        description="Choose 'preset' to use predefined settings or 'custom' for manual control",
    )
    preset_name: Optional[
        Literal["imperfect_skin", "high_end_skin", "smooth_skin", "portrait", "mid_range", "full_body"]
    ] = Field(
        default="high_end_skin",
        title="Preset",
        description="Select a preset (only active when mode is 'preset')",
    )
    cfg: float = Field(
        default=1.0, ge=0.0, le=2.0,
        title="Skin Realism",
        description="Adjust skin realism (only active when mode is 'custom')",
    )
    skin_refinement: int = Field(
        default=30, ge=0, le=100,
        title="Skin Refinement",
        description="Adjust skin refinement level (only active when mode is 'custom')",
    )
    seed: int = Field(default=123456789, title="Random Seed")
    upscale_resolution: Literal[1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072] = Field(
        default=2048,
        title="Upscaler Resolution",
        description="Target resolution for upscaling (only active when mode is 'custom')",
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

        # ── 2. Start ComfyUI immediately (don't wait for models first) ─────
        #    FIX #2: ComfyUI boots in parallel with model downloads.
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
        #    FIX #1: All models download simultaneously via ThreadPoolExecutor.
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

        # ── 5. Warmup in background at multiple resolutions ───────────────
        #    FIX #3: Warmup doesn't block setup completion.
        #    FIX #4: Warmup hits both 512px and 2048px so GPU kernels are
        #            pre-compiled for the resolutions real requests will use.
        threading.Thread(target=self._run_warmup, daemon=True).start()
        debug_log("🔥 Warmup queued in background — setup complete.")

    def _make_dummy_b64(self, size: int) -> str:
        """Create a solid-black dummy PNG at the given square resolution."""
        dummy = PILImage.new("RGB", (size, size), (0, 0, 0))
        buf = BytesIO()
        dummy.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _upload_dummy(self, size: int) -> str:
        """Upload a dummy image and return its filename."""
        image_b64 = self._make_dummy_b64(size)
        image_name = f"warmup_{size}_{uuid.uuid4().hex}.png"
        upload_images([{"name": image_name, "image": image_b64}])
        return image_name

    def _run_warmup(self):
        """
        FIX #4: Warm up at low AND high resolution.
        The GPU's CUDA kernels are JIT-compiled per resolution on first use,
        so warming up at both sizes prevents slowness on the first real request.
        """
        debug_log("🔥 Warmup starting (512px + 2048px)...")
        for warmup_res in (512, 2048):
            try:
                image_name = self._upload_dummy(warmup_res)

                # Build a minimal workflow snapshot for warmup
                job = copy.deepcopy(WORKFLOW_JSON)
                workflow = job["input"]["workflow"]
                workflow["545"]["inputs"]["image"] = image_name

                sampler = workflow["510"]["inputs"]
                sampler["cfg"] = 1.0
                sampler["denoise"] = 0.30
                sampler["seed"] = random.randint(0, 2**32 - 1)

                target_res = warmup_res
                workflow["548"]["inputs"]["resolution"] = target_res
                workflow["548"]["inputs"]["max_resolution"] = 4096
                workflow["548"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
                workflow["549"]["inputs"]["encode_tile_size"] = min(1024, target_res)
                workflow["549"]["inputs"]["decode_tile_size"] = min(1024, target_res)

                client_id = str(uuid.uuid4())
                ws = websocket.WebSocket()
                ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

                resp = requests.post(
                    f"http://{COMFY_HOST}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                    timeout=30,
                )
                if resp.status_code != 200:
                    debug_log(f"⚠️  Warmup {warmup_res}px rejected by ComfyUI (non-fatal): {resp.text}")
                    ws.close()
                    continue

                # Drain websocket until execution finishes
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
            # Fresh deep-copy each request so concurrent calls don't collide
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
                sampler["cfg"]    = p["cfg"]
                sampler["denoise"] = p["denoise"]
                sampler["seed"]   = p.get("seed", input.seed)
                target_resolution = max(p["resolution"], input_image_resolution)
                if p.get("prompt_override"):
                    workflow["506"]["inputs"]["part1"] = p["positive_prompt"]
                    workflow["507"]["inputs"]["text"]  = p["negative_prompt"]
            else:
                sampler["cfg"]    = input.cfg
                sampler["denoise"] = 0.30 + (input.skin_refinement / 100.0) * 0.10
                sampler["seed"]   = input.seed
                target_resolution = max(input.upscale_resolution, input_image_resolution)

            workflow["548"]["inputs"]["resolution"]     = target_resolution
            workflow["548"]["inputs"]["max_resolution"] = 4096
            workflow["548"]["inputs"]["seed"]           = random.randint(0, 2**32 - 1)
            workflow["549"]["inputs"]["encode_tile_size"] = min(1024, target_resolution)
            workflow["549"]["inputs"]["decode_tile_size"] = min(1024, target_resolution)

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
                debug_log(f"ComfyUI Error Response: {resp.text}")
                raise HTTPException(
                    status_code=500,
                    detail=f"ComfyUI rejected workflow: {resp.text}",
                )

            prompt_id = resp.json()["prompt_id"]

            while True:
                out = ws.recv()
                if not isinstance(out, str) or not out.strip().startswith("{"):
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