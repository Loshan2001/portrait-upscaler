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
WS_TIMEOUT  = 300  # max seconds to wait for a single generation


def debug_log(message: str) -> None:
    if DEBUG_LOGS:
        print(message)


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def check_server(url, retries=300, delay=0.5):
    """Poll until ComfyUI is ready. Max wait ~150s."""
    for _ in range(retries):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def image_url_to_base64(image_url: str) -> str:
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    pil = PILImage.open(BytesIO(response.content))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def upload_images(images):
    for img in images:
        blob = base64.b64decode(img["image"])
        files = {"image": (img["name"], BytesIO(blob), "image/png")}
        r = requests.post(f"http://{COMFY_HOST}/upload/image", files=files, timeout=30)
        r.raise_for_status()


def _download_and_link(model: dict) -> None:
    """Download a model only if the symlink isn't already correct."""
    target_path = model["target"]
    cached_path = download_model_weights(model["url"])  # fal cache — fast if already cached
    ensure_dir(target_path)
    # Skip re-linking if symlink already points to the right place
    if os.path.islink(target_path) and os.readlink(target_path) == str(cached_path):
        debug_log(f"✅ Already linked: {target_path}")
        return
    if os.path.exists(target_path) or os.path.islink(target_path):
        os.unlink(target_path)
    os.symlink(cached_path, target_path)
    debug_log(f"✅ Linked: {cached_path} -> {target_path}")


def _wait_for_prompt(ws: websocket.WebSocket, timeout: int = WS_TIMEOUT) -> None:
    """Block until ComfyUI signals execution is done, with a hard timeout."""
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"ComfyUI generation timed out after {timeout}s")
        ws.settimeout(min(remaining, 30))
        try:
            out = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        if not isinstance(out, str) or not out.strip().startswith("{"):
            continue
        msg = json.loads(out)
        if msg.get("type") == "executing" and msg["data"]["node"] is None:
            break


def _submit_workflow(workflow: dict) -> str:
    """Submit a workflow to ComfyUI and return the prompt_id."""
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}", timeout=10)

    resp = requests.post(
        f"http://{COMFY_HOST}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    if resp.status_code != 200:
        ws.close()
        raise HTTPException(
            status_code=500,
            detail=f"ComfyUI rejected workflow: {resp.text}",
        )

    prompt_id = resp.json()["prompt_id"]
    _wait_for_prompt(ws)
    ws.close()
    return prompt_id


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
# Workflow helpers
# -------------------------------------------------
def _make_dummy_b64(size: int) -> str:
    dummy = PILImage.new("RGB", (size, size), (128, 128, 128))  # grey is more realistic than black
    buf = BytesIO()
    dummy.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _build_workflow(image_name: str, cfg: float, denoise: float, seed: int, upscale_by: float) -> dict:
    """Return a fully configured workflow dict ready to submit."""
    job = copy.deepcopy(WORKFLOW_JSON)
    workflow = job["input"]["workflow"]

    workflow["43"]["inputs"]["image"] = image_name

    upscaler = workflow["310:160"]["inputs"]
    upscaler["cfg"]        = cfg
    upscaler["denoise"]    = denoise
    upscaler["seed"]       = seed
    upscaler["upscale_by"] = upscale_by

    workflow["310:148"]["inputs"]["denoise"] = denoise

    return workflow


# -------------------------------------------------
# App
# -------------------------------------------------
class PortraitUpscaler(
    fal.App,
    keep_alive=200,
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

        # ── 2. Start ComfyUI early, parallel with model downloads ──────────
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

        # ── 5. Warmup: both resolutions in parallel, in background ─────────
        threading.Thread(target=self._run_warmup, daemon=True).start()
        debug_log("🔥 Warmup queued in background — setup complete.")

    def _run_warmup(self):
        """Run 512px and 2048px warmup jobs in parallel to pre-compile GPU kernels."""
        debug_log("🔥 Warmup starting (512px + 2048px in parallel)...")

        def _warmup_single(warmup_res: int):
            try:
                image_b64  = _make_dummy_b64(warmup_res)
                image_name = f"warmup_{warmup_res}_{uuid.uuid4().hex}.png"
                upload_images([{"name": image_name, "image": image_b64}])

                workflow = _build_workflow(
                    image_name = image_name,
                    cfg        = 1.0,
                    denoise    = 0.30,
                    seed       = random.randint(0, 2**32 - 1),
                    upscale_by = 2.0,
                )

                client_id = str(uuid.uuid4())
                ws = websocket.WebSocket()
                ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}", timeout=10)

                resp = requests.post(
                    f"http://{COMFY_HOST}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                    timeout=30,
                )
                if resp.status_code != 200:
                    debug_log(f"⚠️  Warmup {warmup_res}px rejected (non-fatal): {resp.text}")
                    ws.close()
                    return

                _wait_for_prompt(ws)
                ws.close()
                debug_log(f"✅ Warmup {warmup_res}px complete.")

            except Exception as e:
                debug_log(f"⚠️  Warmup {warmup_res}px failed (non-fatal): {e}")

        # Run both warmup sizes in parallel
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(_warmup_single, res) for res in (512, 2048)]
            for f in as_completed(futures):
                f.result()  # exceptions already caught inside, this just drains

        debug_log("🔥 Warmup finished.")

    @fal.endpoint("/")
    async def handler(self, input: SkinFixInput, response: Response) -> SkinFixOutput:
        try:
            # ── Fetch and upload input image ───────────────────────────────
            image_b64 = image_url_to_base64(input.image_url)
            pil_img   = PILImage.open(BytesIO(base64.b64decode(image_b64)))
            input_image_resolution = max(pil_img.size)

            image_name = f"input_{uuid.uuid4().hex}.png"
            upload_images([{"name": image_name, "image": image_b64}])

            # ── Build workflow ─────────────────────────────────────────────
            denoise_strength  = 0.30 + (input.skin_refinement / 100.0) * 0.10
            seed              = input.seed if input.seed != -1 else random.randint(0, 2**32 - 1)
            target_resolution = max(input.upscale_resolution, input_image_resolution)
            upscale_by        = round(target_resolution / input_image_resolution, 2)

            workflow = _build_workflow(
                image_name = image_name,
                cfg        = input.cfg,
                denoise    = denoise_strength,
                seed       = seed,
                upscale_by = upscale_by,
            )

            # ── Submit and wait ────────────────────────────────────────────
            prompt_id = _submit_workflow(workflow)

            # ── Collect output images ──────────────────────────────────────
            history = requests.get(
                f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30
            ).json()

            images = []
            for node in history[prompt_id]["outputs"].values():
                for img in node.get("images", []):
                    params = (
                        f"filename={img['filename']}"
                        f"&subfolder={img.get('subfolder', '')}"
                        f"&type={img['type']}"
                    )
                    r = requests.get(f"http://{COMFY_HOST}/view?{params}", timeout=60)
                    pil_image    = PILImage.open(BytesIO(r.content))
                    output_image = Image.from_pil(pil_image, format="png")
                    images.append(output_image)

            response.headers["x-fal-billable-units"] = str(len(images))
            return SkinFixOutput(images=images)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))