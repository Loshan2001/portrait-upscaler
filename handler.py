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

COMFY_HOST  = "127.0.0.1:8188"
DEBUG_LOGS  = os.environ.get("FAL_DEBUG") == "1"
WS_TIMEOUT  = 600


def debug_log(message: str) -> None:
    if DEBUG_LOGS:
        print(message)


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def check_server(url, retries=300, delay=0.5):
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
    target_path = model["target"]
    cached_path = download_model_weights(model["url"])
    ensure_dir(target_path)
    if os.path.islink(target_path) and os.readlink(target_path) == str(cached_path):
        debug_log(f"✅ Already linked: {target_path}")
        return
    if os.path.exists(target_path) or os.path.islink(target_path):
        os.unlink(target_path)
    os.symlink(cached_path, target_path)
    debug_log(f"✅ Linked: {cached_path} -> {target_path}")


def _wait_for_prompt(ws: websocket.WebSocket, timeout: int = WS_TIMEOUT) -> None:
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
    upscale_by: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        title="Upscale Factor",
        description="How much to upscale the image. 2.0 = double the resolution.",
    )
    denoise: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        title="Denoise Strength",
        description="How much to alter the image during upscaling. Lower = closer to original.",
    )
    steps: int = Field(
        default=10,
        ge=1,
        le=30,
        title="Steps",
        description="Number of diffusion steps. More steps = better quality but slower.",
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
    dummy = PILImage.new("RGB", (size, size), (128, 128, 128))
    buf = BytesIO()
    dummy.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _build_workflow(
    image_name: str,
    upscale_by: float,
    denoise: float,
    steps: int,
    seed: int,
) -> dict:
    job      = copy.deepcopy(WORKFLOW_JSON)
    workflow = job["input"]["workflow"]

    # Input image
    workflow["43"]["inputs"]["image"] = image_name

    # Main upscaler node
    upscaler = workflow["310:160"]["inputs"]
    upscaler["upscale_by"] = upscale_by
    upscaler["denoise"]    = denoise
    upscaler["steps"]      = steps
    upscaler["seed"]       = seed
    upscaler["cfg"]        = 1.0  # hardcoded — not exposed to user

    # Scheduler node — keep denoise and steps in sync
    scheduler = workflow["310:148"]["inputs"]
    scheduler["denoise"] = denoise
    scheduler["steps"]   = steps

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
    image        = custom_image
    machine_type = "GPU-H100"
    requirements = ["websockets", "websocket-client"]
    private_logs = True

    def setup(self):
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
            ).strip()
            debug_log(f"🖥️  GPU: {gpu_info}")
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
            futures = {executor.submit(_download_and_link, m): m for m in MODEL_LIST}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    debug_log(f"❌ Model download failed: {e}")
                    raise

        debug_log("✅ All models ready.")

        debug_log("⏳ Waiting for ComfyUI...")
        if not check_server(f"http://{COMFY_HOST}/system_stats"):
            raise RuntimeError("ComfyUI failed to start within the timeout window.")
        debug_log("✅ ComfyUI is ready.")

        threading.Thread(target=self._run_warmup, daemon=True).start()
        debug_log("🔥 Warmup queued — setup complete.")

    def _run_warmup(self):
        debug_log("🔥 Warmup starting (2048px)...")

        # Line ~20
WS_TIMEOUT = 600  # increase from 300 to 600

# In _run_warmup — replace the ThreadPoolExecutor block with:
def _run_warmup(self):
    debug_log("🔥 Warmup starting (2048px)...")
    try:
        image_name = f"warmup_2048_{uuid.uuid4().hex}.png"
        upload_images([{"name": image_name, "image": _make_dummy_b64(2048)}])

        workflow = _build_workflow(
            image_name = image_name,
            upscale_by = 2.0,
            denoise    = 0.2,
            steps      = 10,
            seed       = random.randint(0, 2**32 - 1),
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
            debug_log(f"⚠️  Warmup rejected (non-fatal): {resp.text}")
            ws.close()
            return

        _wait_for_prompt(ws, timeout=600)
        ws.close()
        debug_log("✅ Warmup complete.")

    except Exception as e:
        debug_log(f"⚠️  Warmup failed (non-fatal): {e}")

    @fal.endpoint("/")
    async def handler(self, input: SkinFixInput, response: Response) -> SkinFixOutput:
        try:
            # Fetch and upload input image
            image_b64 = image_url_to_base64(input.image_url)
            image_name = f"input_{uuid.uuid4().hex}.png"
            upload_images([{"name": image_name, "image": image_b64}])

            seed = input.seed if input.seed != -1 else random.randint(0, 2**32 - 1)

            workflow = _build_workflow(
                image_name = image_name,
                upscale_by = input.upscale_by,
                denoise    = input.denoise,
                steps      = input.steps,
                seed       = seed,
            )

            prompt_id = _submit_workflow(workflow)

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
                    output_image = Image.from_pil(PILImage.open(BytesIO(r.content)), format="png")
                    images.append(output_image)

            response.headers["x-fal-billable-units"] = str(len(images))
            return SkinFixOutput(images=images)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))