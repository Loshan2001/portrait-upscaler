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

PWD = Path(__file__).resolve().parent if __file__ else Path(os.getcwd())
dockerfile_path = f"{PWD}/Dockerfile"
custom_image = ContainerImage.from_dockerfile(dockerfile_path)

COMFY_HOST = "127.0.0.1:8188"
DEBUG_LOGS = os.environ.get("FAL_DEBUG") == "1"
WS_TIMEOUT = 600

def debug_log(message: str) -> None:
    if DEBUG_LOGS:
        print(message)

# ------------------------------------------------- 
# Utilities
# -------------------------------------------------
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def check_server(url, retries=120, delay=0.25):
    """
    FIX #3: Faster polling — 0.25s delay, 120 retries = 30s max wait.
    ComfyUI on H100 starts in ~8-15s, so this is plenty.
    """
    for i in range(retries):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        # Exponential backoff up to 1s to avoid hammering
        time.sleep(min(delay * (1.05 ** i), 1.0))
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
    # FIX #1: Check if target already exists (baked into image) before downloading
    if os.path.exists(target_path) and not os.path.islink(target_path):
        debug_log(f"✅ Already in image: {target_path}")
        return
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
    image_url: str = Field(..., title="Input Image", description="URL of the image to enhance and upscale.")
    upscale_by: float = Field(default=2.0, ge=1.0, le=4.0, title="Upscale Factor")
    denoise: float = Field(default=0.35, ge=0.0, le=1.0, title="Denoise Strength")
    steps: int = Field(default=10, ge=1, le=30, title="Steps")
    seed: int = Field(default=123456789, title="Seed")

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

def _build_workflow(image_name, upscale_by, denoise, steps, seed) -> dict:
    job = copy.deepcopy(WORKFLOW_JSON)
    workflow = job["input"]["workflow"]
    workflow["43"]["inputs"]["image"] = image_name
    upscaler = workflow["310:160"]["inputs"]
    upscaler["upscale_by"] = upscale_by
    upscaler["denoise"] = denoise
    upscaler["steps"] = steps
    upscaler["seed"] = seed
    upscaler["cfg"] = 1.0
    scheduler = workflow["310:148"]["inputs"]
    scheduler["denoise"] = denoise
    scheduler["steps"] = steps
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
    image = custom_image
    machine_type = "GPU-H100"
    requirements = ["websockets", "websocket-client"]
    private_logs = True

    def setup(self):
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
            ).strip()
            debug_log(f"🖥️ GPU: {gpu_info}")
        except Exception as e:
            debug_log(f"⚠️ Could not detect GPU: {e}")

        # FIX #1: Download models and start ComfyUI in PARALLEL
        # No reason to wait for models before starting ComfyUI —
        # ComfyUI loads models lazily when a workflow runs, not at startup.
        debug_log("🚀 Starting ComfyUI + downloading models in parallel...")

        self.comfy = subprocess.Popen(
            [
                "python", "-u", "/comfyui/main.py",
                "--disable-auto-launch",
                "--disable-metadata",
                "--listen",
                "--port", "8188",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Download models concurrently while ComfyUI boots
        def _download_all():
            debug_log(f"⬇️ Downloading {len(MODEL_LIST)} models in parallel...")
            with ThreadPoolExecutor(max_workers=min(16, len(MODEL_LIST))) as executor:
                futures = {executor.submit(_download_and_link, m): m for m in MODEL_LIST}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        debug_log(f"❌ Model download failed: {e}")
                        raise
            debug_log("✅ All models ready.")

        model_thread = threading.Thread(target=_download_all, daemon=False)
        model_thread.start()

        # Wait for ComfyUI to be ready
        debug_log("⏳ Waiting for ComfyUI...")
        if not check_server(f"http://{COMFY_HOST}/system_stats"):
            raise RuntimeError("ComfyUI failed to start within the timeout window.")
        debug_log("✅ ComfyUI is ready.")

        # Wait for models to finish before accepting requests
        model_thread.join()

        # FIX #2: Single small warmup (256px) — just enough to warm CUDA kernels
        # Skip the 2048px warmup — it wastes 10-20s and the first real request
        # will be a better warmup anyway.
        threading.Thread(target=self._run_warmup, daemon=True).start()
        debug_log("🔥 Warmup queued — setup complete.")

    def _run_warmup(self):
        """
        FIX #2: One tiny warmup pass at 256px to load CUDA kernels.
        Removed the large 2048px warmup — not worth the cold start cost.
        """
        debug_log("🔥 Warmup starting (256px)...")
        try:
            image_name = f"warmup_{uuid.uuid4().hex}.png"
            upload_images([{"name": image_name, "image": _make_dummy_b64(256)}])
            workflow = _build_workflow(
                image_name=image_name,
                upscale_by=2.0,
                denoise=0.35,
                steps=4,   # Fewer steps = faster warmup
                seed=42,
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
                debug_log(f"⚠️ Warmup rejected (non-fatal): {resp.text}")
                ws.close()
                return
            _wait_for_prompt(ws)
            ws.close()
            debug_log("✅ Warmup complete.")
        except Exception as e:
            debug_log(f"⚠️ Warmup failed (non-fatal): {e}")

    @fal.endpoint("/")
    async def handler(self, input: SkinFixInput, response: Response) -> SkinFixOutput:
        try:
            image_b64 = image_url_to_base64(input.image_url)
            image_name = f"input_{uuid.uuid4().hex}.png"
            upload_images([{"name": image_name, "image": image_b64}])

            seed = input.seed if input.seed != -1 else random.randint(0, 2**32 - 1)
            workflow = _build_workflow(
                image_name=image_name,
                upscale_by=input.upscale_by,
                denoise=input.denoise,
                steps=input.steps,
                seed=seed,
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