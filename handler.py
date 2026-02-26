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
import httpx
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
from io import BytesIO
from PIL import Image as PILImage
from pydantic import BaseModel, Field
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
    for i in range(retries):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(min(delay * (1.05 ** i), 1.0))
    return False

async def image_url_to_base64_async(image_url: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(image_url)
        r.raise_for_status()
        pil = PILImage.open(BytesIO(r.content))
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
    if os.path.exists(target_path) and not os.path.islink(target_path):
        debug_log(f"Already in image: {target_path}")
        return
    cached_path = download_model_weights(model["url"])
    ensure_dir(target_path)
    if os.path.islink(target_path) and os.readlink(target_path) == str(cached_path):
        debug_log(f"Already linked: {target_path}")
        return
    if os.path.exists(target_path) or os.path.islink(target_path):
        os.unlink(target_path)
    os.symlink(cached_path, target_path)
    debug_log(f"Linked: {cached_path} -> {target_path}")

def _wait_for_prompt(ws: websocket.WebSocket, timeout: int = WS_TIMEOUT) -> None:
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"ComfyUI generation timed out after {timeout}s")
        ws.settimeout(min(remaining, 5))
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
        raise HTTPException(status_code=500, detail=f"ComfyUI rejected workflow: {resp.text}")
    prompt_id = resp.json()["prompt_id"]
    _wait_for_prompt(ws)
    ws.close()
    return prompt_id

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
# Input / Output Models
# -------------------------------------------------
class SkinFixInput(BaseModel):
    image_url: str = Field(..., title="Input Image", description="URL of the image to enhance and upscale.")
    upscale_by: float = Field(default=2.0, ge=1.0, le=4.0, title="Upscale Factor")
    denoise: float = Field(default=0.25, ge=0.0, le=1.0, title="Denoise Strength")
    steps: int = Field(default=8, ge=1, le=30, title="Steps")
    seed: int = Field(default=123456789, title="Seed")

class SkinFixOutput(BaseModel):
    images: list[Image] = Field(description="Output images from skin fix processing")

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
    requirements = ["websockets", "websocket-client", "httpx"]
    private_logs = True

    def setup(self):
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
            ).strip()
            debug_log(f"GPU: {gpu_info}")
        except Exception as e:
            debug_log(f"Could not detect GPU: {e}")

        self._warmup_done = threading.Event()
        self._warmup_failed = False

        debug_log("Starting ComfyUI...")
        self.comfy = subprocess.Popen(
            [
                "python", "-u", "/comfyui/main.py",
                "--disable-auto-launch",
                "--disable-metadata",
                "--listen",
                "--port", "8188",
                "--use-sage-attention",
                "--fast",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Model verification — instant since models are baked into image
        debug_log(f"Verifying {len(MODEL_LIST)} models...")
        for m in MODEL_LIST:
            _download_and_link(m)
        debug_log("All models verified.")

        # Wait for ComfyUI to accept HTTP
        debug_log("Waiting for ComfyUI...")
        if not check_server(f"http://{COMFY_HOST}/system_stats"):
            raise RuntimeError("ComfyUI failed to start within the timeout window.")
        debug_log("ComfyUI is ready.")

        # Warmup in background — container becomes READY immediately after setup() returns.
        # Handler waits on _warmup_done only if a request races ahead of warmup (rare).
        threading.Thread(target=self._run_warmup, daemon=True).start()
        debug_log("Warmup queued in background — setup() returning, container is READY.")

    def _run_warmup(self):
        """
        512px dummy pass through the full workflow.
        - Loads all models (UNET, VAE, CLIP, LoRA, upscaler) into VRAM
        - Warms CUDA kernels so first real request is fast
        - Runs in background so setup() isn't blocked
        """
        debug_log("Warmup starting (512px, 6 steps)...")
        try:
            image_name = f"warmup_{uuid.uuid4().hex}.png"
            upload_images([{"name": image_name, "image": _make_dummy_b64(512)}])
            workflow = _build_workflow(
                image_name=image_name,
                upscale_by=2.0,
                denoise=0.25,
                steps=6,
                seed=42,
            )
            _submit_workflow(workflow)
            debug_log("Warmup complete — all models in VRAM, CUDA warm.")
        except Exception as e:
            debug_log(f"Warmup failed (non-fatal): {e}")
            self._warmup_failed = True
        finally:
            self._warmup_done.set()

    @fal.endpoint("/")
    async def handler(self, input: SkinFixInput, response: Response) -> SkinFixOutput:
        try:
            # Safety net: if request arrives before background warmup finishes, wait for it.
            # In practice warmup completes ~15-20s after setup(), well before real traffic.
            if not self._warmup_done.is_set():
                debug_log("Request arrived before warmup — waiting up to 60s...")
                self._warmup_done.wait(timeout=60)

            image_b64 = await image_url_to_base64_async(input.image_url)
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