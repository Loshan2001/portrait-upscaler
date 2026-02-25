FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base
ARG COMFYUI_VERSION=latest
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1

# ---------------------------------------------------------
# System + Python 3.12 (via deadsnakes PPA)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        git git-lfs wget curl ffmpeg libgl1 libglib2.0-0 \
        python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install pip via get-pip (python3-pip on 22.04 targets 3.10, not 3.12)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# PyTorch CUDA 12.8
# ---------------------------------------------------------
RUN pip install torch==2.7.0 -f https://download.pytorch.org/whl/cu128/torch_stable.html

# ---------------------------------------------------------
# Install ComfyUI
# ---------------------------------------------------------
WORKDIR /opt
RUN pip install comfy-cli
RUN yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia
WORKDIR /comfyui

# ---------------------------------------------------------
# Core Dependencies (FLUX compatible)
# ---------------------------------------------------------
RUN pip install \
    accelerate \
    transformers \
    sageattention \
    onnxruntime-gpu==1.18.0 \
    websocket-client \
    websockets \
    requests

# ---------------------------------------------------------
# REQUIRED Custom Nodes for Your Workflow ONLY
# ---------------------------------------------------------
# Essentials
RUN comfy --workspace /comfyui node install comfyui_essentials@1.1.0

# KJNodes (used in workflow)
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes \
    && cd /comfyui/custom_nodes/ComfyUI-KJNodes \
    && pip install -r requirements.txt

# rgthree (LoRA Loader support)
RUN git clone https://github.com/rgthree/rgthree-comfy.git /comfyui/custom_nodes/rgthree-comfy \
    && cd /comfyui/custom_nodes/rgthree-comfy \
    && pip install -r requirements.txt

# Ultimate SD Upscale (provides UltimateSDUpscaleCustomSample)
RUN git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale \
    && cd /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale \
    && pip install -r requirements.txt || true

# Detail Daemon (used in workflow for DetailDaemonSamplerNode)
RUN git clone https://github.com/Jonseed/ComfyUI-Detail-Daemon.git /comfyui/custom_nodes/ComfyUI-Detail-Daemon \
    && cd /comfyui/custom_nodes/ComfyUI-Detail-Daemon \
    && pip install -r requirements.txt || true

# MiDaS depth preprocessor (used in workflow for MiDaS-DepthMapPreprocessor)
RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git /comfyui/custom_nodes/comfyui_controlnet_aux \
    && cd /comfyui/custom_nodes/comfyui_controlnet_aux \
    && pip install -r requirements.txt

# ---------------------------------------------------------
# fal Runtime Requirements
# ---------------------------------------------------------
RUN pip install \
    boto3==1.35.74 \
    protobuf==4.25.1 \
    pydantic==2.10.6

# ---------------------------------------------------------
# Model cache location (fal requirement)
# ---------------------------------------------------------
ENV HF_HOME=/fal-volume/models/huggingface
WORKDIR /comfyui
EXPOSE 8188