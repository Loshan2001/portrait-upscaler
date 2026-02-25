MODEL_LIST = [

    # =========================
    # FLUX UNET MODEL
    # =========================
    {
        "url": "https://huggingface.co/ferzikgetitdone/1488ULTRAREALFINETUNEV4/resolve/main/ultrarealFineTune_v4.safetensors",
        "path": "/data/models/unet/ultrarealFineTune_v4.safetensors",
        "target": "/comfyui/models/unet/ultrarealFineTune_v4.safetensors"
    },

    # =========================
    # VAE
    # =========================
    {
        "url": "https://huggingface.co/lovis93/testllm/resolve/ed9cf1af7465cebca4649157f118e331cf2a084f/ae.safetensors",
        "path": "/data/models/vae/ae.safetensors",
        "target": "/comfyui/models/vae/ae.safetensors"
    },

    # =========================
    # CLIP L
    # =========================
    {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "path": "/data/models/clip/clip_l.safetensors",
        "target": "/comfyui/models/clip/clip_l.safetensors"
    },

    # =========================
    # T5 XXL
    # =========================
    {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        "path": "/data/models/clip/t5xxl_fp8_e4m3fn.safetensors",
        "target": "/comfyui/models/clip/t5xxl_fp8_e4m3fn.safetensors"
    },

    # =========================
    # LORA
    # =========================
    {
        "url": "https://huggingface.co/Raspberry-ai/aidmaRealisticSkin-FLUX-v0.1/resolve/main/aidmaRealisticSkin-FLUX-v0.1.safetensors",
        "path": "/data/models/loras/aidmaRealisticSkin-FLUX-v0.1.safetensors",
        "target": "/comfyui/models/loras/aidmaRealisticSkin-FLUX-v0.1.safetensors"
    },

    # =========================
    # UPSCALER
    # =========================
    {
        "url": "https://huggingface.co/skbhadra/ClearRealityV1/resolve/main/4x-ClearRealityV1.pth",
        "path": "/data/models/upscale_models/4x-ClearRealityV1.pth",
        "target": "/comfyui/models/upscale_models/4x-ClearRealityV1.pth"
    }
]