import json

_WORKFLOW = {
  "input": {
    "uid": "testUid",
    "customNodes": [],
    "customModels": [],
    "images": [
      {
        "name": "de5c01cc.png",
        "image": "base64.."
      }
    ],
    "workflow": {
      "545": {
        "inputs": {
          "image": "input_placeholder.png"
        },
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"}
      },
      "147": {
        "inputs": {
          "vae_name": "ae.safetensors"
        },
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"}
      },
      "149": {
        "inputs": {
          "unet_name": "ultrarealFineTune_v4.safetensors",
          "weight_dtype": "fp8_e4m3fn"
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Diffusion Model"}
      },
      "150": {
        "inputs": {
          "clip_name1": "clip_l.safetensors",
          "clip_name2": "new/t5xxl_fp8_e4m3fn.safetensors",
          "type": "flux",
          "device": "default"
        },
        "class_type": "DualCLIPLoader",
        "_meta": {"title": "DualCLIPLoader"}
      },
      "154": {
        "inputs": {
          "model_name": "4x-ClearRealityV1.pth"
        },
        "class_type": "UpscaleModelLoader",
        "_meta": {"title": "Load Upscale Model"}
      },
      "187": {
        "inputs": {
          "lora_name": "aidmaRealisticSkin-FLUX-v0.1.safetensors",
          "strength_model": 0.8,
          "model": ["149", 0]
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "Load LoRA"}
      },
      # 506 = positive prompt node (handler.py writes to ["inputs"]["part1"])
      "506": {
        "inputs": {
          "part1": "aidmarealisticskin, beautiful realistic human face, authentic skin texture, natural freckles, soft pore details, refined smoothness without blur, subtle tone variation, natural color gradients, soft micro-shadows, subsurface scattering, hydrated skin texture, matte-natural finish, healthy glow, realistic surface roughness, true skin depth, ultra-detailed pore structure, clean complexion, photoreal clarity, 8k ultra photorealistic portrait, cinematic lighting, masterpiece",
          "clip": ["150", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive)"}
      },
      # 507 = negative prompt node (handler.py writes to ["inputs"]["text"])
      "507": {
        "inputs": {
          "text": "score_1, score_2, bad skin, ugly skin, unrealistic, innacurate, bad, ugly, worst quality",
          "clip": ["150", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative)"}
      },
      "310:155": {
        "inputs": {
          "sampler_name": "res_2s"
        },
        "class_type": "KSamplerSelect",
        "_meta": {"title": "KSamplerSelect"}
      },
      "310:156": {
        "inputs": {
          "conditioning": ["507", 0]
        },
        "class_type": "ConditioningZeroOut",
        "_meta": {"title": "ConditioningZeroOut"}
      },
      "310:148": {
        "inputs": {
          "scheduler": "beta",
          "steps": 8,
          "denoise": 0.2,
          "model": ["187", 0]
        },
        "class_type": "BasicScheduler",
        "_meta": {"title": "BasicScheduler"}
      },
      "310:153": {
        "inputs": {
          "max_shift": 1.15,
          "base_shift": 0.5,
          "width": 1024,
          "height": 1024,
          "model": ["187", 0]
        },
        "class_type": "ModelSamplingFlux",
        "_meta": {"title": "ModelSamplingFlux"}
      },
      "310:158": {
        "inputs": {
          "a": 6.283185307179586,
          "bg_threshold": 0.1,
          "resolution": 1024,
          "image": ["545", 0]
        },
        "class_type": "MiDaS-DepthMapPreprocessor",
        "_meta": {"title": "MiDaS Depth Map"}
      },
      "310:159": {
        "inputs": {
          "positive": ["506", 0],
          "negative": ["310:156", 0],
          "vae": ["147", 0],
          "pixels": ["310:158", 0]
        },
        "class_type": "InstructPixToPixConditioning",
        "_meta": {"title": "InstructPixToPixConditioning"}
      },
      "310:157": {
        "inputs": {
          "detail_amount": 0.2,
          "start": 0.2,
          "end": 0.8,
          "bias": 0.5,
          "exponent": 0.25,
          "start_offset": 0,
          "end_offset": 0,
          "fade": 0,
          "smooth": True,
          "cfg_scale_override": 0,
          "sampler": ["310:155", 0]
        },
        "class_type": "DetailDaemonSamplerNode",
        "_meta": {"title": "Detail Daemon Sampler"}
      },
      # 510 = main sampler (handler.py writes cfg, denoise, seed here)
      "510": {
        "inputs": {
          "upscale_by": 2,
          "seed": 166074999420575,
          "steps": 10,
          "cfg": 1.0,
          "sampler_name": "dpmpp_2m",
          "scheduler": "beta",
          "denoise": 0.30,
          "mode_type": "Linear",
          "tile_width": 1024,
          "tile_height": 1024,
          "mask_blur": 8,
          "tile_padding": 32,
          "seam_fix_mode": "None",
          "seam_fix_denoise": 1,
          "seam_fix_width": 64,
          "seam_fix_mask_blur": 8,
          "seam_fix_padding": 16,
          "force_uniform_tiles": True,
          "tiled_decode": False,
          "batch_size": 1,
          "image": ["545", 0],
          "model": ["310:153", 0],
          "positive": ["310:159", 0],
          "negative": ["310:159", 1],
          "vae": ["147", 0],
          "upscale_model": ["154", 0],
          "custom_sampler": ["310:157", 0],
          "custom_sigmas": ["310:148", 0]
        },
        "class_type": "UltimateSDUpscaleCustomSample",
        "_meta": {"title": "Ultimate SD Upscale (Custom Sample)"}
      },
      # 548 = resolution node (handler.py sets resolution, max_resolution, seed)
      "548": {
        "inputs": {
          "resolution": 2048,
          "max_resolution": 4096,
          "seed": 0,
          "image": ["545", 0]
        },
        "class_type": "ImageScaleToResolution",
        "_meta": {"title": "Scale To Resolution"}
      },
      # 549 = VAE encode node (handler.py sets encode_tile_size, decode_tile_size)
      "549": {
        "inputs": {
          "encode_tile_size": 1024,
          "decode_tile_size": 1024,
          "vae": ["147", 0],
          "pixels": ["510", 0]
        },
        "class_type": "VAEEncodeTiled",
        "_meta": {"title": "VAE Encode (Tiled)"}
      },
      "318": {
        "inputs": {
          "filename_prefix": "Final_",
          "images": ["510", 0]
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"}
      },
      "310:189": {
        "inputs": {
          "image": ["545", 0]
        },
        "class_type": "Get Image Size",
        "_meta": {"title": "Get Image Size"}
      },
      "310:191": {
        "inputs": {
          "width": ["310:189", 0],
          "height": ["310:189", 1],
          "upscale_method": "lanczos",
          "keep_proportion": "resize",
          "pad_color": "0, 0, 0",
          "crop_position": "center",
          "divisible_by": 2,
          "device": "cpu",
          "image": ["510", 0]
        },
        "class_type": "ImageResizeKJv2",
        "_meta": {"title": "Resize Image v2"}
      }
    }
  }
}

# Store as JSON string — avoids 'unhashable type: dict' during fal deploy
WORKFLOW_JSON = json.loads(json.dumps(_WORKFLOW))