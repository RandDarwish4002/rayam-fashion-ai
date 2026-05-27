# ============================================================
# ai/services/model_loader.py
# Singleton Model Loader (Florence-2 + CLIP)
# FIXED: FP16 safe + device handling
# ============================================================

import torch
import clip
from transformers import AutoProcessor, AutoModelForCausalLM

# -----------------------------
# Device
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Always safe default for T4
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


class ModelLoader:
    """
    Singleton loader for all models
    """

    _florence_proc = None
    _florence_model = None

    _clip_model = None
    _clip_prep = None

    _clip_text_cache = {}

    # ========================================================
    # Device getter
    # ========================================================
    @classmethod
    def get_device(cls):
        return DEVICE

    # ========================================================
    # Florence-2
    # ========================================================
    @classmethod
    def florence(cls):
        if cls._florence_model is None:
            print("تحميل Florence-2...")

            model_id = "microsoft/Florence-2-base"

            cls._florence_proc = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )

            cls._florence_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=DTYPE
            ).to(DEVICE)

            cls._florence_model.eval()

            print(f"✓ Florence-2 جاهز ({DEVICE}, {DTYPE})")

        return cls._florence_proc, cls._florence_model

    # ========================================================
    # CLIP
    # ========================================================
    @classmethod
    def clip(cls):
        if cls._clip_model is None:
            print("تحميل CLIP...")

            cls._clip_model, cls._clip_prep = clip.load(
                "ViT-B/32",
                device=DEVICE
            )

            cls._clip_model.eval()

            print(f"✓ CLIP جاهز ({DEVICE})")

        return cls._clip_model, cls._clip_prep

    # ========================================================
    # CLIP text cache
    # ========================================================
    @classmethod
    def get_clip_text_features(cls, key: str, labels: list):
        if key in cls._clip_text_cache:
            return cls._clip_text_cache[key]

        model, _ = cls.clip()

        with torch.no_grad():
            tokens = clip.tokenize(labels).to(DEVICE)

            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        cls._clip_text_cache[key] = features
        return features

    # ========================================================
    # cleanup
    # ========================================================
    @classmethod
    def unload_all(cls):
        cls._florence_proc = None
        cls._florence_model = None
        cls._clip_model = None
        cls._clip_prep = None
        cls._clip_text_cache = {}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✓ تم تحرير الذاكرة")
