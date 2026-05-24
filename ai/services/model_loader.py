# ============================================================
# ai/services/model_loader.py
# Singleton Model Loader (Florence + CLIP)
# ============================================================

import torch
import clip
from transformers import AutoProcessor, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


class ModelLoader:
    """
    يحمل النماذج مرة واحدة فقط (Singleton)
    """

    _florence_proc = None
    _florence_model = None
    _clip_model = None
    _clip_preprocess = None

    @classmethod
    def florence(cls):
        if cls._florence_proc is None:
            print("تحميل Florence-2...")

            model_id = "microsoft/Florence-2-base"

            cls._florence_proc = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )

            cls._florence_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation="eager"   # يمنع flash_attn مشاكل
            ).to(DEVICE)

            cls._florence_model.eval()

            print("✓ Florence-2 جاهز")

        return cls._florence_proc, cls._florence_model

    @classmethod
    def clip(cls):
        if cls._clip_model is None:
            print("تحميل CLIP...")

            cls._clip_model, cls._clip_preprocess = clip.load(
                "ViT-B/32",
                device=DEVICE
            )

            cls._clip_model.eval()

            print("✓ CLIP جاهز")

        return cls._clip_model, cls._clip_preprocess

    @classmethod
    def unload_all(cls):
        cls._florence_proc = None
        cls._florence_model = None
        cls._clip_model = None
        cls._clip_preprocess = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✓ تم تفريغ النماذج")
