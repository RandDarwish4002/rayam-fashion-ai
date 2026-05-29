# ============================================================
# ai/services/model_loader.py
# تحميل النماذج مرة واحدة (Singleton Pattern)
# Florence-2 + CLIP
# ============================================================

import torch
import clip
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM
)

# تحديد الجهاز
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# precision حسب الجهاز
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


class ModelLoader:
    """
    Singleton Loader

    يحمّل:
    - Florence-2
    - CLIP

    مرة واحدة فقط ويحفظهم في الذاكرة
    """

    _florence_proc = None
    _florence_model = None

    _clip_model = None
    _clip_prep = None

    # cache لـ text features
    _clip_text_cache = {}

    # ========================================================
    # device
    # ========================================================
    @classmethod
    def get_device(cls) -> str:
        return DEVICE

    # ========================================================
    # Florence-2
    # ========================================================
    @classmethod
    def florence(cls):
        """
        يرجع:
        processor, model
        """

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
        """
        يرجع:
        model, preprocess
        """

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
    # CLIP text feature cache
    # ========================================================
    @classmethod
    def get_clip_text_features(cls, key: str, labels: list):
        """
        يحسب text embeddings مرة واحدة فقط
        """

        if key in cls._clip_text_cache:
            return cls._clip_text_cache[key]

        clip_model, _ = cls.clip()

        with torch.no_grad():
            tokens = clip.tokenize(labels).to(DEVICE)

            text_features = clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True
            )

        cls._clip_text_cache[key] = text_features
        return text_features

    # ========================================================
    # unload
    # ========================================================
    @classmethod
    def unload_all(cls):
        """
        تحرير الذاكرة
        """

        cls._florence_proc = None
        cls._florence_model = None

        cls._clip_model = None
        cls._clip_prep = None

        cls._clip_text_cache = {}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✓ تم تحرير النماذج من الذاكرة")
