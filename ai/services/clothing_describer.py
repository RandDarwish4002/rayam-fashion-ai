# ============================================================
# ai/services/clothing_describer.py
# FIXED: FP16 + Florence-2 safe inference
# ============================================================

import torch
from PIL import Image

from ai.services.model_loader import ModelLoader, DEVICE


class ClothingDescriber:
    """
    Florence-2 detailed captioning
    """

    def __init__(self):
        self.task = "<DETAILED_CAPTION>"

    # ========================================================
    # describe
    # ========================================================
    def describe(self, image: Image.Image) -> str:

        proc, model = ModelLoader.florence()

        # لازم RGB
        image_rgb = image.convert("RGB")

        # -----------------------------
        # FIX 1: no .to() on dict
        # -----------------------------
        inputs = proc(
            text=self.task,
            images=image_rgb,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        # -----------------------------
        # FIX 2: safe FP16 execution
        # -----------------------------
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],

                    max_new_tokens=300,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=False
                )

        # decode
        generated_text = proc.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # post-process
        parsed_answer = proc.post_process_generation(
            generated_text,
            task=self.task,
            image_size=image_rgb.size
        )

        description = parsed_answer.get(
            self.task,
            generated_text
        )

        description = self._clean_text(description)

        print(f"  ✓ الوصف: {description[:100]}...")

        return description

    # ========================================================
    # clean text
    # ========================================================
    def _clean_text(self, text: str) -> str:

        if not text:
            return "No description"

        text = text.strip()

        if text.startswith(self.task):
            text = text.replace(self.task, "").strip()

        while "  " in text:
            text = text.replace("  ", " ")

        return text
