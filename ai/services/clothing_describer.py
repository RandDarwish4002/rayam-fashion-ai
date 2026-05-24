# ============================================================
# ai/services/clothing_describer.py
# يطابق: AIClassifier.describeClothing(image): String
# ============================================================

import torch
from PIL import Image
from ai.services.model_loader import ModelLoader, DEVICE


class ClothingDescriber:
    """
    يطابق Class Diagram:
    AIClassifier.describeClothing(image): String
    """

    def describe(
            self,
            image: Image.Image
    ) -> str:
        """
        المدخل:  صورة PIL (RGBA أو RGB)
        المخرج:  وصف نصي كامل للقطعة
        """
        proc, model = ModelLoader.florence()
        image_rgb   = image.convert("RGB")

        inputs = proc(
            text           = "<DETAILED_CAPTION>",
            images         = image_rgb,
            return_tensors = "pt"
        ).to(DEVICE)

        with torch.no_grad():
            ids = model.generate(
                input_ids      = inputs["input_ids"],
                pixel_values   = inputs["pixel_values"],
                max_new_tokens = 300,
                num_beams      = 3,
                early_stopping = True
            )

        desc = proc.batch_decode(
            ids, skip_special_tokens=True
        )[0]

        print(f"  ✓ الوصف: {desc[:80]}...")
        return desc
