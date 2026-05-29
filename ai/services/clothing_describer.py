
# ============================================================
# ai/services/clothing_describer.py
# AIClassifier.describeClothing(image): String
# Florence-2 detailed captioning
# ============================================================

import torch
from PIL import Image

from ai.services.model_loader import (
    ModelLoader,
    DEVICE
)


class ClothingDescriber:
    """
    Florence-2 وصف تفصيلي للقطعة
    """

    def __init__(self):
        self.task = "<DETAILED_CAPTION>"

    # ========================================================
    # describe
    # ========================================================
    def describe(
            self,
            image: Image.Image
    ) -> str:
        """
        المدخل:
            صورة PIL

        المخرج:
            وصف نصي كامل
        """

        proc, model = ModelLoader.florence()

        # Florence يحتاج RGB
        image_rgb = image.convert("RGB")

        # ====================================================
        # preprocessing
        # ====================================================

        inputs = proc(
            text=self.task,
            images=image_rgb,
            return_tensors="pt"
        )

        # FIX:
        # تحويل الـ tensors إلى float16
        # ليتوافق مع Florence المحمّل بـ FP16
        inputs = {
            k: v.to(
                DEVICE,
                dtype=torch.float16
            ) if v.dtype == torch.float32
            else v.to(DEVICE)
            for k, v in inputs.items()
        }

        # ====================================================
        # generation
        # ====================================================

        with torch.no_grad():

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],

                max_new_tokens=300,
                num_beams=3,
                early_stopping=True,

                do_sample=False
            )

        # ====================================================
        # decode
        # ====================================================

        generated_text = proc.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # ====================================================
        # Florence post-process
        # ====================================================

        parsed_answer = proc.post_process_generation(
            generated_text,
            task=self.task,
            image_size=image_rgb.size
        )

        # استخراج النص الحقيقي
        description = parsed_answer.get(
            self.task,
            generated_text
        )

        # تنظيف
        description = self._clean_text(description)

        print(f"  ✓ الوصف: {description[:100]}...")

        return description

    # ========================================================
    # clean output
    # ========================================================

    def _clean_text(
            self,
            text: str
    ) -> str:
        """
        تنظيف مخرجات Florence
        """

        if not text:
            return "No description"

        text = text.strip()

        # إزالة task token لو ظهر
        if text.startswith("<DETAILED_CAPTION>"):

            text = text.replace(
                "<DETAILED_CAPTION>",
                ""
            ).strip()

        # إزالة double spaces
        while "  " in text:
            text = text.replace("  ", " ")

        return text
