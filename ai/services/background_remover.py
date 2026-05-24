# ============================================================
# ai/services/background_remover.py
# يطابق: AIClassifier.removeBackground(image)
# ============================================================

import io
import numpy as np
from PIL import Image
from rembg import remove


class BackgroundRemover:
    """
    يطابق Class Diagram:
    AIClassifier.removeBackground(image): Image
    """

    def remove(self, image_path: str) -> Image.Image:
        """
        المدخل:  مسار صورة
        المخرج:  صورة RGBA بدون خلفية
        """
        with open(image_path, "rb") as f:
            data = f.read()

        result = remove(data)
        img = Image.open(io.BytesIO(result)).convert("RGBA")

        # تحقق من الجودة
        arr = np.array(img)
        coverage = (arr[:, :, 3] > 128).mean()

        if coverage < 0.03:
            raise ValueError("الصورة غير واضحة — جرب صورة بإضاءة أفضل")

        if coverage > 0.97:
            print("  ⚠ الخلفية قد لم تُزَل بالكامل")

        print(f"  ✓ إزالة الخلفية ({coverage:.1%})")
        return img

    def remove_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """
        المدخل: bytes الصورة (للـ API)
        المخرج: صورة RGBA بدون خلفية
        """
        result = remove(image_bytes)
        img = Image.open(io.BytesIO(result)).convert("RGBA")

        arr = np.array(img)
        coverage = (arr[:, :, 3] > 128).mean()

        if coverage < 0.03:
            raise ValueError("الصورة غير واضحة")

        return img
