
# ============================================================
# ai/services/wardrobe_pipeline.py
# AI Fashion Pipeline
# ============================================================

import os
import json
from typing import Dict, List, Optional

from ai.services.background_remover import BackgroundRemover
from ai.services.clothing_describer import ClothingDescriber
from ai.services.attribute_classifier import AttributeClassifier
from ai.services.color_extractor import ColorExtractor
from ai.services.smart_decision_layer import SmartDecisionLayer


class WardrobePipeline:

    def __init__(self):
        self.bg_remover  = BackgroundRemover()
        self.describer   = ClothingDescriber()
        self.attr_clf    = AttributeClassifier()
        self.color_ext   = ColorExtractor()
        self.smart_layer = SmartDecisionLayer()

    def analyze_item(self, image_path: str) -> Dict:

        print(f"\n{'─'*50}")
        print(f"تحليل: {os.path.basename(image_path)}")
        print(f"{'─'*50}")

        try:
            print("① إزالة الخلفية...")
            image = self.bg_remover.remove(image_path)

            print("② وصف القطعة...")
            description = self.describer.describe(image)

            print("③ تصنيف الـ attributes...")
            raw_attributes = self.attr_clf.classify(image)

            print("③.1 دمج وتحسين النتائج...")
            attributes = self.smart_layer.resolve(raw_attributes, description)

            print("④ استخراج الألوان...")
            colors = self.color_ext.extract(image)

            print("⑤ تصنيف المجموعات اللونية...")
            colors = self.color_ext.classify_groups(colors)

            result = {
                "status": "success",
                "image_path": image_path,
                "description": description,

                "attributes": attributes,

                "confidence": {
                    k: v["confidence"]
                    for k, v in raw_attributes.items()
                },

                "colors": colors,
                "primary_color": colors[0] if colors else None,
            }

            print("✓ اكتمل التحليل")
            return result

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def analyze_item_from_bytes(self, image_bytes: bytes, filename: str = "image.jpg") -> Dict:
        import tempfile

        suffix = os.path.splitext(filename)[1] or ".jpg"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            return self.analyze_item(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def analyze_wardrobe(self, image_paths: List[str], save_path: Optional[str] = None) -> List[Dict]:

        results = []
        success = 0
        failed = 0

        print(f"\nتحليل {len(image_paths)} قطعة...")

        for i, path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}]")
            r = self.analyze_item(path)
            results.append(r)

            if r["status"] == "success":
                success += 1
            else:
                failed += 1

        print(f"\n✓ نجح: {success} | ✗ فشل: {failed}")

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return results
