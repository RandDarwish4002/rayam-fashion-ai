
import os, json
from PIL import Image
from typing import Dict, List, Optional

from ai.services.background_remover      import BackgroundRemover
from ai.services.clothing_describer      import ClothingDescriber
from ai.services.attribute_classifier    import AttributeClassifier
from ai.services.color_extractor         import ColorExtractor
from ai.services.fashion_decision_engine import FashionDecisionEngine
from ai.services.fashion_nlp_extractor   import FashionNLPExtractor


class WardrobePipeline:

    def __init__(self, use_llm: bool = True):
        self.bg_remover   = BackgroundRemover()
        self.describer    = ClothingDescriber()
        self.nlp_extractor = FashionNLPExtractor(
            use_llm=use_llm
        )
        self.attr_clf     = AttributeClassifier()
        self.color_ext    = ColorExtractor()
        self.decision_eng = FashionDecisionEngine()

    def analyze_item(self, image_path: str) -> Dict:

        print(f"\n{'─'*50}")
        print(f"تحليل: {os.path.basename(image_path)}")
        print(f"{'─'*50}")

        try:
            # ① إزالة الخلفية
            print("① إزالة الخلفية...")
            image = self.bg_remover.remove(image_path)

            # ② Florence — وصف نصي
            print("② Florence — وصف نصي...")
            description = self.describer.describe(image)

            # ③ NLP — استخراج منظم من الوصف  ← الجديد
            print("③ NLP — استخراج attributes...")
            nlp_attrs = self.nlp_extractor.extract(
                description
            )

            # ④ CLIP — تصنيف مباشر من الصورة
            print("④ CLIP — تصنيف الصورة...")
            clip_raw = self.attr_clf.classify(image)

            # ⑤ Fashion Engine — دمج وقرار
            print("⑤ Fashion Engine — دمج وقرار...")
            decision = self.decision_eng.decide(
                florence_desc = description,
                clip_result   = clip_raw,
                nlp_attrs     = nlp_attrs,  # ← الجديد
            )

            # طباعة القرارات
            print("\n  ── قرارات الـ Engine:")
            for log in decision["decisions_log"]:
                print(f"    {log}")

            # ⑥ الألوان
            print("\n⑥ استخراج الألوان...")
            colors = self.color_ext.extract(image)
            colors = self.color_ext.classify_groups(colors)

            return {
                "status":       "success",
                "image_path":   image_path,
                "description":  description,
                "attributes":   decision["final_attributes"],
                "confidence":   decision["confidence"],
                "colors":       colors,
                "primary_color":colors[0] if colors else None,
                "engine_log":   decision["decisions_log"],
            }

        except FileNotFoundError:
            return {"status":"error",
                    "message":f"الصورة غير موجودة"}
        except Exception as e:
            return {"status":"error","message":str(e)}

    def analyze_item_from_bytes(
            self, image_bytes: bytes,
            filename: str = "image.jpg"
    ) -> Dict:
        import tempfile
        suffix = os.path.splitext(filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            result = self.analyze_item(tmp_path)
            result["image_path"] = filename
            return result
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def analyze_wardrobe(
            self,
            image_paths: List[str],
            save_path:   Optional[str] = None
    ) -> List[Dict]:
        results = []
        success = 0
        failed  = 0
        print(f"\nتحليل {len(image_paths)} قطعة...")
        for i, path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}]")
            r = self.analyze_item(path)
            results.append(r)
            if r["status"] == "success": success += 1
            else: failed += 1
        print(f"\n✓ نجح: {success} | ✗ فشل: {failed}")
        if save_path:
            with open(save_path,"w",encoding="utf-8") as f:
                json.dump(results,f,ensure_ascii=False,indent=2)
        return results
