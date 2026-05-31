
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

    def __init__(self, use_llm: bool = True, validate_clothing: bool = True):
        """
        Parameters:
        - use_llm: استخدام LLM لاستخراج attributes (أبطأ لكن أدق)
        - validate_clothing: التحقق من أن الصورة تحتوي على ملابس قبل التحليل
        """
        self.bg_remover   = BackgroundRemover()
        self.describer    = ClothingDescriber()
        self.nlp_extractor = FashionNLPExtractor(use_llm=use_llm)
        self.attr_clf     = AttributeClassifier()
        self.color_ext    = ColorExtractor()
        self.decision_eng = FashionDecisionEngine()
        self.validate_clothing = validate_clothing
        
        # قائمة الفئات المسموحة (ملابس وأكسسوارات)
        self.clothing_categories = [
            "dress", "evening gown", "t-shirt", "shirt", "blouse", 
            "pants", "jeans", "skirt", "jacket", "coat", "sweater", 
            "hoodie", "shorts", "shoes", "boots", "sneakers", 
            "sandals", "cardigan", "jumpsuit", "blazer", "suit",
            "turtleneck sweater", "knit sweater", "polo shirt"
        ]
        
        # عتبات الرفض
        self.MIN_COVERAGE = 0.05      # 5% كحد أدنى لتغطية الجسم
        self.MIN_CONFIDENCE = 8.0     # 8% كحد أدنى لثقة CLIP
        self.MAX_CONFIDENCE_NON_CLOTHING = 15.0  # إذا زادت الثقة عن هذا، نعتبره ملابس حتى لو الفئة غريبة

    def _is_clothing_item(self, clip_result: Dict, bg_coverage: float) -> tuple:
        """
        تتحقق إذا كانت الصورة تحتوي على قطعة ملابس حقيقية
        
        Returns:
        - (is_clothing, reason)
        """
        category = clip_result.get("category", {}).get("value", "").lower()
        confidence = clip_result.get("category", {}).get("confidence", 0)
        
        # 1. التحقق من تغطية إزالة الخلفية
        if bg_coverage < self.MIN_COVERAGE:
            return False, f"تغطية الخلفية منخفضة جداً ({bg_coverage:.1%}) - قد لا يوجد جسم واضح"
        
        # 2. التحقق من ثقة CLIP العالية جداً (قد تكون صورة معقدة)
        if confidence > 80:
            # ثقة عالية جداً - قد تكون صورة منوعة
            if not any(cat in category for cat in self.clothing_categories):
                return False, f"ثقة عالية ({confidence:.1f}%) لكن التصنيف '{category}' ليس ملابس"
        
        # 3. التحقق من الفئة
        is_clothing_category = any(cat in category for cat in self.clothing_categories)
        
        if not is_clothing_category and confidence < self.MAX_CONFIDENCE_NON_CLOTHING:
            return False, f"التصنيف '{category}' ليس قطعة ملابس (الثقة: {confidence:.1f}%)"
        
        # 4. إذا كانت الثقة منخفضة جداً
        if confidence < self.MIN_CONFIDENCE and not is_clothing_category:
            return False, f"ثقة منخفضة جداً ({confidence:.1f}%) للتصنيف '{category}'"
        
        return True, "صورة ملابس صالحة"

    def analyze_item(self, image_path: str) -> Dict:
        print(f"\n{'─'*50}")
        print(f"تحليل: {os.path.basename(image_path)}")
        print(f"{'─'*50}")

        try:
            # ① إزالة الخلفية
            print("① إزالة الخلفية...")
            image = self.bg_remover.remove(image_path)
            
            # حساب تغطية الجسم من الـ alpha mask
            import numpy as np
            arr = np.array(image)
            alpha = arr[:, :, 3] if arr.shape[2] == 4 else np.ones(arr.shape[:2]) * 255
            bg_coverage = (alpha > 128).mean()
            
            # ② CLIP سريع للتحقق (قبل Florence عشان نوفر وقت)
            print("② التحقق من الصورة (هل تحتوي على ملابس؟)...")
            clip_raw_fast = self.attr_clf.classify(image)
            
            if self.validate_clothing:
                is_clothing, reason = self._is_clothing_item(clip_raw_fast, bg_coverage)
                if not is_clothing:
                    print(f"  ⚠️ الصورة لا تحتوي على ملابس: {reason}")
                    return {
                        "status": "rejected",
                        "image_path": image_path,
                        "message": reason,
                        "is_clothing": False,
                        "bg_coverage": round(bg_coverage * 100, 1),
                        "clip_confidence": clip_raw_fast.get("category", {}).get("confidence", 0),
                        "clip_category": clip_raw_fast.get("category", {}).get("value", "unknown")
                    }
            
            print(f"  ✓ تم التأكيد: صورة ملابس (تغطية: {bg_coverage:.1%})")

            # ③ Florence — وصف نصي
            print("③ Florence — وصف نصي...")
            description = self.describer.describe(image)

            # ④ NLP — استخراج منظم من الوصف
            print("④ NLP — استخراج attributes...")
            nlp_attrs = self.nlp_extractor.extract(description)

            # ⑤ CLIP كامل (إذا أردنا إعادة الاستخدام، لكننا استخدمنا النتيجة السابقة)
            # نستخدم clip_raw_fast مباشرة عشان نوفر وقت
            clip_raw = clip_raw_fast

            # ⑥ Fashion Engine — دمج وقرار
            print("⑤ Fashion Engine — دمج وقرار...")
            decision = self.decision_eng.decide(
                florence_desc = description,
                clip_result   = clip_raw,
                nlp_attrs     = nlp_attrs,
            )

            # طباعة القرارات
            print("\n  ── قرارات الـ Engine:")
            for log in decision["decisions_log"]:
                print(f"    {log}")

            # ⑦ الألوان
            print("\n⑥ استخراج الألوان...")
            colors = self.color_ext.extract(image)
            colors = self.color_ext.classify_groups(colors)

            return {
                "status": "success",
                "image_path": image_path,
                "description": description,
                "attributes": decision["final_attributes"],
                "confidence": decision["confidence"],
                "colors": colors,
                "primary_color": colors[0] if colors else None,
                "engine_log": decision["decisions_log"],
                "is_clothing": True,
                "bg_coverage": round(bg_coverage * 100, 1)
            }

        except FileNotFoundError:
            return {"status": "error", "message": f"الصورة غير موجودة: {image_path}"}
        except Exception as e:
            return {"status": "error", "message": str(e), "image_path": image_path}

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
            save_path: Optional[str] = None,
            strict_mode: bool = False
    ) -> List[Dict]:
        """
        تحليل خزانة كاملة
        
        Parameters:
        - image_paths: قائمة بمسارات الصور
        - save_path: مسار لحفظ النتائج (اختياري)
        - strict_mode: إذا True، يتم رفض أي صورة ليست ملابس بشكل قاطع
        """
        results = []
        success = 0
        rejected = 0
        failed = 0
        
        print(f"\n{'='*60}")
        print(f"تحليل {len(image_paths)} قطعة...")
        if strict_mode:
            print("⚠ الوضع الصارم: سيتم رفض الصور التي لا تحتوي على ملابس")
        print(f"{'='*60}")
        
        for i, path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}]")
            r = self.analyze_item(path)
            results.append(r)
            
            if r["status"] == "success":
                success += 1
            elif r["status"] == "rejected":
                rejected += 1
                if strict_mode:
                    print(f"  ✗ مرفوض: {r.get('message', 'غير معروف')}")
            else:
                failed += 1
                print(f"  ✗ فشل: {r.get('message', 'خطأ غير معروف')}")
        
        print(f"\n{'─'*50}")
        print(f"📊 النتائج النهائية:")
        print(f"  ✅ نجح: {success}")
        print(f"  ⚠️ مرفوض (ليس ملابس): {rejected}")
        print(f"  ❌ فشل: {failed}")
        print(f"{'─'*50}")
        
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✓ تم حفظ النتائج في: {save_path}")
        
        return results
