
# ============================================================
# main.py — اختبار كامل للمراحل الثلاث
# ============================================================

from ai.pipeline.onboarding_pipeline import OnboardingPipeline
from ai.services.wardrobe_pipeline   import WardrobePipeline
import json

if __name__ == "__main__":

    # ── اختبار المرحلتين الأوليتين ──
    print("═"*50)
    print("اختبار OnboardingPipeline")
    print("═"*50)

    onboarding = OnboardingPipeline()
    user_result = onboarding.run(
        user_id  = "user_001",
        skin_hex = "#C68642",
        height   = 175, weight   = 70,
        chest    = 95,  waist    = 80,
        hip      = 98,  shoulder = 42
    )
    print(f"✓ نوع البشرة: {user_result['skin']['skin_type']}")
    print(f"✓ المانيكان:  {user_result['mannequin']['mesh_path']}")

    # ── اختبار المرحلة الثالثة ──
    print("\n" + "═"*50)
    print("اختبار WardrobePipeline")
    print("═"*50)

    wardrobe = WardrobePipeline()
    item = wardrobe.analyze_item("/content/rayam-fashion-ai/download.jfif")

    if item["status"] == "success":
        print(f"✓ الفئة:    {item['attributes']['category']}")
        print(f"✓ المناسبة: {item['attributes']['occasion']}")
        print(f"✓ اللون:    {item['primary_color']['hex']}"
              f" — {item['primary_color']['group']}")
    else:
        print(f"✗ {item['message']}")

    with open("test_result.json", "w",
              encoding="utf-8") as f:
        json.dump(item, f,
                  ensure_ascii=False, indent=2)
    print("\n✓ محفوظ: test_result.json")
