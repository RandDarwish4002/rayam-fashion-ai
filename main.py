
from ai.pipeline.onboarding_pipeline import OnboardingPipeline
from ai.services.wardrobe_pipeline import WardrobePipeline
import json

if __name__ == "__main__":

    print("="*50)
    print("OnboardingPipeline Test")
    print("="*50)

    onboarding = OnboardingPipeline()
    user_result = onboarding.run(
        user_id="user_001",
        skin_hex="#C68642",
        height=175, weight=70,
        chest=95, waist=80,
        hip=98, shoulder=42
    )

    print("Skin:", user_result["skin"]["skin_type"])
    print("Mannequin:", user_result["mannequin"]["mesh_path"])

    print("\n" + "="*50)
    print("WardrobePipeline Test")
    print("="*50)

    wardrobe = WardrobePipeline()
    item = wardrobe.analyze_item("/content/rayam-fashion-ai/thumbnail-500x500.webp")

    if item["status"] == "success":

        attrs = item.get("attributes", {})

        print("Category:", attrs.get("category", "unknown"))
        print("Occasion:", attrs.get("occasion", "unknown"))   # FIX
        print("Season:", attrs.get("season", "unknown"))
        print("Color:", item.get("primary_color", {}).get("hex", "unknown"))

    else:
        print("Error:", item.get("message", "unknown"))

    with open("test_result.json", "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=2)

    print("\nSaved: test_result.json")
